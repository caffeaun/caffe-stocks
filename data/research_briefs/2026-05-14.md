# Research brief — 2026-05-14

## Registry-bias diagnosis
Last-30-day distribution: xgb_classifier 41% (368), ranker 21% (190), xgb_regressor 13% (111), nn 10% (87), non-parametric 7% (64), linear/kernel 4% (37), tree-other 2% (16) — i.e. **75% of recent iterations are GBDT variants**. Yesterday's brief added the **tabular foundation model** axis (`tabpfn_v25`). Two large inductive-bias gaps remain: (a) **time-series foundation models** pretrained on the Time-300B / GIFT-Eval corpora (TimesFM, Chronos-2, Moirai-2, Time-MoE) — the entire family is absent from the registry; (b) **sparse mixture-of-experts** routing — also absent. Both gaps matter because the regime-shifting bear windows that have repeatedly killed our XGB tweaks (W3, W5) are exactly the kind of "different temporal pattern needs different parameter pathway" problem that MoE solves by construction. Closing both gaps in a single trainer (Time-MoE) is high-leverage and structurally orthogonal to yesterday's TabPFN-2.5 add.

## Recommended technique: Time-MoE (frozen TS-foundation encoder + small classification head)

### Why now
Time-MoE is the first **billion-scale time-series foundation model with sparse mixture-of-experts** ([arxiv:2409.16040](https://arxiv.org/abs/2409.16040)) — pretrained on Time-300B (300B time-series points across 9 domains), open-sourced under Apache-2.0, and now downloaded ~400k times/month from Hugging Face ([Maple728/TimeMoE-50M](https://huggingface.co/Maple728/TimeMoE-50M)). The 2026 ML Mastery toolkit names it among the 5 production-ready TS foundation models ([MachineLearningMastery, 2026](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)), and follow-up work (Time Tracker, [arxiv:2505.15151](https://arxiv.org/abs/2505.15151)) has confirmed that MoE routing inside a TS encoder learns regime-specific experts — exactly the inductive bias our W3/W5 bear windows have lacked. We use the smallest variant (50M params, ~0.1B BF16 weights) as a **frozen encoder**: no fine-tuning, no GPU memory pressure, 30-min wall-time budget per window respected. A small MLP head is the only trainable surface — the rest is in-context-style transfer from the pretrained MoE.

### Deps
```bash
pip install transformers>=4.40.0 torch>=2.1.0 accelerate>=0.30.0
```

### Weights
```bash
huggingface-cli download Maple728/TimeMoE-50M --local-dir ~/.cache/huggingface/hub/models--Maple728--TimeMoE-50M
```

### Scaffold code
```python
# models/trainers.py — append at end (before TRAINERS dict)

import os
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


class _TimeMoEHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchTimeMoETrainer(BaseTrainer):
    """Time-MoE (Maple728/TimeMoE-50M) used as a FROZEN time-series
    foundation encoder, with a small MLP classification head trained on
    pooled embeddings concatenated with the raw tabular features.

    Brings two inductive biases that are absent from the registry:
      (1) time-series foundation-model pretraining (Time-300B), and
      (2) sparse mixture-of-experts routing — pathways activate per
          input pattern, which is the natural fit for the regime-
          shifting bear windows where prior XGB variants have failed.

    Each row's F-dim feature vector is reshaped to a univariate
    sequence of length F and pushed through Time-MoE; the last-layer
    hidden states are mean-pooled to a fixed embedding, optionally
    concatenated with the raw features, then fed to the trainable head.
    The encoder is loaded once on first fit() and never updated.
    """
    name = 'torch_time_moe'
    consumes_sequences = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            raise ImportError(
                "torch_time_moe needs torch>=2.1 and transformers>=4.40. "
                "Run `pip install torch transformers accelerate`."
            )
        self.model_id = str(kwargs.get('model_id', 'Maple728/TimeMoE-50M'))
        self.hidden_dim = int(kwargs.get('hidden_dim', 128))
        self.dropout = float(kwargs.get('dropout', 0.15))
        self.learning_rate = float(kwargs.get('learning_rate', 1e-3))
        self.weight_decay = float(kwargs.get('weight_decay', 1e-4))
        self.batch_size = int(kwargs.get('batch_size', 256))
        self.encode_batch_size = int(kwargs.get('encode_batch_size', 64))
        self.epochs = int(kwargs.get('epochs', 12))
        self.patience = int(kwargs.get('patience', 3))
        self.use_raw_features = bool(kwargs.get('use_raw_features', True))
        self.embed_pool = str(kwargs.get('embed_pool', 'mean'))  # 'mean' | 'last'
        self.random_state = int(kwargs.get('random_state', 42))
        self._encoder = None
        self._head = None
        self._embed_dim = None
        self._n_features = None
        self._device = None

    def _pick_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load_encoder(self):
        if self._encoder is not None:
            return
        self._device = self._pick_device()
        self._encoder = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self._device,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad = False

    def _sanitize(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _encode(self, X):
        # Each row of shape (F,) is treated as a univariate time series of length F.
        # Time-MoE accepts float-valued (B, T) input via its causal-LM forward pass
        # and we ask for hidden states explicitly.
        n = X.shape[0]
        bs = self.encode_batch_size
        out = []
        for i in range(0, n, bs):
            xb = torch.from_numpy(X[i:i + bs]).to(self._device).float()
            try:
                outputs = self._encoder(
                    input_ids=xb,
                    output_hidden_states=True,
                    return_dict=True,
                )
            except TypeError:
                # Some custom Time-MoE forwards expect `inputs_embeds` of shape (B, T, 1)
                outputs = self._encoder(
                    inputs_embeds=xb.unsqueeze(-1),
                    output_hidden_states=True,
                    return_dict=True,
                )
            h = outputs.hidden_states[-1]  # (B, T, D)
            if self.embed_pool == 'last':
                emb = h[:, -1, :]
            else:
                emb = h.mean(dim=1)
            out.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, **kwargs):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._load_encoder()

        X_tr = self._sanitize(X_tr)
        y_tr = np.asarray(y_tr, dtype=np.float32).ravel()
        self._n_features = X_tr.shape[1]
        Z_tr = self._encode(X_tr)
        self._embed_dim = Z_tr.shape[1]

        if self.use_raw_features:
            feed_tr = np.concatenate([Z_tr, X_tr], axis=1)
        else:
            feed_tr = Z_tr
        in_dim = feed_tr.shape[1]

        self._head = _TimeMoEHead(in_dim, self.hidden_dim, self.dropout).to(self._device)
        opt = torch.optim.AdamW(
            self._head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        x_tr_t = torch.from_numpy(feed_tr).to(self._device)
        y_tr_t = torch.from_numpy(y_tr).to(self._device)

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        x_val_t = y_val_t = None
        if has_val:
            X_val_s = self._sanitize(X_val)
            Z_val = self._encode(X_val_s)
            feed_val = np.concatenate([Z_val, X_val_s], axis=1) if self.use_raw_features else Z_val
            x_val_t = torch.from_numpy(feed_val).to(self._device)
            y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32).ravel()).to(self._device)

        best_val = float('inf')
        best_state = None
        bad = 0
        n = x_tr_t.shape[0]
        idx = np.arange(n)
        for ep in range(self.epochs):
            np.random.shuffle(idx)
            self._head.train()
            for i in range(0, n, self.batch_size):
                jb = idx[i:i + self.batch_size]
                xb = x_tr_t[jb]
                yb = y_tr_t[jb]
                opt.zero_grad()
                logits = self._head(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            if has_val:
                self._head.eval()
                with torch.no_grad():
                    val_logits = self._head(x_val_t)
                    val_loss = loss_fn(val_logits, y_val_t).item()
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    bad = 0
                    best_state = {k: v.detach().clone() for k, v in self._head.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self._head.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._head is None:
            raise RuntimeError("torch_time_moe: predict_proba called before fit()")
        self._load_encoder()
        X = self._sanitize(X)
        Z = self._encode(X)
        feed = np.concatenate([Z, X], axis=1) if self.use_raw_features else Z
        self._head.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(feed).to(self._device)
            logits = self._head(x_t)
            proba = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        return proba

    def save(self, model_dir, extra=None):
        os.makedirs(model_dir, exist_ok=True)
        if self._head is not None:
            torch.save(
                {
                    'state_dict': self._head.state_dict(),
                    'embed_dim': self._embed_dim,
                    'n_features': self._n_features,
                    'hidden_dim': self.hidden_dim,
                    'dropout': self.dropout,
                    'use_raw_features': self.use_raw_features,
                    'embed_pool': self.embed_pool,
                },
                os.path.join(model_dir, 'head.pt'),
            )
        meta = {
            'trainer': self.name,
            'model_id': self.model_id,
            'embed_pool': self.embed_pool,
            'use_raw_features': self.use_raw_features,
            'device': self._device,
            'embed_dim': self._embed_dim,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)
```

### Registry registration
```python
'torch_time_moe': TorchTimeMoETrainer,
```

### HP search space
```python
# models/search_spaces.py — entry to add to SEARCH_SPACES dict
'torch_time_moe': {
    'hidden_dim':         [64, 128, 256],
    'dropout':            (0.0, 0.4),
    'learning_rate':      (1e-4, 1e-2),
    'weight_decay':       (1e-6, 1e-3),
    'batch_size':         [128, 256, 512],
    'encode_batch_size':  [32, 64, 128],
    'epochs':             [5, 10, 20],
    'patience':           [2, 3, 5],
    'use_raw_features':   [True, False],
    'embed_pool':         ['mean', 'last'],
    'random_state':       [42, 7, 1337],
},
```

### Integration notes (for claude_mode)
- **Encoder is frozen** — only the MLP head trains. Per-window wall-time on a single GPU (the 50M-param encoder fits comfortably in 1 GB VRAM at FP32) should be ~3-8 min: encoding 20-40k rows × 96 features once, then ~10 head epochs over the cached embeddings. On CPU expect ~15-25 min — reduce `encode_batch_size` to 32 to avoid OOM.
- **Encoder I/O contract**: Time-MoE's HuggingFace forward signature has been stable since Sep 2024, but if `input_ids` is rejected (some custom forwards take `inputs_embeds=(B,T,1)`), the trainer's `_encode` already falls back to the embeds form via try/except. If both fail, the model card shows the canonical pattern in `model.generate(...)` — adapt by extracting the encoder block manually from `self._encoder.model`.
- **Why concat raw features**: Time-MoE was pretrained on naturally temporal univariate signals (electricity, traffic, finance, etc.); our 96-dim feature vector is heterogeneous (price stats, volume, indicators), so the encoder embedding alone may not capture domain-specific signal. The raw-features concat (`use_raw_features=True` default) lets the MLP head still see the original signal — treat the encoder output as a learned regime tag rather than a complete representation.
- **Expected gate behavior**: Threshold sweep should be ALIVE on W1, W2, W4, W6, W7 (fits where prior XGB variants already pass thresholds). The interesting test is W3 and W5 — if MoE routing picks up bear-regime patterns, those should see meaningful improvement. If both bear windows still fail, retry with `use_raw_features=False` to force the head to rely entirely on encoder embeddings (this is a sharper signal-to-noise test of whether the foundation-model transfer is real).
- **Determinism**: encoder is frozen so it adds no run-to-run noise; head training has the usual GPU non-determinism — log seeds for sweep reproducibility.

## Recommended next action for claude_mode
After `scripts/ml_scaffold.py` installs `transformers>=4.40 torch>=2.1 accelerate` and pre-downloads the TimeMoE-50M weights to `~/.cache/huggingface/hub/`, the next claude_mode iteration should run `--model-type torch_time_moe` with default HPs (`hidden_dim=128`, `dropout=0.15`, `lr=1e-3`, `epochs=12`, `patience=3`, `use_raw_features=True`, `embed_pool=mean`) and cite this brief in its iteration log. Because this is the first **time-series foundation model** in the registry and the first **MoE-routing** trainer, do NOT pivot back to xgb_* losses or another tabular variant for at least one full HP-sweep cycle — preserve the slot the way `kernel_logreg` and `tabpfn_v25` were preserved. The two windows most likely to move are **W3 and W5 (bear regimes)**, where the MoE's pattern-specialised experts and Time-300B pretrained priors over volatility regimes are the inductive bias the registry has been missing.

## Sources
- [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts (arxiv:2409.16040)](https://arxiv.org/abs/2409.16040) — primary paper; sparse-MoE TS foundation model pretrained on Time-300B; Apache-2.0; ICLR 2025 spotlight.
- [Maple728/TimeMoE-50M on Hugging Face](https://huggingface.co/Maple728/TimeMoE-50M) — 50M-param checkpoint (~0.1B BF16); 406k downloads/month; canonical loading via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.
- [Time-MoE GitHub (time-moe/time-moe)](https://github.com/time-moe/time-moe) — install instructions, requirements.txt, official inference example, optional flash-attn for speed.
- [Time Tracker: MoE-Enhanced Foundation Time-Series Forecasting (arxiv:2505.15151)](https://arxiv.org/abs/2505.15151) — independent confirmation that sparse MoE routing inside a TS encoder learns regime-specific experts; supports the W3/W5 bear-regime hypothesis.
- [The 2026 Time Series Toolkit: 5 Foundation Models for Autonomous Forecasting (MachineLearningMastery, 2026)](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/) — current-year survey naming Time-MoE among the 5 production-ready TS foundation models.
- [TS-RAG: Retrieval-Augmented TSFM with MoE (OpenReview)](https://openreview.net/forum?id=TJuUelhGQr) — additional 2025 evidence that MoE-augmented TS foundation models post +6.5% over single-expert TSFMs across diverse domains.
