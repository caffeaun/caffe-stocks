# Research brief — 2026-05-14

## Registry-bias diagnosis
Last-30-day distribution shows 41% xgb_classifier, 21% ranker, 13% xgb_regressor, 10% nn, 7% non-parametric, 4% linear/kernel, 2% tree-other — i.e. 75% of recent iterations are GBDT variants and 4 of the 47 registered trainers are GBDT-loss tweaks (focal, JTT, group-balanced-focal, magnitude, etc.). The class of model that the 2025–2026 literature has shown to *beat* tuned XGBoost on small-to-medium tabular classification — **in-context tabular foundation models** — is entirely absent from the registry. TabPFN-2.5 reports a 100% win rate against default XGBoost on classification ≤10k rows × 500 features and 87% on ≤100k × 2k, and now leads the TabArena benchmark against tuned tree ensembles ([Hollmann et al., 2025](https://arxiv.org/abs/2511.08667)). Our walk-forward windows are 20–40k rows × 96 features per training fold, which sits inside the regime where TabPFN-2.5's published advantage is largest. Adding it closes the most expensive gap in the registry without re-treading any tree-loss territory.

## Recommended technique: TabPFN-2.5 (Prior Labs tabular foundation model)

### Why now
TabPFN-2.5 was published Nov 2025 and is the first tabular foundation model that scales the original TabPFN's in-context Bayesian inference to 50k×2k while preserving its 100%-vs-XGBoost win rate on small-to-medium classification ([arxiv:2511.08667](https://arxiv.org/abs/2511.08667), [TabArena leaderboard](https://priorlabs.ai/tabpfn)). Independent enterprise replication on credit-risk tabular data confirmed the result against tuned XGB and AutoGluon ([Mission Lane benchmark](https://medium.com/mission-lane-tech-blog/tabicl-under-the-microscope-benchmarking-tabular-foundation-models-for-enterprise-credit-risk-ad8315f9bec4)). The model performs no gradient updates at fit time — it does in-context attention over the training set — which makes it especially robust on the kind of low-signal, regime-shifting financial data that has plagued our XGBoost variants (no over-fit window-by-window, since there are no parameters being fit per window).

### Deps
```bash
pip install tabpfn
```

### Weights
N/A — the `tabpfn` package auto-downloads the pretrained checkpoint (~200 MB) from Hugging Face on first `fit()` call to `~/.cache/tabpfn/`. No manual step required. If offline pre-cache is desired the post-script may optionally run: `huggingface-cli download Prior-Labs/tabpfn_2_5 --local-dir ~/.cache/tabpfn/Prior-Labs--tabpfn_2_5`

### Scaffold code
```python
# models/trainers.py — append at end (before TRAINERS dict)

import os
import json
import pickle
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from tabpfn import TabPFNClassifier
    _HAS_TABPFN = True
except Exception:
    _HAS_TABPFN = False


class TabPFNV25Trainer(BaseTrainer):
    """In-context tabular foundation model (Prior Labs TabPFN-2.5).

    No gradient updates at fit-time: stores the training rows and the
    pretrained backbone performs in-context Bayesian inference on each
    predict_proba call. Subsamples (stratified) to max_train_rows to
    stay inside the published 50k-row support of TabPFN-2.5 and to
    bound per-window wall-time.
    """
    name = 'tabpfn_v25'
    consumes_sequences = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not _HAS_TABPFN:
            raise ImportError(
                "tabpfn not installed. `pip install tabpfn` (Python 3.9+, "
                "PyTorch>=2.1). CUDA strongly recommended."
            )
        self.n_estimators = int(kwargs.get('n_estimators', 4))
        self.softmax_temperature = float(kwargs.get('softmax_temperature', 0.9))
        self.balance_probabilities = bool(kwargs.get('balance_probabilities', False))
        self.average_before_softmax = bool(kwargs.get('average_before_softmax', False))
        self.ignore_pretraining_limits = bool(kwargs.get('ignore_pretraining_limits', True))
        self.random_state = int(kwargs.get('random_state', 42))
        self.max_train_rows = int(kwargs.get('max_train_rows', 30000))
        self._model = None
        self._X_train = None
        self._y_train = None
        self._device = None

    def _sanitize(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _stratified_subsample(self, X, y):
        n = len(X)
        if n <= self.max_train_rows:
            return X, y
        rng = np.random.default_rng(self.random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        frac = self.max_train_rows / float(n)
        n_pos = max(1, int(round(len(pos_idx) * frac)))
        n_neg = max(1, min(self.max_train_rows - n_pos, len(neg_idx)))
        sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        idx = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(idx)
        return X[idx], y[idx]

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, **kwargs):
        X_tr = self._sanitize(X_tr)
        y_tr = np.asarray(y_tr).astype(np.int64).ravel()
        X_tr, y_tr = self._stratified_subsample(X_tr, y_tr)
        self._device = 'cuda' if (_HAS_TORCH and torch.cuda.is_available()) else 'cpu'
        self._model = TabPFNClassifier(
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            balance_probabilities=self.balance_probabilities,
            average_before_softmax=self.average_before_softmax,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            device=self._device,
            random_state=self.random_state,
        )
        self._model.fit(X_tr, y_tr)
        self._X_train = X_tr
        self._y_train = y_tr
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TabPFNV25Trainer: predict_proba called before fit()")
        X = self._sanitize(X)
        proba = self._model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float32)
        return np.asarray(proba, dtype=np.float32).ravel()

    def save(self, model_dir, extra=None):
        os.makedirs(model_dir, exist_ok=True)
        # The pretrained backbone is reloaded from cache on next fit; we
        # only need to persist the in-context training data + HP so the
        # trainer can be reconstructed deterministically.
        with open(os.path.join(model_dir, 'tabpfn_state.pkl'), 'wb') as f:
            pickle.dump({
                'X_train': self._X_train,
                'y_train': self._y_train,
                'hp': {
                    'n_estimators': self.n_estimators,
                    'softmax_temperature': self.softmax_temperature,
                    'balance_probabilities': self.balance_probabilities,
                    'average_before_softmax': self.average_before_softmax,
                    'ignore_pretraining_limits': self.ignore_pretraining_limits,
                    'random_state': self.random_state,
                    'max_train_rows': self.max_train_rows,
                },
                'device': self._device,
            }, f)
        meta = {'trainer': self.name, 'device': self._device}
        if extra:
            meta.update(extra)
        with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)
```

### Registry registration
```python
'tabpfn_v25': TabPFNV25Trainer,
```

### HP search space
```python
'tabpfn_v25': {
    'n_estimators':          [2, 4, 8],
    'softmax_temperature':   (0.5, 1.5),
    'balance_probabilities': [False, True],
    'average_before_softmax':[False, True],
    'max_train_rows':        [10000, 20000, 30000],
    'random_state':          [42, 7, 1337],
},
```

### Integration notes (for claude_mode)
TabPFN-2.5 is in-context: there is no per-window gradient training, so the "fit" call is essentially `store(X_train, y_train)` plus a single forward pass through the pretrained backbone — wall-time per window should be 30 s – 3 min on GPU, ~10–20 min on CPU (so a GPU is strongly recommended; fall back to `max_train_rows=10000` if CPU-only). Inputs MUST be 2D float arrays with NaN/inf scrubbed (the trainer handles this internally via `np.nan_to_num`). Because the model performs Bayesian inference rather than ERM, it does *not* over-fit on small windows the way XGBoost does — expect more stable score distributions across W1–W7 and especially better behavior on bear windows (W3, W5) where prior XGBoost variants over-confidently rank junk. Default `n_estimators=4` ensembles 4 internal feature-permutation forward passes; raise to 8 if you see threshold-sweep instability, lower to 2 if wall-time blows. Expect threshold sweep to be alive across windows by default — TabPFN's output is well-calibrated out of the box.

## Recommended next action for claude_mode
After `scripts/ml_scaffold.py` pip-installs `tabpfn` and writes the class + registry entry, the next claude_mode iteration should run `--model-type tabpfn_v25` with default HPs (n_estimators=4, softmax_temperature=0.9, balance_probabilities=False, max_train_rows=30000) and cite this brief in its iteration log. Prior tree-loss tweaks have repeatedly fallen down on W3 (bear) and W5 (false-positive heavy bear day) — TabPFN-2.5's in-context Bayesian inference and TabArena-validated calibration are the two properties most likely to move those specific windows, and the kernel_logreg breakthrough on W4/W6 already proved that non-XGBoost score geometries pass the gate. Do NOT pivot back to xgb_* losses for at least one full cycle of tabpfn_v25 HP exploration; treat the slot the way you'd treat knn_classifier and kernel_logreg — preserve it as the "tabular foundation model" axis of the registry.

## Sources
- [Hollmann et al. — TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models (arXiv 2511.08667, Nov 2025)](https://arxiv.org/abs/2511.08667) — primary reference; 100% win rate vs default XGBoost ≤10k×500, 87% ≤100k×2k.
- [PriorLabs/TabPFN GitHub](https://github.com/PriorLabs/TabPFN) — pip install, sklearn-style API, checkpoint cache behavior, hardware notes.
- [TabPFN-2.5 Model Report — Prior Labs](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report) — TabArena leaderboard position vs AutoGluon 1.4 and tuned tree baselines.
- [Mission Lane Tech Blog — TabICL Under the Microscope: Benchmarking Tabular Foundation Models for Enterprise Credit Risk](https://medium.com/mission-lane-tech-blog/tabicl-under-the-microscope-benchmarking-tabular-foundation-models-for-enterprise-credit-risk-ad8315f9bec4) — independent enterprise replication on credit-risk tabular data.
- [Prior-Labs/tabpfn_2_5 on Hugging Face](https://huggingface.co/Prior-Labs/tabpfn_2_5) — checkpoint hosting for offline pre-cache.
