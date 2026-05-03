"""Shared LSTM model class — single source of truth.
Imported by both lstm_trainer.py and lstm_trader.py.
Supports optional ensemble mode (prediction averaging of N sub-models).
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3,
                 use_attention=False, head_dims=None, input_dropout=0.0,
                 hidden_noise_std=0.0, output_temperature=0.5,
                 sequence_normalize=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_attention = use_attention
        self.head_dims = head_dims or []
        self.n_ensemble = 1
        self.hidden_noise_std = hidden_noise_std

        # Within-sequence z-score normalization: for each sequence window,
        # normalize each feature to zero mean / unit std within that window.
        # Eliminates scaler drift across walk-forward splits because the model
        # only sees "how unusual is today relative to the last N days" rather
        # than absolute feature magnitudes. Stored as buffer for save/load.
        self.register_buffer('_seq_normalize',
                             torch.tensor(1 if sequence_normalize else 0))

        # MLP mode: num_layers=0 bypasses LSTM entirely, uses last timestep only.
        # Removes temporal modeling that consistently overfits (best epoch always 0).
        self.use_mlp = (num_layers == 0)

        # Output temperature: sigmoid(logit / T) controls prediction sharpness.
        # T=0.5 (default): sharper sigmoid — logit 0.2 → score 0.60, logit 0.4 → 0.69.
        # Critical under sequence_normalize=True because z-score normalization
        # compresses LSTM logit magnitudes toward zero (variance is normalized out),
        # so with T=0.7 almost no samples exceed the 0.6 live threshold.
        # Recent attempts 256-265 produced 0 trades at WF gate because T=0.7 +
        # sequence_normalize left candidate test_recall at 0.08% (essentially never
        # firing). T=0.5 doubles sensitivity so genuine positives can clear 0.6.
        # Stored as a buffer so it persists through save/load via state_dict.
        # Old models without this buffer get T=1.0 via load_state_dict (backward compat).
        temp_val = float(output_temperature) if output_temperature is not None else 1.0
        self.register_buffer('_output_temp', torch.tensor(temp_val))

        # Input feature dropout: randomly zero input features during training.
        # Forces the model to not rely on any single feature, improving
        # robustness across time periods where feature importance shifts.
        self.input_dropout = nn.Dropout(input_dropout)

        # Single-model modules (default)
        if self.use_mlp:
            # Sequence-aggregated MLP: concatenate [last, mean, last-mean] for 3x features.
            # This captures current state, recent baseline, and deviation from baseline
            # without LSTM temporal overfitting. Deviation features are inherently
            # regime-invariant (they measure change, not absolute level).
            # LayerNorm stabilizes hidden activations across regime shifts.
            self.mlp_hidden = nn.Sequential(
                nn.Linear(input_size * 3, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout)
            # LayerNorm on LSTM output: normalizes hidden states to zero mean
            # and unit variance before the FC layer. This:
            # 1. Stabilizes the FC input scale across different time periods
            #    (train 2023 vs test 2025 hidden states become comparable)
            # 2. Reduces covariate shift between walk-forward splits
            # 3. Helps produce well-calibrated logits (less score compression)
            # Only 2*hidden_size params — negligible capacity increase.
            self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        # When use_attention=True: mean pooling over all timesteps (no learned params).
        # Mean pooling forces every timestep to contribute, preventing the model
        # from memorizing position-specific temporal patterns that don't generalize.
        # No self.attention linear layer — mean pooling has zero learnable params.
        # MLP head: optional hidden layers between LSTM output and classification
        if self.head_dims:
            head_layers = []
            prev_dim = hidden_size
            for dim in self.head_dims:
                head_layers.append(nn.Linear(prev_dim, dim))
                head_layers.append(nn.ReLU())
                head_layers.append(nn.Dropout(dropout))
                prev_dim = dim
            head_layers.append(nn.Linear(prev_dim, 1))
            self.fc = nn.Sequential(*head_layers)
        else:
            self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Positive FC bias init: tilt initial logits toward positive so early
        # sigmoid outputs sit slightly above 0.5 rather than exactly at it.
        # Combined with output_temperature=0.5 (sharp sigmoid), a +0.5 logit
        # bias produces initial scores around sigmoid(1.0)=0.73 — well above
        # the 0.6 live threshold. Training then has to actively *push down* on
        # negatives rather than *push up* on positives — a more stable gradient
        # landscape that avoids the "never be confident" local minimum observed
        # in attempts 266-269 (all produced 0 trades at WF with Q99(scores)<0.5).
        # The majority-negative class will naturally pull most scores down, but
        # high-signal positives retain enough confidence to clear 0.6.
        if self.head_dims:
            nn.init.constant_(self.fc[-1].bias, 0.5)
        else:
            nn.init.constant_(self.fc.bias, 0.5)

    def _setup_ensemble(self, n):
        """Dynamically create N sub-models for ensemble prediction averaging.
        Supports both LSTM and MLP (num_layers=0) sub-models.
        """
        if self.n_ensemble == n and hasattr(self, 'models'):
            return  # Already set up
        self.n_ensemble = n
        # Remove single-model modules to avoid state_dict conflicts
        for attr in ('lstm', 'mlp_hidden', 'dropout', 'attention', 'fc', 'output_norm'):
            if hasattr(self, attr):
                delattr(self, attr)
        self.models = nn.ModuleList()
        for _ in range(n):
            sub = nn.ModuleDict()
            if self.use_mlp:
                sub['mlp_hidden'] = nn.Sequential(
                    nn.Linear(self.input_size * 3, self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                )
            else:
                sub['lstm'] = nn.LSTM(self.input_size, self.hidden_size,
                                      num_layers=self.num_layers,
                                      batch_first=True, dropout=self.dropout_rate)
                sub['dropout'] = nn.Dropout(self.dropout_rate)
            # use_attention=True uses mean pooling (no learnable attention layer)
            # Support head_dims MLP in ensemble mode
            if self.head_dims:
                head_layers = []
                prev_dim = self.hidden_size
                for dim in self.head_dims:
                    head_layers.append(nn.Linear(prev_dim, dim))
                    head_layers.append(nn.ReLU())
                    head_layers.append(nn.Dropout(self.dropout_rate))
                    prev_dim = dim
                head_layers.append(nn.Linear(prev_dim, 1))
                sub['fc'] = nn.Sequential(*head_layers)
            else:
                sub['fc'] = nn.Linear(self.hidden_size, 1)
            self.models.append(sub)

    def _normalize_sequence(self, x):
        """Per-sequence z-score normalization.

        For each (batch, seq_len, features) input, normalize each feature
        to zero mean / unit std within the seq_len window. This makes the
        model see "how unusual is today vs the last N days" instead of
        absolute feature levels — eliminating scaler drift across time periods.
        """
        # x: (batch, seq_len, features)
        mean = x.mean(dim=1, keepdim=True)   # (batch, 1, features)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)  # avoid div-by-zero
        return (x - mean) / std

    def _aggregate_sequence(self, x):
        """Concatenate [last_timestep, seq_mean, last - mean] for MLP input.

        Gives the model 3 views of each feature:
          - last: current state (what the market looks like right now)
          - mean: recent baseline (what's normal over the sequence window)
          - last - mean: deviation (how unusual is today — regime-invariant)
        """
        last = x[:, -1, :]           # (batch, features)
        mean = x.mean(dim=1)         # (batch, features)
        return torch.cat([last, mean, last - mean], dim=-1)  # (batch, 3*features)

    def _forward_mlp(self, x):
        """MLP forward: sequence-aggregated features, no temporal modeling.
        Note: mlp_hidden already includes dropout — no extra dropout here.
        """
        out = self._aggregate_sequence(x)  # (batch, 3*input_size)
        out = self.mlp_hidden(out)  # (batch, hidden_size) — includes LayerNorm + dropout
        logit = self.fc(out).squeeze(-1)
        return self.sigmoid(logit / self._output_temp)

    def _forward_mlp_logits(self, x):
        """MLP forward returning raw logits (before sigmoid)."""
        out = self._aggregate_sequence(x)
        out = self.mlp_hidden(out)  # includes LayerNorm + dropout
        return self.fc(out).squeeze(-1)

    def _forward_single(self, x, lstm, dropout, fc, attention=None):
        out, _ = lstm(x)
        if self.training and self.hidden_noise_std > 0:
            out = out + torch.randn_like(out) * self.hidden_noise_std
        if self.use_attention:
            # Recency-weighted pooling: exponential decay giving ~7x more weight
            # to last timestep vs first. Captures full sequence context while
            # emphasizing recent price action (most predictive for 10-day trades).
            # Zero learnable parameters — no overfitting risk.
            T = out.size(1)
            w = torch.exp(torch.linspace(-2.0, 0.0, T, device=out.device))
            w = w / w.sum()
            out = (out * w.view(1, -1, 1)).sum(dim=1)
            out = dropout(out)
        else:
            out = dropout(out[:, -1, :])
        if hasattr(self, 'output_norm'):
            out = self.output_norm(out)
        logit = fc(out).squeeze(-1)
        return self.sigmoid(logit / self._output_temp)

    def _forward_single_logits(self, x, lstm, dropout, fc, attention=None):
        """Return raw logits (before sigmoid) for a single sub-model."""
        out, _ = lstm(x)
        if self.training and self.hidden_noise_std > 0:
            out = out + torch.randn_like(out) * self.hidden_noise_std
        if self.use_attention:
            # Recency-weighted pooling (matches forward)
            T = out.size(1)
            w = torch.exp(torch.linspace(-2.0, 0.0, T, device=out.device))
            w = w / w.sum()
            out = (out * w.view(1, -1, 1)).sum(dim=1)
            out = dropout(out)
        else:
            out = dropout(out[:, -1, :])
        if hasattr(self, 'output_norm'):
            out = self.output_norm(out)
        return fc(out).squeeze(-1)

    def get_logits(self, x):
        """Return raw logits before sigmoid. For use with BCEWithLogitsLoss."""
        if self._seq_normalize.item():
            x = self._normalize_sequence(x)
        x = self.input_dropout(x)
        if self.training and self.INPUT_NOISE_STD > 0:
            x = x + torch.randn_like(x) * self.INPUT_NOISE_STD
        if self.use_mlp:
            if self.n_ensemble > 1:
                agg = self._aggregate_sequence(x)
                logits = []
                for sub in self.models:
                    out = sub['mlp_hidden'](agg)
                    logits.append(sub['fc'](out).squeeze(-1))
                return torch.stack(logits).mean(0)
            return self._forward_mlp_logits(x)
        if self.n_ensemble > 1:
            logits = []
            for sub in self.models:
                logit = self._forward_single_logits(x, sub['lstm'], sub['dropout'],
                                                     sub['fc'])
                logits.append(logit)
            return torch.stack(logits).mean(0)
        else:
            return self._forward_single_logits(x, self.lstm, self.dropout, self.fc)

    # Training-time input noise: add Gaussian noise to scaled features.
    # Forces the model to learn robust patterns rather than memorizing exact
    # feature values. This is the single highest-WR technique (53% avg WR
    # in attempt 139) but previously only applied in candidate training —
    # now embedded in the model so walk-forward retraining also uses it.
    # Reduced from 0.08 to 0.03: attempts 259-268 all produced 0 trades at 0.6
    # threshold with Q99(scores)=0.47. The regularizer stack (0.08 noise +
    # 0.3 dropout + 1e-3 weight_decay + 3.5x pos_weight + sequence_normalize)
    # collapsed scores into "never confident" local minimum. 0.03 preserves
    # robustness to feature magnitude noise while letting the model commit
    # to positive logits for genuine high-signal patterns, unblocking trades.
    INPUT_NOISE_STD = 0.03

    def forward(self, x):
        if self._seq_normalize.item():
            x = self._normalize_sequence(x)
        x = self.input_dropout(x)
        if self.training and self.INPUT_NOISE_STD > 0:
            x = x + torch.randn_like(x) * self.INPUT_NOISE_STD
        if self.use_mlp:
            if self.n_ensemble > 1:
                agg = self._aggregate_sequence(x)
                preds = []
                for sub in self.models:
                    out = sub['mlp_hidden'](agg)
                    logit = sub['fc'](out).squeeze(-1)
                    pred = self.sigmoid(logit / self._output_temp)
                    preds.append(pred)
                return torch.stack(preds).mean(0)
            return self._forward_mlp(x)
        if self.n_ensemble > 1:
            preds = []
            for sub in self.models:
                pred = self._forward_single(x, sub['lstm'], sub['dropout'],
                                            sub['fc'])
                preds.append(pred)
            return torch.stack(preds).mean(0)
        else:
            return self._forward_single(x, self.lstm, self.dropout, self.fc)

    def load_state_dict(self, state_dict, strict=True):
        # Auto-detect ensemble from key prefix "models."
        ensemble_keys = [k for k in state_dict if k.startswith('models.')]
        if ensemble_keys:
            n = max(int(k.split('.')[1]) for k in ensemble_keys) + 1
            self._setup_ensemble(n)
        # Drop stale attention keys from old models that used a learned attention layer
        # (current implementation uses mean pooling with no learnable params)
        stale_keys = [k for k in state_dict if k.startswith('attention.')]
        if stale_keys and not hasattr(self, 'attention'):
            for k in stale_keys:
                del state_dict[k]
        # Backward compat: old models don't have output_norm keys.
        # If loading an old model, remove output_norm from this module so
        # hasattr check in forward skips it (effectively identity transform).
        if hasattr(self, 'output_norm') and 'output_norm.weight' not in state_dict:
            delattr(self, 'output_norm')
        # Backward compat: old models don't have _output_temp buffer.
        # Use 1.0 (original default) so old models behave identically.
        if '_output_temp' not in state_dict:
            state_dict['_output_temp'] = torch.tensor(1.0)
        # Backward compat: old models don't have _seq_normalize buffer.
        # Default to 0 (disabled) so old models behave identically.
        if '_seq_normalize' not in state_dict:
            state_dict['_seq_normalize'] = torch.tensor(0)
        super().load_state_dict(state_dict, strict=strict)
