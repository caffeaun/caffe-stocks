"""Pluggable trainer abstraction for the v1+ ML pipeline.

A Trainer wraps a binary classifier with a uniform interface used by
scripts/trainer.py and scripts/return_gate.py. Add new model types by
subclassing BaseTrainer and registering in TRAINERS.

v1 ships LightGBM (default) and XGBoost. The NN slot is reserved
intentionally — when the strategy team wants to revisit sequence models,
add NeuralTrainer in a separate file and register it here. The trainer
script and walk-forward gate keep working with no other changes.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Optional

import numpy as np

# Workaround for current host: NVML library (595.71) is newer than the loaded
# kernel driver, so torch.cuda's NVML-driven device-count check inside the
# CUDA caching allocator hits "NVML_SUCCESS == nvmlInit_v2_() ASSERT FAILED"
# on multi-GPU enumeration. backend:native skips the NVML path, and pinning
# to GPU 0 sidesteps the multi-device enumeration entirely. Both via
# setdefault so an explicit cron-level override still wins.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'backend:native')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


class BaseTrainer:
    """Interface for binary classifiers in the v1 pipeline."""

    name: str = 'base'

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        """Fit the model. pnl_train/pnl_val carry per-trade realized P&L
        (fraction; 0.05 = +5%) so regression-head trainers can predict EV
        directly. Classifier trainers ignore them — the binary y_train remains
        the canonical target for them. dates_train/dates_val carry the entry
        date strings so ranker-head trainers can group rows by date for
        pairwise loss; other trainers ignore them. All extras are always
        passed by the gate so trainers can opt in without API churn."""
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        """Return P(y=1) for each row of X. Shape: (N,)."""
        raise NotImplementedError

    def save(self, output_dir: str, extra: Optional[dict] = None) -> dict:
        """Persist artifacts. Returns dict of paths written."""
        raise NotImplementedError

    @classmethod
    def load(cls, output_dir: str):
        """Load a previously saved trainer."""
        raise NotImplementedError

    def feature_importance(self) -> Optional[np.ndarray]:
        """Per-feature importance score, or None if not supported."""
        return None

    @property
    def best_iteration(self) -> Optional[int]:
        """Best iteration / round / epoch found during fit, if applicable."""
        return None

    @property
    def hyperparams(self) -> dict:
        """Public view of the hyperparameter dict for logging / metadata."""
        return {}


# --------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------- #
class LightGBMTrainer(BaseTrainer):
    name = 'lightgbm'

    def __init__(self,
                 num_leaves: int = 31,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 n_estimators: int = 500,
                 min_child_samples: int = 50,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 0.1,
                 early_stopping_rounds: int = 30,
                 pos_class_weight: float = 1.5,
                 random_state: int = 42):
        self._params = dict(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            early_stopping_rounds=early_stopping_rounds,
            pos_class_weight=pos_class_weight,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import lightgbm as lgb

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        self.clf = lgb.LGBMClassifier(
            objective='binary',
            num_leaves=p['num_leaves'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            n_estimators=p['n_estimators'],
            min_child_samples=p['min_child_samples'],
            subsample=p['subsample'],
            subsample_freq=1,
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            class_weight={0: 1.0, 1: p['pos_class_weight']},
            random_state=p['random_state'],
            n_jobs=-1,
            verbose=-1,
        )
        callbacks = [lgb.early_stopping(stopping_rounds=p['early_stopping_rounds'],
                                          verbose=verbose)]
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))

        self.clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=callbacks,
        )
        self._best_iteration = (self.clf.best_iteration_
                                  if self.clf.best_iteration_ else self.clf.n_estimators)
        return self

    def predict_proba(self, X):
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict_proba(X)[:, 1]

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.booster_.feature_importance(importance_type='gain')

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.txt')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.booster_.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import lightgbm as lgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.txt')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = lgb.LGBMClassifier()
        inst.clf._Booster = lgb.Booster(model_file=booster_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------- #
class XGBoostTrainer(BaseTrainer):
    name = 'xgboost'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 early_stopping_rounds: int = 30,
                 pos_class_weight: Optional[float] = None,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            early_stopping_rounds=early_stopping_rounds,
            pos_class_weight=pos_class_weight,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        # Auto-compute scale_pos_weight if not provided
        if p['pos_class_weight'] is None:
            pos_rate = float(np.mean(y_train))
            spw = float(min((1.0 - pos_rate) / max(pos_rate, 1e-6), 15.0))
        else:
            spw = float(p['pos_class_weight'])

        self.clf = xgb.XGBClassifier(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            scale_pos_weight=spw,
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
        )
        self._best_iteration = getattr(self.clf, 'best_iteration', None) or p['n_estimators']
        return self

    def predict_proba(self, X):
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict_proba(X)[:, 1]

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.feature_importances_

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = xgb.XGBClassifier()
        inst.clf.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# XGBoost Regressor — predicts realized pnl, not P(win)
# --------------------------------------------------------------------- #
# Motivation (from claude iter #10 lesson): every classifier in the loop has
# pinned WR ~30-42% with avg_pnl mildly negative. P(win)-ranking doesn't
# translate to positive EV because clean-win cap squeezes 1-class wins to
# +0.02 while losses keep -0.04 (commission + stop). Selection by predicted
# pnl directly optimises EV — a 60%-WR trade with 1% avg_pnl loses to a
# 35%-WR trade with 4% avg_pnl, and only a regression head sees that.
#
# predict_proba returns sigmoid(pred_pnl * SIGMOID_SCALE) so the existing
# SCORE_THRESHOLDS [0.30..0.70] in return_gate.py still carve a meaningful
# slice of the prediction range (typical pred_pnl ∈ [-0.05, +0.10] → score
# ∈ [0.27, 0.88]). Rank order within a date is preserved exactly.
class XGBoostRegressorTrainer(BaseTrainer):
    name = 'xgb_regressor'

    SIGMOID_SCALE = 20.0

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L). '
                'The gate must pass them through trainer.fit().')

        p = self._params
        self.clf = xgb.XGBRegressor(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='reg:squarederror',
            eval_metric='rmse',
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf.fit(
            X_train, np.asarray(pnl_train, dtype=np.float32),
            eval_set=[(X_val, np.asarray(pnl_val, dtype=np.float32))],
            verbose=verbose,
        )
        self._best_iteration = getattr(self.clf, 'best_iteration', None) or p['n_estimators']
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        pred_pnl = self.clf.predict(X)
        return 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * pred_pnl))

    def predict_pnl(self, X) -> np.ndarray:
        """Raw predicted P&L (fraction). Useful for diagnostics."""
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict(X)

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.feature_importances_

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = xgb.XGBRegressor()
        inst.clf.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# Bagged XGBoost Regressor — variance-reducing ensemble with confidence-
# penalized scoring.
# --------------------------------------------------------------------- #
# Motivation: single-model xgb_regressor produces high cross-window variance
# (recent iters #361-#368 wp=0..4/7 with ann swinging from -27% to +49%
# across HPs that are nearly identical). The 100%-pass gate rejects every
# such model. The same booster that hits ann=+48.9% in one HP config (#365)
# also crashes to wp=3/7 — meaning per-window picks are unstable, not the HP.
#
# Bagging K=5 xgb_regressor learners with different seeds AND row-bootstrap
# subsamples reduces prediction variance by ~1/K on independent components and
# leaves correlated bias unchanged (classic Breiman 1996). On its own this
# helps but doesn't necessarily move the per-window pass rate; what does is
# the confidence-penalized scoring:
#
#     score(row) = mean_k(pred_k(row)) - conf_lambda * std_k(pred_k(row))
#
# Rows where the K bags AGREE on a positive EV survive the penalty; rows
# where bags disagree (one bag predicts +5%, another predicts -2% — the
# lottery-ticket signature) are pushed down by the std term. Combined with
# the gate's top-K-per-day selection, this filters trades by *consensus*
# rather than by *aggressive single-model conviction*. The expected effect:
# fewer big-ann/low-wp HP configs, more cross-window stability around a
# slightly lower per-window ann — exactly the trade the 100%-pass gate
# requires.
#
# Same XGBoost-tree HP space as xgb_regressor (sub-bag HP intuitions carry
# over) plus three ensemble knobs: n_bags ∈ [3,5,7], conf_lambda ∈ [0.0, 3.0],
# bootstrap_frac ∈ [0.7, 1.0]. conf_lambda=0 collapses to plain bagging
# (variance reduction without confidence gating); >2.0 makes the scorer
# heavily defensive (often produces 0 trades on noisy windows). The sweep
# should find the slope where consensus filtering peaks WR without starving
# trade count below the 20-trade-per-window floor.
class BaggedXGBRegressorTrainer(BaseTrainer):
    name = 'bagged_xgb_regressor'

    SIGMOID_SCALE = 20.0

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 early_stopping_rounds: int = 30,
                 n_bags: int = 5,
                 conf_lambda: float = 1.0,
                 bootstrap_frac: float = 0.85,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            early_stopping_rounds=early_stopping_rounds,
            n_bags=n_bags,
            conf_lambda=conf_lambda,
            bootstrap_frac=bootstrap_frac,
            random_state=random_state,
        )
        self.bags: list = []
        self._best_iteration = None

    def _make_bag(self, seed: int):
        import xgboost as xgb
        p = self._params
        return xgb.XGBRegressor(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='reg:squarederror',
            eval_metric='rmse',
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=seed,
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (regression-on-pnl head).')

        p = self._params
        n_train = len(X_train)
        n_sample = max(int(round(p['bootstrap_frac'] * n_train)), 200)
        rs_root = np.random.RandomState(p['random_state'])
        pnl_train_arr = np.asarray(pnl_train, dtype=np.float32)
        pnl_val_arr = np.asarray(pnl_val, dtype=np.float32)

        self.bags = []
        best_iters = []
        for k in range(p['n_bags']):
            # Row-level bootstrap (with replacement) — classic bagging. Each bag
            # sees a different ~bootstrap_frac slice of train rows, which in
            # combination with per-bag seeds + XGBoost's internal subsample
            # produces decorrelated predictors without over-shrinking individual
            # train sets (which would hurt early stopping signal quality).
            sample_seed = int(rs_root.randint(0, 2**31 - 1))
            sub_idx = np.random.RandomState(sample_seed).choice(
                n_train, size=n_sample, replace=True)
            X_sub = X_train[sub_idx]
            pnl_sub = pnl_train_arr[sub_idx]

            booster_seed = int(rs_root.randint(0, 2**31 - 1))
            clf = self._make_bag(booster_seed)
            clf.fit(
                X_sub, pnl_sub,
                eval_set=[(X_val, pnl_val_arr)],
                verbose=verbose,
            )
            self.bags.append(clf)
            best_iters.append(getattr(clf, 'best_iteration', None) or p['n_estimators'])

        self._best_iteration = int(np.mean(best_iters))
        return self

    def _bag_predictions(self, X) -> np.ndarray:
        if not self.bags:
            raise RuntimeError('Model not fit')
        preds = np.stack([clf.predict(X) for clf in self.bags], axis=0)
        return preds  # shape (n_bags, n_rows)

    def predict_proba(self, X) -> np.ndarray:
        preds = self._bag_predictions(X)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0, ddof=0)
        score = mean_pred - self._params['conf_lambda'] * std_pred
        return 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * score))

    def predict_pnl(self, X) -> np.ndarray:
        """Confidence-penalized predicted P&L (fraction). Useful for diagnostics."""
        preds = self._bag_predictions(X)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0, ddof=0)
        return mean_pred - self._params['conf_lambda'] * std_pred

    def feature_importance(self):
        if not self.bags:
            return None
        # Average of bag-level importances. Each bag normalized to sum=1 so the
        # average is in comparable units regardless of n_estimators per bag.
        fis = []
        for clf in self.bags:
            fi = getattr(clf, 'feature_importances_', None)
            if fi is None:
                continue
            s = fi.sum()
            fis.append(fi / s if s > 0 else fi)
        if not fis:
            return None
        return np.mean(np.stack(fis, axis=0), axis=0)

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        bag_paths = []
        for k, clf in enumerate(self.bags):
            bag_path = os.path.join(output_dir, f'bag_{k:02d}.json')
            clf.save_model(bag_path)
            bag_paths.append(bag_path)
        meta_path = os.path.join(output_dir, 'metadata.json')
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
            'n_bags_saved': len(bag_paths),
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'bags': bag_paths, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        n_saved = meta.get('n_bags_saved', 0)
        inst.bags = []
        for k in range(n_saved):
            bag_path = os.path.join(output_dir, f'bag_{k:02d}.json')
            clf = xgb.XGBRegressor()
            clf.load_model(bag_path)
            inst.bags.append(clf)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# XGBoost Huber Regressor — pseudo-Huber loss, outlier-robust EV head
# --------------------------------------------------------------------- #
# Motivation: xgb_regressor uses reg:squarederror, which is convex but its
# gradient grows linearly with residual magnitude — a single +15% target hit
# (mean pnl ≈ +13.9% net) produces 5× the gradient of a -3% stop. On thin
# 2024 train chunks the squared-error model chases the few rare big wins,
# producing high-variance picks (recent iters #346-#353: ann swings -24%..
# +27% across HPs while WR stays stuck at 28-39%, never clearing the 40%
# per-window floor).
#
# Pseudo-Huber loss (reg:pseudohubererror, slope δ) is quadratic for
# residuals |r|<δ and linear beyond — same convexity / efficiency as MSE in
# the dense middle of the pnl distribution but bounded gradient on the
# 15%-target / -3%-stop tails. The expected effect: fewer "lottery-ticket"
# picks, more density in the 4-7% gain zone where MIN_PROFIT_PCT=0.04
# decides the WR floor.
#
# huber_slope ∈ [0.02, 0.10] spans the relevant regime: 0.02 ≈ MIN_PROFIT_PCT/2
# (treats almost every trade as a tail event), 0.10 ≈ trailing-trigger / max-pnl
# bounds (close to plain L2). The sweep should find a slope where the gradient
# transitions right around the win/loss boundary.
#
# Identical sigmoid-on-prediction trick as xgb_regressor so SCORE_THRESHOLDS
# in return_gate.py keep working without changes. Distinct from
# xgb_quantile (P70 instead of robust mean) and xgb_dual_quantile (asymmetric
# upper-vs-lower estimators) — all three approach the right-skew problem
# differently, giving the pipeline genuine algorithmic diversity.
class XGBoostHuberRegressorTrainer(BaseTrainer):
    name = 'xgb_huber_regressor'

    SIGMOID_SCALE = 20.0

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 huber_slope: float = 0.05,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            huber_slope=huber_slope,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L). '
                'The gate must pass them through trainer.fit().')

        p = self._params
        self.clf = xgb.XGBRegressor(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='reg:pseudohubererror',
            huber_slope=p['huber_slope'],
            eval_metric='mphe',
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf.fit(
            X_train, np.asarray(pnl_train, dtype=np.float32),
            eval_set=[(X_val, np.asarray(pnl_val, dtype=np.float32))],
            verbose=verbose,
        )
        self._best_iteration = getattr(self.clf, 'best_iteration', None) or p['n_estimators']
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        pred_pnl = self.clf.predict(X)
        return 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * pred_pnl))

    def predict_pnl(self, X) -> np.ndarray:
        """Raw predicted P&L (fraction). Useful for diagnostics."""
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict(X)

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.feature_importances_

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = xgb.XGBRegressor()
        inst.clf.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# XGBoost Quantile Regressor — predicts an upper quantile of pnl
# --------------------------------------------------------------------- #
# Motivation (from claude iter #31 lesson): xgb_regressor with squared-error
# loss was the best run to date (WR 44%, DD 9.9%, 5/7 windows positive) but
# blew up in early-2024 windows where WR fell to 23-29%. Squared-error
# regression predicts the conditional MEAN E[pnl|X], which gets dragged by
# rare big wins in the right tail; in noisy regimes the mean stays mildly
# positive even when the bulk of trades lose.
#
# Quantile regression at alpha=0.7 instead predicts P70(pnl|X). A positive
# P70 means at least 70% of conditional outcomes are above zero, which bakes
# in a precision filter aligned with the gate's MIN_WR=30% (alpha=0.7 → at
# least 30% of trades must be wins for the prediction to be positive).
# This should reject noisy-regime trades the squared-error model accepted.
#
# Identical sigmoid-on-prediction trick as xgb_regressor so SCORE_THRESHOLDS
# in return_gate.py keep working without changes.
class XGBoostQuantileTrainer(BaseTrainer):
    name = 'xgb_quantile'

    SIGMOID_SCALE = 20.0

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 quantile_alpha: float = 0.7,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            quantile_alpha=quantile_alpha,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L). '
                'The gate must pass them through trainer.fit().')

        p = self._params
        self.clf = xgb.XGBRegressor(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='reg:quantileerror',
            quantile_alpha=p['quantile_alpha'],
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf.fit(
            X_train, np.asarray(pnl_train, dtype=np.float32),
            eval_set=[(X_val, np.asarray(pnl_val, dtype=np.float32))],
            verbose=verbose,
        )
        self._best_iteration = getattr(self.clf, 'best_iteration', None) or p['n_estimators']
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        pred_q = self.clf.predict(X)
        return 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * pred_q))

    def predict_pnl(self, X) -> np.ndarray:
        """Raw predicted quantile of P&L (fraction). Useful for diagnostics."""
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict(X)

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.feature_importances_

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = xgb.XGBRegressor()
        inst.clf.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# LightGBM Regressor — predicts realized pnl, leaf-wise growth + GOSS
# --------------------------------------------------------------------- #
# Motivation: xgb_regressor (#31) is the best structural change to date —
# 5/7 positive windows, 44% WR, 9.9% DD — confirming EV-ranking beats
# P(win)-ranking. Three of four registered trainers are XGBoost variants;
# all share level-wise tree growth and the same histogram binning. The model
# space has algorithmic monoculture even after adding the regression head.
#
# LightGBM's leaf-wise growth picks the leaf with max delta-loss (more
# aggressive splits, better at finding rare-event patterns) and GOSS keeps
# all large-gradient samples while sub-sampling small-gradient ones — both
# diverge from XGBoost's defaults in ways that matter on noisy ~200-1000 row
# train chunks. An EV-prediction head built on this different bias should
# disagree with xgb_regressor on the marginal trades, which is exactly the
# diversity that fails the simple avg-prob ensemble (#10) needed.
#
# Same sigmoid-on-prediction trick as xgb_regressor so SCORE_THRESHOLDS in
# return_gate.py keep working without changes.
class LightGBMRegressorTrainer(BaseTrainer):
    name = 'lightgbm_regressor'

    SIGMOID_SCALE = 20.0

    def __init__(self,
                 num_leaves: int = 31,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 n_estimators: int = 500,
                 min_child_samples: int = 50,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 0.1,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import lightgbm as lgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L). '
                'The gate must pass them through trainer.fit().')

        p = self._params
        self.clf = lgb.LGBMRegressor(
            objective='regression',
            num_leaves=p['num_leaves'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            n_estimators=p['n_estimators'],
            min_child_samples=p['min_child_samples'],
            subsample=p['subsample'],
            subsample_freq=1,
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            random_state=p['random_state'],
            n_jobs=-1,
            verbose=-1,
        )
        callbacks = [lgb.early_stopping(stopping_rounds=p['early_stopping_rounds'],
                                          verbose=verbose)]
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))

        self.clf.fit(
            X_train, np.asarray(pnl_train, dtype=np.float32),
            eval_set=[(X_val, np.asarray(pnl_val, dtype=np.float32))],
            eval_metric='rmse',
            callbacks=callbacks,
        )
        self._best_iteration = (self.clf.best_iteration_
                                  if self.clf.best_iteration_ else self.clf.n_estimators)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        pred_pnl = self.clf.predict(X)
        return 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * pred_pnl))

    def predict_pnl(self, X) -> np.ndarray:
        """Raw predicted P&L (fraction). Useful for diagnostics."""
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict(X)

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.booster_.feature_importance(importance_type='gain')

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.txt')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.booster_.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import lightgbm as lgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.txt')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = lgb.LGBMRegressor()
        inst.clf._Booster = lgb.Booster(model_file=booster_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# XGBoost Dual-Quantile — joint upper + lower quantile, downside-aware ranking
# --------------------------------------------------------------------- #
# Motivation (from claude iter #32 lesson, finally getting around to it):
# xgb_quantile@0.7 (#32) was a precision filter — fixed window 4 (+16% → +65%)
# but regressed on windows 5, 7. The single-tail view captured upside but was
# blind to the left tail: when right-tail signal is real but downside is
# unbounded (the noisy ~200-1000 row train chunks are most exposed to this),
# alpha=0.7 still picked bets whose P25 was deeply negative.
#
# Dual quantile fits TWO XGBRegressors at alpha_lower (e.g. 0.25) and
# alpha_upper (e.g. 0.75). Ranking score combines:
#     score = pred_upper - dd_penalty * max(0, -pred_lower)
# When pred_lower is negative (likely loss in the lower 25% of cases), the
# penalty subtracts from upside; when pred_lower is positive (clean upside-
# only bet), full upper passes through. This is the conditional-VaR-aware
# selection rule explicitly proposed in #32's lesson — it should reject the
# noisy-regime trades that single-quantile and squared-error regressors picked
# even when downside was uncomfortable.
#
# Two independent XGBRegressor objects (XGBoost can't fit two quantiles in a
# single model with reg:quantileerror — only one alpha per call). Sigmoid trick
# preserved so SCORE_THRESHOLDS in return_gate.py keep working.
class XGBoostDualQuantileTrainer(BaseTrainer):
    name = 'xgb_dual_quantile'

    SIGMOID_SCALE = 20.0

    def __init__(self,
                 max_depth: int = 6,
                 learning_rate: float = 0.07,
                 n_estimators: int = 400,
                 subsample: float = 0.7,
                 colsample_bytree: float = 0.78,
                 reg_alpha: float = 0.37,
                 reg_lambda: float = 1.46,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.01,
                 alpha_lower: float = 0.35,
                 alpha_upper: float = 0.65,
                 dd_penalty: float = 1.0,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        # Defaults seeded from iter #58's best tree HPs + tight quantiles + low
        # dd_penalty=1.0 (symmetric weighting). Wider tails (0.25/0.75) with
        # dd_penalty=2.0 aggressively over-penalized noisy 200-1000 row chunks
        # at the gate (avg ann -28.7% at first probe). Train mode will sweep
        # the full ranges in models/search_spaces.py.
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            dd_penalty=dd_penalty,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf_lower = None
        self.clf_upper = None
        self._best_iteration = None

    def _make_regressor(self, alpha):
        import xgboost as xgb
        p = self._params
        return xgb.XGBRegressor(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L). '
                'The gate must pass them through trainer.fit().')

        p = self._params
        pnl_tr = np.asarray(pnl_train, dtype=np.float32)
        pnl_va = np.asarray(pnl_val, dtype=np.float32)

        self.clf_lower = self._make_regressor(p['alpha_lower'])
        self.clf_lower.fit(X_train, pnl_tr, eval_set=[(X_val, pnl_va)], verbose=verbose)

        self.clf_upper = self._make_regressor(p['alpha_upper'])
        self.clf_upper.fit(X_train, pnl_tr, eval_set=[(X_val, pnl_va)], verbose=verbose)

        bi_lower = getattr(self.clf_lower, 'best_iteration', None) or p['n_estimators']
        bi_upper = getattr(self.clf_upper, 'best_iteration', None) or p['n_estimators']
        self._best_iteration = max(bi_lower, bi_upper)
        return self

    def _combined_score(self, X) -> np.ndarray:
        pred_lower = self.clf_lower.predict(X)
        pred_upper = self.clf_upper.predict(X)
        downside = np.maximum(0.0, -pred_lower)
        return pred_upper - self._params['dd_penalty'] * downside

    def predict_proba(self, X) -> np.ndarray:
        if self.clf_lower is None or self.clf_upper is None:
            raise RuntimeError('Model not fit')
        score = self._combined_score(X)
        return 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * score))

    def predict_pnl(self, X) -> np.ndarray:
        """Raw combined score (upper_q - dd_penalty * downside). Diagnostics."""
        if self.clf_lower is None or self.clf_upper is None:
            raise RuntimeError('Model not fit')
        return self._combined_score(X)

    def feature_importance(self):
        if self.clf_upper is None:
            return None
        # Average of two heads — both contribute to the ranking score.
        fi_upper = self.clf_upper.feature_importances_
        fi_lower = self.clf_lower.feature_importances_
        return 0.5 * (fi_upper + fi_lower)

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        lower_path = os.path.join(output_dir, 'booster_lower.json')
        upper_path = os.path.join(output_dir, 'booster_upper.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf_lower.save_model(lower_path)
        self.clf_upper.save_model(upper_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model_lower': lower_path, 'model_upper': upper_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        lower_path = os.path.join(output_dir, 'booster_lower.json')
        upper_path = os.path.join(output_dir, 'booster_upper.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf_lower = xgb.XGBRegressor()
        inst.clf_lower.load_model(lower_path)
        inst.clf_upper = xgb.XGBRegressor()
        inst.clf_upper.load_model(upper_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# XGBoost Pairwise Ranker — date-grouped learning-to-rank
# --------------------------------------------------------------------- #
# Motivation: at gate time the deployment task is per-date ranking — pick the
# highest-scoring symbol on each trading date, only trade when score >=
# threshold. Every prior head (classifier, EV regressor, quantile, dual-
# quantile) predicts an absolute magnitude (P(win), E[pnl], Q_alpha(pnl),
# upper_q - penalty * downside) and *hopes* good rankings emerge. A direct
# pairwise ranking loss trains on the actual ranking task: minimize the count
# of pairs (a, b) within a date group where pred(less-pnl) > pred(more-pnl).
#
# Structurally distinct from xgb_dual_quantile (#64, the best result to date
# at avg ann +41.9%, WR 45.7%, DD 11.6% but only 2/7 windows passing): same
# tree family but the loss only sees *relative ordering within each date
# group*, never absolute pnl. Robustness arguments:
#   - regime pnl-scale shifts: a +2% bull day and a -1% chop day are the same
#     per-date ranking problem; mean/quantile losses see them as different
#     prediction tasks because the conditional means/quantiles shift.
#   - outlier-driven mean drag: pairwise loss is bounded per-pair (logistic-
#     style on margin), not dominated by a single +15% target hit pulling the
#     conditional mean upward.
#   - cross-day calibration noise: ranker scores need not be on a consistent
#     absolute scale across dates — dual_quantile/regressor heads need to
#     predict on a coherent absolute scale across all regimes for the gate's
#     SCORE_THRESHOLDS sweep to work, the ranker only needs within-date order.
#
# Group structure: dates_train / dates_val arrays passed via fit() are bucketed
# into per-date groups (consecutive same-date rows after stable sort). XGBRanker
# learns pairwise comparisons of pnl across rows within each date.
#
# Predict-time: raw ranker output → sigmoid(SIGMOID_SCALE * pred). Output is
# unbounded but typically O(1) magnitude with default tree HPs, so SIGMOID_SCALE
# = 1.0 maps the practical [-3, +3] range to roughly (0.05, 0.95) which spans
# the gate's SCORE_THRESHOLDS [0.30..0.70]. Order is preserved exactly so the
# gate's per-date highest-score selection works correctly.
class XGBoostRankerTrainer(BaseTrainer):
    name = 'xgb_ranker'

    # Ranker output is unbounded; empirically scores have std ~0.05-0.3 on real
    # data depending on tree depth / n_estimators. SCALE=10 maps a typical
    # ±0.2 range onto sigmoid(±2) → roughly (0.12, 0.88) which spans the
    # gate's SCORE_THRESHOLDS [0.30..0.70] with discrimination at every level.
    SIGMOID_SCALE = 10.0

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    @staticmethod
    def _group_counts(dates):
        """Sort indices by date and return contiguous-group counts.

        XGBRanker requires `group=[count_g0, count_g1, ...]` where rows are
        already laid out so each group's rows are contiguous. We achieve that
        with a stable sort on dates, then run-length-encode the sorted dates.
        Returns (sort_order, group_counts).
        """
        d = np.asarray(dates)
        order = np.argsort(d, kind='stable')
        # np.unique on the sorted array gives counts in sorted order — exactly
        # what XGBoost expects since after `[order]` the rows are contiguous.
        _, counts = np.unique(d[order], return_counts=True)
        return order, counts.astype(np.int64)

    @staticmethod
    def _within_group_rank(values, group_counts):
        """Replace each value with its 0-indexed rank within its group.

        XGBoost ranking requires integer relevance labels. For pairwise loss
        the absolute scale doesn't matter — only ordinal position within
        each group does. So per-group rank IS the natural relevance: the
        row with highest pnl in a date gets rank=cnt-1, lowest gets rank=0.
        """
        out = np.empty_like(values, dtype=np.int32)
        idx = 0
        for cnt in group_counts:
            chunk = values[idx:idx + cnt]
            # argsort(argsort(...)) gives 0..cnt-1 ordinal ranks
            ranks = np.argsort(np.argsort(chunk, kind='stable'), kind='stable')
            out[idx:idx + cnt] = ranks.astype(np.int32)
            idx += cnt
        return out

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L).')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for date-group ranking. '
                'The gate must pass them through trainer.fit().')

        order_tr, groups_tr = self._group_counts(dates_train)
        order_va, groups_va = self._group_counts(dates_val)
        X_tr_sorted = np.asarray(X_train)[order_tr]
        pnl_tr_sorted = np.asarray(pnl_train, dtype=np.float32)[order_tr]
        X_va_sorted = np.asarray(X_val)[order_va]
        pnl_va_sorted = np.asarray(pnl_val, dtype=np.float32)[order_va]

        # Pairwise ranking needs at least one group with >= 2 samples; in
        # practice the ML pipeline has ~10-50 rows per date so this never trips.
        if (groups_tr < 2).all():
            raise ValueError(f'{self.name}: no train date has >=2 samples — '
                             'pairwise ranking is impossible')

        # XGBoost requires non-negative integer relevance labels (NDCG eval is
        # the default for ranker even with rank:pairwise objective). Per-group
        # rank of pnl gives the model exactly the within-date ordering signal it
        # needs without leaking absolute pnl scale.
        rel_tr = self._within_group_rank(pnl_tr_sorted, groups_tr)
        rel_va = self._within_group_rank(pnl_va_sorted, groups_va)

        p = self._params
        # ndcg_exp_gain=False — default exponential gain caps relevance at 31,
        # but per-group rank exceeds this on dates with >32 stocks (typical SET
        # universe). Linear DCG gain has no cap and works equally well as the
        # eval metric for early stopping.
        self.clf = xgb.XGBRanker(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='rank:pairwise',
            ndcg_exp_gain=False,
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf.fit(
            X_tr_sorted, rel_tr,
            group=groups_tr,
            eval_set=[(X_va_sorted, rel_va)],
            eval_group=[groups_va],
            verbose=verbose,
        )
        self._best_iteration = getattr(self.clf, 'best_iteration', None) or p['n_estimators']
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        pred = self.clf.predict(X)
        return 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * pred))

    def predict_score(self, X) -> np.ndarray:
        """Raw ranker score (unbounded). Useful for diagnostics."""
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict(X)

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.feature_importances_

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = xgb.XGBRanker()
        inst.clf.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# XGBoost Ranker (DART booster) — pairwise ranker with stochastic tree dropout
# --------------------------------------------------------------------- #
# Motivation: HP sweeping on xgb_ranker (#136-#143) plateaus at 5/7 windows
# with avg WR ~45% and DD <10% — never 7/7. The two failing windows are
# regime-specific: the model latches onto patterns that don't transfer
# across the calendar splits. This is overfitting expressed as regime
# variance, not training-set variance — early stopping caught the latter
# already.
#
# DART (Dropouts meet Multiple Additive Regression Trees, Rashmi & Gilad-
# Bachrach, AISTATS 2015) is XGBoost's native answer to this. Each boosting
# round drops a random subset of already-grown trees (rate_drop) before
# fitting the new tree against the *partial* ensemble's residuals; with
# probability skip_drop the round skips dropping entirely (so training
# remains progressive). Effects:
#   - Forces each tree to encode a redundant, broadly-useful signal rather
#     than a specialized correction to a specific narrow residual pattern.
#   - Acts like a stochastic deep ensemble inside the single booster: at
#     inference, all trees are kept, but they were trained against many
#     random sub-ensembles.
#   - Empirically reduces train/test divergence on small/noisy tabular data
#     where standard GBT overfits the latest few residual patterns.
#
# Why this trainer (not just a flag on xgb_ranker): DART introduces two
# extra HPs (rate_drop, skip_drop) and changes the inference path subtly —
# e.g. early stopping on DART evaluates against the dropped sub-ensemble,
# so its best_iteration semantics differ. Keeping it as a sibling trainer
# lets train mode HP-sweep DART independently without contaminating the
# xgb_ranker search space, and lets feedback attribute wins cleanly to
# either standard boosting or DART regularization.
#
# Inherits all the date-group ranking machinery from XGBoostRankerTrainer:
# same _group_counts / _within_group_rank helpers, same SIGMOID_SCALE
# mapping (DART scores have similar magnitude to GBT ranker scores —
# trees are merged at inference, only the training schedule differs).
class XGBoostRankerDartTrainer(XGBoostRankerTrainer):
    name = 'xgb_ranker_dart'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 rate_drop: float = 0.1,
                 skip_drop: float = 0.5,
                 sample_type: str = 'uniform',
                 normalize_type: str = 'tree',
                 early_stopping_rounds: int = 50,
                 random_state: int = 42):
        # Skip the XGBoostRankerTrainer __init__ — we own _params layout.
        # rate_drop ∈ [0, 1]: fraction of existing trees to drop each round.
        #   0.0 collapses to GBT; >0.3 is unstable on small data. Default 0.1
        #   is the value Rashmi & Gilad-Bachrach found best across UCI tasks.
        # skip_drop ∈ [0, 1]: probability a round skips dropping entirely.
        #   0.5 means half the rounds train against the full prior ensemble
        #   (fast progress); the other half against random sub-ensembles
        #   (regularization). Decoupling these two knobs is what gives DART
        #   its tuneable expressiveness vs plain bagging.
        # sample_type: 'uniform' draws drop set uniformly; 'weighted' favors
        #   trees with larger residual contribution — usually slightly better
        #   on noisy data, but more sensitive to outliers. Default 'uniform'.
        # normalize_type: 'tree' rescales each surviving tree by 1/(k+1) where
        #   k = #dropped trees that round, preserving total tree weight.
        #   'forest' rescales by 1/(k+lr); typically marginal effect.
        # early_stopping_rounds raised 30→50: DART's stochastic dropout means
        #   eval-loss curves are noisier than GBT's, so early stopping needs
        #   a wider patience window to avoid stopping on noise.
        BaseTrainer.__init__(self)
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            rate_drop=rate_drop,
            skip_drop=skip_drop,
            sample_type=sample_type,
            normalize_type=normalize_type,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L).')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for date-group ranking. '
                'The gate must pass them through trainer.fit().')

        order_tr, groups_tr = self._group_counts(dates_train)
        order_va, groups_va = self._group_counts(dates_val)
        X_tr_sorted = np.asarray(X_train)[order_tr]
        pnl_tr_sorted = np.asarray(pnl_train, dtype=np.float32)[order_tr]
        X_va_sorted = np.asarray(X_val)[order_va]
        pnl_va_sorted = np.asarray(pnl_val, dtype=np.float32)[order_va]

        if (groups_tr < 2).all():
            raise ValueError(f'{self.name}: no train date has >=2 samples — '
                             'pairwise ranking is impossible')

        rel_tr = self._within_group_rank(pnl_tr_sorted, groups_tr)
        rel_va = self._within_group_rank(pnl_va_sorted, groups_va)

        p = self._params
        self.clf = xgb.XGBRanker(
            booster='dart',
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            rate_drop=p['rate_drop'],
            skip_drop=p['skip_drop'],
            sample_type=p['sample_type'],
            normalize_type=p['normalize_type'],
            objective='rank:pairwise',
            ndcg_exp_gain=False,
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf.fit(
            X_tr_sorted, rel_tr,
            group=groups_tr,
            eval_set=[(X_va_sorted, rel_va)],
            eval_group=[groups_va],
            verbose=verbose,
        )
        self._best_iteration = getattr(self.clf, 'best_iteration', None) or p['n_estimators']
        return self

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}


# --------------------------------------------------------------------- #
# XGBoost WinRanker — date-grouped LambdaRank with BINARY win relevance
# --------------------------------------------------------------------- #
# Motivation: xgb_ranker / xgb_ranker_dart use within-group rank of pnl as the
# relevance label, so the lambdarank loss is dominated by magnitude — putting
# +14% (target hit) above +5% (trailing) carries the same gradient as putting
# +5% above -3% (stop). The gate, however, scores WR≥40% per window — it does
# not care whether the picked symbol gained 5% or 14%, only whether pnl >
# MIN_PROFIT_PCT (0.04). The pnl-rank ranker therefore optimizes a strictly
# stronger objective than the gate cares about, and the extra capacity goes
# into chasing tail-magnitude patterns that don't generalize across regimes.
#
# Train mode just spent 8 iterations sweeping bagged_xgb_regressor HPs (#371-
# #378): WR landed in [27%, 37%], never clearing the 40% per-window gate
# despite +22.6% ann in the best-WR run. The bottleneck is a discrimination
# problem — the regression head can't separate winners from losers, only
# extrapolate magnitude — and HP tuning won't fix it.
#
# This trainer uses the BINARY y label (1 if pnl > MIN_PROFIT_PCT else 0) as
# relevance directly, with rank:ndcg + ndcg@2 evaluation. NDCG@2 matches
# MAX_OPEN_POSITIONS=2 in return_gate.simulate_window — it scores a perfect 1.0
# when both top-2 picks per date are winners, and degrades smoothly as winners
# fall out of the top-2. Since the gate picks top-K-per-date, this is the
# closest objective the modeling layer can express to the gate's actual
# scoreboard.
#
# Three properties make this distinct from the existing rankers:
#   1. Binary relevance: pairwise gradients are uniform — winner > loser is the
#      *only* signal. No tail-chasing toward +15% targets, no penalty for
#      misranking two winners against each other.
#   2. NDCG@K objective: gradient is concentrated on top-K positions per group.
#      With ~30-60 candidates per date and K=2, ~95% of pairs the loss "sees"
#      involve at least one row that's currently in or near the top-2.
#   3. ndcg_at HP: Sweep [1, 2, 3] lets train mode test whether the gate's
#      effective K (via the [0.30..0.70] threshold sweep + max_open=2 cap) is
#      best matched by single-symbol-of-the-day, top-2, or a slight slack.
#
# SIGMOID_SCALE=10 mirrors xgb_ranker — raw rank:ndcg scores live in roughly
# the same magnitude range, and reusing the scale lets the gate's
# SCORE_THRESHOLDS work without per-trainer calibration.
class XGBoostWinRankerTrainer(XGBoostRankerTrainer):
    name = 'xgb_win_ranker'

    SIGMOID_SCALE = 10.0

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 ndcg_at: int = 2,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        BaseTrainer.__init__(self)
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            ndcg_at=int(ndcg_at),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for date-group ranking.')

        order_tr, groups_tr = self._group_counts(dates_train)
        order_va, groups_va = self._group_counts(dates_val)
        X_tr_sorted = np.asarray(X_train)[order_tr]
        y_tr_sorted = np.asarray(y_train, dtype=np.int32)[order_tr]
        X_va_sorted = np.asarray(X_val)[order_va]
        y_va_sorted = np.asarray(y_val, dtype=np.int32)[order_va]

        if (groups_tr < 2).all():
            raise ValueError(f'{self.name}: no train date has >=2 samples — '
                             'pairwise ranking is impossible')
        # rank:ndcg with all-zero (or all-one) groups gives zero gradient for
        # those groups, which is fine — the loss simply ignores them and
        # focuses signal on the groups where at least one winner coexists with
        # at least one loser.

        p = self._params
        # ndcg_exp_gain=False forces linear gain g(rel)=rel; with binary
        # {0,1} relevance this is equivalent to the default exp gain
        # 2^rel - 1, but it is the explicit choice we want documented (no
        # implicit non-linearity on top of an already-binary signal).
        self.clf = xgb.XGBRanker(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='rank:ndcg',
            eval_metric=f'ndcg@{p["ndcg_at"]}',
            ndcg_exp_gain=False,
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf.fit(
            X_tr_sorted, y_tr_sorted,
            group=groups_tr,
            eval_set=[(X_va_sorted, y_va_sorted)],
            eval_group=[groups_va],
            verbose=verbose,
        )
        self._best_iteration = getattr(self.clf, 'best_iteration', None) or p['n_estimators']
        return self


# --------------------------------------------------------------------- #
# EV-Gated Ranker — date-grouped win-relevance ranker × EV-regressor gate
# --------------------------------------------------------------------- #
# Motivation: the xgb_win_ranker / xgb_ranker family produces a top-K-per-day
# ordering signal but cannot abstain in regimes where every candidate is
# negative-EV. Recent xgb_win_ranker HP sweep (#387-#394) is stuck at WR
# 26-35% — well below the 40% per-window gate — even when annualized return
# is high (#389: ann=+52.7% but WR=33%, only 2/7 windows pass). The model
# picks the best of a bad bunch on regime-shift dates because the listwise
# objective only sees within-date pairs, not absolute EV.
#
# The closest any iteration has come to a candidate was claude #159
# (ev_gated_ranker, avg WR 65.5%, 6/7 windows passed; gate failed only on
# Split 2 where ev_blend=1.0 still admitted 31 negative-EV trades). Its
# trainer code was inadvertently dropped from trainers.py during a prior
# cleanup — only its .pyc remains. This reconstruction implements the same
# core idea with explicit gating semantics.
#
# Architecture: compose a date-grouped XGBRanker (binary-win relevance,
# rank:ndcg + ndcg@K eval) with a XGBRegressor predicting per-trade pnl
# (reg:squarederror). At inference the two outputs combine multiplicatively:
#
#     ranker_prob = sigmoid(SIGMOID_SCALE * ranker_raw)        # [0, 1]
#     ev_gate_raw = sigmoid(ev_scale * ev_pred)                # [0, 1], hinge at ev_pred=0
#     ev_gate     = ev_floor + (1 - ev_floor) * ev_gate_raw    # [ev_floor, 1]
#     final       = ranker_prob * ev_gate
#
# Behaviour:
#   ev_pred → +∞ : ev_gate → 1, final → ranker_prob (pure ranker, no shrinkage)
#   ev_pred = 0  : ev_gate = ev_floor + (1 - ev_floor)/2 (intermediate)
#   ev_pred → -∞ : ev_gate → ev_floor, final ≤ ev_floor (regime abstention)
#
# This means: when the EV regressor predicts negative pnl, the gate compresses
# the ranker score toward `ranker_prob × ev_floor`, dropping it below the
# threshold sweep's [0.30..0.70] range and effectively skipping that date.
# When EV is positive, the ranker's ordering is preserved unchanged.
#
# HPs (sweep target for train mode):
#   ev_scale ∈ [10, 100]: sigmoid steepness around ev_pred=0. Large = sharp
#     gate (small EV moves flip from 0→1); small = gentle gate (smooth
#     gradient). 30 ≈ "5% pnl gives gate ≈ 0.82, -3% pnl gives gate ≈ 0.29".
#   ev_floor ∈ [0.0, 0.5]: minimum gate value. 0.0 = hard gate (negative-EV
#     trades scored ≈ 0); 0.5 = soft gate (negative-EV trades penalized but
#     still selectable). Lower floor = more aggressive abstention.
#   ndcg_at ∈ {1, 2, 3}: top-K position the LambdaRank loss puts gradient
#     weight on. Same as xgb_win_ranker — K=2 matches MAX_OPEN_POSITIONS.
#
# Both sub-models share tree-family HPs (max_depth, learning_rate,
# n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda,
# min_child_weight, gamma) for parsimony — joint sweep first; if attribution
# becomes informative, split the HP space later. The regressor uses
# random_state+1 to break seed-correlation between the two sub-models.
class EVGatedRankerTrainer(XGBoostRankerTrainer):
    name = 'ev_gated_ranker'

    # Mirrors xgb_win_ranker — empirical raw-score range ±0.2 maps to
    # sigmoid(±2) ≈ (0.12, 0.88), spanning the gate's [0.30, 0.70] sweep.
    SIGMOID_SCALE = 10.0

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 ev_scale: float = 30.0,
                 ev_floor: float = 0.1,
                 ndcg_at: int = 2,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        BaseTrainer.__init__(self)
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            ev_scale=float(ev_scale),
            ev_floor=float(ev_floor),
            ndcg_at=int(ndcg_at),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.ranker = None
        self.regressor = None
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L).')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for date-group ranking.')

        # --- Stage 1: ranker (binary-win relevance, ndcg@K) ---
        order_tr, groups_tr = self._group_counts(dates_train)
        order_va, groups_va = self._group_counts(dates_val)
        X_tr_sorted = np.asarray(X_train)[order_tr]
        y_tr_sorted = np.asarray(y_train, dtype=np.int32)[order_tr]
        X_va_sorted = np.asarray(X_val)[order_va]
        y_va_sorted = np.asarray(y_val, dtype=np.int32)[order_va]

        if (groups_tr < 2).all():
            raise ValueError(f'{self.name}: no train date has >=2 samples — '
                             'pairwise ranking is impossible')

        p = self._params
        self.ranker = xgb.XGBRanker(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='rank:ndcg',
            eval_metric=f'ndcg@{p["ndcg_at"]}',
            ndcg_exp_gain=False,
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.ranker.fit(
            X_tr_sorted, y_tr_sorted,
            group=groups_tr,
            eval_set=[(X_va_sorted, y_va_sorted)],
            eval_group=[groups_va],
            verbose=verbose,
        )

        # --- Stage 2: EV regressor (squared error, ungrouped) ---
        # The regressor predicts per-trade pnl directly. No date grouping —
        # squared-error loss is row-independent. Different random_state breaks
        # seed-correlation with the ranker so the two sub-models contribute
        # genuine diversity at inference.
        pnl_tr = np.asarray(pnl_train, dtype=np.float32)
        pnl_va = np.asarray(pnl_val, dtype=np.float32)
        self.regressor = xgb.XGBRegressor(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='reg:squarederror',
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'] + 1,
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.regressor.fit(
            np.asarray(X_train), pnl_tr,
            eval_set=[(np.asarray(X_val), pnl_va)],
            verbose=verbose,
        )
        bi_r = getattr(self.ranker, 'best_iteration', None) or p['n_estimators']
        bi_e = getattr(self.regressor, 'best_iteration', None) or p['n_estimators']
        self._best_iteration = int(max(bi_r, bi_e))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.ranker is None or self.regressor is None:
            raise RuntimeError('Model not fit')
        ranker_raw = self.ranker.predict(X)
        ranker_prob = 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * ranker_raw))
        ev_pred = self.regressor.predict(X)
        p = self._params
        # Clip the linear part of the sigmoid to avoid float32 overflow on
        # large ev_scale × |ev_pred| products; -700 is well past the point
        # where np.exp underflows to 0 in IEEE doubles.
        z = np.clip(p['ev_scale'] * ev_pred, -700.0, 700.0)
        ev_gate_raw = 1.0 / (1.0 + np.exp(-z))
        ev_gate = p['ev_floor'] + (1.0 - p['ev_floor']) * ev_gate_raw
        return ranker_prob * ev_gate

    def predict_score(self, X) -> np.ndarray:
        """Final composite score (same as predict_proba). For diagnostics."""
        return self.predict_proba(X)

    def feature_importance(self):
        if self.ranker is None:
            return None
        # Average ranker + regressor importances after per-model normalization
        # (each XGB model normalizes differently, so re-normalize before mixing).
        fi_r = self.ranker.feature_importances_
        fi_e = self.regressor.feature_importances_ if self.regressor is not None else None
        if fi_e is None:
            return fi_r
        sr = fi_r.sum(); se = fi_e.sum()
        nr = fi_r / sr if sr > 0 else fi_r
        ne = fi_e / se if se > 0 else fi_e
        return 0.5 * (nr + ne)

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        ranker_path = os.path.join(output_dir, 'ranker.json')
        regressor_path = os.path.join(output_dir, 'regressor.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.ranker.save_model(ranker_path)
        self.regressor.save_model(regressor_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'ranker': ranker_path, 'regressor': regressor_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        ranker_path = os.path.join(output_dir, 'ranker.json')
        regressor_path = os.path.join(output_dir, 'regressor.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.ranker = xgb.XGBRanker()
        inst.ranker.load_model(ranker_path)
        inst.regressor = xgb.XGBRegressor()
        inst.regressor.load_model(regressor_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# Bagged EV-Gated Ranker — variance-reduced composite ranker × EV gate
# --------------------------------------------------------------------- #
# Motivation: ev_gated_ranker (the historical best architectural class —
# claude #159 cleared 6/7 windows at 65.5% WR) has two problems in recent
# iterations (#395-#410, all 0-2/7 windows passing):
#
#   1. Threshold sweep instability: single-trainer bagless predictions have
#      enough run-to-run variance that the per-window "best threshold" hops
#      between thr=0.0 (rank-only fallback) and thr=0.60+ (selective). At
#      thr=0.0 the gate degrades to top-K-per-day with ~62 trades/window
#      (#410 W7) and WR collapses to ~27%. At thr=0.65+ the EV gate cuts
#      most days but the few that pass are still single-tree noisy.
#   2. The 14% positive rate makes the binary-win ranker target very
#      imbalanced; one decision tree's first-split choice on a noisy feature
#      can bias the whole ensemble. A 5-bag ensemble decorrelates that
#      first-split bias by giving each bag a different bootstrap sample +
#      seed, then averaging the [0,1] composite score.
#
# Historical evidence (lost in pyc-only state, recovered from feedback DB):
#   #286: bagged_ev_gated_ranker → wp=5/7  ann=+4.8% wr=46.1% dd=9.9% trades=154
#   #284: bagged_ev_gated_ranker → wp=4/7  ann=+8.8% wr=49.7% dd=6.5% trades=145
#   #278: bagged_ev_gated_ranker → wp=4/7  ann=+5.7% wr=43.1% dd=4.9% trades=119
# vs current single ev_gated_ranker (sweep range #395-#410):
#   best #403 → wp=1/7  ann=+41.0% wr=34.5% dd=14.1% trades=225
#   median   → wp=0-1/7 wr=27-31%
#
# The bagged variant traded ~30% trade count for ~12 percentage points of WR
# and ~5 percentage points of DD reduction — exactly the v1 gate's preferred
# direction (MIN_WR=40% and MAX_DD=20% are both binding constraints; trade
# count comfortably exceeds MIN_TRADES_PER_WINDOW=20 with K=2 slot cap).
#
# Implementation:
#   - Wrap N copies of EVGatedRankerTrainer with date-block bootstrap + seed
#     diversity. Each bag fits independently on its bootstrap sample;
#     predict_proba returns the composite ranker_prob × ev_gate score in [0,1]
#     per bag.
#   - Aggregate: mean - conf_lambda * std, clipped to [0,1]. conf_lambda=0
#     reduces to plain bagging (variance reduction); conf_lambda~0.5-1.0 adds
#     consensus filtering — predictions where bags disagree get pulled below
#     the threshold sweep, abstaining from low-confidence regimes.
#   - Bootstrap unit is the **trading date**, not the row. We sample
#     ceil(bootstrap_frac × n_unique_dates) dates with replacement and pull
#     every row belonging to each sampled date into the bag's training set.
#     This (a) preserves date-group integrity so EVGatedRankerTrainer's
#     pairwise rank:ndcg still operates on intact within-date pairs (row
#     bootstrap shatters groups — a 0.85-row sample drops ~15% of every
#     date's pairs uniformly, deflating gradient signal without giving the
#     bags genuinely different views), (b) gives each bag a regime-distinct
#     view (bag A might over-sample 2024-Q1, bag B over-samples 2025-Q3), so
#     the cross-bag std (the consensus signal driving conf_lambda) actually
#     measures regime-disagreement instead of within-date sub-sampling
#     noise. Historical evidence: iter #286 used date-block bootstrap and
#     cleared 5/7 windows at 46.1% WR; the row-bootstrap reconstruction
#     (#411-#422) plateaus at 1/7 windows / 25-30% WR.
#
# Wall-time: 5 bags × 2 sub-models × 7 windows ≈ 70 XGBoost fits at ~3-4s each
# ≈ 4-5 min — well inside the 30 min iter budget.
class BaggedEVGatedRankerTrainer(BaseTrainer):
    name = 'bagged_ev_gated_ranker'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 ev_scale: float = 30.0,
                 ev_floor: float = 0.1,
                 ndcg_at: int = 2,
                 early_stopping_rounds: int = 30,
                 n_bags: int = 5,
                 conf_lambda: float = 0.5,
                 bootstrap_frac: float = 0.85,
                 random_state: int = 42):
        BaseTrainer.__init__(self)
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            ev_scale=float(ev_scale),
            ev_floor=float(ev_floor),
            ndcg_at=int(ndcg_at),
            early_stopping_rounds=early_stopping_rounds,
            n_bags=int(n_bags),
            conf_lambda=float(conf_lambda),
            bootstrap_frac=float(bootstrap_frac),
            random_state=random_state,
        )
        self.bags: list = []
        self._best_iteration = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L).')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for date-group ranking.')

        p = self._params
        X_tr_arr = np.asarray(X_train)
        y_tr_arr = np.asarray(y_train)
        pnl_tr_arr = np.asarray(pnl_train, dtype=np.float32)
        dates_tr_arr = np.asarray(dates_train)

        rs_root = np.random.RandomState(p['random_state'])

        # Date-block bootstrap setup. Pre-compute the rows belonging to each
        # unique date once so per-bag sampling is O(n_dates_sample) instead of
        # O(n_train) per bag.
        unique_dates = np.unique(dates_tr_arr)
        rows_per_date = {d: np.flatnonzero(dates_tr_arr == d)
                         for d in unique_dates}
        n_dates_sample = max(int(round(p['bootstrap_frac'] * len(unique_dates))), 5)

        self.bags = []
        best_iters = []
        for k in range(p['n_bags']):
            sample_seed = int(rs_root.randint(0, 2**31 - 1))
            date_choice = np.random.RandomState(sample_seed).choice(
                len(unique_dates), size=n_dates_sample, replace=True)
            sampled_dates = unique_dates[date_choice]
            # Concatenate all rows from sampled dates. Duplicate dates appear
            # multiple times in the bag (multinomial bootstrap weighting);
            # the inner trainer sorts by date before computing rank groups,
            # so duplicate-date rows merge into a single larger group with
            # M× the within-date pairs — equivalent to upweighting that date
            # in the ranker's pairwise loss.
            sub_idx = np.concatenate([rows_per_date[d] for d in sampled_dates])
            X_sub = X_tr_arr[sub_idx]
            y_sub = y_tr_arr[sub_idx]
            pnl_sub = pnl_tr_arr[sub_idx]
            dates_sub = dates_tr_arr[sub_idx]

            booster_seed = int(rs_root.randint(0, 2**31 - 1))
            bag = EVGatedRankerTrainer(
                max_depth=p['max_depth'],
                learning_rate=p['learning_rate'],
                n_estimators=p['n_estimators'],
                subsample=p['subsample'],
                colsample_bytree=p['colsample_bytree'],
                reg_alpha=p['reg_alpha'],
                reg_lambda=p['reg_lambda'],
                min_child_weight=p['min_child_weight'],
                gamma=p['gamma'],
                ev_scale=p['ev_scale'],
                ev_floor=p['ev_floor'],
                ndcg_at=p['ndcg_at'],
                early_stopping_rounds=p['early_stopping_rounds'],
                random_state=booster_seed,
            )
            bag.fit(X_sub, y_sub, X_val, y_val, verbose=verbose,
                    pnl_train=pnl_sub, pnl_val=pnl_val,
                    dates_train=dates_sub, dates_val=dates_val)
            self.bags.append(bag)
            bi = bag.best_iteration if bag.best_iteration is not None else p['n_estimators']
            best_iters.append(int(bi))

        self._best_iteration = int(np.mean(best_iters)) if best_iters else None
        return self

    def _bag_predictions(self, X) -> np.ndarray:
        if not self.bags:
            raise RuntimeError('Model not fit')
        # Each bag's predict_proba returns the composite ranker × ev_gate score
        # already bounded in [0, 1]. Stack to (n_bags, n_rows).
        return np.stack([bag.predict_proba(X) for bag in self.bags], axis=0)

    def predict_proba(self, X) -> np.ndarray:
        preds = self._bag_predictions(X)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0, ddof=0)
        # Confidence-penalised aggregate: when bags disagree, score is reduced.
        # Both terms are in [0,1] so without clipping the result lies in
        # [-conf_lambda, 1]; clipping to [0,1] keeps the score within the
        # gate's threshold-sweep range (SCORE_THRESHOLDS ⊂ [0,1]).
        score = mean_pred - self._params['conf_lambda'] * std_pred
        return np.clip(score, 0.0, 1.0).astype(np.float64)

    def predict_score(self, X) -> np.ndarray:
        return self.predict_proba(X)

    def feature_importance(self):
        if not self.bags:
            return None
        fis = []
        for bag in self.bags:
            fi = bag.feature_importance()
            if fi is None:
                continue
            s = float(fi.sum())
            fis.append(fi / s if s > 0 else fi)
        if not fis:
            return None
        return np.mean(np.stack(fis, axis=0), axis=0)

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        bag_dirs = []
        for k, bag in enumerate(self.bags):
            bag_dir = os.path.join(output_dir, f'bag_{k:02d}')
            bag.save(bag_dir)
            bag_dirs.append(bag_dir)
        meta_path = os.path.join(output_dir, 'metadata.json')
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'n_bags_saved': len(bag_dirs),
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'bags': bag_dirs, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        n_saved = meta.get('n_bags_saved', 0)
        inst.bags = []
        for k in range(n_saved):
            bag_dir = os.path.join(output_dir, f'bag_{k:02d}')
            bag = EVGatedRankerTrainer.load(bag_dir)
            inst.bags.append(bag)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# LightGBM LambdaRank — date-grouped listwise ranking with leaf-wise+GOSS bias
# --------------------------------------------------------------------- #
# Motivation: xgb_ranker (#80) was the most consistent claude-mode result —
# avg WR 49.5% (best of any iter to date), avg DD 10.7%, ZERO negative-ann
# windows — but only 1/7 windows cleared the 50% annualized gate. The
# regime-variance bottleneck remains: WR 49.5% × ~3-5% avg per-trade × ~10
# trades / 4mo gives 15-20% raw return, short of the 50% gate.
#
# Two structural levers stack here:
#   1. lambdarank vs rank:pairwise — LightGBM's lambdarank objective re-weights
#      each pair by its delta-NDCG contribution, putting ~10-100x more loss
#      gradient on swaps near the top of the list than swaps in the middle.
#      The gate selects exactly one symbol per date (the highest score), so
#      top-of-list ranking is *the* objective. Pairwise loss treats all swaps
#      equally and "wastes" capacity on mid-list ordering that the gate never
#      sees.
#   2. leaf-wise growth + GOSS — #48's lesson confirmed LightGBM finds signals
#      XGBoost misses (split 4: +168.6% ann at 66.7% WR). Pairing the proven
#      ranker-loss insight (#80) with the proven LGB algorithmic-diversity
#      insight (#48) targets two independent failure modes simultaneously.
#
# Implementation notes:
#   - Same _group_counts / _within_group_rank helpers as xgb_ranker (sort by
#     date, run-length-encode, integer rank-within-group as relevance).
#   - LightGBM's default label_gain is exponential and tops out at ~30; on
#     SET universe dates with up to ~70 stocks, max relevance can exceed this.
#     Solution: pass label_gain=list(range(max_rel+1)) (linear gain), which is
#     the lambdarank-equivalent of XGBoost's ndcg_exp_gain=False.
#   - SIGMOID_SCALE=10 mirrors xgb_ranker — raw lambdarank scores have similar
#     magnitude to XGBRanker's scores, both empirically O(±0.2) on this data,
#     so the same scale maps to the gate's [0.30, 0.70] threshold range.
class LightGBMRankerTrainer(BaseTrainer):
    name = 'lightgbm_ranker'

    SIGMOID_SCALE = 10.0

    def __init__(self,
                 num_leaves: int = 31,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 min_child_samples: int = 50,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 0.1,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    @staticmethod
    def _group_counts(dates):
        """Sort indices by date and return (sort_order, group_counts).
        Mirrors XGBoostRankerTrainer._group_counts — same contract."""
        d = np.asarray(dates)
        order = np.argsort(d, kind='stable')
        _, counts = np.unique(d[order], return_counts=True)
        return order, counts.astype(np.int64)

    @staticmethod
    def _within_group_rank(values, group_counts):
        """0-indexed integer rank of each value within its date group.
        Mirrors XGBoostRankerTrainer — pairwise/listwise loss only sees ordinal
        position within each group, not absolute pnl scale."""
        out = np.empty_like(values, dtype=np.int32)
        idx = 0
        for cnt in group_counts:
            chunk = values[idx:idx + cnt]
            ranks = np.argsort(np.argsort(chunk, kind='stable'), kind='stable')
            out[idx:idx + cnt] = ranks.astype(np.int32)
            idx += cnt
        return out

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import lightgbm as lgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L).')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for date-group ranking.')

        order_tr, groups_tr = self._group_counts(dates_train)
        order_va, groups_va = self._group_counts(dates_val)
        X_tr_sorted = np.asarray(X_train)[order_tr]
        pnl_tr_sorted = np.asarray(pnl_train, dtype=np.float32)[order_tr]
        X_va_sorted = np.asarray(X_val)[order_va]
        pnl_va_sorted = np.asarray(pnl_val, dtype=np.float32)[order_va]

        if (groups_tr < 2).all():
            raise ValueError(f'{self.name}: no train date has >=2 samples — '
                             'pairwise ranking is impossible')

        rel_tr = self._within_group_rank(pnl_tr_sorted, groups_tr)
        rel_va = self._within_group_rank(pnl_va_sorted, groups_va)

        # Linear label_gain replaces LightGBM's default exponential gain (which
        # caps at ~30 levels). With per-date stock counts up to ~70, integer
        # ranks can exceed the default cap; linear gain avoids the silent
        # truncation that would corrupt the lambdarank loss.
        max_rel = int(max(rel_tr.max(), rel_va.max()))
        label_gain = list(range(max_rel + 1))

        p = self._params
        self.clf = lgb.LGBMRanker(
            objective='lambdarank',
            num_leaves=p['num_leaves'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            n_estimators=p['n_estimators'],
            min_child_samples=p['min_child_samples'],
            subsample=p['subsample'],
            subsample_freq=1,
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            random_state=p['random_state'],
            n_jobs=-1,
            verbose=-1,
            label_gain=label_gain,
        )
        callbacks = [lgb.early_stopping(stopping_rounds=p['early_stopping_rounds'],
                                          verbose=verbose)]
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))

        self.clf.fit(
            X_tr_sorted, rel_tr,
            group=groups_tr,
            eval_set=[(X_va_sorted, rel_va)],
            eval_group=[groups_va],
            callbacks=callbacks,
        )
        self._best_iteration = (self.clf.best_iteration_
                                  if self.clf.best_iteration_ else self.clf.n_estimators)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        pred = self.clf.predict(X)
        return 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * pred))

    def predict_score(self, X) -> np.ndarray:
        """Raw lambdarank score (unbounded). Useful for diagnostics."""
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict(X)

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.booster_.feature_importance(importance_type='gain')

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.txt')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.booster_.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'sigmoid_scale': self.SIGMOID_SCALE,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import lightgbm as lgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.txt')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = lgb.LGBMRanker()
        inst.clf._Booster = lgb.Booster(model_file=booster_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# Stacked Ranker — validation-fold-weighted ensemble of three diverse heads
# --------------------------------------------------------------------- #
# Motivation: per-window analysis of the three best claude iterations shows
# the heads disagree by REGIME, not by quality:
#   w1 (2023-Q4): only xgb_ranker(#80) +; dual_quantile(#64) and lgb_ranker(#96) deeply -
#   w3 (2024-summer): xgb_dual_quantile(#64) +42%; xgb_ranker(#80) -1%
#   w5 (2025-Q1): xgb_ranker(#80) +26%; lgb_ranker(#96) -27%
#   w6 (2025-summer): dual_quantile(#64) +99% / lgb_ranker(#96) +71%; xgb_ranker(#80) +2%
# A regime-aware ensemble that learns per-fold weights from the held-out
# validation slice should systematically pick the right head per regime,
# converting the loop's "1/7 windows pass" plateau into a higher pass rate.
#
# Distinct from the failed avg-prob ensemble (#10):
#   - #10 averaged TWO classifiers at default HPs with EQUAL weights
#   - This stacks THREE different loss functions (rank:pairwise / lambdarank /
#     dual-quantile EV) with VALIDATION-LEARNED weights
#   - Weight objective: Spearman ρ(weighted_score, pnl_val) — directly aligned
#     with the gate's per-date highest-score selection task
#
# Sub-trainer HPs are seeded from the published best-of-loop configs:
#   xgb_ranker: #80's HPs (depth=3, lr=0.03, n_est=800, min_child_weight=20)
#   lgb_ranker: #96's HPs (num_leaves=15, depth=4, lr=0.03, n_est=1000, min_child=100)
#   dual_quantile: defaults (which are #58's best HPs + tight quantiles)
# These sub-HPs are FROZEN — train mode tunes the stacker's own knobs
# (weight grid resolution, sub-trainer subset toggles) per search_spaces.py.
#
# Output: weighted avg of sub-trainer predict_proba outputs. Each sub-trainer
# already maps its raw output through a sigmoid into [0,1], so the weighted
# avg is also in [0,1] and SCORE_THRESHOLDS [0.30..0.70] keep working.
class StackedRankerTrainer(BaseTrainer):
    name = 'stacked_ranker'

    # Sub-trainer HPs frozen at best-known configs; only the random_state is
    # exposed (so train mode can do seed-stability sweeps).
    XGB_RANKER_HP = dict(max_depth=3, learning_rate=0.03, n_estimators=800,
                         min_child_weight=20.0, gamma=0.1, subsample=0.8,
                         colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0)
    LGB_RANKER_HP = dict(num_leaves=15, max_depth=4, learning_rate=0.03,
                         n_estimators=1000, min_child_samples=100,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=0.1, reg_lambda=0.5)
    DUAL_QUANTILE_HP = dict(max_depth=6, learning_rate=0.07, n_estimators=400,
                            subsample=0.7, colsample_bytree=0.78,
                            reg_alpha=0.37, reg_lambda=1.46,
                            min_child_weight=5.0, gamma=0.01,
                            alpha_lower=0.35, alpha_upper=0.65, dd_penalty=1.0)

    def __init__(self,
                 weight_grid_step: float = 0.1,
                 use_uniform_fallback: bool = True,
                 min_concordance: float = 0.0,
                 aggregation: str = 'weighted_max',
                 random_state: int = 42,
                 early_stopping_rounds: int = 30):
        # weight_grid_step: resolution of the simplex grid search (0.1 → ~66 combos)
        # use_uniform_fallback: when val concordance for the best learned combo
        #   is below min_concordance, fall back to equal weights (safer than
        #   over-fitting to a small noisy val fold)
        # min_concordance: Spearman ρ floor for accepting learned weights
        # aggregation: 'weighted_avg' compresses scores around 0.5 when sub-models
        #   disagree (gate run #1 result: 0 trades in 4/7 splits because avg of
        #   3 sigmoids stayed under thr=0.30). 'weighted_max' takes the
        #   weight-scaled max per row — preserves spread, lets the most-confident
        #   sub-model on each row drive the prediction, but uses learned weights
        #   to *attenuate* low-quality sub-models rather than zero them out:
        #     score(row) = max_i (w_i * p_i(row))
        #   When all three w_i are similar, behaves close to plain max
        #   (high-confidence picks survive). When learned weights are skewed
        #   to one model, behaves close to that model (winner-take-all).
        self._params = dict(
            weight_grid_step=weight_grid_step,
            use_uniform_fallback=use_uniform_fallback,
            min_concordance=min_concordance,
            aggregation=aggregation,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.xgb_ranker = None
        self.lgb_ranker = None
        self.dual_quantile = None
        self.weights = None  # (w_xgb_ranker, w_lgb_ranker, w_dual_quantile)
        self._best_iteration = None
        self._val_concordance = None
        self._learned_weights = None  # weights chosen by grid search before fallback

    @staticmethod
    def _spearman(a, b):
        """Spearman ρ via Pearson on ranks. Returns 0.0 if either input is constant."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.std() == 0 or b.std() == 0:
            return 0.0
        ra = np.argsort(np.argsort(a, kind='stable'), kind='stable').astype(np.float64)
        rb = np.argsort(np.argsort(b, kind='stable'), kind='stable').astype(np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        if denom == 0:
            return 0.0
        return float((ra * rb).sum() / denom)

    def _aggregate(self, p1, p2, p3, weights):
        w1, w2, w3 = weights
        if self._params['aggregation'] == 'weighted_avg':
            return w1 * p1 + w2 * p2 + w3 * p3
        # weighted_max: row-wise maximum of weight-scaled component probabilities
        stacked = np.vstack([w1 * p1, w2 * p2, w3 * p3])
        return stacked.max(axis=0)

    def _learn_weights(self, p1, p2, p3, pnl_val):
        """Grid search on the 3-simplex for weights maximising Spearman ρ(score, pnl).

        Returns (best_weights, best_rho).
        """
        step = self._params['weight_grid_step']
        n_steps = int(round(1.0 / step))
        best_w = (1/3, 1/3, 1/3)
        best_rho = -2.0
        for i in range(n_steps + 1):
            for j in range(n_steps + 1 - i):
                k = n_steps - i - j
                w1, w2, w3 = i * step, j * step, k * step
                score = self._aggregate(p1, p2, p3, (w1, w2, w3))
                rho = self._spearman(score, pnl_val)
                if rho > best_rho:
                    best_rho = rho
                    best_w = (w1, w2, w3)
        return best_w, best_rho

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train/pnl_val (its xgb_dual_quantile component '
                'predicts EV directly).')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train/dates_val (its ranker components '
                'use date-grouped pairwise/listwise ranking loss).')

        rs = self._params['random_state']
        esr = self._params['early_stopping_rounds']

        self.xgb_ranker = XGBoostRankerTrainer(
            random_state=rs, early_stopping_rounds=esr,
            **self.XGB_RANKER_HP)
        self.xgb_ranker.fit(X_train, y_train, X_val, y_val, verbose=verbose,
                            pnl_train=pnl_train, pnl_val=pnl_val,
                            dates_train=dates_train, dates_val=dates_val)

        self.lgb_ranker = LightGBMRankerTrainer(
            random_state=rs, early_stopping_rounds=esr,
            **self.LGB_RANKER_HP)
        self.lgb_ranker.fit(X_train, y_train, X_val, y_val, verbose=verbose,
                            pnl_train=pnl_train, pnl_val=pnl_val,
                            dates_train=dates_train, dates_val=dates_val)

        self.dual_quantile = XGBoostDualQuantileTrainer(
            random_state=rs, early_stopping_rounds=esr,
            **self.DUAL_QUANTILE_HP)
        self.dual_quantile.fit(X_train, y_train, X_val, y_val, verbose=verbose,
                               pnl_train=pnl_train, pnl_val=pnl_val,
                               dates_train=dates_train, dates_val=dates_val)

        # Learn weights on the validation fold using Spearman concordance
        p_xgb = self.xgb_ranker.predict_proba(X_val)
        p_lgb = self.lgb_ranker.predict_proba(X_val)
        p_dual = self.dual_quantile.predict_proba(X_val)
        pnl_arr = np.asarray(pnl_val, dtype=np.float64)

        learned, rho = self._learn_weights(p_xgb, p_lgb, p_dual, pnl_arr)
        self._learned_weights = learned
        self._val_concordance = float(rho)

        # Fallback: when val concordance is too weak (small/noisy fold), uniform
        # weights are a safer prior than over-fitting to a single fold's quirks.
        if (self._params['use_uniform_fallback']
                and rho < self._params['min_concordance']):
            self.weights = (1/3, 1/3, 1/3)
        else:
            self.weights = learned

        # Composite best_iteration: max of components (used for logging only).
        bi_xgb = self.xgb_ranker.best_iteration or 0
        bi_lgb = self.lgb_ranker.best_iteration or 0
        bi_dual = self.dual_quantile.best_iteration or 0
        self._best_iteration = max(bi_xgb, bi_lgb, bi_dual)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.xgb_ranker is None or self.lgb_ranker is None or self.dual_quantile is None:
            raise RuntimeError('Model not fit')
        if self.weights is None:
            raise RuntimeError('Stacker weights not set — fit() must run first')
        p1 = self.xgb_ranker.predict_proba(X)
        p2 = self.lgb_ranker.predict_proba(X)
        p3 = self.dual_quantile.predict_proba(X)
        return self._aggregate(p1, p2, p3, self.weights)

    def feature_importance(self):
        if self.xgb_ranker is None:
            return None
        # Stacker's importance is a weighted blend of component importances —
        # treats sub-trainer absent importance arrays as zero contributions.
        w1, w2, w3 = self.weights
        fi_xgb = self.xgb_ranker.feature_importance()
        fi_lgb = self.lgb_ranker.feature_importance()
        fi_dual = self.dual_quantile.feature_importance()
        if fi_xgb is None or fi_lgb is None or fi_dual is None:
            return None
        # Each sub-trainer normalizes importances differently; normalize each
        # to sum=1 before weighting so the blend is in comparable units.
        def _norm(fi):
            s = fi.sum()
            return fi / s if s > 0 else fi
        return w1 * _norm(fi_xgb) + w2 * _norm(fi_lgb) + w3 * _norm(fi_dual)

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        xgb_dir = os.path.join(output_dir, 'xgb_ranker')
        lgb_dir = os.path.join(output_dir, 'lgb_ranker')
        dual_dir = os.path.join(output_dir, 'dual_quantile')
        self.xgb_ranker.save(xgb_dir)
        self.lgb_ranker.save(lgb_dir)
        self.dual_quantile.save(dual_dir)
        meta_path = os.path.join(output_dir, 'metadata.json')
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'weights': list(self.weights),
            'learned_weights': list(self._learned_weights) if self._learned_weights else None,
            'val_concordance': self._val_concordance,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {
            'xgb_ranker_dir': xgb_dir,
            'lgb_ranker_dir': lgb_dir,
            'dual_quantile_dir': dual_dir,
            'metadata': meta_path,
        }

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.xgb_ranker = XGBoostRankerTrainer.load(os.path.join(output_dir, 'xgb_ranker'))
        inst.lgb_ranker = LightGBMRankerTrainer.load(os.path.join(output_dir, 'lgb_ranker'))
        inst.dual_quantile = XGBoostDualQuantileTrainer.load(os.path.join(output_dir, 'dual_quantile'))
        inst.weights = tuple(meta.get('weights', (1/3, 1/3, 1/3)))
        lw = meta.get('learned_weights')
        inst._learned_weights = tuple(lw) if lw else None
        inst._val_concordance = meta.get('val_concordance')
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# Magnitude-Weighted XGBoost Classifier
# --------------------------------------------------------------------- #
# Motivation: every existing classifier in the registry treats every train row
# equally — a +14% target hit and a +4.5% trailing-stop exit both contribute
# the same gradient. Recent train sweeps (#422-#429) on bagged_ev_gated_ranker
# are stuck at WR ~30% because the binary loss has no way to tell the model
# which "wins" matter. The regressors (xgb_regressor, xgb_huber) implicitly
# weight by pnl² (squared loss) or |residual| (Huber), but they predict
# unbounded pnl so their score distribution sigmoid-compresses through
# SIGMOID_SCALE — leading to the threshold-sweep collapse we see in iter #428
# where only thr=0.0 has any trades and WR craters.
#
# The clean middle ground: keep the binary y∈{0,1} target so predict_proba
# stays naturally in [0,1] (no sigmoid compression), but pass per-row
# sample_weight = (|pnl| × magnitude_scale) + base_weight to xgb.fit. The
# gradient on each row is now scaled by trade impact:
#   - +14% target hit gets ~3-4× the learning gradient of a +4.5% trailing
#   - -3% stop loss gets ~1.5× the gradient of a marginal -0.5% close
#   - Marginal trades (|pnl|<1%) get base_weight only — they're noise
# So the model focuses on patterns that drive HIGH-MAGNITUDE outcomes, both
# wins and losses. This is qualitatively different from class_weight (which
# only scales the 0/1 ratio) and from regression heads (which predict pnl
# directly and lose the [0,1] calibration).
#
# Why this hasn't been tried: every prior magnitude-aware approach in the
# registry put the magnitude into the LABEL (rank: pnl-rank, regressor: pnl,
# huber: pnl with capped gradient). Putting it into the per-row sample WEIGHT
# while keeping the binary y is novel for this loop — and it's the only path
# that combines "model focuses on high-impact rows" with "predict_proba
# returns calibrated [0,1] scores that the threshold sweep can actually use."
#
# Implementation notes:
#   - sample_weight scaled to mean ≈ 1.0 per row so XGBoost's per-leaf hessian
#     normalization (min_child_weight) keeps its meaningful unit.
#   - Three knobs: magnitude_scale ∈ [5, 30] (gradient amplification on high-
#     pnl rows), base_weight ∈ [0.3, 1.5] (floor for marginal rows so the
#     model doesn't collapse onto outliers only), pos_class_weight ∈ [1.0, 6.0]
#     (existing scale_pos_weight tradition for the imbalanced ~12% pos rate).
#   - val sample_weight mirrors train so early stopping evaluates on the same
#     loss landscape the model is optimized for.
class XGBoostMagnitudeWeightedTrainer(BaseTrainer):
    name = 'xgb_magnitude_classifier'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 magnitude_scale: float = 15.0,
                 base_weight: float = 0.6,
                 pos_class_weight: float = 3.0,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            magnitude_scale=float(magnitude_scale),
            base_weight=float(base_weight),
            pos_class_weight=float(pos_class_weight),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None

    def _build_weights(self, pnl):
        """Per-row sample weight: |pnl|*scale + base. Normalised so mean=1
        across the train set (preserves XGBoost's min_child_weight unit)."""
        p = self._params
        pnl_arr = np.asarray(pnl, dtype=np.float64)
        raw = np.abs(pnl_arr) * p['magnitude_scale'] + p['base_weight']
        m = raw.mean()
        if m <= 0:
            return np.ones_like(raw, dtype=np.float32)
        return (raw / m).astype(np.float32)

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade realized P&L) '
                'for magnitude weighting.')
        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        sw_tr = self._build_weights(pnl_train)
        sw_va = self._build_weights(pnl_val)

        self.clf = xgb.XGBClassifier(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            scale_pos_weight=p['pos_class_weight'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf.fit(
            np.asarray(X_train), np.asarray(y_train),
            sample_weight=sw_tr,
            eval_set=[(np.asarray(X_val), np.asarray(y_val))],
            sample_weight_eval_set=[sw_va],
            verbose=verbose,
        )
        self._best_iteration = getattr(self.clf, 'best_iteration', None) or p['n_estimators']
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict_proba(np.asarray(X))[:, 1]

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.feature_importances_

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = xgb.XGBClassifier()
        inst.clf.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        return inst


# --------------------------------------------------------------------- #
# Strict-Win Classifier — magnitude-weighted, target = (pnl > 0)
# --------------------------------------------------------------------- #
# Motivation: the WR bottleneck across iters #438-#445 (xgb_magnitude_classifier
# train sweep) is structural, not HP-shaped. The base classifier trains on
# y_clean = (pnl > MIN_PROFIT_PCT AND max_adverse_excursion <= CLEAN_WIN_MAX_DD)
# — sequence_loader's inline override produces ~12% positive rate. But the
# gate's WR metric (scripts/return_gate.py L210: `wins = pnls > 0`) counts any
# positive-pnl trade as a win, including rocky wins capped at MIN_PROFIT_PCT/2
# (= +0.02). The base trainer therefore learns to discriminate clean wins vs
# {real losses ∪ rocky wins} — the latter two are pooled into class 0 even
# though rocky wins are class 1 at gate eval time. This blurs the loss
# gradient: a "rocky win" pattern that the gate would score as a +2% gain
# is told "you predicted wrong" during training.
#
# Fix: re-derive y from realized pnl with the gate's exact rule, y = (pnl > 0).
# This raises the positive rate to ~22% (rocky wins flip from class 0 to 1),
# halves the imbalance ratio (so pos_class_weight needs less amplification),
# and aligns the training target with the metric the gate scores. Sample
# weighting (|pnl|*scale + base_weight, normalized to mean=1) is preserved
# unchanged from the base — magnitude weighting still emphasizes extreme
# outcomes (-3% stops, +15% targets) regardless of the label boundary.
#
# Concretely the change is one line in fit(): override y_train/y_val with
# (pnl > 0).astype(int32) before delegating to the base. All other behavior
# (HP space, predict_proba, save/load) is inherited — train mode can sweep
# the same HPs against this trainer with no further plumbing.
class XGBoostStrictWinClassifierTrainer(XGBoostMagnitudeWeightedTrainer):
    name = 'xgb_strict_win'

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade '
                'realized P&L) to derive the strict-win label.')
        y_train_strict = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_val_strict = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        return super().fit(
            X_train, y_train_strict, X_val, y_val_strict,
            verbose=verbose,
            pnl_train=pnl_train, pnl_val=pnl_val,
            dates_train=dates_train, dates_val=dates_val,
        )


# --------------------------------------------------------------------- #
# Focal-Loss XGBoost Classifier — y = (pnl > 0), focal binary CE
# --------------------------------------------------------------------- #
# Motivation: every xgb_strict_win HP sweep (#454-#461, 16 iters) saturates
# at 4/7 windows. Failing windows (W2 2024-Q1, W5 2025-Q1, W7 2025Q4-2026Q1)
# all fail on WR<40% — the top-K-per-day picks are losers in regime-shift
# periods. Standard binary log-loss treats every (X, y=win) row as equally
# important; the gradient is dominated by easy positives (clear bullish
# setups in 2024-H2 training data) and the model overfits to those patterns.
# When the test window's regime differs (Q1 chop, late-2025 reversal), the
# learned discriminant fails because it never saw enough hard-case gradient.
#
# Focal loss (Lin et al. 2017, "Focal Loss for Dense Object Detection"):
#   FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
# where p_t = p if y=1 else 1-p. The (1-p_t)^gamma factor multiplicatively
# down-weights well-classified examples (p_t high) and up-weights hard ones
# (p_t low). gamma=0 reduces to weighted binary CE; gamma=2 is the canonical
# default for object-detection imbalance.
#
# Hypothesis: forcing the gradient onto regime-edge / ambiguous training rows
# (instead of bullish-period easy wins) yields a more discriminative model
# that holds up in W2/W5/W7. This is a structurally distinct knob from
# pos_class_weight (uniform scalar) and magnitude_scale (|pnl|-driven row
# weighting): focal loss is a per-row loss-shape modulation that depends on
# the model's CURRENT prediction confidence, not the row's static properties.
#
# Implementation: subclass strict-win (so y = pnl > 0 alignment with the gate
# is preserved) and override fit() to use xgb.train with a custom objective
# closure. Hessian is approximated as p*(1-p) (the logistic CE hess) — the
# exact focal hessian is unstable for p near 0/1 and standard implementations
# use this simplification. predict_proba derives sigmoid from raw margin
# since custom-objective boosters output margins, not probabilities.
def _make_focal_loss_obj(alpha: float, gamma: float):
    """Binary focal-loss custom objective for xgb.train.

    xgb.train passes the closure `(y_pred, dtrain)` where y_pred is the raw
    margin array and dtrain is a DMatrix carrying the labels. Per the analytic
    derivation:
      For y=1: g = alpha * (1-p)^gamma * (gamma * p * log(p) + p - 1)
      For y=0: g = (1-alpha) * p^gamma * (p - gamma * (1-p) * log(1-p))
    Hess uses logistic-CE approximation: h = max(p*(1-p), 1e-6).
    """
    def _obj(y_pred, dtrain):
        z = np.clip(np.asarray(y_pred, dtype=np.float64), -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
        one_minus_p = 1.0 - p_safe
        log_p = np.log(p_safe)
        log_1mp = np.log(one_minus_p)
        y = np.asarray(dtrain.get_label(), dtype=np.float64)
        pow_one_minus_p = np.power(one_minus_p, gamma)
        pow_p = np.power(p_safe, gamma)
        g_pos = alpha * pow_one_minus_p * (gamma * p_safe * log_p + p_safe - 1.0)
        g_neg = (1.0 - alpha) * pow_p * (p_safe - gamma * one_minus_p * log_1mp)
        grad = np.where(y == 1.0, g_pos, g_neg)
        hess = np.maximum(p_safe * one_minus_p, 1e-6)
        return grad.astype(np.float64), hess.astype(np.float64)
    return _obj


class XGBoostFocalLossClassifierTrainer(BaseTrainer):
    name = 'xgb_focal_loss'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 focal_alpha: float = 0.5,
                 focal_gamma: float = 2.0,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            focal_alpha=float(focal_alpha),
            focal_gamma=float(focal_gamma),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.booster = None
        self._best_iteration = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade '
                'realized P&L) to derive the strict-win label.')

        # Strict-win target: realized pnl > 0. Aligns with the gate's WR metric.
        y_tr = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_va = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        self._n_features = Xt.shape[1]
        dtrain = xgb.DMatrix(Xt, label=y_tr.astype(np.float32))
        dval = xgb.DMatrix(Xv, label=y_va.astype(np.float32))

        params = {
            'tree_method': 'hist',
            'max_depth': int(p['max_depth']),
            'learning_rate': float(p['learning_rate']),
            'subsample': float(p['subsample']),
            'colsample_bytree': float(p['colsample_bytree']),
            'reg_alpha': float(p['reg_alpha']),
            'reg_lambda': float(p['reg_lambda']),
            'min_child_weight': float(p['min_child_weight']),
            'gamma': float(p['gamma']),
            'seed': int(p['random_state']),
            'verbosity': 0,
            'eval_metric': 'logloss',
            # Disable_default_eval_metric prevents XGBoost from inferring an
            # objective-tied default (which fails for custom objectives).
            'disable_default_eval_metric': 0,
        }
        focal_obj = _make_focal_loss_obj(
            alpha=float(p['focal_alpha']),
            gamma=float(p['focal_gamma']),
        )
        self.booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(p['n_estimators']),
            evals=[(dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            obj=focal_obj,
            verbose_eval=verbose,
        )
        self._best_iteration = int(getattr(self.booster, 'best_iteration', 0))
        return self

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        if self.booster is None:
            raise RuntimeError('Model not fit')
        dmat = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        # Custom-objective boosters emit raw margins; apply sigmoid for P(y=1).
        raw = self.booster.predict(dmat, output_margin=True)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0)))

    def feature_importance(self):
        if self.booster is None or self._n_features is None:
            return None
        score = self.booster.get_score(importance_type='gain')
        out = np.zeros(self._n_features, dtype=np.float64)
        for k, v in score.items():
            if k.startswith('f'):
                out[int(k[1:])] = v
        return out

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.booster.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.booster = xgb.Booster()
        inst.booster.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        inst._n_features = meta.get('n_features')
        return inst


# --------------------------------------------------------------------- #
# Group-Balanced Focal Loss XGBoost Classifier
# --------------------------------------------------------------------- #
# Motivation: xgb_focal_loss (#470-#477, 8 train iters) saturates at 4/7
# windows. Default-HP run #477 reached ann=+1.6%, DD=11.3%, trades=187 — but
# WR=36.1% with 3 windows under the 40% floor (W2 2024-Q1, W5 2025-Q1, W7
# 2025Q4-2026Q1, all regime-shift periods). The focal-loss objective focuses
# gradient on hard-classified rows WITHIN the training distribution, but does
# not address the orthogonal failure mode: the training data is dominated by
# 2024-H2 / 2025-H2 bullish quarters, and the learned discriminant projects
# poorly onto chop / regime-shift quarters in the test windows.
#
# Hypothesis: combine focal loss (within-distribution hard-example focus) with
# per-quarter GROUP-BALANCED sample weighting (across-distribution regime
# balance). Weight each training row by w_i = (N_total / K_groups) / N_group,
# so every calendar quarter contributes equal total gradient regardless of
# how many rows fall in it. The classifier then cannot ignore Q1-style chop
# quarters just because Q3 bull periods are over-represented in the training
# panel. Focal loss still does its job inside each group; group balancing
# does its job across groups. Two independent levers stacked.
#
# This is structurally distinct from existing trainers:
#   - xgb_focal_loss: focal loss only (no regime balancing)
#   - xgb_strict_win / xgb_magnitude_classifier: weighted CE, no regime balancing
#   - bagged_*: row/date bootstrap diversity, but bags collapse back to the
#     dominant-regime distribution on average
#   - ev_gated_ranker: pairwise ranking, no per-row regime weighting
#
# Group balancing is also distinct from pos_class_weight (uniform scalar over
# the positive class) and from focal_alpha (uniform scalar over class types) —
# it is per-ROW and per-GROUP, attacking the regime-imbalance dimension that
# none of the existing knobs touches.
#
# Group definition: YYYYQq derived from the date string. With ~6-month training
# windows, this produces ~6 groups, giving the inverse-frequency weighting
# enough granularity to differentiate Q1 vs Q3 patterns without splintering
# into too-small per-group samples.
def _quarter_group(date_val) -> str:
    s = str(date_val)[:7]
    yr, mo = s.split('-')
    q = (int(mo) - 1) // 3 + 1
    return f'{yr}Q{q}'


def _group_balanced_weights(dates_arr, group_fn=_quarter_group) -> np.ndarray:
    """Inverse-frequency sample weights so each group contributes equal total
    weight. Returns float64 array same length as dates_arr; sum equals N.
    """
    groups = np.array([group_fn(d) for d in dates_arr])
    uniq, counts = np.unique(groups, return_counts=True)
    n_total = len(groups)
    n_groups = max(1, len(uniq))
    per_group = n_total / n_groups
    g_to_w = {g: per_group / max(1, c) for g, c in zip(uniq, counts)}
    weights = np.array([g_to_w[g] for g in groups], dtype=np.float64)
    return weights


class XGBoostGroupBalancedFocalLossTrainer(BaseTrainer):
    name = 'xgb_group_balanced_focal'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 focal_alpha: float = 0.5,
                 focal_gamma: float = 2.0,
                 group_balance_strength: float = 1.0,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            focal_alpha=float(focal_alpha),
            focal_gamma=float(focal_gamma),
            group_balance_strength=float(group_balance_strength),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.booster = None
        self._best_iteration = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade '
                'realized P&L) to derive the strict-win label.')
        if dates_train is None:
            raise ValueError(
                f'{self.name} requires dates_train for per-quarter group balancing.')

        # Strict-win target — same alignment with the gate's WR metric as
        # xgb_focal_loss. Group-balanced weighting is layered on top.
        y_tr = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_va = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params

        # Per-quarter inverse-frequency weights, then optionally tempered toward
        # uniform via group_balance_strength ∈ [0,1]. strength=0 → all weights
        # 1.0 (no group balancing, falls back to plain focal loss);
        # strength=1 → fully balanced inverse-frequency.
        gb_w = _group_balanced_weights(np.asarray(dates_train), _quarter_group)
        s = float(p['group_balance_strength'])
        weights_tr = (1.0 - s) * np.ones_like(gb_w) + s * gb_w
        # Renormalize so mean weight = 1.0 (preserves XGBoost's effective
        # learning-rate scale; gradient magnitudes match the un-weighted run
        # in expectation).
        weights_tr = weights_tr * (len(weights_tr) / weights_tr.sum())

        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        self._n_features = Xt.shape[1]
        dtrain = xgb.DMatrix(Xt, label=y_tr.astype(np.float32),
                             weight=weights_tr.astype(np.float32))
        dval = xgb.DMatrix(Xv, label=y_va.astype(np.float32))

        params = {
            'tree_method': 'hist',
            'max_depth': int(p['max_depth']),
            'learning_rate': float(p['learning_rate']),
            'subsample': float(p['subsample']),
            'colsample_bytree': float(p['colsample_bytree']),
            'reg_alpha': float(p['reg_alpha']),
            'reg_lambda': float(p['reg_lambda']),
            'min_child_weight': float(p['min_child_weight']),
            'gamma': float(p['gamma']),
            'seed': int(p['random_state']),
            'verbosity': 0,
            'eval_metric': 'logloss',
            'disable_default_eval_metric': 0,
        }
        focal_obj = _make_focal_loss_obj(
            alpha=float(p['focal_alpha']),
            gamma=float(p['focal_gamma']),
        )
        self.booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(p['n_estimators']),
            evals=[(dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            obj=focal_obj,
            verbose_eval=verbose,
        )
        self._best_iteration = int(getattr(self.booster, 'best_iteration', 0))
        return self

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        if self.booster is None:
            raise RuntimeError('Model not fit')
        dmat = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        raw = self.booster.predict(dmat, output_margin=True)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0)))

    def feature_importance(self):
        if self.booster is None or self._n_features is None:
            return None
        score = self.booster.get_score(importance_type='gain')
        out = np.zeros(self._n_features, dtype=np.float64)
        for k, v in score.items():
            if k.startswith('f'):
                out[int(k[1:])] = v
        return out

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.booster.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.booster = xgb.Booster()
        inst.booster.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        inst._n_features = meta.get('n_features')
        return inst


# --------------------------------------------------------------------- #
# Temporal Mixup XGBoost Classifier
# --------------------------------------------------------------------- #
# Motivation: 5 distinct trainer families saturate at 4/7 windows across the
# last ~150 iterations (xgb_focal_loss #470-#489 best 4/7; xgb_strict_win
# #446-#461 best 4/7; xgb_magnitude_classifier #430-#445 best 3/7 ann=+95%;
# xgb_group_balanced_focal #495-#538 best 4/7; bagged_ev_gated_ranker best
# 5/7 only on the pre-fix annualization bug). The failing windows cluster on
# regime-shift periods (W2 2024-Q1 chop, W5 2025-Q1 selloff, W7 2025Q4-2026Q1)
# with WR<40%. Each tried loss-function tweak (focal, group balance, magnitude
# weighting, class weighting) reweights *existing* training samples but never
# expands the model's implicit knowledge beyond the empirical distribution.
#
# Hypothesis: regime-specific overfitting is the structural ceiling. The
# learned discriminant memorises bull-quarter-specific patterns and projects
# poorly onto chop-quarter test data. Mixup augmentation (Zhang et al. 2018,
# C-Mixup 2022, and recent 2024 tabular extensions) generates synthetic
# training samples that linearly interpolate features+labels between two real
# samples, explicitly creating training points OUTSIDE the empirical
# distribution. When the partner is drawn from a temporally-distant quarter,
# the synthetic samples bridge regime boundaries — forcing the model to learn
# decision boundaries that are smooth across regimes rather than memorising
# the dominant-quarter pattern.
#
# This is structurally distinct from every prior knob:
#   - focal / strict-win / magnitude: per-row weighting on real samples only
#   - group_balanced_focal: per-quarter weighting on real samples only
#   - bagged_*: bootstrap diversity, but bags still draw from real samples
#   - SMOTE-like minority oversampling: random partners, no regime bridging
# Mixup is the only mechanism that creates training points outside the
# empirical distribution — which is what regime-shift generalisation
# fundamentally requires. Soft labels (in [0,1]) require reg:logistic
# objective rather than focal loss; this is the trade-off — focal loss
# operates on hard targets only, so the two innovations are not stackable
# in the same trainer without rederiving the focal gradient.
def _temporal_mixup_partners(base_idx: np.ndarray, dates_arr: np.ndarray,
                              min_quarters_apart: int,
                              rng: 'np.random.Generator') -> np.ndarray:
    """For each entry in base_idx, return a partner index drawn from a sample
    in a quarter at least ``min_quarters_apart`` away. Falls back to any-other
    when no distant quarter exists in the training data.
    """
    quarters = np.array([_quarter_group(d) for d in dates_arr])

    def _q_to_int(q: str) -> int:
        yr, qq = q.split('Q')
        return int(yr) * 4 + int(qq)

    q_int = np.array([_q_to_int(q) for q in quarters])
    n = len(dates_arr)

    # Per-quarter "far pool" cache so we don't recompute the mask per row.
    far_pool: dict[int, np.ndarray] = {}
    for q in np.unique(q_int):
        far_idx = np.where(np.abs(q_int - q) >= max(0, min_quarters_apart))[0]
        if len(far_idx) == 0:
            # Fallback 1: any sample from a different quarter.
            far_idx = np.where(q_int != q)[0]
        if len(far_idx) == 0:
            # Fallback 2: any sample (all in same quarter — degenerate).
            far_idx = np.arange(n)
        far_pool[int(q)] = far_idx

    partners = np.empty(len(base_idx), dtype=np.int64)
    for k, i in enumerate(base_idx):
        pool = far_pool[int(q_int[i])]
        partners[k] = pool[rng.integers(0, len(pool))]
    return partners


class XGBoostTemporalMixupTrainer(BaseTrainer):
    """XGBoost binary classifier trained on real ∪ temporally-mixed samples.

    For each real training row, generates a synthetic partner row by linearly
    combining its features and (strict-win) label with a sample drawn from a
    quarter at least ``mixup_min_quarters`` away. Trains with reg:logistic
    objective (which accepts soft labels in [0,1]) on the augmented set.
    Validation set is left untouched — early stopping evaluates on real
    held-out data only.
    """
    name = 'xgb_temporal_mixup'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 mixup_alpha: float = 0.4,
                 mixup_ratio: float = 1.0,
                 mixup_min_quarters: int = 1,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            mixup_alpha=float(mixup_alpha),
            mixup_ratio=float(mixup_ratio),
            mixup_min_quarters=int(mixup_min_quarters),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.booster = None
        self._best_iteration = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade '
                'realized P&L) to derive the strict-win label.')
        if dates_train is None:
            raise ValueError(
                f'{self.name} requires dates_train for temporal partner selection.')

        # Strict-win label aligns training target with the gate's WR metric,
        # matching xgb_focal_loss / xgb_group_balanced_focal.
        y_tr = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.float64)
        y_va = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.float64)
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        Xt = np.asarray(X_train, dtype=np.float64)
        Xv = np.asarray(X_val, dtype=np.float32)
        self._n_features = Xt.shape[1]

        rng = np.random.default_rng(int(p['random_state']))
        n = len(Xt)
        n_aug = int(n * float(p['mixup_ratio']))

        if n_aug > 0:
            base_idx = rng.integers(0, n, size=n_aug)
            partner_idx = _temporal_mixup_partners(
                base_idx, np.asarray(dates_train),
                min_quarters_apart=int(p['mixup_min_quarters']),
                rng=rng,
            )
            alpha = float(p['mixup_alpha'])
            lam_raw = rng.beta(alpha, alpha, size=n_aug)
            # Asymmetric mixup: ensure lam ≥ 0.5 so the synthetic sample stays
            # closer to its base parent than to the temporal partner. Empirically
            # this stabilises soft-label training; symmetric (lam ~ 0.5) labels
            # collapse the gradient signal on rows where parents disagree.
            lam = np.maximum(lam_raw, 1.0 - lam_raw)
            lam_col = lam[:, None]

            X_mix = lam_col * Xt[base_idx] + (1.0 - lam_col) * Xt[partner_idx]
            y_mix = lam * y_tr[base_idx] + (1.0 - lam) * y_tr[partner_idx]

            X_combined = np.vstack([Xt, X_mix]).astype(np.float32)
            y_combined = np.concatenate([y_tr, y_mix]).astype(np.float32)
        else:
            X_combined = Xt.astype(np.float32)
            y_combined = y_tr.astype(np.float32)

        dtrain = xgb.DMatrix(X_combined, label=y_combined)
        dval = xgb.DMatrix(Xv, label=y_va.astype(np.float32))

        params = {
            'tree_method': 'hist',
            # reg:logistic supports continuous labels in [0,1] — required for
            # soft mixup labels. Standard binary:logistic with hard labels would
            # round-trip through round() and lose the mixup signal.
            'objective': 'reg:logistic',
            'max_depth': int(p['max_depth']),
            'learning_rate': float(p['learning_rate']),
            'subsample': float(p['subsample']),
            'colsample_bytree': float(p['colsample_bytree']),
            'reg_alpha': float(p['reg_alpha']),
            'reg_lambda': float(p['reg_lambda']),
            'min_child_weight': float(p['min_child_weight']),
            'gamma': float(p['gamma']),
            'seed': int(p['random_state']),
            'verbosity': 0,
            'eval_metric': 'logloss',
        }

        self.booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(p['n_estimators']),
            evals=[(dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        self._best_iteration = int(getattr(self.booster, 'best_iteration', 0))
        return self

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        if self.booster is None:
            raise RuntimeError('Model not fit')
        dmat = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        # reg:logistic returns probabilities in [0,1] directly.
        return self.booster.predict(dmat)

    def feature_importance(self):
        if self.booster is None or self._n_features is None:
            return None
        score = self.booster.get_score(importance_type='gain')
        out = np.zeros(self._n_features, dtype=np.float64)
        for k, v in score.items():
            if k.startswith('f'):
                out[int(k[1:])] = v
        return out

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.booster.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.booster = xgb.Booster()
        inst.booster.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        inst._n_features = meta.get('n_features')
        return inst


# --------------------------------------------------------------------- #
# Just-Train-Twice XGBoost classifier (Liu et al. 2021)
# --------------------------------------------------------------------- #
# Motivation: every prior loss-shape / sample-weighting trainer (focal,
# strict-win, magnitude, group-balanced-focal, temporal-mixup) saturates at
# 4/7 windows. The chronic failure cluster is W3 (2024-05..2024-08) and W6
# (2025-05..2025-08) — Thai-summer regimes where WR collapses to 30-37%.
# Static reweighting schemes (per-row by |pnl|, per-quarter by inverse-
# frequency, per-row by current model confidence) all reweight the SAME
# distribution the ERM model already sees: they don't change which subset
# of the input space is "hard" — only how loudly the existing hard set
# screams.
#
# Hypothesis: the truly hard subset is empirically defined — the rows an
# ERM-trained XGBoost model gets WRONG. JTT (Liu et al. 2021, ICML, "Just
# Train Twice: Improving Group Robustness without Training Group
# Information") implements implicit Group DRO without requiring group
# labels:
#   Pass 1 (identifier): train a deliberately-weak ERM model
#   Pass 2 (final):      retrain a fresh model with samples misclassified
#                        by the identifier upweighted by `lambda_up`
# Without group labels, the misclassified-by-pass-1 set is the empirical
# worst-case subgroup. Liu et al. show this matches or beats Group DRO
# (which requires explicit group labels) on Waterbirds / CelebA / CivilComments.
#
# Structurally distinct from existing trainers:
#   - focal_loss:    per-row weighting by CURRENT prediction confidence
#                    (single-pass, weights computed inside loss closure)
#   - group_balanced: per-quarter weighting by FREQUENCY (single-pass,
#                    weights computed before any training)
#   - magnitude:     per-row weighting by |pnl| (single-pass, static)
#   - temporal_mixup: synthetic samples (data augmentation, single-pass)
#   - bagged_*:      bootstrap diversity, but bags are still ERM
#   - JTT:           per-row weighting by FIRST-PASS MISTAKES (two-pass,
#                    weights derived from a separately-trained model)
# The "weights from a learned model" mechanism is genuinely orthogonal —
# it's the only one that lets the identifier discover hard subsets the
# loss / labels themselves couldn't reveal.
#
# Why this should help W3/W6 specifically: the dominant 2024-H2 / 2025-H2
# bull-quarter rows in training are easy for the identifier; the chop /
# regime-shift rows are hard. Upweighting the latter forces the pass-2
# model to fit decision boundaries that survive in summer regimes — even
# when those rows aren't from the test windows themselves, they're closer
# in feature distribution than the bull-quarter rows are.
#
# Implementation: pure-strict-win label (matches focal_loss / group_balanced /
# temporal_mixup), binary:logistic objective (no custom obj — JTT works
# equally well with built-in CE since the DRO comes from the weights, not
# the loss shape). Pass 1 deliberately weakened by training only
# `pass1_estimators_frac` of n_estimators (per JTT paper §3 — early-stopped
# identifier, not over-trained one, otherwise mistakes ≈ 0 and DRO degrades
# to ERM). Pass 1 and pass 2 use distinct seeds so pass-2 isn't a strict
# refinement of pass-1's tree structure.
class XGBoostJTTTrainer(BaseTrainer):
    """Just Train Twice (Liu et al. 2021): two-pass training with implicit
    Group DRO via mistake-set upweighting. No group labels required.
    """
    name = 'xgb_jtt'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 lambda_up: float = 50.0,
                 pass1_estimators_frac: float = 0.5,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            lambda_up=float(lambda_up),
            pass1_estimators_frac=float(pass1_estimators_frac),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.booster = None
        self._best_iteration = None
        self._n_features = None
        self._mistake_rate = None  # diagnostic: fraction of train rows the identifier got wrong

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade '
                'realized P&L) to derive the strict-win label.')

        # Strict-win label aligns target with the gate's WR metric (matches
        # xgb_focal_loss / xgb_strict_win / xgb_temporal_mixup).
        y_tr = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_va = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        self._n_features = Xt.shape[1]

        n_total = int(p['n_estimators'])
        n_pass1 = max(50, int(n_total * float(p['pass1_estimators_frac'])))

        common = {
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'max_depth': int(p['max_depth']),
            'learning_rate': float(p['learning_rate']),
            'subsample': float(p['subsample']),
            'colsample_bytree': float(p['colsample_bytree']),
            'reg_alpha': float(p['reg_alpha']),
            'reg_lambda': float(p['reg_lambda']),
            'min_child_weight': float(p['min_child_weight']),
            'gamma': float(p['gamma']),
            'verbosity': 0,
            'eval_metric': 'logloss',
        }

        # PASS 1 — ERM identifier (weakened capacity via fewer rounds)
        dtrain_uniform = xgb.DMatrix(Xt, label=y_tr.astype(np.float32))
        dval = xgb.DMatrix(Xv, label=y_va.astype(np.float32))
        params_p1 = {**common, 'seed': int(p['random_state'])}
        identifier = xgb.train(
            params=params_p1,
            dtrain=dtrain_uniform,
            num_boost_round=n_pass1,
            evals=[(dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )

        # Identify mistakes on the training set (hard 0.5 threshold — the
        # canonical JTT criterion; using the actual model decisions, not
        # margin slack).
        pass1_proba = identifier.predict(dtrain_uniform)
        pass1_class = (pass1_proba >= 0.5).astype(np.int32)
        mistakes = (pass1_class != y_tr)
        mistake_rate = float(mistakes.mean())
        self._mistake_rate = mistake_rate

        # Build pass-2 weights: misclassified rows × lambda_up, correct rows × 1.
        # Renormalize to mean = 1 so XGBoost's effective learning rate matches
        # the un-weighted pass (gradient magnitudes preserved in expectation).
        weights = np.where(mistakes, float(p['lambda_up']), 1.0).astype(np.float64)
        weights = weights * (len(weights) / weights.sum())

        # PASS 2 — final model with mistake-upweighted DMatrix. Distinct seed
        # so pass 2 isn't a deterministic refinement of pass 1's tree structure
        # (would defeat the DRO mechanism by recovering pass 1's mistakes).
        dtrain_w = xgb.DMatrix(
            Xt,
            label=y_tr.astype(np.float32),
            weight=weights.astype(np.float32),
        )
        params_p2 = {**common, 'seed': int(p['random_state']) + 1}
        self.booster = xgb.train(
            params=params_p2,
            dtrain=dtrain_w,
            num_boost_round=n_total,
            evals=[(dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        self._best_iteration = int(getattr(self.booster, 'best_iteration', 0))
        return self

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        if self.booster is None:
            raise RuntimeError('Model not fit')
        dmat = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        # binary:logistic returns probabilities directly.
        return self.booster.predict(dmat)

    def feature_importance(self):
        if self.booster is None or self._n_features is None:
            return None
        score = self.booster.get_score(importance_type='gain')
        out = np.zeros(self._n_features, dtype=np.float64)
        for k, v in score.items():
            if k.startswith('f'):
                out[int(k[1:])] = v
        return out

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.booster.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'n_features': self._n_features,
            'mistake_rate': self._mistake_rate,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.booster = xgb.Booster()
        inst.booster.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        inst._n_features = meta.get('n_features')
        inst._mistake_rate = meta.get('mistake_rate')
        return inst


# --------------------------------------------------------------------- #
# Quarterly Group DRO XGBoost Classifier
# --------------------------------------------------------------------- #
# Motivation: across the last ~150 iterations every loss-side knob
# (focal_loss, group_balanced_focal, magnitude, strict_win, temporal_mixup,
# JTT) saturates at 3-5/7 windows. Per-window pass-rates over the last 50
# xgb_temporal_mixup train-mode iters: W1 23%, W3 40%, W4 30%, W6 10%.
# W1/W3/W4 fail with WR ~33% (low-quality picks despite enough trades);
# W6 fails on n_trades=13<20 with WR=60% (rotation-limited high-conviction
# regime). Existing reweighting axes:
#   - focal: per-row CONFIDENCE (down-weights easy)
#   - group_balanced_focal: per-quarter ROW-COUNT (inverse-frequency)
#   - magnitude: per-row |PNL| (up-weights tail trades)
#   - JTT: per-row MISTAKES from an ERM identifier (implicit groups)
# What's missing: per-quarter LOSS-based reweighting. JTT looks at row-level
# mistakes; group_balanced_focal weights by group SIZE. Quarterly-DRO weights
# rows by their group's RISK (logloss after a pass-1 identifier), so a
# quarter where many rows are individually "correct" but the model's
# probability calibration is poor still gets upweighted. This is the natural
# group-level analogue of JTT.
#
# Algorithm:
#   1. Pass 1: train ERM XGBoost on uniform-weight strict-win labels.
#   2. Compute per-quarter mean BCE loss on the training set using pass-1
#      probabilities (clipped to (eps, 1-eps) to avoid log(0)).
#   3. Reweight: w_q = ((R_q + smoothing*R_mean) / (R_mean + smoothing*R_mean))
#                       ^ dro_strength
#      Smoothing prevents one outlier-low-loss quarter from collapsing weights.
#      dro_strength controls aggressiveness: 0 = uniform (ERM), 1 = linear in
#      relative loss, 2+ = quadratic (worst quarter dominates).
#   4. Renormalize per-row weights to mean=1 (preserves XGBoost effective
#      learning-rate scale).
#   5. Pass 2: train final XGBoost with these per-row weights.
#
# Distinct from JTT: JTT identifies hard ROWS; this trainer identifies hard
# GROUPS. If a quarter's logloss is high uniformly across its rows (regime
# the model fits poorly overall), JTT would only upweight individual
# misclassified rows in that quarter; quarterly-DRO upweights ALL rows of
# that quarter, including the ones the identifier got right. The latter
# is appropriate when the failure mode is regime-level (the model's whole
# decision boundary is wrong for that quarter), not row-level (a few hard
# samples).
class XGBoostQuarterlyDROTrainer(BaseTrainer):
    """Group DRO with explicit calendar-quarter groups, weighted by per-quarter
    pass-1 logloss. Two-pass: ERM identifier → quarter-loss reweighted retrain.
    """
    name = 'xgb_quarterly_dro'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 dro_strength: float = 1.0,
                 pass1_estimators_frac: float = 0.5,
                 weight_smoothing: float = 0.1,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            dro_strength=float(dro_strength),
            pass1_estimators_frac=float(pass1_estimators_frac),
            weight_smoothing=float(weight_smoothing),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.booster = None
        self._best_iteration = None
        self._n_features = None
        self._quarter_logloss = None  # diagnostic
        self._quarter_weights = None  # diagnostic

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val (per-trade '
                'realized P&L) to derive the strict-win label.')
        if dates_train is None:
            raise ValueError(
                f'{self.name} requires dates_train for per-quarter group DRO.')

        # Strict-win label — same alignment with the gate's WR metric as
        # xgb_focal_loss / xgb_jtt / xgb_temporal_mixup.
        y_tr = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_va = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        self._n_features = Xt.shape[1]

        n_total = int(p['n_estimators'])
        n_pass1 = max(50, int(n_total * float(p['pass1_estimators_frac'])))

        common = {
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'max_depth': int(p['max_depth']),
            'learning_rate': float(p['learning_rate']),
            'subsample': float(p['subsample']),
            'colsample_bytree': float(p['colsample_bytree']),
            'reg_alpha': float(p['reg_alpha']),
            'reg_lambda': float(p['reg_lambda']),
            'min_child_weight': float(p['min_child_weight']),
            'gamma': float(p['gamma']),
            'verbosity': 0,
            'eval_metric': 'logloss',
        }

        # PASS 1 — ERM identifier
        dtrain_uniform = xgb.DMatrix(Xt, label=y_tr.astype(np.float32))
        dval = xgb.DMatrix(Xv, label=y_va.astype(np.float32))
        params_p1 = {**common, 'seed': int(p['random_state'])}
        identifier = xgb.train(
            params=params_p1,
            dtrain=dtrain_uniform,
            num_boost_round=n_pass1,
            evals=[(dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )

        # Per-row BCE loss using pass-1 probabilities
        proba_p1 = identifier.predict(dtrain_uniform)
        proba_p1 = np.clip(proba_p1, 1e-7, 1.0 - 1e-7)
        bce_per_row = -(y_tr * np.log(proba_p1)
                        + (1 - y_tr) * np.log(1.0 - proba_p1))

        # Per-quarter mean logloss
        quarters = np.array([_quarter_group(d) for d in dates_train])
        uniq_q = np.unique(quarters)
        q_logloss: dict[str, float] = {}
        for q in uniq_q:
            mask = (quarters == q)
            q_logloss[q] = float(bce_per_row[mask].mean())
        self._quarter_logloss = q_logloss

        # Smoothed relative-loss weights, then raised to dro_strength
        loss_vals = np.array(list(q_logloss.values()), dtype=np.float64)
        mean_R = float(loss_vals.mean()) if len(loss_vals) else 1.0
        smoothing = float(p['weight_smoothing']) * mean_R
        eta = float(p['dro_strength'])
        q_weights: dict[str, float] = {}
        for q, R in q_logloss.items():
            q_weights[q] = float(
                ((R + smoothing) / (mean_R + smoothing)) ** eta
            )
        self._quarter_weights = q_weights

        weights = np.array([q_weights[q] for q in quarters], dtype=np.float64)
        # Renormalize to mean=1 so XGBoost's effective learning rate matches
        # the un-weighted pass (gradient magnitudes preserved in expectation).
        if weights.sum() > 0:
            weights = weights * (len(weights) / weights.sum())

        # PASS 2 — final model with quarter-loss-upweighted DMatrix.
        # Distinct seed prevents pass 2 from deterministically recovering
        # pass 1's tree structure and undoing the DRO mechanism.
        dtrain_w = xgb.DMatrix(
            Xt,
            label=y_tr.astype(np.float32),
            weight=weights.astype(np.float32),
        )
        params_p2 = {**common, 'seed': int(p['random_state']) + 1}
        self.booster = xgb.train(
            params=params_p2,
            dtrain=dtrain_w,
            num_boost_round=n_total,
            evals=[(dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        self._best_iteration = int(getattr(self.booster, 'best_iteration', 0))
        return self

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        if self.booster is None:
            raise RuntimeError('Model not fit')
        dmat = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        return self.booster.predict(dmat)

    def feature_importance(self):
        if self.booster is None or self._n_features is None:
            return None
        score = self.booster.get_score(importance_type='gain')
        out = np.zeros(self._n_features, dtype=np.float64)
        for k, v in score.items():
            if k.startswith('f'):
                out[int(k[1:])] = v
        return out

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.booster.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'n_features': self._n_features,
            'quarter_logloss': self._quarter_logloss,
            'quarter_weights': self._quarter_weights,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.booster = xgb.Booster()
        inst.booster.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        inst._n_features = meta.get('n_features')
        inst._quarter_logloss = meta.get('quarter_logloss')
        inst._quarter_weights = meta.get('quarter_weights')
        return inst


# --------------------------------------------------------------------- #
# Top-K Classifier — y = (trade was top-K of its date by pnl AND pnl > 0)
# --------------------------------------------------------------------- #
# Motivation: the loss-engineering family (focal, group_focal, JTT, DRO,
# magnitude, strict_win, temporal_mixup) all saturate at 3-4/7 windows. They
# REWEIGHT example losses but keep a STATIC binary target — "was this trade
# profitable?". The gate, however, doesn't ask that: it asks "is this trade in
# the top-K-per-date by score, with score above threshold?" — a per-day
# competitive selection rule. The model is being optimized for a global
# criterion (any win) and selected by a relative criterion (best-of-day),
# producing high-WR-at-high-threshold but flat-WR-at-low-threshold (the
# gate's MIN_TRADES=20 floor forces the latter, which is where W3-W5 fail).
#
# Per-day TopK label: y=1 iff the trade is in the top-K (default K=2) of its
# entry date by realized pnl AND pnl > 0. K=2 mirrors MAX_OPEN_POSITIONS in
# return_gate.simulate_window — the gate physically takes top-2 per date.
# Positives become ~2-3% of train rows (vs 22% for strict-win), but the model
# now sees the EXACT discrimination rule the gate scores: "what makes a stock
# the best of the day?". On bear days where no candidate has pnl > 0, no
# positives are emitted — the model implicitly learns to abstain when even
# the day's best is a loser, reducing the W3-W5 false-positive band where
# 0.4-0.5-scored trades currently destroy WR.
#
# Distinct from xgb_win_ranker (NDCG@2): the ranker fits a pairwise/listwise
# loss whose gradient depends on within-date pred ordering — it can rank
# trades correctly even if all candidates are losers. This trainer's BCE
# gradient depends only on the binary positive/negative split — it can
# converge on "no positives today" patterns where the ranker still emits
# ordered scores. Distinct from xgb_strict_win: that trains on (pnl > 0)
# globally; ~22% pos_rate, no per-date relativization. The ranker and the
# strict-win classifier collectively over-call the regime-edge windows;
# this trainer's tighter label is hypothesised to under-call and abstain.
class XGBoostTopKClassifierTrainer(BaseTrainer):
    """Per-date relative top-K classifier with pos_weighted XGBoost CE.

    The training label is reconstructed inside fit() from pnl_train + dates_train:
    rows in the top-K of their entry date by pnl (with pnl > 0) get y=1, else 0.
    """
    name = 'xgb_topk_classifier'

    def __init__(self,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 top_k: int = 2,
                 pos_class_weight: float = 20.0,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            top_k=int(top_k),
            pos_class_weight=float(pos_class_weight),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.clf = None
        self._best_iteration = None
        self._n_features = None
        self._train_pos_rate = None  # diagnostic

    @staticmethod
    def _topk_labels(pnl, dates, k: int) -> np.ndarray:
        """Per-date top-K positive label: y=1 iff row is in top-K by pnl AND pnl > 0.

        Vectorised path: rank within each date by pnl (descending). Rows whose
        rank <= k AND pnl > 0 are positive. Argsort-based per-date rank avoids
        a Python-level groupby loop.
        """
        pnl = np.asarray(pnl, dtype=np.float64)
        dates = np.asarray(dates)
        n = len(pnl)
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        # Sort by (date asc, pnl desc) so consecutive equal-date rows are
        # ranked highest-pnl-first. Then rank-within-date is just a counter
        # that resets on date change.
        order = np.lexsort((-pnl, dates))
        sorted_dates = dates[order]
        # Rank within date: position since last date change.
        # New-date marker: 1 at index 0 and where date != prev date.
        new_date = np.empty(n, dtype=bool)
        new_date[0] = True
        new_date[1:] = sorted_dates[1:] != sorted_dates[:-1]
        # group_id increments by 1 at every new_date; group_start[i] = index
        # of the first row of group_id[i].
        group_id = np.cumsum(new_date) - 1
        group_start = np.zeros(group_id[-1] + 1, dtype=np.int64)
        group_start[group_id[new_date]] = np.where(new_date)[0]
        rank_in_date = np.arange(n) - group_start[group_id]
        # Apply mask: rank < k AND pnl > 0
        sorted_pnl = pnl[order]
        is_topk = (rank_in_date < k) & (sorted_pnl > 0.0)
        # Map back to original row order.
        y_sorted = is_topk.astype(np.int32)
        y = np.empty(n, dtype=np.int32)
        y[order] = y_sorted
        return y

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val to derive the '
                'top-K-per-date label.')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for per-date '
                'top-K label construction.')

        p = self._params
        k = int(p['top_k'])
        y_tr = self._topk_labels(pnl_train, dates_train, k)
        y_va = self._topk_labels(pnl_val, dates_val, k)

        # Defensive: degenerate splits where every date has no positive can
        # still happen on weird windows. Fall back to strict-win if so.
        if y_tr.sum() < 5 or y_va.sum() < 2:
            y_tr = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
            y_va = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        self._train_pos_rate = float(y_tr.mean())
        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        self._n_features = Xt.shape[1]

        self.clf = xgb.XGBClassifier(
            n_estimators=int(p['n_estimators']),
            max_depth=int(p['max_depth']),
            learning_rate=float(p['learning_rate']),
            subsample=float(p['subsample']),
            colsample_bytree=float(p['colsample_bytree']),
            scale_pos_weight=float(p['pos_class_weight']),
            reg_alpha=float(p['reg_alpha']),
            reg_lambda=float(p['reg_lambda']),
            min_child_weight=float(p['min_child_weight']),
            gamma=float(p['gamma']),
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=int(p['early_stopping_rounds']),
            random_state=int(p['random_state']),
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf.fit(
            Xt, y_tr,
            eval_set=[(Xv, y_va)],
            verbose=verbose,
        )
        self._best_iteration = getattr(self.clf, 'best_iteration', None) or int(p['n_estimators'])
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict_proba(np.asarray(X))[:, 1]

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.feature_importances_

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.clf.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'n_features': self._n_features,
            'train_pos_rate': self._train_pos_rate,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.clf = xgb.XGBClassifier()
        inst.clf.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        inst._n_features = meta.get('n_features')
        inst._train_pos_rate = meta.get('train_pos_rate')
        return inst


# --------------------------------------------------------------------- #
# Adversarial-validation reweighting — feature-distribution-aware shift fix.
#
# Every prior loss-engineering trainer (focal, JTT, group_balanced_focal,
# quarterly_dro, mixup, topk_classifier, strict_win, magnitude) saturates at
# 3-4/7 windows across iters #430-#586. They differ in HOW they reweight rows
# (per-row mistakes, per-quarter losses, per-quarter counts, per-row magnitude,
# synthetic interpolation), but they all share a structural blind spot: NONE
# of them know which TRAIN rows resemble the upcoming TEST distribution. They
# only know about the labels and losses on the train side. recency_huber (#207,
# 1/7) tried to fix this with a raw exponential decay over date position — too
# crude, since 2024-Q1 features may look like 2023-Q1 features regardless of
# the calendar.
#
# Adversarial validation (Kaggle-canonical for time-series competitions) fits
# this exact gap. A small inner classifier C predicts P(row is from the LATE
# fraction of train | features), giving every train row a continuous
# "test-likeness" score derived from its FEATURES, not its date. Re-train the
# main classifier on (pnl > 0) with sample_weight = C(x)^adv_alpha so the
# gradient concentrates on rows whose feature distribution most resembles
# the upcoming test window — the exact axis recency-decay couldn't reach.
#
# Distinct from every existing trainer:
#   - quarterly_dro: per-quarter LOSSES (label-conditioned). AVR: per-row
#     feature-distribution similarity to test (label-agnostic upstream).
#   - JTT: per-row pass-1 errors. AVR: per-row pass-1 distribution similarity.
#   - temporal_mixup: synthesizes new rows linearly between regimes. AVR:
#     reweights real rows by feature-space similarity.
#   - recency_huber: raw exp(decay * t) over date position only. AVR: features
#     drive the weight, so a 2023-Q4-dated row whose features happen to match
#     the upcoming Q1-shifted distribution gets a HIGH weight, which raw
#     date-decay can't express.
#   - focal_loss / strict_win / topk_classifier: change the LABEL or the LOSS
#     SHAPE on a fixed train pool. AVR: changes which TRAIN ROWS the gradient
#     attends to, with the same binary (pnl > 0) target.
#
# Defaults chosen to be conservative:
#   - adv_test_frac=0.25: last 25% of train rows by date are pseudo-test for
#     the inner classifier. Roughly matches the inner-train/val split that
#     evaluate_window already uses (0.80 cutoff = 20% val) — so the AVR
#     classifier learns "what does the val distribution look like" which is
#     the closest proxy to the actual test distribution available at fit time.
#   - adv_alpha=1.0: P(test-like) raised to power 1 (no sharpening or softening).
#     alpha=0 reduces to ERM; alpha>1 sharpens toward most-test-like rows;
#     alpha<1 softens. Sweep range covers all three regimes.
#   - weight_clip=10.0: caps any single row's weight at 10× the mean to prevent
#     a handful of extreme test-like rows from dominating gradient (a known
#     failure mode of importance-weighting under sparse domain overlap).
#   - adv_n_estimators=100, adv_max_depth=4: small inner classifier — must NOT
#     overfit pseudo-labels (else weights collapse to {0, 1} and AVR degenerates
#     into hard truncation of train data).
class XGBoostAdversarialValidationTrainer(BaseTrainer):
    """Two-stage XGBoost: (1) inner classifier predicts test-likeness from
    features; (2) main binary classifier on (pnl > 0) with sample_weight
    derived from inner classifier's per-row probabilities.
    """
    name = 'xgb_adv_val'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 adv_alpha: float = 1.0,
                 adv_test_frac: float = 0.25,
                 adv_n_estimators: int = 100,
                 adv_max_depth: int = 4,
                 weight_clip: float = 10.0,
                 weight_floor: float = 0.05,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            adv_alpha=float(adv_alpha),
            adv_test_frac=float(adv_test_frac),
            adv_n_estimators=int(adv_n_estimators),
            adv_max_depth=int(adv_max_depth),
            weight_clip=float(weight_clip),
            weight_floor=float(weight_floor),
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.booster = None
        self._best_iteration = None
        self._n_features = None
        self._adv_auc = None         # diagnostic: how separable late vs early
        self._weight_stats = None    # diagnostic

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val to derive the '
                'strict-win label.')
        if dates_train is None:
            raise ValueError(
                f'{self.name} requires dates_train to define the pseudo-test '
                'fraction for adversarial validation.')

        # Strict-win label (matches xgb_jtt / xgb_quarterly_dro / xgb_focal_loss)
        y_tr = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_va = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        self._n_features = Xt.shape[1]

        # ---- STAGE 1: adversarial classifier ---------------------------------
        # Pseudo-test = last `adv_test_frac` fraction of UNIQUE train dates.
        # Per-row label = 1 iff row's date is in the late tail. This gives the
        # inner classifier a date-derived but feature-driven decision boundary.
        dates_arr = np.asarray(dates_train)
        unique_dates = np.sort(np.unique(dates_arr))
        cutoff_idx = int((1.0 - float(p['adv_test_frac'])) * len(unique_dates))
        cutoff_idx = max(1, min(cutoff_idx, len(unique_dates) - 1))
        late_cutoff = unique_dates[cutoff_idx]
        adv_y = (dates_arr >= late_cutoff).astype(np.float32)

        # Guard: if the cutoff produces a single class, fall back to uniform
        # weights (degenerate window — too few unique dates for AVR to bite).
        if len(set(adv_y.tolist())) < 2:
            weights = np.ones(len(Xt), dtype=np.float64)
            self._adv_auc = None
        else:
            adv_dtrain = xgb.DMatrix(Xt, label=adv_y)
            adv_params = {
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'max_depth': int(p['adv_max_depth']),
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbosity': 0,
                'eval_metric': 'auc',
                'seed': int(p['random_state']) + 7,
            }
            adv_booster = xgb.train(
                params=adv_params,
                dtrain=adv_dtrain,
                num_boost_round=int(p['adv_n_estimators']),
                verbose_eval=verbose,
            )
            p_test_like = adv_booster.predict(adv_dtrain)
            p_test_like = np.clip(p_test_like, 1e-6, 1.0 - 1e-6)

            # Adversarial AUC for diagnostics: > 0.7 = clear distribution shift,
            # ~ 0.5 = no shift detected (AVR collapses to ERM-ish weights).
            try:
                from sklearn.metrics import roc_auc_score
                self._adv_auc = float(roc_auc_score(adv_y, p_test_like))
            except Exception:
                self._adv_auc = None

            # Per-row weight = (P(test-like))^alpha, with floor and clip.
            alpha = float(p['adv_alpha'])
            w = np.power(p_test_like, alpha)
            w = np.maximum(w, float(p['weight_floor']))
            # Renormalize to mean=1 BEFORE clip so the clip threshold is
            # interpretable in mean-multiples.
            w = w * (len(w) / w.sum())
            w = np.minimum(w, float(p['weight_clip']))
            # Renormalize again post-clip.
            if w.sum() > 0:
                weights = w * (len(w) / w.sum())
            else:
                weights = np.ones(len(Xt), dtype=np.float64)

        self._weight_stats = {
            'min': float(weights.min()),
            'max': float(weights.max()),
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'frac_above_1.0': float((weights > 1.0).mean()),
            'late_frac': float(adv_y.mean()),
        }

        # ---- STAGE 2: main classifier with AVR weights -----------------------
        common = {
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'max_depth': int(p['max_depth']),
            'learning_rate': float(p['learning_rate']),
            'subsample': float(p['subsample']),
            'colsample_bytree': float(p['colsample_bytree']),
            'reg_alpha': float(p['reg_alpha']),
            'reg_lambda': float(p['reg_lambda']),
            'min_child_weight': float(p['min_child_weight']),
            'gamma': float(p['gamma']),
            'verbosity': 0,
            'eval_metric': 'logloss',
            'seed': int(p['random_state']),
        }
        dtrain = xgb.DMatrix(
            Xt,
            label=y_tr.astype(np.float32),
            weight=weights.astype(np.float32),
        )
        dval = xgb.DMatrix(Xv, label=y_va.astype(np.float32))
        self.booster = xgb.train(
            params=common,
            dtrain=dtrain,
            num_boost_round=int(p['n_estimators']),
            evals=[(dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        self._best_iteration = int(getattr(self.booster, 'best_iteration', 0))
        return self

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        if self.booster is None:
            raise RuntimeError('Model not fit')
        dmat = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        return self.booster.predict(dmat)

    def feature_importance(self):
        if self.booster is None or self._n_features is None:
            return None
        score = self.booster.get_score(importance_type='gain')
        out = np.zeros(self._n_features, dtype=np.float64)
        for k, v in score.items():
            if k.startswith('f'):
                out[int(k[1:])] = v
        return out

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        booster_path = os.path.join(output_dir, 'booster.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.booster.save_model(booster_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'n_features': self._n_features,
            'adv_auc': self._adv_auc,
            'weight_stats': self._weight_stats,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': booster_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        booster_path = os.path.join(output_dir, 'booster.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.booster = xgb.Booster()
        inst.booster.load_model(booster_path)
        inst._best_iteration = meta.get('best_iteration')
        inst._n_features = meta.get('n_features')
        inst._adv_auc = meta.get('adv_auc')
        inst._weight_stats = meta.get('weight_stats')
        return inst


# --------------------------------------------------------------------- #
# Meta-labeling — López de Prado, Advances in Financial ML, Ch. 3.
#
# Every prior loss-engineering / reweighting trainer (#430-#617) is a SINGLE-
# stage classifier predicting P(pnl > 0). HP sweeps of xgb_adv_val (#610-617)
# top out at 5/7 windows, avg ann +5%, with WR stuck at 36-42%. Threshold-sweep
# diagnostics show the model has reasonable RANK ordering (high thresholds
# produce small, high-WR baskets) but POOR PRECISION at the n_trades >= 20
# operating point — exactly the failure pattern meta-labeling is designed for.
#
# Meta-labeling separates SIDE from SIZE:
#   - Stage 1 (side):  P(pnl > 0 | features)        — same target as today
#   - Stage 2 (size):  P(pnl > 0 | features, stage1_pred)  — confidence filter
# Stage 2 takes stage 1's output as an input feature and re-learns "given the
# side model says positive, is it actually a winner?" This concentrates trades
# on the high-confidence subset and lifts WR at the cost of trade count — the
# exact direction we need (W4-W7 cluster at 36-39% WR, just below the 40%
# gate, and even default thresholds yield 24-35 trades on those windows).
#
# Distinct from every existing trainer:
#   - xgb_adv_val: inner classifier predicts P(test-like), outputs WEIGHTS for
#     stage 2. Meta-labeling: inner classifier predicts P(pnl>0), outputs a
#     FEATURE for stage 2. Different epistemic axis (distribution vs confidence).
#   - xgb_jtt: stage 2 reweights pass-1 errors but predicts the same label
#     with the same input features. Meta-labeling: stage 2 has stage 1's
#     prediction as a NEW input feature, enabling it to learn calibration.
#   - ev_gated_ranker: regressor → ranker pipeline; both predict the same
#     pnl-related quantity. Meta-labeling: stage 2 explicitly targets
#     "filter the side model" not "rank within side-model picks".
#   - stacked_ranker: averages predictions across base learners. Meta-labeling
#     is hierarchical (one model's output feeds the next), not parallel.
#
# Implementation:
#   1. Time-split train into early/late halves by unique date.
#   2. Train stage_1_first on EARLY half; predict on LATE half + X_val to get
#      OOF stage-1 scores (no leakage — late-half rows never touched stage 1).
#   3. Train stage_2 on LATE half with [X | stage1_oof_pred] as input,
#      label = (pnl > 0). Early-stop on X_val with stage1's OOF preds on X_val.
#   4. Refit stage_1_final on ALL of train (uses full data for inference).
#   5. predict_proba(X_test): stage_1_final.predict(X_test) → stack with X_test
#      → stage_2.predict — final score is stage 2's probability.
#
# Defaults chosen conservatively:
#   - stage1_train_frac=0.5: half-and-half. Smaller fractions give stage 1 less
#     to learn from; larger fractions starve stage 2's training set.
#   - stage 2 deeper trees and lower min_child_weight than typical for stage 1
#     would cause overfit on the smaller late-half train pool. Defaults keep
#     stage 2 SHALLOWER (max_depth 3 vs 4) and HIGHER min_child_weight (10 vs 5).
#   - Strict-win label (pnl > 0), matching xgb_strict_win / xgb_jtt /
#     xgb_quarterly_dro / xgb_focal_loss / xgb_adv_val.
class XGBoostMetaLabelingTrainer(BaseTrainer):
    """Two-stage XGBoost meta-labeling: side (stage 1) + size (stage 2)."""
    name = 'xgb_meta_label'

    def __init__(self,
                 # Stage 1 (side classifier) hyperparameters
                 stage1_max_depth: int = 4,
                 stage1_learning_rate: float = 0.05,
                 stage1_n_estimators: int = 400,
                 stage1_min_child_weight: float = 5.0,
                 stage1_subsample: float = 0.8,
                 stage1_colsample_bytree: float = 0.7,
                 stage1_reg_alpha: float = 0.1,
                 stage1_reg_lambda: float = 1.0,
                 stage1_gamma: float = 0.1,
                 # Stage 2 (size / meta classifier) hyperparameters — by
                 # default tighter regularization than stage 1.
                 stage2_max_depth: int = 3,
                 stage2_learning_rate: float = 0.05,
                 stage2_n_estimators: int = 400,
                 stage2_min_child_weight: float = 10.0,
                 stage2_subsample: float = 0.8,
                 stage2_colsample_bytree: float = 0.7,
                 stage2_reg_alpha: float = 0.1,
                 stage2_reg_lambda: float = 1.0,
                 stage2_gamma: float = 0.1,
                 # Time-based train split for stage-1 OOF.
                 stage1_train_frac: float = 0.5,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            stage1_max_depth=int(stage1_max_depth),
            stage1_learning_rate=float(stage1_learning_rate),
            stage1_n_estimators=int(stage1_n_estimators),
            stage1_min_child_weight=float(stage1_min_child_weight),
            stage1_subsample=float(stage1_subsample),
            stage1_colsample_bytree=float(stage1_colsample_bytree),
            stage1_reg_alpha=float(stage1_reg_alpha),
            stage1_reg_lambda=float(stage1_reg_lambda),
            stage1_gamma=float(stage1_gamma),
            stage2_max_depth=int(stage2_max_depth),
            stage2_learning_rate=float(stage2_learning_rate),
            stage2_n_estimators=int(stage2_n_estimators),
            stage2_min_child_weight=float(stage2_min_child_weight),
            stage2_subsample=float(stage2_subsample),
            stage2_colsample_bytree=float(stage2_colsample_bytree),
            stage2_reg_alpha=float(stage2_reg_alpha),
            stage2_reg_lambda=float(stage2_reg_lambda),
            stage2_gamma=float(stage2_gamma),
            stage1_train_frac=float(stage1_train_frac),
            early_stopping_rounds=int(early_stopping_rounds),
            random_state=int(random_state),
        )
        self.stage1_final = None
        self.stage2 = None
        self._best_iteration = None
        self._n_features = None
        self._stage1_oof_auc = None
        self._stage2_val_auc = None
        self._stage1_first_iter = None

    def _xgb_params(self, prefix: str) -> dict:
        p = self._params
        return {
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'max_depth': int(p[f'{prefix}_max_depth']),
            'learning_rate': float(p[f'{prefix}_learning_rate']),
            'subsample': float(p[f'{prefix}_subsample']),
            'colsample_bytree': float(p[f'{prefix}_colsample_bytree']),
            'min_child_weight': float(p[f'{prefix}_min_child_weight']),
            'gamma': float(p[f'{prefix}_gamma']),
            'reg_alpha': float(p[f'{prefix}_reg_alpha']),
            'reg_lambda': float(p[f'{prefix}_reg_lambda']),
            'verbosity': 0,
            'eval_metric': 'logloss',
            'seed': int(p['random_state']) + (0 if prefix == 'stage1' else 11),
        }

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val to derive the '
                'strict-win label.')
        if dates_train is None:
            raise ValueError(
                f'{self.name} requires dates_train for the time-based '
                'OOF split.')

        # Strict-win label (matches xgb_jtt / xgb_quarterly_dro / xgb_adv_val)
        y_tr = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_va = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        self._n_features = Xt.shape[1]
        p = self._params

        # ---- Time split of train into early / late halves --------------------
        dates_arr = np.asarray(dates_train)
        unique_dates = np.sort(np.unique(dates_arr))
        cutoff_idx = int(float(p['stage1_train_frac']) * len(unique_dates))
        cutoff_idx = max(1, min(cutoff_idx, len(unique_dates) - 1))
        late_cutoff = unique_dates[cutoff_idx]
        early_mask = dates_arr < late_cutoff
        late_mask = ~early_mask

        # Guard: degenerate split (rare, e.g. < 4 unique dates). Fall back to
        # random 50/50 to keep the trainer evaluable rather than aborting.
        if early_mask.sum() < 50 or late_mask.sum() < 50:
            rng = np.random.RandomState(int(p['random_state']))
            shuffle = rng.permutation(len(Xt))
            half = len(Xt) // 2
            early_idx = shuffle[:half]
            late_idx = shuffle[half:]
            early_mask = np.zeros(len(Xt), dtype=bool)
            late_mask = np.zeros(len(Xt), dtype=bool)
            early_mask[early_idx] = True
            late_mask[late_idx] = True

        # Need both classes present in the early half for stage 1 to fit
        if len(set(y_tr[early_mask].tolist())) < 2 or \
           len(set(y_tr[late_mask].tolist())) < 2:
            raise ValueError(
                f'{self.name}: degenerate class distribution after time-split — '
                'early or late half is single-class.')

        # ---- STAGE 1 (first pass): train on early half -----------------------
        stage1_first_dtrain = xgb.DMatrix(
            Xt[early_mask],
            label=y_tr[early_mask].astype(np.float32),
        )
        # Use the late half as stage-1's val set for early stopping. This keeps
        # stage-1's OOF predictions on the late half generated by a model
        # tuned to that distribution boundary (best-iteration on the boundary
        # we'll predict on next).
        stage1_first_dval = xgb.DMatrix(
            Xt[late_mask],
            label=y_tr[late_mask].astype(np.float32),
        )
        stage1_first_booster = xgb.train(
            params=self._xgb_params('stage1'),
            dtrain=stage1_first_dtrain,
            num_boost_round=int(p['stage1_n_estimators']),
            evals=[(stage1_first_dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        self._stage1_first_iter = int(
            getattr(stage1_first_booster, 'best_iteration', 0))

        # OOF stage-1 predictions on the late half (these rows were never seen
        # during stage 1 first-pass training).
        stage1_oof_late = stage1_first_booster.predict(stage1_first_dval)
        # Stage-1 prediction on val (also OOF — val rows aren't in the early
        # half by construction; train_mask precedes val_mask in the gate).
        stage1_pred_val = stage1_first_booster.predict(
            xgb.DMatrix(Xv))

        try:
            from sklearn.metrics import roc_auc_score
            self._stage1_oof_auc = float(
                roc_auc_score(y_tr[late_mask], stage1_oof_late))
        except Exception:
            self._stage1_oof_auc = None

        # ---- STAGE 2: meta-classifier over [features | stage1_oof_pred] ------
        Xt_late = Xt[late_mask]
        # Stack stage-1 prediction as an additional column.
        X_stage2_tr = np.column_stack([
            Xt_late,
            stage1_oof_late.astype(np.float32),
        ]).astype(np.float32)
        X_stage2_val = np.column_stack([
            Xv,
            stage1_pred_val.astype(np.float32),
        ]).astype(np.float32)

        stage2_dtrain = xgb.DMatrix(
            X_stage2_tr,
            label=y_tr[late_mask].astype(np.float32),
        )
        stage2_dval = xgb.DMatrix(
            X_stage2_val,
            label=y_va.astype(np.float32),
        )
        self.stage2 = xgb.train(
            params=self._xgb_params('stage2'),
            dtrain=stage2_dtrain,
            num_boost_round=int(p['stage2_n_estimators']),
            evals=[(stage2_dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        self._best_iteration = int(getattr(self.stage2, 'best_iteration', 0))
        try:
            from sklearn.metrics import roc_auc_score
            stage2_val_pred = self.stage2.predict(stage2_dval)
            self._stage2_val_auc = float(roc_auc_score(y_va, stage2_val_pred))
        except Exception:
            self._stage2_val_auc = None

        # ---- STAGE 1 (final): refit on full train ----------------------------
        # At inference, the stage-1 input to stage 2 should come from a model
        # trained on as much data as possible. The early-half model was used
        # only to manufacture OOF predictions during training. There is a
        # mild distribution shift between (early-only-trained) stage-1 preds
        # used at stage-2 train time and (full-train) stage-1 preds at
        # inference — empirically this is small relative to the gain from
        # using all of train at inference, and matches the canonical LdP
        # implementation.
        stage1_final_dtrain = xgb.DMatrix(
            Xt,
            label=y_tr.astype(np.float32),
        )
        stage1_final_dval = xgb.DMatrix(
            Xv,
            label=y_va.astype(np.float32),
        )
        self.stage1_final = xgb.train(
            params=self._xgb_params('stage1'),
            dtrain=stage1_final_dtrain,
            num_boost_round=int(p['stage1_n_estimators']),
            evals=[(stage1_final_dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        if self.stage1_final is None or self.stage2 is None:
            raise RuntimeError('Model not fit')
        X_arr = np.asarray(X, dtype=np.float32)
        stage1_pred = self.stage1_final.predict(xgb.DMatrix(X_arr))
        X_meta = np.column_stack([
            X_arr,
            stage1_pred.astype(np.float32),
        ]).astype(np.float32)
        return self.stage2.predict(xgb.DMatrix(X_meta))

    def feature_importance(self):
        # Return importance over the ORIGINAL feature set (drop the stage-1
        # meta column, which is at index n_features in stage 2's input).
        if self.stage2 is None or self._n_features is None:
            return None
        score = self.stage2.get_score(importance_type='gain')
        out = np.zeros(self._n_features, dtype=np.float64)
        for k, v in score.items():
            if k.startswith('f'):
                idx = int(k[1:])
                if idx < self._n_features:
                    out[idx] = v
        return out

    @property
    def best_iteration(self):
        return self._best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        stage1_path = os.path.join(output_dir, 'stage1.json')
        stage2_path = os.path.join(output_dir, 'stage2.json')
        meta_path = os.path.join(output_dir, 'metadata.json')
        self.stage1_final.save_model(stage1_path)
        self.stage2.save_model(stage2_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_iteration': self._best_iteration,
            'n_features': self._n_features,
            'stage1_oof_auc': self._stage1_oof_auc,
            'stage2_val_auc': self._stage2_val_auc,
            'stage1_first_iter': self._stage1_first_iter,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'stage1': stage1_path, 'stage2': stage2_path,
                'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import xgboost as xgb
        meta_path = os.path.join(output_dir, 'metadata.json')
        stage1_path = os.path.join(output_dir, 'stage1.json')
        stage2_path = os.path.join(output_dir, 'stage2.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.stage1_final = xgb.Booster()
        inst.stage1_final.load_model(stage1_path)
        inst.stage2 = xgb.Booster()
        inst.stage2.load_model(stage2_path)
        inst._best_iteration = meta.get('best_iteration')
        inst._n_features = meta.get('n_features')
        inst._stage1_oof_auc = meta.get('stage1_oof_auc')
        inst._stage2_val_auc = meta.get('stage2_val_auc')
        inst._stage1_first_iter = meta.get('stage1_first_iter')
        return inst


# --------------------------------------------------------------------- #
# Regime-Aware Meta-Labeling (xgb_meta_label_regime) — extends López de
# Prado meta-labeling by feeding stage 2 three per-date aggregates of
# stage 1's predictions (mean, std, within-date rank-pct). Stage 2 thus
# sees model-implied regime context that no per-row feature directly
# encodes: "is the whole cohort bullish today?" and "where does this row
# rank within today's cohort?". The hypothesis is that the W2-W4 false-
# positive floods in transformer trainers (torch_patchtst #1298-#1299:
# WR 14-19% on bull-train→bear-test) come from over-confident per-row
# scoring with no cross-sectional confidence prior; stage 2 with regime
# features can learn to demote bullish stage-1 calls when stage 1 is
# uniformly bullish on what looks like a low-breadth day, mirroring the
# day_gate_monotonic #1031 lesson ("need hard p_day<0.3 suppression").
# --------------------------------------------------------------------- #
class XGBoostMetaLabelingRegimeTrainer(XGBoostMetaLabelingTrainer):
    """Meta-labeling with regime-aware stage 2.

    Stage 2 input = [features | stage1_oof_pred | day_mean(stage1_pred) |
    day_std(stage1_pred) | day_rank_pct(stage1_pred)].
    At predict time, ``set_predict_context(dates)`` must be called by the
    gate so per-date aggregates can be computed from the test cross-
    section; if dates are unavailable the entire batch is treated as one
    cohort (degraded mode).
    """
    name = 'xgb_meta_label_regime'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._predict_dates = None
        self._n_regime_extra = 3

    def set_predict_context(self, dates):
        self._predict_dates = (
            np.asarray(dates) if dates is not None else None)

    @staticmethod
    def _compute_day_regime_features(preds, dates):
        preds = np.asarray(preds, dtype=np.float64)
        n = len(preds)
        out = np.empty((n, 3), dtype=np.float32)
        if dates is None:
            day_mean = float(np.mean(preds)) if n > 0 else 0.5
            day_std = float(np.std(preds)) if n > 1 else 0.0
            denom = max(n - 1, 1)
            order = np.argsort(np.argsort(preds))
            out[:, 0] = day_mean
            out[:, 1] = day_std
            out[:, 2] = (order.astype(np.float64) / denom).astype(np.float32)
            return out
        dates = np.asarray(dates)
        for d in np.unique(dates):
            mask = dates == d
            sl = preds[mask]
            if len(sl) <= 1:
                out[mask, 0] = float(sl[0]) if len(sl) == 1 else 0.5
                out[mask, 1] = 0.0
                out[mask, 2] = 0.5
                continue
            out[mask, 0] = float(np.mean(sl))
            out[mask, 1] = float(np.std(sl))
            order = np.argsort(np.argsort(sl))
            denom = max(len(sl) - 1, 1)
            out[mask, 2] = (order.astype(np.float64) / denom).astype(np.float32)
        return out

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val.')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for '
                'regime aggregates.')

        y_tr = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_va = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        self._n_features = Xt.shape[1]
        p = self._params

        dates_arr = np.asarray(dates_train)
        unique_dates = np.sort(np.unique(dates_arr))
        cutoff_idx = int(float(p['stage1_train_frac']) * len(unique_dates))
        cutoff_idx = max(1, min(cutoff_idx, len(unique_dates) - 1))
        late_cutoff = unique_dates[cutoff_idx]
        early_mask = dates_arr < late_cutoff
        late_mask = ~early_mask

        if early_mask.sum() < 50 or late_mask.sum() < 50:
            rng = np.random.RandomState(int(p['random_state']))
            shuffle = rng.permutation(len(Xt))
            half = len(Xt) // 2
            early_mask = np.zeros(len(Xt), dtype=bool)
            late_mask = np.zeros(len(Xt), dtype=bool)
            early_mask[shuffle[:half]] = True
            late_mask[shuffle[half:]] = True

        if len(set(y_tr[early_mask].tolist())) < 2 or \
           len(set(y_tr[late_mask].tolist())) < 2:
            raise ValueError(
                f'{self.name}: degenerate class distribution after time-split.')

        stage1_first_dtrain = xgb.DMatrix(
            Xt[early_mask], label=y_tr[early_mask].astype(np.float32))
        stage1_first_dval = xgb.DMatrix(
            Xt[late_mask], label=y_tr[late_mask].astype(np.float32))
        stage1_first_booster = xgb.train(
            params=self._xgb_params('stage1'),
            dtrain=stage1_first_dtrain,
            num_boost_round=int(p['stage1_n_estimators']),
            evals=[(stage1_first_dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        self._stage1_first_iter = int(
            getattr(stage1_first_booster, 'best_iteration', 0))

        stage1_oof_late = stage1_first_booster.predict(stage1_first_dval)
        stage1_pred_val = stage1_first_booster.predict(xgb.DMatrix(Xv))

        try:
            from sklearn.metrics import roc_auc_score
            self._stage1_oof_auc = float(
                roc_auc_score(y_tr[late_mask], stage1_oof_late))
        except Exception:
            self._stage1_oof_auc = None

        # NEW: per-date regime aggregates of stage 1 predictions.
        late_dates = dates_arr[late_mask]
        regime_tr = self._compute_day_regime_features(
            stage1_oof_late, late_dates)
        regime_val = self._compute_day_regime_features(
            stage1_pred_val, np.asarray(dates_val))

        X_stage2_tr = np.column_stack([
            Xt[late_mask],
            stage1_oof_late.astype(np.float32).reshape(-1, 1),
            regime_tr,
        ]).astype(np.float32)
        X_stage2_val = np.column_stack([
            Xv,
            stage1_pred_val.astype(np.float32).reshape(-1, 1),
            regime_val,
        ]).astype(np.float32)

        stage2_dtrain = xgb.DMatrix(
            X_stage2_tr, label=y_tr[late_mask].astype(np.float32))
        stage2_dval = xgb.DMatrix(
            X_stage2_val, label=y_va.astype(np.float32))
        self.stage2 = xgb.train(
            params=self._xgb_params('stage2'),
            dtrain=stage2_dtrain,
            num_boost_round=int(p['stage2_n_estimators']),
            evals=[(stage2_dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        self._best_iteration = int(getattr(self.stage2, 'best_iteration', 0))
        try:
            from sklearn.metrics import roc_auc_score
            self._stage2_val_auc = float(roc_auc_score(
                y_va, self.stage2.predict(stage2_dval)))
        except Exception:
            self._stage2_val_auc = None

        stage1_final_dtrain = xgb.DMatrix(Xt, label=y_tr.astype(np.float32))
        stage1_final_dval = xgb.DMatrix(Xv, label=y_va.astype(np.float32))
        self.stage1_final = xgb.train(
            params=self._xgb_params('stage1'),
            dtrain=stage1_final_dtrain,
            num_boost_round=int(p['stage1_n_estimators']),
            evals=[(stage1_final_dval, 'val')],
            early_stopping_rounds=int(p['early_stopping_rounds']),
            verbose_eval=verbose,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        if self.stage1_final is None or self.stage2 is None:
            raise RuntimeError('Model not fit')
        X_arr = np.asarray(X, dtype=np.float32)
        stage1_pred = self.stage1_final.predict(xgb.DMatrix(X_arr))
        regime = self._compute_day_regime_features(
            stage1_pred, self._predict_dates)
        X_meta = np.column_stack([
            X_arr,
            stage1_pred.astype(np.float32).reshape(-1, 1),
            regime,
        ]).astype(np.float32)
        return self.stage2.predict(xgb.DMatrix(X_meta))


# --------------------------------------------------------------------- #
# MC-Dropout Feature Mask Classifier — predict-time uncertainty via random
# feature masking. Inherits strict-win training (y = pnl > 0); overrides
# predict_proba to run K stochastic passes, masking drop_rate fraction of
# features per row to NaN (XGB's native missing → default-direction routing
# at each split). Final score = mean(p_k) − conf_lambda · std(p_k), so rows
# whose prediction depends on a single fragile feature get penalised, and
# only predictions stable under input perturbation pass the gate threshold.
#
# Motivation: the xgb_meta_label sweep (#619-#632) keeps producing high-WR
# threshold bands JUST below the n>=20 cliff (e.g. #624 W6: 17 trades / 70.6%
# WR / +22.4% ann — would pass at n>=20; #631 W4: 19 trades / 42.1% WR — same
# story). This is the signature of confident-but-narrow predictions: the
# model bets hard on a small cluster of (stock, day) rows whose score depends
# on a specific feature value being intact. When the regime shifts (W4 2024-Q4,
# W5 2025-Q1), the same feature drifts and the cluster's WR collapses.
#
# Feature-dropout uncertainty is structurally orthogonal to every existing
# trainer family in the registry: bagging averages across MODELS, mixup
# perturbs TRAINING rows, DRO reweights TRAINING quarters, adv-val reweights
# train ROWS by test-similarity. None perturb the predict-time INPUT. The
# Gal & Ghahramani (2016) MC-dropout intuition — that dropout at inference
# yields a Bayesian uncertainty estimate — transfers to trees via XGB's
# native NaN routing: each masked feature is replaced by the default-direction
# vote, and the spread across K masks measures how much the prediction
# depends on the SPECIFIC unmasked feature combination.
#
# Concretely: a (stock, day) row predicted +0.70 across all 15 masks (std=0.02)
# is robust → score ≈ 0.70 − 0.5·0.02 = 0.69. A row predicted +0.70 in 8
# passes and +0.40 in 7 passes (std=0.15) is fragile → score ≈ 0.70 − 0.5·0.15
# = 0.625, demoted below the n=20-yielding threshold.
class XGBoostMCDropoutClassifierTrainer(XGBoostStrictWinClassifierTrainer):
    name = 'xgb_mcdropout_classifier'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 magnitude_scale: float = 15.0,
                 base_weight: float = 0.6,
                 pos_class_weight: float = 3.0,
                 early_stopping_rounds: int = 30,
                 drop_rate: float = 0.20,
                 n_dropout_passes: int = 15,
                 conf_lambda: float = 0.5,
                 random_state: int = 42):
        super().__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            magnitude_scale=magnitude_scale,
            base_weight=base_weight,
            pos_class_weight=pos_class_weight,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self._params['drop_rate'] = float(drop_rate)
        self._params['n_dropout_passes'] = int(n_dropout_passes)
        self._params['conf_lambda'] = float(conf_lambda)
        self._last_mean_uncertainty = None  # diagnostic, set in predict_proba

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        p = self._params
        X_arr = np.asarray(X, dtype=np.float64)
        n_rows, n_feat = X_arr.shape
        K = max(1, int(p['n_dropout_passes']))
        drop_rate = max(0.0, min(0.95, float(p['drop_rate'])))
        n_drop = int(round(drop_rate * n_feat))

        if K == 1 or n_drop == 0:
            # Degenerate config — fall back to plain XGB predict
            return self.clf.predict_proba(X_arr)[:, 1]

        rng = np.random.default_rng(int(p['random_state']))
        probs = np.empty((K, n_rows), dtype=np.float64)
        row_idx_template = np.repeat(np.arange(n_rows), n_drop)

        for k in range(K):
            X_masked = X_arr.copy()
            # argpartition on random scores → indices of n_drop smallest per row
            scores = rng.random((n_rows, n_feat))
            drop_cols = np.argpartition(scores, n_drop - 1, axis=1)[:, :n_drop]
            X_masked[row_idx_template, drop_cols.reshape(-1)] = np.nan
            probs[k] = self.clf.predict_proba(X_masked)[:, 1]

        mean_p = probs.mean(axis=0)
        std_p = probs.std(axis=0)
        self._last_mean_uncertainty = float(std_p.mean())
        score = mean_p - float(p['conf_lambda']) * std_p
        # Clip into [0, 1] so threshold semantics in return_gate.py
        # (SCORE_THRESHOLDS) stay valid even when conf_lambda is aggressive.
        return np.clip(score, 0.0, 1.0)


# --------------------------------------------------------------------- #
# Diverse-Objective Rank-Fusion Ensemble (xgb_rank_fusion)
# --------------------------------------------------------------------- #
# Motivation: post-gate-fix history (#200+) shows distinct trainers fail in
# distinct walk-forward windows:
#   - iter #589 xgb_quarterly_dro: 6/7 (avg_ann 26.2%) — fails only W7
#     (2025-09..2026-02, regime shift).
#   - iter #631 xgb_meta_label:    5/7 (avg_ann 18.6%) — fails W2/W4, passes W7.
#   - iter #565 xgb_temporal_mixup:5/7 (avg_ann 17.7%) — fails different windows.
# No single trainer has ever passed 7/7. The failure modes are window-specific
# and trainer-specific — uncorrelated noise across objectives. That is the
# textbook signature of an ensembling opportunity: if base failures are not
# perfectly correlated, a consensus filter has a strictly lower joint-failure
# probability than any individual base.
#
# Prior ensemble attempts and why this is different:
#   - StackedRanker (16 iters, 0 pass): learns simplex weights on val via grid
#     search → weights overfit the train-period regime; out-of-sample weights
#     wrong. We avoid LEARNED weights entirely.
#   - BaggedEVGatedRanker / BaggedXGBRegressor (10+45 iters, 0 pass): bagging
#     with same base trainer + seed shuffle → low diversity, all bags share
#     the same loss family's regime-edge blind spots.
#
# Design: three structurally-distinct XGBoost bases (different objectives,
# different loss landscapes), each fit independently, predictions fused via
# WITHIN-TEST-SET QUANTILE-RANK GEOMETRIC MEAN (no learned weights).
#
#   Base 1: Strict-win BCE classifier   — direct P(pnl>0)
#   Base 2: Huber regressor on pnl       — continuous EV, different gradient
#   Base 3: Quarterly DRO classifier     — per-quarter loss-reweighted (regime)
#
# Fusion: for each base i, convert raw score p_i to within-batch quantile rank
# q_i ∈ (0, 1]. Combined score = (q1 * q2 * q3) ** (1/3). Geometric mean is
# harsh on disagreement — if any one base ranks a row low, the combined rank
# drops sharply. Equivalent in log-space to mean(log q_i), so any single
# very-bad base ≈ kills the consensus. This implements an implicit "all-must-
# agree" filter without any tunable mixing coefficients.
#
# Why geometric-mean of rank quantiles (not weighted avg of raw scores):
#   1. Rank quantile is regime-invariant by construction — bounded [0, 1]
#      with a uniform marginal, regardless of how the base model's raw score
#      drifts across regimes. The StackedRanker over-fit failure came from
#      weighting raw scores whose scales drift; quantile rank removes drift.
#   2. Geometric mean encodes consensus harshly — arithmetic average lets a
#      single confident base outvote two uncertain ones; geometric mean
#      requires all bases to agree (any q→0 kills the product).
#   3. No tunable mixing parameters → nothing to overfit on val. The bases
#      themselves are tuned (HPs) but the FUSION is a fixed function.
#
# Expected behavior: combined score has the same per-day rank order as a
# soft AND of the three bases' rankings → top-K-per-day picks are stocks
# that ALL three bases independently rank highly. Should reduce false-positive
# regime-edge picks (where one base over-calls but the others abstain) at
# minor cost to true-positive rate (some W2/W3 wins where only quarterly_dro
# called correctly won't survive the consensus). Net effect: trading TRADE-
# COUNT-AT-WR-FLOOR for HIGHER WR — the exact tradeoff failing windows need.
class XGBoostRankFusionTrainer(BaseTrainer):
    name = 'xgb_rank_fusion'

    def __init__(self,
                 # Shared XGBoost knobs (applied to all three bases)
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 300,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 # Strict-win classifier-only knobs
                 pos_class_weight: float = 3.0,
                 magnitude_scale: float = 15.0,
                 base_weight: float = 0.6,
                 # Huber regressor-only knobs
                 huber_slope: float = 0.05,
                 # Quarterly DRO-only knobs
                 dro_strength: float = 1.0,
                 pass1_estimators_frac: float = 0.5,
                 weight_smoothing: float = 0.1,
                 # Common
                 early_stopping_rounds: int = 30,
                 # Hard abstention floor: rows where any base's within-batch
                 # rank quantile < abstain_min_q have fused score zeroed (so
                 # the threshold sweep excludes them). Default 0.0 = no
                 # abstention (backward-compatible with iter 661's 6/7 config).
                 abstain_min_q: float = 0.0,
                 random_state: int = 42):
        self._params = dict(
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
            min_child_weight=float(min_child_weight),
            gamma=float(gamma),
            pos_class_weight=float(pos_class_weight),
            magnitude_scale=float(magnitude_scale),
            base_weight=float(base_weight),
            huber_slope=float(huber_slope),
            dro_strength=float(dro_strength),
            pass1_estimators_frac=float(pass1_estimators_frac),
            weight_smoothing=float(weight_smoothing),
            early_stopping_rounds=int(early_stopping_rounds),
            abstain_min_q=float(abstain_min_q),
            random_state=int(random_state),
        )
        self.classifier = None  # XGBoostStrictWinClassifierTrainer
        self.regressor = None   # XGBoostHuberRegressorTrainer
        self.dro = None         # XGBoostQuarterlyDROTrainer
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train/pnl_val for its three '
                'sub-trainers (strict-win label, huber pnl target, DRO).')
        if dates_train is None:
            raise ValueError(
                f'{self.name} requires dates_train for the DRO sub-trainer.')

        p = self._params
        self._n_features = int(np.asarray(X_train).shape[1])

        # Distinct sub-seeds prevent the three bases from deterministically
        # collapsing onto identical tree structures — adds genuine sample
        # diversity on top of objective diversity.
        rs = int(p['random_state'])

        self.classifier = XGBoostStrictWinClassifierTrainer(
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            n_estimators=p['n_estimators'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            magnitude_scale=p['magnitude_scale'],
            base_weight=p['base_weight'],
            pos_class_weight=p['pos_class_weight'],
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=rs,
        )
        self.classifier.fit(
            X_train, y_train, X_val, y_val, verbose=False,
            pnl_train=pnl_train, pnl_val=pnl_val,
            dates_train=dates_train, dates_val=dates_val,
        )

        self.regressor = XGBoostHuberRegressorTrainer(
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            n_estimators=p['n_estimators'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            huber_slope=p['huber_slope'],
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=rs + 7,
        )
        self.regressor.fit(
            X_train, y_train, X_val, y_val, verbose=False,
            pnl_train=pnl_train, pnl_val=pnl_val,
            dates_train=dates_train, dates_val=dates_val,
        )

        self.dro = XGBoostQuarterlyDROTrainer(
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            n_estimators=p['n_estimators'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            dro_strength=p['dro_strength'],
            pass1_estimators_frac=p['pass1_estimators_frac'],
            weight_smoothing=p['weight_smoothing'],
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=rs + 13,
        )
        self.dro.fit(
            X_train, y_train, X_val, y_val, verbose=False,
            pnl_train=pnl_train, pnl_val=pnl_val,
            dates_train=dates_train, dates_val=dates_val,
        )
        return self

    @staticmethod
    def _quantile_rank(x: np.ndarray) -> np.ndarray:
        """Within-batch quantile rank ∈ (0, 1]. Average-rank for ties."""
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        if n == 0:
            return x
        # argsort-of-argsort gives 0-indexed ranks (ties broken by order, but
        # consistent across runs). Add 1 so output ∈ [1/n, 1] — strictly > 0
        # so the geometric mean's log is finite.
        ranks = np.argsort(np.argsort(x, kind='stable'), kind='stable')
        return (ranks + 1).astype(np.float64) / n

    def predict_proba(self, X) -> np.ndarray:
        if self.classifier is None or self.regressor is None or self.dro is None:
            raise RuntimeError('Model not fit')
        p1 = self.classifier.predict_proba(X)
        p2 = self.regressor.predict_proba(X)
        p3 = self.dro.predict_proba(X)
        q1 = self._quantile_rank(p1)
        q2 = self._quantile_rank(p2)
        q3 = self._quantile_rank(p3)
        # Geometric mean in log-space (numerically stable for n large).
        log_geo = (np.log(q1) + np.log(q2) + np.log(q3)) / 3.0
        fused = np.exp(log_geo)
        # Hard abstention: any base ranking the row below abstain_min_q
        # collapses the fused score to zero so the threshold sweep skips it.
        # This is stricter than the soft penalty geometric-mean already applies
        # — it forbids trades where one base actively dissents.
        amq = float(self._params.get('abstain_min_q', 0.0))
        if amq > 0.0:
            min_q = np.minimum(np.minimum(q1, q2), q3)
            fused = np.where(min_q >= amq, fused, 0.0)
        return fused

    @property
    def best_iteration(self):
        if self.classifier is None:
            return None
        return self.classifier.best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)


# --------------------------------------------------------------------- #
# Regime-Blend Classifier — two strict-win XGB heads (one trained on FULL
# data, one trained on BEAR-regime-only subset), softly blended at predict
# time by per-row market_breadth_adv. Bear-regime test rows pull the
# bear-specialist head; bull-regime test rows pull the generalist head.
#
# Motivation: the rank_fusion / mcdropout / meta_label / adv_val / quarterly_dro
# sweep (#618-#661, 130+ iters) plateaus at 3-6/7 windows. The failing
# windows (W1 Nov2023-Feb2024 with WR=17.6%, W2 Jan-Apr2024 WR=30.8%, W5
# Jan-Apr2025 WR=31.8%) ALL correspond to bear-regime test periods, while
# the passing windows (W3 May-Aug2024 WR=55%, W4 Sep-Dec2024 WR=41%, W6
# May-Aug2025 WR=46%) are recovery / bullish periods. In W1 the model picks
# losers WORSE THAN RANDOM (17.6% vs 50% random baseline) — this is signal
# INVERSION, where features that predict winners in the training (bullish)
# period predict losers in the test (bearish) period.
#
# Every prior structural change reweights TRAINING rows (focal, group-DRO,
# adversarial val, JTT, magnitude weighting) — they all train ONE model on
# the SAME set, hoping the loss-shape forces regime invariance. None hit
# 100% because a single XGB tree split that's predictive in bull regime
# IS NOT predictive in bear, no matter how the train rows are weighted.
#
# Regime blend is structurally different: it trains TWO classifiers in
# parallel, and ROUTES at predict time. Model A is the generalist (trained
# on all rows). Model B is the bear specialist (trained only on rows where
# the market regime feature is in the lower quantile of training). At test
# time, a row's market_breadth_adv value determines a soft blend weight:
# bear-like rows (low breadth) draw more from B, bull-like rows (high
# breadth) draw more from A. The blend is smooth (sigmoid-based) so there
# is no hard expert routing — bull rows still see B's signal at low weight,
# and the (limited) bear training data is amplified only where it matters.
#
# Why soft blend not hard routing: hard routing fails when the bear-regime
# test data outsizes the bear-regime train data (exactly the W1/W2/W5
# pattern — test is mostly bear, train is mostly bull, so the bear expert
# has few training rows). The soft blend ensures we always have generalist
# A as a fallback when B is undertrained, AND the blend weight is bounded
# in [0, 1] regardless of expert size mismatch.
# --------------------------------------------------------------------- #
class XGBoostRegimeBlendTrainer(BaseTrainer):
    """Regime-conditioned 2-model blend (generalist + bear-specialist).

    fit() trains:
      - model_A: XGB strict-win classifier on FULL training data
      - model_B: XGB strict-win classifier on the bear-quantile subset
        of training rows (rows where the regime feature is below
        ``bear_quantile`` percentile of the training distribution).

    predict_proba(X) returns a per-row blend:
      bear_w = sigmoid((threshold - X[:, regime_feature_idx]) / temperature)
      score  = (1 - bear_w) * model_A.predict(X) + bear_w * model_B.predict(X)

    The regime feature defaults to column 15 of the aggregated tabular X,
    which is ``last__market_breadth_adv`` under the curated feature set
    (see models/feature_eng.py CURATED_FEATURES). If ``model_B``'s training
    subset is smaller than ``min_specialist_samples``, ``model_B`` is skipped
    and the trainer degenerates to ``model_A`` alone — preserving graceful
    fallback when a window's training data has too few bear samples.
    """
    name = 'xgb_regime_blend'

    def __init__(self,
                 # Shared XGB tree-family knobs (both heads use these)
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 # Strict-win classifier weighting knobs (both heads)
                 magnitude_scale: float = 15.0,
                 base_weight: float = 0.6,
                 pos_class_weight: float = 2.5,
                 # Regime-blend mechanics. Defaults err on the side of
                 # NEAR-HARD routing (temperature=0.05) so the bull-regime
                 # rows get pure model_A and bear rows get pure model_B —
                 # soft blending in the transition zone produced noisy
                 # ranking contamination at default HPs.
                 regime_feature_idx: int = 15,
                 bear_quantile: float = 0.30,
                 temperature: float = 0.05,
                 min_specialist_samples: int = 500,
                 # Common
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self._params = dict(
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
            min_child_weight=float(min_child_weight),
            gamma=float(gamma),
            magnitude_scale=float(magnitude_scale),
            base_weight=float(base_weight),
            pos_class_weight=float(pos_class_weight),
            regime_feature_idx=int(regime_feature_idx),
            bear_quantile=float(bear_quantile),
            temperature=float(temperature),
            min_specialist_samples=int(min_specialist_samples),
            early_stopping_rounds=int(early_stopping_rounds),
            random_state=int(random_state),
        )
        self.model_a = None
        self.model_b = None
        self._regime_threshold = None
        self._specialist_fit = False
        self._n_features = None

    def _build_strict_win(self, seed_offset: int = 0):
        p = self._params
        return XGBoostStrictWinClassifierTrainer(
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            n_estimators=p['n_estimators'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            magnitude_scale=p['magnitude_scale'],
            base_weight=p['base_weight'],
            pos_class_weight=p['pos_class_weight'],
            early_stopping_rounds=p['early_stopping_rounds'],
            random_state=int(p['random_state']) + int(seed_offset),
        )

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train/pnl_val (strict-win label).')
        X_tr = np.asarray(X_train, dtype=np.float32)
        X_va = np.asarray(X_val, dtype=np.float32)
        self._n_features = X_tr.shape[1]
        p = self._params

        idx = int(p['regime_feature_idx'])
        if idx < 0 or idx >= self._n_features:
            raise ValueError(
                f'{self.name}: regime_feature_idx={idx} out of bounds for '
                f'X with {self._n_features} columns.')

        # Quantile boundary on the SCALED training feature.
        # X has already been RobustScaler-transformed by the gate, so the
        # quantile here is on the scaled-units space — and the same scaling
        # is applied at predict, so it's consistent.
        regime_train = X_tr[:, idx]
        self._regime_threshold = float(
            np.quantile(regime_train, p['bear_quantile']))

        # --- Head A: generalist on FULL train --------------------------------
        self.model_a = self._build_strict_win(seed_offset=0)
        self.model_a.fit(
            X_tr, np.asarray(y_train), X_va, np.asarray(y_val),
            verbose=False,
            pnl_train=pnl_train, pnl_val=pnl_val,
            dates_train=dates_train, dates_val=dates_val,
        )

        # --- Head B: bear-regime specialist on subset ------------------------
        bear_mask = regime_train <= self._regime_threshold
        n_bear = int(bear_mask.sum())
        # Require both bear samples AND class diversity in the subset.
        if n_bear >= int(p['min_specialist_samples']):
            pnl_bear = np.asarray(pnl_train)[bear_mask]
            y_bear_strict = (pnl_bear > 0.0).astype(np.int32)
            if len(set(y_bear_strict.tolist())) >= 2:
                # Bear-subset val: use rows from X_val whose regime feature is
                # also bear-side, so early stopping picks a model_B iteration
                # tuned to bear-distribution loss, not full-val (bull-leaning)
                # loss. Falls back to full val if bear-subset val is too thin
                # (< 30 rows or single-class).
                val_regime = X_va[:, idx]
                val_bear_mask = val_regime <= self._regime_threshold
                y_val_strict = (np.asarray(pnl_val) > 0.0).astype(np.int32)
                if (val_bear_mask.sum() >= 30
                        and len(set(y_val_strict[val_bear_mask].tolist())) >= 2):
                    Xv_for_b = X_va[val_bear_mask]
                    yv_for_b = np.asarray(y_val)[val_bear_mask]
                    pnlv_for_b = np.asarray(pnl_val)[val_bear_mask]
                    datesv_for_b = (np.asarray(dates_val)[val_bear_mask]
                                    if dates_val is not None else None)
                else:
                    Xv_for_b = X_va
                    yv_for_b = np.asarray(y_val)
                    pnlv_for_b = np.asarray(pnl_val)
                    datesv_for_b = dates_val
                self.model_b = self._build_strict_win(seed_offset=11)
                self.model_b.fit(
                    X_tr[bear_mask], np.asarray(y_train)[bear_mask],
                    Xv_for_b, yv_for_b,
                    verbose=False,
                    pnl_train=pnl_bear,
                    pnl_val=pnlv_for_b,
                    dates_train=(np.asarray(dates_train)[bear_mask]
                                 if dates_train is not None else None),
                    dates_val=datesv_for_b,
                )
                self._specialist_fit = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model_a is None:
            raise RuntimeError(f'{self.name}: not fit')
        X_arr = np.asarray(X, dtype=np.float32)
        p_a = self.model_a.predict_proba(X_arr)
        if not self._specialist_fit or self.model_b is None:
            return p_a

        p = self._params
        idx = int(p['regime_feature_idx'])
        thr = float(self._regime_threshold)
        T = max(1e-6, float(p['temperature']))
        # bear_w high when regime feature < threshold; smooth via sigmoid.
        # Clip the logit to avoid overflow on extreme outliers (RobustScaler
        # uses IQR, so 1.5-sigma outliers can be in [-3, 3]; bigger blowups
        # are rare but possible on tail rows). The clip keeps exp() finite.
        logit = (thr - X_arr[:, idx]) / T
        logit = np.clip(logit, -30.0, 30.0)
        bear_w = 1.0 / (1.0 + np.exp(-logit))
        p_b = self.model_b.predict_proba(X_arr)
        return (1.0 - bear_w) * p_a + bear_w * p_b

    @property
    def best_iteration(self):
        if self.model_a is None:
            return None
        return self.model_a.best_iteration

    @property
    def hyperparams(self):
        return dict(self._params)


# --------------------------------------------------------------------- #
# Recency-Consensus Classifier — two strict-win XGB heads (full-history
# view + exponential-decay recent-history view), geometric-mean fused at
# predict time. Picks must score high in BOTH temporal lenses to clear
# the consensus filter.
#
# Motivation: the failing windows (W1 Nov2023-Feb2024 WR=17.6%, W4
# Sep-Dec2024 WR=20.8%, W5 Jan-Apr2025 WR=31.8%, W7 Sep2025-Feb2026
# WR=21.3% — all on rank_fusion default) follow a pattern: test period
# is a bear/transition immediately AFTER a bull-dominated train. Every
# prior trainer (focal, DRO, JTT, adv_val, meta_label, rank_fusion,
# regime_blend, mcdropout) treats train rows with uniform TEMPORAL
# weight — only their PER-ROW row weighting (magnitude, focal, DRO
# group, adversarial-importance) differs. This means an outdated bullish
# regime signal from the early-train months dominates the gradient
# equally with the late-train signal that should better preview test.
#
# Recency consensus is a structurally distinct knob: it weights train
# rows by their AGE (exp decay from train_end). The "recent" head sees
# late-train rows at full weight and old-train rows at min_recent_weight,
# capturing the regime signal closest to test. The "full" head sees all
# rows uniformly (the existing strict_win behavior). Geometric-mean
# fusion at predict keeps only stocks where BOTH lenses agree — a stock
# only flagged by the full-history head (likely an outdated bullish
# pattern) gets suppressed by the recent head; a stock only flagged by
# the recent head (likely a one-off late-train fluke) gets suppressed
# by the full head. The intersection-style consensus is structurally
# different from rank_fusion's geo-mean-of-different-objectives (which
# uses the SAME train data for all three bases) — here the two heads
# disagree EXACTLY when train-time regime is non-stationary, which is
# the failure mode we need to filter out.
# --------------------------------------------------------------------- #
class XGBoostRecencyConsensusTrainer(BaseTrainer):
    """Recency-consensus strict-win classifier (full + recent heads, geo mean)."""
    name = 'xgb_recency_consensus'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 magnitude_scale: float = 15.0,
                 base_weight: float = 0.6,
                 pos_class_weight: float = 3.0,
                 recency_halflife_days: float = 60.0,
                 min_recent_weight: float = 0.20,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        # recency_halflife_days: shorter half-life concentrates the recent
        #   head onto the last weeks of train. 30d ≈ last quarter dominates;
        #   180d ≈ roughly uniform across 6-mo train. Default 60d puts ~75%
        #   of weight on the most-recent half of a 6-mo train window.
        # min_recent_weight: floor for the oldest train rows in the recent
        #   head, AFTER exp decay. 0.0 means oldest rows ignored (high
        #   variance from small ESS); 1.0 means uniform (no recency
        #   advantage). 0.20 keeps the old rows as a soft regularizer.
        self._params = dict(
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
            min_child_weight=float(min_child_weight),
            gamma=float(gamma),
            magnitude_scale=float(magnitude_scale),
            base_weight=float(base_weight),
            pos_class_weight=float(pos_class_weight),
            recency_halflife_days=float(recency_halflife_days),
            min_recent_weight=float(min_recent_weight),
            early_stopping_rounds=int(early_stopping_rounds),
            random_state=int(random_state),
        )
        self.clf_full = None
        self.clf_recent = None
        self._best_iteration_full = None
        self._best_iteration_recent = None

    def _build_magnitude_weights(self, pnl):
        p = self._params
        pnl_arr = np.asarray(pnl, dtype=np.float64)
        raw = np.abs(pnl_arr) * p['magnitude_scale'] + p['base_weight']
        m = raw.mean()
        if m <= 0:
            return np.ones_like(raw, dtype=np.float32)
        return (raw / m).astype(np.float32)

    @staticmethod
    def _to_ordinal(dates) -> np.ndarray:
        """Convert assorted date inputs to numeric days. Works for numpy
        datetime64, pandas Timestamp, datetime.date/datetime, or ISO date
        strings — falls back to row-index if all parsing fails."""
        arr = np.asarray(dates)
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype('datetime64[D]').astype(np.int64).astype(np.float64)
        try:
            import pandas as pd
            ts = pd.to_datetime(arr)
            return ts.values.astype('datetime64[D]').astype(np.int64).astype(np.float64)
        except Exception:
            return np.arange(len(arr), dtype=np.float64)

    def _build_recency_weights(self, dates, ref_day: float) -> np.ndarray:
        p = self._params
        days = self._to_ordinal(dates)
        age = np.maximum(0.0, ref_day - days)
        halflife = max(1.0, p['recency_halflife_days'])
        w = np.power(0.5, age / halflife)
        w = np.maximum(w, p['min_recent_weight'])
        return w.astype(np.float32)

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val for strict-win '
                'labels and magnitude weighting.')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for '
                'recency weighting.')

        p = self._params

        y_tr_strict = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_va_strict = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr_strict)) < 2:
            raise ValueError('Train set has only one class after strict-win — fit aborted')

        mag_tr = self._build_magnitude_weights(pnl_train)
        mag_va = self._build_magnitude_weights(pnl_val)

        days_tr = self._to_ordinal(dates_train)
        days_va = self._to_ordinal(dates_val)
        ref_day = float(max(days_tr.max(), days_va.max()))
        rec_tr = self._build_recency_weights(dates_train, ref_day)
        rec_va = self._build_recency_weights(dates_val, ref_day)

        comb_tr = mag_tr * rec_tr
        m_tr = comb_tr.mean()
        if m_tr > 0:
            comb_tr = (comb_tr / m_tr).astype(np.float32)
        comb_va = mag_va * rec_va
        m_va = comb_va.mean()
        if m_va > 0:
            comb_va = (comb_va / m_va).astype(np.float32)

        common_kw = dict(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            scale_pos_weight=p['pos_class_weight'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=p['early_stopping_rounds'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )

        self.clf_full = xgb.XGBClassifier(
            random_state=p['random_state'],
            **common_kw,
        )
        self.clf_full.fit(
            np.asarray(X_train), y_tr_strict,
            sample_weight=mag_tr,
            eval_set=[(np.asarray(X_val), y_va_strict)],
            sample_weight_eval_set=[mag_va],
            verbose=verbose,
        )
        self._best_iteration_full = getattr(self.clf_full, 'best_iteration', None) or p['n_estimators']

        self.clf_recent = xgb.XGBClassifier(
            random_state=p['random_state'] + 7,
            **common_kw,
        )
        self.clf_recent.fit(
            np.asarray(X_train), y_tr_strict,
            sample_weight=comb_tr,
            eval_set=[(np.asarray(X_val), y_va_strict)],
            sample_weight_eval_set=[comb_va],
            verbose=verbose,
        )
        self._best_iteration_recent = getattr(self.clf_recent, 'best_iteration', None) or p['n_estimators']
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf_full is None or self.clf_recent is None:
            raise RuntimeError('Model not fit')
        X_arr = np.asarray(X)
        p_full = self.clf_full.predict_proba(X_arr)[:, 1]
        p_recent = self.clf_recent.predict_proba(X_arr)[:, 1]
        eps = 1e-7
        log_geo = (np.log(np.clip(p_full, eps, 1.0))
                   + np.log(np.clip(p_recent, eps, 1.0))) / 2.0
        return np.exp(log_geo)

    @property
    def best_iteration(self):
        if self._best_iteration_full is None:
            return None
        return max(self._best_iteration_full, self._best_iteration_recent or 0)

    @property
    def hyperparams(self):
        return dict(self._params)


# --------------------------------------------------------------------- #
# Day-Quality Consensus — three-head strict-win classifier. Heads A and B
# match recency_consensus (uniform vs exp-decay temporal weighting). Head
# C is the structural novelty: a per-day-quality regressor supervised by
# the per-DATE mean strict-win rate (a quantity that is constant within a
# date but varies across dates). Geometric-mean-of-three fusion at predict.
#
# Motivation: even the best xgb_recency_consensus run (iter #688 default,
# 5/7 default, +27% avg_ann) still fails W5 (Jan-Apr2025 WR 34.4%) and
# W7 (Sep2025-Feb2026 WR 21.2%) — both bear/transition windows where the
# per-row classifier picks 30+ trades whose WR sits below the 40% floor.
# All existing trainers (focal, DRO, JTT, adv_val, meta_label, mcdropout,
# rank_fusion, regime_blend, recency_consensus) train heads on the SAME
# row-level signal (strict-win or magnitude-weighted variants), so each
# head, no matter the temporal weighting, optimizes for "which row is a
# winner?" — never for "is the regime supportive of trading today?".
#
# Per-day mean strict-win rate is an inherently day-level signal: the per
# row value is a CONSTANT for all rows sharing a date, so a regressor
# trained on it cannot use intra-day cross-symbol signals (atr_pct,
# volume_ratio, etc.) to discriminate — only features that ARE
# day-level (market_breadth_*, set_above_sma20, up_days_5d,
# sector_breadth, market_new_highs, sector_avg_*, etc.) carry the
# gradient. This explicitly forces the third head to leverage the regime
# features that are already in the feature set but dominated by per-row
# features in heads A/B's gradient.
#
# At predict: head C output (in [0,1] via reg:logistic) is floored at
# day_quality_floor so a clearly-bearish-day prediction can suppress but
# not zero out, then folded into the geometric mean with adjustable
# weight. Result: stocks only clear the gate when (a) the full-history
# classifier likes them, AND (b) the recent-tilt classifier likes them,
# AND (c) the regime regressor says it's a tradeable day.
# --------------------------------------------------------------------- #
class XGBoostDayQualityConsensusTrainer(BaseTrainer):
    """Three-head consensus: full + recent strict-win + day-quality regressor."""
    name = 'xgb_day_quality_consensus'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 400,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.7,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 5.0,
                 gamma: float = 0.1,
                 magnitude_scale: float = 15.0,
                 base_weight: float = 0.6,
                 pos_class_weight: float = 3.0,
                 recency_halflife_days: float = 120.0,
                 min_recent_weight: float = 0.30,
                 day_quality_floor: float = 0.30,
                 day_quality_weight: float = 1.0,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42):
        # day_quality_floor: floor for head C's effective output BEFORE the
        #   geo-mean. 0.0 = head C can fully zero a score on bad days; 1.0 =
        #   head C is bypassed entirely. 0.30 lets head C dampen by ~70% at
        #   most, keeping the consensus from collapsing on borderline days.
        # day_quality_weight: exponent applied to head C in the geo-mean.
        #   The fused score = (p_a * p_b * p_dq_eff^w)^(1/(2+w)). w=0 is
        #   plain two-head recency_consensus; w=1 weights C equally with A,B;
        #   w=2 makes C dominant. Train mode should search [0.0, 2.5].
        self._params = dict(
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
            min_child_weight=float(min_child_weight),
            gamma=float(gamma),
            magnitude_scale=float(magnitude_scale),
            base_weight=float(base_weight),
            pos_class_weight=float(pos_class_weight),
            recency_halflife_days=float(recency_halflife_days),
            min_recent_weight=float(min_recent_weight),
            day_quality_floor=float(day_quality_floor),
            day_quality_weight=float(day_quality_weight),
            early_stopping_rounds=int(early_stopping_rounds),
            random_state=int(random_state),
        )
        self.clf_full = None
        self.clf_recent = None
        self.reg_dq = None
        self._best_iteration_full = None
        self._best_iteration_recent = None
        self._best_iteration_dq = None

    def _build_magnitude_weights(self, pnl):
        p = self._params
        pnl_arr = np.asarray(pnl, dtype=np.float64)
        raw = np.abs(pnl_arr) * p['magnitude_scale'] + p['base_weight']
        m = raw.mean()
        if m <= 0:
            return np.ones_like(raw, dtype=np.float32)
        return (raw / m).astype(np.float32)

    @staticmethod
    def _to_ordinal(dates) -> np.ndarray:
        arr = np.asarray(dates)
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype('datetime64[D]').astype(np.int64).astype(np.float64)
        try:
            import pandas as pd
            ts = pd.to_datetime(arr)
            return ts.values.astype('datetime64[D]').astype(np.int64).astype(np.float64)
        except Exception:
            return np.arange(len(arr), dtype=np.float64)

    def _build_recency_weights(self, dates, ref_day: float) -> np.ndarray:
        p = self._params
        days = self._to_ordinal(dates)
        age = np.maximum(0.0, ref_day - days)
        halflife = max(1.0, p['recency_halflife_days'])
        w = np.power(0.5, age / halflife)
        w = np.maximum(w, p['min_recent_weight'])
        return w.astype(np.float32)

    @staticmethod
    def _per_date_mean(dates, values) -> np.ndarray:
        """Broadcast per-date mean(values) back onto each row. Constant
        within a date so regressors can only fit it via day-level features."""
        dates_arr = np.asarray(dates)
        v_arr = np.asarray(values, dtype=np.float64)
        try:
            import pandas as pd
            s = pd.Series(v_arr).groupby(pd.Series(dates_arr)).transform('mean')
            return s.values.astype(np.float32)
        except Exception:
            sums: dict = {}
            counts: dict = {}
            for d, v in zip(dates_arr, v_arr):
                sums[d] = sums.get(d, 0.0) + float(v)
                counts[d] = counts.get(d, 0) + 1
            means = {d: sums[d] / counts[d] for d in sums}
            return np.asarray([means[d] for d in dates_arr], dtype=np.float32)

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb

        if pnl_train is None or pnl_val is None:
            raise ValueError(
                f'{self.name} requires pnl_train and pnl_val for strict-win '
                'labels and magnitude weighting.')
        if dates_train is None or dates_val is None:
            raise ValueError(
                f'{self.name} requires dates_train and dates_val for '
                'recency weighting and day-quality supervision.')

        p = self._params

        y_tr_strict = (np.asarray(pnl_train, dtype=np.float64) > 0.0).astype(np.int32)
        y_va_strict = (np.asarray(pnl_val, dtype=np.float64) > 0.0).astype(np.int32)
        if len(set(y_tr_strict)) < 2:
            raise ValueError('Train set has only one class after strict-win — fit aborted')

        mag_tr = self._build_magnitude_weights(pnl_train)
        mag_va = self._build_magnitude_weights(pnl_val)

        days_tr = self._to_ordinal(dates_train)
        days_va = self._to_ordinal(dates_val)
        ref_day = float(max(days_tr.max(), days_va.max()))
        rec_tr = self._build_recency_weights(dates_train, ref_day)
        rec_va = self._build_recency_weights(dates_val, ref_day)

        comb_tr = mag_tr * rec_tr
        m_tr = comb_tr.mean()
        if m_tr > 0:
            comb_tr = (comb_tr / m_tr).astype(np.float32)
        comb_va = mag_va * rec_va
        m_va = comb_va.mean()
        if m_va > 0:
            comb_va = (comb_va / m_va).astype(np.float32)

        # Head-C target: per-DATE mean strict-win rate, broadcast to rows.
        # Constant within a date → only day-level features can fit it.
        dq_tr = self._per_date_mean(dates_train, y_tr_strict)
        dq_va = self._per_date_mean(dates_val, y_va_strict)

        cls_kw = dict(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            scale_pos_weight=p['pos_class_weight'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=p['early_stopping_rounds'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )

        self.clf_full = xgb.XGBClassifier(
            random_state=p['random_state'],
            **cls_kw,
        )
        self.clf_full.fit(
            np.asarray(X_train), y_tr_strict,
            sample_weight=mag_tr,
            eval_set=[(np.asarray(X_val), y_va_strict)],
            sample_weight_eval_set=[mag_va],
            verbose=verbose,
        )
        self._best_iteration_full = (getattr(self.clf_full, 'best_iteration', None)
                                      or p['n_estimators'])

        self.clf_recent = xgb.XGBClassifier(
            random_state=p['random_state'] + 7,
            **cls_kw,
        )
        self.clf_recent.fit(
            np.asarray(X_train), y_tr_strict,
            sample_weight=comb_tr,
            eval_set=[(np.asarray(X_val), y_va_strict)],
            sample_weight_eval_set=[comb_va],
            verbose=verbose,
        )
        self._best_iteration_recent = (getattr(self.clf_recent, 'best_iteration', None)
                                        or p['n_estimators'])

        # Head C: regressor supervised by per-date mean WR. reg:logistic
        # keeps output in (0,1), aligning naturally with the geo-mean fold.
        reg_kw = dict(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='reg:logistic',
            eval_metric='rmse',
            early_stopping_rounds=p['early_stopping_rounds'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.reg_dq = xgb.XGBRegressor(
            random_state=p['random_state'] + 13,
            **reg_kw,
        )
        eps = 1e-3
        dq_tr_clip = np.clip(dq_tr, eps, 1.0 - eps).astype(np.float32)
        dq_va_clip = np.clip(dq_va, eps, 1.0 - eps).astype(np.float32)
        self.reg_dq.fit(
            np.asarray(X_train), dq_tr_clip,
            eval_set=[(np.asarray(X_val), dq_va_clip)],
            verbose=verbose,
        )
        self._best_iteration_dq = (getattr(self.reg_dq, 'best_iteration', None)
                                    or p['n_estimators'])
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf_full is None or self.clf_recent is None or self.reg_dq is None:
            raise RuntimeError('Model not fit')
        X_arr = np.asarray(X)
        p_full = self.clf_full.predict_proba(X_arr)[:, 1]
        p_recent = self.clf_recent.predict_proba(X_arr)[:, 1]
        p_dq = np.clip(self.reg_dq.predict(X_arr), 0.0, 1.0)
        floor = self._params['day_quality_floor']
        w = self._params['day_quality_weight']
        p_dq_eff = floor + (1.0 - floor) * p_dq
        eps = 1e-7
        # Weighted geo-mean: (p_a * p_b * p_dq_eff^w)^(1/(2+w)).
        log_score = (np.log(np.clip(p_full, eps, 1.0))
                     + np.log(np.clip(p_recent, eps, 1.0))
                     + w * np.log(np.clip(p_dq_eff, eps, 1.0)))
        return np.exp(log_score / (2.0 + w))

    @property
    def best_iteration(self):
        if self._best_iteration_full is None:
            return None
        return max(
            self._best_iteration_full,
            self._best_iteration_recent or 0,
            self._best_iteration_dq or 0,
        )

    @property
    def hyperparams(self):
        return dict(self._params)


# --------------------------------------------------------------------- #
# Torch Attentive MLP — first NN-family trainer post-pyc loss
# --------------------------------------------------------------------- #
# Diagnosis (claude iter following #700): across the last 20 iter the XGB
# family plateaus on bear/choppy windows — W4 (avg ann -16.8%), W5
# (-29.0%), W7 (-16.6%) — while W6 (bull, +85.1% avg ann) passes 13/20.
# W5 corresponds to SET -9.7% Jan–Apr 2025 with breadth 0.32, the deepest
# bear in the panel. With 30+ XGB-loss variants and several thousand HP
# samples, the failure pattern is regime-shaped, not loss-shaped: piecewise
# constant tree boundaries cluster around the same losing patterns when the
# regime flips, and the registry has no NN inductive bias to compare
# against (the torch_* trainers from earlier iterations were lost when
# trainers.py was rebuilt and only the .pyc remained — see iter #696 data
# request).
#
# Hypothesis: a small MLP with a *sample-level feature-attention gate*
# (sigmoid mask over the 92 aggregated features, learned per row) lets the
# network down-weight features that mislead in bear regimes (high
# momentum / high volume look like wins in bull but get faded in bear)
# while keeping useful ones (atr_pct, market_breadth_*). This is the
# minimal "TabNet-style" gating that fits in a 30-min wall budget without
# new package dependencies. Cross-feature smoothness (NN) plus per-sample
# masking (gate) gives XGB-orthogonal coverage of the same input space.
#
# Architecture (parameters small to keep CPU training under ~30s/window):
#   Linear(F → 128) + GELU + Dropout → context z
#   Linear(z → F) → sigmoid → feature gate g (per-sample, per-feature)
#   x' = x * g (Hadamard); Linear(F → 128) + GELU + Dropout
#   Linear(128 → 64) + GELU + Dropout → Linear(64 → 1) → logit
# Loss: BCEWithLogitsLoss(pos_weight=pos_class_weight)
# Optim: AdamW, cosine schedule, early-stop on val ROC-AUC patience=5.
class TorchAttentiveMLPTrainer(BaseTrainer):
    name = 'torch_attentive_mlp'

    def __init__(self,
                 hidden_dim: int = 128,
                 bottleneck_dim: int = 64,
                 dropout: float = 0.3,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 max_epochs: int = 40,
                 patience: int = 6,
                 pos_class_weight: float = 2.0,
                 random_state: int = 42):
        self._params = dict(
            hidden_dim=int(hidden_dim),
            bottleneck_dim=int(bottleneck_dim),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            max_epochs=int(max_epochs),
            patience=int(patience),
            pos_class_weight=float(pos_class_weight),
            random_state=int(random_state),
        )
        self.net = None
        self._best_epoch = None
        self._n_features = None

    def _build_net(self, n_features):
        import torch
        import torch.nn as nn
        p = self._params

        class AttentiveMLP(nn.Module):
            def __init__(self, F, H, B, dp):
                super().__init__()
                self.ctx = nn.Sequential(
                    nn.Linear(F, H), nn.GELU(), nn.Dropout(dp),
                )
                self.gate = nn.Linear(H, F)
                self.head = nn.Sequential(
                    nn.Linear(F, H), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(H, B), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(B, 1),
                )

            def forward(self, x):
                z = self.ctx(x)
                g = torch.sigmoid(self.gate(z))
                return self.head(x * g).squeeze(-1)

        return AttentiveMLP(n_features, p['hidden_dim'],
                            p['bottleneck_dim'], p['dropout'])

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        # 92-dim input × small MLP — OMP overhead dominates on >4 threads.
        # Pin to 4 to keep per-window wall-time predictable under the 30 min budget.
        torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.manual_seed(p['random_state'])
        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        # Clip very large standardized values to keep tanh/sigmoid gradients sane
        Xt = np.clip(Xt, -8.0, 8.0)
        Xv = np.clip(Xv, -8.0, 8.0)
        yt = np.asarray(y_train, dtype=np.float32)
        yv = np.asarray(y_val, dtype=np.float32)
        self._n_features = Xt.shape[1]
        self.net = self._build_net(self._n_features)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([p['pos_class_weight']], device=device))
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=p['learning_rate'],
                                weight_decay=p['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=p['max_epochs'])

        ds = TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
        dl = DataLoader(ds, batch_size=p['batch_size'], shuffle=True,
                        drop_last=False)
        Xv_t = torch.from_numpy(Xv).to(device)
        yv_t = torch.from_numpy(yv).to(device)

        best_auc = -1.0
        best_state = None
        bad = 0
        for epoch in range(p['max_epochs']):
            self.net.train()
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = self.net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            sched.step()
            # Val AUC
            self.net.eval()
            with torch.no_grad():
                vl = torch.sigmoid(self.net(Xv_t)).cpu().numpy()
            auc = _auc_safe(yv, vl)
            if verbose:
                print(f'  ep{epoch:02d} val_auc={auc:.4f}')
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().clone() for k, v in
                              self.net.state_dict().items()}
                self._best_epoch = epoch
                bad = 0
            else:
                bad += 1
                if bad >= p['patience']:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        if self.net is None:
            raise RuntimeError('Model not fit')
        device = next(self.net.parameters()).device
        X = np.asarray(X, dtype=np.float32)
        X = np.clip(X, -8.0, 8.0)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X).to(device))
            return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    @property
    def best_iteration(self):
        return self._best_epoch

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        import torch
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pt')
        meta_path = os.path.join(output_dir, 'metadata.json')
        torch.save({'state_dict': self.net.state_dict(),
                    'n_features': self._n_features}, model_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_epoch': self._best_epoch,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import torch
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'model.pt')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        inst.net = inst._build_net(inst._n_features)
        state = torch.load(model_path, weights_only=True)
        inst.net.load_state_dict(state['state_dict'])
        inst._best_epoch = meta.get('best_epoch')
        return inst


def _auc_safe(y_true, y_score):
    """ROC-AUC that returns 0.5 on degenerate input instead of raising."""
    from sklearn.metrics import roc_auc_score
    try:
        if len(set(np.asarray(y_true).astype(int).tolist())) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


# Iter #711 — first trainer with `consumes_sequences = True`. Every existing
# trainer (XGB / LightGBM / torch_attentive_mlp) trains on the 92-dim
# aggregated tabular vector `[last, mean, std, last-mean]` produced by
# `aggregate_sequence`. That projection collapses the 20-step temporal
# trajectory into 4 summary statistics per feature — sufficient when the
# signal is stationary, lossy when the test window is a regime shift.
#
# Diagnosis (last 20 iters, all `xgb_day_quality_consensus`): W4 (test
# 2024-09..12) avg_ann=-22% / pass=4/20 and W5 (test 2025-01..04)
# avg_ann=-23% / pass=4/20 are chronic failures across every HP setting.
# Both test slices follow bull-ish training periods and contain a clear
# regime flip mid-window. The tabular aggregator throws away exactly the
# temporal evolution signal a model would need to recognise that flip.
#
# Hypothesis: a small GRU that consumes the raw (N, 20, F) sequence can
# read the temporal arc — accelerating volatility, breadth decay,
# foreign-flow zscore trend — and produce regime-aware probabilities.
# This is genuinely orthogonal inductive bias to the 30 GBDT / XGB-loss
# variants already in the registry.
#
# Architecture (kept small for CPU + 30-min wall budget):
#   GRU(F → H, 1 layer, batch_first) → take last hidden h_T
#   LayerNorm(H) → Dropout → Linear(H → H//2) → GELU → Dropout
#   Linear(H//2 → 1) → logit
# Loss: BCEWithLogitsLoss(pos_weight=pos_class_weight)
# Optim: AdamW + cosine, early-stop on val ROC-AUC, patience=4.
class TorchSeqGRUTrainer(BaseTrainer):
    name = 'torch_seq_gru'
    consumes_sequences = True

    def __init__(self,
                 hidden_dim: int = 64,
                 dropout: float = 0.30,
                 learning_rate: float = 1.5e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 max_epochs: int = 12,
                 patience: int = 4,
                 pos_class_weight: float = 2.0,
                 random_state: int = 42):
        self._params = dict(
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            max_epochs=int(max_epochs),
            patience=int(patience),
            pos_class_weight=float(pos_class_weight),
            random_state=int(random_state),
        )
        self.net = None
        self._best_epoch = None
        self._n_features = None
        self._seq_len = None

    def _build_net(self, n_features):
        import torch
        import torch.nn as nn
        p = self._params

        class SeqGRU(nn.Module):
            def __init__(self, F, H, dp):
                super().__init__()
                self.gru = nn.GRU(input_size=F, hidden_size=H,
                                  num_layers=1, batch_first=True)
                self.norm = nn.LayerNorm(H)
                self.drop = nn.Dropout(dp)
                self.head = nn.Sequential(
                    nn.Linear(H, max(8, H // 2)), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(max(8, H // 2), 1),
                )

            def forward(self, x):
                # x: (B, T, F)
                _, h_T = self.gru(x)            # h_T: (1, B, H)
                z = self.norm(h_T.squeeze(0))   # (B, H)
                z = self.drop(z)
                return self.head(z).squeeze(-1)

        return SeqGRU(n_features, p['hidden_dim'], p['dropout'])

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if X_train.ndim != 3:
            raise ValueError(
                f'torch_seq_gru expects 3D sequence input (N, T, F); got '
                f'shape {X_train.shape}. The gate must thread X_seq, not '
                f'aggregated X_tab.')
        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.manual_seed(p['random_state'])
        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        Xt = np.clip(Xt, -8.0, 8.0)
        Xv = np.clip(Xv, -8.0, 8.0)
        yt = np.asarray(y_train, dtype=np.float32)
        yv = np.asarray(y_val, dtype=np.float32)
        self._n_features = Xt.shape[-1]
        self._seq_len = Xt.shape[1]
        self.net = self._build_net(self._n_features)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([p['pos_class_weight']], device=device))
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=p['learning_rate'],
                                weight_decay=p['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=p['max_epochs'])

        ds = TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
        dl = DataLoader(ds, batch_size=p['batch_size'], shuffle=True,
                        drop_last=False)
        Xv_t = torch.from_numpy(Xv).to(device)

        best_auc = -1.0
        best_state = None
        bad = 0
        for epoch in range(p['max_epochs']):
            self.net.train()
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = self.net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            sched.step()
            self.net.eval()
            with torch.no_grad():
                vl = torch.sigmoid(self.net(Xv_t)).cpu().numpy()
            auc = _auc_safe(yv, vl)
            if verbose:
                print(f'  ep{epoch:02d} val_auc={auc:.4f}')
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().clone() for k, v in
                              self.net.state_dict().items()}
                self._best_epoch = epoch
                bad = 0
            else:
                bad += 1
                if bad >= p['patience']:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        if self.net is None:
            raise RuntimeError('Model not fit')
        if X.ndim != 3:
            raise ValueError(
                f'torch_seq_gru.predict_proba expects 3D input (N, T, F); '
                f'got shape {X.shape}.')
        device = next(self.net.parameters()).device
        X = np.asarray(X, dtype=np.float32)
        X = np.clip(X, -8.0, 8.0)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X).to(device))
            return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    @property
    def best_iteration(self):
        return self._best_epoch

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        import torch
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pt')
        meta_path = os.path.join(output_dir, 'metadata.json')
        torch.save({'state_dict': self.net.state_dict(),
                    'n_features': self._n_features,
                    'seq_len': self._seq_len}, model_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_epoch': self._best_epoch,
            'n_features': self._n_features,
            'seq_len': self._seq_len,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import torch
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'model.pt')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        inst._seq_len = meta.get('seq_len')
        inst.net = inst._build_net(inst._n_features)
        state = torch.load(model_path, weights_only=True)
        inst.net.load_state_dict(state['state_dict'])
        inst._best_epoch = meta.get('best_epoch')
        return inst


# Iter #712 — Transformer encoder over the time dimension, second
# `consumes_sequences=True` trainer (after #711 torch_seq_gru).
#
# Diagnosis from last-20-iter cross-tab: W7 (test 2026-01..05) has 0/20
# passes, avg_wr 30.9%, avg_ann -15%; W4/W5 chronic. Iter #711 torch_seq_gru
# delivered massive ANN signal on W6 (+96%) / W7 (+75%) but ALL failures
# were WR-bound (14-37%), never DD-bound. Regime stats show W7 is actually
# a +15% bull market with 8% vol — failure cannot be blamed on a hostile
# regime, so the GRU's last-hidden-state compression is dropping the
# selectivity signal mid-sequence.
#
# Hypothesis: a Transformer encoder reads the full (B, T, F) sequence via
# self-attention and pools through a learnable [CLS] token. Unlike the GRU
# which sequentially compresses, attention can identify the single timestep
# inside the 20-day window where a regime flip occurred (breadth drop,
# foreign-flow inversion, vol acceleration). Different inductive bias from
# everything in the registry: GBDT/XGB-loss variants (aggregated tabular),
# torch_attentive_mlp (feature-gated MLP, no time axis), torch_seq_gru
# (recurrent compression to last state).
#
# Architecture (small, CPU-friendly, ~3× GRU param count at same d_model):
#   Linear(F → D) per timestep → x_proj  (B, T, D)
#   prepend learnable [CLS] token       → (B, T+1, D)
#   add learnable positional embedding  (1, T+1, D)
#   N-layer nn.TransformerEncoder (H heads, GELU FFN, dropout)
#   read [CLS] embedding                → (B, D)
#   LayerNorm → Linear(D → D/2) → GELU → Dropout → Linear(D/2 → 1)
# Loss: BCEWithLogitsLoss(pos_weight)
# Optim: AdamW + CosineAnnealingLR; early-stop on val ROC-AUC, patience=4.
class TorchSeqTransformerTrainer(BaseTrainer):
    name = 'torch_seq_transformer'
    consumes_sequences = True

    def __init__(self,
                 d_model: int = 48,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.35,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 max_epochs: int = 12,
                 patience: int = 4,
                 pos_class_weight: float = 2.0,
                 random_state: int = 42):
        self._params = dict(
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            max_epochs=int(max_epochs),
            patience=int(patience),
            pos_class_weight=float(pos_class_weight),
            random_state=int(random_state),
        )
        self.net = None
        self._best_epoch = None
        self._n_features = None
        self._seq_len = None

    def _build_net(self, n_features, seq_len):
        import torch
        import torch.nn as nn
        p = self._params
        # d_model must be divisible by nhead; clamp if mis-sampled.
        d = p['d_model']
        h = p['nhead']
        if d % h != 0:
            d = (d // h) * h
            if d <= 0:
                d = h
            p['d_model'] = d

        class SeqTransformer(nn.Module):
            def __init__(self, F, T, D, H, L, FF, dp):
                super().__init__()
                self.proj = nn.Linear(F, D)
                self.cls = nn.Parameter(torch.zeros(1, 1, D))
                nn.init.normal_(self.cls, std=0.02)
                self.pos = nn.Parameter(torch.zeros(1, T + 1, D))
                nn.init.normal_(self.pos, std=0.02)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=D, nhead=H, dim_feedforward=FF,
                    dropout=dp, activation='gelu', batch_first=True,
                    norm_first=True)
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=L)
                self.norm = nn.LayerNorm(D)
                self.drop = nn.Dropout(dp)
                self.head = nn.Sequential(
                    nn.Linear(D, max(8, D // 2)), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(max(8, D // 2), 1),
                )

            def forward(self, x):
                # x: (B, T, F)
                B = x.size(0)
                z = self.proj(x)                                # (B, T, D)
                cls = self.cls.expand(B, -1, -1)                # (B, 1, D)
                z = torch.cat([cls, z], dim=1)                  # (B, T+1, D)
                z = z + self.pos                                # broadcast pos
                z = self.encoder(z)                             # (B, T+1, D)
                cls_out = self.norm(z[:, 0, :])                 # (B, D)
                cls_out = self.drop(cls_out)
                return self.head(cls_out).squeeze(-1)

        return SeqTransformer(
            n_features, seq_len, p['d_model'], p['nhead'],
            p['num_layers'], p['dim_feedforward'], p['dropout'])

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if X_train.ndim != 3:
            raise ValueError(
                f'torch_seq_transformer expects 3D sequence input (N, T, F); '
                f'got shape {X_train.shape}. The gate must thread X_seq.')
        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.manual_seed(p['random_state'])
        Xt = np.asarray(X_train, dtype=np.float32)
        Xv = np.asarray(X_val, dtype=np.float32)
        Xt = np.clip(Xt, -8.0, 8.0)
        Xv = np.clip(Xv, -8.0, 8.0)
        yt = np.asarray(y_train, dtype=np.float32)
        yv = np.asarray(y_val, dtype=np.float32)
        self._n_features = Xt.shape[-1]
        self._seq_len = Xt.shape[1]
        self.net = self._build_net(self._n_features, self._seq_len)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([p['pos_class_weight']], device=device))
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=p['learning_rate'],
                                weight_decay=p['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=p['max_epochs'])

        ds = TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
        dl = DataLoader(ds, batch_size=p['batch_size'], shuffle=True,
                        drop_last=False)
        Xv_t = torch.from_numpy(Xv).to(device)

        best_auc = -1.0
        best_state = None
        bad = 0
        for epoch in range(p['max_epochs']):
            self.net.train()
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = self.net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            sched.step()
            self.net.eval()
            with torch.no_grad():
                vl = torch.sigmoid(self.net(Xv_t)).cpu().numpy()
            auc = _auc_safe(yv, vl)
            if verbose:
                print(f'  ep{epoch:02d} val_auc={auc:.4f}')
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().clone() for k, v in
                              self.net.state_dict().items()}
                self._best_epoch = epoch
                bad = 0
            else:
                bad += 1
                if bad >= p['patience']:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        if self.net is None:
            raise RuntimeError('Model not fit')
        if X.ndim != 3:
            raise ValueError(
                f'torch_seq_transformer.predict_proba expects 3D input '
                f'(N, T, F); got shape {X.shape}.')
        device = next(self.net.parameters()).device
        X = np.asarray(X, dtype=np.float32)
        X = np.clip(X, -8.0, 8.0)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X).to(device))
            return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    @property
    def best_iteration(self):
        return self._best_epoch

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        import torch
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pt')
        meta_path = os.path.join(output_dir, 'metadata.json')
        torch.save({'state_dict': self.net.state_dict(),
                    'n_features': self._n_features,
                    'seq_len': self._seq_len}, model_path)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_epoch': self._best_epoch,
            'n_features': self._n_features,
            'seq_len': self._seq_len,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import torch
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'model.pt')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        inst._seq_len = meta.get('seq_len')
        inst.net = inst._build_net(inst._n_features, inst._seq_len)
        state = torch.load(model_path, weights_only=True)
        inst.net.load_state_dict(state['state_dict'])
        inst._best_epoch = meta.get('best_epoch')
        return inst


# Iter #713 — deep-ensemble GRU with disagreement-based selectivity.
#
# Diagnosis (last 20 iters cross-tab):
#   W7 0/20 passes — strong-bull regime that nobody trades successfully.
#   W1/W4/W5 4-5/20; W3/W6 11-13/20. Even sequence-aware iter #711 (GRU)
#   captured massive ANN on W6/W7 (+96% / +75%) but failed every per-window
#   gate on WR (best 37%, need 40%). Iter #712 (Transformer) hit the same
#   WR ceiling — adding more attention-capacity did not help selectivity.
#
# Hypothesis: the WR ceiling reflects EPISTEMIC uncertainty the GRU cannot
# express. A single sigmoid output gives mean prediction but no confidence;
# the gate's threshold sweep can only re-rank, not abstain. A deep ensemble
# of K=5 independently-initialised GRUs yields a per-sample posterior; the
# disagreement (std across members) is well-known to correlate with
# prediction errors (Lakshminarayanan 2017). Score = mean − λ·std penalises
# high-disagreement samples so the gate's "take top-K per day" picks
# *confident* positives — directly targeting the diagnosed WR-bound failure
# rather than searching for yet another encoder.
#
# Architecture: K independent SeqGRU instances (same hyperparams, seeds
# {seed, seed+1, ..., seed+K-1}). At inference: clip(mean − λ·std, 0, 1).
# Score remains in [0, 1] so the gate's SCORE_THRESHOLDS sweep works
# unchanged; high-disagreement samples drop near zero and are excluded
# even at threshold 0.0 (since top-K is by score-desc).
#
# Cost budget: K=5 × ~25s/window × 7 windows ≈ 14 min train, within the
# 30-min wall. The K=5 default is conservative; train mode can sweep K
# and λ within search_spaces.
class TorchSeqGRUEnsembleTrainer(BaseTrainer):
    name = 'torch_seq_gru_ensemble'
    consumes_sequences = True

    def __init__(self,
                 n_models: int = 5,
                 hidden_dim: int = 64,
                 dropout: float = 0.30,
                 learning_rate: float = 1.5e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 max_epochs: int = 10,
                 patience: int = 3,
                 pos_class_weight: float = 2.0,
                 disagreement_penalty: float = 1.5,
                 random_state: int = 42):
        self._params = dict(
            n_models=int(n_models),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            max_epochs=int(max_epochs),
            patience=int(patience),
            pos_class_weight=float(pos_class_weight),
            disagreement_penalty=float(disagreement_penalty),
            random_state=int(random_state),
        )
        self.nets = []
        self._best_epochs = []
        self._n_features = None
        self._seq_len = None

    def _build_net(self, n_features):
        import torch
        import torch.nn as nn
        p = self._params

        class SeqGRU(nn.Module):
            def __init__(self, F, H, dp):
                super().__init__()
                self.gru = nn.GRU(input_size=F, hidden_size=H,
                                  num_layers=1, batch_first=True)
                self.norm = nn.LayerNorm(H)
                self.drop = nn.Dropout(dp)
                self.head = nn.Sequential(
                    nn.Linear(H, max(8, H // 2)), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(max(8, H // 2), 1),
                )

            def forward(self, x):
                _, h_T = self.gru(x)
                z = self.norm(h_T.squeeze(0))
                z = self.drop(z)
                return self.head(z).squeeze(-1)

        return SeqGRU(n_features, p['hidden_dim'], p['dropout'])

    def _train_one(self, seed, Xt, yt, Xv, yv, device, verbose=False):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        p = self._params
        torch.manual_seed(int(seed))
        net = self._build_net(self._n_features).to(device)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([p['pos_class_weight']], device=device))
        opt = torch.optim.AdamW(net.parameters(),
                                lr=p['learning_rate'],
                                weight_decay=p['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=p['max_epochs'])
        ds = TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
        dl = DataLoader(ds, batch_size=p['batch_size'], shuffle=True,
                        drop_last=False,
                        generator=torch.Generator().manual_seed(int(seed)))
        Xv_t = torch.from_numpy(Xv).to(device)

        best_auc = -1.0
        best_state = None
        best_ep = 0
        bad = 0
        for epoch in range(p['max_epochs']):
            net.train()
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                vl = torch.sigmoid(net(Xv_t)).cpu().numpy()
            auc = _auc_safe(yv, vl)
            if verbose:
                print(f'    seed={seed} ep{epoch:02d} val_auc={auc:.4f}')
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().clone() for k, v in
                              net.state_dict().items()}
                best_ep = epoch
                bad = 0
            else:
                bad += 1
                if bad >= p['patience']:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_ep

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import torch

        if X_train.ndim != 3:
            raise ValueError(
                f'torch_seq_gru_ensemble expects 3D sequence input (N, T, F); '
                f'got shape {X_train.shape}.')
        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        torch.set_num_threads(min(4, torch.get_num_threads()))
        Xt = np.clip(np.asarray(X_train, dtype=np.float32), -8.0, 8.0)
        Xv = np.clip(np.asarray(X_val, dtype=np.float32), -8.0, 8.0)
        yt = np.asarray(y_train, dtype=np.float32)
        yv = np.asarray(y_val, dtype=np.float32)
        self._n_features = Xt.shape[-1]
        self._seq_len = Xt.shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets = []
        self._best_epochs = []
        base = p['random_state']
        for k in range(p['n_models']):
            seed = base + k
            if verbose:
                print(f'  ensemble member {k+1}/{p["n_models"]} (seed={seed})')
            net, best_ep = self._train_one(
                seed, Xt, yt, Xv, yv, device, verbose=verbose)
            self.nets.append(net)
            self._best_epochs.append(int(best_ep))
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        if not self.nets:
            raise RuntimeError('Model not fit')
        if X.ndim != 3:
            raise ValueError(
                f'torch_seq_gru_ensemble.predict_proba expects 3D input '
                f'(N, T, F); got shape {X.shape}.')
        device = next(self.nets[0].parameters()).device
        X = np.clip(np.asarray(X, dtype=np.float32), -8.0, 8.0)
        Xt = torch.from_numpy(X).to(device)
        probs = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                probs.append(
                    torch.sigmoid(net(Xt)).cpu().numpy().astype(np.float64))
        P = np.stack(probs, axis=0)              # (K, N)
        p_mean = P.mean(axis=0)
        p_std = P.std(axis=0)
        lam = float(self._params['disagreement_penalty'])
        score = p_mean - lam * p_std
        return np.clip(score, 0.0, 1.0)

    @property
    def best_iteration(self):
        if not self._best_epochs:
            return None
        return int(round(float(np.mean(self._best_epochs))))

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        import torch
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, 'metadata.json')
        member_paths = []
        for i, net in enumerate(self.nets):
            mp = os.path.join(output_dir, f'model_{i}.pt')
            torch.save({'state_dict': net.state_dict()}, mp)
            member_paths.append(mp)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_epochs': self._best_epochs,
            'n_features': self._n_features,
            'seq_len': self._seq_len,
            'member_paths': member_paths,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'metadata': meta_path, 'members': member_paths}

    @classmethod
    def load(cls, output_dir):
        import torch
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        inst._seq_len = meta.get('seq_len')
        inst.nets = []
        for mp in meta.get('member_paths', []):
            net = inst._build_net(inst._n_features)
            state = torch.load(mp, weights_only=True)
            net.load_state_dict(state['state_dict'])
            inst.nets.append(net)
        inst._best_epochs = meta.get('best_epochs', [])
        return inst


# Iter #713 — selectivity layer via hard epistemic abstention.
#
# Diagnosis (Part A, last 20 iters): W7 (test 2026-01..05) is a +15% bull
# regime with low vol (16%) and positive breadth (54%), yet 0/20 iterations
# pass — failure cannot be blamed on regime hostility. Iter #711 GRU and
# iter #712 Transformer both produced massive ANN signal (+75% and +20% on
# W7) but win-rates 33-37% — i.e. the signal is there, precision is not.
# Iter #712 lessons explicitly call for a *selectivity layer*: deep-ensemble
# disagreement filter or conformal abstention, not another encoder.
#
# Why the existing torch_seq_gru_ensemble is not enough: it does soft
# re-ranking via `score = mean - λ·std` clipped to [0,1]. At threshold=0.0
# the gate still picks top-K per date — re-ranking only changes WHICH high-
# uncertainty trade gets taken, not WHETHER any is. Result: 0/7 pass.
#
# This trainer adds HARD abstention. After fitting K GRU members, we
# compute per-row epistemic uncertainty (std across members) on the inner-
# val set and store the q-th quantile as τ. At predict time, any row with
# p_std > τ has its score forced to -1e9, which is below every gate
# threshold including 0.0 — so the gate's `score < threshold: continue`
# branch literally skips that signal. The next-best (lower-uncertainty)
# symbol on the same date can fill the slot, or the slot stays empty if
# every same-day candidate is uncertain. This is a *day-aware* filter
# even though it operates on (date, symbol) rows: hostile days where all
# K members disagree on every symbol will drop out entirely.
#
# τ-calibration on inner-val (not test) is critical for honesty — it's the
# same split used for early stopping, so no test peeking.
class TorchSeqGRUAbstainTrainer(BaseTrainer):
    # Iter #1737 structural pivot from exhausted modernnca brief:
    # the trainer's prior per-row epistemic abstention helps within a date but
    # cannot kill an entire hostile day (W5: -12.7% set return, 31% breadth,
    # 1/20 passes across last-20 iter cross-tab). Memory `feedback_day_abstain_anomaly_gated.md`
    # and iter #1499 (xgb_rank_fusion abstain_min_q=0.3) both showed that a
    # DATE-level abstention floor is the lever for breaking the W5 wall.
    # This iteration adds a `date_abstain_q` mechanism: at fit time we record
    # the train-date mean-prob distribution; at predict time, dates whose mean
    # ensemble prob falls below the q-th quantile of train-date-means get ALL
    # their rows hard-zeroed regardless of per-row uncertainty. The hook is
    # `set_predict_context`, which the gate already calls (see return_gate.py
    # L351). Defaults: q=0.30 (skip ~30% of low-confidence days), n_models=3
    # and max_epochs=6 (time budget — 5*10 was borderline for 30-min wall).
    name = 'torch_seq_gru_abstain'
    consumes_sequences = True

    def __init__(self,
                 n_models: int = 3,
                 hidden_dim: int = 64,
                 dropout: float = 0.30,
                 learning_rate: float = 1.5e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 max_epochs: int = 6,
                 patience: int = 3,
                 pos_class_weight: float = 2.0,
                 abstain_quantile: float = 0.50,
                 date_abstain_q: float = 0.30,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            n_models=int(n_models),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            max_epochs=int(max_epochs),
            patience=int(patience),
            pos_class_weight=float(pos_class_weight),
            abstain_quantile=float(abstain_quantile),
            date_abstain_q=float(date_abstain_q),
            random_state=int(random_state),
        )
        self.nets = []
        self._best_epochs = []
        self._n_features = None
        self._seq_len = None
        self._tau_std = None  # calibrated per-row abstention threshold
        self._date_mean_floor = None  # calibrated per-date mean-prob floor
        self._predict_dates = None  # injected by set_predict_context

    def _build_net(self, n_features):
        import torch
        import torch.nn as nn
        p = self._params

        class SeqGRU(nn.Module):
            def __init__(self, F, H, dp):
                super().__init__()
                self.gru = nn.GRU(input_size=F, hidden_size=H,
                                  num_layers=1, batch_first=True)
                self.norm = nn.LayerNorm(H)
                self.drop = nn.Dropout(dp)
                self.head = nn.Sequential(
                    nn.Linear(H, max(8, H // 2)), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(max(8, H // 2), 1),
                )

            def forward(self, x):
                _, h_T = self.gru(x)
                z = self.norm(h_T.squeeze(0))
                z = self.drop(z)
                return self.head(z).squeeze(-1)

        return SeqGRU(n_features, p['hidden_dim'], p['dropout'])

    def _train_one(self, seed, Xt, yt, Xv, yv, device, verbose=False):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        p = self._params
        torch.manual_seed(int(seed))
        net = self._build_net(self._n_features).to(device)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([p['pos_class_weight']], device=device))
        opt = torch.optim.AdamW(net.parameters(),
                                lr=p['learning_rate'],
                                weight_decay=p['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=p['max_epochs'])
        ds = TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
        dl = DataLoader(ds, batch_size=p['batch_size'], shuffle=True,
                        drop_last=False,
                        generator=torch.Generator().manual_seed(int(seed)))
        Xv_t = torch.from_numpy(Xv).to(device)

        best_auc = -1.0
        best_state = None
        best_ep = 0
        bad = 0
        for epoch in range(p['max_epochs']):
            net.train()
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                vl = torch.sigmoid(net(Xv_t)).cpu().numpy()
            auc = _auc_safe(yv, vl)
            if verbose:
                print(f'    seed={seed} ep{epoch:02d} val_auc={auc:.4f}')
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().clone() for k, v in
                              net.state_dict().items()}
                best_ep = epoch
                bad = 0
            else:
                bad += 1
                if bad >= p['patience']:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_ep

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import torch

        if X_train.ndim != 3:
            raise ValueError(
                f'torch_seq_gru_abstain expects 3D sequence input (N, T, F); '
                f'got shape {X_train.shape}.')
        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        torch.set_num_threads(min(4, torch.get_num_threads()))
        Xt = np.clip(np.asarray(X_train, dtype=np.float32), -8.0, 8.0)
        Xv = np.clip(np.asarray(X_val, dtype=np.float32), -8.0, 8.0)
        yt = np.asarray(y_train, dtype=np.float32)
        yv = np.asarray(y_val, dtype=np.float32)
        self._n_features = Xt.shape[-1]
        self._seq_len = Xt.shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets = []
        self._best_epochs = []
        base = p['random_state']
        for k in range(p['n_models']):
            seed = base + k
            if verbose:
                print(f'  abstain ensemble member {k+1}/{p["n_models"]} '
                      f'(seed={seed})')
            net, best_ep = self._train_one(
                seed, Xt, yt, Xv, yv, device, verbose=verbose)
            self.nets.append(net)
            self._best_epochs.append(int(best_ep))

        # Calibrate the abstention threshold τ on inner-val predictions.
        # We collect K member probabilities on the val set, compute per-row
        # std, and pick the q-th quantile so that exactly (1-q) of val rows
        # exceed τ — those are the rows we will abstain on at test time.
        Xv_t = torch.from_numpy(Xv).to(device)
        probs = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                probs.append(
                    torch.sigmoid(net(Xv_t)).cpu().numpy().astype(np.float64))
        P = np.stack(probs, axis=0)  # (K, N_val)
        val_std = P.std(axis=0)
        q = float(np.clip(p['abstain_quantile'], 0.05, 0.95))
        self._tau_std = float(np.quantile(val_std, q))
        if verbose:
            print(f'  abstention τ (val std @ q={q:.2f}) = {self._tau_std:.4f} '
                  f'— {(1.0 - q) * 100:.0f}% of val rows would abstain')

        # Calibrate the per-DATE mean-prob floor on TRAIN-set predictions.
        # The gate's hostile-day failure mode (W5: 1/20 across last 20 iters)
        # is that on deep-bear days every row's confidence is low but ranked
        # the same — the gate still picks the top-K per date. Hard-killing
        # those dates entirely is the only mechanism that survived (#1499
        # rank_fusion abstain_min_q=0.3 broke W5).
        Xt_t = torch.from_numpy(Xt).to(device)
        tr_probs = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                tr_probs.append(
                    torch.sigmoid(net(Xt_t)).cpu().numpy().astype(np.float64))
        P_tr = np.stack(tr_probs, axis=0).mean(axis=0)  # (N_train,)
        if dates_train is not None and len(dates_train) == len(P_tr):
            d_tr = np.asarray(dates_train)
            uniq = np.unique(d_tr)
            if len(uniq) >= 10:
                date_means = np.array(
                    [P_tr[d_tr == d].mean() for d in uniq], dtype=np.float64)
                q_date = float(np.clip(p['date_abstain_q'], 0.0, 0.95))
                if q_date > 0.0:
                    self._date_mean_floor = float(np.quantile(date_means, q_date))
                    if verbose:
                        print(f'  date-abstain floor (train-date-mean @ '
                              f'q={q_date:.2f}) = {self._date_mean_floor:.4f}')
                else:
                    self._date_mean_floor = None
            else:
                self._date_mean_floor = None
        else:
            self._date_mean_floor = None
        return self

    def set_predict_context(self, dates):
        """Receive test-set dates from return_gate before predict_proba so the
        date-level mean-prob abstention can be applied per-date."""
        self._predict_dates = np.asarray(dates) if dates is not None else None

    def predict_proba(self, X) -> np.ndarray:
        import torch
        if not self.nets:
            raise RuntimeError('Model not fit')
        if X.ndim != 3:
            raise ValueError(
                f'torch_seq_gru_abstain.predict_proba expects 3D input '
                f'(N, T, F); got shape {X.shape}.')
        device = next(self.nets[0].parameters()).device
        X = np.clip(np.asarray(X, dtype=np.float32), -8.0, 8.0)
        Xt = torch.from_numpy(X).to(device)
        probs = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                probs.append(
                    torch.sigmoid(net(Xt)).cpu().numpy().astype(np.float64))
        P = np.stack(probs, axis=0)              # (K, N)
        p_mean = P.mean(axis=0)
        p_std = P.std(axis=0)

        score = p_mean.copy()
        if self._tau_std is not None and self._tau_std > 0:
            # Hard abstention: anything above τ gets a sentinel score that is
            # strictly below every entry in scripts/return_gate.SCORE_THRESHOLDS
            # (lowest is 0.0), so the gate's `if score < threshold: continue`
            # skips it for every sweep iteration.
            abstain_mask = p_std > self._tau_std
            score[abstain_mask] = -1e9

        # Date-level abstention: dates whose mean-prob is below the train
        # quantile floor get ALL their rows hard-zeroed. Targets hostile-day
        # collapse (W5 bear regime) — the gate cannot rescue any row from
        # a date the model itself thinks is low-quality.
        if (self._date_mean_floor is not None
                and self._predict_dates is not None
                and len(self._predict_dates) == len(score)):
            d_te = np.asarray(self._predict_dates)
            for d in np.unique(d_te):
                mask = (d_te == d)
                if mask.sum() == 0:
                    continue
                day_mean = float(np.mean(p_mean[mask]))
                if day_mean < self._date_mean_floor:
                    score[mask] = -1e9
        return score

    @property
    def best_iteration(self):
        if not self._best_epochs:
            return None
        return int(round(float(np.mean(self._best_epochs))))

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        import torch
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, 'metadata.json')
        member_paths = []
        for i, net in enumerate(self.nets):
            mp = os.path.join(output_dir, f'model_{i}.pt')
            torch.save({'state_dict': net.state_dict()}, mp)
            member_paths.append(mp)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_epochs': self._best_epochs,
            'n_features': self._n_features,
            'seq_len': self._seq_len,
            'tau_std': self._tau_std,
            'date_mean_floor': self._date_mean_floor,
            'member_paths': member_paths,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'metadata': meta_path, 'members': member_paths}

    @classmethod
    def load(cls, output_dir):
        import torch
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        inst._seq_len = meta.get('seq_len')
        inst._tau_std = meta.get('tau_std')
        inst._date_mean_floor = meta.get('date_mean_floor')
        inst.nets = []
        for mp in meta.get('member_paths', []):
            net = inst._build_net(inst._n_features)
            state = torch.load(mp, weights_only=True)
            net.load_state_dict(state['state_dict'])
            inst.nets.append(net)
        inst._best_epochs = meta.get('best_epochs', [])
        return inst


# Iter #714 — date-level day-quality gate over the #711 single GRU.
#
# Part A diagnosis: W7 (test 2026-01..05, +14.5% mkt, breadth 53.6%) fails
# 0/20 in last-20-iter cross-tab — best regime in the panel, yet zero passes,
# i.e. precision-bound on good days. W5 (test 2025-01..04, -12.7% mkt,
# breadth 31.7%) fails 4/20 — genuine bear regime hostility. W3 (-6.9% mkt
# but 11/20 pass) shows precision and regime are nearly orthogonal: the
# model can find signal in a mild bear if it's not catastrophically wrong on
# the precision side. Single-GRU (#711) had +96% ann on W6 / +75% on W7
# with WR 33-37% — within 3-7 pp of the 40% pass.
#
# Iter #713 lessons explicitly call for option (b): "learned day-quality
# head over daily aggregate features (breadth, vol, foreign-flow) that
# abstains at the date level rather than per-row". Per-row hard abstain
# (#713) hurt because high-EV ↔ high p_std are correlated in this task —
# removing variance removed the right tail. Date-level abstention sidesteps
# that: it doesn't penalize disagreement within a day, it filters whether
# the DAY itself is tradeable based on regime aggregates that the GRU's
# 20-step recurrence may compress away by the last hidden state.
#
# Architecture:
#   Stage 1: single GRU (identical to TorchSeqGRUTrainer) → p_gru
#   Stage 2: XGBoost classifier on per-date features → p_day_quality
#     - Per-date label: mean(pnl on that date across all rows) > 0
#     - Per-date features: take last timestep of every sequence, then
#       deduplicate to one row per date (mean across symbols on that date)
#       so XGB learns date-level patterns, not symbol-level.
#     - Trained on train+val combined dates (small N ~120 dates) with
#       conservative HPs to avoid overfitting.
#   Inference: score = p_gru * (0.5 + 0.5 * p_day_quality)
#     - Soft multiplier in [0.5*p_gru, 1.0*p_gru] preserves p_gru ranking
#       on good days, demotes (not erases) right-tail picks on bad days.
#     - Compatible with the gate's threshold sweep — no API change.
class TorchSeqGRUDayGateTrainer(BaseTrainer):
    name = 'torch_seq_gru_day_gate'
    consumes_sequences = True

    def __init__(self,
                 hidden_dim: int = 64,
                 dropout: float = 0.30,
                 learning_rate: float = 1.5e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 max_epochs: int = 12,
                 patience: int = 4,
                 pos_class_weight: float = 2.0,
                 day_gate_max_depth: int = 3,
                 day_gate_n_estimators: int = 60,
                 day_gate_learning_rate: float = 0.05,
                 day_gate_min_child_weight: float = 5.0,
                 day_gate_blend: float = 0.5,
                 random_state: int = 42):
        self._params = dict(
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            max_epochs=int(max_epochs),
            patience=int(patience),
            pos_class_weight=float(pos_class_weight),
            day_gate_max_depth=int(day_gate_max_depth),
            day_gate_n_estimators=int(day_gate_n_estimators),
            day_gate_learning_rate=float(day_gate_learning_rate),
            day_gate_min_child_weight=float(day_gate_min_child_weight),
            day_gate_blend=float(day_gate_blend),
            random_state=int(random_state),
        )
        self.net = None
        self.day_gate = None
        self._best_epoch = None
        self._n_features = None
        self._seq_len = None

    def _build_net(self, n_features):
        import torch
        import torch.nn as nn
        p = self._params

        class SeqGRU(nn.Module):
            def __init__(self, F, H, dp):
                super().__init__()
                self.gru = nn.GRU(input_size=F, hidden_size=H,
                                  num_layers=1, batch_first=True)
                self.norm = nn.LayerNorm(H)
                self.drop = nn.Dropout(dp)
                self.head = nn.Sequential(
                    nn.Linear(H, max(8, H // 2)), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(max(8, H // 2), 1),
                )

            def forward(self, x):
                _, h_T = self.gru(x)
                z = self.norm(h_T.squeeze(0))
                z = self.drop(z)
                return self.head(z).squeeze(-1)

        return SeqGRU(n_features, p['hidden_dim'], p['dropout'])

    def _fit_gru(self, Xt, yt, Xv, yv, device, verbose=False):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        p = self._params
        torch.manual_seed(p['random_state'])
        self.net = self._build_net(self._n_features).to(device)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([p['pos_class_weight']], device=device))
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=p['learning_rate'],
                                weight_decay=p['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=p['max_epochs'])
        ds = TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
        dl = DataLoader(ds, batch_size=p['batch_size'], shuffle=True,
                        drop_last=False)
        Xv_t = torch.from_numpy(Xv).to(device)

        best_auc = -1.0
        best_state = None
        bad = 0
        for epoch in range(p['max_epochs']):
            self.net.train()
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = self.net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            sched.step()
            self.net.eval()
            with torch.no_grad():
                vl = torch.sigmoid(self.net(Xv_t)).cpu().numpy()
            auc = _auc_safe(yv, vl)
            if verbose:
                print(f'  GRU ep{epoch:02d} val_auc={auc:.4f}')
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().clone() for k, v in
                              self.net.state_dict().items()}
                self._best_epoch = epoch
                bad = 0
            else:
                bad += 1
                if bad >= p['patience']:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _build_day_features(self, X, dates):
        """Aggregate (N, T, F) sequences into one row per unique date.

        Per-row daily features = mean across symbols on that date of the
        LAST timestep of the sequence. Daily-aggregate cols (breadth,
        market_new_highs, foreign_flow_pctrank, ...) are already constant
        within a date so the mean is exact for them; per-symbol cols
        (rsi_xrank, ret_*_xrank, ...) collapse to ~0.5 for xranks but the
        XGB can still pick up the residual market-tilt features (atr_pct
        median, volume_ratio median, up_days_5d median) that vary day-by-day.

        Returns (X_day, date_keys) where X_day is (D, F) and date_keys is
        (D,) parallel.
        """
        last = X[:, -1, :]  # (N, F)
        dates_arr = np.asarray(dates)
        unique_dates = np.sort(np.unique(dates_arr))
        X_day = np.zeros((len(unique_dates), last.shape[1]), dtype=np.float32)
        for i, d in enumerate(unique_dates):
            mask = dates_arr == d
            X_day[i] = last[mask].mean(axis=0)
        return X_day, unique_dates

    def _build_day_labels(self, pnl, dates):
        """Per-date label: mean pnl across all rows on that date > 0.

        This proxies day-quality: a date where random picks are net-positive
        on average is a "tradeable day"; a date where the average row loses
        money (bear day, sell-off, low-quality candidates) is "untradeable".
        """
        pnl_arr = np.asarray(pnl)
        dates_arr = np.asarray(dates)
        unique_dates = np.sort(np.unique(dates_arr))
        y_day = np.zeros(len(unique_dates), dtype=np.int32)
        for i, d in enumerate(unique_dates):
            mask = dates_arr == d
            y_day[i] = int(pnl_arr[mask].mean() > 0.0)
        return y_day, unique_dates

    # Per-date regime/breadth features that should monotonically increase
    # P(good day). Mirrors iter #936's HistGBMonotonicTrainer breakthrough —
    # same prior knowledge applied to the day-gate sub-model so it can't
    # learn "high breadth → bad day" from a bear-heavy train slice. These are
    # the 8 features whose mean across symbols within a date is exact (they
    # are already constant within the date) — per-symbol xranks (collapse to
    # ~0.5 in the aggregate) are intentionally left unconstrained.
    _DAY_GATE_MONOTONIC_INCREASING = (
        'sector_breadth',
        'foreign_net_monthly_pctrank',
        'market_breadth_adv',
        'market_breadth_above_sma20',
        'up_days_5d',
        'market_new_highs',
        'market_sector_aligned',
        'set_ret_5d_zscore_60d',
    )

    def _build_day_gate_monotonic(self, n_features: int) -> Optional[tuple]:
        try:
            from models.feature_eng import CURATED_FEATURES
        except Exception:
            return None
        if n_features != len(CURATED_FEATURES):
            return None
        idx = {n: i for i, n in enumerate(CURATED_FEATURES)}
        cst = [0] * n_features
        for name in self._DAY_GATE_MONOTONIC_INCREASING:
            i = idx.get(name)
            if i is not None:
                cst[i] = 1
        return tuple(cst)

    def _fit_day_gate(self, X_train, X_val, pnl_train, pnl_val,
                      dates_train, dates_val, verbose=False):
        """Train XGBoost classifier on per-date aggregates → P(good day).

        Combines train and val dates (~120 unique dates) since the GRU has
        no second-level model that needs holdout for early stopping; XGB
        uses fixed n_estimators with conservative HPs (depth=3, n=60) to
        avoid overfitting the small N.
        """
        import xgboost as xgb
        if dates_train is None or pnl_train is None:
            # Without dates/pnl we cannot build the day gate; fall back to
            # all-days-equal (p_day=0.5) by leaving day_gate=None.
            return

        # Build per-date training set from train + val combined
        X_all = np.concatenate([X_train, X_val], axis=0)
        pnl_all = np.concatenate([np.asarray(pnl_train), np.asarray(pnl_val)])
        dates_all = np.concatenate([np.asarray(dates_train),
                                    np.asarray(dates_val) if dates_val is not None
                                    else np.array([])])
        X_day, day_keys_x = self._build_day_features(X_all, dates_all)
        y_day, day_keys_y = self._build_day_labels(pnl_all, dates_all)
        assert (day_keys_x == day_keys_y).all()

        # Degenerate guard: if all days are good or all bad, skip the gate.
        if len(set(y_day.tolist())) < 2:
            if verbose:
                print(f'  day-gate: degenerate label distribution '
                      f'(all={y_day.mean():.2f}); skipping')
            return

        p = self._params
        # Class balance: positive rate of "good days" is ~0.4-0.6 typically;
        # use scale_pos_weight = neg/pos to balance.
        n_pos = float(y_day.sum())
        n_neg = float(len(y_day) - y_day.sum())
        spw = max(0.25, min(4.0, n_neg / max(1.0, n_pos)))

        mono_cst = self._build_day_gate_monotonic(X_day.shape[1])

        self.day_gate = xgb.XGBClassifier(
            max_depth=p['day_gate_max_depth'],
            n_estimators=p['day_gate_n_estimators'],
            learning_rate=p['day_gate_learning_rate'],
            min_child_weight=p['day_gate_min_child_weight'],
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.5,
            scale_pos_weight=spw,
            tree_method='hist',
            monotone_constraints=mono_cst,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=p['random_state'],
            n_jobs=1,
            verbosity=0,
        )
        self.day_gate.fit(X_day, y_day)
        if verbose:
            train_pred = self.day_gate.predict_proba(X_day)[:, 1]
            print(f'  day-gate: trained on {len(y_day)} unique dates '
                  f'(pos={n_pos:.0f}/{len(y_day)}, spw={spw:.2f}); '
                  f'mean_pred={train_pred.mean():.3f} '
                  f'range=[{train_pred.min():.3f}, {train_pred.max():.3f}]')

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import torch

        if X_train.ndim != 3:
            raise ValueError(
                f'torch_seq_gru_day_gate expects 3D sequence input (N, T, F); '
                f'got shape {X_train.shape}.')
        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            raise ValueError('Train or val set has only one class — fit aborted')

        p = self._params
        torch.set_num_threads(min(4, torch.get_num_threads()))
        Xt = np.clip(np.asarray(X_train, dtype=np.float32), -8.0, 8.0)
        Xv = np.clip(np.asarray(X_val, dtype=np.float32), -8.0, 8.0)
        yt = np.asarray(y_train, dtype=np.float32)
        yv = np.asarray(y_val, dtype=np.float32)
        self._n_features = Xt.shape[-1]
        self._seq_len = Xt.shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Stage 1: GRU
        self._fit_gru(Xt, yt, Xv, yv, device, verbose=verbose)

        # Stage 2: day-quality XGB on per-date aggregates
        self._fit_day_gate(Xt, Xv, pnl_train, pnl_val,
                            dates_train, dates_val, verbose=verbose)
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        if self.net is None:
            raise RuntimeError('Model not fit')
        if X.ndim != 3:
            raise ValueError(
                f'torch_seq_gru_day_gate.predict_proba expects 3D input '
                f'(N, T, F); got shape {X.shape}.')
        device = next(self.net.parameters()).device
        X = np.clip(np.asarray(X, dtype=np.float32), -8.0, 8.0)
        self.net.eval()
        # Chunked inference to survive shared-GPU contention (other processes
        # may be holding most VRAM). Falls back to CPU automatically if a
        # CUDA OOM is hit mid-stream.
        chunk = 4096
        preds = []
        with torch.no_grad():
            for s in range(0, X.shape[0], chunk):
                xb = torch.from_numpy(X[s:s+chunk]).to(device)
                try:
                    lb = self.net(xb)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self.net.to('cpu')
                    device = torch.device('cpu')
                    xb = xb.to('cpu')
                    lb = self.net(xb)
                preds.append(torch.sigmoid(lb).cpu().numpy().astype(np.float64))
            p_gru = np.concatenate(preds, axis=0)

        if self.day_gate is None:
            # No day gate (degenerate fit); return raw GRU prob.
            return p_gru

        # Day gate inference: predict on per-row last-timestep features.
        # Same-date rows naturally get similar p_day because the daily-
        # aggregate columns (breadth, market_new_highs, foreign_flow_pctrank)
        # are constant within a date — XGB on those dominates the score.
        last = X[:, -1, :].astype(np.float32)
        p_day = self.day_gate.predict_proba(last)[:, 1].astype(np.float64)

        # Soft multiplier in [blend*p_gru, p_gru] (default blend=0.5).
        # Preserves p_gru ranking on good days, demotes on bad days without
        # erasing the right tail (unlike #713's hard abstain).
        blend = float(self._params['day_gate_blend'])
        return p_gru * (blend + (1.0 - blend) * p_day)

    @property
    def best_iteration(self):
        return self._best_epoch

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        import torch
        import pickle
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pt')
        meta_path = os.path.join(output_dir, 'metadata.json')
        gate_path = os.path.join(output_dir, 'day_gate.pkl')
        torch.save({'state_dict': self.net.state_dict(),
                    'n_features': self._n_features,
                    'seq_len': self._seq_len}, model_path)
        if self.day_gate is not None:
            with open(gate_path, 'wb') as f:
                pickle.dump(self.day_gate, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_epoch': self._best_epoch,
            'n_features': self._n_features,
            'seq_len': self._seq_len,
            'has_day_gate': self.day_gate is not None,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path,
                'day_gate': gate_path if self.day_gate is not None else None}

    @classmethod
    def load(cls, output_dir):
        import torch
        import pickle
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'model.pt')
        gate_path = os.path.join(output_dir, 'day_gate.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        inst._seq_len = meta.get('seq_len')
        inst.net = inst._build_net(inst._n_features)
        state = torch.load(model_path, weights_only=True)
        inst.net.load_state_dict(state['state_dict'])
        inst._best_epoch = meta.get('best_epoch')
        if meta.get('has_day_gate') and os.path.exists(gate_path):
            with open(gate_path, 'rb') as f:
                inst.day_gate = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# Extremely Randomized Trees — sklearn ExtraTreesClassifier
#
# Why this trainer / what gap it fills:
#   Diagnosis over 300+ post-fix iters shows every XGBoost-family variant
#   (xgb_rank_fusion, xgb_quarterly_dro, xgb_meta_label, xgb_temporal_mixup,
#   xgb_recency_consensus, xgb_adv_val, …) plateaus at 5-6/7 windows. W5
#   (Jan-Apr 2025 SET -17.4%) and W7 (Jan-May 2026 SET +18.3% with foreign-
#   flow divergence) are the systematic killers — both involve a train→test
#   regime flip the XGB greedy splitter latches onto training artefacts that
#   do NOT generalise.
#
#   ExtraTrees uses a fundamentally different inductive bias: at each split
#   it samples a RANDOM threshold (not the best greedy threshold) on a
#   random subset of features, and averages predictions across many fully
#   grown trees. The combination of randomised splits + bagging is well
#   known (Geurts et al. 2006) to be less prone to overfit specific
#   training patterns than greedy GBDT, at the cost of slightly higher bias.
#   For a regime-shifted out-of-sample window the bias-variance trade is
#   often favourable.
#
#   This is the novel-inductive-bias slot §6 calls out — pure XGBoost-loss
#   sweeps cannot reach it, only a structural change can.
# --------------------------------------------------------------------- #
class SklearnExtraTreesTrainer(BaseTrainer):
    name = 'sklearn_extra_trees'

    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 20,
                 min_samples_split: int = 10,
                 max_features: str = 'sqrt',
                 bootstrap: bool = False,
                 pos_class_weight: float = 2.0,
                 random_state: int = 42):
        self._params = dict(
            n_estimators=int(n_estimators),
            max_depth=None if max_depth in (None, 0, -1) else int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            min_samples_split=int(min_samples_split),
            max_features=str(max_features),
            bootstrap=bool(bootstrap),
            pos_class_weight=float(pos_class_weight),
            random_state=int(random_state),
        )
        self.clf = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.ensemble import ExtraTreesClassifier

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        # ExtraTrees has no early stopping; fit val into the same tree pool
        # via concatenation so all available labelled data informs splits.
        # The inner val rows are still kept distinct in evaluate_window for
        # the gate's threshold sweep, which is the real selection mechanism.
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self._n_features = X_full.shape[1]

        self.clf = ExtraTreesClassifier(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            min_samples_leaf=p['min_samples_leaf'],
            min_samples_split=p['min_samples_split'],
            max_features=p['max_features'],
            bootstrap=p['bootstrap'],
            class_weight={0: 1.0, 1: p['pos_class_weight']},
            random_state=p['random_state'],
            n_jobs=-1,
        )
        self.clf.fit(X_full, y_full)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict_proba(X)[:, 1]

    def feature_importance(self):
        if self.clf is None:
            return None
        return self.clf.feature_importances_

    @property
    def best_iteration(self):
        return self._params['n_estimators']

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.clf, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'model.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        with open(model_path, 'rb') as f:
            inst.clf = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# Logistic regression with elastic net + polynomial interaction features.
#
# Iter #750 hypothesis (diagnostic-driven, see claude_iter #750 report):
#   All 37 prior trainers are tree-based (XGB/LGBM/ExtraTrees, axis-aligned
#   step-function splits) or NN-based (GRU/Transformer/MLP, smooth via
#   ReLU/sigmoid stacks). Part-A diagnosis shows W1 (bull +6.3%, only 6 mo
#   train, 0/20 pass) and W5 (hostile bear -12.7%, breadth 0.10, 1/20 pass)
#   are the binding regimes. Both failure modes are classic high-variance
#   estimator problems: trees overfit small-sample regimes (W1) and produce
#   wild axis-aligned scores on out-of-distribution rows (W5).
#
# A linear classifier with degree-2 interaction features is a genuinely
# different inductive bias: convex MLE objective (no local minima), smooth
# logistic decision surface, provably calibrated P(y=1|x) under MLE, and a
# strong inductive bias (linearity in φ(x)) that lowers variance for small
# samples. Elastic net (L1+L2) handles correlated curated features and
# auto-sparsifies; pairwise interactions let the linear surface model
# "regime × signal" interactions that single-feature linear cannot.
# --------------------------------------------------------------------- #
class LogisticElasticNetTrainer(BaseTrainer):
    name = 'logistic_elastic_net'

    def __init__(self,
                 C: float = 0.1,
                 l1_ratio: float = 0.5,
                 pos_class_weight: float = 2.0,
                 # degree=1 default: pure linear in 96 aggregate features —
                 # SAGA fit ~5-15s/window vs ~2-4min/window at degree=2
                 # (4656 polynomial features). Train mode can sweep degree=2
                 # over the next hours if degree=1 shows promise.
                 degree: int = 1,
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            C=float(C),
            l1_ratio=float(l1_ratio),
            pos_class_weight=float(pos_class_weight),
            degree=int(degree),
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=int(random_state),
        )
        self.pipe = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        # No inner-val early stopping for LR — concatenate to maximize the
        # MLE sample. Threshold sweep in the gate handles selection.
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self._n_features = X_full.shape[1]

        steps = []
        if p['degree'] > 1:
            # interaction_only=True: drop x_i^2 terms (RobustScaler-centered
            # inputs already capture magnitude via |x|; squares are redundant
            # and double the parameter count).
            steps.append(('poly', PolynomialFeatures(
                degree=p['degree'], interaction_only=True, include_bias=False,
            )))
        # Re-standardize after PolynomialFeatures so the L1 penalty applies
        # uniformly across feature scales (products of [0,1] features stay
        # small but products of std/slope aggregates can blow up).
        steps.append(('scaler', StandardScaler(with_mean=True, with_std=True)))
        # sklearn 1.8 deprecated explicit penalty='elasticnet' / n_jobs in favor
        # of inferring penalty from l1_ratio alone; pass C + l1_ratio only.
        steps.append(('clf', LogisticRegression(
            C=p['C'],
            l1_ratio=p['l1_ratio'],
            solver='saga',
            class_weight={0: 1.0, 1: p['pos_class_weight']},
            max_iter=p['max_iter'],
            tol=p['tol'],
            random_state=p['random_state'],
            verbose=1 if verbose else 0,
        )))
        self.pipe = Pipeline(steps)
        self.pipe.fit(X_full, y_full)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError('Model not fit')
        return self.pipe.predict_proba(X)[:, 1]

    def feature_importance(self):
        # Polynomial expansion makes per-input importance lossy; skip.
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pipe.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipe, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'pipe.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        with open(model_path, 'rb') as f:
            inst.pipe = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# KNN classifier (sklearn) — non-parametric, memory-based, locally adaptive.
#
# Motivation (claude iter #756):
# Per-window regime diagnosis shows W1 is the smallest train slice (~8k rows
# vs 20–40k for W2..W7) AND fails 0/20 across the last 20 iterations
# regardless of trainer family. W3 (bear -6.6%, breadth 0.378) and W4
# (mildly bear -1.6%) also fail 0/20. Every prior trainer in the registry is
# either a tree (axis-aligned splits, high variance on small samples / OOD
# rows) or a parametric NN (likewise needs many samples) or a global linear
# (low variance but cannot model local nonlinearity — caps WR at ~35%).
#
# KNN occupies a strictly different bias slot:
#   * Non-parametric — no global parameters, can't overfit at the model
#     level. Variance is bounded by 1/sqrt(K).
#   * Locally adaptive — each test row's score uses only its K nearest
#     training rows in feature space. Regime drift is handled implicitly:
#     if a test row is far from any training row, its K-nearest are still
#     the "best available" historical analogs.
#   * Cheap to fit (essentially memorization), bounded predict cost via
#     ball_tree (~O(d log n) per query).
#
# Design choices:
#   * StandardScaler on top of the gate's RobustScaler. KNN distance is
#     scale-sensitive; we want each feature on roughly the same scale.
#   * Optional PCA reduction (default 0=off; HP sweep can enable). 96-d
#     curated aggregates have heavy redundancy (last/mean/std/dev of the
#     same 24 features); PCA collapses correlated axes and mitigates the
#     curse of dimensionality.
#   * weights='distance' so nearby rows count more — sharpens the
#     probability estimate vs uniform K-vote.
#   * metric='manhattan' (L1) default. With heavy-tailed atr_pct /
#     volume_ratio aggregates, L1 is less dominated by single-feature
#     outliers than L2.
# --------------------------------------------------------------------- #
class KNNClassifierTrainer(BaseTrainer):
    name = 'knn_classifier'

    def __init__(self,
                 n_neighbors: int = 100,
                 weights: str = 'distance',
                 metric: str = 'manhattan',
                 leaf_size: int = 30,
                 pca_components: int = 0,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            n_neighbors=int(n_neighbors),
            weights=str(weights),
            metric=str(metric),
            leaf_size=int(leaf_size),
            pca_components=int(pca_components),
            random_state=int(random_state),
        )
        self.pipe = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        # KNN has no inner-val early stopping concept — concatenate train+val
        # to maximize the support set, mirroring logistic_elastic_net.
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self._n_features = X_full.shape[1]

        steps = [('scaler', StandardScaler(with_mean=True, with_std=True))]
        if p['pca_components'] > 0:
            n_comp = min(p['pca_components'], X_full.shape[1], X_full.shape[0])
            steps.append(('pca', PCA(
                n_components=n_comp, random_state=p['random_state'])))
        # ball_tree works with manhattan / euclidean / minkowski; let sklearn
        # auto-select based on metric. n_jobs=1 — within-window fit is fast
        # enough and parallel KNN queries blow up RSS on the 96-d feature set.
        steps.append(('clf', KNeighborsClassifier(
            n_neighbors=p['n_neighbors'],
            weights=p['weights'],
            metric=p['metric'],
            algorithm='auto',
            leaf_size=p['leaf_size'],
            n_jobs=1,
        )))
        self.pipe = Pipeline(steps)
        self.pipe.fit(X_full, y_full)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError('Model not fit')
        return self.pipe.predict_proba(X)[:, 1]

    def feature_importance(self):
        # KNN has no per-feature importance.
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pipe.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipe, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'pipe.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        with open(model_path, 'rb') as f:
            inst.pipe = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# QuadraticDiscriminantAnalysis (iter #772) — generative probabilistic
# classifier. Diagnosis pointed at W4 (0/20 pass rate over last 20 iters,
# avg_ann -39%, avg_wr 19%). In the iter #770 knn report, W4 had only
# 5 trades at thr=0.5 (60% WR, +39% ann, pass on WR/DD) and 21 trades at
# thr=0.3 (19% WR, fail on WR & DD) — no "sweet spot" of 10-20 trades at
# 40%+ WR exists for KNN's coarse, distance-weighted probabilities.
#
# QDA fits class-conditional Gaussians P(X|Y=k) and applies Bayes' rule
# for P(Y|X). This produces:
#   * Strictly continuous posterior probabilities (no quantization like
#     KNN's k+1 levels), giving the threshold sweep finer granularity.
#   * Probabilities that are correctly calibrated *if* features within a
#     class are approximately Gaussian — under that assumption no extra
#     Platt/isotonic step is needed. With the curated features being a mix
#     of bounded xranks ([0,1]) and continuous magnitudes, the assumption
#     holds well enough for the posterior ranking to be useful even when
#     the absolute calibration drifts.
#   * Quadratic decision surface in feature space (per-class covariance) —
#     genuinely different inductive bias from anything currently in the
#     registry: trees partition recursively, linear/elastic-net is hyperplane,
#     KNN is local distance, GRU is sequential. QDA is the first generative
#     model in the panel.
#
# Design choices:
#   * StandardScaler before QDA. Class-conditional Gaussian fits are scale-
#     sensitive (covariance matrix conditioning); same convention as KNN.
#   * Optional PCA reduction (default 0=off; HP sweep can enable). The 24-d
#     CURATED feature aggregate becomes 96-d after the gate's last/mean/
#     std/dev expansion, which puts QDA's per-class covariance estimation
#     into a ~96×96 / 2 ≈ 4.6k-parameter regime that is still well-
#     conditioned on ~30k-row windows, but PCA can sharpen the signal by
#     collapsing redundant temporal aggregates.
#   * reg_param ∈ [0,1] = shrinkage toward spherical covariance. With non-
#     perfectly-Gaussian features and ~30k rows per window, mild
#     regularization (~0.1) stabilizes the per-class covariance estimate.
#   * priors=None (sklearn default = empirical class proportions). The
#     ~25% positive rate is reflected in the posterior.
#   * tol=1e-4 default; controls rank estimation in covariance inversion.
# --------------------------------------------------------------------- #
class QDAClassifierTrainer(BaseTrainer):
    name = 'qda_classifier'

    def __init__(self,
                 reg_param: float = 0.6,
                 tol: float = 1e-4,
                 pca_components: int = 16,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            reg_param=float(reg_param),
            tol=float(tol),
            pca_components=int(pca_components),
            random_state=int(random_state),
        )
        self.pipe = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        # No early stopping for closed-form generative fit — use full
        # train+val to maximize support, mirroring KNN / logistic_elastic_net.
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self._n_features = X_full.shape[1]

        steps = [('scaler', StandardScaler(with_mean=True, with_std=True))]
        if p['pca_components'] > 0:
            n_comp = min(p['pca_components'], X_full.shape[1], X_full.shape[0])
            steps.append(('pca', PCA(
                n_components=n_comp, random_state=p['random_state'])))
        steps.append(('clf', QuadraticDiscriminantAnalysis(
            reg_param=p['reg_param'],
            tol=p['tol'],
            store_covariance=False,
        )))
        self.pipe = Pipeline(steps)
        self.pipe.fit(X_full, y_full)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError('Model not fit')
        return self.pipe.predict_proba(X)[:, 1]

    def feature_importance(self):
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pipe.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipe, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'pipe.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        with open(model_path, 'rb') as f:
            inst.pipe = pickle.load(f)
        return inst


class GaussianNaiveBayesClassifierTrainer(BaseTrainer):
    """Gaussian Naive Bayes — generative classifier with feature-independence prior.

    Iter #788: pivot from qda_classifier (16 sweeps, flat threshold sweep —
    saturated probabilities make SCORE_THRESHOLDS useless, scores collapsed
    to {≈0, ≈1} for QDA's full-covariance discriminant). QDA's W3/W4/W5
    over-trading (78-93 trades, 12-17% WR, 40-60% DD) is structural to
    full-covariance modelling on a 96-d aggregate where bear-regime test
    rows fall inside the trained "win" class manifold and threshold gating
    cannot filter them out.

    GaussianNB factorizes class-conditional likelihood per feature
    (independence assumption). Combined with upstream PCA decorrelation,
    the independence assumption holds by construction on the orthogonalized
    components, and the resulting posterior is a product of Gaussian CDFs
    that varies SMOOTHLY across [0,1] — restoring threshold-sweep
    sensitivity that QDA loses to its bimodal score distribution.

    Distinct inductive bias from all 40 prior trainers in the registry:
    neither trees (XGB / LightGBM / sklearn_extra_trees), nor neural nets
    (torch_*), nor distance (KNN), nor full-covariance generative (QDA),
    nor discriminative linear (logistic_elastic_net) factorize the
    likelihood per feature the way Gaussian NB does. The factorization
    means rows with macro features inconsistent with the trained "win"
    Gaussians (low breadth, negative SET-zscore, bear foreign flow) get
    multiplicatively suppressed posteriors — the right inductive bias
    for the W3/W4/W5 bear-regime bottleneck identified in Part A.3.
    """
    name = 'gaussian_nb'

    def __init__(self,
                 var_smoothing: float = 1e-9,
                 pca_components: int = 16,
                 prior_class1: float = 0.0,  # 0 = estimate from data
                 random_state: int = 42,
                 **_):
        self._params = dict(
            var_smoothing=float(var_smoothing),
            pca_components=int(pca_components),
            prior_class1=float(prior_class1),
            random_state=int(random_state),
        )
        self.pipe = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.decomposition import PCA
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self._n_features = X_full.shape[1]

        priors = None
        if 0.0 < p['prior_class1'] < 1.0:
            priors = [1.0 - p['prior_class1'], p['prior_class1']]

        steps = [('scaler', StandardScaler(with_mean=True, with_std=True))]
        if p['pca_components'] > 0:
            n_comp = min(p['pca_components'], X_full.shape[1], X_full.shape[0])
            steps.append(('pca', PCA(
                n_components=n_comp, random_state=p['random_state'])))
        steps.append(('clf', GaussianNB(
            priors=priors,
            var_smoothing=p['var_smoothing'],
        )))
        self.pipe = Pipeline(steps)
        self.pipe.fit(X_full, y_full)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError('Model not fit')
        return self.pipe.predict_proba(X)[:, 1]

    def feature_importance(self):
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pipe.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipe, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'pipe.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        with open(model_path, 'rb') as f:
            inst.pipe = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# XGBoost Isotonic-Calibrated Classifier (iter #804)
#
# Diagnosis (Part A — this iteration):
#   * Per-window failure pattern across last 20 iters: W3 (-35% avg_ann,
#     20% WR, 76 trades), W4 (-72%, 17% WR, 69 trades), W5 (-77%, 15% WR,
#     76 trades). All three are bear regimes (W3 SET -6.9%, W4 -1.6%,
#     W5 -12.4%, breadth 32-43%, FF -23k to -59k monthly).
#   * gaussian_nb (last 16 sweeps) and qda_classifier (16 prior sweeps)
#     both fail with SATURATED posteriors: thr=0.0/0.3/0.5/0.7 all yield
#     IDENTICAL trade counts in the same window. The score CDF is bimodal
#     near {0, 1} so SCORE_THRESHOLDS becomes a no-op, leaving the trader
#     to take top-K per date even when ALL same-day scores are spurious
#     (bear-regime stop-outs in W3/W4/W5).
#   * Trade counts of 56-77 per window vs gate floor of 20 confirm the
#     model is far from over-selective — the WR/DD bar is missed because
#     too many marginal predictions slip through, not because the model
#     is too cautious.
#
# Hypothesis: isotonic post-hoc calibration via CV gives XGB an
# empirically-monotonic score → win-rate map. After calibration, thr=0.55
# means "training WR ≥ 55%" by construction, so the gate's threshold
# sweep BITES in bear windows where the calibrated score quietly drops
# (less of training was at high empirical WR for the bear-similar feature
# regions). Trade count should fall from ~70/win to ~20-30/win in
# bear windows, lifting WR above the 40% gate floor.
#
# Distinct inductive bias from registry:
#   * vs xgboost: raw logits → sigmoid; uncalibrated, often overconfident
#     under class imbalance (scale_pos_weight stretches scores toward {0,1})
#   * vs xgb_meta_label: meta-labeling predicts whether to ACCEPT another
#     model's signal; this is a single-stage calibrated classifier
#   * vs xgb_strict_win: hard-label retraining on stricter class; this
#     keeps the v1 label and corrects only the score function
#   * vs xgb_focal_loss: re-weighted loss with same uncalibrated output
#   * vs gaussian_nb / qda_classifier: generative posteriors that saturate;
#     this is discriminative + post-hoc isotonic (no parametric assumption
#     about p(x|y))
#
# Implementation note: CalibratedClassifierCV with method='isotonic' and
# cv=3 fits 3 inner XGB models on different folds of the train data, uses
# each model's out-of-fold predictions to fit an isotonic regressor, then
# averages the 3 calibrated predictions at inference. The inner XGB does
# NOT get early stopping (sklearn's CV interface doesn't pass an eval_set)
# but the n_estimators is held modest (300) and reg_lambda elevated to
# prevent the smaller-fold over-fit that early stopping would otherwise
# catch. Time budget per window: ~6-8s (3× the base XGB fit).
# --------------------------------------------------------------------- #
class XGBoostIsotonicCalibratedTrainer(BaseTrainer):
    """XGB classifier wrapped in CV-based isotonic calibration."""

    name = 'xgb_iso_calibrated'

    def __init__(self,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 n_estimators: int = 300,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 2.0,
                 min_child_weight: float = 10.0,
                 gamma: float = 0.1,
                 calib_cv: int = 3,
                 calib_method: str = 'isotonic',
                 pos_class_weight: float = 0.0,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
            min_child_weight=float(min_child_weight),
            gamma=float(gamma),
            calib_cv=int(calib_cv),
            calib_method=str(calib_method),
            pos_class_weight=float(pos_class_weight),
            random_state=int(random_state),
        )
        self.clf = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import xgboost as xgb
        from sklearn.calibration import CalibratedClassifierCV

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        # Use full train+val data for calibration fitting (more rows → better
        # isotonic regression). The CV inside CalibratedClassifierCV provides
        # the train/calib separation needed to avoid the in-bag calibration
        # bias that would arise from fitting on the same rows used for XGB.
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self._n_features = X_full.shape[1]

        if p['pos_class_weight'] > 0:
            spw = p['pos_class_weight']
        else:
            pos_rate = float(np.mean(y_full))
            spw = float(min((1.0 - pos_rate) / max(pos_rate, 1e-6), 15.0))

        base_xgb = xgb.XGBClassifier(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            scale_pos_weight=spw,
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            min_child_weight=p['min_child_weight'],
            gamma=p['gamma'],
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=p['random_state'],
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
        )
        self.clf = CalibratedClassifierCV(
            base_xgb,
            method=p['calib_method'],
            cv=p['calib_cv'],
            n_jobs=1,
        )
        self.clf.fit(X_full, y_full)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict_proba(X)[:, 1]

    def feature_importance(self):
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'calibrated.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.clf, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'calibrated.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        with open(model_path, 'rb') as f:
            inst.clf = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# Kernel Logistic Regression via Nyström RBF approximation (iter #819)
#
# Diagnosis (Part A — this iteration):
#   * Per-window failure across last 20 iters (#799–818, all gaussian_nb):
#       W1 wr=0.31 ann=+146% dd=0.20 (close — fails on WR floor 0.40)
#       W2 wr=0.28 ann=+27%  dd=0.26 (fails on WR + DD)
#       W3 wr=0.21 ann=-23%  dd=0.31 (bear — uniformly hostile)
#       W4 wr=0.17 ann=-76%  dd=0.43 (mild SET decline but high DD)
#       W5 wr=0.14 ann=-79%  dd=0.45 (severe bear, breadth 32%)
#       W6 wr=0.31 ann=+143% dd=0.20 (close)
#       W7 wr=0.25 ann=-16%  dd=0.23 (close-to-passing)
#     gaussian_nb is fully saturated (31 iters, 0 pass).
#   * Regime stats: W3 (-7% SET, breadth 38%), W4 (-2% SET, breadth 43%),
#     W5 (-13% SET, breadth 32%) are the bear-regime test windows; trainers
#     fail uniformly here regardless of family. W4 is mild macro but still
#     a 43% DD trainwreck — model picks wrong stocks, not too many.
#   * Baseline reference: random_topk = -30% avg_ann (1/7 pass via W5
#     happenstance); rule_only = -28% (0/7). The model's job is to beat
#     these by picking BETTER per-date trades, not by trading less.
#
# Hypothesis: kernel methods are the largest under-represented inductive-
# bias slot in the 41-trainer registry. Existing families are all axis-
# aligned trees (XGB/LightGBM/ExtraTrees), distance (KNN), linear
# (logistic_elastic_net), full-covariance generative (QDA),
# factorized generative (GaussianNB), sequence-recurrent (torch_seq_*),
# or attention-MLP (torch_attentive_mlp). NONE map features into an RBF
# kernel space and learn a max-margin / max-likelihood boundary there.
#
# RBF kernel logistic regression captures *local* conjunctions:
#   "high volume_ratio AND positive sector_breadth AND atr_pct in [0.02,0.05]"
# becomes a single bump in kernel space, vs trees fragmenting it into
# many leaves. In bear regimes (W3/W5) where the win-class manifold is
# sparser, kernel methods can model the thin winning region without the
# spurious axis-aligned splits that trees fall into when training on a
# bull-heavy mix.
#
# Implementation:
#   * Nyström RBF approximation with n_components=300: O(N*m^2) vs O(N^2)
#     for exact kernel SVM. 300 components are enough to span the local
#     structure of the 96-d aggregate feature space without blowing wall
#     time. Random landmark sampling.
#   * Logistic regression (sklearn LogisticRegression with lbfgs) on the
#     Nyström-transformed features. Gives true probabilities (well
#     calibrated by construction, unlike NB/QDA saturating posteriors).
#   * class_weight='balanced' to compensate for ~22% positive class rate
#     without hard re-sampling.
#   * StandardScaler first (kernel methods are scale-sensitive, RBF
#     gamma is computed in scaled space).
#   * Optional PCA reduction before Nyström (de-correlate the 96-d
#     last/mean/std/dev aggregate; defaults to off for the gate run since
#     PCA loses the absolute-level info that breadth/SET features carry).
#
# Distinct inductive bias vs registry: first kernel-space classifier.
# vs KNN: KNN is local but unsmoothed and uniform-weight; this is local
#         but smoothed (Gaussian kernel) and max-likelihood weighted.
# vs logistic_elastic_net: same final classifier head, but in 300-d
#         nonlinear kernel feature space instead of 96-d raw features.
# vs trees: smooth nonlinear surface vs axis-aligned step function.
# vs QDA/NB: discriminative (no class-conditional Gaussian assumption);
#         calibrated proba (no saturation pathology).
# --------------------------------------------------------------------- #
class KernelLogRegTrainer(BaseTrainer):
    """RBF-kernel logistic regression via Nyström approximation.

    iter-#1300 (claude_mode, brief-exhausted pivot from torch_patchtst):
    optional cross-validated isotonic / sigmoid calibration on top of the
    Nyström+LR pipeline. PatchTST iter #1298/#1299 failed structurally on
    transition windows W2-W4 (WR 14-19%, FP flood) because the patch
    transformer assigned high probabilities to mixed-regime samples that
    turned out to be losers — i.e., it was *over-confident* in unfamiliar
    regimes. Isotonic calibration via CalibratedClassifierCV is the textbook
    fix: it learns a monotone, piecewise-constant remap from raw
    probabilities to empirical positive rates on held-out folds, which
    pulls extreme high-confidence predictions back toward the base rate
    in regions where the raw model was systematically overconfident. This
    directly attacks the transition-window FP problem without touching
    features or labels, and is structurally novel for the registry
    (`kernel_logreg` uses raw LR; `xgb_iso_calibrated` calibrates an XGB,
    not a kernel map; none combine kernel features with isotonic
    calibration).
    """

    name = 'kernel_logreg'

    def __init__(self,
                 n_components: int = 300,
                 gamma: float = 0.0,  # 0 → 1/(n_features * X.var())
                 C: float = 1.0,
                 max_iter: int = 200,
                 pca_components: int = 0,
                 class_weight: str = 'balanced',
                 calibrate: str = 'none',   # 'none' | 'isotonic' | 'sigmoid'
                 calibrate_cv: int = 3,
                 gamma_secondary: float = 0.0,
                 n_components_secondary: int = 0,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            n_components=int(n_components),
            gamma=float(gamma),
            C=float(C),
            max_iter=int(max_iter),
            pca_components=int(pca_components),
            class_weight=str(class_weight),
            calibrate=str(calibrate),
            calibrate_cv=int(calibrate_cv),
            gamma_secondary=float(gamma_secondary),
            n_components_secondary=int(n_components_secondary),
            random_state=int(random_state),
        )
        self.pipe = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.decomposition import PCA
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline, FeatureUnion
        from sklearn.preprocessing import StandardScaler

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self._n_features = X_full.shape[1]

        # Cap n_components by available rows (Nyström subsamples landmarks
        # from X; can't sample more than the row count).
        ncomp = min(p['n_components'], X_full.shape[0])
        gamma = p['gamma'] if p['gamma'] > 0 else 'scale'

        cw = p['class_weight'] if p['class_weight'] != 'none' else None

        steps = [('scaler', StandardScaler(with_mean=True, with_std=True))]
        if p['pca_components'] > 0:
            n_pca = min(p['pca_components'], X_full.shape[1], X_full.shape[0])
            steps.append(('pca', PCA(
                n_components=n_pca, random_state=p['random_state'])))
        # Nyström: 1/(n_features * X.var()) when gamma='scale'. Per sklearn
        # 1.8 API, Nystroem.gamma accepts None to auto-set; we mimic that
        # with explicit 'scale' handling: compute on post-scaler features.
        nys_primary = Nystroem(
            kernel='rbf',
            gamma=None if gamma == 'scale' else gamma,
            n_components=ncomp,
            random_state=p['random_state'],
        )
        # Multi-scale Nyström: when gamma_secondary > 0 AND
        # n_components_secondary > 0, build a second RBF map at a different
        # bandwidth and concatenate via FeatureUnion. The LR head then sees
        # both scales — fine-grained (small gamma → wider RBF) and coarse
        # (large gamma → narrow RBF) similarity. Motivation: kernel_logreg
        # iter #1379 hit 5/7 at single-gamma 0.5 but failed W2 by n=19/20
        # (one trade short — insufficient high-confidence picks in the
        # transition window's mixed regime) and W5 wr=32% (over-confident on
        # deep-bear OOD samples). A second bandwidth shifts the RKHS to
        # mix local and global structure: small-gamma features sharpen
        # discrimination in the bull-train domain (helps W2 get the 20th
        # confident pick) while large-gamma features broaden the
        # "similar-to-training" notion in regime-shifted W5 (pulling
        # over-confident OOD predictions toward the prior).
        if (p['gamma_secondary'] > 0
                and p['n_components_secondary'] > 0):
            ncomp2 = min(p['n_components_secondary'], X_full.shape[0])
            nys_secondary = Nystroem(
                kernel='rbf',
                gamma=p['gamma_secondary'],
                n_components=ncomp2,
                random_state=p['random_state'] + 1,
            )
            steps.append(('nystroem', FeatureUnion([
                ('nys_primary', nys_primary),
                ('nys_secondary', nys_secondary),
            ])))
        else:
            steps.append(('nystroem', nys_primary))

        base_clf = LogisticRegression(
            C=p['C'],
            solver='lbfgs',
            max_iter=p['max_iter'],
            class_weight=cw,
            random_state=p['random_state'],
        )

        if p['calibrate'] in ('isotonic', 'sigmoid'):
            # Wrap the LR head in CalibratedClassifierCV. cv folds re-fit
            # the LR on each train fold and learn the isotonic/sigmoid
            # remap on the held-out fold; final predict_proba averages
            # the cv calibrated estimators. The Nyström features must be
            # computed BEFORE the cv split so all folds share the same
            # landmarks — hence we wrap only the LR head, not the whole
            # pipeline.
            clf = CalibratedClassifierCV(
                base_clf,
                method=p['calibrate'],
                cv=max(2, p['calibrate_cv']),
            )
        else:
            clf = base_clf
        steps.append(('clf', clf))
        self.pipe = Pipeline(steps)
        self.pipe.fit(X_full, y_full)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError('Model not fit')
        return self.pipe.predict_proba(X)[:, 1]

    def feature_importance(self):
        # No per-input-feature importance after the Nyström map.
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pipe.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipe, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'pipe.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        with open(model_path, 'rb') as f:
            inst.pipe = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# GaussianProcessClassifierTrainer (iter #835) — full Bayesian non-parametric
#
# Motivation (Part A diagnosis, iter #835):
#   * Last-50 per-window pass rate: W4 4%, W5 4%, W3 6%, W6 10%, W1 10%, W7 8%,
#     W2 16%. Even the easiest regime (W7: +14.9% SET, breadth 73%, +FF) fails
#     92% of the time. The pattern is uniform-across-trainers, suggesting models
#     mis-rank picks even in benign macro — not a regime-feature gap.
#   * Across 42 trainers and 830 iters, 0 passes. Tree-loss variants, kernel-
#     map MAP regression (kernel_logreg), generative Gaussian (gaussian_nb,
#     qda_classifier), distance (knn_classifier) all saturated. The remaining
#     under-explored slot in §6.B.2: full-Bayesian probabilistic classifier.
#   * kernel_logreg is the closest neighbor in inductive bias: both use RBF
#     features. But kernel_logreg is MAP (point-estimate logistic head on
#     Nyström-approximated kernel features). A Gaussian Process Classifier
#     instead carries a full posterior over the latent function, calibrated
#     via Laplace approximation, and learns the RBF length-scale via marginal
#     likelihood maximization (no manual gamma sweep). In feature regions far
#     from training data (e.g. W5's regime-shifted bear-vol slice that is rare
#     in W4's bull-heavy training), the posterior naturally widens → predicted
#     probabilities pull toward the prior mean (~22% positive rate). The
#     downstream threshold sweep filters those low-confidence picks out, so
#     hostile-regime false positives should drop.
#
# Distinct inductive bias vs registry:
#   vs kernel_logreg:  Bayesian posterior (Laplace) vs MAP logistic; learned
#                      length-scale via marginal-likelihood vs fixed gamma; no
#                      Nyström approximation (exact kernel matrix on subsample).
#   vs qda/gaussian_nb: non-parametric (kernel) vs parametric Gaussian density;
#                       no class-conditional Gaussian assumption.
#   vs trees: smooth function over feature space vs axis-aligned step function.
#   vs knn:    smoothed (kernel) probability vs unsmoothed point distance.
#
# Implementation:
#   * sklearn.gaussian_process.GaussianProcessClassifier with RBF + ConstantKernel.
#   * Laplace approximation (default) for the binary-classification posterior.
#   * Marginal likelihood is exact but O(N^3) in N_train; therefore subsample
#     to ``n_inducing`` rows (default 600), stratified by class to preserve the
#     ~22% positive rate, and within-class stratified by date order (uniform
#     across the train window) to retain temporal coverage. This makes one
#     fit cost ~600^3 ≈ 2e8 ops → <15s per fold on CPU, 7 folds ≤ 2 min total,
#     well inside the 30-min wall.
#   * StandardScaler upstream — GP RBF length-scale is interpreted in scaled
#     feature units; un-scaled axes (atr_pct ~ 0.03 vs market_breadth_adv ~ 0.5)
#     would force a single learned length-scale to be a bad compromise.
# --------------------------------------------------------------------- #
class GaussianProcessClassifierTrainer(BaseTrainer):
    """Full-posterior Bayesian Matern GP classifier on PCA-projected features.

    iter-#1099 structural change (claude_mode, brief-exhausted pivot):
      * The iter-#1015 / #835 failure mode was documented: GP-RBF on the
        96-d curated feature space saturates — sklearn's marginal-likelihood
        optimizer pushes length_scale to whatever upper bound is set
        (1000 → 10 both saturated), the kernel goes flat, predict_proba
        under-discriminates, every window fails on WR < 40%.
      * Root cause: curse-of-dimensionality on the aggregated curated panel
        (24 features × 4 aggregations [last/mean/std/last-mean] = 96 dims).
        In 96-d, RBF distances concentrate (all points look equidistant),
        so the only way to maintain training accuracy is to lengthen the
        kernel scale until it spans the full data cloud — losing all local
        discrimination.
      * Three coordinated fixes (one structural intervention, "make GP
        robust to high-D noisy financial data"):
          (1) PCA(n_components=20) preprocessing — projects to the 20 most-
              variant components so RBF distances regain meaning.
          (2) Matern(nu=2.5) replaces RBF — finite differentiability
              (≈2 derivatives) is a better prior for noisy financial signals
              than RBF's infinite analytic smoothness; less prone to the
              flat-kernel saturation pattern.
          (3) + WhiteKernel(noise_level) — explicit noise term lets the GP
              attribute residual variance to label noise instead of stretching
              the length-scale to fit every training point exactly.
      * Hypothesis (cites Part A): iTransformer (iter #1072, the brief's pick)
        failed by over-trading W2/W4/W5 (53/36/71 trades, WR 17-21%, DD 25-35%)
        — a deterministic NN that fires too confidently on hostile bear-regime
        days. A well-calibrated Bayesian classifier should naturally
        under-fire on regime-shifted test slices (the GP's posterior variance
        explodes OOD), trading less on W3/W4/W5 and preserving WR.
    """

    name = 'gaussian_process_classifier'

    def __init__(self,
                 n_inducing: int = 1000,
                 n_components: int = 20,
                 kernel_type: str = 'matern',
                 length_scale: float = 2.0,
                 length_scale_bounds_lo: float = 1e-1,
                 length_scale_bounds_hi: float = 1e1,
                 matern_nu: float = 2.5,
                 noise_level: float = 0.5,
                 noise_level_bounds_lo: float = 1e-3,
                 noise_level_bounds_hi: float = 1e1,
                 constant_value: float = 1.0,
                 n_restarts_optimizer: int = 1,
                 max_iter_predict: int = 100,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            n_inducing=int(n_inducing),
            n_components=int(n_components),
            kernel_type=str(kernel_type),
            length_scale=float(length_scale),
            length_scale_bounds_lo=float(length_scale_bounds_lo),
            length_scale_bounds_hi=float(length_scale_bounds_hi),
            matern_nu=float(matern_nu),
            noise_level=float(noise_level),
            noise_level_bounds_lo=float(noise_level_bounds_lo),
            noise_level_bounds_hi=float(noise_level_bounds_hi),
            constant_value=float(constant_value),
            n_restarts_optimizer=int(n_restarts_optimizer),
            max_iter_predict=int(max_iter_predict),
            random_state=int(random_state),
        )
        self.pipe = None
        self._n_features = None

    def _stratified_subsample(self, X, y, dates):
        """Stratified by class, then within-class uniform-by-date subsample."""
        p = self._params
        n_target = min(int(p['n_inducing']), X.shape[0])
        rng = np.random.RandomState(p['random_state'])
        y_arr = np.asarray(y).astype(int)
        idx_pos = np.where(y_arr == 1)[0]
        idx_neg = np.where(y_arr == 0)[0]
        # Preserve class ratio
        pos_frac = len(idx_pos) / max(1, len(y_arr))
        n_pos = max(2, int(round(n_target * pos_frac)))
        n_neg = max(2, n_target - n_pos)
        n_pos = min(n_pos, len(idx_pos))
        n_neg = min(n_neg, len(idx_neg))

        def _date_strided(idx_class, k):
            if len(idx_class) <= k:
                return idx_class
            if dates is None:
                return rng.choice(idx_class, size=k, replace=False)
            # Uniform-by-date: sort by date, take every step-th index.
            dts = np.asarray(dates)[idx_class]
            order = np.argsort(dts)
            sorted_idx = idx_class[order]
            step = max(1, len(sorted_idx) // k)
            picks = sorted_idx[::step][:k]
            if len(picks) < k:
                # Pad by random fill from remaining
                remaining = np.setdiff1d(sorted_idx, picks, assume_unique=False)
                if len(remaining) > 0:
                    extra = rng.choice(remaining,
                                       size=k - len(picks), replace=False)
                    picks = np.concatenate([picks, extra])
            return picks

        keep = np.concatenate([
            _date_strided(idx_pos, n_pos),
            _date_strided(idx_neg, n_neg),
        ])
        return keep

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import (
            RBF, ConstantKernel, Matern, WhiteKernel,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if len(set(np.asarray(y_train).tolist())) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
        if dates_train is not None and dates_val is not None:
            dates_full = np.concatenate(
                [np.asarray(dates_train), np.asarray(dates_val)])
        else:
            dates_full = None
        self._n_features = X_full.shape[1]

        keep_idx = self._stratified_subsample(X_full, y_full, dates_full)
        X_sub = X_full[keep_idx]
        y_sub = y_full[keep_idx]

        ls_bounds = (p['length_scale_bounds_lo'], p['length_scale_bounds_hi'])
        if p['kernel_type'] == 'matern':
            base = Matern(
                length_scale=p['length_scale'],
                length_scale_bounds=ls_bounds,
                nu=p['matern_nu'],
            )
        else:
            base = RBF(
                length_scale=p['length_scale'],
                length_scale_bounds=ls_bounds,
            )
        noise_bounds = (
            p['noise_level_bounds_lo'], p['noise_level_bounds_hi'],
        )
        kernel = (
            ConstantKernel(
                p['constant_value'], constant_value_bounds=(1e-3, 1e3))
            * base
            + WhiteKernel(
                noise_level=p['noise_level'],
                noise_level_bounds=noise_bounds,
            )
        )
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=p['n_restarts_optimizer'],
            max_iter_predict=p['max_iter_predict'],
            random_state=p['random_state'],
            warm_start=False,
            copy_X_train=False,
        )
        n_components = max(2, min(p['n_components'], X_sub.shape[1] - 1))
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components,
                        random_state=p['random_state'])),
            ('gpc', gpc),
        ])
        self.pipe.fit(X_sub, y_sub)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError('Model not fit')
        return self.pipe.predict_proba(X)[:, 1]

    def feature_importance(self):
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pipe.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipe, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'pipe.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        with open(model_path, 'rb') as f:
            inst.pipe = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# MLPClassifierTrainer (iter #851) — pure tabular feed-forward neural net
#
# Motivation (Part A diagnosis, iter #851):
#   * Per-window pass rate across last 200 mixed-trainer iters:
#       W1 14%, W2 26%, W3 24%, W4 17%, W5 15%, W6 23%, W7 12%.
#     W4 and W5 are the hostile bear-vol regimes (W4: SET -3.5%, vol 12%,
#     breadth 43%; W5: SET -15.2%, vol 19.1%, breadth 31.7% — both with
#     persistent foreign outflows). Both have avg WR (28.7% / 27.6%) BELOW
#     random_topk's 31.7% — models *anti-select* in these regimes,
#     i.e. the learned score signal points the wrong way under regime shift.
#   * The registry has 41 trainers but no pure tabular MLP: the torch_*
#     trainers are sequence-based (3D X_seq input). On the 96-dim aggregate
#     [last, mean, std, last-mean], the model space is dominated by tree
#     ensembles (axis-aligned step boundaries) + kernel maps (Nystroem) +
#     QDA/GNB (parametric Gaussian) + KNN (local distance). A regularized
#     feed-forward MLP produces a smooth, learned non-linear decision
#     surface — fundamentally different from all the above and the natural
#     under-filled inductive-bias slot in §6.B.2's NN-family bullet.
#
# Distinct inductive bias vs registry:
#   vs trees (xgb/lgbm/extra_trees): smooth learned non-linearity vs
#       axis-aligned step splits. MLP's weight regularization (alpha) pulls
#       toward simpler interpolating functions, less prone to memorizing
#       bull-regime feature thresholds that invert in W4/W5.
#   vs kernel_logreg: learned hidden representation vs fixed Nyström RBF
#       map. MLP optimizes the basis jointly with the head, so the learned
#       features are task-aware rather than data-density-aware.
#   vs torch_attentive_mlp: pure tabular on aggregated features (96-d) with
#       sklearn's LBFGS solver and built-in early-stopping — no PyTorch
#       overhead, no sequence attention, no per-symbol grouping. Different
#       inductive bias by being structurally simpler.
#   vs logistic_elastic_net: non-linear vs linear; hidden layer learns
#       feature interactions that elastic-net cannot represent.
#
# Implementation:
#   * sklearn.neural_network.MLPClassifier with two hidden layers
#     (default (64, 32)). On ~30k-row windows × 96 features this is
#     ~6k + 2k + 33 ≈ 8k parameters — well below the overfitting cliff for
#     this data size; comparable to xgboost's effective complexity at
#     n_estimators=500 max_depth=6.
#   * StandardScaler upstream: MLP weight initialization (Glorot/He) is
#     unit-variance-tuned; unscaled atr_pct (~0.03) vs market_breadth (~0.5)
#     would push some hidden units to saturation/death.
#   * solver='adam' with early_stopping=True and validation_fraction=0.15
#     gives sklearn-native early stopping on a held-out slice of the
#     train+val concatenation — robust regularizer for noisy financial
#     labels. n_iter_no_change=15 prevents premature stop on flat loss.
#   * alpha (L2 reg) defaults to 1e-3 (10x sklearn default) — financial
#     data is high-noise; stronger regularization toward smoother boundaries
#     is the structural lever that should dampen W4/W5 anti-selection by
#     preventing the network from memorizing bull-regime patterns.
#   * One quiet caveat: MLPClassifier converges with adam stochastically;
#     fix random_state for reproducibility and let train mode sample over
#     hidden_layer_sizes / alpha / learning_rate_init in the HP space.
# --------------------------------------------------------------------- #
class MLPClassifierTrainer(BaseTrainer):
    """Two-layer feed-forward MLP for binary win/loss classification."""

    name = 'mlp_classifier'

    def __init__(self,
                 hidden_layer_1: int = 64,
                 hidden_layer_2: int = 32,
                 alpha: float = 1e-3,
                 learning_rate_init: float = 1e-3,
                 max_iter: int = 300,
                 batch_size: int = 256,
                 activation: str = 'relu',
                 early_stopping: bool = True,
                 validation_fraction: float = 0.15,
                 n_iter_no_change: int = 15,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            hidden_layer_1=int(hidden_layer_1),
            hidden_layer_2=int(hidden_layer_2),
            alpha=float(alpha),
            learning_rate_init=float(learning_rate_init),
            max_iter=int(max_iter),
            batch_size=int(batch_size),
            activation=str(activation),
            early_stopping=bool(early_stopping),
            validation_fraction=float(validation_fraction),
            n_iter_no_change=int(n_iter_no_change),
            random_state=int(random_state),
        )
        self.pipe = None
        self._n_features = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        # Concatenate train+val: MLP's early_stopping uses an internal split.
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val]).astype(int)
        self._n_features = X_full.shape[1]

        hidden = tuple(
            x for x in (p['hidden_layer_1'], p['hidden_layer_2']) if x > 0
        )
        if not hidden:
            hidden = (32,)

        # batch_size capped by n_samples to avoid sklearn warning.
        bs = min(p['batch_size'], X_full.shape[0])

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=p['activation'],
            solver='adam',
            alpha=p['alpha'],
            batch_size=bs,
            learning_rate_init=p['learning_rate_init'],
            max_iter=p['max_iter'],
            shuffle=True,
            random_state=p['random_state'],
            early_stopping=p['early_stopping'],
            validation_fraction=p['validation_fraction'],
            n_iter_no_change=p['n_iter_no_change'],
            verbose=False,
        )

        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', mlp),
        ])
        self.pipe.fit(X_full, y_full)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError('Model not fit')
        return self.pipe.predict_proba(X)[:, 1]

    def feature_importance(self):
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pipe.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipe, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'pipe.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        with open(model_path, 'rb') as f:
            inst.pipe = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# Registry — add new model types here
# --------------------------------------------------------------------- #


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

    Explicit kwargs (vs the prior **kwargs sink) so train_mode's HP sweep
    can actually reach this trainer — get_trainer() filters kwargs by
    __init__.co_varnames and was previously dropping every HP, locking
    this trainer at defaults forever (zero rows in feedback DB across
    994 iters).
    """
    name = 'tabpfn_v25'
    consumes_sequences = False

    def __init__(self,
                 n_estimators: int = 2,
                 softmax_temperature: float = 0.9,
                 balance_probabilities: bool = False,
                 average_before_softmax: bool = False,
                 ignore_pretraining_limits: bool = True,
                 max_train_rows: int = 10000,
                 random_state: int = 42,
                 **_):
        if not _HAS_TABPFN:
            raise ImportError(
                "tabpfn not installed. `pip install tabpfn` (Python 3.9+, "
                "PyTorch>=2.1). CUDA strongly recommended."
            )
        self.n_estimators = int(n_estimators)
        self.softmax_temperature = float(softmax_temperature)
        self.balance_probabilities = bool(balance_probabilities)
        self.average_before_softmax = bool(average_before_softmax)
        self.ignore_pretraining_limits = bool(ignore_pretraining_limits)
        self.random_state = int(random_state)
        self.max_train_rows = int(max_train_rows)
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
        if not torch.cuda.is_available():
            return 'cpu'
        # Pick the CUDA device with the most free memory — shared workstations
        # often have one GPU pinned by another process.
        best_idx, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except Exception:
                free = 0
            if free > best_free:
                best_free, best_idx = free, i
        return f'cuda:{best_idx}'

    def _load_encoder(self):
        if self._encoder is not None:
            return
        self._device = self._pick_device()
        self._encoder = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self._device,
            trust_remote_code=True,
            dtype=torch.float32,
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
        # Time-MoE accepts float-valued (B, T) input via its causal-LM forward pass.
        # use_cache=False bypasses DynamicCache.from_legacy_cache() which was removed
        # in transformers v5 (Time-MoE's modeling code still calls it via past_key_values).
        n = X.shape[0]
        bs = self.encode_batch_size
        out = []
        for i in range(0, n, bs):
            xb = torch.from_numpy(X[i:i + bs]).to(self._device).float()
            outputs = self._encoder(
                input_ids=xb,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
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


# models/trainers.py — append at end (before TRAINERS dict)

import os
import json
import numpy as np

class TabMClassifierTrainer(BaseTrainer):
    name = 'tabm_classifier'
    consumes_sequences = False

    def __init__(self,
                 k=32,
                 n_blocks=3,
                 d_block=512,
                 dropout=0.1,
                 lr=2e-3,
                 weight_decay=3e-4,
                 batch_size=512,
                 max_epochs=120,
                 patience=12,
                 grad_clip=1.0,
                 device=None,
                 seed=42,
                 **kwargs):
        self.k = int(k)
        self.n_blocks = int(n_blocks)
        self.d_block = int(d_block)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.grad_clip = float(grad_clip)
        self.seed = int(seed)
        self.device = device or self._pick_device()
        self.model = None
        self.imputer = None
        self.scaler = None
        self.n_features_ = None

    @staticmethod
    def _pick_device(min_free_mb=1024):
        import torch as _torch
        try:
            if not _torch.cuda.is_available():
                return 'cpu'
            best_idx, best_free = -1, -1
            for i in range(_torch.cuda.device_count()):
                try:
                    free, _ = _torch.cuda.mem_get_info(i)
                except Exception:
                    free = 0
                if free > best_free:
                    best_free, best_idx = free, i
            if best_free < min_free_mb * 1024 * 1024:
                return 'cpu'
            return f'cuda:{best_idx}'
        except Exception:
            return 'cpu'

    def _make_model(self, n_features):
        import torch
        from tabm import TabM
        return TabM.make(
            n_num_features=n_features,
            cat_cardinalities=[],
            d_out=1,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
            k=self.k,
        ).to(self.device)

    def _to_2d(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X[:, -1, :]
        return X

    def _preprocess(self, X, fit=False):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        X = self._to_2d(X)
        if fit:
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = StandardScaler()
            X = self.imputer.fit_transform(X)
            X = self.scaler.fit_transform(X)
            self.n_features_ = X.shape[1]
        else:
            X = self.imputer.transform(X)
            X = self.scaler.transform(X)
        return np.clip(X.astype(np.float32), -10.0, 10.0)

    def fit(self, X_tr, y_tr, X_val, y_val, **kwargs):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_tr_p = self._preprocess(X_tr, fit=True)
        X_val_p = self._preprocess(X_val, fit=False)
        y_tr_a = np.asarray(y_tr, dtype=np.float32).reshape(-1)
        y_val_a = np.asarray(y_val, dtype=np.float32).reshape(-1)

        self.model = self._make_model(self.n_features_)
        opt = optim.AdamW(self.model.parameters(),
                          lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        X_tr_t = torch.from_numpy(X_tr_p)
        y_tr_t = torch.from_numpy(y_tr_a).unsqueeze(1)
        X_val_t = torch.from_numpy(X_val_p).to(self.device)
        y_val_t = torch.from_numpy(y_val_a).unsqueeze(1).to(self.device)

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        best_val = float('inf')
        best_state = None
        bad = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)  # (B, k, 1)
                yb_rep = yb.unsqueeze(1).expand_as(logits)
                loss = loss_fn(logits, yb_rep)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()

            self.model.eval()
            with torch.no_grad():
                logits_val = self.model(X_val_t)
                yv_rep = y_val_t.unsqueeze(1).expand_as(logits_val)
                val_loss = loss_fn(logits_val, yv_rep).item()

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        X_p = self._preprocess(X, fit=False)
        self.model.eval()
        X_t = torch.from_numpy(X_p).to(self.device)
        probs_chunks = []
        bs = 4096
        with torch.no_grad():
            for i in range(0, X_t.shape[0], bs):
                logits = self.model(X_t[i:i + bs])  # (B, k, 1)
                p = torch.sigmoid(logits).mean(dim=1).squeeze(-1)
                probs_chunks.append(p.detach().cpu().numpy())
        return np.concatenate(probs_chunks, axis=0).astype(np.float32)

    def save(self, model_dir, extra=None):
        import torch
        import joblib
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(),
                   os.path.join(model_dir, 'tabm_state.pt'))
        joblib.dump(
            {'imputer': self.imputer, 'scaler': self.scaler},
            os.path.join(model_dir, 'tabm_preproc.joblib'),
        )
        meta = {
            'name': self.name,
            'k': self.k,
            'n_blocks': self.n_blocks,
            'd_block': self.d_block,
            'dropout': self.dropout,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'grad_clip': self.grad_clip,
            'n_features': self.n_features_,
            'seed': self.seed,
        }
        if extra:
            meta.update(extra)
        with open(os.path.join(model_dir, 'tabm_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)


class TabMLCBClassifierTrainer(TabMClassifierTrainer):
    # TabM with Lower-Confidence-Bound scoring: score = mean(p_k) - lam * std(p_k).
    # Iter #921 diagnosis: WR < 40% is the dominant gate failure (17-19/20 across
    # last 20 iters). TabM's plain mean throws away the k=32 BatchEnsemble
    # dispersion — exactly the signal that distinguishes "all members agree this
    # wins" from "members disagree, mean happens to be high". Precedent:
    # torch_seq_gru_ensemble uses the same form on a 5-net independent ensemble
    # and is the only trainer that explicitly penalises disagreement.
    name = 'tabm_lcb'

    # NOTE: get_trainer() filters kwargs via __init__.__code__.co_varnames, so
    # the full HP surface must be named here even though TabMClassifierTrainer
    # already exposes them.
    def __init__(self,
                 k=32,
                 n_blocks=3,
                 d_block=512,
                 dropout=0.1,
                 lr=2e-3,
                 weight_decay=3e-4,
                 batch_size=512,
                 max_epochs=120,
                 patience=12,
                 grad_clip=1.0,
                 disagreement_penalty=0.3,
                 device=None,
                 seed=42,
                 **kwargs):
        super().__init__(k=k, n_blocks=n_blocks, d_block=d_block, dropout=dropout,
                         lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                         max_epochs=max_epochs, patience=patience, grad_clip=grad_clip,
                         device=device, seed=seed, **kwargs)
        self.disagreement_penalty = float(disagreement_penalty)

    def predict_proba(self, X) -> np.ndarray:
        import torch
        X_p = self._preprocess(X, fit=False)
        self.model.eval()
        X_t = torch.from_numpy(X_p).to(self.device)
        score_chunks = []
        bs = 4096
        lam = self.disagreement_penalty
        with torch.no_grad():
            for i in range(0, X_t.shape[0], bs):
                logits = self.model(X_t[i:i + bs])      # (B, k, 1)
                p = torch.sigmoid(logits).squeeze(-1)    # (B, k)
                mu = p.mean(dim=1)
                sd = p.std(dim=1, unbiased=False)
                lcb = (mu - lam * sd).clamp(0.0, 1.0)
                score_chunks.append(lcb.detach().cpu().numpy())
        return np.concatenate(score_chunks, axis=0).astype(np.float32)


# --------------------------------------------------------------------- #
# HistGradientBoostingClassifier with monotonic constraints (iter #936)
#
# Diagnosis (Part A, this iteration):
#   * Cross-tab over last 20 iters: W2/W4/W5/W6 fail in 19/20+ runs across
#     all trainer families. Per-window WR averages 25.7-27.1% — BELOW the
#     ~30% random-topk WR baseline. Models are anti-selecting in mild-bear
#     and even bull-but-low-foreign-flow regimes.
#   * The most striking finding: W6 (bull market, +4.9% SET return, 51.7%
#     breadth, mid vol) has 0/20 pass rate with 26.5% WR. The model picks
#     WORSE than random in a regime that "looks" like the passing W1 (+4.5%
#     SET, 47.5% breadth) and W7 (+3.4%, 48.4% breadth). Hypothesis: as
#     train window grows, bear-heavy 2024 data dominates and biases the
#     model to learn spurious "high breadth → SHORT" type correlations
#     that fire in W6.
#   * TabM family (last 14 iters #922-#935) saturated at 2/7 max via
#     lambda sweep. New inductive bias needed.
#
# Hypothesis: Monotonic constraints on regime features prevent the W6
# anti-selection. Forcing the model to respect "more breadth → more
# bullish" / "higher foreign-flow rank → more bullish" as economic priors
# blocks the spurious negative correlations the bear-heavy training set
# would otherwise teach. HistGradientBoostingClassifier supports per-
# feature monotonic_cst natively (sklearn >= 0.23) and uses histogram-
# based binning + level-wise tree growth — structurally distinct from
# XGBoost's leaf-wise growth and the existing xgb_regime_blend (which is
# a soft regime gate, not a hard monotonic constraint).
#
# Distinct inductive bias vs registry:
#   vs XGBoost / LightGBM: histogram binning + level-wise growth + native
#     per-feature monotonic constraints (XGB has them too but our XGB
#     trainers don't use them; this is the first monotonic-constrained
#     trainer in the registry).
#   vs sklearn_extra_trees / random forest: gradient-boosted (sequential,
#     residual-fitting) vs bagged.
#   vs xgb_regime_blend: hard guarantee on monotonicity vs soft mixture
#     weights — the model CAN'T learn "high breadth → short" no matter
#     what the gradient says.
#
# Constrained features (monotonically increasing → P(y=1)):
#   sector_breadth, foreign_net_monthly_pctrank, market_breadth_adv,
#   market_breadth_above_sma20, up_days_5d, market_new_highs,
#   market_sector_aligned, set_ret_5d_zscore_60d
#
# Constraints applied to the `last` and `mean` aggregations (the level
# information). `std` and `dev` (last - mean) carry change information
# and are left unconstrained — we don't want to force "higher std →
# bullish" since vol regimes are not directionally signed.
# --------------------------------------------------------------------- #
class HistGBMonotonicTrainer(BaseTrainer):
    """HistGradientBoostingClassifier with monotonic constraints on regime features."""

    name = 'histgb_monotonic'

    # Curated feature names that should be monotonically increasing in P(y=1).
    # Names must match models.feature_eng.CURATED_FEATURES exactly.
    #
    # Regime/breadth features (8) — iter #936 set. Block W6-style
    # "high breadth → SHORT" anti-selection from bear-heavy training data.
    #
    # Stock-level momentum features (4) — iter #951 extension. Same
    # mechanism at the stock level. Best prior (iter #946) hit 5/7 with
    # W1 (+4.5% SET bull) at 28% WR and W3 (-7.6% SET bear) at 32% WR —
    # both anti-selection within the universe. With these four added the
    # gate run produced 3/7 PASS but avg_ann +43.2% (vs +30.7% best prior)
    # and three of the four failing windows are 1.5-2.5pp short of the WR
    # bar — clear room for pos_class_weight HP tuning to close the gap.
    # Volume-coupled features are PAIRED with price-momentum features:
    # the 2-feature ablation (price-only) regressed to 2/7 WR 34.8%, so
    # the volume terms balance the pure-momentum overcommitment.
    _MONOTONIC_INCREASING = (
        # Regime/breadth (8)
        'sector_breadth',
        'foreign_net_monthly_pctrank',
        'market_breadth_adv',
        'market_breadth_above_sma20',
        'up_days_5d',
        'market_new_highs',
        'market_sector_aligned',
        'set_ret_5d_zscore_60d',
        # Stock-level momentum (4) — price + volume confirmation pair
        'volume_ratio_xrank',
        'ret_5d_xrank',
        'macd_xrank',
        'momentum_volume_cross',
    )

    def __init__(self,
                 max_iter: int = 400,
                 max_leaf_nodes: int = 31,
                 max_depth: Optional[int] = None,
                 learning_rate: float = 0.05,
                 min_samples_leaf: int = 50,
                 l2_regularization: float = 1.0,
                 max_bins: int = 255,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.15,
                 n_iter_no_change: int = 20,
                 tol: float = 1e-4,
                 pos_class_weight: float = 2.5,
                 use_monotonic: bool = True,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            max_iter=int(max_iter),
            max_leaf_nodes=int(max_leaf_nodes),
            max_depth=None if max_depth in (None, 0, -1) else int(max_depth),
            learning_rate=float(learning_rate),
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=float(l2_regularization),
            max_bins=int(max_bins),
            early_stopping=bool(early_stopping),
            validation_fraction=float(validation_fraction),
            n_iter_no_change=int(n_iter_no_change),
            tol=float(tol),
            pos_class_weight=float(pos_class_weight),
            use_monotonic=bool(use_monotonic),
            random_state=int(random_state),
        )
        self.clf = None
        self._n_features = None
        self._monotonic_cst = None

    def _build_monotonic_constraints(self, n_features: int) -> Optional[np.ndarray]:
        """Map curated feature names → indices in the (4F,) aggregated vector.

        The aggregator emits [last_*, mean_*, std_*, dev_*] in CURATED_FEATURES
        order. We constrain only `last` and `mean` (level info) — std/dev are
        change/dispersion info that isn't directionally signed.
        """
        if not self._params['use_monotonic']:
            return None
        try:
            from models.feature_eng import CURATED_FEATURES
        except Exception:
            return None
        F = len(CURATED_FEATURES)
        if n_features != 4 * F:
            # If the gate ever changes aggregation shape, silently fall back.
            return None
        cst = np.zeros(n_features, dtype=np.int8)
        name_to_idx = {n: i for i, n in enumerate(CURATED_FEATURES)}
        for name in self._MONOTONIC_INCREASING:
            i = name_to_idx.get(name)
            if i is None:
                continue
            cst[i] = 1            # last_<name>
            cst[F + i] = 1        # mean_<name>
        return cst

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.ensemble import HistGradientBoostingClassifier

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        # Concatenate train+val: HistGBM has its own internal validation split
        # for early stopping (validation_fraction), so the gate's outer val is
        # not needed for fit-time selection. Same convention as ExtraTrees.
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self._n_features = X_full.shape[1]
        self._monotonic_cst = self._build_monotonic_constraints(self._n_features)

        # Sample weights for class imbalance — sklearn HistGBM doesn't accept
        # class_weight='balanced', so up-weight positives explicitly.
        sw = np.where(y_full == 1, p['pos_class_weight'], 1.0).astype(np.float32)

        self.clf = HistGradientBoostingClassifier(
            max_iter=p['max_iter'],
            max_leaf_nodes=p['max_leaf_nodes'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            min_samples_leaf=p['min_samples_leaf'],
            l2_regularization=p['l2_regularization'],
            max_bins=p['max_bins'],
            early_stopping=p['early_stopping'],
            validation_fraction=p['validation_fraction'] if p['early_stopping'] else None,
            n_iter_no_change=p['n_iter_no_change'],
            tol=p['tol'],
            monotonic_cst=self._monotonic_cst,
            random_state=p['random_state'],
            verbose=0,
        )
        self.clf.fit(X_full, y_full, sample_weight=sw)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError('Model not fit')
        return self.clf.predict_proba(X)[:, 1]

    def feature_importance(self):
        # sklearn HistGBM exposes no built-in feature_importances_ (binning
        # makes that ambiguous); permutation importance is a separate call.
        return None

    @property
    def best_iteration(self):
        if self.clf is None:
            return None
        return int(getattr(self.clf, 'n_iter_', self._params['max_iter']))

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.clf, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
            'monotonic_cst': None if self._monotonic_cst is None
                              else [int(v) for v in self._monotonic_cst.tolist()],
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'model.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        cst = meta.get('monotonic_cst')
        inst._monotonic_cst = None if cst is None else np.array(cst, dtype=np.int8)
        with open(model_path, 'rb') as f:
            inst.clf = pickle.load(f)
        return inst


# --------------------------------------------------------------------- #
# HistGBMonotonicBaggedTrainer (iter #1131, claude_mode, brief-exhausted pivot)
#
# Pivot context: brief #1131 picked torch_itransformer (variate-axis attention)
# but it exhausted at 2 attempts wp=0. Per directive, pivoting to a trainer
# family OUTSIDE the NN bucket. histgb_monotonic just hit wp=5/7 (iter #1128
# ann +74.2% wr 40.0% trades 191) — the strongest base learner in the registry.
# The remaining gap is the WR plateau: avg_wr lives in 33-40% range and 2-3
# windows fail by <2pp. WR is a precision metric — to push it up, we either
# (a) train a better discriminator, or (b) abstain more aggressively on
# low-confidence picks.
#
# Structural change: date-bagged ensemble of K=3 HistGBMonotonic learners
# with LCB-style score = mean(p) - lambda * std(p). Different from:
#   * histgb_monotonic (single model) — adds inter-bag disagreement abstention
#   * tabm_lcb (TabM/BatchEnsemble NN-based) — uses tree boosting base + date
#     bagging (block bootstrap) rather than parameter-shared NN ensembling
#   * bagged_xgb_regressor (XGB regression, row-bootstrap) — uses HistGBM
#     classification + DATE-block bootstrap (whole-day blocks resampled with
#     replacement, preserving within-day cross-sectional structure) +
#     monotonic constraints retained from base learner
#
# Why date-block bootstrap (not row bootstrap): with cross-sectional features
# (sector_breadth, market_breadth_adv) being identical for all symbols on the
# same date, row bootstrap leaks the same date into multiple bags. Sampling
# whole days with replacement preserves the cross-sectional context while
# still varying the date set across bags — closer to the temporal block
# bootstrap recommended for time-series ensembling.
# --------------------------------------------------------------------- #
class HistGBMonotonicBaggedTrainer(BaseTrainer):
    """K-bag ensemble of HistGBMonotonic with pluggable aggregation.

    iter #1131 (LCB at lam=1.0, bag_frac=0.85) hit only wp=1/7 (avg_wr 31.3%
    vs base histgb_monotonic 38.5%). Failure was structural: LCB demotion
    (mean - λ·std) and bag_frac<1.0 stack the same demotion vector, leaving
    a thin top-of-day score band that the gate threshold cannot exploit.
    Defaults below pivot to MEDIAN aggregation with bag_frac=1.0 (standard
    bootstrap, all unique dates eligible) so the bagging contributes
    robustness-to-outlier-bag without an extra confidence penalty stacked
    on top. LCB is retained as a sweep-time option.
    """

    name = 'histgb_monotonic_bagged'

    _AGGREGATIONS = ('median', 'lcb', 'trimmed_lcb', 'mean')

    def __init__(self,
                 n_bags: int = 3,
                 aggregation: str = 'median',
                 conf_lambda: float = 0.0,
                 bag_frac: float = 1.0,
                 max_iter: int = 400,
                 max_leaf_nodes: int = 31,
                 max_depth: Optional[int] = None,
                 learning_rate: float = 0.03,
                 min_samples_leaf: int = 50,
                 l2_regularization: float = 1.0,
                 max_bins: int = 255,
                 n_iter_no_change: int = 20,
                 pos_class_weight: float = 3.0,
                 use_monotonic: bool = True,
                 random_state: int = 42,
                 **_):
        agg = str(aggregation).lower()
        if agg not in self._AGGREGATIONS:
            raise ValueError(
                f'aggregation={aggregation!r} not in {self._AGGREGATIONS}')
        self._params = dict(
            n_bags=int(n_bags),
            aggregation=agg,
            conf_lambda=float(conf_lambda),
            bag_frac=float(bag_frac),
            max_iter=int(max_iter),
            max_leaf_nodes=int(max_leaf_nodes),
            max_depth=None if max_depth in (None, 0, -1) else int(max_depth),
            learning_rate=float(learning_rate),
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=float(l2_regularization),
            max_bins=int(max_bins),
            n_iter_no_change=int(n_iter_no_change),
            pos_class_weight=float(pos_class_weight),
            use_monotonic=bool(use_monotonic),
            random_state=int(random_state),
        )
        self.clfs = []
        self._monotonic_cst = None
        self._n_features = None

    def _build_monotonic_constraints(self, n_features: int) -> Optional[np.ndarray]:
        """Same constraint vector as HistGBMonotonicTrainer."""
        if not self._params['use_monotonic']:
            return None
        try:
            from models.feature_eng import CURATED_FEATURES
        except Exception:
            return None
        F = len(CURATED_FEATURES)
        if n_features != 4 * F:
            return None
        names = HistGBMonotonicTrainer._MONOTONIC_INCREASING
        cst = np.zeros(n_features, dtype=np.int8)
        name_to_idx = {n: i for i, n in enumerate(CURATED_FEATURES)}
        for n in names:
            i = name_to_idx.get(n)
            if i is None:
                continue
            cst[i] = 1
            cst[F + i] = 1
        return cst

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.ensemble import HistGradientBoostingClassifier

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        if dates_train is not None and dates_val is not None:
            dates_full = np.concatenate([np.asarray(dates_train),
                                         np.asarray(dates_val)])
        else:
            dates_full = None
        self._n_features = X_full.shape[1]
        self._monotonic_cst = self._build_monotonic_constraints(self._n_features)

        rng_master = np.random.default_rng(p['random_state'])
        K = max(1, p['n_bags'])
        self.clfs = []

        for k in range(K):
            seed_k = int(rng_master.integers(0, 2**31 - 1))
            rng_k = np.random.default_rng(seed_k)

            if dates_full is not None:
                unique_dates = np.unique(dates_full)
                n_sample = max(1, int(round(p['bag_frac'] * len(unique_dates))))
                sampled_dates = rng_k.choice(
                    unique_dates, size=n_sample, replace=True)
                # Build a mapping for membership using a Counter — duplicates
                # become row repetitions for that date.
                from collections import Counter
                cnt = Counter(sampled_dates.tolist())
                pieces_X, pieces_y, pieces_w = [], [], []
                for d, m in cnt.items():
                    mask = dates_full == d
                    if mask.sum() == 0:
                        continue
                    pieces_X.append(X_full[mask])
                    pieces_y.append(y_full[mask])
                    pieces_w.append(np.full(mask.sum(), m, dtype=np.float32))
                if not pieces_X:
                    Xb = X_full
                    yb = y_full
                    wb_base = np.ones(len(y_full), dtype=np.float32)
                else:
                    Xb = np.vstack(pieces_X)
                    yb = np.concatenate(pieces_y)
                    wb_base = np.concatenate(pieces_w)
            else:
                # Row bootstrap fallback (no dates available).
                n = X_full.shape[0]
                idx = rng_k.integers(0, n, size=n)
                Xb = X_full[idx]
                yb = y_full[idx]
                wb_base = np.ones(n, dtype=np.float32)

            if len(set(yb)) < 2:
                # Degenerate bag — skip.
                continue

            sw = wb_base * np.where(
                yb == 1, p['pos_class_weight'], 1.0).astype(np.float32)

            clf = HistGradientBoostingClassifier(
                max_iter=p['max_iter'],
                max_leaf_nodes=p['max_leaf_nodes'],
                max_depth=p['max_depth'],
                learning_rate=p['learning_rate'],
                min_samples_leaf=p['min_samples_leaf'],
                l2_regularization=p['l2_regularization'],
                max_bins=p['max_bins'],
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=p['n_iter_no_change'],
                tol=1e-4,
                monotonic_cst=self._monotonic_cst,
                random_state=seed_k,
                verbose=0,
            )
            clf.fit(Xb, yb, sample_weight=sw)
            self.clfs.append(clf)

        if not self.clfs:
            raise RuntimeError('All bags collapsed — refusing to predict')
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.clfs:
            raise RuntimeError('Model not fit')
        # Stack per-bag P(y=1) → shape (K, N)
        probs = np.stack(
            [clf.predict_proba(X)[:, 1] for clf in self.clfs], axis=0)
        p = self._params
        agg = p['aggregation']
        if agg == 'median':
            score = np.median(probs, axis=0)
        elif agg == 'mean':
            score = probs.mean(axis=0)
        elif agg == 'lcb':
            score = probs.mean(axis=0) - p['conf_lambda'] * probs.std(axis=0)
        elif agg == 'trimmed_lcb':
            # Drop max and min bag probability per row, then mean - λ·std on
            # the middle K-2 bags. Needs n_bags >= 3 to leave a non-empty
            # middle; with K=3 reduces to "median - 0 (std degenerate)".
            K = probs.shape[0]
            if K < 3:
                score = probs.mean(axis=0) - p['conf_lambda'] * probs.std(axis=0)
            else:
                srt = np.sort(probs, axis=0)
                mid = srt[1:-1]  # drop extremes
                score = mid.mean(axis=0) - p['conf_lambda'] * mid.std(axis=0)
        else:
            score = probs.mean(axis=0)
        # Clip to [0, 1] so the downstream threshold sweep still has a
        # comparable scale (the SCORE_THRESHOLDS list spans 0..0.7).
        return np.clip(score, 0.0, 1.0)

    def feature_importance(self):
        return None

    @property
    def best_iteration(self):
        if not self.clfs:
            return None
        return int(np.mean([getattr(c, 'n_iter_', 0) for c in self.clfs]))

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(self.clfs, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
            'n_bags_fit': len(self.clfs),
            'monotonic_cst': None if self._monotonic_cst is None
                              else [int(v) for v in self._monotonic_cst.tolist()],
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'model.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        cst = meta.get('monotonic_cst')
        inst._monotonic_cst = None if cst is None else np.array(cst, dtype=np.int8)
        with open(model_path, 'rb') as f:
            inst.clfs = pickle.load(f)
        return inst


# models/trainers.py — append at end (before TRAINERS dict)

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _ITransformerBlock(nn.Module):
    """Pre-LayerNorm block: variate-axis multi-head self-attention + FFN."""

    def __init__(self, dim, heads, dim_head, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.ln_attn = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.attn_out = nn.Linear(inner, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ln_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, V, dim) — V is num_variates; attention runs across V.
        b, v, d = x.shape
        h = self.ln_attn(x)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, val = (t.reshape(b, v, self.heads, self.dim_head).transpose(1, 2) for t in qkv)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, val).transpose(1, 2).reshape(b, v, self.heads * self.dim_head)
        x = x + self.attn_out(out)
        x = x + self.ffn(self.ln_ffn(x))
        return x


class _ITransformerClassifier(nn.Module):
    def __init__(self, num_variates, lookback_len=1, dim=128, depth=3,
                 heads=4, dim_head=32, dropout=0.1):
        super().__init__()
        self.variate_embed = nn.Linear(lookback_len, dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _ITransformerBlock(dim, heads, dim_head, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        # Accept (B, F) for 2D tabular OR (B, T, F) for sequence input.
        if x.dim() == 2:
            x = x.unsqueeze(-1)              # (B, F, 1) — variates with lookback 1
        else:
            x = x.permute(0, 2, 1).contiguous()  # (B, T, F) -> (B, F, T) — variates as tokens
        tokens = self.variate_embed(x)       # (B, V, dim)
        tokens = self.input_dropout(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.ln_out(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class TorchITransformerTrainer(BaseTrainer):
    """iTransformer-style classifier (Liu et al., ICLR 2024).

    Inverts the conventional transformer axis: each feature becomes a token and
    self-attention runs across features (variate-axis), modelling feature-feature
    correlations explicitly. Mean-pools variate tokens, then a 2-layer MLP head
    produces a scalar logit. Pure PyTorch — no external dependency beyond torch.
    """

    name = 'torch_itransformer'
    consumes_sequences = False

    def __init__(self, dim=128, depth=3, heads=4, dim_head=32, dropout=0.1,
                 learning_rate=1e-3, weight_decay=1e-4, batch_size=512,
                 epochs=40, patience=8, pos_weight=None, device=None, **kwargs):
        self.dim = int(dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.pos_weight = pos_weight
        if device is None:
            device = 'cpu'
            try:
                if torch.cuda.is_available():
                    best_idx, best_free = -1, -1
                    for i in range(torch.cuda.device_count()):
                        try:
                            free, _ = torch.cuda.mem_get_info(i)
                        except Exception:
                            free = 0
                        if free > best_free:
                            best_free, best_idx = free, i
                    if best_free >= 1_500_000_000:
                        device = f'cuda:{best_idx}'
            except Exception:
                pass
        self.device = device
        self.model = None
        self.feature_mean_ = None
        self.feature_std_ = None

    def _make_model(self, num_variates, lookback_len):
        return _ITransformerClassifier(
            num_variates=num_variates,
            lookback_len=lookback_len,
            dim=self.dim, depth=self.depth,
            heads=self.heads, dim_head=self.dim_head,
            dropout=self.dropout,
        ).to(self.device)

    def _normalize(self, X, fit=False):
        flat = X.reshape(-1, X.shape[-1])
        if fit:
            self.feature_mean_ = np.nanmean(flat, axis=0).astype(np.float32)
            self.feature_std_ = (np.nanstd(flat, axis=0) + 1e-6).astype(np.float32)
        Xn = (X - self.feature_mean_) / self.feature_std_
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return Xn

    def fit(self, X_tr, y_tr, X_val, y_val, **kwargs):
        X_tr = np.asarray(X_tr, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_tr = np.asarray(y_tr, dtype=np.float32).ravel()
        y_val = np.asarray(y_val, dtype=np.float32).ravel()
        X_tr = self._normalize(X_tr, fit=True)
        X_val = self._normalize(X_val, fit=False)
        if X_tr.ndim == 2:
            num_variates = X_tr.shape[1]
            lookback_len = 1
        else:
            num_variates = X_tr.shape[2]
            lookback_len = X_tr.shape[1]
        torch.manual_seed(42)
        self.model = self._make_model(num_variates, lookback_len)
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        pos_w = None
        if self.pos_weight is not None:
            pos_w = torch.tensor(float(self.pos_weight), device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)
        best_val = float('inf')
        best_state = None
        bad_epochs = 0
        for _epoch in range(self.epochs):
            self.model.train()
            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            self.model.eval()
            with torch.no_grad():
                val_losses = []
                for i in range(0, X_val_t.shape[0], self.batch_size):
                    xb = X_val_t[i:i + self.batch_size].to(self.device, non_blocking=True)
                    yb = y_val_t[i:i + self.batch_size].to(self.device, non_blocking=True)
                    vl = self.model(xb)
                    val_losses.append(float(loss_fn(vl, yb).item()) * xb.shape[0])
                val_loss = sum(val_losses) / max(X_val_t.shape[0], 1)
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize(X, fit=False)
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, Xn.shape[0], self.batch_size):
                xb = torch.from_numpy(Xn[i:i + self.batch_size]).to(self.device)
                logits = self.model(xb)
                probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
                out.append(probs)
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    def save(self, model_dir, extra=None):
        os.makedirs(model_dir, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(model_dir, 'model.pt'))
        cfg = {
            'name': self.name,
            'dim': self.dim, 'depth': self.depth, 'heads': self.heads,
            'dim_head': self.dim_head, 'dropout': self.dropout,
            'learning_rate': self.learning_rate, 'weight_decay': self.weight_decay,
            'batch_size': self.batch_size, 'epochs': self.epochs,
            'patience': self.patience, 'pos_weight': self.pos_weight,
        }
        if extra:
            cfg.update(extra)
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        if self.feature_mean_ is not None:
            np.save(os.path.join(model_dir, 'feature_mean.npy'), self.feature_mean_)
            np.save(os.path.join(model_dir, 'feature_std.npy'), self.feature_std_)


# --------------------------------------------------------------------- #
# TorchTabNetTrainer (claude_mode iter #1147, brief-exhausted pivot)
#
# Brief #1147 picked torch_itransformer (variate-axis attention) but it
# exhausted after 2 claude attempts (wp=0 each, iter #1061 and #1072) plus
# 20+ train-mode HP sweeps (best wp=2). Pivoting to a structurally different
# NN family per §6.B.0(b).
#
# TabNet (Arik & Pfister, AAAI 2021) is a tabular DL family with a
# fundamentally different inductive bias from anything currently registered:
#   - SEQUENTIAL decision steps (3-5), each step picks a small SPARSE subset
#     of features via a sparsemax-attention mask
#   - Per-step feature transformer (shared GLU blocks + step-specific GLU
#     blocks, residual scaled 1/sqrt(2))
#   - Mask-reuse prior (gamma > 1) discourages a step from re-selecting
#     features used in earlier steps -> features get distributed across steps
#   - Entropy penalty on the per-step mask encourages SPARSITY (few features
#     per step) but the DIFFERENT features-across-steps means the model
#     collectively uses many features
#
# Why this addresses W1/W4 stealth-bear failure pattern (Part A diagnosis):
#   - W1 (set_ret=+5.2%, breadth 47%, foreign_flow -47.6B) and
#     W4 (set_ret=-1.7%, breadth 43%, foreign_flow -23.2B) are "stealth bear"
#     regimes: roughly flat SET but bearish flows; the pattern fails in
#     18/20 recent iterations regardless of trainer family
#   - Dense models (iTransformer attention, MLP, XGB w/ multi-feature splits)
#     learn to predict UP for ANY pattern resembling a bullish-train day
#   - TabNet's per-row SPARSE feature selection (sparsemax mask) means each
#     row gets evaluated by a FEW features, not all of them — this is a
#     strong inductive bias against the dense-overfitting failure mode
#   - The mask-reuse penalty diversifies what features get selected, which
#     means the model has multiple "lenses" through which it evaluates a row,
#     similar in spirit to monotonic constraints but learned per-row
#
# Structurally different from registered families:
#   * iTransformer (variate-axis attention)   — DENSE feature mixing
#   * MLP / TorchAttentiveMLP                 — DENSE linear/attention mixing
#   * XGB / LightGBM / HistGB(monotonic)      — greedy tree splits, no per-row
#                                               feature selection
#   * Kernel logreg                           — fixed kernel, dense
#   * TabM / TabM_LCB                         — k-ensemble of dense MLPs
#   * KNN / QDA / GaussianNB                  — distance/density methods, all
#                                               features
#   * TimeMoE / GRU / Transformer (time-axis) — sequence models
# --------------------------------------------------------------------- #


def _tabnet_sparsemax(z: 'torch.Tensor', dim: int = -1) -> 'torch.Tensor':
    """Sparsemax (Martins & Astudillo, ICML 2016).

    Projection onto the probability simplex with sparsity — output is a
    non-negative distribution that sums to 1 along ``dim``, with many EXACT
    zeros. PyTorch autograd handles the backward via sort/cumsum/gather/clamp;
    the clamp's subgradient at 0 plus the sort indices being detached
    reproduce the analytic Jacobian from the paper.
    """
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim)
    n = z.size(dim)
    range_idx = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.ndim
    shape[dim] = n
    range_idx = range_idx.view(*shape)
    is_above = (1.0 + range_idx * z_sorted > z_cumsum).to(z.dtype)
    k = is_above.sum(dim=dim, keepdim=True).clamp(min=1.0)
    tau = (z_cumsum.gather(dim, k.long() - 1) - 1.0) / k
    return torch.clamp(z - tau, min=0.0)


class _TabNetGLU(nn.Module):
    """Gated linear unit: y = a · sigmoid(b), where [a; b] = BN(W x)."""

    def __init__(self, dim_in: int, dim_out: int, momentum: float = 0.02):
        super().__init__()
        self.fc = nn.Linear(dim_in, 2 * dim_out, bias=False)
        self.bn = nn.BatchNorm1d(2 * dim_out, momentum=momentum)

    def forward(self, x):
        h = self.bn(self.fc(x))
        a, b = h.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class _TabNetFeatureTransformer(nn.Module):
    """Per-step feature transformer: 2 shared GLU blocks + 2 step-specific
    GLU blocks, residual connections scaled by 1/sqrt(2) (TabNet appendix).
    """

    def __init__(self, shared_first: '_TabNetGLU',
                 shared_rest: 'nn.ModuleList',
                 dim_out: int, momentum: float = 0.02):
        super().__init__()
        # shared blocks are passed BY REFERENCE so they're truly shared
        # across all per-step feature transformers (param sharing).
        self.shared_first = shared_first
        self.shared_rest = shared_rest
        self.step = nn.ModuleList([
            _TabNetGLU(dim_out, dim_out, momentum=momentum) for _ in range(2)
        ])
        self.scale = 0.5 ** 0.5

    def forward(self, x):
        h = self.shared_first(x)
        for layer in self.shared_rest:
            h = (h + layer(h)) * self.scale
        for layer in self.step:
            h = (h + layer(h)) * self.scale
        return h


class _TabNetAttentiveTransformer(nn.Module):
    """Computes the sparsemax mask over input features for one decision step.

    Input ``a`` is the "attention" half of the previous step's output; ``prior``
    accumulates (gamma - mask) factors to discourage feature reuse across steps.
    """

    def __init__(self, dim_a: int, n_features: int, momentum: float = 0.02):
        super().__init__()
        self.fc = nn.Linear(dim_a, n_features, bias=False)
        self.bn = nn.BatchNorm1d(n_features, momentum=momentum)

    def forward(self, a, prior):
        h = self.bn(self.fc(a))
        return _tabnet_sparsemax(h * prior, dim=-1)


class _TabNetClassifier(nn.Module):
    """TabNet binary classifier (Arik & Pfister, AAAI 2021).

    Returns (logits, sparse_loss). sparse_loss is the per-step mask entropy
    averaged over steps — caller adds lambda_sparse * sparse_loss to BCE.
    """

    def __init__(self, n_features: int, n_d: int = 16, n_a: int = 16,
                 n_steps: int = 3, gamma: float = 1.3,
                 dropout: float = 0.0, momentum: float = 0.02):
        super().__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        dim_out = n_d + n_a
        # Input batchnorm.
        self.input_bn = nn.BatchNorm1d(n_features, momentum=momentum)
        # 2 shared GLU blocks. The first projects n_features -> dim_out;
        # the rest are dim_out -> dim_out.
        shared_first = _TabNetGLU(n_features, dim_out, momentum=momentum)
        shared_rest = nn.ModuleList([
            _TabNetGLU(dim_out, dim_out, momentum=momentum),
        ])
        # The initial feature transformer (no mask yet — uses ALL features).
        # Produces 'a' that seeds the first attention step. Its 'd' part is
        # discarded (decision sum starts at 0).
        self.initial_ft = _TabNetFeatureTransformer(
            shared_first, shared_rest, dim_out, momentum=momentum)
        # Per-step attentive + feature transformers (param-sharing across
        # steps via shared_first/shared_rest references).
        self.attn_transformers = nn.ModuleList([
            _TabNetAttentiveTransformer(n_a, n_features, momentum=momentum)
            for _ in range(n_steps)
        ])
        self.feature_transformers = nn.ModuleList([
            _TabNetFeatureTransformer(shared_first, shared_rest, dim_out,
                                      momentum=momentum)
            for _ in range(n_steps)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(n_d, 1)

    def forward(self, x):
        x = self.input_bn(x)
        # Initial step: get 'a' for the first attention without using a mask.
        h_init = self.initial_ft(x)
        a = h_init[:, self.n_d:]
        prior = torch.ones_like(x)
        decision_sum = x.new_zeros(x.size(0), self.n_d)
        sparse_loss = x.new_zeros(())
        for step in range(self.n_steps):
            mask = self.attn_transformers[step](a, prior)
            # Mask entropy penalty: encourages high entropy within a step
            # would push toward UNIFORM, which is the WRONG direction for
            # sparsity. The TabNet paper uses NEGATIVE entropy as the loss
            # contribution so minimizing it encourages LOW entropy (sparse).
            # We compute -sum(M log M) and ADD it -> minimizing increases it
            # (encourages uniform). To encourage SPARSITY we want to MINIMIZE
            # -(-mask*log(mask)) i.e. ADD +mask*log(mask). But mask is sparse
            # (from sparsemax) so log(mask) is undefined at 0. Standard
            # TabNet uses the entropy form below WITH the SIGN FLIPPED in
            # the paper's loss formula L_sparse = sum (-M log M / Nsteps)
            # added to the cross-entropy. Looking at the official Google
            # research codebase, the sparsity loss IS positive-entropy added,
            # making the loss higher when mask is uniform -> minimizer
            # pushes toward sparse mask. Keep the standard form.
            sparse_loss = sparse_loss + (
                -mask * torch.log(mask + 1e-15)
            ).sum(dim=-1).mean()
            prior = prior * (self.gamma - mask)
            masked_x = x * mask
            h = self.feature_transformers[step](masked_x)
            d_step = torch.relu(h[:, :self.n_d])
            a = h[:, self.n_d:]
            decision_sum = decision_sum + d_step
        logits = self.head(self.dropout(decision_sum)).squeeze(-1)
        return logits, sparse_loss / max(1, self.n_steps)


class TorchTabNetTrainer(BaseTrainer):
    """TabNet binary classifier with sparsemax per-row feature selection.

    Pure PyTorch (no external pytorch_tabnet dep). Default HPs follow the
    paper's "small dataset" recommendations (n_d=n_a=16, n_steps=3, gamma=1.3,
    lambda_sparse=1e-3). Operates on the 2D aggregated tabular features
    (consumes_sequences=False) -> same input shape as iTransformer, but with
    sparse per-row feature selection instead of dense variate attention.
    """

    name = 'torch_tabnet'
    consumes_sequences = False

    def __init__(self,
                 n_d: int = 16, n_a: int = 16, n_steps: int = 3,
                 gamma: float = 1.3, dropout: float = 0.1,
                 lambda_sparse: float = 1e-3,
                 lr: float = 2e-2, weight_decay: float = 1e-4,
                 batch_size: int = 256, epochs: int = 60, patience: int = 10,
                 momentum: float = 0.02,
                 pos_weight: 'float | None' = None,
                 device: 'str | None' = None,
                 seed: int = 42,
                 **kwargs):
        self.n_d = int(n_d)
        self.n_a = int(n_a)
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.dropout = float(dropout)
        self.lambda_sparse = float(lambda_sparse)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.momentum = float(momentum)
        self.pos_weight = pos_weight
        self.seed = int(seed)
        if device is None:
            device = 'cpu'
            try:
                if torch.cuda.is_available():
                    best_idx, best_free = -1, -1
                    for i in range(torch.cuda.device_count()):
                        try:
                            free, _ = torch.cuda.mem_get_info(i)
                        except Exception:
                            free = 0
                        if free > best_free:
                            best_free, best_idx = free, i
                    if best_free >= 1_000_000_000:
                        device = f'cuda:{best_idx}'
            except Exception:
                pass
        self.device = device
        self.model = None
        self.feature_mean_ = None
        self.feature_std_ = None
        self._n_features = None

    def _normalize(self, X, fit: bool = False):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X[:, -1, :]
        if fit:
            self.feature_mean_ = np.nanmean(X, axis=0).astype(np.float32)
            self.feature_std_ = (np.nanstd(X, axis=0) + 1e-6).astype(np.float32)
            self._n_features = X.shape[1]
        Xn = (X - self.feature_mean_) / self.feature_std_
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(Xn, -10.0, 10.0).astype(np.float32)

    def fit(self, X_tr, y_tr, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X_tr_p = self._normalize(X_tr, fit=True)
        X_val_p = self._normalize(X_val, fit=False)
        y_tr_a = np.asarray(y_tr, dtype=np.float32).reshape(-1)
        y_val_a = np.asarray(y_val, dtype=np.float32).reshape(-1)

        if len(set(y_tr_a.astype(int))) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        self.model = _TabNetClassifier(
            n_features=self._n_features,
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, dropout=self.dropout, momentum=self.momentum,
        ).to(self.device)
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # LR decay matching original TabNet schedule (0.9 every 20 epochs).
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.9)
        pos_w = None
        if self.pos_weight is not None:
            pos_w = torch.tensor(float(self.pos_weight), device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        X_tr_t = torch.from_numpy(X_tr_p)
        y_tr_t = torch.from_numpy(y_tr_a)
        X_val_t = torch.from_numpy(X_val_p).to(self.device)
        y_val_t = torch.from_numpy(y_val_a).to(self.device)
        ds = TensorDataset(X_tr_t, y_tr_t)
        bs = min(self.batch_size, len(ds))
        if bs < 2:
            bs = max(2, len(ds))
        # drop_last=True so BatchNorm always has >1 sample in a batch.
        dl = DataLoader(ds, batch_size=bs, shuffle=True,
                        drop_last=(len(ds) > bs))
        best_val = float('inf')
        best_state = None
        bad = 0
        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in dl:
                if xb.size(0) < 2:
                    continue  # BatchNorm requires batch > 1 in train mode
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits, sparse_loss = self.model(xb)
                loss = loss_fn(logits, yb) + self.lambda_sparse * sparse_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            scheduler.step()
            self.model.eval()
            with torch.no_grad():
                val_logits, _ = self.model(X_val_t)
                val_loss = float(loss_fn(val_logits, y_val_t).item())
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('Model not fit')
        X = self._normalize(X, fit=False)
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                xb = torch.from_numpy(X[i:i + self.batch_size]).to(self.device)
                logits, _ = self.model(xb)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                out.append(p)
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    @property
    def hyperparams(self):
        return dict(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, dropout=self.dropout,
            lambda_sparse=self.lambda_sparse,
            lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, epochs=self.epochs,
            patience=self.patience, momentum=self.momentum,
            pos_weight=self.pos_weight, seed=self.seed,
        )

    def feature_importance(self):
        return None

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(),
                       os.path.join(output_dir, 'model.pt'))
        cfg = dict(self.hyperparams)
        cfg['n_features'] = self._n_features
        cfg['name'] = self.name
        if extra:
            cfg.update(extra)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        if self.feature_mean_ is not None:
            np.save(os.path.join(output_dir, 'feature_mean.npy'),
                    self.feature_mean_)
            np.save(os.path.join(output_dir, 'feature_std.npy'),
                    self.feature_std_)
        return {'model': os.path.join(output_dir, 'model.pt'),
                'config': os.path.join(output_dir, 'config.json')}


# models/trainers.py — append at end (before TRAINERS dict)

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class _MambaRMSNorm(nn.Module):
    """RMSNorm — version-agnostic implementation (PyTorch <2.4 lacks nn.RMSNorm)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class _SelectiveSSM(nn.Module):
    """Mamba-1 selective state-space (S6) mixer.

    Discretized linear recurrence h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B_t·x_t with
    input-dependent (Δ_t, B_t, C_t). Sequential scan along the L axis in pure
    PyTorch — no CUDA kernel or external triton dependency.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.d_inner = int(expand) * int(d_model)
        self.dt_rank = max(1, math.ceil(self.d_model / 16))
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, self.d_conv,
            groups=self.d_inner, padding=self.d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            )
            inv_dt = dt + torch.log(-torch.expm1(-dt) + 1e-8)
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        x_ = x_.transpose(1, 2)
        x_ = self.conv1d(x_)[:, :, :l]
        x_ = x_.transpose(1, 2)
        x_ = F.silu(x_)
        x_dbl = self.x_proj(x_)
        dt_in, Bp, Cp = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_in))
        A = -torch.exp(self.A_log)
        deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = dt.unsqueeze(-1) * Bp.unsqueeze(2) * x_.unsqueeze(-1)
        h = x.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        for t in range(l):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            y_t = (h * Cp[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = y + x_ * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class _MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = _MambaRMSNorm(d_model)
        self.mixer = _SelectiveSSM(d_model, d_state, d_conv, expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mixer(self.norm(x)))


class _MambaTabularClassifier(nn.Module):
    """Tabular Mamba — each input feature is a token in a length-F pseudo-sequence.
    Selective SSM blocks scan along the variate axis with content-dependent
    (Δ, B, C); mean-pool over tokens, MLP head to a scalar logit.
    """

    def __init__(self, num_features, d_model=48, n_layers=2, d_state=12,
                 d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.tok_embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, num_features, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = _MambaRMSNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = x.unsqueeze(-1)
        h = self.tok_embed(x) + self.pos
        h = self.input_dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


class TorchMambaTrainer(BaseTrainer):
    """Mamba (selective state-space, S6) tabular classifier.

    Treats each of the ~96 input features as a token in a length-F pseudo-
    sequence and runs a stack of Mamba blocks across that axis. Attention-free,
    O(L) per layer, with input-dependent (Δ, B, C). Pure PyTorch — no CUDA
    kernel or triton dependency. Defaults are sized for CPU feasibility
    (d_model=48, n_layers=2, d_state=12); GPU users can scale up.
    """

    name = 'torch_mamba'
    consumes_sequences = False

    def __init__(self, d_model=48, n_layers=2, d_state=12, d_conv=4,
                 expand=2, dropout=0.1, learning_rate=2e-3, weight_decay=1e-4,
                 batch_size=256, epochs=25, patience=6, pos_weight=1.5,
                 device=None, **kwargs):
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.pos_weight = pos_weight
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_mean_ = None
        self.feature_std_ = None
        self.num_features_ = None

    def _normalize(self, X, fit=False):
        flat = X.reshape(-1, X.shape[-1])
        if fit:
            self.feature_mean_ = np.nanmean(flat, axis=0).astype(np.float32)
            self.feature_std_ = (np.nanstd(flat, axis=0) + 1e-6).astype(np.float32)
        Xn = (X - self.feature_mean_) / self.feature_std_
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return Xn

    def fit(self, X_tr, y_tr, X_val, y_val, **kwargs):
        X_tr = np.asarray(X_tr, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_tr = np.asarray(y_tr, dtype=np.float32).ravel()
        y_val = np.asarray(y_val, dtype=np.float32).ravel()
        if X_tr.ndim != 2:
            X_tr = X_tr.reshape(X_tr.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
        X_tr = self._normalize(X_tr, fit=True)
        X_val = self._normalize(X_val, fit=False)
        self.num_features_ = int(X_tr.shape[1])
        torch.manual_seed(42)
        self.model = _MambaTabularClassifier(
            num_features=self.num_features_,
            d_model=self.d_model, n_layers=self.n_layers,
            d_state=self.d_state, d_conv=self.d_conv,
            expand=self.expand, dropout=self.dropout,
        ).to(self.device)
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        pos_w = None
        if self.pos_weight is not None:
            pos_w = torch.tensor(float(self.pos_weight), device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)
        X_val_t = torch.from_numpy(X_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)
        best_val = float('inf')
        best_state = None
        bad_epochs = 0
        for _epoch in range(self.epochs):
            self.model.train()
            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            self.model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_count = 0
                for i in range(0, X_val_t.shape[0], self.batch_size):
                    xb = X_val_t[i:i + self.batch_size]
                    yb = y_val_t[i:i + self.batch_size]
                    val_loss_sum += float(loss_fn(self.model(xb), yb).item()) * int(xb.shape[0])
                    val_count += int(xb.shape[0])
                val_loss = val_loss_sum / max(val_count, 1)
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            X = X.reshape(X.shape[0], -1)
        Xn = self._normalize(X, fit=False)
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, Xn.shape[0], self.batch_size):
                xb = torch.from_numpy(Xn[i:i + self.batch_size]).to(self.device)
                logits = self.model(xb)
                probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
                out.append(probs)
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    def save(self, model_dir, extra=None):
        os.makedirs(model_dir, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(model_dir, 'model.pt'))
        cfg = {
            'name': self.name,
            'd_model': self.d_model, 'n_layers': self.n_layers,
            'd_state': self.d_state, 'd_conv': self.d_conv,
            'expand': self.expand, 'dropout': self.dropout,
            'learning_rate': self.learning_rate, 'weight_decay': self.weight_decay,
            'batch_size': self.batch_size, 'epochs': self.epochs,
            'patience': self.patience, 'pos_weight': self.pos_weight,
            'num_features': self.num_features_,
        }
        if extra:
            cfg.update(extra)
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        if self.feature_mean_ is not None:
            np.save(os.path.join(model_dir, 'feature_mean.npy'), self.feature_mean_)
            np.save(os.path.join(model_dir, 'feature_std.npy'), self.feature_std_)


# models/trainers.py — append at end (before TRAINERS dict)

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TorchPatchTSTTrainer(BaseTrainer):
    """PatchTST classifier — channel-independent patch transformer over the
    L=20 day axis (consumes_sequences=True path).

    Tokenization: each of the F per-day features is treated as an independent
    univariate series of length L. Each series is sliced into patches of
    length P with stride S, projected to d_model, and run through a shared
    Transformer encoder. A CLS token is prepended (use_cls_token=True) and
    the resulting CLS embedding is concatenated across channels and fed to
    a linear head with `num_targets` outputs. Forward returns
    `prediction_logits` of shape (batch, num_targets); we softmax over the
    class axis to produce predict_proba.

    Optional SSL masked pretraining stage: if ssl_pretrain_epochs > 0, a
    PatchTSTForPretraining model is trained first on the concatenation of
    train + val sequences with `random_mask_ratio` of patches masked out
    and reconstructed; the matched-shape backbone weights are then
    transferred to the classification head before fine-tuning.
    """

    name = 'torch_patchtst'
    consumes_sequences = True

    def __init__(self,
                 d_model=128,
                 num_hidden_layers=3,
                 num_attention_heads=4,
                 patch_length=4,
                 patch_stride=4,
                 ffn_dim=256,
                 channel_attention=True,
                 use_cls_token=True,
                 pooling_type='mean',
                 share_embedding=True,
                 dropout=0.1,
                 attention_dropout=0.1,
                 path_dropout=0.0,
                 learning_rate=1e-3,
                 weight_decay=1e-5,
                 batch_size=256,
                 epochs=20,
                 patience=5,
                 pos_weight=1.5,
                 ssl_pretrain_epochs=0,
                 random_mask_ratio=0.4,
                 device=None,
                 **kwargs):
        self.d_model = int(d_model)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.ffn_dim = int(ffn_dim)
        self.channel_attention = bool(channel_attention)
        self.use_cls_token = bool(use_cls_token)
        self.pooling_type = str(pooling_type)
        self.share_embedding = bool(share_embedding)
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)
        self.path_dropout = float(path_dropout)
        lr = float(learning_rate)
        # Defensive clamp: return_gate hard-codes learning_rate=0.05 (LightGBM-shaped);
        # an AdamW transformer at lr=5e-2 diverges. Treat anything > 1e-2 as a passthrough
        # artifact and use the AdamW-appropriate default instead.
        if lr > 1e-2:
            lr = 1e-3
        self.learning_rate = lr
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.pos_weight = pos_weight
        self.ssl_pretrain_epochs = int(ssl_pretrain_epochs)
        self.random_mask_ratio = float(random_mask_ratio)
        if device is None:
            device = 'cpu'
            try:
                if torch.cuda.is_available():
                    best_idx, best_free = -1, -1
                    for i in range(torch.cuda.device_count()):
                        try:
                            free, _ = torch.cuda.mem_get_info(i)
                        except Exception:
                            free = 0
                        if free > best_free:
                            best_free, best_idx = free, i
                    if best_free >= 1_500_000_000:
                        device = f'cuda:{best_idx}'
            except Exception:
                pass
        self.device = str(device)
        self.model = None
        self.num_channels_ = None
        self.context_length_ = None

    def _ensure_3d(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, :, None]
        if X.ndim != 3:
            raise ValueError(f'TorchPatchTSTTrainer expects 3D (N, L, F); got {X.shape}')
        return X

    def _build_config(self, num_channels, context_length, num_classes,
                      for_pretrain=False):
        from transformers import PatchTSTConfig
        p_len = max(1, min(self.patch_length, context_length))
        p_str = max(1, min(self.patch_stride, context_length))
        return PatchTSTConfig(
            num_input_channels=int(num_channels),
            context_length=int(context_length),
            patch_length=int(p_len),
            patch_stride=int(p_str),
            d_model=int(self.d_model),
            num_hidden_layers=int(self.num_hidden_layers),
            num_attention_heads=int(self.num_attention_heads),
            ffn_dim=int(self.ffn_dim),
            channel_attention=bool(self.channel_attention),
            use_cls_token=bool(self.use_cls_token),
            pooling_type=str(self.pooling_type),
            share_embedding=bool(self.share_embedding),
            attention_dropout=float(self.attention_dropout),
            ff_dropout=float(self.dropout),
            path_dropout=float(self.path_dropout),
            head_dropout=float(self.dropout),
            num_targets=int(num_classes),
            scaling=None,
            do_mask_input=bool(for_pretrain),
            mask_type='random',
            random_mask_ratio=float(self.random_mask_ratio),
        )

    def _pretrain_backbone(self, X_all):
        from transformers import PatchTSTForPretraining
        N, L, Fc = X_all.shape
        cfg = self._build_config(Fc, L, num_classes=2, for_pretrain=True)
        pre = PatchTSTForPretraining(cfg).to(self.device)
        opt = torch.optim.AdamW(
            pre.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        ds = TensorDataset(torch.from_numpy(X_all))
        dl = DataLoader(ds, batch_size=self.batch_size,
                        shuffle=True, drop_last=False)
        pre.train()
        for _ep in range(self.ssl_pretrain_epochs):
            for (xb,) in dl:
                xb = xb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = pre(past_values=xb)
                loss = out.loss
                if loss is None:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pre.parameters(), 1.0)
                opt.step()
        return pre.state_dict()

    def _transfer_backbone(self, src_state):
        dst_state = self.model.state_dict()
        n_copied = 0
        for k, v in src_state.items():
            if k in dst_state and dst_state[k].shape == v.shape:
                dst_state[k].copy_(v)
                n_copied += 1
        self.model.load_state_dict(dst_state)
        return n_copied

    def fit(self, X_tr, y_tr, X_val, y_val, **kwargs):
        from transformers import PatchTSTForClassification
        X_tr = self._ensure_3d(X_tr)
        X_val = self._ensure_3d(X_val)
        N, L, Fc = X_tr.shape
        self.num_channels_ = int(Fc)
        self.context_length_ = int(L)
        y_tr = np.asarray(y_tr).astype(np.int64).ravel()
        y_val = np.asarray(y_val).astype(np.int64).ravel()

        if self.ssl_pretrain_epochs > 0:
            X_all = np.concatenate([X_tr, X_val], axis=0)
            src_state = self._pretrain_backbone(X_all)
        else:
            src_state = None

        cfg = self._build_config(Fc, L, num_classes=2, for_pretrain=False)
        self.model = PatchTSTForClassification(cfg).to(self.device)
        if src_state is not None:
            self._transfer_backbone(src_state)

        if self.pos_weight is not None:
            pw = float(self.pos_weight)
        else:
            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            pw = max(n_neg / max(n_pos, 1), 0.5)
        class_w = torch.tensor([1.0, pw], dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_w)

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size,
                           shuffle=True, drop_last=False)
        X_val_t = torch.from_numpy(X_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)

        best_val = float('inf')
        best_state = None
        bad_epochs = 0
        for _epoch in range(self.epochs):
            self.model.train()
            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = self.model(past_values=xb)
                loss = loss_fn(out.prediction_logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            self.model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_count = 0
                for i in range(0, X_val_t.shape[0], self.batch_size):
                    xb = X_val_t[i:i + self.batch_size]
                    yb = y_val_t[i:i + self.batch_size]
                    out = self.model(past_values=xb)
                    val_loss_sum += float(
                        loss_fn(out.prediction_logits, yb).item()
                    ) * int(xb.shape[0])
                    val_count += int(xb.shape[0])
                val_loss = val_loss_sum / max(val_count, 1)
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().clone()
                              for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, X):
        X = self._ensure_3d(X)
        self.model.eval()
        out_chunks = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                xb = torch.from_numpy(X[i:i + self.batch_size]).to(self.device)
                out = self.model(past_values=xb)
                probs = torch.softmax(out.prediction_logits, dim=-1)[:, 1]
                out_chunks.append(probs.detach().cpu().numpy())
        if not out_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    def save(self, model_dir, extra=None):
        os.makedirs(model_dir, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(),
                       os.path.join(model_dir, 'model.pt'))
        cfg = {
            'name': self.name,
            'd_model': self.d_model,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'patch_length': self.patch_length,
            'patch_stride': self.patch_stride,
            'ffn_dim': self.ffn_dim,
            'channel_attention': self.channel_attention,
            'use_cls_token': self.use_cls_token,
            'pooling_type': self.pooling_type,
            'share_embedding': self.share_embedding,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'path_dropout': self.path_dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'pos_weight': self.pos_weight,
            'ssl_pretrain_epochs': self.ssl_pretrain_epochs,
            'random_mask_ratio': self.random_mask_ratio,
            'num_channels_': self.num_channels_,
            'context_length_': self.context_length_,
        }
        if extra:
            cfg.update(extra)
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)


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
    from tabicl import TabICLClassifier
    _HAS_TABICL = True
except Exception:
    _HAS_TABICL = False


class TabICLv2Trainer(BaseTrainer):
    """In-context tabular foundation model (Inria Soda TabICLv2, Apr 2026).

    A second-generation ICL foundation model that supersedes TabPFN-2.5 on
    TabArena/TALENT by 5-10% relative error at 5-10x lower inference cost.
    The crucial structural change vs `tabpfn_v25` is the Query-Aware
    Scalable Softmax (QASSMax) attention: element-wise temperature scaling
    that depends on context length n, so the softmax does not collapse as
    in-context training rows grow into the hundreds of thousands. That
    lifts the practical training-row ceiling from ~10k (TabPFN-2.5) toward
    1M, which means we can finally show the model the *full* per-window
    train slice (~150k rows) instead of stratified-subsampling it.

    No gradient updates at fit-time: stores the (optionally subsampled)
    training rows; the pretrained backbone performs in-context inference
    on each predict_proba call. Fit-time and predict-time wall-clock are
    both dominated by the single forward pass through the backbone.
    """
    name = 'tabicl_v2'
    consumes_sequences = False

    def __init__(self,
                 n_estimators: int = 1,
                 softmax_temperature: float = 0.9,
                 average_logits: bool = True,
                 norm_methods: str = 'none',
                 feat_shuffle_method: str = 'latin',
                 class_shuffle_method: str = 'shift',
                 outlier_threshold: float = 4.0,
                 batch_size: int = 1,
                 use_amp: bool = True,
                 use_fa3: bool = False,
                 kv_cache=False,
                 max_train_rows: int = 15000,
                 test_chunk_size: int = 2000,
                 offload_mode: str = 'cpu',
                 device: str = 'auto',
                 checkpoint_version: str = 'tabicl-classifier-v2-20260212.ckpt',
                 random_state: int = 42,
                 use_modal: bool = False,
                 modal_gpu: str = 'A100-40GB',
                 **_):
        # When use_modal=True we route fit+predict to scripts/modal_runner.py
        # so the local 12 GB GPU is bypassed entirely. The tabicl package only
        # needs to be importable for the local path.
        self.use_modal = bool(use_modal)
        self.modal_gpu = str(modal_gpu)
        if not self.use_modal and not _HAS_TABICL:
            raise ImportError(
                "tabicl not installed. `pip install tabicl` (Python >=3.10, "
                "PyTorch>=2.1). CUDA strongly recommended; CPU works with n_jobs."
            )
        self.n_estimators = int(n_estimators)
        self.softmax_temperature = float(softmax_temperature)
        self.average_logits = bool(average_logits)
        self.norm_methods = str(norm_methods)
        self.feat_shuffle_method = str(feat_shuffle_method)
        self.class_shuffle_method = str(class_shuffle_method)
        self.outlier_threshold = float(outlier_threshold)
        self.batch_size = int(batch_size)
        self.use_amp = bool(use_amp)
        self.use_fa3 = bool(use_fa3)
        # kv_cache accepts bool or str ('repr' = memory-efficient cache, True/'kv' = full).
        # 'repr' uses ~24x less GPU memory than 'kv' so it fits beside ollama on a 12 GB card.
        self.kv_cache = kv_cache
        self.max_train_rows = int(max_train_rows)
        self.test_chunk_size = int(test_chunk_size)
        self.offload_mode = str(offload_mode)
        self.requested_device = str(device)
        self.checkpoint_version = str(checkpoint_version)
        self.random_state = int(random_state)
        self._model = None
        self._X_train = None
        self._y_train = None
        self._device = None
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

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

    def _modal_hp(self) -> dict:
        return {
            'n_estimators': self.n_estimators,
            'softmax_temperature': self.softmax_temperature,
            'average_logits': self.average_logits,
            'norm_methods': self.norm_methods,
            'feat_shuffle_method': self.feat_shuffle_method,
            'class_shuffle_method': self.class_shuffle_method,
            'outlier_threshold': self.outlier_threshold,
            'batch_size': self.batch_size,
            'use_amp': self.use_amp,
            'use_fa3': self.use_fa3,
            'kv_cache': self.kv_cache,
            'offload_mode': self.offload_mode,
            'checkpoint_version': self.checkpoint_version,
        }

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, **kwargs):
        X_tr = self._sanitize(X_tr)
        y_tr = np.asarray(y_tr).astype(np.int64).ravel()
        X_tr, y_tr = self._stratified_subsample(X_tr, y_tr)
        # ICL model: fit() just stashes the support set; the actual forward
        # pass is in predict_proba. When use_modal=True we never build a
        # local TabICLClassifier — the support set ships to Modal at
        # predict time and the remote function does both fit and predict.
        self._X_train = X_tr
        self._y_train = y_tr
        if self.use_modal:
            self._device = f'modal:{self.modal_gpu}'
            self._model = 'modal'   # sentinel so predict_proba sees fit() ran
            return self
        import gc
        if self.requested_device == 'cpu':
            self._device = 'cpu'
        elif self.requested_device == 'cuda':
            self._device = 'cuda'
        else:
            self._device = 'cuda' if (_HAS_TORCH and torch.cuda.is_available()) else 'cpu'
        gc.collect()
        if _HAS_TORCH and self._device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self._model = TabICLClassifier(
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            average_logits=self.average_logits,
            norm_methods=self.norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=self.class_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            batch_size=self.batch_size,
            use_amp=self.use_amp,
            use_fa3=self.use_fa3,
            kv_cache=self.kv_cache,
            offload_mode=self.offload_mode,
            checkpoint_version=self.checkpoint_version,
            device=self._device,
            random_state=self.random_state,
            allow_auto_download=True,
        )
        self._model.fit(X_tr, y_tr)
        return self

    def _predict_via_modal(self, X) -> np.ndarray:
        import time
        import modal
        from scripts.modal_budget import (
            check_budget, record_call, estimate_cost,
        )
        # Conservative pre-flight estimate: 60 s per call covers all
        # n_estimators ∈ [1,4] sweeps without false-positive budget trips.
        check_budget(estimated_duration_s=60.0, gpu=self.modal_gpu)
        fn = modal.Function.from_name('caffe-stocks-modal', 'train_predict_tabicl')
        t0 = time.monotonic()
        try:
            proba = fn.remote(
                self._X_train, self._y_train, X, self._modal_hp(),
                self.random_state,
            )
            duration = time.monotonic() - t0
            record_call(
                trainer=self.name, duration_s=duration, gpu=self.modal_gpu,
                status='ok',
            )
        except Exception:
            duration = time.monotonic() - t0
            record_call(
                trainer=self.name, duration_s=duration, gpu=self.modal_gpu,
                status='error',
            )
            raise
        proba = np.asarray(proba, dtype=np.float32)
        return proba.ravel().astype(np.float32)

    def predict_proba(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TabICLv2Trainer: predict_proba called before fit()")
        X = self._sanitize(X)
        if self.use_modal:
            return self._predict_via_modal(X)
        if _HAS_TORCH and self._device == 'cuda':
            torch.cuda.empty_cache()
        n = X.shape[0]
        chunk = self.test_chunk_size
        # Chunked test-set inference: reuses the KV cache built in fit() across
        # mini-batches so wall-time scales linearly with test size and peak VRAM
        # stays bounded by chunk × cache-size. test_chunk_size=0 disables chunking.
        if chunk > 0 and n > chunk:
            parts = []
            for i in range(0, n, chunk):
                part = self._model.predict_proba(X[i:i + chunk])
                parts.append(np.asarray(part, dtype=np.float32))
                if _HAS_TORCH and self._device == 'cuda':
                    torch.cuda.empty_cache()
            proba = np.concatenate(parts, axis=0)
        else:
            proba = self._model.predict_proba(X)
        if _HAS_TORCH and self._device == 'cuda':
            torch.cuda.empty_cache()
        proba = np.asarray(proba, dtype=np.float32)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float32)
        return proba.ravel().astype(np.float32)

    def save(self, model_dir, extra=None):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'tabicl_state.pkl'), 'wb') as f:
            pickle.dump({
                'X_train': self._X_train,
                'y_train': self._y_train,
                'hp': {
                    'n_estimators': self.n_estimators,
                    'softmax_temperature': self.softmax_temperature,
                    'average_logits': self.average_logits,
                    'norm_methods': self.norm_methods,
                    'feat_shuffle_method': self.feat_shuffle_method,
                    'class_shuffle_method': self.class_shuffle_method,
                    'outlier_threshold': self.outlier_threshold,
                    'batch_size': self.batch_size,
                    'use_amp': self.use_amp,
                    'kv_cache': self.kv_cache,
                    'max_train_rows': self.max_train_rows,
                    'checkpoint_version': self.checkpoint_version,
                    'random_state': self.random_state,
                },
                'device': self._device,
            }, f)
        meta = {'trainer': self.name, 'device': self._device,
                'checkpoint_version': self.checkpoint_version}
        if extra:
            meta.update(extra)
        with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        return {
            'state': os.path.join(model_dir, 'tabicl_state.pkl'),
            'meta': os.path.join(model_dir, 'meta.json'),
        }

    @property
    def hyperparams(self) -> dict:
        return {
            'n_estimators': self.n_estimators,
            'softmax_temperature': self.softmax_temperature,
            'average_logits': self.average_logits,
            'norm_methods': self.norm_methods,
            'feat_shuffle_method': self.feat_shuffle_method,
            'class_shuffle_method': self.class_shuffle_method,
            'outlier_threshold': self.outlier_threshold,
            'batch_size': self.batch_size,
            'use_amp': self.use_amp,
            'use_fa3': self.use_fa3,
            'kv_cache': self.kv_cache,
            'max_train_rows': self.max_train_rows,
            'test_chunk_size': self.test_chunk_size,
            'offload_mode': self.offload_mode,
            'checkpoint_version': self.checkpoint_version,
        }


# models/trainers.py — append at end (before TRAINERS dict)

import os
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class TorchChronos2EmbedTrainer(BaseTrainer):
    """Frozen Chronos-2 encoder → mean-pooled embedding → linear head.

    Consumes the existing L=20 close-price sequence (univariate).
    Embeddings are computed once per fit() call in batches, optionally
    concatenated with the tabular X, then a LogisticRegression head is
    fit on (embedding [+ X]) -> y. Encoder is never updated.
    """

    name = 'torch_chronos2'
    consumes_sequences = True

    def __init__(
        self,
        chronos_model: str = 'amazon/chronos-2',
        seq_channel: str = 'close',
        embedding_pool: str = 'mean',
        concat_tabular: bool = True,
        head_C: float = 1.0,
        head_max_iter: int = 2000,
        batch_size: int = 256,
        device: str = None,
        local_weights_dir: str = 'models/checkpoints/chronos2',
        **kwargs,
    ):
        self.chronos_model = chronos_model
        self.seq_channel = seq_channel
        self.embedding_pool = embedding_pool
        self.concat_tabular = concat_tabular
        self.head_C = head_C
        self.head_max_iter = head_max_iter
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_weights_dir = local_weights_dir
        self.pipeline_ = None
        self.scaler_ = None
        self.head_ = None
        self.emb_dim_ = None

    def _load_pipeline(self):
        if self.pipeline_ is not None:
            return
        from chronos import Chronos2Pipeline
        src = self.local_weights_dir if os.path.isdir(self.local_weights_dir) else self.chronos_model
        self.pipeline_ = Chronos2Pipeline.from_pretrained(src, device_map=self.device)

    def _extract_channel(self, seq: np.ndarray) -> np.ndarray:
        # seq: (N, L, F) — pick the channel matching self.seq_channel.
        # Convention: channel 0 = close (claude_mode wires the actual mapping
        # via sequence_loader.SEQUENCE_FEATURE_COLUMNS). Fallback: column 0.
        if seq.ndim == 3:
            return seq[:, :, 0].astype(np.float32)
        if seq.ndim == 2:
            return seq.astype(np.float32)
        raise ValueError(f"sequence ndim must be 2 or 3, got {seq.ndim}")

    def _embed(self, univariate: np.ndarray) -> np.ndarray:
        # univariate: (N, L) — returns (N, D).
        self._load_pipeline()
        outs = []
        N = univariate.shape[0]
        for i in range(0, N, self.batch_size):
            batch = univariate[i:i + self.batch_size]
            tensors = [torch.from_numpy(row) for row in batch]
            with torch.no_grad():
                result = self.pipeline_.embed(tensors, batch_size=self.batch_size)
            # Chronos-2 v2.2.x: returns (list[Tensor(1,T,D)], list[aux]).
            if isinstance(result, tuple) and len(result) == 2:
                embs_list = result[0]
            else:
                embs_list = result
            arr = np.stack([self._pool(e) for e in embs_list], axis=0)
            outs.append(arr)
        return np.concatenate(outs, axis=0).astype(np.float32)

    def _pool(self, e):
        e = e.detach().cpu().numpy() if hasattr(e, 'detach') else np.asarray(e)
        if e.ndim == 3 and e.shape[0] == 1:
            e = e[0]
        if e.ndim == 1:
            return e
        if self.embedding_pool == 'mean':
            return e.mean(axis=0)
        if self.embedding_pool == 'last':
            return e[-1]
        if self.embedding_pool == 'max':
            return e.max(axis=0)
        return e.mean(axis=0)

    def _build_features(self, seq):
        # seq: (N, L, F). Embed channel-0 univariate, optionally concat
        # aggregated tabular (last/mean/std/dev across the L window) of all F.
        seq = np.asarray(seq, dtype=np.float32)
        uni = self._extract_channel(seq)
        emb = self._embed(uni)
        if self.concat_tabular:
            last = seq[:, -1, :]
            mean = seq.mean(axis=1)
            std = seq.std(axis=1)
            tab = np.concatenate([last, mean, std, last - mean], axis=1)
            tab = np.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
            return np.concatenate([emb, tab], axis=1)
        return emb

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        if not hasattr(X_train, 'ndim') or X_train.ndim != 3:
            raise ValueError(
                f'torch_chronos2 expects 3D sequence input (N, L, F); got '
                f'shape {getattr(X_train, "shape", None)}. '
                f'consumes_sequences=True must thread X_seq.')
        Z_tr = self._build_features(X_train)
        self.scaler_ = StandardScaler().fit(Z_tr)
        Z_tr_s = self.scaler_.transform(Z_tr)
        self.emb_dim_ = Z_tr_s.shape[1]
        self.head_ = LogisticRegression(
            C=self.head_C, max_iter=self.head_max_iter, solver='lbfgs', n_jobs=1,
        )
        self.head_.fit(Z_tr_s, np.asarray(y_train).astype(int))
        return self

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        if not hasattr(X, 'ndim') or X.ndim != 3:
            raise ValueError(
                f'torch_chronos2.predict_proba expects 3D sequence input; got '
                f'shape {getattr(X, "shape", None)}.')
        Z = self._build_features(X)
        Z_s = self.scaler_.transform(Z)
        return self.head_.predict_proba(Z_s)[:, 1].astype(np.float32)

    def save(self, model_dir, extra=None):
        os.makedirs(model_dir, exist_ok=True)
        import joblib
        joblib.dump(self.head_, os.path.join(model_dir, 'head.joblib'))
        joblib.dump(self.scaler_, os.path.join(model_dir, 'scaler.joblib'))
        meta = {
            'name': self.name,
            'chronos_model': self.chronos_model,
            'seq_channel': self.seq_channel,
            'embedding_pool': self.embedding_pool,
            'concat_tabular': self.concat_tabular,
            'head_C': self.head_C,
            'batch_size': self.batch_size,
            'emb_dim': self.emb_dim_,
        }
        if extra:
            meta.update(extra)
        with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)


# --------------------------------------------------------------------- #
# AnomalyGatedHistGBTrainer (iter #1499 claude_mode, brief-exhausted pivot)
#
# Brief said torch_chronos2 is exhausted (2 attempts, best wp=1/7). Part A
# of this iter confirms W5 is a 0/21 wall across the last 21 iterations and
# W3/W4 fail in 20/21 and 16/21 respectively; W5 is the deep-bear regime
# (SET -15.1% / 19% vol / 31% breadth). Every model family — XGB-loss
# variants, transformer-from-scratch, foundation-model encoders — converges
# on the same failure mode: ~30 trades in W5 at ~25% WR with ~25% DD. The
# structural shared cause is distribution shift: bull-heavy training data
# does not contain enough W5-like deep-bear feature patterns, so all
# trainers learn over-confident classifiers that fire too often on test
# data that looks unlike training.
#
# Family in scope: anomaly-gated supervised classification — not present in
# the current registry. IsolationForest fit on training X learns the typical
# training-feature regime. At inference, samples that fall in the low
# quantile of IF score (relative to training) are flagged as OOD and have
# their classifier proba damped toward 0. This produces a natural,
# data-driven abstention behaviour without changing the underlying label
# definition (preserves the gate's contract).
#
# Base classifier reuses HistGBMonotonicTrainer's monotonic constraint set
# (iter #936's 4/7 winning architecture). This concentrates the novelty on
# the gating mechanism rather than the supervised loss surface.
# --------------------------------------------------------------------- #
class AnomalyGatedHistGBTrainer(BaseTrainer):
    """IsolationForest-gated HistGB classifier for OOD abstention on bear windows.

    Iter #1675 addition: ``day_abstain_q`` adds a *date-level* abstention floor
    on top of the existing per-row IF gate.

    Iter #1723 addition: ``anomaly_as_feature`` feeds the per-row IF anomaly
    score as an ADDITIONAL HistGB feature (with monotonic +1 constraint —
    more normal → higher win probability). Before, IF was used only as a
    post-hoc multiplicative gate; now the model can learn NONLINEAR
    INTERACTIONS between anomaly and other features (e.g., "high atr_pct
    AND high anomaly → reject"). This addresses the failure mode of
    torch_timesfm (iter #1721/#1722), where regime-uniform input produced
    score collapse on bear windows — feeding an explicit per-row
    anomaly signal as a tree feature gives the classifier within-day
    dispersion that the multiplicative gate alone cannot produce."""

    name = 'anomaly_gated_histgb'

    def __init__(self,
                 # IsolationForest knobs
                 if_n_estimators: int = 200,
                 if_max_samples: float = 0.5,
                 if_contamination: float = 0.1,
                 if_random_state: int = 42,
                 # Gating: rows below `gate_threshold` quantile of training IF
                 # scores get partial damping (linear ramp to 0 at the most-
                 # anomalous training point). alpha controls damp aggression.
                 gate_threshold: float = 0.255,
                 gating_alpha: float = 0.693,
                 # Day-level abstention: zero out all rows on test dates whose
                 # mean IF score lies below this quantile of training-day means.
                 # 0.0 disables. Iter #1675.
                 day_abstain_q: float = 0.25,
                 # Anomaly score concatenated as HistGB feature (iter #1723).
                 # Adds 1 column to X with monotonic +1 constraint (higher IF
                 # score = more normal = higher win prob).
                 anomaly_as_feature: bool = True,
                 # HistGB knobs (default to iter #1527 best 5/7 HPs)
                 max_iter: int = 400,
                 max_leaf_nodes: int = 31,
                 max_depth: Optional[int] = None,
                 learning_rate: float = 0.05,
                 min_samples_leaf: int = 50,
                 l2_regularization: float = 1.0,
                 max_bins: int = 255,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.15,
                 n_iter_no_change: int = 20,
                 tol: float = 1e-4,
                 pos_class_weight: float = 3.371,
                 use_monotonic: bool = True,
                 if_contamination_default_keep: bool = False,  # accept legacy kw
                 random_state: int = 42,
                 **_):
        self._params = dict(
            if_n_estimators=int(if_n_estimators),
            if_max_samples=float(if_max_samples),
            if_contamination=float(if_contamination),
            if_random_state=int(if_random_state),
            gate_threshold=float(gate_threshold),
            gating_alpha=float(gating_alpha),
            day_abstain_q=float(day_abstain_q),
            anomaly_as_feature=bool(anomaly_as_feature),
            max_iter=int(max_iter),
            max_leaf_nodes=int(max_leaf_nodes),
            max_depth=None if max_depth in (None, 0, -1) else int(max_depth),
            learning_rate=float(learning_rate),
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=float(l2_regularization),
            max_bins=int(max_bins),
            early_stopping=bool(early_stopping),
            validation_fraction=float(validation_fraction),
            n_iter_no_change=int(n_iter_no_change),
            tol=float(tol),
            pos_class_weight=float(pos_class_weight),
            use_monotonic=bool(use_monotonic),
            random_state=int(random_state),
        )
        self.iso_ = None
        self.histgb_ = None
        self._train_anom_sorted = None
        self._train_day_anom_sorted = None
        self._predict_dates = None
        self._n_features = None
        self._monotonic_cst = None

    def _build_monotonic_constraints(self, n_features: int) -> Optional[np.ndarray]:
        if not self._params['use_monotonic']:
            return None
        try:
            from models.feature_eng import CURATED_FEATURES
        except Exception:
            return None
        F = len(CURATED_FEATURES)
        extra = 1 if self._params.get('anomaly_as_feature', False) else 0
        if n_features != 4 * F + extra:
            return None
        cst = np.zeros(n_features, dtype=np.int8)
        name_to_idx = {n: i for i, n in enumerate(CURATED_FEATURES)}
        for name in HistGBMonotonicTrainer._MONOTONIC_INCREASING:
            i = name_to_idx.get(name)
            if i is None:
                continue
            cst[i] = 1
            cst[F + i] = 1
        if extra:
            # The anomaly column is appended at the end. Higher IF
            # score = more normal = higher predicted win probability.
            cst[-1] = 1
        return cst

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        # Fit IsolationForest on training distribution. Lower max_samples
        # increases tree diversity and produces a smoother quantile rank.
        self.iso_ = IsolationForest(
            n_estimators=p['if_n_estimators'],
            max_samples=p['if_max_samples'],
            contamination=p['if_contamination'],
            random_state=p['if_random_state'],
            n_jobs=-1,
        )
        self.iso_.fit(X_full)
        # Sorted training anomaly scores → used for quantile-rank lookup at
        # inference. score_samples returns higher = more normal.
        train_anom = self.iso_.score_samples(X_full).astype(np.float32)
        self._train_anom_sorted = np.sort(train_anom)

        if p.get('anomaly_as_feature', False):
            X_full_fit = np.hstack(
                [X_full, train_anom.reshape(-1, 1).astype(X_full.dtype)])
        else:
            X_full_fit = X_full
        self._n_features = X_full_fit.shape[1]
        self._monotonic_cst = self._build_monotonic_constraints(self._n_features)

        # Per-training-date mean IF score → distribution of "how OOD is the
        # whole day's cross-section". Used by the day-level abstention gate at
        # predict time. Concatenate dates from inner-train + inner-val (full
        # training span) so the day quantile reflects all training cohorts.
        if p['day_abstain_q'] > 0 and dates_train is not None and dates_val is not None:
            try:
                d_full = np.concatenate([np.asarray(dates_train), np.asarray(dates_val)])
                if len(d_full) == len(train_anom):
                    uniq = np.unique(d_full)
                    day_means = np.empty(len(uniq), dtype=np.float32)
                    for i, d in enumerate(uniq):
                        day_means[i] = float(np.mean(train_anom[d_full == d]))
                    self._train_day_anom_sorted = np.sort(day_means)
            except Exception:
                self._train_day_anom_sorted = None

        sw = np.where(y_full == 1, p['pos_class_weight'], 1.0).astype(np.float32)
        self.histgb_ = HistGradientBoostingClassifier(
            max_iter=p['max_iter'],
            max_leaf_nodes=p['max_leaf_nodes'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            min_samples_leaf=p['min_samples_leaf'],
            l2_regularization=p['l2_regularization'],
            max_bins=p['max_bins'],
            early_stopping=p['early_stopping'],
            validation_fraction=p['validation_fraction'] if p['early_stopping'] else None,
            n_iter_no_change=p['n_iter_no_change'],
            tol=p['tol'],
            monotonic_cst=self._monotonic_cst,
            random_state=p['random_state'],
            verbose=0,
        )
        self.histgb_.fit(X_full_fit, y_full, sample_weight=sw)
        return self

    def _gate(self, X) -> np.ndarray:
        anom = self.iso_.score_samples(X).astype(np.float32)
        n = len(self._train_anom_sorted)
        # Quantile rank vs training distribution: 0 = more anomalous than
        # any training row, 1 = more normal than any training row.
        ranks = np.searchsorted(self._train_anom_sorted, anom, side='right').astype(np.float32) / max(n, 1)
        thr = self._params['gate_threshold']
        # Linear ramp: rank>=thr → gate=1.0, rank=0 → gate=0.0.
        if thr > 0:
            gate = np.clip(ranks / thr, 0.0, 1.0)
        else:
            gate = np.ones_like(ranks)
        alpha = self._params['gating_alpha']
        if alpha != 1.0:
            gate = gate ** alpha
        return gate

    def set_predict_context(self, dates):
        self._predict_dates = (
            np.asarray(dates) if dates is not None else None)

    def predict_proba(self, X) -> np.ndarray:
        if self.histgb_ is None:
            raise RuntimeError('Model not fit')
        anom_test = self.iso_.score_samples(X).astype(np.float32)
        if self._params.get('anomaly_as_feature', False):
            X_pred = np.hstack([X, anom_test.reshape(-1, 1).astype(X.dtype)])
        else:
            X_pred = X
        p_clf = self.histgb_.predict_proba(X_pred)[:, 1].astype(np.float32)
        gate = self._gate(X)
        scores = (p_clf * gate).astype(np.float32)

        q = self._params.get('day_abstain_q', 0.0)
        if (q > 0 and self._predict_dates is not None
                and self._train_day_anom_sorted is not None
                and len(self._train_day_anom_sorted) > 0
                and len(self._predict_dates) == len(scores)):
            d = np.asarray(self._predict_dates)
            sorted_day = self._train_day_anom_sorted
            n_day = len(sorted_day)
            for u in np.unique(d):
                mask = d == u
                day_mean = float(np.mean(anom_test[mask]))
                rank_q = np.searchsorted(sorted_day, day_mean, side='right') / n_day
                if rank_q < q:
                    scores[mask] = 0.0
        return scores

    def feature_importance(self):
        return None

    @property
    def best_iteration(self):
        if self.histgb_ is None:
            return None
        return int(getattr(self.histgb_, 'n_iter_', self._params['max_iter']))

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'iso': self.iso_,
                'histgb': self.histgb_,
                'train_anom_sorted': self._train_anom_sorted,
                'train_day_anom_sorted': self._train_day_anom_sorted,
            }, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'n_features': self._n_features,
            'monotonic_cst': None if self._monotonic_cst is None
                              else [int(v) for v in self._monotonic_cst.tolist()],
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'model.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        cst = meta.get('monotonic_cst')
        inst._monotonic_cst = None if cst is None else np.array(cst, dtype=np.int8)
        with open(model_path, 'rb') as f:
            blob = pickle.load(f)
        inst.iso_ = blob['iso']
        inst.histgb_ = blob['histgb']
        inst._train_anom_sorted = blob['train_anom_sorted']
        inst._train_day_anom_sorted = blob.get('train_day_anom_sorted')
        return inst


# --------------------------------------------------------------------- #
# Denoising Autoencoder + LogReg classifier (iter #1530)
#
# Pivot from torch_chronos2 (exhausted, 2 attempts at wp=1). The brief's
# foundation-model angle was "pretrained representation → linear head"; this
# is the same pattern but with the encoder pretrained on OUR data via
# denoising reconstruction — sidesteps the OOM-on-12GB-RTX-5070 issue with
# Chronos-2 weights and gives a representation tuned to SET tabular features
# rather than univariate close-price sequences.
#
# Three signals feed the LR head:
#   1. Original X (24-d) — preserves linear baseline
#   2. Bottleneck embedding (low-d non-linear regime-aware compression)
#   3. Per-row reconstruction error — OOD signal (cf. anomaly_gated_histgb's
#      IsolationForest, but learned rather than density-based)
#
# Failure-pattern motivation: W5 100% fail across last 29 iters, W3/W4 76%.
# These are bear-regime test slices following bull-train; a learned encoder
# trained on train+val + reconstruction-error OOD flag may distinguish
# regime-transition rows from in-distribution rows better than the
# IsolationForest density estimate used by anomaly_gated_histgb.
# --------------------------------------------------------------------- #
class DAELogRegTrainer(BaseTrainer):
    """Denoising Autoencoder pretraining + LogReg head on [X | z | recon_err]."""

    name = 'dae_logreg'

    def __init__(self,
                 bottleneck_dim: int = 8,
                 hidden_dim: int = 32,
                 noise_std: float = 0.15,
                 dropout: float = 0.20,
                 ae_learning_rate: float = 1e-3,
                 ae_weight_decay: float = 1e-5,
                 ae_batch_size: int = 512,
                 ae_max_epochs: int = 25,
                 ae_patience: int = 4,
                 head_C: float = 1.0,
                 head_max_iter: int = 2000,
                 include_recon_err: int = 1,
                 include_raw_x: int = 1,
                 pos_class_weight: float = 1.5,
                 # Anomaly-conditional abstention (iter #1530 lesson option b).
                 # The DAE's own per-row reconstruction error doubles as an
                 # OOD signal: rows whose features cannot be reconstructed by
                 # the learned bottleneck are by definition out of the train
                 # distribution and get probabilistically demoted by an
                 # exponential multiplier. Differs structurally from
                 # AnomalyGatedHistGB (IsolationForest + HistGB) by using the
                 # SAME network that produced the bottleneck z to also score
                 # anomaly, so abstention and representation are coupled.
                 # abstain_strength=0.0 ⇒ off (legacy behaviour). The
                 # threshold is the abstain_recon_q quantile of training
                 # recon_err; only rows ABOVE it are demoted.
                 abstain_strength: float = 2.0,
                 abstain_recon_q: float = 0.70,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            bottleneck_dim=int(bottleneck_dim),
            hidden_dim=int(hidden_dim),
            noise_std=float(noise_std),
            dropout=float(dropout),
            ae_learning_rate=float(ae_learning_rate),
            ae_weight_decay=float(ae_weight_decay),
            ae_batch_size=int(ae_batch_size),
            ae_max_epochs=int(ae_max_epochs),
            ae_patience=int(ae_patience),
            head_C=float(head_C),
            head_max_iter=int(head_max_iter),
            include_recon_err=int(include_recon_err),
            include_raw_x=int(include_raw_x),
            pos_class_weight=float(pos_class_weight),
            abstain_strength=float(abstain_strength),
            abstain_recon_q=float(np.clip(abstain_recon_q, 0.10, 0.95)),
            random_state=int(random_state),
        )
        self.scaler_ = None
        self.ae_ = None
        self.head_ = None
        self._n_features = None
        self._best_ae_epoch = None
        self._recon_err_thresh = None

    def _build_ae(self, n_features):
        import torch
        import torch.nn as nn
        p = self._params

        class DAE(nn.Module):
            def __init__(self, F, H, B, dp):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Linear(F, H), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(H, B), nn.GELU(),
                )
                self.dec = nn.Sequential(
                    nn.Linear(B, H), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(H, F),
                )

            def forward(self, x_noisy):
                z = self.enc(x_noisy)
                return self.dec(z), z

        return DAE(n_features, p['hidden_dim'],
                   p['bottleneck_dim'], p['dropout'])

    def _features_and_err(self, X_scaled):
        """Return ([X | z | recon_err] features, raw_recon_err vector)."""
        import torch
        p = self._params
        device = next(self.ae_.parameters()).device
        X_t = torch.from_numpy(np.asarray(X_scaled, dtype=np.float32)).to(device)
        self.ae_.eval()
        with torch.no_grad():
            recon, z = self.ae_(X_t)
            err = ((recon - X_t) ** 2).mean(dim=1, keepdim=True)
        err_np = err.cpu().numpy()
        feats = [z.cpu().numpy()]
        if p['include_raw_x']:
            feats.insert(0, np.asarray(X_scaled, dtype=np.float64))
        if p['include_recon_err']:
            feats.append(err_np)
        return (np.concatenate(feats, axis=1).astype(np.float64),
                err_np.reshape(-1).astype(np.float64))

    def _features(self, X_scaled):
        return self._features_and_err(X_scaled)[0]

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.manual_seed(p['random_state'])
        np.random.seed(p['random_state'])

        # Concat train+val for AE pretraining (semi-supervised use of all rows),
        # but the LR head still fits on the train+val combined (matches
        # logistic_elastic_net / kernel_logreg behavior).
        X_full_raw = np.vstack([np.asarray(X_train, dtype=np.float32),
                                np.asarray(X_val, dtype=np.float32)])
        y_full = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
        self._n_features = X_full_raw.shape[1]

        self.scaler_ = StandardScaler(with_mean=True, with_std=True)
        X_full = self.scaler_.fit_transform(X_full_raw).astype(np.float32)
        X_full = np.clip(X_full, -8.0, 8.0)

        # ---- Phase 1: pretrain DAE on X_full (no labels) ----
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ae_ = self._build_ae(self._n_features).to(device)
        opt = torch.optim.AdamW(self.ae_.parameters(),
                                lr=p['ae_learning_rate'],
                                weight_decay=p['ae_weight_decay'])
        # Held-out 10% slice for AE early-stop on reconstruction loss
        n = X_full.shape[0]
        rs = np.random.RandomState(p['random_state'])
        idx = rs.permutation(n)
        n_val = max(1, n // 10)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        Xtr = torch.from_numpy(X_full[tr_idx])
        Xva = torch.from_numpy(X_full[val_idx]).to(device)

        ds = TensorDataset(Xtr)
        dl = DataLoader(ds, batch_size=p['ae_batch_size'], shuffle=True)
        best_loss = float('inf')
        best_state = None
        bad = 0
        for epoch in range(p['ae_max_epochs']):
            self.ae_.train()
            for (xb,) in dl:
                xb = xb.to(device, non_blocking=True)
                # Gaussian noise on input; clean target.
                noise = torch.randn_like(xb) * p['noise_std']
                opt.zero_grad(set_to_none=True)
                recon, _ = self.ae_(xb + noise)
                loss = nn.functional.mse_loss(recon, xb)
                loss.backward()
                opt.step()
            # Val recon loss (clean inputs both ways)
            self.ae_.eval()
            with torch.no_grad():
                recon_v, _ = self.ae_(Xva)
                vloss = nn.functional.mse_loss(recon_v, Xva).item()
            if verbose:
                print(f'  AE ep{epoch:02d} val_mse={vloss:.4f}')
            if vloss < best_loss - 1e-5:
                best_loss = vloss
                best_state = {k: v.detach().clone() for k, v in
                              self.ae_.state_dict().items()}
                self._best_ae_epoch = epoch
                bad = 0
            else:
                bad += 1
                if bad >= p['ae_patience']:
                    break
        if best_state is not None:
            self.ae_.load_state_dict(best_state)

        # ---- Phase 2: build features and fit LR head ----
        F_full, err_full = self._features_and_err(X_full)
        # class_weight balanced w/ pos boost
        cw = {0: 1.0, 1: float(p['pos_class_weight'])}
        self.head_ = LogisticRegression(
            C=p['head_C'],
            max_iter=p['head_max_iter'],
            class_weight=cw,
            random_state=p['random_state'],
            solver='lbfgs',
        )
        self.head_.fit(F_full, y_full)
        # Record the abstention threshold from training recon_err so that
        # predict-time gating is anchored on the in-distribution mass.
        self._recon_err_thresh = float(np.quantile(err_full, p['abstain_recon_q']))
        if verbose:
            print(f'  abstain_recon_q={p["abstain_recon_q"]:.2f} '
                  f'thresh={self._recon_err_thresh:.4f} '
                  f'(p99={float(np.quantile(err_full, 0.99)):.4f})')
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.ae_ is None or self.head_ is None or self.scaler_ is None:
            raise RuntimeError('Model not fit')
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler_.transform(X).astype(np.float32)
        Xs = np.clip(Xs, -8.0, 8.0)
        F, err = self._features_and_err(Xs)
        p_raw = self.head_.predict_proba(F)[:, 1]
        strength = float(self._params.get('abstain_strength', 0.0))
        thresh = self._recon_err_thresh
        if strength > 0.0 and thresh is not None and thresh > 0.0:
            # Exponential demote only ABOVE the training-quantile threshold.
            # excess > 0 ⇒ row is in the tail the DAE could not reconstruct
            # ⇒ likely OOD ⇒ exp(-strength · excess) shrinks the prob.
            excess = np.maximum(0.0, (err - thresh) / thresh)
            mult = np.exp(-strength * excess)
            p_raw = p_raw * mult
        return p_raw

    def feature_importance(self):
        return None

    @property
    def best_iteration(self):
        return self._best_ae_epoch

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        import torch
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pt')
        head_path = os.path.join(output_dir, 'head.pkl')
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        torch.save({'state_dict': self.ae_.state_dict(),
                    'n_features': self._n_features}, model_path)
        with open(head_path, 'wb') as f:
            pickle.dump(self.head_, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler_, f)
        meta = {
            'trainer': self.name,
            'hyperparams': self._params,
            'best_ae_epoch': self._best_ae_epoch,
            'n_features': self._n_features,
        }
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'head': head_path,
                'scaler': scaler_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        import torch
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'model.pt')
        head_path = os.path.join(output_dir, 'head.pkl')
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst._n_features = meta.get('n_features')
        inst.ae_ = inst._build_ae(inst._n_features)
        state = torch.load(model_path, weights_only=True)
        inst.ae_.load_state_dict(state['state_dict'])
        with open(head_path, 'rb') as f:
            inst.head_ = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            inst.scaler_ = pickle.load(f)
        inst._best_ae_epoch = meta.get('best_ae_epoch')
        return inst


# models/trainers.py — append at end (before TRAINERS dict)

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _frets_complex_linear(z, W_r, W_i, b_r, b_i):
    """Complex-valued linear: y = (W_r + jW_i) @ z + (b_r + jb_i),
    with ReLU applied to real and imaginary parts separately."""
    z_r, z_i = z.real, z.imag
    y_r = F.relu(z_r @ W_r - z_i @ W_i + b_r)
    y_i = F.relu(z_r @ W_i + z_i @ W_r + b_i)
    return torch.complex(y_r, y_i)


class _FreTSNet(nn.Module):
    """Frequency-domain MLP classifier inspired by FreTS (NeurIPS 2023).
    Input: (B, L, C) sequences. Architecture:
      embed scalar -> E-dim
      FFT along channel axis -> complex MLP -> iFFT  (Frequency Channel Learner)
      FFT along time axis    -> complex MLP -> iFFT  (Frequency Temporal Learner)
      mean-pool over (L, C) -> head -> logit
    """
    def __init__(self, seq_len, channels, embed_size=64, hidden_size=128, dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.channels = int(channels)
        self.embed_size = int(embed_size)
        self.embed = nn.Linear(1, self.embed_size)
        scale = 1.0 / float(np.sqrt(self.embed_size))
        E = self.embed_size
        self.Wc_r = nn.Parameter(torch.randn(E, E) * scale)
        self.Wc_i = nn.Parameter(torch.randn(E, E) * scale)
        self.bc_r = nn.Parameter(torch.zeros(E))
        self.bc_i = nn.Parameter(torch.zeros(E))
        self.Wt_r = nn.Parameter(torch.randn(E, E) * scale)
        self.Wt_i = nn.Parameter(torch.randn(E, E) * scale)
        self.bt_r = nn.Parameter(torch.zeros(E))
        self.bt_i = nn.Parameter(torch.zeros(E))
        self.head = nn.Sequential(
            nn.LayerNorm(E),
            nn.Linear(E, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(self, x):
        B, L, C = x.shape
        x = x.unsqueeze(-1)
        x = self.embed(x)
        z = torch.fft.rfft(x, dim=2)
        z = _frets_complex_linear(z, self.Wc_r, self.Wc_i, self.bc_r, self.bc_i)
        x = torch.fft.irfft(z, n=C, dim=2)
        z = torch.fft.rfft(x, dim=1)
        z = _frets_complex_linear(z, self.Wt_r, self.Wt_i, self.bt_r, self.bt_i)
        x = torch.fft.irfft(z, n=L, dim=1)
        pooled = x.mean(dim=(1, 2))
        logit = self.head(pooled).squeeze(-1)
        return logit


class TorchFreTSTrainer(BaseTrainer):
    name = 'torch_frets'
    consumes_sequences = True

    def __init__(self, embed_size=64, hidden_size=128, dropout=0.1,
                 learning_rate=1e-3, weight_decay=1e-4, pos_weight=1.5,
                 epochs=20, batch_size=256, device=None, seed=42, **kwargs):
        self.embed_size = int(embed_size)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.pos_weight = float(pos_weight)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(seed)
        self.model = None
        self.seq_len = None
        self.channels = None

    def _build_model(self, seq_len, channels):
        torch.manual_seed(self.seed)
        m = _FreTSNet(seq_len=seq_len, channels=channels,
                      embed_size=self.embed_size, hidden_size=self.hidden_size,
                      dropout=self.dropout)
        return m.to(self.device)

    @staticmethod
    def _as_tensor_X(X):
        arr = np.asarray(X, dtype=np.float32)
        return torch.from_numpy(arr)

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, **kwargs):
        X_t = self._as_tensor_X(X_tr)
        if X_t.ndim != 3:
            raise ValueError(f"TorchFreTSTrainer expects (N, L, C) sequences, got {tuple(X_t.shape)}")
        N, L, C = X_t.shape
        self.seq_len, self.channels = int(L), int(C)
        y_t = torch.from_numpy(np.asarray(y_tr, dtype=np.float32).reshape(-1))
        self.model = self._build_model(L, C)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.AdamW(self.model.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        pos_w = torch.tensor([float(self.pos_weight)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        self.model.train()
        for _ep in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                logit = self.model(xb)
                loss = loss_fn(logit, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("TorchFreTSTrainer.predict_proba called before fit")
        X_t = self._as_tensor_X(X)
        if X_t.ndim != 3:
            raise ValueError(f"TorchFreTSTrainer expects (N, L, C) sequences, got {tuple(X_t.shape)}")
        out = []
        bs = max(int(self.batch_size), 1)
        self.model.eval()
        for i in range(0, X_t.shape[0], bs):
            xb = X_t[i:i + bs].to(self.device)
            logit = self.model(xb)
            p = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
            out.append(p)
        return np.concatenate(out, axis=0)

    def save(self, model_dir, extra=None):
        os.makedirs(model_dir, exist_ok=True)
        ckpt = {
            'state_dict': (self.model.state_dict() if self.model is not None else None),
            'seq_len': self.seq_len,
            'channels': self.channels,
            'hparams': {
                'embed_size': self.embed_size,
                'hidden_size': self.hidden_size,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'pos_weight': self.pos_weight,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'seed': self.seed,
            },
        }
        torch.save(ckpt, os.path.join(model_dir, 'torch_frets.pt'))
        meta = {'name': self.name, 'consumes_sequences': self.consumes_sequences}
        if extra:
            meta.update(extra)
        with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        return model_dir


# --------------------------------------------------------------------- #
# Kernel-Anomaly Blend (iter #1564) — pivot from torch_frets (exhausted).
# Part A cross-tab: anomaly_gated_histgb (#1499) uniquely hits W1+W7;
# kernel_logreg (#1344) uniquely hits W3+W5+W6. MAX-fusion of their
# probabilities lets each base contribute its top-K-per-date picks
# without absolute-score calibration — addresses torch_frets' failure
# mode (score collapse below 0.3 → flood at thr=0.0).
# --------------------------------------------------------------------- #
class KernelAnomalyBlendTrainer(BaseTrainer):
    """Score-MAX fusion of anomaly_gated_histgb and kernel_logreg."""

    name = 'kernel_anomaly_blend'

    def __init__(self,
                 if_n_estimators: int = 200,
                 if_max_samples: float = 0.5,
                 if_contamination: float = 0.1,
                 gate_threshold: float = 0.2,
                 gating_alpha: float = 1.0,
                 anom_max_iter: int = 400,
                 anom_max_leaf_nodes: int = 31,
                 anom_learning_rate: float = 0.05,
                 anom_min_samples_leaf: int = 50,
                 anom_l2_regularization: float = 1.0,
                 anom_pos_class_weight: float = 2.5,
                 anom_use_monotonic: bool = True,
                 kernel_n_components: int = 300,
                 kernel_gamma: float = 0.5,
                 kernel_C: float = 3.0,
                 kernel_max_iter: int = 200,
                 kernel_pca_components: int = 32,
                 kernel_class_weight: str = 'none',
                 fusion_mode: str = 'max',
                 random_state: int = 42,
                 **_):
        self._params = dict(
            if_n_estimators=int(if_n_estimators),
            if_max_samples=float(if_max_samples),
            if_contamination=float(if_contamination),
            gate_threshold=float(gate_threshold),
            gating_alpha=float(gating_alpha),
            anom_max_iter=int(anom_max_iter),
            anom_max_leaf_nodes=int(anom_max_leaf_nodes),
            anom_learning_rate=float(anom_learning_rate),
            anom_min_samples_leaf=int(anom_min_samples_leaf),
            anom_l2_regularization=float(anom_l2_regularization),
            anom_pos_class_weight=float(anom_pos_class_weight),
            anom_use_monotonic=bool(anom_use_monotonic),
            kernel_n_components=int(kernel_n_components),
            kernel_gamma=float(kernel_gamma),
            kernel_C=float(kernel_C),
            kernel_max_iter=int(kernel_max_iter),
            kernel_pca_components=int(kernel_pca_components),
            kernel_class_weight=str(kernel_class_weight),
            fusion_mode=str(fusion_mode),
            random_state=int(random_state),
        )
        self.anom_ = None
        self.kernel_ = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        p = self._params
        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        self.anom_ = AnomalyGatedHistGBTrainer(
            if_n_estimators=p['if_n_estimators'],
            if_max_samples=p['if_max_samples'],
            if_contamination=p['if_contamination'],
            gate_threshold=p['gate_threshold'],
            gating_alpha=p['gating_alpha'],
            max_iter=p['anom_max_iter'],
            max_leaf_nodes=p['anom_max_leaf_nodes'],
            learning_rate=p['anom_learning_rate'],
            min_samples_leaf=p['anom_min_samples_leaf'],
            l2_regularization=p['anom_l2_regularization'],
            pos_class_weight=p['anom_pos_class_weight'],
            use_monotonic=p['anom_use_monotonic'],
            random_state=p['random_state'],
        )
        self.anom_.fit(X_train, y_train, X_val, y_val, verbose=False)

        self.kernel_ = KernelLogRegTrainer(
            n_components=p['kernel_n_components'],
            gamma=p['kernel_gamma'],
            C=p['kernel_C'],
            max_iter=p['kernel_max_iter'],
            pca_components=p['kernel_pca_components'],
            class_weight=p['kernel_class_weight'],
            calibrate='none',
            random_state=p['random_state'] + 1,
        )
        self.kernel_.fit(X_train, y_train, X_val, y_val, verbose=False)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.anom_ is None or self.kernel_ is None:
            raise RuntimeError('Model not fit')
        p_a = np.asarray(self.anom_.predict_proba(X), dtype=np.float32)
        p_k = np.asarray(self.kernel_.predict_proba(X), dtype=np.float32)
        mode = self._params['fusion_mode']
        if mode == 'mean':
            return ((p_a + p_k) * 0.5).astype(np.float32)

        def _rank01(v):
            n = len(v)
            if n == 0:
                return v
            order = np.argsort(v, kind='mergesort')
            ranks = np.empty(n, dtype=np.float32)
            ranks[order] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
            return ranks

        if mode == 'rank_max':
            return np.maximum(_rank01(p_a), _rank01(p_k)).astype(np.float32)
        if mode == 'rank_min':
            # Consensus picker: only rows BOTH models rank highly survive.
            # Targets bull-regime over-trade where max-fusion lets in
            # anomaly-only positives that dilute WR (e.g. W7 #1607/#1611).
            return np.minimum(_rank01(p_a), _rank01(p_k)).astype(np.float32)
        return np.maximum(p_a, p_k).astype(np.float32)

    def feature_importance(self):
        return None

    @property
    def best_iteration(self):
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        anom_dir = os.path.join(output_dir, 'anom')
        kernel_dir = os.path.join(output_dir, 'kernel')
        self.anom_.save(anom_dir)
        self.kernel_.save(kernel_dir)
        meta_path = os.path.join(output_dir, 'metadata.json')
        meta = {'trainer': self.name, 'hyperparams': self._params}
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'metadata': meta_path, 'anom_dir': anom_dir, 'kernel_dir': kernel_dir}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.anom_ = AnomalyGatedHistGBTrainer.load(os.path.join(output_dir, 'anom'))
        inst.kernel_ = KernelLogRegTrainer.load(os.path.join(output_dir, 'kernel'))
        return inst


# --------------------------------------------------------------------- #
# ROCKET (iter #1595) — pivot from torch_frets (exhausted, 2/7 best in 2
# claude attempts). Part A failure pattern (last 21 iters):
#   W1 48%  W2 57%  W3 19%  W4 10%  W5 0%  W6 29%  W7 52%
# W3/W4/W5 (transition+bear) blocked across every family tried. FreTS's
# DFT-MLP framed signal as global periodicity — failed because deep-bear
# regimes (W5) have non-stationary transient patterns, not stationary
# spectra. ROCKET (Dempster et al. 2020, arXiv 1910.13051) instead uses
# THOUSANDS of *random* non-trained dilated 1D conv kernels: each kernel
# captures a different local transient shape at a different time scale,
# and PPV (proportion of positive values) makes the representation
# location-invariant. The ridge head is convex (no SGD instability that
# collapsed FreTS scores below 0.3). The inductive bias — multi-scale
# random transient detectors — is genuinely absent from the registry:
# trees (axis splits), kernel_logreg (RBF in flat 96-d), GRU/Mamba/
# Transformer (learned recurrence/attention), FreTS (frequency MLP).
# Random fixed weights also bypass the train/test distribution-shift
# problem that hurts deep nets in W5: no parameters can overfit a
# bull-train regime if the kernels never learn anything from data.
# --------------------------------------------------------------------- #
class ROCKETClassifierTrainer(BaseTrainer):
    name = 'rocket_classifier'
    consumes_sequences = True

    def __init__(self,
                 n_kernels: int = 2000,
                 kernel_length: int = 9,
                 max_channels_per_kernel: int = 9,
                 ridge_alpha: float = 1.0,
                 class_weight: str = 'balanced',
                 chunk_size: int = 4096,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            n_kernels=int(n_kernels),
            kernel_length=int(kernel_length),
            max_channels_per_kernel=int(max_channels_per_kernel),
            ridge_alpha=float(ridge_alpha),
            class_weight=str(class_weight),
            chunk_size=int(chunk_size),
            random_state=int(random_state),
        )
        self._kernels = None
        self._seq_len = None
        self._channels = None
        self._scaler = None
        self._clf = None

    def _generate_kernels(self, seq_len: int, channels: int):
        p = self._params
        rng = np.random.RandomState(p['random_state'])
        K = p['n_kernels']
        klen = p['kernel_length']
        # dilation: pick among {1, 2} so that effective span ((klen-1)*d + 1)
        # fits within seq_len. For klen=9 seq_len=20: d=1 span 9; d=2 span 17.
        max_dil = max(1, (seq_len - 1) // (klen - 1))
        dilations = rng.choice(
            [d for d in (1, 2) if d <= max_dil],
            size=K).astype(np.int32)

        # weights: standard normal, zero-mean within each kernel (Dempster trick)
        weights = rng.randn(K, klen).astype(np.float32)
        weights -= weights.mean(axis=1, keepdims=True)

        # biases: drawn from output distribution proxy (uniform [-1, 1])
        biases = rng.uniform(-1.0, 1.0, size=K).astype(np.float32)

        # per-kernel channel subset: sample uniform 1..max_channels_per_kernel
        max_ch = min(p['max_channels_per_kernel'], channels)
        channel_masks = np.zeros((K, channels), dtype=np.float32)
        chan_signs = np.zeros((K, channels), dtype=np.float32)
        for k in range(K):
            n_sel = rng.randint(1, max_ch + 1)
            sel = rng.choice(channels, size=n_sel, replace=False)
            channel_masks[k, sel] = 1.0
            chan_signs[k, sel] = rng.choice([-1.0, 1.0], size=n_sel)

        # effective channel weight: ±1 sign × mask, normalized by sqrt(n_sel)
        n_sel_per_kernel = channel_masks.sum(axis=1, keepdims=True)
        n_sel_per_kernel = np.maximum(n_sel_per_kernel, 1.0)
        channel_proj = (chan_signs / np.sqrt(n_sel_per_kernel)).astype(np.float32)

        return dict(
            weights=weights, biases=biases, dilations=dilations,
            channel_proj=channel_proj, kernel_length=klen,
        )

    def _transform(self, X3d: np.ndarray) -> np.ndarray:
        """X3d: (N, L, C) → features (N, K) via PPV per kernel."""
        p = self._params
        N, L, C = X3d.shape
        K = self._kernels['weights'].shape[0]
        klen = self._kernels['kernel_length']
        weights = self._kernels['weights']
        biases = self._kernels['biases']
        dilations = self._kernels['dilations']
        channel_proj = self._kernels['channel_proj']

        # group kernels by dilation for vectorized 9-gram extraction
        out = np.zeros((N, K), dtype=np.float32)
        chunk = max(int(p['chunk_size']), 64)
        unique_dils = np.unique(dilations)
        for d in unique_dils:
            kmask = (dilations == d)
            k_idx = np.where(kmask)[0]
            Kd = len(k_idx)
            span = (klen - 1) * int(d) + 1
            T = L - span + 1
            if T <= 0:
                continue
            # collapse channels per-kernel first: produce X_proj[n, t, k] of
            # shape (N, L, Kd) via X3d (N,L,C) @ channel_proj[k_idx].T (C, Kd)
            cp = channel_proj[k_idx].T.astype(np.float32)  # (C, Kd)
            w_d = weights[k_idx]  # (Kd, klen)
            b_d = biases[k_idx]   # (Kd,)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                # (m, L, C) @ (C, Kd) → (m, L, Kd)
                Xp = np.einsum('mlc,ck->mlk', X3d[s:e], cp, optimize=True)
                # build 9-gram windows along time axis with dilation d
                # window indices: [t, t+d, t+2d, ..., t+(klen-1)d]
                grams = np.stack(
                    [Xp[:, t:t + klen * int(d):int(d), :] for t in range(T)],
                    axis=1,  # (m, T, klen, Kd)
                )
                # apply per-kernel time weights: out_tk = sum_i w[k,i] * grams[m,t,i,k]
                conv = np.einsum('mtik,ki->mtk', grams, w_d, optimize=True)
                conv -= b_d[None, None, :]
                ppv = (conv > 0).mean(axis=1).astype(np.float32)  # (m, Kd)
                out[s:e, k_idx] = ppv
        return out

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if X_train.ndim != 3:
            raise ValueError(
                f'rocket_classifier expects 3D (N, L, C); got {X_train.shape}.')
        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        X = np.asarray(X_train, dtype=np.float32)
        X = np.clip(X, -8.0, 8.0)
        N, L, C = X.shape
        self._seq_len, self._channels = int(L), int(C)
        # also fold val into the fit (ROCKET has no early stopping concept)
        Xv = np.asarray(X_val, dtype=np.float32)
        Xv = np.clip(Xv, -8.0, 8.0)
        X_full = np.concatenate([X, Xv], axis=0)
        y_full = np.concatenate([
            np.asarray(y_train).reshape(-1),
            np.asarray(y_val).reshape(-1),
        ])

        self._kernels = self._generate_kernels(L, C)
        Z = self._transform(X_full)  # (N_full, K)

        # standardize PPV features
        self._scaler = StandardScaler()
        Z = self._scaler.fit_transform(Z)

        p = self._params
        # ridge-regularized logistic regression — convex optimization,
        # ~30k rows × 2000 features is fast in scikit's lbfgs.
        self._clf = LogisticRegression(
            penalty='l2',
            C=1.0 / max(p['ridge_alpha'], 1e-6),
            solver='lbfgs',
            max_iter=500,
            class_weight=p['class_weight'] if p['class_weight'] else None,
            random_state=p['random_state'],
            n_jobs=1,
        )
        self._clf.fit(Z, y_full)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError('Model not fit')
        X = np.asarray(X, dtype=np.float32)
        X = np.clip(X, -8.0, 8.0)
        Z = self._transform(X)
        Z = self._scaler.transform(Z)
        return self._clf.predict_proba(Z)[:, 1]

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'rocket.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(dict(
                kernels=self._kernels,
                scaler=self._scaler,
                clf=self._clf,
                seq_len=self._seq_len,
                channels=self._channels,
            ), f)
        meta = {'trainer': self.name, 'hyperparams': self._params}
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'rocket.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        with open(model_path, 'rb') as f:
            blob = pickle.load(f)
        inst._kernels = blob['kernels']
        inst._scaler = blob['scaler']
        inst._clf = blob['clf']
        inst._seq_len = blob['seq_len']
        inst._channels = blob['channels']
        return inst


# models/trainers.py — append at end (before TRAINERS dict)

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _XLSTMClassifierNet(nn.Module):
    """xLSTM-based classifier for time-series day windows.

    Input: (B, L, C) sequences where L=day window (typically 20) and C=channels (per-day features).
    Architecture:
      Linear(C -> E) input projection
      xLSTMBlockStack with mLSTM-only blocks (slstm_at=[] -> no custom-CUDA dependency)
        - matrix memory C ∈ R^{d_k × d_v} per head
        - exponential input/forget gates with stabilizer state
      Mean-pool over L
      Linear(E -> 1) classification head -> logit
    """

    def __init__(self, seq_len, channels, embedding_dim=64, num_blocks=2,
                 num_heads=4, dropout=0.1):
        super().__init__()
        # xlstm's slstm submodule imports cuda_init at top of package, which
        # requires CUDA_HOME even though we only use mLSTM blocks. Provide a
        # harmless fallback so import succeeds on systems without a full CUDA
        # toolkit (we never JIT-compile sLSTM kernels here).
        os.environ.setdefault("CUDA_HOME", "/usr")
        from xlstm import (
            xLSTMBlockStack, xLSTMBlockStackConfig,
            mLSTMBlockConfig, mLSTMLayerConfig,
        )
        self.seq_len = int(seq_len)
        self.channels = int(channels)
        self.embedding_dim = int(embedding_dim)
        # num_heads must divide embedding_dim; clamp defensively.
        nh = int(num_heads)
        while nh > 1 and (self.embedding_dim % nh) != 0:
            nh //= 2
        self.num_heads = max(1, nh)

        self.input_proj = nn.Linear(self.channels, self.embedding_dim)
        self.in_drop = nn.Dropout(float(dropout))

        mlstm_cfg = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(num_heads=self.num_heads)
        )
        stack_cfg = xLSTMBlockStackConfig(
            mlstm_block=mlstm_cfg,
            slstm_block=None,
            context_length=self.seq_len,
            num_blocks=int(num_blocks),
            embedding_dim=self.embedding_dim,
            slstm_at=[],
            dropout=float(dropout),
        )
        self.stack = xLSTMBlockStack(stack_cfg)
        self.head_drop = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        # x: (B, L, C) — project to (B, L, E), run xLSTM stack, mean-pool, head.
        h = self.input_proj(x)
        h = self.in_drop(h)
        h = self.stack(h)
        h = h.mean(dim=1)
        h = self.head_drop(h)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------- #
# KernelHistGBStack (claude iter #1629, brief-exhausted pivot from torch_xlstm)
#
# Pivot evidence (Part A.3 diagnosis): two distinct 6/7 ceiling-hitters in
# the registry fail on DIFFERENT single windows:
#   iter #1218 HistGBMonotonic (pure tree, regime+momentum monotone cst):
#     PASS W1,W2,W3,W5,W6,W7  FAIL W4 (33% WR, +3% ann, 14% DD)
#   iter #1607 KernelAnomalyBlend (IsolationForest-gated HistGB + kernel-LR):
#     PASS W1..W6              FAIL W7 (34% WR, +17% ann, 11% DD)
# Both single-window failures miss the 40% WR bar by ~5-7pp. Their PASS
# windows mostly overlap (W1,W2,W3,W5,W6), but their FAIL windows are
# orthogonal — implying the two engines have complementary inductive
# blind spots:
#   * HistGB axis-splits + regime monotonic cst sees breadth-driven bear
#     reversals cleanly (cracks W5/W6/W7) but mis-prices transition W4.
#   * Kernel-RBF over a flat 96-d projection sees similarity-based bull/
#     transition patterns (W1-W6) but gets gated out of W7's late
#     transition by the IsolationForest contamination filter.
# A rank-mean fusion of the two engines should retain both PASS sets and
# average down each engine's weak-window false positives.
#
# Structurally NEW vs. existing registry:
#   * stacked_ranker — intra-XGB stacking (LightGBM/XGB/different losses)
#   * kernel_anomaly_blend — fuses anomaly_gated_histgb + kernel_logreg
#     (both internally; uses max/mean/rank_max/rank_min over those two
#     COMPONENT outputs, NOT over the full histgb_monotonic engine)
#   * xgb_rank_fusion — fuses multiple XGB heads only
# This trainer fuses TWO TOP-LEVEL families (HistGB-monotone tree +
# KernelAnomalyBlend composite) — a cross-family stack absent from the
# registry.
# --------------------------------------------------------------------- #
class KernelHistGBStackTrainer(BaseTrainer):
    """Rank-mean cross-family ensemble of HistGBMonotonic + KernelAnomalyBlend."""

    name = 'kernel_histgb_stack'

    def __init__(self,
                 # HistGBMonotonic base — defaults from #1218 winning HPs.
                 histgb_max_iter: int = 400,
                 histgb_max_leaf_nodes: int = 63,
                 histgb_learning_rate: float = 0.05,
                 histgb_min_samples_leaf: int = 50,
                 histgb_l2_regularization: float = 1.0,
                 histgb_pos_class_weight: float = 3.0,
                 histgb_use_monotonic: bool = True,
                 # KernelAnomalyBlend base — defaults from #1607 winning HPs.
                 kab_gate_threshold: float = 0.20,
                 kab_gating_alpha: float = 1.0,
                 kab_anom_max_iter: int = 400,
                 kab_anom_learning_rate: float = 0.05,
                 kab_anom_pos_class_weight: float = 2.5,
                 kab_kernel_n_components: int = 300,
                 kab_kernel_gamma: float = 0.5,
                 kab_kernel_C: float = 3.0,
                 kab_fusion_mode: str = 'max',
                 # Stack-level.
                 stack_weight: float = 0.5,
                 fusion_mode: str = 'regime_aware',
                 # Regime-aware fusion: choose stack weight per-row from input
                 # breadth feature. Index points into the flattened 4F vector
                 # = [last_*, mean_*, std_*, dev_*] in CURATED_FEATURES order;
                 # 16 = last_market_breadth_above_sma20, 41 = mean_market_breadth_above_sma20.
                 # Iter #1644 lesson: equal-weight rank-mean halved HistGB's
                 # bear-window edge → HistGB-heavy under low breadth, KAB-
                 # heavy under high breadth.
                 regime_breadth_col_idx: int = 16,
                 regime_threshold: float = 0.4,
                 regime_bear_stack_weight: float = 0.85,
                 regime_bull_stack_weight: float = 0.40,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            histgb_max_iter=int(histgb_max_iter),
            histgb_max_leaf_nodes=int(histgb_max_leaf_nodes),
            histgb_learning_rate=float(histgb_learning_rate),
            histgb_min_samples_leaf=int(histgb_min_samples_leaf),
            histgb_l2_regularization=float(histgb_l2_regularization),
            histgb_pos_class_weight=float(histgb_pos_class_weight),
            histgb_use_monotonic=bool(histgb_use_monotonic),
            kab_gate_threshold=float(kab_gate_threshold),
            kab_gating_alpha=float(kab_gating_alpha),
            kab_anom_max_iter=int(kab_anom_max_iter),
            kab_anom_learning_rate=float(kab_anom_learning_rate),
            kab_anom_pos_class_weight=float(kab_anom_pos_class_weight),
            kab_kernel_n_components=int(kab_kernel_n_components),
            kab_kernel_gamma=float(kab_kernel_gamma),
            kab_kernel_C=float(kab_kernel_C),
            kab_fusion_mode=str(kab_fusion_mode),
            stack_weight=float(stack_weight),
            fusion_mode=str(fusion_mode),
            regime_breadth_col_idx=int(regime_breadth_col_idx),
            regime_threshold=float(regime_threshold),
            regime_bear_stack_weight=float(regime_bear_stack_weight),
            regime_bull_stack_weight=float(regime_bull_stack_weight),
            random_state=int(random_state),
        )
        self.histgb_ = None
        self.kab_ = None

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        p = self._params
        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        self.histgb_ = HistGBMonotonicTrainer(
            max_iter=p['histgb_max_iter'],
            max_leaf_nodes=p['histgb_max_leaf_nodes'],
            learning_rate=p['histgb_learning_rate'],
            min_samples_leaf=p['histgb_min_samples_leaf'],
            l2_regularization=p['histgb_l2_regularization'],
            pos_class_weight=p['histgb_pos_class_weight'],
            use_monotonic=p['histgb_use_monotonic'],
            random_state=p['random_state'],
        )
        self.histgb_.fit(X_train, y_train, X_val, y_val, verbose=False)

        self.kab_ = KernelAnomalyBlendTrainer(
            gate_threshold=p['kab_gate_threshold'],
            gating_alpha=p['kab_gating_alpha'],
            anom_max_iter=p['kab_anom_max_iter'],
            anom_learning_rate=p['kab_anom_learning_rate'],
            anom_pos_class_weight=p['kab_anom_pos_class_weight'],
            kernel_n_components=p['kab_kernel_n_components'],
            kernel_gamma=p['kab_kernel_gamma'],
            kernel_C=p['kab_kernel_C'],
            fusion_mode=p['kab_fusion_mode'],
            random_state=p['random_state'] + 1,
        )
        self.kab_.fit(X_train, y_train, X_val, y_val, verbose=False)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.histgb_ is None or self.kab_ is None:
            raise RuntimeError('Model not fit')
        p_h = np.asarray(self.histgb_.predict_proba(X), dtype=np.float32)
        p_k = np.asarray(self.kab_.predict_proba(X), dtype=np.float32)
        mode = self._params['fusion_mode']
        w = float(self._params['stack_weight'])
        w = max(0.0, min(1.0, w))

        def _rank01(v):
            n = len(v)
            if n == 0:
                return v
            order = np.argsort(v, kind='mergesort')
            ranks = np.empty(n, dtype=np.float32)
            ranks[order] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
            return ranks

        r_h = _rank01(p_h)
        r_k = _rank01(p_k)

        if mode == 'rank_mean':
            return (w * r_h + (1.0 - w) * r_k).astype(np.float32)
        if mode == 'prob_mean':
            return (w * p_h + (1.0 - w) * p_k).astype(np.float32)
        if mode == 'rank_max':
            return np.maximum(r_h, r_k).astype(np.float32)
        if mode == 'rank_min':
            return np.minimum(r_h, r_k).astype(np.float32)
        if mode == 'regime_aware':
            # Per-row stack-weight derived from the input row's breadth signal.
            # Bear day (breadth low) → HistGB-heavy. Bull day → KAB-heavy.
            Xa = np.asarray(X, dtype=np.float32)
            if Xa.ndim == 3:
                # (N, L, C): use last timestep's breadth column (raw curated index).
                col_raw = int(self._params['regime_breadth_col_idx']) % Xa.shape[-1]
                breadth = Xa[:, -1, col_raw]
            else:
                col = int(self._params['regime_breadth_col_idx'])
                col = max(0, min(col, Xa.shape[-1] - 1))
                breadth = Xa[:, col]
            thr = float(self._params['regime_threshold'])
            w_bear = float(self._params['regime_bear_stack_weight'])
            w_bull = float(self._params['regime_bull_stack_weight'])
            # Smooth (sigmoid) transition over a narrow band so neighbouring
            # rows don't flip discretely between engines — keeps the per-date
            # ranking stable.
            band = 0.05
            z = np.clip((breadth - thr) / max(band, 1e-6), -50.0, 50.0)
            soft = 1.0 / (1.0 + np.exp(z))
            w_per_row = (w_bear * soft + w_bull * (1.0 - soft)).astype(np.float32)
            w_per_row = np.clip(w_per_row, 0.0, 1.0)
            return (w_per_row * r_h + (1.0 - w_per_row) * r_k).astype(np.float32)
        # Default fallback: rank-mean
        return (w * r_h + (1.0 - w) * r_k).astype(np.float32)

    def feature_importance(self):
        return None

    @property
    def best_iteration(self):
        return None

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        histgb_dir = os.path.join(output_dir, 'histgb')
        kab_dir = os.path.join(output_dir, 'kab')
        self.histgb_.save(histgb_dir)
        self.kab_.save(kab_dir)
        meta_path = os.path.join(output_dir, 'metadata.json')
        meta = {'trainer': self.name, 'hyperparams': self._params}
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'metadata': meta_path, 'histgb_dir': histgb_dir, 'kab_dir': kab_dir}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        inst.histgb_ = HistGBMonotonicTrainer.load(os.path.join(output_dir, 'histgb'))
        inst.kab_ = KernelAnomalyBlendTrainer.load(os.path.join(output_dir, 'kab'))
        return inst


class TorchXLSTMTrainer(BaseTrainer):
    name = 'torch_xlstm'
    consumes_sequences = True

    def __init__(self, embedding_dim=64, num_blocks=2, num_heads=4,
                 dropout=0.1, learning_rate=1e-3, weight_decay=1e-4,
                 pos_weight=1.5, epochs=20, batch_size=256, device=None,
                 seed=42, **kwargs):
        self.embedding_dim = int(embedding_dim)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.pos_weight = float(pos_weight)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(seed)
        self.model = None
        self.seq_len = None
        self.channels = None

    def _build_model(self, seq_len, channels):
        torch.manual_seed(self.seed)
        m = _XLSTMClassifierNet(
            seq_len=seq_len, channels=channels,
            embedding_dim=self.embedding_dim, num_blocks=self.num_blocks,
            num_heads=self.num_heads, dropout=self.dropout,
        )
        return m.to(self.device)

    @staticmethod
    def _as_tensor_X(X):
        arr = np.asarray(X, dtype=np.float32)
        return torch.from_numpy(arr)

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, **kwargs):
        X_t = self._as_tensor_X(X_tr)
        if X_t.ndim != 3:
            raise ValueError(
                f"TorchXLSTMTrainer expects (N, L, C) sequences, got {tuple(X_t.shape)}"
            )
        N, L, C = X_t.shape
        self.seq_len, self.channels = int(L), int(C)
        y_t = torch.from_numpy(np.asarray(y_tr, dtype=np.float32).reshape(-1))
        self.model = self._build_model(L, C)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        pos_w = torch.tensor([float(self.pos_weight)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        self.model.train()
        for _ep in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                logit = self.model(xb)
                loss = loss_fn(logit, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("TorchXLSTMTrainer.predict_proba called before fit")
        X_t = self._as_tensor_X(X)
        if X_t.ndim != 3:
            raise ValueError(
                f"TorchXLSTMTrainer expects (N, L, C) sequences, got {tuple(X_t.shape)}"
            )
        out = []
        bs = max(int(self.batch_size), 1)
        self.model.eval()
        for i in range(0, X_t.shape[0], bs):
            xb = X_t[i:i + bs].to(self.device)
            logit = self.model(xb)
            p = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
            out.append(p)
        return np.concatenate(out, axis=0)

    def save(self, model_dir, extra=None):
        os.makedirs(model_dir, exist_ok=True)
        state_path = os.path.join(model_dir, 'xlstm_state.pt')
        torch.save({
            'model_state': self.model.state_dict() if self.model is not None else None,
            'seq_len': self.seq_len,
            'channels': self.channels,
            'embedding_dim': self.embedding_dim,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
        }, state_path)
        meta = {
            'name': self.name,
            'seq_len': self.seq_len,
            'channels': self.channels,
            'embedding_dim': self.embedding_dim,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'pos_weight': self.pos_weight,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }
        if extra:
            meta.update(extra)
        meta_path = os.path.join(model_dir, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'state': state_path, 'meta': meta_path}


# --------------------------------------------------------------------- #
# RocketBaggedCalibratedTrainer (claude iter #1660, brief-exhausted pivot)
#
# Brief pivot: torch_xlstm (matrix-memory recurrence) tried twice at 2/7;
# its W5 collapse (25.8% WR, -52.9% ann, 23.5% DD) matched the universal
# W5 wall (1/19 pass rate over last 20 iters across all families). The
# sequence-NN family doesn't solve the structural W5 problem.
#
# Pivot rationale (Part A): ROCKET (iter #1625) had 1/7 (W6 only) but
# also collapsed W5 (22.2% WR, -58.8% ann). The collapse mode in both
# cases is "score saturation"—predictions cluster around 0.5 and the
# threshold sweep is forced to thr=0.0, picking the worst possible 36-40
# examples in the worst regime. Three structural fixes vs vanilla
# rocket_classifier, each addressing a distinct failure mode:
#   (1) Date-block bagging (K=5) — variance reduction in the linear head
#       for hostile-regime test slices; train bags on disjoint date blocks
#       so each bag sees different regime mixtures, then average.
#   (2) Isotonic calibration on val — anchors output prob distribution
#       to true val-set frequency, breaking the score-collapse failure
#       where every output ≈ 0.5 and threshold sweep degenerates.
#   (3) Per-date z-score normalization on inference — converts raw scores
#       into cross-sectional ranks within each test date, so picks are
#       made by within-day relative quality even when absolute scores
#       degrade in hostile regimes. Critical for W5 where the model is
#       genuinely uncertain but must still pick a daily top.
#
# Structurally NEW vs registry:
#   * rocket_classifier — single LR head, no bagging, no calibration,
#     no per-date norm (the W6 score-collapse iter #1625 victim)
#   * histgb_monotonic_bagged — date-block bagging exists but on
#     HistGBM trees, NOT on ROCKET random-conv features
#   * xgb_iso_calibrated — isotonic exists but on raw XGB, not chained
#     to ROCKET features
#   * kernel_anomaly_blend — bagging exists internally but in a kernel
#     pipeline, NOT after random-conv feature extraction
# No other trainer composes (random conv kernels) → (date-block bag) →
# (isotonic calibration) → (per-date z-score). This is the structural
# composition gap.
#
# Hypothesis: ROCKET's W6 +126% ann iter #1625 result proves the random
# conv features are bull-regime predictive when scored confidently; the
# W5 collapse was a head-side failure (LR overfit train-bull distribution).
# Bagging + iso-calibration + per-date norm directly attack the
# head-side score-collapse without changing the feature extractor.
# --------------------------------------------------------------------- #
class ROCKETBaggedCalibratedTrainer(BaseTrainer):
    """ROCKET random-conv features + date-block bagged Ridge + isotonic
    calibration + per-date z-score normalization. Pivot from torch_xlstm
    targeting the W5 score-collapse failure mode."""

    name = 'rocket_bagged_calibrated'
    consumes_sequences = True

    def __init__(self,
                 n_kernels: int = 4000,
                 kernel_length: int = 9,
                 max_channels_per_kernel: int = 9,
                 n_bags: int = 5,
                 bag_frac: float = 0.7,
                 ridge_alpha: float = 1.0,
                 pos_class_weight: float = 1.5,
                 calibrate: str = 'isotonic',
                 per_date_zscore: bool = True,
                 chunk_size: int = 4096,
                 random_state: int = 42,
                 **_):
        self._params = dict(
            n_kernels=int(n_kernels),
            kernel_length=int(kernel_length),
            max_channels_per_kernel=int(max_channels_per_kernel),
            n_bags=max(1, int(n_bags)),
            bag_frac=float(np.clip(bag_frac, 0.3, 1.0)),
            ridge_alpha=float(ridge_alpha),
            pos_class_weight=float(pos_class_weight),
            calibrate=str(calibrate),
            per_date_zscore=bool(per_date_zscore),
            chunk_size=int(chunk_size),
            random_state=int(random_state),
        )
        self._kernels = None
        self._seq_len = None
        self._channels = None
        self._scaler = None
        self._bags = []          # list of LogisticRegression
        self._iso = None         # IsotonicRegression or None
        self._predict_dates = None

    # ROCKET kernel generation and transform are identical to the
    # vanilla rocket_classifier (kept inline so this trainer is
    # self-contained — see ROCKETClassifierTrainer for design notes).
    def _generate_kernels(self, seq_len: int, channels: int):
        p = self._params
        rng = np.random.RandomState(p['random_state'])
        K = p['n_kernels']
        klen = p['kernel_length']
        max_dil = max(1, (seq_len - 1) // (klen - 1))
        dilations = rng.choice(
            [d for d in (1, 2) if d <= max_dil],
            size=K).astype(np.int32)
        weights = rng.randn(K, klen).astype(np.float32)
        weights -= weights.mean(axis=1, keepdims=True)
        biases = rng.uniform(-1.0, 1.0, size=K).astype(np.float32)
        max_ch = min(p['max_channels_per_kernel'], channels)
        channel_masks = np.zeros((K, channels), dtype=np.float32)
        chan_signs = np.zeros((K, channels), dtype=np.float32)
        for k in range(K):
            n_sel = rng.randint(1, max_ch + 1)
            sel = rng.choice(channels, size=n_sel, replace=False)
            channel_masks[k, sel] = 1.0
            chan_signs[k, sel] = rng.choice([-1.0, 1.0], size=n_sel)
        n_sel_per_kernel = channel_masks.sum(axis=1, keepdims=True)
        n_sel_per_kernel = np.maximum(n_sel_per_kernel, 1.0)
        channel_proj = (chan_signs / np.sqrt(n_sel_per_kernel)).astype(np.float32)
        return dict(
            weights=weights, biases=biases, dilations=dilations,
            channel_proj=channel_proj, kernel_length=klen,
        )

    def _transform(self, X3d: np.ndarray) -> np.ndarray:
        p = self._params
        N, L, C = X3d.shape
        K = self._kernels['weights'].shape[0]
        klen = self._kernels['kernel_length']
        weights = self._kernels['weights']
        biases = self._kernels['biases']
        dilations = self._kernels['dilations']
        channel_proj = self._kernels['channel_proj']
        out = np.zeros((N, K), dtype=np.float32)
        chunk = max(int(p['chunk_size']), 64)
        unique_dils = np.unique(dilations)
        for d in unique_dils:
            kmask = (dilations == d)
            k_idx = np.where(kmask)[0]
            Kd = len(k_idx)
            span = (klen - 1) * int(d) + 1
            T = L - span + 1
            if T <= 0:
                continue
            cp = channel_proj[k_idx].T.astype(np.float32)
            w_d = weights[k_idx]
            b_d = biases[k_idx]
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                Xp = np.einsum('mlc,ck->mlk', X3d[s:e], cp, optimize=True)
                grams = np.stack(
                    [Xp[:, t:t + klen * int(d):int(d), :] for t in range(T)],
                    axis=1,
                )
                conv = np.einsum('mtik,ki->mtk', grams, w_d, optimize=True)
                conv -= b_d[None, None, :]
                ppv = (conv > 0).mean(axis=1).astype(np.float32)
                out[s:e, k_idx] = ppv
        return out

    def set_predict_context(self, dates):
        """Receive test-set dates from return_gate before predict_proba
        so per-date z-score normalization is anchored to the right
        date partition (the gate passes test_dates here)."""
        self._predict_dates = np.asarray(dates)

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.isotonic import IsotonicRegression

        if X_train.ndim != 3:
            raise ValueError(
                f'rocket_bagged_calibrated expects 3D (N, L, C); got {X_train.shape}.')
        if len(set(y_train)) < 2:
            raise ValueError('Train set has only one class — fit aborted')

        p = self._params
        X = np.clip(np.asarray(X_train, dtype=np.float32), -8.0, 8.0)
        Xv = np.clip(np.asarray(X_val, dtype=np.float32), -8.0, 8.0)
        N, L, C = X.shape
        self._seq_len, self._channels = int(L), int(C)

        # 1. Build random conv kernels and transform train + val
        self._kernels = self._generate_kernels(L, C)
        Z_tr = self._transform(X)
        Z_va = self._transform(Xv)
        self._scaler = StandardScaler()
        Z_tr = self._scaler.fit_transform(Z_tr)
        Z_va = self._scaler.transform(Z_va)

        # 2. Date-block bagging on the train set. Each bag gets a random
        #    subset of unique train dates (bag_frac), trained as Ridge LR.
        rng = np.random.RandomState(p['random_state'])
        if dates_train is not None:
            train_dates = np.asarray(dates_train)
            unique_dates = np.sort(np.unique(train_dates))
        else:
            train_dates = None
            unique_dates = None

        y_tr_arr = np.asarray(y_train).reshape(-1)
        cw = {0: 1.0, 1: float(p['pos_class_weight'])}

        self._bags = []
        for b in range(p['n_bags']):
            if unique_dates is not None and len(unique_dates) >= 8:
                # Sample bag_frac of unique dates (without replacement)
                n_sample = max(4, int(p['bag_frac'] * len(unique_dates)))
                sel_dates = rng.choice(unique_dates, size=n_sample, replace=False)
                row_mask = np.isin(train_dates, sel_dates)
                Z_b = Z_tr[row_mask]
                y_b = y_tr_arr[row_mask]
            else:
                # Fallback: random row bootstrap of bag_frac
                idx = rng.choice(len(Z_tr), size=int(p['bag_frac'] * len(Z_tr)),
                                 replace=False)
                Z_b = Z_tr[idx]
                y_b = y_tr_arr[idx]
            if len(set(y_b)) < 2:
                # Skip degenerate bag rather than crash
                continue
            clf = LogisticRegression(
                penalty='l2',
                C=1.0 / max(p['ridge_alpha'], 1e-6),
                solver='lbfgs',
                max_iter=500,
                class_weight=cw,
                random_state=int(p['random_state']) + b,
                n_jobs=1,
            )
            clf.fit(Z_b, y_b)
            self._bags.append(clf)

        if not self._bags:
            raise RuntimeError('All bags degenerate — fit aborted.')

        # 3. Isotonic calibration on val (mean of bag probs vs y_val)
        y_va_arr = np.asarray(y_val).reshape(-1)
        if p['calibrate'] == 'isotonic' and len(set(y_va_arr)) == 2:
            p_va = np.mean([c.predict_proba(Z_va)[:, 1] for c in self._bags],
                           axis=0)
            self._iso = IsotonicRegression(out_of_bounds='clip',
                                            y_min=0.0, y_max=1.0)
            self._iso.fit(p_va, y_va_arr)
        else:
            self._iso = None
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self._bags:
            raise RuntimeError('Model not fit')
        X = np.clip(np.asarray(X, dtype=np.float32), -8.0, 8.0)
        Z = self._transform(X)
        Z = self._scaler.transform(Z)
        # Mean of bag probabilities
        p_raw = np.mean([c.predict_proba(Z)[:, 1] for c in self._bags], axis=0)
        # Optional isotonic calibration
        if self._iso is not None:
            p_raw = self._iso.transform(p_raw)
        # Optional per-date z-score normalization (then sigmoid)
        if (self._params['per_date_zscore'] and self._predict_dates is not None
                and len(self._predict_dates) == len(p_raw)):
            out = np.empty_like(p_raw, dtype=np.float32)
            dates_arr = np.asarray(self._predict_dates)
            for d in np.unique(dates_arr):
                mask = dates_arr == d
                vals = p_raw[mask]
                if len(vals) <= 1:
                    out[mask] = 0.5
                    continue
                mu = float(np.mean(vals))
                sd = float(np.std(vals))
                if sd < 1e-9:
                    out[mask] = 0.5
                    continue
                z = (vals - mu) / sd
                # map z back to [0, 1] via logistic with tau=1.0
                out[mask] = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
            return out
        return np.asarray(p_raw, dtype=np.float32)

    @property
    def hyperparams(self):
        return dict(self._params)

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'rocket_bagged.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(model_path, 'wb') as f:
            pickle.dump(dict(
                kernels=self._kernels,
                scaler=self._scaler,
                bags=self._bags,
                iso=self._iso,
                seq_len=self._seq_len,
                channels=self._channels,
            ), f)
        meta = {'trainer': self.name, 'hyperparams': self._params}
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return {'model': model_path, 'metadata': meta_path}

    @classmethod
    def load(cls, output_dir):
        meta_path = os.path.join(output_dir, 'metadata.json')
        model_path = os.path.join(output_dir, 'rocket_bagged.pkl')
        with open(meta_path) as f:
            meta = json.load(f)
        params = meta.get('hyperparams', {})
        inst = cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})
        with open(model_path, 'rb') as f:
            blob = pickle.load(f)
        inst._kernels = blob['kernels']
        inst._scaler = blob['scaler']
        inst._bags = blob['bags']
        inst._iso = blob['iso']
        inst._seq_len = blob['seq_len']
        inst._channels = blob['channels']
        return inst


# Iter #1706 — TimesFM-2.5 (Google patched decoder-only TS foundation model)
# frozen-backbone zero-shot quantile forecast → MLP classifier head. Brief
# §5b 2026-05-23 recommends this as a complementary information channel to
# the registered Chronos-2 (value-token encoder) and Time-MoE (sparse MoE
# decoder) foundation-model slots — TimesFM patches continuous values
# instead of tokenizing them, so its quantile spread carries independently
# calibrated regime-conditional uncertainty.
#
# Inductive prior: pass one univariate channel (default = last column =
# set_ret_5d_zscore_60d, a regime-invariant return z-score) into TimesFM,
# get 1 point + 10 quantile forecasts per row, concatenate with the
# last-step tabular row, train a small MLP head on top.
#
# Host workarounds (this machine, May-2026):
#   - NVML library/driver mismatch: PYTORCH_CUDA_ALLOC_CONF=backend:native
#     and CUDA_VISIBLE_DEVICES=0 (set at module top above).
#   - ForecastConfig max_context capped at 512 / per_core_batch_size=64 —
#     larger values trigger the failing NVML path inside the KV-cache
#     allocation. seq_len=20 << 512 so the input fits without truncation.
class _TimesFMHead:
    """Lazy import lives in the trainer to keep module-level imports light."""

    @staticmethod
    def build(in_dim, hidden, dropout):
        import torch.nn as nn

        class _Head(nn.Module):
            def __init__(self, D, H, dp):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(D, H), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(H, H), nn.GELU(), nn.Dropout(dp),
                    nn.Linear(H, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return _Head(int(in_dim), int(hidden), float(dropout))


# Module-level singleton — TimesFM weights are ~800MB; load once across the
# 7 gate windows.
_TIMESFM_CACHE: dict = {}


class TorchTimesFMTrainer(BaseTrainer):
    """Frozen TimesFM-2.5 backbone → MLP classifier head."""

    name = 'torch_timesfm'
    consumes_sequences = True

    def __init__(self,
                 model_id: str = 'google/timesfm-2.5-200m-pytorch',
                 horizon: int = 1,
                 channel_index: int = -1,
                 head_hidden: int = 128,
                 dropout: float = 0.15,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 pos_weight: float = 1.5,
                 epochs: int = 30,
                 batch_size: int = 512,
                 inference_batch: int = 1024,
                 patience: int = 5,
                 max_context: int = 512,
                 max_horizon: int = 64,
                 per_core_batch_size: int = 64,
                 seq_stats_channels=None,
                 device: str = None,
                 random_state: int = 42,
                 **_):
        self.model_id = str(model_id)
        self.horizon = int(horizon)
        self.channel_index = int(channel_index)
        self.head_hidden = int(head_hidden)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.pos_weight = float(pos_weight)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.inference_batch = int(inference_batch)
        self.patience = int(patience)
        self.max_context = int(max_context)
        self.max_horizon = int(max_horizon)
        self.per_core_batch_size = int(per_core_batch_size)
        # seq_stats_channels: list of channel indices to compute per-row
        # sequence summary stats over (mean, std, min, max, last-first delta).
        # Provides cheap per-stock dispersion features to complement the
        # regime-invariant TimesFM forecast on channel_index. None = disabled.
        if seq_stats_channels is None or (isinstance(seq_stats_channels, (list, tuple))
                                          and len(seq_stats_channels) == 0):
            self.seq_stats_channels = None
        else:
            self.seq_stats_channels = [int(c) for c in seq_stats_channels]
        self.requested_device = device
        self.random_state = int(random_state)
        self._head = None
        self._scaler_mean = None
        self._scaler_std = None
        self._in_dim = None
        self._device = None

    def _resolve_device(self):
        import torch
        if self.requested_device:
            return self.requested_device
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load_backbone(self):
        key = (self.model_id, self.max_context, self.max_horizon,
               self.per_core_batch_size)
        if key in _TIMESFM_CACHE:
            return _TIMESFM_CACHE[key]
        import timesfm
        from huggingface_hub import hf_hub_download
        weight_file = hf_hub_download(repo_id=self.model_id,
                                       filename='model.safetensors')
        tfm = timesfm.TimesFM_2p5_200M_torch()
        tfm.model.load_checkpoint(weight_file, torch_compile=False)
        device = self._resolve_device()
        tfm.model.to(device)
        tfm.compile(timesfm.ForecastConfig(
            max_context=self.max_context,
            max_horizon=self.max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
            per_core_batch_size=self.per_core_batch_size,
        ))
        _TIMESFM_CACHE[key] = tfm
        self._device = device
        return tfm

    def _encode(self, X_3d):
        """Run TimesFM forecast on one channel per row. Returns (N, 11*horizon)."""
        tfm = self._load_backbone()
        N, L, C = X_3d.shape
        ci = self.channel_index if self.channel_index >= 0 else (C + self.channel_index)
        ci = max(0, min(int(ci), C - 1))
        channel = X_3d[:, :, ci].astype(np.float32)
        channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
        out_chunks = []
        bs = max(self.inference_batch, self.per_core_batch_size)
        for i in range(0, N, bs):
            j = min(N, i + bs)
            inputs = [channel[k] for k in range(i, j)]
            point_fc, quant_fc = tfm.forecast(horizon=self.horizon, inputs=inputs)
            pf = np.asarray(point_fc, dtype=np.float32).reshape(j - i, self.horizon, 1)
            qf = np.asarray(quant_fc, dtype=np.float32).reshape(j - i, self.horizon, -1)
            emb = np.concatenate([pf, qf], axis=2).reshape(j - i, -1)
            out_chunks.append(emb)
        return np.concatenate(out_chunks, axis=0)

    def _stack_features(self, X_3d, emb):
        last = X_3d[:, -1, :].astype(np.float32)
        last = np.nan_to_num(last, nan=0.0, posinf=0.0, neginf=0.0)
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        parts = [last, emb]
        if self.seq_stats_channels:
            N, L, C = X_3d.shape
            stat_blocks = []
            for raw_idx in self.seq_stats_channels:
                ci = raw_idx if raw_idx >= 0 else (C + raw_idx)
                ci = max(0, min(int(ci), C - 1))
                seq = X_3d[:, :, ci].astype(np.float32)
                seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
                # 5 stats per channel: mean, std, min, max, last-first delta
                stat_blocks.append(seq.mean(axis=1, keepdims=True))
                stat_blocks.append(seq.std(axis=1, keepdims=True))
                stat_blocks.append(seq.min(axis=1, keepdims=True))
                stat_blocks.append(seq.max(axis=1, keepdims=True))
                stat_blocks.append((seq[:, -1] - seq[:, 0]).reshape(-1, 1))
            stats = np.concatenate(stat_blocks, axis=1)
            stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            parts.append(stats)
        return np.concatenate(parts, axis=1)

    def _fit_scaler(self, F):
        self._scaler_mean = F.mean(axis=0).astype(np.float32)
        self._scaler_std = (F.std(axis=0) + 1e-6).astype(np.float32)

    def _apply_scaler(self, F):
        return ((F - self._scaler_mean) / self._scaler_std).astype(np.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose: bool = False,
            pnl_train=None, pnl_val=None,
            dates_train=None, dates_val=None):
        import torch
        import torch.nn as nn

        X_tr_3d = np.asarray(X_train, dtype=np.float32)
        if X_tr_3d.ndim != 3:
            raise ValueError(
                f'torch_timesfm expects 3D sequence input (N, L, F); '
                f'got shape {X_tr_3d.shape}.')
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        emb_tr = self._encode(X_tr_3d)
        F_tr = self._stack_features(X_tr_3d, emb_tr)
        self._fit_scaler(F_tr)
        F_tr = self._apply_scaler(F_tr)
        self._in_dim = int(F_tr.shape[1])

        has_val = (X_val is not None) and (y_val is not None) and len(y_val) > 0
        if has_val:
            X_val_3d = np.asarray(X_val, dtype=np.float32)
            emb_val = self._encode(X_val_3d)
            F_val = self._apply_scaler(self._stack_features(X_val_3d, emb_val))
        else:
            F_val = None

        device = self._device or self._resolve_device()
        self._head = _TimesFMHead.build(self._in_dim, self.head_hidden,
                                         self.dropout).to(device)
        opt = torch.optim.AdamW(self._head.parameters(),
                                 lr=self.learning_rate,
                                 weight_decay=self.weight_decay)
        pos_w = torch.tensor([self.pos_weight], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        F_tr_t = torch.from_numpy(F_tr).to(device)
        y_tr_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32).reshape(-1)).to(device)
        if has_val:
            F_val_t = torch.from_numpy(F_val).to(device)
            y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32).reshape(-1)).to(device)

        best_val = float('inf')
        best_state = None
        bad = 0
        N = F_tr_t.size(0)
        for _ep in range(self.epochs):
            self._head.train()
            perm = torch.randperm(N, device=device)
            for i in range(0, N, self.batch_size):
                idx = perm[i:i + self.batch_size]
                logit = self._head(F_tr_t[idx])
                loss = loss_fn(logit, y_tr_t[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._head.parameters(), 1.0)
                opt.step()
            if has_val:
                self._head.eval()
                with torch.no_grad():
                    vloss = loss_fn(self._head(F_val_t), y_val_t).item()
                if vloss < best_val - 1e-4:
                    best_val = vloss
                    best_state = {k: v.detach().clone()
                                  for k, v in self._head.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self._head.load_state_dict(best_state)
        self._head.eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        if self._head is None:
            raise RuntimeError('TorchTimesFMTrainer.predict_proba before fit')
        X_3d = np.asarray(X, dtype=np.float32)
        if X_3d.ndim != 3:
            raise ValueError(
                f'torch_timesfm.predict_proba expects 3D input (N, L, F); '
                f'got shape {X_3d.shape}.')
        emb = self._encode(X_3d)
        F = self._apply_scaler(self._stack_features(X_3d, emb))
        device = self._device or self._resolve_device()
        out = []
        self._head.eval()
        bs = max(self.batch_size, 1)
        with torch.no_grad():
            for i in range(0, F.shape[0], bs):
                xb = torch.from_numpy(F[i:i + bs]).to(device)
                p = torch.sigmoid(self._head(xb)).cpu().numpy().astype(np.float64)
                out.append(p)
        return np.concatenate(out, axis=0)

    @property
    def hyperparams(self):
        return dict(
            model_id=self.model_id, horizon=self.horizon,
            channel_index=self.channel_index, head_hidden=self.head_hidden,
            dropout=self.dropout, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, pos_weight=self.pos_weight,
            epochs=self.epochs, batch_size=self.batch_size,
            inference_batch=self.inference_batch, patience=self.patience,
            max_context=self.max_context, max_horizon=self.max_horizon,
            per_core_batch_size=self.per_core_batch_size,
            seq_stats_channels=self.seq_stats_channels,
            random_state=self.random_state,
        )

    def save(self, output_dir, extra=None):
        import torch
        os.makedirs(output_dir, exist_ok=True)
        state_path = os.path.join(output_dir, 'timesfm_head.pt')
        torch.save({
            'state_dict': self._head.state_dict() if self._head is not None else None,
            'in_dim': self._in_dim,
            'scaler_mean': self._scaler_mean,
            'scaler_std': self._scaler_std,
        }, state_path)
        meta = dict(self.hyperparams, in_dim=self._in_dim)
        if extra:
            meta.update(extra)
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        return {'state': state_path, 'meta': meta_path}


# models/trainers.py — append at end (before TRAINERS dict)
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _PLRLiteEncoder(nn.Module):
    """Periodic-Linear-ReLU 'lite' encoder for numerical features (ModernNCA, ICLR 2025).

    For each feature x_j, build cos/sin(2*pi * x_j * c_jk) over K learned coefficients
    c_jk, then a per-feature Linear+ReLU into d_emb dims. Outputs (B, n_features * d_emb).
    """

    def __init__(self, n_features: int, n_frequencies: int = 24,
                 sigma: float = 1.0, d_emb: int = 16):
        super().__init__()
        self.n_features = n_features
        self.n_frequencies = n_frequencies
        # one set of coefficients per feature
        self.coefficients = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        self.linear = nn.Linear(2 * n_frequencies, d_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, n_features)
        # broadcast: (B, F, 1) * (1, F, K) -> (B, F, K)
        z = 2 * math.pi * x.unsqueeze(-1) * self.coefficients.unsqueeze(0)
        z = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)  # (B, F, 2K)
        z = F.relu(self.linear(z))  # (B, F, d_emb)
        return z.flatten(start_dim=1)  # (B, F * d_emb)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        h = self.bn(x)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class _ModernNCAEncoder(nn.Module):
    """Deep encoder φ that maps (B, n_features) -> (B, d_embed)."""

    def __init__(self, n_features: int, d_embed: int = 128, n_blocks: int = 2,
                 hidden_mult: int = 2, dropout: float = 0.1,
                 plr_freq: int = 24, plr_sigma: float = 1.0, plr_d_emb: int = 16):
        super().__init__()
        self.num_enc = _PLRLiteEncoder(n_features, n_frequencies=plr_freq,
                                       sigma=plr_sigma, d_emb=plr_d_emb)
        d_after_plr = n_features * plr_d_emb
        self.proj_in = nn.Linear(d_after_plr, d_embed)
        self.blocks = nn.ModuleList([
            _ResidualBlock(d_embed, d_embed * hidden_mult, dropout)
            for _ in range(n_blocks)
        ])
        self.proj_out = nn.Linear(d_embed, d_embed)

    def forward(self, x):
        h = self.num_enc(x)
        h = self.proj_in(h)
        for blk in self.blocks:
            h = blk(h)
        return self.proj_out(h)


class ModernNCATrainer(BaseTrainer):
    """ModernNCA — differentiable deep nearest-neighbour (ICLR 2025).

    Reference: Ye et al., "Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular
    Baseline Two Decades Later", arXiv:2407.03257.

    Architecture:
        x  ->  PLR-lite per-feature embedding  ->  residual MLP encoder φ  ->  z
        p(y=1 | x) = sum_j softmax(-||z - φ(x_j)||^2)_j * 1[y_j = 1]

    Training: cross-entropy on soft-NN probabilities. Stochastic Neighbourhood Sampling
    (SNS): each minibatch's "neighbour bank" is a uniform-random subsample (rate
    sns_rate) of the training set, so the encoder learns from many candidate sets
    rather than a single fixed graph.

    Inference: the full training set (post-fit) is the retrieval bank; this trainer
    therefore keeps X_train / y_train in memory and re-encodes them when predict_proba
    is called. Memory for our 190k x 96 setup: O(190k * d_embed) ≈ 100MB on GPU.
    """

    name = 'modernnca'
    consumes_sequences = False

    def __init__(self, d_embed: int = 128, n_blocks: int = 2, hidden_mult: int = 2,
                 dropout: float = 0.1, plr_freq: int = 24, plr_sigma: float = 1.0,
                 plr_d_emb: int = 16, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 1024, sns_rate: float = 0.2, epochs: int = 50,
                 patience: int = 8, inference_batch: int = 256, seed: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.d_embed = int(d_embed)
        self.n_blocks = int(n_blocks)
        self.hidden_mult = int(hidden_mult)
        self.dropout = float(dropout)
        self.plr_freq = int(plr_freq)
        self.plr_sigma = float(plr_sigma)
        self.plr_d_emb = int(plr_d_emb)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.sns_rate = float(sns_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.inference_batch = int(inference_batch)
        self.seed = int(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._encoder = None
        self._X_train_t = None
        self._y_train_t = None
        self._x_mean = None
        self._x_std = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._x_mean = X.mean(axis=0).astype(np.float32)
            self._x_std = (X.std(axis=0) + 1e-6).astype(np.float32)
        return ((X - self._x_mean) / self._x_std).astype(np.float32)

    def fit(self, X_tr, y_tr, X_val, y_val, **kwargs):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X_tr = self._standardize(np.asarray(X_tr, dtype=np.float32), fit=True)
        X_val = self._standardize(np.asarray(X_val, dtype=np.float32), fit=False)
        y_tr = np.asarray(y_tr, dtype=np.int64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.int64).reshape(-1)

        n_features = X_tr.shape[1]
        self._encoder = _ModernNCAEncoder(
            n_features=n_features, d_embed=self.d_embed, n_blocks=self.n_blocks,
            hidden_mult=self.hidden_mult, dropout=self.dropout,
            plr_freq=self.plr_freq, plr_sigma=self.plr_sigma, plr_d_emb=self.plr_d_emb,
        ).to(self.device)

        X_tr_t = torch.from_numpy(X_tr).to(self.device)
        y_tr_t = torch.from_numpy(y_tr).to(self.device)
        X_val_t = torch.from_numpy(X_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)
        self._X_train_t = X_tr_t
        self._y_train_t = y_tr_t

        opt = torch.optim.AdamW(self._encoder.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        N = X_tr_t.shape[0]
        n_cand = max(128, int(N * self.sns_rate))

        best_val = float('inf')
        bad_epochs = 0
        best_state = None

        for epoch in range(self.epochs):
            self._encoder.train()
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                idx = perm[i:i + self.batch_size]
                x_q = X_tr_t[idx]
                y_q = y_tr_t[idx]

                # SNS: stochastic candidate pool (uniform sample from training set)
                cand_idx = torch.randint(0, N, (n_cand,), device=self.device)
                x_c = X_tr_t[cand_idx]
                y_c = y_tr_t[cand_idx]

                e_q = self._encoder(x_q)            # (B, d)
                e_c = self._encoder(x_c)            # (n_cand, d)
                # squared euclidean distance (ModernNCA uses Euclidean, not squared
                # in the equation but standard implementations square it inside softmax)
                d2 = torch.cdist(e_q, e_c, p=2.0) ** 2  # (B, n_cand)
                w = torch.softmax(-d2, dim=1)       # (B, n_cand)
                p1 = (w * (y_c == 1).float().unsqueeze(0)).sum(dim=1)
                p1 = p1.clamp(1e-7, 1 - 1e-7)
                target = (y_q == 1).float()
                loss = -(target * torch.log(p1) + (1 - target) * torch.log(1 - p1)).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

            # validation: soft-NN over the full training set
            self._encoder.eval()
            with torch.no_grad():
                p_val = self._predict_proba_torch(X_val_t)
                p_val = p_val.clamp(1e-7, 1 - 1e-7)
                val_loss = F.binary_cross_entropy(p_val, (y_val_t == 1).float()).item()

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                bad_epochs = 0
                best_state = {k: v.detach().clone() for k, v in self._encoder.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._encoder.load_state_dict(best_state)
        self._encoder.eval()

    def _predict_proba_torch(self, X_t: torch.Tensor) -> torch.Tensor:
        """Soft-NN probability of class 1 for each query against full training set."""
        self._encoder.eval()
        N_train = self._X_train_t.shape[0]
        with torch.no_grad():
            # encode full train set once, chunked for memory
            chunks = []
            for i in range(0, N_train, self.batch_size):
                chunks.append(self._encoder(self._X_train_t[i:i + self.batch_size]))
            train_emb = torch.cat(chunks, dim=0)  # (N_train, d)
            y_train_one = (self._y_train_t == 1).float()  # (N_train,)

            out = []
            for i in range(0, X_t.shape[0], self.inference_batch):
                e_q = self._encoder(X_t[i:i + self.inference_batch])
                d2 = torch.cdist(e_q, train_emb, p=2.0) ** 2
                w = torch.softmax(-d2, dim=1)
                p1 = (w * y_train_one.unsqueeze(0)).sum(dim=1)
                out.append(p1)
            return torch.cat(out, dim=0)

    def predict_proba(self, X) -> np.ndarray:
        X = self._standardize(np.asarray(X, dtype=np.float32), fit=False)
        X_t = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            p = self._predict_proba_torch(X_t).cpu().numpy()
        return p.astype(np.float32)

    def save(self, model_dir, extra=None):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'encoder_state': self._encoder.state_dict(),
            'X_train': self._X_train_t.cpu().numpy(),
            'y_train': self._y_train_t.cpu().numpy(),
            'x_mean': self._x_mean,
            'x_std': self._x_std,
            'd_embed': self.d_embed,
            'n_blocks': self.n_blocks,
            'hidden_mult': self.hidden_mult,
            'dropout': self.dropout,
            'plr_freq': self.plr_freq,
            'plr_sigma': self.plr_sigma,
            'plr_d_emb': self.plr_d_emb,
        }, str(model_dir / 'modernnca.pt'))
        meta = {
            'name': self.name,
            'd_embed': self.d_embed,
            'n_blocks': self.n_blocks,
            'hidden_mult': self.hidden_mult,
            'dropout': self.dropout,
            'plr_freq': self.plr_freq,
            'plr_sigma': self.plr_sigma,
            'plr_d_emb': self.plr_d_emb,
            'sns_rate': self.sns_rate,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'patience': self.patience,
            'seed': self.seed,
        }
        if extra:
            meta.update(extra)
        with open(model_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)


# Iter #1757 — Sundial (Tsinghua THUML, ICML 2025 Oral, arXiv:2502.00816).
# First flow-matching TS foundation model in the registry. The frozen
# backbone runs in a SINGLE forward() (bypassing transformers' broken
# generate() — Sundial was published against transformers==4.40.1 and
# its prepare_inputs_for_generation hooks reference DynamicCache.seen_tokens
# / get_usable_length / get_max_length which are gone in 4.43+). Direct
# forward returns (B, num_samples, 720) trajectories per row; we summarise
# them into 13 path statistics (mean, std, exceedance probs, IQR, MDD, skew)
# and feed an MLP head jointly with the last-step tabular vector.
_SUNDIAL_CACHE: dict = {}


class _SundialPathStatHead(torch.nn.Module):
    def __init__(self, n_stat_features: int, n_tab_features: int,
                 hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        in_dim = n_stat_features + n_tab_features
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [torch.nn.Linear(d, hidden_dim), torch.nn.GELU(),
                       torch.nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, stat_feats, tab_feats):
        x = torch.cat([stat_feats, tab_feats], dim=-1)
        return self.net(x).squeeze(-1)


class TorchSundialTrainer(BaseTrainer):
    """Frozen Sundial flow-matching backbone -> per-row path-stat MLP head.

    Per row: pass last `lookback` steps of channel `channel_index` through
    SundialForPrediction.forward(num_samples=K). Output is (B, K, 720)
    sampled future patches; truncate to `forecast_length`, summarise into
    13 distribution-shape statistics, concat with last-step tabular vector,
    train MLP classifier head. Backbone is frozen and cached across windows.
    """

    name = 'torch_sundial'
    consumes_sequences = True

    def __init__(self,
                 model_id: str = 'thuml/sundial-base-128m',
                 model_dir: str = 'data/sundial_base_128m',
                 channel_index: int = -1,
                 lookback: int = 20,
                 forecast_length: int = 5,
                 num_samples: int = 16,
                 hidden_dim: int = 128,
                 n_head_layers: int = 2,
                 dropout: float = 0.15,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 n_epochs: int = 40,
                 pos_weight: float = 1.3,
                 use_tabular_features: bool = True,
                 inference_batch_size: int = 256,
                 device: str = None,
                 seed: int = 1234,
                 **kwargs):
        self.model_id = str(model_id)
        self.model_dir = model_dir
        self.channel_index = int(channel_index)
        self.lookback = int(lookback)
        self.forecast_length = int(forecast_length)
        self.num_samples = int(num_samples)
        self.hidden_dim = int(hidden_dim)
        self.n_head_layers = int(n_head_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.pos_weight = float(pos_weight)
        self.use_tabular_features = bool(use_tabular_features)
        self.inference_batch_size = int(inference_batch_size)
        self.seed = int(seed)
        self._device = torch.device(device) if device else None
        self._backbone = None
        self._head = None
        self._n_stat_features = 13
        self._tab_dim = None
        self._scaler_mean = None
        self._scaler_std = None

    def _resolve_device(self):
        if self._device is not None:
            return self._device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_backbone(self):
        if self._backbone is not None:
            return self._backbone
        device = self._resolve_device()
        key = (self.model_id, self.model_dir, str(device))
        if key in _SUNDIAL_CACHE:
            self._backbone = _SUNDIAL_CACHE[key]
            return self._backbone
        # Compat shims: Sundial's modeling code uses transformers<=4.42 cache API.
        from transformers.cache_utils import DynamicCache, Cache
        if not hasattr(DynamicCache, 'seen_tokens'):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(Cache, 'get_max_length'):
            Cache.get_max_length = lambda self: None
        if not hasattr(DynamicCache, 'get_usable_length'):
            DynamicCache.get_usable_length = (
                lambda self, new_seq_length=None, layer_idx=0:
                self.get_seq_length(layer_idx)
            )
        from transformers import AutoModelForCausalLM
        src = self.model_dir if (self.model_dir and os.path.isdir(self.model_dir)) \
            else self.model_id
        backbone = AutoModelForCausalLM.from_pretrained(
            src, trust_remote_code=True).to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        _SUNDIAL_CACHE[key] = backbone
        self._backbone = backbone
        return backbone

    @torch.no_grad()
    def _sample_paths(self, seqs: np.ndarray) -> np.ndarray:
        """seqs: (N, lookback) -> (N, num_samples, forecast_length)."""
        backbone = self._load_backbone()
        device = self._resolve_device()
        # Sundial's forward(revin=True) has a broken broadcast shape; do revin
        # manually so we can pass revin=False inside.
        out_chunks = []
        n = seqs.shape[0]
        bs = max(1, self.inference_batch_size)
        for i in range(0, n, bs):
            chunk = torch.from_numpy(seqs[i:i + bs]).float().to(device)
            means = chunk.mean(1, keepdim=True)
            stdev = chunk.std(1, keepdim=True, unbiased=False).clamp_min(1e-2)
            chunk_n = (chunk - means) / stdev
            out = backbone(input_ids=chunk_n, num_samples=self.num_samples,
                           use_cache=False, return_dict=True, revin=False)
            preds = out.logits  # (B, K, 720)
            preds = preds * stdev.unsqueeze(1) + means.unsqueeze(1)
            preds = preds[:, :, :self.forecast_length]
            out_chunks.append(preds.detach().float().cpu().numpy())
        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    @staticmethod
    def _path_stats(paths: np.ndarray) -> np.ndarray:
        """paths: (N, K, T) -> (N, 13). Treat each sample's K-step path as a
        candidate return trajectory; summarise distribution shape."""
        path_mean = paths.mean(axis=(1, 2))
        path_std = paths.std(axis=(1, 2))
        last_mean = paths[:, :, -1].mean(axis=1)
        last_std = paths[:, :, -1].std(axis=1)
        cumret = paths.sum(axis=2)  # (N, K)
        p_up = (cumret > 0).mean(axis=1).astype(np.float32)
        p_strong_up = (cumret > 0.02).mean(axis=1).astype(np.float32)
        p_strong_dn = (cumret < -0.02).mean(axis=1).astype(np.float32)
        q25 = np.quantile(cumret, 0.25, axis=1)
        q50 = np.quantile(cumret, 0.50, axis=1)
        q75 = np.quantile(cumret, 0.75, axis=1)
        iqr = q75 - q25
        cum = np.cumsum(paths, axis=2)
        running_max = np.maximum.accumulate(cum, axis=2)
        drawdown = (cum - running_max).min(axis=2)
        mdd_mean = drawdown.mean(axis=1)
        sigma = cumret.std(axis=1) + 1e-9
        skew = ((cumret - cumret.mean(axis=1, keepdims=True)) ** 3).mean(axis=1) / (sigma ** 3)
        stats = np.stack([path_mean, path_std, last_mean, last_std,
                          p_up, p_strong_up, p_strong_dn,
                          q25, q50, q75, iqr, mdd_mean, skew], axis=-1).astype(np.float32)
        return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_inputs(self, X_seq: np.ndarray):
        """X_seq: (N, L, F) -> (stats (N, 13), tab (N, F or 0))."""
        L = X_seq.shape[1]
        F = X_seq.shape[2]
        ci = self.channel_index if self.channel_index >= 0 else (F + self.channel_index)
        ci = max(0, min(int(ci), F - 1))
        take = min(self.lookback, L)
        seqs = X_seq[:, -take:, ci].astype(np.float32)
        seqs = np.nan_to_num(seqs, nan=0.0, posinf=0.0, neginf=0.0)
        if take < self.lookback:
            pad = np.zeros((seqs.shape[0], self.lookback - take), dtype=seqs.dtype)
            seqs = np.concatenate([pad, seqs], axis=1)
        paths = self._sample_paths(seqs)
        stats = self._path_stats(paths)
        if self.use_tabular_features:
            tab = X_seq[:, -1, :].astype(np.float32)
            tab = np.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            tab = np.zeros((X_seq.shape[0], 0), dtype=np.float32)
        return stats, tab

    def _fit_scaler(self, F):
        self._scaler_mean = F.mean(axis=0).astype(np.float32)
        self._scaler_std = (F.std(axis=0) + 1e-6).astype(np.float32)

    def _apply_scaler(self, F):
        return ((F - self._scaler_mean) / self._scaler_std).astype(np.float32)

    def fit(self, X_train, y_train, X_val, y_val, verbose=False,
            pnl_train=None, pnl_val=None, dates_train=None, dates_val=None,
            **kwargs):
        X_tr = np.asarray(X_train, dtype=np.float32)
        if X_tr.ndim != 3:
            raise ValueError(
                f'torch_sundial expects 3D sequence input (N, L, F); got shape '
                f'{X_tr.shape}.')
        X_va = np.asarray(X_val, dtype=np.float32) if X_val is not None else None
        y_tr = np.asarray(y_train, dtype=np.float32).reshape(-1)
        y_va = np.asarray(y_val, dtype=np.float32).reshape(-1) if y_val is not None else None

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        stat_tr, tab_tr = self._build_inputs(X_tr)
        # Fit per-feature scaler on the concatenated [stat, tab] matrix so
        # the MLP head sees comparable scales (path stats are O(1) but tab
        # features are already RobustScaler'd by the gate).
        F_tr = np.concatenate([stat_tr, tab_tr], axis=1)
        self._fit_scaler(F_tr)
        F_tr = self._apply_scaler(F_tr)
        self._tab_dim = tab_tr.shape[1]

        has_val = X_va is not None and y_va is not None and len(y_va) > 0
        if has_val:
            stat_va, tab_va = self._build_inputs(X_va)
            F_va = self._apply_scaler(np.concatenate([stat_va, tab_va], axis=1))
        else:
            F_va = None

        device = self._resolve_device()
        self._head = _SundialPathStatHead(
            n_stat_features=self._n_stat_features,
            n_tab_features=self._tab_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_head_layers,
            dropout=self.dropout,
        ).to(device)
        opt = torch.optim.AdamW(self._head.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        pw = torch.tensor([self.pos_weight], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

        stat_tr_t = torch.from_numpy(F_tr[:, :self._n_stat_features]).to(device)
        tab_tr_t = torch.from_numpy(F_tr[:, self._n_stat_features:]).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        if has_val:
            stat_va_t = torch.from_numpy(F_va[:, :self._n_stat_features]).to(device)
            tab_va_t = torch.from_numpy(F_va[:, self._n_stat_features:]).to(device)
            y_va_t = torch.from_numpy(y_va).to(device)

        n = stat_tr_t.shape[0]
        best_val = float('inf')
        best_state = None
        patience, bad = 6, 0
        for epoch in range(self.n_epochs):
            self._head.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                logits = self._head(stat_tr_t[idx], tab_tr_t[idx])
                loss = loss_fn(logits, y_tr_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._head.parameters(), 1.0)
                opt.step()
            if has_val:
                self._head.eval()
                with torch.no_grad():
                    vloss = loss_fn(self._head(stat_va_t, tab_va_t), y_va_t).item()
                if vloss < best_val - 1e-5:
                    best_val = vloss
                    best_state = {k: v.detach().clone()
                                  for k, v in self._head.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self._head.load_state_dict(best_state)
        self._head.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        if self._head is None:
            raise RuntimeError('TorchSundialTrainer.predict_proba before fit')
        X3 = np.asarray(X, dtype=np.float32)
        if X3.ndim != 3:
            raise ValueError(
                f'torch_sundial.predict_proba expects 3D input; got {X3.shape}.')
        stat, tab = self._build_inputs(X3)
        F_te = self._apply_scaler(np.concatenate([stat, tab], axis=1))
        device = self._resolve_device()
        out = []
        bs = max(self.batch_size, 1)
        self._head.eval()
        for i in range(0, F_te.shape[0], bs):
            chunk = F_te[i:i + bs]
            st = torch.from_numpy(chunk[:, :self._n_stat_features]).to(device)
            tb = torch.from_numpy(chunk[:, self._n_stat_features:]).to(device)
            p = torch.sigmoid(self._head(st, tb)).cpu().numpy().astype(np.float64)
            out.append(p)
        return np.concatenate(out, axis=0)

    @property
    def hyperparams(self):
        return dict(
            model_id=self.model_id, model_dir=self.model_dir,
            channel_index=self.channel_index, lookback=self.lookback,
            forecast_length=self.forecast_length, num_samples=self.num_samples,
            hidden_dim=self.hidden_dim, n_head_layers=self.n_head_layers,
            dropout=self.dropout, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, batch_size=self.batch_size,
            n_epochs=self.n_epochs, pos_weight=self.pos_weight,
            use_tabular_features=self.use_tabular_features,
            inference_batch_size=self.inference_batch_size, seed=self.seed,
        )

    def save(self, output_dir, extra=None):
        os.makedirs(output_dir, exist_ok=True)
        state_path = os.path.join(output_dir, 'sundial_head.pt')
        torch.save({
            'state_dict': self._head.state_dict() if self._head is not None else None,
            'scaler_mean': self._scaler_mean,
            'scaler_std': self._scaler_std,
            'tab_dim': self._tab_dim,
        }, state_path)
        meta = dict(self.hyperparams, tab_dim=self._tab_dim)
        if extra:
            meta.update(extra)
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        return {'state': state_path, 'meta': meta_path}


TRAINERS = {
    'lightgbm': LightGBMTrainer,
    'lightgbm_regressor': LightGBMRegressorTrainer,
    'lightgbm_ranker': LightGBMRankerTrainer,
    'xgboost': XGBoostTrainer,
    'xgb_regressor': XGBoostRegressorTrainer,
    'bagged_xgb_regressor': BaggedXGBRegressorTrainer,
    'xgb_huber_regressor': XGBoostHuberRegressorTrainer,
    'xgb_quantile': XGBoostQuantileTrainer,
    'xgb_dual_quantile': XGBoostDualQuantileTrainer,
    'xgb_ranker': XGBoostRankerTrainer,
    'xgb_ranker_dart': XGBoostRankerDartTrainer,
    'xgb_win_ranker': XGBoostWinRankerTrainer,
    'ev_gated_ranker': EVGatedRankerTrainer,
    'bagged_ev_gated_ranker': BaggedEVGatedRankerTrainer,
    'stacked_ranker': StackedRankerTrainer,
    'xgb_magnitude_classifier': XGBoostMagnitudeWeightedTrainer,
    'xgb_strict_win': XGBoostStrictWinClassifierTrainer,
    'xgb_focal_loss': XGBoostFocalLossClassifierTrainer,
    'xgb_group_balanced_focal': XGBoostGroupBalancedFocalLossTrainer,
    'xgb_temporal_mixup': XGBoostTemporalMixupTrainer,
    'xgb_jtt': XGBoostJTTTrainer,
    'xgb_quarterly_dro': XGBoostQuarterlyDROTrainer,
    'xgb_topk_classifier': XGBoostTopKClassifierTrainer,
    'xgb_adv_val': XGBoostAdversarialValidationTrainer,
    'xgb_meta_label': XGBoostMetaLabelingTrainer,
    'xgb_meta_label_regime': XGBoostMetaLabelingRegimeTrainer,
    'xgb_mcdropout_classifier': XGBoostMCDropoutClassifierTrainer,
    'xgb_rank_fusion': XGBoostRankFusionTrainer,
    'xgb_regime_blend': XGBoostRegimeBlendTrainer,
    'xgb_recency_consensus': XGBoostRecencyConsensusTrainer,
    'xgb_day_quality_consensus': XGBoostDayQualityConsensusTrainer,
    'torch_attentive_mlp': TorchAttentiveMLPTrainer,
    'torch_seq_gru': TorchSeqGRUTrainer,
    'torch_seq_transformer': TorchSeqTransformerTrainer,
    'torch_seq_gru_ensemble': TorchSeqGRUEnsembleTrainer,
    'torch_seq_gru_abstain': TorchSeqGRUAbstainTrainer,
    'torch_seq_gru_day_gate': TorchSeqGRUDayGateTrainer,
    'sklearn_extra_trees': SklearnExtraTreesTrainer,
    'logistic_elastic_net': LogisticElasticNetTrainer,
    'knn_classifier': KNNClassifierTrainer,
    'qda_classifier': QDAClassifierTrainer,
    'gaussian_nb': GaussianNaiveBayesClassifierTrainer,
    'xgb_iso_calibrated': XGBoostIsotonicCalibratedTrainer,
    'kernel_logreg': KernelLogRegTrainer,
    'gaussian_process_classifier': GaussianProcessClassifierTrainer,
    'mlp_classifier': MLPClassifierTrainer,
    'tabpfn_v25': TabPFNV25Trainer,
    'torch_time_moe': TorchTimeMoETrainer,
    'tabm_classifier': TabMClassifierTrainer,
    'tabm_lcb': TabMLCBClassifierTrainer,
    'histgb_monotonic': HistGBMonotonicTrainer,
    'histgb_monotonic_bagged': HistGBMonotonicBaggedTrainer,
    'torch_itransformer': TorchITransformerTrainer,
    'torch_tabnet': TorchTabNetTrainer,
    'torch_mamba': TorchMambaTrainer,
    'torch_patchtst': TorchPatchTSTTrainer,
    'tabicl_v2': TabICLv2Trainer,
    'torch_chronos2': TorchChronos2EmbedTrainer,
    'anomaly_gated_histgb': AnomalyGatedHistGBTrainer,
    'dae_logreg': DAELogRegTrainer,
    'torch_frets': TorchFreTSTrainer,
    'kernel_anomaly_blend': KernelAnomalyBlendTrainer,
    'rocket_classifier': ROCKETClassifierTrainer,
    'torch_xlstm': TorchXLSTMTrainer,
    'kernel_histgb_stack': KernelHistGBStackTrainer,
    'rocket_bagged_calibrated': ROCKETBaggedCalibratedTrainer,
    'torch_timesfm': TorchTimesFMTrainer,
    'modernnca': ModernNCATrainer,
    'torch_sundial': TorchSundialTrainer,
}


def get_trainer(name: str, **kwargs) -> BaseTrainer:
    if name not in TRAINERS:
        raise ValueError(
            f'Unknown trainer {name!r}. Choices: {list(TRAINERS)}')
    cls = TRAINERS[name]
    valid_kwargs = {k: v for k, v in kwargs.items()
                    if k in cls.__init__.__code__.co_varnames}
    return cls(**valid_kwargs)
