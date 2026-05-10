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
# Registry — add new model types here
# --------------------------------------------------------------------- #
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
    # 'neural': NeuralTrainer,  # add when v2 NN trainer lands
}


def get_trainer(name: str, **kwargs) -> BaseTrainer:
    if name not in TRAINERS:
        raise ValueError(
            f'Unknown trainer {name!r}. Choices: {list(TRAINERS)}')
    cls = TRAINERS[name]
    valid_kwargs = {k: v for k, v in kwargs.items()
                    if k in cls.__init__.__code__.co_varnames}
    return cls(**valid_kwargs)
