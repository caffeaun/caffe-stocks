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

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False):
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

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False):
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

    def fit(self, X_train, y_train, X_val, y_val, verbose: bool = False):
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
# Registry — add new model types here
# --------------------------------------------------------------------- #
TRAINERS = {
    'lightgbm': LightGBMTrainer,
    'xgboost': XGBoostTrainer,
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
