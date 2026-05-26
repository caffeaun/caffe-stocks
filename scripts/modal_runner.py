"""Modal app for foundation-model trainers that don't fit on the local 12 GB GPU.

Functions are dispatched by trainer name. Each function builds its own image
with just the deps it needs so the cold-start cost is minimised per call.

Deploy once after edits:
    venv/bin/modal deploy scripts/modal_runner.py

Local invocation from trainers.py:
    import modal
    fn = modal.Function.lookup('caffe-stocks-modal', 'train_predict_tabicl')
    proba = fn.remote(X_train, y_train, X_test, hp_dict)
        # → numpy array of shape (n_test,) — P(class=1)

Monthly spend cap is enforced in scripts/modal_budget.py — adapter code in
trainers.py is expected to call check_budget() before .remote() and
record_call() after.
"""

from __future__ import annotations

import modal

APP_NAME = 'caffe-stocks-modal'
app = modal.App(APP_NAME)


# Image: torch (CUDA build), numpy, tabicl. Checkpoint download is deferred
# to first call; Modal caches /root between calls within a warm container,
# and the function is configured to keep the container warm for `keep_warm`.
# Two pip steps: torch from the PyTorch CUDA index (only place torch+cu124
# lives), then the rest from PyPI (PyTorch index lacks scikit-learn/scipy).
tabicl_image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install(
        'torch==2.5.0',
        extra_options='--index-url https://download.pytorch.org/whl/cu124',
    )
    .pip_install('numpy', 'scipy', 'scikit-learn', 'tabicl')
)


@app.function(
    image=tabicl_image,
    gpu='A100-40GB',
    timeout=900,            # 15 min per call — covers full 7-split gate
    memory=32768,           # 32 GB RAM
    scaledown_window=120,   # keep warm for 2 min between calls
)
def train_predict_tabicl(
    X_train,     # numpy.ndarray (n_train, n_features)
    y_train,     # numpy.ndarray (n_train,)
    X_test,      # numpy.ndarray (n_test, n_features)
    hp: dict,    # TabICLClassifier kwargs (excluding device/random_state)
    random_state: int = 42,
) -> 'numpy.ndarray':
    """One-shot fit+predict on Modal A100. Returns P(class=1) of shape (n_test,)."""
    import numpy as np
    import torch
    from tabicl import TabICLClassifier

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64).ravel()
    X_test = np.asarray(X_test, dtype=np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[modal/tabicl] device={device}  X_train={X_train.shape}  '
          f'X_test={X_test.shape}  hp={hp}')

    model = TabICLClassifier(
        device=device,
        random_state=random_state,
        allow_auto_download=True,
        **hp,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    proba = np.asarray(proba, dtype=np.float32)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.ravel()


@app.local_entrypoint()
def smoke():
    """`modal run scripts/modal_runner.py` to verify the function deploys.

    Generates 200 synthetic rows, fits + predicts, prints first 5 probas.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype('float32')
    y = (X[:, 0] > 0).astype('int64')
    hp = {
        'n_estimators': 2,
        'batch_size': 64,
        'use_amp': True,
        'kv_cache': False,
        'offload_mode': 'auto',
    }
    proba = train_predict_tabicl.remote(X[:150], y[:150], X[150:], hp)
    print(f'smoke OK: proba[:5] = {proba[:5]}')
