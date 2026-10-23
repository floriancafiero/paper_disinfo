"""Explicit CPU/cuML baseline selection; production metrics remain NumPy based."""
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import os
import sys
import time

import numpy as np


def resolve_backend(backend=None):
    backend = backend if backend is not None else os.environ.get('BASELINE_BACKEND', 'cpu')
    if backend not in ('cpu', 'gpu'):
        raise ValueError('BASELINE_BACKEND must be cpu or gpu')
    return backend


@contextmanager
def backend_environment(backend):
    previous = os.environ.get('BASELINE_BACKEND')
    os.environ['BASELINE_BACKEND'] = resolve_backend(backend)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('BASELINE_BACKEND', None)
        else:
            os.environ['BASELINE_BACKEND'] = previous


def require_gpu():
    try:
        import cupy as cp
        import cuml
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device available')
        cp.cuda.runtime.free(0)
    except Exception as exc:
        raise RuntimeError('GPU baseline requested, but cuML/CUDA is unavailable. '
                           'Use the paper_disinfo GPU environment or select cpu.') from exc
    return cp


def synchronize(backend):
    if resolve_backend(backend) == 'gpu':
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()


def build_baselines(backend=None, *, logreg_random_state=None, record=True):
    backend = resolve_backend(backend)
    if backend == 'cpu':
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        models = {
            'logreg': LogisticRegression(max_iter=1000, n_jobs=-1,
                                         random_state=logreg_random_state),
            'rf': RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
        }
    else:
        require_gpu()
        from cuml.linear_model import LogisticRegression
        from cuml.ensemble import RandomForestClassifier
        models = {
            'logreg': LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,
                                        max_iter=1000, tol=1e-4, solver='qn',
                                        output_type='numpy'),
            'rf': RandomForestClassifier(n_estimators=250, split_criterion='gini',
                                        bootstrap=True, max_samples=1.0,
                                        max_features='sqrt', max_depth=None,
                                        min_samples_split=2, min_samples_leaf=1,
                                        random_state=42, n_bins=128, n_streams=4,
                                        output_type='numpy'),
        }
    if record:
        record_manifest(backend, models)
    return models


def runtime_metadata():
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'cuml-cu13', 'cupy-cuda13x'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'versions': versions}


def record_manifest(backend, models):
    folder = Path(__file__).resolve().parents[1] / 'revision_results/backend_manifests'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {**runtime_metadata(), 'backend': backend, 'pid': os.getpid(),
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'models': {name: model.get_params() for name, model in models.items()}}
    (folder / f'{os.getpid()}.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    print(f'Baseline backend: {backend}; versions={manifest["versions"]}', flush=True)


def positive_probability(model, features):
    classes = model.classes_
    if hasattr(classes, 'get'):
        classes = classes.get()
    classes = np.asarray(classes)
    column = np.flatnonzero(classes == 1)
    if len(column) != 1:
        raise ValueError('Expected a fitted binary classifier with positive class 1')
    values = model.predict_proba(features)
    if hasattr(values, 'get'):
        values = values.get()
    values = np.asarray(values)
    if values.shape != (len(features), len(classes)):
        raise ValueError('Unexpected predict_proba shape')
    result = values[:, column[0]]
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError('Invalid predicted probabilities')
    return result


def fit_evaluate(model, X, y, fit, cal, test, *, backend=None, label='baseline'):
    from imbalanced_evaluation import evaluate_with_calibration
    backend = resolve_backend(backend)
    print(f'{label}: backend={backend}, fit starting', flush=True)
    synchronize(backend)
    start = time.perf_counter()
    model.fit(X[fit], y[fit])
    synchronize(backend)
    fitted = time.perf_counter()
    p_test = positive_probability(model, X[test])
    p_cal = positive_probability(model, X[cal])
    synchronize(backend)
    predicted = time.perf_counter()
    metrics = evaluate_with_calibration(y[test], p_test, y[cal], p_cal)
    end = time.perf_counter()
    times = {'fit_seconds': fitted - start, 'predict_seconds': predicted - fitted,
             'calibration_evaluation_seconds': end - predicted, 'total_seconds': end - start}
    print(f'{label}: ' + ', '.join(f'{k}={v:.3f}' for k, v in times.items()), flush=True)
    return metrics, times
