#!/usr/bin/env python3
"""Separate real-data CPU/GPU benchmark; exports aggregates only, never predictions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_backend import (build_baselines, fit_evaluate, require_gpu,
                              runtime_metadata, synchronize)
from fakenewsnet_prefix import make_splits, split_fingerprint

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('accuracy', 'auc', 'pr_auc', 'f1', 'f1_tuned')


def sample_indices(y, samples=10000):
    if not 10 <= samples <= len(y):
        raise ValueError('samples must be between 10 and the dataset size')
    if samples == len(y):
        return np.arange(len(y))
    selected, _ = train_test_split(np.arange(len(y)), train_size=samples,
                                   stratify=y, random_state=42)
    return selected


def load_features(mode, indices):
    import torch
    suffix = '' if mode == 'bert' else '_mistral'
    parts = []
    for kind in ('title', 'desc'):
        values = torch.load(ROOT / f'evons/data/{kind}_embeddings{suffix}.pt', map_location='cpu')
        parts.append(torch.as_tensor(values, dtype=torch.float32)[indices].numpy().copy())
        del values
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def write_report(out, rows, manifest):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'results.csv', index=False)
    comparisons = []
    for encoder in frame.encoder.unique():
        for name in ('logreg', 'rf'):
            pair = frame[(frame.encoder == encoder) & (frame.model == name)]
            if set(pair.backend) != {'cpu', 'gpu'}:
                continue
            cpu, gpu = (pair[pair.backend == b].iloc[0] for b in ('cpu', 'gpu'))
            comparison = {'encoder': encoder, 'model': name,
                          'fit_speedup': cpu.fit_seconds / gpu.fit_seconds,
                          'total_speedup': cpu.total_seconds / gpu.total_seconds}
            comparison.update({f'delta_{m}': gpu[m] - cpu[m] for m in METRICS})
            comparison['quality_review'] = any(abs(comparison[f'delta_{m}']) > .03
                                               for m in ('auc', 'pr_auc', 'f1_tuned'))
            comparisons.append(comparison)
    compared = pd.DataFrame(comparisons)
    compared.to_csv(out / 'comparison.csv', index=False)
    manifest['completed_cases'] = len(rows)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    report = ['# Confronto baseline CPU/GPU', '',
              f'Campione stratificato EVONS: {manifest["samples"]} righe; seed 42; primo split 60/20/20.',
              'BERT e Mistral condividono campione e split. Nessun addestramento MLP.', '',
              'I delta sono GPU meno CPU; speedup > 1 indica GPU più veloce.', '',
              compared.to_markdown(index=False, floatfmt='.4f') if len(compared) else 'Confronto in corso.', '',
              '## Tempi e metriche per esecuzione', '',
              frame[['encoder', 'model', 'backend', 'fit_seconds', 'predict_seconds',
                     'calibration_evaluation_seconds', 'total_seconds', *METRICS]].to_markdown(index=False, floatfmt='.4f'), '',
              '## Limiti e metodo', '',
              '- Un solo split su un campione: nessuna stima di variabilità o garanzia sullo speedup del dataset completo.',
              '- RF cuML usa split quantizzati (128 bin); LogReg usa QN. Le differenze non sono soltanto floating point.',
              '- Campionamento, caricamento e warmup sono esclusi dai tempi operativi; i trasferimenti nei fit/predict sono inclusi.',
              '- Calibrazione e metriche usano lo stesso codice CPU di produzione, senza ottimizzazioni.',
              '- Scarti assoluti > 0,03 in ROC-AUC, PR-AUC o F1 calibrato sono segnalati per approfondimento.',
              '- Versioni, parametri, fingerprint e stato dei processi iniziali sono nel manifest.',
              f'- Concorrenza rilevata all’avvio: {manifest["concurrent_training"]}.', '']
    (out / 'report.md').write_text('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'revision_results/backend_benchmark')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cp = require_gpu()
    start = time.perf_counter()
    warm_X = np.random.default_rng(42).normal(size=(256, 8)).astype(np.float32)
    warm_y = np.arange(256) % 2
    for model in build_baselines('gpu', record=False).values():
        model.fit(warm_X, warm_y)
        model.predict_proba(warm_X)
    synchronize('gpu')
    warmup = time.perf_counter() - start
    y_all = pd.read_csv(ROOT / 'evons/data/evons.csv', usecols=['is_fake']).is_fake.astype(int).to_numpy()
    indices = sample_indices(y_all, args.samples)
    y = y_all[indices]
    fit, cal, test = make_splits(y, n_splits=1)[0]
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
    concurrent = [line for line in processes if any(token in line for token in (
        'colleague_revision_run.py --stage', 'evons_disinformation_detection.py', 'ipykernel_launcher'))]
    manifest = {**runtime_metadata(), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'samples': len(y), 'seed': 42, 'warmup_seconds': warmup,
                'sample_fingerprint': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                'split_fingerprint': split_fingerprint(fit, cal, test),
                'concurrent_training': concurrent, 'models': {}, 'feature_loading_seconds': {},
                'gpu': {str(k): v for k, v in cp.cuda.runtime.getDeviceProperties(0).items()
                        if k in ('name', 'totalGlobalMem', 'major', 'minor')},
                'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
                'cuda_driver': cp.cuda.runtime.driverGetVersion(),
                'benchmark_policy': 'Single split, one timed fit per backend; fresh estimator instances after GPU warmup.',
                'cpu_timing_policy': 'Original n_jobs=-1 and native thread pool defaults'}
    rows = []
    for encoder in ('bert', 'mistral'):
        start = time.perf_counter()
        X = load_features(encoder, indices)
        manifest['feature_loading_seconds'][encoder] = time.perf_counter() - start
        for name in ('logreg', 'rf'):
            for backend in ('cpu', 'gpu'):
                models = build_baselines(backend, record=False)
                model = models[name]
                manifest['models'][f'{backend}_{name}'] = model.get_params()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    metrics, times = fit_evaluate(model, X, y, fit, cal, test,
                                                 backend=backend, label=f'{encoder} {name}')
                row = {'encoder': encoder, 'model': name, 'backend': backend,
                       'fold': 1, 'n_fit': len(fit), 'n_calibration': len(cal),
                       'split_fingerprint': manifest['split_fingerprint'],
                       **metrics, **times, 'warnings': ' | '.join(str(w.message) for w in caught)}
                if rows and set(row) != set(rows[0]):
                    raise RuntimeError('CPU/GPU result schemas differ')
                rows.append(row)
                write_report(args.out_dir, rows, manifest)
                del model, models
        del X
    print(f'Report: {args.out_dir / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
