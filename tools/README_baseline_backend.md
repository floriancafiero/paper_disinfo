# Baseline CPU/GPU selezionabili

RF e regressione logistica possono usare scikit-learn (`cpu`, predefinito) oppure
cuML diretto (`gpu`). MLP e XGBoost mantengono il comportamento precedente.
Non viene usato `cuml.accel` e non esiste fallback automatico CPU.

## Ambiente e avvio

Usare l'ambiente conda `paper_disinfo`, anche per il kernel dei notebook:

```bash
conda activate paper_disinfo
python -m pip install -r requirements-gpu.txt
python -m ipykernel install --user --name paper_disinfo --display-name 'Python (paper_disinfo)'
python tools/colleague_revision_run.py --stage train --baseline-backend gpu --kernel-name paper_disinfo > training_gpu.log 2>&1
```

Il kernel è già stato registrato su questa istanza. La dipendenza GPU è opzionale:
un ambiente CPU non deve installare cuML. cuML richiede toolkit CUDA 13 con NVRTC;
la sola disponibilità di PyTorch CUDA non dimostra che questi requisiti siano soddisfatti.
Su questa istanza sono stati verificati CUDA 13.2, driver 595 e NVIDIA L4.

Per CPU, usare `--baseline-backend cpu --kernel-name paper_disinfo`.
Il runner dei quantili accetta le stesse opzioni. L'opzione CLI prevale su
`BASELINE_BACKEND`; senza entrambe si usa CPU. Negli script diretti e nei notebook:

```bash
BASELINE_BACKEND=gpu python colab_no_embeddings/evons_disinformation_detection.py
python tools/run_virality_quantile_grid.py --dataset fakenewsnet --baseline-backend gpu --kernel-name paper_disinfo
```

Per esecuzione interattiva, impostare `os.environ['BASELINE_BACKEND'] = 'gpu'`
prima della cella iniziale e scegliere il kernel `paper_disinfo`.
La scelta viene propagata ai sottoprocessi e ai kernel lanciati dai runner.
GPU non disponibile, dipendenze mancanti ed esaurimento della memoria causano un errore.

## Contratto e parametri

Restano invariati nomi dei file CSV, colonne, nomi dei modelli, split, target,
feature, calibrazione e aggregazione. Le probabilità della classe 1 tornano in
NumPy; le metriche e la scelta della soglia continuano sulla CPU. Le esecuzioni
complete CPU/GPU scrivono negli stessi percorsi di produzione: archiviare i risultati
precedenti prima di confrontare due training completi.

I parametri CPU rimangono quelli originali. GPU LogReg usa L2, C=1, intercetta,
1.000 iterazioni massime, tolleranza 1e-4 e solver QN. GPU RF usa 250 alberi,
Gini, bootstrap con rapporto 1, max_features=sqrt, profondità illimitata,
min_samples_split=2, min_samples_leaf=1, seed 42, 128 bin e 4 stream.
RF cuML costruisce split quantizzati e LogReg usa un solver differente:
non si tratta soltanto di differenze floating point. Il seed non garantisce
identità numerica tra backend o esecuzioni GPU.

Backend, versioni e parametri sono registrati separatamente in
`revision_results/backend_manifests/<pid>.json`. Il log riporta fit, predizione e
calibrazione/valutazione per modello, encoder e split. I manifest non entrano
nell'archivio aggregati esistente; la sua allowlist resta invariata.

## Benchmark separato

```bash
conda activate paper_disinfo
python tools/benchmark_baseline_backends.py --samples 10000
```

Il benchmark usa un campione stratificato EVONS (seed 42), entrambi gli encoder,
un solo split 60/20/20 e otto fit: due modelli per due encoder per due backend.
Non avvia MLP, non cambia i risultati di produzione e non esporta dati individuali.
Scrive CSV aggregati, manifest e `report.md` in
`revision_results/backend_benchmark/`; `--out-dir` permette una destinazione distinta.
Le misure operative includono i trasferimenti nei fit/predict, escludono caricamento
e warmup e separano il costo CPU della calibrazione. Il benchmark mantiene n_jobs=-1
e i thread nativi del training CPU; non applica ottimizzazioni ai thread.
Il rapporto segnala scarti assoluti >0,03 in AUC, PR-AUC o F1 calibrato.
Un campione e uno split non permettono di stimare variabilità o speedup sull'intero corpus.

Test del contratto e integrazione CUDA:

```bash
RUN_GPU_TESTS=1 python -m unittest discover -s tests -p test_baseline_backend.py -v
```

Riferimenti: [RF cuML](https://docs.nvidia.com/cuml/26.10/api/generated/cuml.ensemble.RandomForestClassifier/),
[LogReg cuML](https://docs.nvidia.com/cuml/26.10/api/generated/cuml.linear_model.LogisticRegression/),
[installazione RAPIDS](https://docs.rapids.ai/install/).
