# Confronto baseline CPU/GPU

Campione stratificato EVONS: 10000 righe; seed 42; primo split 60/20/20.
BERT e Mistral condividono campione e split. Nessun addestramento MLP.

I delta sono GPU meno CPU; speedup > 1 indica GPU più veloce.

| encoder   | model   |   fit_speedup |   total_speedup |   delta_accuracy |   delta_auc |   delta_pr_auc |   delta_f1 |   delta_f1_tuned | quality_review   |
|:----------|:--------|--------------:|----------------:|-----------------:|------------:|---------------:|-----------:|-----------------:|:-----------------|
| bert      | logreg  |        8.6029 |          1.3935 |           0.0000 |      0.0001 |         0.0001 |     0.0001 |           0.0015 | False            |
| bert      | rf      |        0.4827 |          0.5684 |          -0.0020 |      0.0016 |         0.0016 |    -0.0022 |           0.0055 | False            |
| mistral   | logreg  |        3.5979 |          1.2417 |          -0.0005 |      0.0000 |         0.0000 |    -0.0006 |           0.0000 | False            |
| mistral   | rf      |        0.4108 |          0.4861 |          -0.0015 |      0.0001 |         0.0008 |    -0.0012 |          -0.0012 | False            |

## Tempi e metriche per esecuzione

| encoder   | model   | backend   |   fit_seconds |   predict_seconds |   calibration_evaluation_seconds |   total_seconds |   accuracy |    auc |   pr_auc |     f1 |   f1_tuned |
|:----------|:--------|:----------|--------------:|------------------:|---------------------------------:|----------------:|-----------:|-------:|---------:|-------:|-----------:|
| bert      | logreg  | cpu       |        1.8989 |            0.0700 |                           5.4609 |          7.4297 |     0.9510 | 0.9892 |   0.9887 | 0.9461 |     0.9454 |
| bert      | logreg  | gpu       |        0.2207 |            0.0375 |                           5.0737 |          5.3319 |     0.9510 | 0.9893 |   0.9888 | 0.9462 |     0.9469 |
| bert      | rf      | cpu       |        5.0902 |            0.6423 |                           0.6762 |          6.4087 |     0.9280 | 0.9782 |   0.9769 | 0.9199 |     0.9162 |
| bert      | rf      | gpu       |       10.5445 |            0.0164 |                           0.7131 |         11.2740 |     0.9260 | 0.9798 |   0.9785 | 0.9177 |     0.9217 |
| mistral   | logreg  | cpu       |        0.2973 |            0.0177 |                           6.1014 |          6.4165 |     0.9600 | 0.9937 |   0.9931 | 0.9561 |     0.9576 |
| mistral   | logreg  | gpu       |        0.0826 |            0.0222 |                           5.0625 |          5.1674 |     0.9595 | 0.9938 |   0.9931 | 0.9556 |     0.9576 |
| mistral   | rf      | cpu       |        5.0842 |            0.6402 |                           0.5931 |          6.3175 |     0.9160 | 0.9796 |   0.9779 | 0.9034 |     0.9186 |
| mistral   | rf      | gpu       |       12.3771 |            0.0200 |                           0.5990 |         12.9962 |     0.9145 | 0.9797 |   0.9787 | 0.9022 |     0.9174 |

## Limiti e metodo

- Un solo split su un campione: nessuna stima di variabilità o garanzia sullo speedup del dataset completo.
- RF cuML usa split quantizzati (128 bin); LogReg usa QN. Le differenze non sono soltanto floating point.
- Campionamento, caricamento e warmup sono esclusi dai tempi operativi; i trasferimenti nei fit/predict sono inclusi.
- Calibrazione e metriche usano lo stesso codice CPU di produzione, senza ottimizzazioni.
- Scarti assoluti > 0,03 in ROC-AUC, PR-AUC o F1 calibrato sono segnalati per approfondimento.
- Versioni, parametri, fingerprint e stato dei processi iniziali sono nel manifest.
- Concorrenza rilevata all’avvio: [].

## Validazione dell’implementazione

47 test superati, inclusi fit CPU/GPU reali e roundtrip CSV per EVONS e i due notebook FakeNewsNet. Kernel `paper_disinfo` verificato con interprete `/venv/paper_disinfo/bin/python`. Dipendenze coerenti secondo `pip check`. Dettagli in `validation.json` e `validation.log`.
