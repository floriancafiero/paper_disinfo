# %% [markdown]
# # Evons — Disinformation detection (merged standalone) -- file python copied from `evons_disinformation_detection.ipynb`
# 
# Ce notebook contient un pipeline autonome pour la tâche de détection de désinformation sur Evons.
# Il ne lance aucun autre notebook.
# 
# Modèles exécutés:
# - MLP sur embeddings BERT pré-calculés (`title_embeddings.pt`, `desc_embeddings.pt`)
# - MLP sur embeddings Mistral pré-calculés (`title_embeddings_mistral.pt`, `desc_embeddings_mistral.pt`)
# - Baselines classiques (LogReg, RF, XGBoost si disponible) sur concaténation des embeddings
# 

# %%

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('WANDB_MODE', 'disabled')

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path().cwd()
DATA_DIR = ROOT / 'evons' / 'data'
CSV_PATH = DATA_DIR / 'evons.csv'

required = [
    CSV_PATH,
    DATA_DIR / 'title_embeddings.pt',
    DATA_DIR / 'desc_embeddings.pt',
    DATA_DIR / 'title_embeddings_mistral.pt',
    DATA_DIR / 'desc_embeddings_mistral.pt',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError('Fichiers manquants\n' + '\n'.join(missing))

print('All required files found.')
print('XGBoost available:', HAS_XGB)


# %%

def load_disinfo_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df = df[['title', 'description', 'is_fake']]
    y = df['is_fake'].astype(int).to_numpy()
    return df, y


def load_embeddings(data_dir: Path, mode='bert'):
    if mode == 'bert':
        t = torch.load(data_dir / 'title_embeddings.pt', map_location='cpu')
        d = torch.load(data_dir / 'desc_embeddings.pt', map_location='cpu')
    elif mode == 'mistral':
        t = torch.load(data_dir / 'title_embeddings_mistral.pt', map_location='cpu')
        d = torch.load(data_dir / 'desc_embeddings_mistral.pt', map_location='cpu')
    else:
        raise ValueError('mode must be bert or mistral')
    if len(t) != len(d):
        raise ValueError('Title/desc embeddings size mismatch.')
    return t.float(), d.float()


class PairDataset(Dataset):
    def __init__(self, title_emb, desc_emb, y):
        self.t = title_emb
        self.d = desc_emb
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.cat([self.t[idx], self.d[idx]], dim=0), self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        out['auc'] = roc_auc_score(y_true, y_prob)
    else:
        out['auc'] = np.nan
    return out


# %%

def train_eval_mlp(title_emb, desc_emb, y, epochs=12, batch_size=64, lr=1e-3, n_splits=10):
    X = torch.cat([title_emb, desc_emb], dim=1)
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_metrics = []
    for fold, (tr, va) in enumerate(skf.split(X.numpy(), y), 1):
        model = MLP(input_dim=X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        train_ds = PairDataset(title_emb[tr], desc_emb[tr], y[tr])
        val_ds = PairDataset(title_emb[va], desc_emb[va], y[va])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model.train()
        for _ in tqdm(range(epochs), desc=f'Fold {fold}/{n_splits}'):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                probs.append(torch.sigmoid(model(xb)).cpu().numpy())
        probs = np.concatenate(probs)
        m = compute_metrics(y[va], probs)
        m['fold'] = fold
        fold_metrics.append(m)

    return pd.DataFrame(fold_metrics)


def run_baselines(title_emb, desc_emb, y, n_splits=10):
    X = torch.cat([title_emb, desc_emb], dim=1).numpy()
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    models = {
        'logreg': LogisticRegression(max_iter=1000, n_jobs=-1),
        'rf': RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
        )

    rows = []
    for name, model in models.items():
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            model.fit(X[tr], y[tr])
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X[va])[:, 1]
            else:
                prob = model.decision_function(X[va])
                prob = 1 / (1 + np.exp(-prob))
            m = compute_metrics(y[va], prob)
            m['model'] = name
            m['fold'] = fold
            rows.append(m)
    return pd.DataFrame(rows)


# %%

df, y = load_disinfo_data(CSV_PATH)
print('Rows:', len(df), '| Positive rate:', y.mean().round(4))

results = []
for mode in ['bert', 'mistral']:
    print(f'\n=== Running MLP ({mode}) ===')
    t, d = load_embeddings(DATA_DIR, mode=mode)
    if len(t) != len(y):
        raise ValueError(f'Embeddings ({mode}) and CSV size mismatch: {len(t)} vs {len(y)}')
    mlp_df = train_eval_mlp(t, d, y)
    mlp_df['model'] = f'mlp_{mode}'
    results.append(mlp_df)

    base_df = run_baselines(t, d, y)
    base_df['model'] = base_df['model'] + f'_{mode}'
    results.append(base_df)

all_results = pd.concat(results, ignore_index=True)
summary = all_results.groupby('model')[['accuracy', 'f1', 'precision', 'recall', 'auc']].mean().sort_values('f1', ascending=False)
summary


# %%

OUT_DIR = ROOT / 'colab_no_embeddings' / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
all_results.to_csv(OUT_DIR / 'evons_disinformation_merged_cv_results.csv', index=False)
summary.to_csv(OUT_DIR / 'evons_disinformation_merged_summary.csv')
print('Saved results to', OUT_DIR)


# %%
PUB_SUMMARY_FILENAME = 'evons_disinformation_publication_summary.md'

# %%
# Build a publication-ready markdown synthesis from summary dataframe
summary_for_pub = summary.copy()
summary_for_pub = summary_for_pub.reset_index()
summary_for_pub[['accuracy', 'f1', 'precision', 'recall', 'auc']] = summary_for_pub[['accuracy', 'f1', 'precision', 'recall', 'auc']].round(4)

top_model = summary_for_pub.iloc[0]
md_lines = [
    '# Synthèse des résultats',
    '',
    f"- Nombre de modèles comparés: **{len(summary_for_pub)}**",
    f"- Meilleur modèle (selon F1): **{top_model['model']}**",
    f"- F1 du meilleur modèle: **{top_model['f1']:.4f}**",
    '',
    '## Tableau comparatif (moyenne CV)',
    '',
    summary_for_pub.to_markdown(index=False),
    '',
    '## Top 3 (F1)',
    '',
    summary_for_pub.head(3).to_markdown(index=False),
]

pub_md_path = OUT_DIR / PUB_SUMMARY_FILENAME
pub_md_path.write_text('\n'.join(md_lines), encoding='utf-8')
print('Saved publication summary to', pub_md_path)

# %%



