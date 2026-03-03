# %% [markdown]
# # Evons — Virality prediction (merged standalone) -- file python copied from `evons_virality_prediction.ipynb`
# 
# Ce notebook contient un pipeline autonome pour la tâche de viralité sur Evons.
# Il ne lance aucun autre notebook.
# 
# Modèles exécutés:
# - MLP (text-only) sur embeddings BERT
# - MLP + source embedding (BERT)
# - MLP + average engagement feature (BERT)
# - Gating model (BERT)
# - Gating model (Mistral)
# 

# %%

from pathlib import Path
import os
import warnings

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('WANDB_MODE', 'disabled')

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


# %%

def load_virality_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['media_source'] = df['media_source'].astype('category').cat.codes

    thr = df['fb_engagements'].quantile(0.95)
    df['is_viral'] = (df['fb_engagements'] > thr).astype(int)
    y = df['is_viral'].to_numpy()

    source = df['media_source'].to_numpy()
    fb_eng = df['fb_engagements'].to_numpy(dtype=np.float32)
    return df, y, source, fb_eng


def compute_avg_eng_per_source(source_tr, fb_eng_tr, source_va):
    """Compute per-source mean engagement on train split, then map to train and val indices.
    Sources unseen in train fall back to the global train mean."""
    src_to_avg = {}
    for s in np.unique(source_tr):
        mask = source_tr == s
        src_to_avg[s] = fb_eng_tr[mask].mean()
    global_avg = fb_eng_tr.mean()
    avg_tr = np.array([src_to_avg[s] for s in source_tr], dtype=np.float32)
    avg_va = np.array([src_to_avg.get(s, global_avg) for s in source_va], dtype=np.float32)
    mu, sigma = avg_tr.mean(), avg_tr.std() + 1e-6
    return (avg_tr - mu) / sigma, (avg_va - mu) / sigma


def load_embeddings(data_dir: Path, mode='bert'):
    if mode == 'bert':
        t = torch.load(data_dir / 'title_embeddings.pt', map_location='cpu')
        d = torch.load(data_dir / 'desc_embeddings.pt', map_location='cpu')
    else:
        t = torch.load(data_dir / 'title_embeddings_mistral.pt', map_location='cpu')
        d = torch.load(data_dir / 'desc_embeddings_mistral.pt', map_location='cpu')
    return t.float(), d.float()


def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    out['auc'] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return out


# %%

class MultiInputDataset(Dataset):
    def __init__(self, t, d, source, avg_eng, y):
        self.t, self.d = t, d
        self.source = torch.tensor(source, dtype=torch.long)
        self.avg_eng = torch.tensor(avg_eng, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.t[idx], self.d[idx], self.source[idx], self.avg_eng[idx], self.y[idx]


class TextMLP(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden//2, 1))
    def forward(self, x):
        return self.net(x).squeeze(1)


class SourceMLP(nn.Module):
    def __init__(self, in_dim, n_sources, src_dim=32, hidden=256):
        super().__init__()
        self.src_emb = nn.Embedding(n_sources, src_dim)
        self.net = nn.Sequential(nn.Linear(in_dim + src_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden//2, 1))
    def forward(self, x, src):
        z = torch.cat([x, self.src_emb(src)], dim=1)
        return self.net(z).squeeze(1)


class AvgEngMLP(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim + 1, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden//2, 1))
    def forward(self, x, avg_eng):
        z = torch.cat([x, avg_eng.unsqueeze(1)], dim=1)
        return self.net(z).squeeze(1)


class GatingMLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(emb_dim*2 + 1, 128), nn.ReLU(), nn.Linear(128, emb_dim), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(emb_dim + 1, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, t, d, avg_eng):
        g = self.gate(torch.cat([t, d, avg_eng.unsqueeze(1)], dim=1))
        fused = g * t + (1-g) * d
        return self.head(torch.cat([fused, avg_eng.unsqueeze(1)], dim=1)).squeeze(1)


# %%

def run_cv(title_emb, desc_emb, y, source, fb_eng, model_name, n_splits=10, epochs=10, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_concat = torch.cat([title_emb, desc_emb], dim=1)
    n_sources = int(source.max()) + 1
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    rows = []
    for fold, (tr, va) in enumerate(skf.split(X_concat.numpy(), y), 1):
        # Compute per-source avg engagement from training data only (no leakage)
        avg_eng_tr, avg_eng_va = compute_avg_eng_per_source(source[tr], fb_eng[tr], source[va])

        ds_tr = MultiInputDataset(title_emb[tr], desc_emb[tr], source[tr], avg_eng_tr, y[tr])
        ds_va = MultiInputDataset(title_emb[va], desc_emb[va], source[va], avg_eng_va, y[va])
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

        if model_name == 'mlp_text':
            model = TextMLP(X_concat.shape[1]).to(device)
        elif model_name == 'mlp_source':
            model = SourceMLP(X_concat.shape[1], n_sources=n_sources).to(device)
        elif model_name == 'mlp_avg_eng':
            model = AvgEngMLP(X_concat.shape[1]).to(device)
        elif model_name == 'gating':
            model = GatingMLP(title_emb.shape[1]).to(device)
        else:
            raise ValueError(model_name)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in tqdm(range(epochs), desc=f'Fold {fold}'):
            model.train()
            for t, d, src, avg, yt in dl_tr:
                t, d, src, avg, yt = t.to(device), d.to(device), src.to(device), avg.to(device), yt.to(device)
                opt.zero_grad()
                if model_name == 'mlp_text':
                    logits = model(torch.cat([t, d], dim=1))
                elif model_name == 'mlp_source':
                    logits = model(torch.cat([t, d], dim=1), src)
                elif model_name == 'mlp_avg_eng':
                    logits = model(torch.cat([t, d], dim=1), avg)
                else:
                    logits = model(t, d, avg)
                loss = loss_fn(logits, yt)
                loss.backward()
                opt.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for t, d, src, avg, _ in dl_va:
                t, d, src, avg = t.to(device), d.to(device), src.to(device), avg.to(device)
                if model_name == 'mlp_text':
                    p = torch.sigmoid(model(torch.cat([t, d], dim=1)))
                elif model_name == 'mlp_source':
                    p = torch.sigmoid(model(torch.cat([t, d], dim=1), src))
                elif model_name == 'mlp_avg_eng':
                    p = torch.sigmoid(model(torch.cat([t, d], dim=1), avg))
                else:
                    p = torch.sigmoid(model(t, d, avg))
                probs.append(p.cpu().numpy())
        probs = np.concatenate(probs)
        m = compute_metrics(y[va], probs)
        m['model'] = model_name
        m['fold'] = fold
        rows.append(m)

    return pd.DataFrame(rows)


# %%

df, y, source, fb_eng = load_virality_data(CSV_PATH)
print('Rows:', len(df), '| Positive rate:', y.mean().round(4))

bert_t, bert_d = load_embeddings(DATA_DIR, mode='bert')
mis_t, mis_d = load_embeddings(DATA_DIR, mode='mistral')

if not (len(bert_t) == len(mis_t) == len(y)):
    raise ValueError('Embeddings and CSV size mismatch')

res = []
res.append(run_cv(bert_t, bert_d, y, source, fb_eng, model_name='mlp_text'))
res.append(run_cv(bert_t, bert_d, y, source, fb_eng, model_name='mlp_source'))
res.append(run_cv(bert_t, bert_d, y, source, fb_eng, model_name='mlp_avg_eng'))
res.append(run_cv(bert_t, bert_d, y, source, fb_eng, model_name='gating').assign(model='gating_bert'))
res.append(run_cv(mis_t, mis_d, y, source, fb_eng, model_name='gating').assign(model='gating_mistral'))

all_results = pd.concat(res, ignore_index=True)
summary = all_results.groupby('model')[['accuracy', 'f1', 'precision', 'recall', 'auc']].mean().sort_values('f1', ascending=False)
summary


# %%

OUT_DIR = ROOT / 'colab_no_embeddings' / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
all_results.to_csv(OUT_DIR / 'evons_virality_merged_cv_results.csv', index=False)
summary.to_csv(OUT_DIR / 'evons_virality_merged_summary.csv')
print('Saved results to', OUT_DIR)


# %%
PUB_SUMMARY_FILENAME = 'evons_virality_publication_summary.md'

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



