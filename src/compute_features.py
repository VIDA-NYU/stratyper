import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from utils import standardize_column
from tqdm import tqdm
from collections import OrderedDict
import string
import nltk 
import math
from scipy.stats import kurtosis, skew
from constants import NAN_VALUES
import torch

def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def extract_bag_of_characters_features(data)-> OrderedDict[str, float]:
    characters_to_check = (
            ['[' + c + ']' for c in string.printable if c not in ('\n', '\\', '\v', '\r', '\t', '^')]
    )

    f = OrderedDict()

    data_no_null = data.dropna()
    all_value_features = OrderedDict()

    for c in characters_to_check:
        all_value_features['n_{}'.format(c)] = data_no_null.str.count(c)

    for value_feature_name, value_features in all_value_features.items():
        f['{}-agg-any'.format(value_feature_name)] = 1 if any(value_features) else 0
        f['{}-agg-all'.format(value_feature_name)] = 1 if all(value_features) else 0
        f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
        f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
        f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
        f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
        f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
        f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
        kurt = kurtosis(value_features)
        skewness = skew(value_features)
        f['{}-agg-kurtosis'.format(value_feature_name)] = kurt if not np.isnan(kurt) else 0
        f['{}-agg-skewness'.format(value_feature_name)] = skewness if not np.isnan(skewness) else 0

    return f


def extract_bag_of_words_features(data, n_val)->OrderedDict[str, float]:
    f = OrderedDict()
    data = data.dropna()

    # Entropy of column
    freq_dist = nltk.FreqDist(data)
    probs = [freq_dist.freq(l) for l in freq_dist]
    f['col_entropy'] = -sum(p * math.log(p, 2) for p in probs)

    # Fraction of cells with unique content
    num_unique = data.nunique()
    f['frac_unique'] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    num_cells = np.sum(data.str.contains('[0-9]', regex=True))
    text_cells = np.sum(data.str.contains('[a-z]|[A-Z]', regex=True))
    f['frac_numcells'] = num_cells / n_val
    f['frac_textcells'] = text_cells / n_val

    # Average + std number of numeric tokens in cells
    num_reg = '[0-9]'
    f['avg_num_cells'] = np.mean(data.str.count(num_reg))
    f['std_num_cells'] = np.std(data.str.count(num_reg))

    # Average + std number of textual tokens in cells
    text_reg = '[a-z]|[A-Z]'
    f['avg_text_cells'] = np.mean(data.str.count(text_reg))
    f['std_text_cells'] = np.std(data.str.count(text_reg))

    # Average + std number of special characters in each cell
    spec_reg = '[[!@#$%^&*(),.?":{}|<>]]'
    f['avg_spec_cells'] = np.mean(data.str.count(spec_reg))
    f['std_spec_cells'] = np.std(data.str.count(spec_reg))

    # Average number of words in each cell
    space_reg = '[" "]'
    f['avg_word_cells'] = np.mean(data.str.count(space_reg) + 1)
    f['std_word_cells'] = np.std(data.str.count(space_reg) + 1)

    all_value_features = OrderedDict()

    data_no_null = data.dropna()

    f['n_values'] = n_val

    all_value_features['length'] = data_no_null.apply(len)

    for value_feature_name, value_features in all_value_features.items():
        f['{}-agg-any'.format(value_feature_name)] = 1 if any(value_features) else 0
        f['{}-agg-all'.format(value_feature_name)] =1 if all(value_features) else 0
        f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
        f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
        f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
        f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
        f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
        f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
        kurt = kurtosis(value_features)
        skewness = skew(value_features)
        f['{}-agg-kurtosis'.format(value_feature_name)] = kurt if not np.isnan(kurt) else 0
        f['{}-agg-skewness'.format(value_feature_name)] = skewness if not np.isnan(skewness) else 0

    n_none = data.size - data_no_null.size - len([e for e in data if e == ''])
    f['none-agg-has'] = 1 if n_none > 0 else 0 
    f['none-agg-percent'] = n_none / len(data)
    f['none-agg-num'] = n_none
    f['none-agg-all'] = 1 if n_none == len(data) else 0

    return f


def histogram_features(series:pd.Series, bins:int=100) -> np.ndarray:
    hist, _ = np.histogram(series, bins=bins, density=True)
    return hist

def magnitude_features(values: np.ndarray) -> np.ndarray:
    pcts = np.percentile(values, [5, 25, 50, 75, 95])
    signed_log_pcts = np.sign(pcts) * np.log1p(np.abs(pcts))
    has_negatives = 1.0 if pcts[0] < 0 else 0.0
    int_frac = float(np.mean(values == np.round(values)))
    return np.concatenate([signed_log_pcts, [has_negatives, int_frac]])


def statistical_features(series:pd.Series) -> np.ndarray:
    return np.array([
        np.mean(series),
        np.std(series),
        np.min(series),
        np.max(series),
        np.median(series),
        np.percentile(series, 25),
        np.percentile(series, 75),
        series.skew(),
        series.kurtosis()
    ])


def get_individual_features(data: pd.DataFrame) -> pd.DataFrame:

    data = data.T
    list_values = data.values.tolist()
    data = pd.DataFrame(data={'values': list_values})

    data_columns = data['values']

    features_list = []

    for column in data_columns:
        column = pd.Series(column).astype(str)

        f = OrderedDict(list(extract_bag_of_characters_features(column).items()) + list(
            extract_bag_of_words_features(column, len(column)).items()))

        features_list.append(f)

    return pd.DataFrame(features_list).reset_index(drop=True) * 1


def compute_column_statistics(datapath:str) -> dict[str, dict[str, np.ndarray]]:    
        
    column_statistics = dict()
        
    for filename in tqdm(os.listdir(datapath)):
        column_statistics[filename] = dict()
        file_path = os.path.join(datapath, filename)
        df = pd.read_csv(file_path, na_values=NAN_VALUES)

        col_features = get_individual_features(df)
        cols = df.columns.tolist()
        feature_list = col_features.values.tolist()

        for i in range(len(cols)):
            column_statistics[filename][cols[i]] = feature_list[i]

    return column_statistics


def compute_column_name_embeddings(datapath:str, model_name:str='all-MiniLM-L6-v2') -> dict[str, dict[str, np.ndarray]]:

    column_name_embeddings = dict()
    device = _get_device()
    if model_name == 'dunzhang/stella_en_400M_v5':
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    else:
        model = SentenceTransformer(model_name, device=device)

    # Collect all (filename, col) pairs in one pass — avoids per-column encode overhead
    pairs = []
    for filename in tqdm(os.listdir(datapath), desc="Reading column names"):
        file_path = os.path.join(datapath, filename)
        df = pd.read_csv(file_path, na_values=NAN_VALUES)
        column_name_embeddings[filename] = dict()
        for col in df.columns:
            pairs.append((filename, col))

    # Single batched encode for all column names
    all_embs = model.encode([col for _, col in pairs], batch_size=512, show_progress_bar=True)

    for (filename, col), emb in zip(pairs, all_embs):
        column_name_embeddings[filename][col] = emb

    return column_name_embeddings


def compute_column_value_embeddings(datapath:str, model_name:str='all-MiniLM-L6-v2', hist_bins:int=20) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:

    column_value_embeddings = dict()
    column_num_value_embeddings = dict()
    device = _get_device()
    if model_name == 'dunzhang/stella_en_400M_v5':
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    else:
        model = SentenceTransformer(model_name, device=device)

    # Process one file at a time — bounded memory, no global DataFrame cache
    for filename in tqdm(os.listdir(datapath), desc="Computing embeddings"):
        file_path = os.path.join(datapath, filename)
        df = pd.read_csv(file_path, na_values=NAN_VALUES)

        df_non_numeric = df.select_dtypes(exclude=['number'])
        df_numeric = df.select_dtypes(include=['number'])

        column_num_value_embeddings[filename] = dict()

        if not df_non_numeric.empty:
            column_value_embeddings[filename] = dict()

            # Collect unique values per column (capped), encode the whole file in one batch
            col_vals_map = {}
            file_unique = set()
            for col in df_non_numeric.columns:
                vals = df_non_numeric[col].dropna().astype(str).unique()#[:max_vals_per_col]
                col_vals_map[col] = vals
                file_unique.update(vals)

            unique_list = list(file_unique)
            embs = model.encode(unique_list, batch_size=256, show_progress_bar=False)
            val_to_idx = {v: i for i, v in enumerate(unique_list)}

            for col, vals in col_vals_map.items():
                idxs = [val_to_idx[v] for v in vals if v in val_to_idx]
                if idxs:
                    column_value_embeddings[filename][col] = embs[idxs].mean(axis=0)

            # Release the embedding matrix and drain the MPS allocator cache
            del embs, unique_list, val_to_idx
            if device == "mps":
                torch.mps.empty_cache()

        for col in df_numeric.columns:
            values = df_numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                # Column had no finite values — emit a zero vector so shapes stay consistent
                column_num_value_embeddings[filename][col] = np.zeros(hist_bins + 7)
                continue
            st_values = standardize_column(values)
            feats_hist = histogram_features(st_values, bins=hist_bins)
            feats_mag = magnitude_features(values.to_numpy())
            column_num_value_embeddings[filename][col] = np.concatenate([feats_hist, feats_mag])

    return column_value_embeddings, column_num_value_embeddings
