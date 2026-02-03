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


def compute_column_name_embeddings(datapath:str, model_name:str='paraphrase-mpnet-base-v2') -> dict[str, dict[str, np.ndarray]]:

    column_name_embeddings = dict()
    if model_name == 'dunzhang/stella_en_400M_v5':
        model = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        model = SentenceTransformer(model_name)

    for filename in tqdm(os.listdir(datapath)):
        column_name_embeddings[filename] = dict()
        file_path = os.path.join(datapath, filename)
        df = pd.read_csv(file_path, na_values=NAN_VALUES)

        for col in df.columns:
            column_name_emb = model.encode(col)
            column_name_embeddings[filename][col] = column_name_emb

    return column_name_embeddings


def compute_column_value_embeddings(datapath:str, model_name:str='paraphrase-mpnet-base-v2', hist_bins:int=100) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:

    column_value_embeddings = dict()
    column_num_value_embeddings = dict()
    if model_name == 'dunzhang/stella_en_400M_v5':
        model = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        model = SentenceTransformer(model_name)

    for filename in tqdm(os.listdir(datapath)):
        column_num_value_embeddings[filename] = dict()
        file_path = os.path.join(datapath, filename)
        df = pd.read_csv(file_path, na_values=NAN_VALUES)

        df_non_numeric = df.select_dtypes(exclude=['number'])
        if not df_non_numeric.empty:
            column_value_embeddings[filename] = dict()

        df_numeric = df.select_dtypes(include=['number'])

        if not df_numeric.empty:
            column_num_value_embeddings[filename] = dict()

        for col in df_non_numeric.columns:
            values = list(set(df_non_numeric[col].dropna().astype(str)))
            values_embs = model.encode(values)
            column_value_embeddings[filename][col] = np.mean(values_embs, axis=0)

        for col in df_numeric.columns:
            values = df_numeric[col].dropna()
            try:
                st_values = standardize_column(values)
            except Exception as e:
                print(filename, col)
                print(e)
                raise Exception("Error")
            feats_hist = histogram_features(st_values, bins=hist_bins)
            column_num_value_embeddings[filename][col] =  feats_hist
                    

    return column_value_embeddings, column_num_value_embeddings
