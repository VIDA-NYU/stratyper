import json
import os
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from json_repair import repair_json
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
import inspect
from datetime import datetime, UTC
from collections import Counter
from random import choice, sample
import sys
import contextlib
from openai import OpenAI
from portkey_ai import Portkey
from dotenv import load_dotenv
import unicodedata

def normalize_string(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = s.replace('_', ' ')
    s = ' '.join(s.split())
    return s

def df_to_text(df:pd.DataFrame, column_names:bool=True, num_rows:int=5)->str:

    if column_names:
        columns = '| ' + ' | '.join(df.columns) + ' |'
    else:
        columns = '| ' + ' | '.join([f'Col{i}' for i in range(len(df.columns))]) + ' |'
    dashes = '|' + (' --- |' * len(df.columns))

    rows = [columns, dashes]

    df_sorted = df.loc[df.isnull().sum(axis=1).argsort()]
    for _, row in df_sorted.head(num_rows).iterrows():
        formatted_row = '| ' + ' | '.join(map(str, row)) + ' |'
        rows.append(formatted_row)

    markdown_table = '\n'.join(rows)
    return markdown_table + '\n'

def extract_dict_from_response(response_text: str) -> dict | None:
    """
    Extract a dictionary from any JSON-formatted LLM response. Handles:
      - Plain JSON, Python-dict syntax (single quotes), unquoted keys, trailing commas
      - Markdown code fences (```json, ```python, or unspecified)
      - JSON embedded inside surrounding prose
      - Truncated / malformed JSON (via json_repair)
      - Top-level lists: merged if non-conflicting single-key dicts, or
        pivoted from [{"column": X, "types": Y}, ...] shape
    """
    if not response_text or not isinstance(response_text, str):
        return None

    # Candidate payloads in order of decreasing confidence
    candidates: list[str] = []

    # 1. Markdown code fences (most reliable signal)
    code_block_pattern = r'```(?:json|python)?\s*(.*?)\s*```'
    candidates.extend(re.findall(code_block_pattern, response_text, re.DOTALL))

    # 2. Top-level balanced {...} or [...] spans in the raw text
    candidates.extend(_find_balanced_spans(response_text))

    # 3. Whole response as a last resort
    candidates.append(response_text)

    for cand in candidates:
        cleaned = cand.strip().replace('{%', '{').replace('%}', '}')
        if not cleaned:
            continue
        try:
            result = repair_json(cleaned, return_objects=True)
        except Exception:
            continue
        normalized = _normalize_parsed_result(result)
        if normalized:
            return normalized

    return None

def _find_balanced_spans(text: str) -> list[str]:
    """Return every top-level balanced {...} or [...] span in text, quote-aware."""
    spans = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] in '{[':
            open_char = text[i]
            close_char = '}' if open_char == '{' else ']'
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == '\\':
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == open_char:
                        depth += 1
                    elif c == close_char:
                        depth -= 1
                        if depth == 0:
                            spans.append(text[i:j + 1])
                            break
                j += 1
            i = j + 1
        else:
            i += 1
    return spans

def _normalize_parsed_result(result) -> dict | None:
    """Coerce parsed JSON into a non-empty dict if possible."""
    if isinstance(result, dict):
        return result if result else None

    if not (isinstance(result, list) and result):
        return None

    # Pivot shape: [{"column": X, "types": Y}, ...]
    col_key_names = {'column', 'column_name', 'col', 'name'}
    type_key_names = {'type', 'types', 'semantic_type', 'semantic_types',
                      'annotation', 'annotations'}
    pivoted: dict = {}
    pivotable = True
    for item in result:
        if not isinstance(item, dict):
            pivotable = False
            break
        lowered = {k.lower(): k for k in item.keys()}
        ck = next((lowered[k] for k in col_key_names if k in lowered), None)
        tk = next((lowered[k] for k in type_key_names if k in lowered), None)
        if ck is None or tk is None:
            pivotable = False
            break
        pivoted[str(item[ck])] = item[tk]
    if pivotable and pivoted:
        return pivoted

    # Merge shape: [{"colA": [...]}, {"colB": [...]}] with no key conflicts
    if all(isinstance(x, dict) for x in result):
        merged: dict = {}
        for item in result:
            if any(k in merged for k in item):
                return None  # conflict → can't safely merge
            merged.update(item)
        return merged or None

    return None

def remove_nan_columns(df:pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that contain only NaN values from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: A new DataFrame without columns that contain only NaN values.
    """
    return df.dropna(axis=1, how='all')

def identify_numeric_string_columns(df:pd.DataFrame) -> list[str]:
    """
    Identifies columns in a Pandas DataFrame that are strings but contain numeric data.

    Args:
      df: The Pandas DataFrame to analyze.

    Returns:
      A list of column names that are strings but contain numeric data.
    """
    numeric_string_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Or str, depending on Pandas version
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_string_cols.append(col)
            except (ValueError, TypeError):
                pass
    return numeric_string_cols

def standardize_column(series:pd.Series) -> np.ndarray:
    """Standardize a numeric column to have mean 0 and variance 1."""
    mean, std = series.mean(), series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return np.zeros(len(series))
    return ((series - mean) / std).to_numpy()

def combine_embeddings(column_name_embeddings:dict[str, dict[str, np.ndarray]], column_value_embeddings:dict[str, dict[str, np.ndarray]], split:bool=False) \
    -> dict[str, dict[str, np.ndarray]] | tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """
    Combines column name and value embeddings into a single dictionary.

    Args:
      column_name_embeddings: A dictionary mapping filenames to dictionaries of column name embeddings.
      column_value_embeddings: A dictionary mapping filenames to dictionaries of column value embeddings.

    Returns:
      A dictionary mapping filenames to dictionaries of combined column embeddings.
    """
    combined_embeddings = dict()
    for filename in column_name_embeddings:
        combined_embeddings[filename] = dict()
        for col in column_name_embeddings[filename]:
            name_emb = column_name_embeddings[filename][col]
            value_emb = column_value_embeddings[filename][col]
            combined_embeddings[filename][col] = np.concatenate([name_emb, value_emb])

    if split:
        combined_numeric_embeddings = dict()
        combined_non_numeric_embeddings = dict()

        lengths = [emb.shape[0] for file in combined_embeddings.values() for emb in file.values()]

        min_length = min(lengths)
        for filename, column_embeddings in combined_embeddings.items():
            combined_numeric_embeddings[filename] = dict()
            combined_non_numeric_embeddings[filename] = dict()
            for col, emb in column_embeddings.items():
                if len(emb) == min_length:
                    combined_numeric_embeddings[filename][col] = emb
                else:
                    combined_non_numeric_embeddings[filename][col] = emb
        return combined_numeric_embeddings, combined_non_numeric_embeddings

    return combined_embeddings

def get_embeddings_index(embeddings_dict:dict[str, dict[str, np.ndarray]], scale:bool=True) -> tuple[np.ndarray | list[np.ndarray], dict[int, tuple[str, str]]]:

    all_embs = []
    index = 0
    index_file_column = dict()
    for f, cd in embeddings_dict.items():
        for c, emb in cd.items():
            index_file_column[index] = [f, c]
            all_embs.append(emb)
            index += 1

    if scale:
        return StandardScaler().fit_transform(all_embs), index_file_column
    return all_embs, index_file_column

def remove_nan_columns(df:pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how='all')

def find_centroid_embedding(embeddings:np.ndarray) -> int:
    """
    Find the embedding that minimizes the total distance to all other embeddings.
    
    Args:
        embeddings (np.ndarray): A (N, D) array where N is the number of embeddings and D is the embedding dimension.
    
    Returns:
        int: Index of the centroid embedding.
    """
    distances = cdist(embeddings, embeddings, metric='euclidean')  # Compute pairwise distances
    total_distances = distances.sum(axis=1)  # Sum distances for each embedding
    centroid_index = np.argmin(total_distances)  # Find index of embedding with minimum total distance
    return centroid_index.item()


def sample_values_to_text(dfs:list[pd.DataFrame], columns:list[str], sampling:str='length', inverted_index:dict[str, set[str]]=None, num_samples:int=10) -> str:

    column_values = []
    for index, df in enumerate(dfs):
        column_values += list(df[columns[index]].dropna().astype(str))      
    unique_values = list(set(column_values))


    if len(unique_values) <= num_samples:
        sampled_values = unique_values
    else:
        

        types_to_uvs = dict()

        if inverted_index:
            for uv in unique_values:
                if uv in inverted_index:
                    for t in inverted_index[uv]:
                        if t in types_to_uvs:
                            types_to_uvs[t].append(uv)
                        else:
                            types_to_uvs[t] = [uv]
            non_typed_uvs = list(set(unique_values) - set(inverted_index.keys()))
        else:
            non_typed_uvs = unique_values

        if non_typed_uvs:
            if sampling in ['sbert', 'tf-idf']:
                if sampling == 'sbert':
                    model = SentenceTransformer('paraphrase-mpnet-base-v2')
                    embeddings = model.encode(non_typed_uvs, show_progress_bar=True)
                elif sampling == 'tf-idf':
                    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4))
                    embeddings = vectorizer.fit_transform(non_typed_uvs).toarray()
                kmeans = KMeans(n_clusters=min(len(non_typed_uvs), num_samples), random_state=0, verbose=0)
                kmeans.fit(embeddings)
                
                cluster_centers = kmeans.cluster_centers_
                closest_indices = np.argmin(cdist(cluster_centers, embeddings), axis=1)
                sampled_values = [non_typed_uvs[i] for i in closest_indices]
            elif sampling == 'length':
                lengths = [len(uv) for uv in non_typed_uvs]
                length_dict = Counter(lengths)

                length_values = {length:[] for length in set(lengths)}

                for uv in non_typed_uvs:
                    length_values[len(uv)].append(uv)

                top_length_strata= length_dict.most_common(min(len(length_dict), num_samples))

                sampled_values = []
                if len(length_dict) < num_samples:
                    budgets = dict()
                    mod = num_samples % len(length_dict)
                    div = num_samples // len(length_dict)
                    for length, _ in top_length_strata:
                        budgets[length] = div + (1 if mod > 0 else 0)
                        mod -= 1
                    for length, _ in top_length_strata:
                        samples = sample(length_values[length], min(len(length_values[length]), budgets[length]))
                        sampled_values.extend(samples)
                else:
                    for length, _ in top_length_strata:
                        sample_value = choice(length_values[length])
                        sampled_values.append(sample_value)    
            elif sampling == 'random':
                sampled_values = sample(unique_values, min(len(unique_values), num_samples))
            else:
                raise ValueError('Value Sampling method is not supported!')
            
            # augment sampled values with typed values
            for _, values in types_to_uvs.items():
                sampled_values.append(choice(values))
            
        else:
            samples_per_type = num_samples // len(types_to_uvs)
            sampled_values = []
            for _, values in types_to_uvs.items():
                sample_set = sample(values, min(len(values), samples_per_type))

                for v in sample_set:
                    values.remove(v)
                sampled_values.extend(sample_set)

            remaining_samples = num_samples - len(sampled_values)

            if remaining_samples > 0:
                keys_to_sample = []
                for value_type, values in types_to_uvs.items():
                    if values:
                        keys_to_sample.append(value_type)
                random_types = sample(keys_to_sample, min(len(keys_to_sample), remaining_samples))
                for t in random_types:
                    sampled_values.append(choice(types_to_uvs[t]))

            sampled_values = list(set(sampled_values))
    
    return ' | '.join(sampled_values)

def compute_metrics(samples: dict[str, dict[str, list[tuple[str, str]]]], annotations: dict[str, dict[str, list[str]]], 
                    ground_truth: dict[str, dict[str, dict[str, dict[str, list[str]]]]], 
                    stages: list[str], numeric: bool = None, print_results: bool = False) -> tuple[float, float, float]:

    def compute_tp_fp_fn(samples: dict[str, list[tuple[str, str]]], annotations: dict[str, dict[str, list[str]]], 
                         ground_truth: dict[str, dict[str, dict[str, list[str]]]], 
                         numeric: bool) -> tuple[int, int, int]:
        tp = 0
        fp = 0
        fn = 0

        num = '' if numeric else 'non_'
        for fname, cname in samples[f'{num}numeric']:
            ans = annotations[fname][cname]
            gt = ground_truth[f'{num}numeric'][fname][cname]
            fp += len(set(ans).difference(set(gt)))
            tp += len(set(ans).intersection(set(gt)))
            if numeric:
                if len(set(ans).intersection(set(gt))) == 0:
                    fn += 1
            else:
                fn += len(set(gt).difference(set(ans)))
        return tp, fp, fn


    if len(stages) == 1:
        stage = stages[0]
        if numeric is not None:
            tp, fp, fn = compute_tp_fp_fn(samples[stage], annotations, ground_truth[stage], numeric)
        else:
            tp_num, fp_num, fn_num = compute_tp_fp_fn(samples[stage], annotations, ground_truth[stage], True)
            tp_non, fp_non, fn_non = compute_tp_fp_fn(samples[stage], annotations, ground_truth[stage], False)

            tp = tp_num + tp_non
            fp = fp_num + fp_non
            fn = fn_num + fn_non
    else:
        stage1 = stages[0]
        stage2 = stages[1]
        if numeric is not None:
            tp1, fp1, fn1 = compute_tp_fp_fn(samples[stage1], annotations, ground_truth[stage1], numeric)
            tp2, fp2, fn2 = compute_tp_fp_fn(samples[stage2], annotations, ground_truth[stage2], numeric)     
                
            tp = tp1 + tp2
            fp = fp1 + fp2
            fn = fn1 + fn2
        else:
            tp1_num, fp1_num, fn1_num = compute_tp_fp_fn(samples[stage1], annotations, ground_truth[stage1], True)
            tp1_non, fp1_non, fn1_non = compute_tp_fp_fn(samples[stage1], annotations, ground_truth[stage1], False)
            tp2_num, fp2_num, fn2_num = compute_tp_fp_fn(samples[stage2], annotations, ground_truth[stage2], True)
            tp2_non, fp2_non, fn2_non = compute_tp_fp_fn(samples[stage2], annotations, ground_truth[stage2], False)

            tp = tp1_num + tp1_non + tp2_num + tp2_non
            fp = fp1_num + fp1_non + fp2_num + fp2_non
            fn = fn1_num + fn1_non + fn2_num + fn2_non
          

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)

    if print_results:
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'F1-score: {f1_score:.3f}')

    return precision, recall, f1_score

def results_summary(samples: dict[str, dict[str, list[tuple[str, str]]]], annotations: dict[str, dict[str, list[str]]], 
                    ground_truth: dict[str, dict[str, dict[str, dict[str, list[str]]]]]) -> pd.DataFrame:
    

    df_results = pd.DataFrame({
    'Score Aggregation': ['Stage 1 - Numeric', 'Stage 1 - Non-Numeric', 'Stage 1 - All', 'Stage 2 - Numeric', 'Stage 2 - Non-Numeric', 'Stage 2 - All', 'Overall - Numeric', 'Overall - Non-Numeric', 'Overall - All'],         
    'Precision': [None] * 9,
    'Recall': [None] * 9,
    'F1-score': [None] * 9
    })

    rows = dict()

    rows[0] = compute_metrics(samples, annotations, ground_truth, stages=['stage_1'], numeric=True)

    rows[1] = compute_metrics(samples, annotations, ground_truth, stages=['stage_1'], numeric=False)

    rows[2] = compute_metrics(samples, annotations, ground_truth, stages=['stage_1'])

    rows[3] = compute_metrics(samples, annotations, ground_truth, stages=['stage_2'], numeric=True)

    rows[4] = compute_metrics(samples, annotations, ground_truth, stages=['stage_2'], numeric=False)

    rows[5] = compute_metrics(samples, annotations, ground_truth, stages=['stage_2'])

    rows[6] = compute_metrics(samples, annotations, ground_truth, stages=['stage_1', 'stage_2'], numeric=True)

    rows[7] = compute_metrics(samples, annotations, ground_truth, stages=['stage_1', 'stage_2'], numeric=False)

    rows[8] = compute_metrics(samples, annotations, ground_truth, stages=['stage_1', 'stage_2'])

    for index, _ in df_results.iterrows():
        df_results.at[index, 'Precision'] = f'{rows[index][0]:.3f}'
        df_results.at[index, 'Recall'] = f'{rows[index][1]:.3f}'
        df_results.at[index, 'F1-score'] = f'{rows[index][2]:.3f}'

    return df_results




def log_llm_call(logpath:str, prompt:str, response:str, model:str, **kwargs) -> None:
    """
    Logs the LLM call to a file.

    Args:
        prompt (str): The prompt sent to the LLM.
        response (str): The response received from the LLM.
        model (str): The model used for the LLM call.
    """
    log_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": inspect.stack()[1].function,
        "model": model,
        "prompt": prompt,
        "response": response
    }
    if kwargs:
        log_entry.update({"filepath": kwargs["filepath"], "column": kwargs["column"]})

    with open(logpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def get_client(provider: str) -> OpenAI | Portkey:
    """
    Factory to create LLM clients. 
    Supports: 'ollama', 'openrouter', 'portkey', 'openai'
    """

    load_dotenv()
    provider = provider.lower()

    configs = {
        "ollama": {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "api_key": "ollama", # Required by client, ignored by Ollama
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }

    if provider == "portkey":
        return Portkey(
            api_key=os.getenv("PORTKEY_API_KEY"),
            virtual_key=os.getenv("PORTKEY_VIRTUAL_API_KEY")
        )
    elif provider in configs:
        conf = configs[provider]
        if not conf.get("api_key") and provider != "ollama":
            raise ValueError(f"Missing API Key for {provider}. Check your .env file or export it properly.")
        
        return OpenAI(**conf)


    raise ValueError(f"Unsupported provider: {provider}")

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
