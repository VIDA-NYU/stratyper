from prompts import column_type_annotation, \
    closed_column_type_annotation, cluster_type_annotation_prompt, judge_column_type_prompt
from utils import extract_dict_from_response, log_llm_call, suppress_output, get_client
import argparse
import pandas as pd
from tqdm import tqdm
import os
import pickle as pkl
from pandas.api.types import is_numeric_dtype
from constants import NAN_VALUES
import re
from difflib import get_close_matches
import yaml
import numpy as np 

def llm_cta_directory(dataset_path:str, log_filepath:str=None, api:str='ollama', model:str="phi4:latest", type_reuse:bool=False,
                  coverage:str = "single", multiple_types:bool=False, column:str = None, temperature:float=0.3, number_of_rows:int=3, prompt_type:str='stratyper') -> dict[str, dict[str, list[str]]]:
    
    client = get_client(api)
    pattern = r"^(.+?):\s*(.*)"
    datasets = os.listdir(dataset_path)
    results = {}
    total_input_tokens = 0
    total_output_tokens = 0
    semantic_types = set()
    for dataset in tqdm(datasets, desc="Performing CTA"):
        if dataset.endswith(".csv"):
            df = pd.read_csv(os.path.join(dataset_path, dataset), na_values=NAN_VALUES)
            cta_system_message, cta_user_message = column_type_annotation(df, coverage=coverage, num_rows=number_of_rows, column=column, multiple_types=multiple_types, semantic_types=list(semantic_types) if semantic_types else None, prompt_type=prompt_type)

            try:
                completion = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "developer",
                            "content": cta_system_message,
                            "role": "user",
                            "content": cta_user_message
                        }
                    ]
                )
            except:
                 print(dataset)
                 raise Exception("Error")

            while not completion.choices: # until completion is not None
                completion = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": cta_system_message,
                            "role": "user",
                            "content": cta_user_message
                        }
                    ]
                )
            response = completion.choices[0].message.content
            if log_filepath:
                log_llm_call(logpath=log_filepath, prompt=cta_user_message, response=response, model=model, filepath=dataset, column=column)

            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            extracted_dict = extract_dict_from_response(response)

            cleaned_dict = dict()

            if extracted_dict:

                for cname, types in extracted_dict.items():
                    if isinstance(types, list):
                        cleaned_types = []
                        for t in types:
                            match = re.search(pattern, t)
                            if match:
                                t_new = match.group(2).strip()
                                cleaned_types.append(t_new)
                            else:
                                cleaned_types.append(t)
                        cleaned_dict[cname] = cleaned_types
                    else:
                        match = re.search(pattern, types)
                        if match:
                            t_new = match.group(2).strip()
                            cleaned_dict[cname] = [t_new]
                        else:
                            cleaned_dict[cname] = [types]

                if type_reuse:
                    for _, types in cleaned_dict.items():
                        for t in types:
                            semantic_types.add(t)
                

                # Strip surrounding quotes the LLM sometimes wraps column names in
                cleaned_dict = {k.strip().strip("\"'"): v for k, v in cleaned_dict.items()}

                invalid_columns = set(cleaned_dict.keys()) - set(df.columns)
                valid_columns = set(df.columns) - set(cleaned_dict.keys())

                for cname in list(invalid_columns):
                    types = cleaned_dict.pop(cname)
                    close_matches = get_close_matches(cname, valid_columns, n=1, cutoff=0.6)
                    if close_matches:
                        corrected_name = close_matches[0]
                        cleaned_dict[corrected_name] = types
                        valid_columns.remove(corrected_name)
                    else:
                        print(f"Warning: column '{cname}' not found in '{dataset}', skipping.")
            else:
                cleaned_dict = dict()

            results[dataset] = cleaned_dict

    print(f"Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}")

    return results

def closed_cta_single(file_path:str, log_path:str=None, api:str='ollama_nyu', model:str="phi4:latest", column:str=None, semantic_types:list[str]=None, 
                      sampling:str='length', inverted_index:dict[str, set[str]]=None, num_rows:int=3, 
                      num_samples:int=5, temperature:float=0.3, refine_output:bool=True) -> tuple[dict[str, str], int, int]:

    client = get_client(api)

    df = pd.read_csv(file_path, na_values=NAN_VALUES)
    ccta_system_message, ccta_user_message = closed_column_type_annotation(df, column=column,  semantic_types=semantic_types, 
                                                                           sampling=sampling, num_rows=num_rows, num_samples=num_samples,
                                                                           numerical=is_numeric_dtype(df[column]), 
                                                                           inverted_index=inverted_index, multiple_types=False if is_numeric_dtype(df[column]) else True)

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": ccta_system_message,
                "role": "user",
                "content": ccta_user_message
            }
        ]
    )

    while not completion.choices: # until completion is not None
                completion = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": ccta_system_message,
                            "role": "user",
                            "content": ccta_user_message
                        }
                    ]
                )
    response = completion.choices[0].message.content

    # Log the LLM call
    if log_path:
        log_llm_call(logpath=log_path, prompt=ccta_user_message, response=response, model=model, filepath=file_path, column=column)

    extracted_dict = extract_dict_from_response(response)

    if refine_output and extracted_dict:
        response_types = list(extracted_dict.values())[0]

        if isinstance(response_types, list):
            new_types = list(set(response_types).intersection(set(semantic_types)))
            extracted_dict = {list(extracted_dict.keys())[0]: new_types}
        else:
            if response_types in semantic_types:
                extracted_dict = {list(extracted_dict.keys())[0]: [response_types]}
            else:
                extracted_dict = {list(extracted_dict.keys())[0]: []}

    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens

    return extracted_dict, input_tokens, output_tokens


def cluster_type_annotation(file_path:str, columns:list[tuple[str, str]], log_path:str=None, api:str='openrouter', model:str="google/gemini-2.5-flash", 
                            num_samples:int=5, semantic_types:list[str|tuple[str]]=None, inverted_index:dict[str, set[str]]=None,
                            sampling:str='length', context:str=None,
                            multiple_types:bool=False, whole_cluster:bool = False, temperature:float=0.3) -> tuple[tuple[str, str], int, int]:
    
    client = get_client(api)

    dfs = []
    cols = []
    for column in columns:
        df = pd.read_csv(os.path.join(file_path, column[0]), na_values=NAN_VALUES)
        dfs.append(df)
        cols.append(column[1])

    cluster_system_message, cluster_user_message = cluster_type_annotation_prompt(dfs, cols, num_samples=num_samples, semantic_types=semantic_types, 
                                                                                  sampling=sampling,
                                                                                  whole_cluster_samples=whole_cluster, multiple_types=multiple_types,
                                                                                  inverted_index=inverted_index, context=context)

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": cluster_system_message,
                "role": "user",
                "content": cluster_user_message
            }
        ]
    )

    while not completion.choices: # until completion is not None
                completion = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": cluster_system_message,
                            "role": "user",
                            "content": cluster_user_message
                        }
                    ]
                )
    response = completion.choices[0].message.content

    # Log the LLM call
    if log_path:
        log_llm_call(logpath=log_path, prompt=cluster_user_message, response=response, model=model)

    extracted_dict = extract_dict_from_response(response)

    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens

    return extracted_dict, input_tokens, output_tokens
     

def judge_column_type(file_path:str, semantic_types:list[str],log_path:str=None, api:str='openrouter', model:str="openai/gpt-4.1-mini", column:str=None, 
                      col_description:bool=False, sampling:str='length', explanation:bool=False, 
                      inverted_index:dict[str, set[str]]=None, num_samples:int=5, num_rows:int=3, temperature:float=0.3) -> tuple[tuple[str, str], int, int]:
    
    client = get_client(api)

    df = pd.read_csv(file_path, na_values=NAN_VALUES)
    judge_system_message, judge_user_message = judge_column_type_prompt(df, column=column, semantic_types=semantic_types, col_description=col_description, 
                                                                        explanation=explanation, sampling=sampling, inverted_index=inverted_index, 
                                                                        num_samples=num_samples, num_rows=num_rows)

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": judge_system_message,
                "role": "user",
                "content": judge_user_message
            }
        ]
    )

    while not completion.choices: # until completion is not None
                completion = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": judge_system_message,
                            "role": "user",
                            "content": judge_user_message
                        }
                    ]
                )
    response = completion.choices[0].message.content

    # Log the LLM call
    if log_path:
        log_llm_call(logpath=log_path, prompt=judge_user_message, response=response, model=model)

    extracted_dict = extract_dict_from_response(response)


    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    return extracted_dict, input_tokens, output_tokens


def retrieve_seed_types(file_path:str, log_path_ctd:str, clusters:list[list[tuple[str, str]]],
                        api_ctd:str='openrouter', model_ctd:str="google/gemini-3-flash",  
                        num_samples:int=10, whole_cluster:bool=True, context:str=None, ccta_refinement:bool=False,
                        logpath_ccta_refinement:str=None, api_ccta_refinement:str=None, model_ccta_refinement:str=None, num_rows:int=None,
                        sampling:str='length', enforce_types:bool=False, temperature:float=0.3) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[int, list[str]], int, int]: 
      
    inverted_index = dict() # store semantic types associated with each value
    communities_types = dict()
    annotations_stage_1 = dict()


    semantic_types = {'numeric': set(), 'non_numeric': set()}
    total_input_seed = 0
    total_output_seed = 0
    for community_index, community in tqdm(enumerate(clusters)):

        community_values = set()
        numeric_flag = False
        non_numeric_flag = False
        df_cache = {}
        for column in community:
            fname_col, cname_col = column[0], column[1]
            if fname_col not in df_cache:
                df_cache[fname_col] = pd.read_csv(os.path.join(file_path, fname_col), na_values=NAN_VALUES)
            df = df_cache[fname_col]
            col_vals = df[cname_col].dropna().unique()
            community_values.update(col_vals)
            if is_numeric_dtype(df[cname_col]):
                numeric_flag = True
            else:
                non_numeric_flag = True
        possible_types = set()
        for value in community_values:
             if value in inverted_index:
                possible_types.update(inverted_index[value])
        response = None
        while not response:
            with suppress_output():
                if possible_types:
                    types_to_use = possible_types
                else:
                    if numeric_flag:
                        if non_numeric_flag:
                            types_to_use = semantic_types['numeric'].union(semantic_types['non_numeric'])
                        else:
                            types_to_use = semantic_types['numeric']
                    else:
                        types_to_use = semantic_types['non_numeric']

                if numeric_flag:
                     multiple_types = False
                else:
                     multiple_types = True
                response, input_tokens, output_tokens = \
                cluster_type_annotation(file_path, community, log_path=log_path_ctd, 
                                        api=api_ctd, model=model_ctd, num_samples=num_samples, whole_cluster=whole_cluster, 
                                        sampling=sampling, multiple_types=multiple_types, semantic_types=types_to_use, inverted_index=inverted_index, temperature=temperature, context=context)
            total_input_seed += input_tokens
            total_output_seed += output_tokens
        
        answer = response['answer']
        if not isinstance(answer, list):
            answer = [answer]
        communities_types[community_index] = answer
        if enforce_types: # if we want to enforce the types, we need to add the possible types to the answer
            communities_types[community_index] = set(answer).union(possible_types)

        for t in answer:
            if numeric_flag:
                semantic_types['numeric'].add(t)
            if non_numeric_flag:
                semantic_types['non_numeric'].add(t)


        if ccta_refinement and len(answer) > 1:
             for column in community:
                fname, cname = column
                df = df_cache[fname] if fname in df_cache else pd.read_csv(os.path.join(file_path, fname), na_values=NAN_VALUES)
                unique_vals = pd.Series(df[cname].dropna().astype(str).unique())
                lengths = unique_vals.str.len()
                num_like = unique_vals.str.match(r'^\s*-?\d+(\.\d+)?\s*$').mean()
                sorted_ulens = np.sort(lengths.unique())
                iqr = np.percentile(lengths, 75) - np.percentile(lengths, 25)
                has_length_gap = len(sorted_ulens) > 1 and np.any(np.diff(sorted_ulens) > max(iqr, 1))
                column_is_homogeneous = not has_length_gap and (num_like < 0.1 or num_like > 0.9)
                if not column_is_homogeneous:
                    continue
                response_ccta, _, _ = closed_cta_single(os.path.join(file_path, fname), log_path=logpath_ccta_refinement, api=api_ccta_refinement, model=model_ccta_refinement,
                                                        num_samples=num_samples, num_rows=num_rows, column=cname, semantic_types=communities_types[community_index],
                                                        sampling=sampling, inverted_index=inverted_index, temperature=temperature)
                
                if not response_ccta:
                     response_types = []
                else:
                    response_types = list(response_ccta.values())
                    if isinstance(response_types[0], list):
                        if response_types[0] == ['None']:
                            response_types = []
                        else:
                            response_types = response_types[0]
                    elif response_types[0] == 'None':
                        response_types = []
                    elif not response_types[0]:
                        response_types = []

                if response_types: # if the model was able to assign a type
                    df = df_cache[fname] if fname in df_cache else pd.read_csv(os.path.join(file_path, fname), na_values=NAN_VALUES)
                    for value in df[cname].dropna().unique():
                        if value in inverted_index:
                            inverted_index[value].update(response_types)
                        else:
                            inverted_index[value] = set(response_types)
                    if fname in annotations_stage_1:
                        annotations_stage_1[fname][cname] = response_types
                    else:
                        annotations_stage_1[fname] = {cname: response_types}
        else:
            for column in community:
                fname, cname = column
                if fname in annotations_stage_1:
                    annotations_stage_1[fname][cname] = answer
                else:
                    annotations_stage_1[fname] = {cname: answer}
            for value in community_values:
                if value in inverted_index:
                    inverted_index[value].update(answer)
                else:
                    inverted_index[value] = set(answer)

    return semantic_types, inverted_index, communities_types, annotations_stage_1, total_input_seed, total_output_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/llm_baseline.yaml')
    parser.add_argument('-sp', '--store_path', type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    results = llm_cta_directory(**config)

    if config.get('multiple_types', False):
         mp_param = 'multiple_types_'
    else:
         mp_param = ''

    if config.get('type_reuse', False):
         ccta_param = '_type-reuse'
    else:
         ccta_param = ''

    #Store results
    with open(os.path.join(args.store_path,f'cta_{config["prompt_type"]}_{config["api"]}_{config["model"].replace("/", "_")}_{config["coverage"]}_{mp_param}rows_{config["number_of_rows"]}_temperature_{config["temperature"]}{ccta_param}.pickle'), 'wb') as file:
        pkl.dump(results, file)
