import argparse
import os
import pandas as pd
import yaml
from constants import NAN_VALUES
from llm_inference import retrieve_seed_types, cluster_type_annotation, closed_cta_single, judge_column_type
from compute_features import compute_column_name_embeddings, compute_column_value_embeddings
from utils import get_embeddings_index, suppress_output
from sentence_transformers import  SentenceTransformer, util
from sklearn.preprocessing import normalize
import pickle
import json
from tqdm import tqdm
from pandas.api.types import is_numeric_dtype

class Stratyper:

    def __init__(self, dataset_path: str, respath: str, metadata_path: str, other_annotations_paths: dict[str, str], thresholds: dict[str, list[float]], 
                 number_of_samples_ctd: int, number_of_samples_ccta: int,number_of_rows_ccta: int, number_of_rows_judge:int,
                 min_cluster_size: int, sampling_type: str, providers: dict[str, str], models: dict[str, str], 
                 temperatures: dict[str, float], context:str):
        self.dataset_path= dataset_path
        self.respath = respath
        self.metadata_path = metadata_path
        self.other_annotations_paths = other_annotations_paths
        self.thresholds = thresholds
        self.number_of_rows_ccta = number_of_rows_ccta
        self.number_of_rows_judge = number_of_rows_judge
        self.context = context
        self.number_of_samples_ctd = number_of_samples_ctd
        self.number_of_samples_ccta = number_of_samples_ccta
        self.min_cluster_size = min_cluster_size
        self.providers = providers
        self.models = models
        self.temperatures = temperatures
        self.sampling_type = sampling_type
        self.dataframes = dict()
        self._is_num = dict()
        self.annotations = {'stage_1': dict(), 'stage_2': dict()}
        self.communities_names_columns = dict()
        self.communities_values_columns = dict()
        self.clusters_combined = dict()
        self.input_tokens_ccta = 0
        self.output_tokens_ccta = 0

        if not os.path.exists(self.respath):
            os.makedirs(self.respath)

        for fname in os.listdir(self.dataset_path):
            if fname.endswith('.csv'):
                df = pd.read_csv(os.path.join(self.dataset_path, fname), na_values=NAN_VALUES)
                self.dataframes[fname] = df

                if fname not in self._is_num:
                    self._is_num[fname] = dict()

                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        self._is_num[fname][col] = True
                    else:
                        self._is_num[fname][col] = False

    def compute_clusters(self, embedding_path: str, store_files: bool = True, print_progress: bool = True, recompute: bool = False):

        def communities_columns(communities:list[list[int]], index_file_column: dict[int, tuple[str, str]]) -> list[list[tuple[str, str]]]:
            communities_columns = [list() for i in range(len(communities))]
            
            for index, community in enumerate(communities):
            
                for c_index in community:
                    file, col = index_file_column[c_index]
                    to_append = [file, col]
                    communities_columns[index].append(to_append)   
            return communities_columns

        if os.path.exists(os.path.join(embedding_path, f'column_name_embeddings_{self.models["embedding_model"]}.pickle')) and not recompute:
            with open(os.path.join(embedding_path, f'column_name_embeddings_{self.models["embedding_model"]}.pickle'), 'rb') as file:
                column_name_embeddings = pickle.load(file)
        else:
            column_name_embeddings = compute_column_name_embeddings(self.dataset_path, model_name=self.models['embedding_model'])
            if store_files:
                with open(os.path.join(embedding_path, f'column_name_embeddings_{self.models["embedding_model"]}.pickle'), 'wb') as file:
                    pickle.dump(column_name_embeddings, file)

        if os.path.exists(os.path.join(embedding_path, f'column_value_embeddings_{self.models["embedding_model"]}.pickle')) and not recompute:
            with open(os.path.join(embedding_path, f'column_value_embeddings_{self.models["embedding_model"]}.pickle'), 'rb') as file:
                column_value_embeddings = pickle.load(file)
            with open(os.path.join(embedding_path, f'column_num_value_embeddings_{self.models["embedding_model"]}.pickle'), 'rb') as file:
                column_num_value_embeddings = pickle.load(file)
        else:
            column_value_embeddings, column_num_value_embeddings = compute_column_value_embeddings(self.dataset_path, model_name=self.models['embedding_model'])
            if store_files:
                with open(os.path.join(embedding_path, f'column_value_embeddings_{self.models["embedding_model"]}.pickle'), 'wb') as file:
                    pickle.dump(column_value_embeddings, file)
                with open(os.path.join(embedding_path, f'column_num_value_embeddings_{self.models["embedding_model"]}.pickle'), 'wb') as file:
                    pickle.dump(column_num_value_embeddings, file)

        name_embeddings, index_file_column = get_embeddings_index(column_name_embeddings)
        name_embeddings_norm = normalize(name_embeddings, norm='l2')

        value_embeddings_num, index_column_value_num = get_embeddings_index(column_num_value_embeddings)
        value_embeddings, index_column_value = get_embeddings_index(column_value_embeddings)

        value_embeddings_norm = normalize(value_embeddings, norm='l2')
        value_embeddings_num_norm = normalize(value_embeddings_num, norm='l2')

        for name_threshold, value_threshold in zip(self.thresholds['name_clusters'], self.thresholds['value_clusters']):
            if print_progress:
                print(f'Computing clusters for thresholds: name clusters - {name_threshold}, value clusters - {value_threshold}')
            communities_names = util.community_detection(name_embeddings_norm, threshold=name_threshold, min_community_size=self.min_cluster_size)
            communities_names_columns = communities_columns(communities_names, index_file_column)

            communities_values = util.community_detection(value_embeddings_norm, threshold=value_threshold, min_community_size=self.min_cluster_size)
            communities_values_columns = communities_columns(communities_values, index_column_value)

            communities_values_num =  util.community_detection(value_embeddings_num_norm, threshold=value_threshold, min_community_size=self.min_cluster_size)
            communities_values_num_columns = communities_columns(communities_values_num, index_column_value_num)

            communities_values_columns.extend(communities_values_num_columns)

            self.communities_names_columns[name_threshold] = communities_names_columns
            self.communities_values_columns[value_threshold] = communities_values_columns

            column_to_indices = dict()

            for index, community in enumerate(communities_names_columns):
                for col in community:
                    column_to_indices[tuple(col)] = [index]

            for index, community in enumerate(communities_values_columns):
                for col in community:
                    if tuple(col) in column_to_indices:
                        column_to_indices[tuple(col)].append(index)

            keys_list = list(column_to_indices.keys())
            for col in keys_list:
                indices = column_to_indices[col]
                if len(indices) < 2:
                    del column_to_indices[col]

            clusters_combined_dict = dict()

            for col, indices in column_to_indices.items():

                key_val = str(indices[0]) + '-' +  str(indices[1])

                if key_val in clusters_combined_dict:
                    clusters_combined_dict[key_val].append(col)
                else:
                    clusters_combined_dict[key_val] = [col]

            clusters_combined = list(clusters_combined_dict.values())
            clusters_combined = sorted(clusters_combined, key=len, reverse=True)
    
            clusters_combined_filtered = []

            for cluster in clusters_combined:
                if len(cluster) >= self.min_cluster_size:
                    clusters_combined_filtered.append(cluster)
            
            clusters_combined = clusters_combined_filtered

            comnam_indices = set()
            for key in clusters_combined_dict:
                comnam_indices.add(int(key.split('-')[0]))

            # Add remaining name communities that do not appear in combined clusters - do this only for non-numeric columns
            for i in range(len(communities_names_columns)):
                if i not in comnam_indices:
                    if len(communities_names_columns[i]) > self.min_cluster_size:
                        if not self._is_num[communities_names_columns[i][0][0]][communities_names_columns[i][0][1]]:
                            community = [tuple(col) for col in communities_names_columns[i]]
                            clusters_combined.append(community)
        

            self.clusters_combined[(name_threshold, value_threshold)] = clusters_combined


    def column_type_discovery(self, store_files: bool = True, print_progress: bool = True):

        if os.path.exists(os.path.join(self.respath, 'annotations_stage_1.pickle')):
            with open(os.path.join(self.respath, 'annotations_stage_1.pickle'), 'rb') as file:
                annotations_stage_1 = pickle.load(file)
            with open(os.path.join(self.respath, 'communities_types.pickle'), 'rb') as file:
                communities_types = pickle.load(file)
            with open(os.path.join(self.respath, 'inverted_index.pickle'), 'rb') as file:
                inverted_index = pickle.load(file)
            with open(os.path.join(self.respath, 'semantic_types.pickle'), 'rb') as file:
                semantic_types = pickle.load(file)
            self.annotations['stage_1'] = annotations_stage_1
            self.semantic_types = semantic_types
            self.inverted_index = inverted_index
            self.communities_types = communities_types
            return

        if len(self.clusters_combined) == 1: # If only one combination of thresholds has been used
            semantic_types, inverted_index, communities_types, annotations_stage_1, total_input_tokens, total_output_tokens =\
                retrieve_seed_types(file_path=self.dataset_path, log_path_ctd=os.path.join(self.respath, 'logs_ctd.jsonl'), 
                                    clusters=self.clusters_combined[(self.thresholds['name_clusters'][0], self.thresholds['value_clusters'][0])],
                                    api_ctd=self.providers['ctd'], model_ctd=self.models['ctd'], sampling=self.sampling_type, 
                                    num_samples=self.number_of_samples_ctd, temperature=self.temperatures['ctd'], context=self.context)
        else:

            
            name_threshold, value_threshold = self.thresholds['name_clusters'][0], self.thresholds['value_clusters'][0]

            if print_progress:
                print(f'Running CTD for thresholds: name clusters - {name_threshold}, value clusters - {value_threshold}')

            semantic_types, inverted_index, communities_types, annotations_stage_1, total_input_tokens, total_output_tokens =\
            retrieve_seed_types(file_path=self.dataset_path, log_path_ctd=os.path.join(self.respath, 'logs_ctd.jsonl'), 
                                clusters=self.clusters_combined[(name_threshold, value_threshold)],
                                api_ctd=self.providers['ctd'], model_ctd=self.models['ctd'], sampling=self.sampling_type,
                                num_samples=self.number_of_samples_ctd, temperature=self.temperatures['ctd'], context=self.context)


            for name_threshold, value_threshold in zip(self.thresholds['name_clusters'][1:], self.thresholds['value_clusters'][1:]):

                if print_progress:
                    print(f'Running CTD and closed-set CTA for thresholds: name clusters - {name_threshold}, value clusters - {value_threshold}')


                clusters_combined_difference = []
                possible_types = dict()

                for cluster in self.clusters_combined[(name_threshold, value_threshold)]:
                    
                    cluster_new = []

                    types_cluster = set()
                    for fname, cname in cluster:

                        if not fname in annotations_stage_1 or not cname in annotations_stage_1[fname]:
                            cluster_new.append((fname, cname))
                        else:
                            types_cluster.update(annotations_stage_1[fname][cname])

                    if cluster_new:
                        clusters_combined_difference.append(cluster_new)
    
    
                    for fname, cname in cluster_new:
                                
                        possible_types[(fname, cname)] = list(types_cluster)

                for cluster in clusters_combined_difference:
                    for fname, cname in cluster:

                        if fname in annotations_stage_1 and cname in annotations_stage_1[fname]:
                            continue
                        else:
                            types = possible_types[(fname, cname)]
                    
                            if types:

                                response, input_tokens_ccta, output_tokens_ccta = closed_cta_single(os.path.join(self.dataset_path,fname), log_path=os.path.join(self.respath, 'logs_stage_1_ccta.jsonl'),api=self.providers['ccta'], 
                                                                                                    model=self.models['ccta'], column=cname, semantic_types=types, num_samples=self.number_of_samples_ccta, num_rows=self.number_of_rows_ccta, temperature=self.temperatures['ccta'])
                        
                                try:

                                    self.input_tokens_ccta += input_tokens_ccta
                                    self.output_tokens_ccta += output_tokens_ccta

                                    if not response:
                                        response_types = []
                                    else:
                                        response_types = list(response.values())
                                        if isinstance(response_types[0], list):
                                            if response_types[0] == ['None']:
                                                response_types = []
                                            else:
                                                response_types = response_types[0]
                                        elif response_types[0] == 'None':
                                            response_types = []
                                        elif not response_types[0]:
                                            response_types = []
                                    if response_types:
                                        if fname in annotations_stage_1:
                                            annotations_stage_1[fname][cname] = response_types
                                        else:
                                            annotations_stage_1[fname] = {cname:response_types}
                                except:
                                    print(fname)
                                    print(cname)
                                    print(response)
                                    print('------------')
               

                clusters_combined_difference_ctd = []
                for cluster in clusters_combined_difference:

                    cols = []
                    for fname, cname in cluster:

                        if not fname in annotations_stage_1 or not cname in annotations_stage_1[fname]:
                            cols.append((fname, cname))

                    if len(cols) > 1:
                        clusters_combined_difference_ctd.append(cols)

                print(f'Running CTD for {len(clusters_combined_difference_ctd)} clusters')

                for cluster in tqdm(clusters_combined_difference_ctd):
                    numeric_flag = False
                    non_numeric_flag = False
                    community_values = set()
                    for column in cluster:
                        df = self.dataframes[column[0]]
                        community_values.update(df[column[1]].dropna().unique())
                        if is_numeric_dtype(df[column[1]]):
                            numeric_flag = True
                        else:
                            non_numeric_flag = True
                    types = set()
                    for value in community_values:
                        if value in inverted_index:
                            types.update(inverted_index[value])
                    response = None
                    while not response:
                        with suppress_output():
                            if types:
                                types_to_use = types
                            else:
                                if numeric_flag:
                                    if non_numeric_flag:
                                        types_to_use = semantic_types['numeric'].union(semantic_types['non_numeric'])
                                    else:
                                        types_to_use = semantic_types['numeric']
                                else:
                                    types_to_use = semantic_types['non_numeric']

                            if numeric_flag and not non_numeric_flag:
                                multiple_types = False
                            else:
                                multiple_types = True
                            response, input_tokens, output_tokens = \
                                        cluster_type_annotation(self.dataset_path, cluster, log_path=os.path.join(self.respath, 'logs_ctd.jsonl'), 
                                                                api=self.providers['ctd'], model=self.models['ctd'], num_samples=self.number_of_samples_ctd, whole_cluster=True, 
                                                                sampling=self.sampling_type, multiple_types=multiple_types, semantic_types=types_to_use,
                                                                inverted_index=inverted_index, temperature=self.temperatures['ctd'], context=self.context)
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                    answer = response['answer']
                    if not isinstance(answer, list):
                        answer = [answer]

                    for t in answer:
                        if numeric_flag:
                            semantic_types['numeric'].add(t)
                        if non_numeric_flag:
                            semantic_types['non_numeric'].add(t)

                    for fname, cname in cluster:
                        if fname in annotations_stage_1:
                            annotations_stage_1[fname][cname] = answer
                        else:
                            annotations_stage_1[fname] = {cname: answer}

                
                    for value in community_values:
                        if value in inverted_index:
                            inverted_index[value].update(answer)
                        else:
                            inverted_index[value] = set(answer)
    

        
        if store_files:
            with open(os.path.join(self.respath, 'annotations_stage_1.pickle'), 'wb') as file:
                pickle.dump(annotations_stage_1, file)
            with open(os.path.join(self.respath, 'communities_types.pickle'), 'wb') as file:
                pickle.dump(communities_types, file)
            with open(os.path.join(self.respath, 'inverted_index.pickle'), 'wb') as file:
                pickle.dump(inverted_index, file)
            with open(os.path.join(self.respath, 'semantic_types.pickle'), 'wb') as file:
                pickle.dump(semantic_types, file)
        
        self.annotations['stage_1'] = annotations_stage_1
        self.semantic_types = semantic_types
        self.inverted_index = inverted_index
        self.communities_types = communities_types
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens
   

    def ccta_step(self, col_to_possible_types: dict[str, dict[str, dict[str, set[str]]]], numerical: bool, type_source: str):

        for fname, cname_types in tqdm(col_to_possible_types.items(), total=len(col_to_possible_types)):

            for cname, category_types in cname_types.items():
                if (self._is_num[fname][cname] and numerical) or (not self._is_num[fname][cname] and not numerical):

                    if fname in self.annotations['stage_2'] and cname in self.annotations['stage_2'][fname]:
                        continue
                    else:
                        possible_types = list(category_types[type_source])

                        if possible_types:

                            response, input_tokens_ccta, output_tokens_ccta = closed_cta_single(os.path.join(self.dataset_path,fname), log_path=os.path.join(self.respath, 'logs_stage_2_ccta.jsonl'),api=self.providers['ccta'], 
                                                                                                model=self.models['ccta'], column=cname, semantic_types=possible_types, num_samples=self.number_of_samples_ccta, 
                                                                                                temperature=self.temperatures['ccta'], num_rows=self.number_of_rows_ccta)
                            try:

                                self.input_tokens_ccta += input_tokens_ccta
                                self.output_tokens_ccta += output_tokens_ccta

                                if not response:
                                    response_types = []
                                else:
                                    response_types = list(response.values())
                                    if isinstance(response_types[0], list):
                                        if response_types[0] == ['None']:
                                            response_types = []
                                        else:
                                            response_types = response_types[0]
                                    elif response_types[0] == 'None':
                                        response_types = []
                                if response_types:
                                    if fname in self.annotations['stage_2']:
                                        if numerical:
                                            self.annotations['stage_2'][fname][cname] = response_types
                                        else:
                                            if cname in self.annotations['stage_2'][fname]:
                                                self.annotations['stage_2'][fname][cname].extend(response_types)
                                            else:
                                                self.annotations['stage_2'][fname][cname] = response_types
                                    else:
                                        self.annotations['stage_2'][fname] = {cname:response_types}
                            except:
                                print(fname)
                                print(cname)
                                print(response)
                                print('------------')

    def closed_cta(self, store_files: bool = True, print_progress: bool = True):

        if os.path.exists(os.path.join(self.respath, 'annotations_stage_2.pickle')):
            with open(os.path.join(self.respath, 'annotations_stage_2.pickle'), 'rb') as file:
                annotations_stage_2 = pickle.load(file)
            self.annotations['stage_2'] = annotations_stage_2
            return

        assigned_cols = set()
        for fname, cans in self.annotations['stage_1'].items():
            for cname in cans:
                assigned_cols.add((fname, cname))


        for name_threshold, value_threshold in zip(self.thresholds['name_clusters'], self.thresholds['value_clusters']):

            if print_progress:
                print(f'Running closed CCTA for thresholds: name clusters - {name_threshold}, value clusters - {value_threshold}')

            communities_values_columns_remaining = []
            comval_to_types = dict()
            for community in self.communities_values_columns[value_threshold]:
                rem_community = set([tuple(col) for col in community]).difference(assigned_cols)
                
                if rem_community:
                    communities_values_columns_remaining.append(list(rem_community))
                    types = set()
                    for col in community:
                        fname, cname = col
                        if fname in self.annotations['stage_1'] and cname in self.annotations['stage_1'][fname]:
                            if isinstance(self.annotations['stage_1'][fname][cname], list):
                                types.update(self.annotations['stage_1'][fname][cname])
                            else:
                                types.add(self.annotations['stage_1'][fname][cname])
                    if types:
                        comval_to_types[len(communities_values_columns_remaining) - 1] = types

            communities_names_columns_remaining = []
            comnam_to_types = dict()
            for community in self.communities_names_columns[name_threshold]:
                rem_community = set([tuple(col) for col in community]).difference(assigned_cols)
                
                if rem_community:
                    communities_names_columns_remaining.append(list(rem_community))
                    types = set()
                    for col in community:
                        fname, cname = col
                        if fname in self.annotations['stage_1'] and cname in self.annotations['stage_1'][fname]:
                            if isinstance(self.annotations['stage_1'][fname][cname], list):
                                types.update(self.annotations['stage_1'][fname][cname])
                            else:
                                types.add(self.annotations['stage_1'][fname][cname])
                    if types:
                        comnam_to_types[len(communities_names_columns_remaining) - 1] = types
            
            col_to_possible_types = dict()

            for index, types in comnam_to_types.items():

                columns = communities_names_columns_remaining[index]

                for fname, cname in columns:

                    if fname not in col_to_possible_types:
                        col_to_possible_types[fname] = dict()

                    if cname not in col_to_possible_types[fname]:
                        
                        col_to_possible_types[fname][cname] = {'comnam': set(), 'comval': set(), 'inverted_index': set()}

                        df = self.dataframes[fname]

                        values = df[cname].dropna().unique()

                        for value in values:
                            if value in self.inverted_index:
                                col_to_possible_types[fname][cname]['inverted_index'].update(self.inverted_index[value])

                    col_to_possible_types[fname][cname]['comnam'].update(types)

            for index, types in comval_to_types.items():

                columns = communities_values_columns_remaining[index]

                for fname, cname in columns:

                    if fname not in col_to_possible_types:
                        col_to_possible_types[fname] = dict()

                    if cname not in col_to_possible_types[fname]:
                        
                        col_to_possible_types[fname][cname] = {'comnam': set(), 'comval': set(), 'inverted_index': set()}

                        df = self.dataframes[fname]

                        values = df[cname].dropna().unique()

                        for value in values:
                            if value in self.inverted_index:
                                col_to_possible_types[fname][cname]['inverted_index'].update(self.inverted_index[value])

                    col_to_possible_types[fname][cname]['comval'].update(types)


            self.ccta_step(col_to_possible_types, numerical=True, type_source='comnam')
            for type_source in ['comnam', 'comval', 'inverted_index']:
                self.ccta_step(col_to_possible_types, numerical=False, type_source=type_source)
            

            for fname, cans in self.annotations['stage_2'].items():
                for cname in cans:
                    assigned_cols.add((fname, cname))
              
        

        if store_files:
            with open(os.path.join(self.respath, 'annotations_stage_2.pickle'), 'wb') as file:
                pickle.dump(self.annotations['stage_2'], file)

    def combine_annotations(self, store_files: bool = True):

        all_annotations = dict()

        for fname, cans in self.annotations['stage_1'].items():
            for cname, ans in cans.items():
                if fname in all_annotations:
                    all_annotations[fname][cname] = ans
                else:
                    all_annotations[fname] = {cname: ans}

        for fname, cans in self.annotations['stage_2'].items():
            for cname, ans in cans.items():
                if fname in all_annotations:
                    all_annotations[fname][cname] = ans
                else:
                    all_annotations[fname] = {cname: ans}

        if store_files:
            with open(os.path.join(self.respath, 'all_annotations.pickle'), 'wb') as file:
                pickle.dump(all_annotations, file)


        self.annotations['all'] = all_annotations

    def get_token_usage(self, include_ccta: bool = False):

        if os.path.exists(os.path.join(self.respath, 'token_usage.txt')):
            with open(os.path.join(self.respath, 'token_usage.txt'), 'r') as file:
                print(file.read())
            return
        else:
            if hasattr(self, 'total_input_tokens') and hasattr(self, 'total_output_tokens'):
                print(f'Total CTD input tokens: {self.total_input_tokens}')
                print(f'Total CTD output tokens: {self.total_output_tokens}')
            else:
                print('Token usage information for CTD not available.')
            
            if include_ccta:
                print(f'Total CCTA input tokens: {self.input_tokens_ccta}')
                print(f'Total CCTA output tokens: {self.output_tokens_ccta}')
                if hasattr(self, 'total_input_tokens') and hasattr(self, 'total_output_tokens'):
                    print(f'Total input tokens (CTD + CCTA): {self.total_input_tokens + self.input_tokens_ccta}')
                    print(f'Total output tokens (CTD + CCTA): {self.total_output_tokens + self.output_tokens_ccta}')

            with open(os.path.join(self.respath, 'token_usage.txt'), 'w') as file:
                if hasattr(self, 'total_input_tokens') and hasattr(self, 'total_output_tokens'):
                    file.write(f'Total CTD input tokens: {self.total_input_tokens}\n')
                    file.write(f'Total CTD output tokens: {self.total_output_tokens}\n')
                else:
                    file.write('Token usage information for CTD not available.\n')
                
                if include_ccta:
                    file.write(f'Total CCTA input tokens: {self.input_tokens_ccta}\n')
                    file.write(f'Total CCTA output tokens: {self.output_tokens_ccta}\n')
                    if hasattr(self, 'total_input_tokens') and hasattr(self, 'total_output_tokens'):
                        file.write(f'Total input tokens (CTD + CCTA): {self.total_input_tokens + self.input_tokens_ccta}\n')
                        file.write(f'Total output tokens (CTD + CCTA): {self.total_output_tokens + self.output_tokens_ccta}\n')

    def set_token_usage(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens = input_tokens
        self.total_output_tokens = output_tokens

    def get_number_of_types(self):
        if hasattr(self, 'semantic_types'):
            print(f'Number of semantic types discovered: {len(self.semantic_types["numeric"].union(self.semantic_types["non_numeric"]))}')
        else:
            print('Semantic types information not available.')

    def load_columns_descriptions(self):
        columns_descriptions = dict()
        if not self.metadata_path:
            print('Metadata path does not exist. Column descriptions not loaded. Proceeding without them.')
            self.columns_descriptions = columns_descriptions
            return
        for filename in os.listdir(self.metadata_path):
            filedir = filename[:-4]
            columns_descriptions[f'{filedir}.csv'] = dict()
            with open(os.path.join(self.metadata_path, filename), 'r') as file:
                cds = json.load(file)

            for sd in cds:
                columns_descriptions[f'{filedir}.csv'][sd['column_name']] = sd['description']

        self.columns_descriptions = columns_descriptions

    def get_coverage(self):
        total_cols = 0

        for _, df in self.dataframes.items():
            total_cols += len(df.columns)
        
        annotated_cols = 0
        for _, cans in self.annotations['all'].items():
            annotated_cols += len(cans)

        print(f'Column Coverage: {annotated_cols}/{total_cols} = {annotated_cols/total_cols:.2%}')
    
    def load_evaluation_dict(self):
        if os.path.exists(os.path.join(self.respath, 'eval_dict.pickle')):
            with open(os.path.join(self.respath, 'eval_dict.pickle'), 'rb') as file:
                self.eval_dict = pickle.load(file)
        else:
            print('Evaluation dictionary not found.')

    def evaluate(self, restart: bool = False, store_dict: bool = True) -> pd.DataFrame:
        # Load other annotations

        if self.other_annotations_paths:
            other_annotations = dict()
            for method_name, path in self.other_annotations_paths.items():
                with open(path, 'rb') as file:
                    other_annotations[method_name] = pickle.load(file)
        else:
            other_annotations = None

        #if restart is True, ignore existing eval_dict

        if not hasattr(self, 'eval_dict'):

            if os.path.exists(os.path.join(self.respath, 'eval_dict.pickle')) and not restart:
                with open(os.path.join(self.respath, 'eval_dict.pickle'), 'rb') as file:
                    self.eval_dict = pickle.load(file)
            else:
                self.eval_dict = dict()

        model = SentenceTransformer(self.models['embedding_model'])

        cos_sum_desc = {'StraTyper': 0.0}
        sum_correct = {'StraTyper': 0}
        sum_correct_non_num = {'StraTyper': 0}
        sum_correct_num = {'StraTyper': 0}
        hits = {'StraTyper': 0}
        hits_non_num = {'StraTyper': 0}
        hits_num = {'StraTyper': 0}
        count_desc = {'StraTyper': 0}
        sum_all_num = {'StraTyper': 0}
        sum_all_non_num = {'StraTyper': 0}

        if other_annotations:
            for method_name in other_annotations:
                cos_sum_desc[method_name] = 0.0
                sum_correct[method_name] = 0
                sum_correct_non_num[method_name] = 0
                sum_correct_num[method_name] = 0
                hits[method_name] = 0
                hits_non_num[method_name] = 0
                hits_num[method_name] = 0
                count_desc[method_name] = 0
                sum_all_num[method_name] = 0
                sum_all_non_num[method_name] = 0
        count_num = 0
        count_non_num = 0
        for fname, col_annotations in tqdm(self.annotations['all'].items()):
            if fname not in self.eval_dict:
                self.eval_dict[fname] = dict()

            for col, annotation_list in col_annotations.items():


                col_description = None
                if fname in self.columns_descriptions and col in self.columns_descriptions[fname]:
                    col_description = self.columns_descriptions[fname][col]
                method_types = dict()
              
                method_types['StraTyper'] = annotation_list

                if other_annotations:
                    for method_name, method_annotations in other_annotations.items():
                        if fname in method_annotations and col in method_annotations[fname]:
                            method_types[method_name] = method_annotations[fname][col]
                        else:
                            method_types[method_name] = []

                types_to_use = list(set().union(*method_types.values()))

                if col in self.eval_dict[fname]:
                    eval_response = self.eval_dict[fname][col]
                else:
                    eval_response = None
                    while not eval_response:
                        with suppress_output():
                        
                            eval_response, _, _ = judge_column_type(os.path.join(self.dataset_path, fname), log_path=os.path.join(self.respath, 'logs_judge.jsonl'), 
                                                            column=col, api=self.providers['judge'], model=self.models['judge'], 
                                                            col_description=col_description, semantic_types=types_to_use, 
                                                            num_samples=self.number_of_samples_ctd, inverted_index=self.inverted_index,
                                                            temperature=self.temperatures['judge'], explanation=False)

                    
                    if col in self.eval_dict[fname]:
                        for col_type in eval_response['judgment']:
                            self.eval_dict[fname][col]['judgment'][col_type] = eval_response['judgment'][col_type]
                    else:
                        self.eval_dict[fname][col] = eval_response

                    if not isinstance(eval_response, dict):
                        print(f"[\"{fname}\"][\"{col}\"]")
                        print(eval_response)
                        raise Exception


                if self._is_num[fname][col]: 
                    count_num += 1
                    for method in method_types:
                        sum_all_num[method] += len(method_types[method])
                else:
                    count_non_num +=1
                    for method in method_types:
                        sum_all_non_num[method] += len(method_types[method])


                try:
                    sum_correct_col = dict()
                    for method in method_types:
                        sum_correct_col[method] = 0
                    for annotation, judgment in self.eval_dict[fname][col]['judgment'].items():
                        if judgment == 'correct':
                            for method_name, annotations in method_types.items():
                                if annotation in annotations:
                                    sum_correct_col[method_name] += 1
                                    if self._is_num[fname][col]:
                                        sum_correct_num[method_name] += 1
                                    else:
                                        sum_correct_non_num[method_name] += 1

                    for method in method_types:
                        sum_correct[method] += sum_correct_col[method]

                    for method, sc in sum_correct_col.items():
                        if sc > 0:
                            hits[method] += 1
                            if self._is_num[fname][col]:
                                hits_num[method] += 1
                            else:
                                hits_non_num[method] += 1
                except:
                    print(f"[\"{fname}\"][\"{col}\"]")

                    print(eval_response)
                    raise Exception
                    


                for method_name , method_annotations in method_types.items():

                    for annotation in method_annotations:
                        annotated_emb = model.encode(annotation)

                        if col_description:
                            desc_emb = model.encode(col_description)
                            cos_sim_desc = util.cos_sim(annotated_emb, desc_emb)
                            cos_sum_desc[method_name] += cos_sim_desc.item()
                            count_desc[method_name] +=1


        if store_dict and not os.path.exists(os.path.join(self.respath, 'eval_dict.pickle')):
            with open(os.path.join(self.respath, 'eval_dict.pickle'), 'wb') as file:
                pickle.dump(self.eval_dict, file)

        
        if other_annotations:
            df_results = pd.DataFrame({
            'Method': [None] * (1 + len(other_annotations)),      
            'BERTScore Description Similarity': [None] * (1 + len(other_annotations)),  
            'Hits': [None] * (1 + len(other_annotations)),
            'Precision': [None] * (1 + len(other_annotations)),
            'Hits (NUMERICAL)': [None] * (1 + len(other_annotations)),
            'Precision (NUMERICAL)': [None] * (1 + len(other_annotations)),
            'Hits (TEXTUAL)': [None] * (1 + len(other_annotations)),
            'Precision (TEXTUAL)': [None] * (1 + len(other_annotations))
            })
        else:
            df_results = pd.DataFrame({
            'Method': [None],      
            'BERTScore Description Similarity': [None],  
            'Hits': [None],
            'Precision': [None],
            'Hits (NUMERICAL)': [None],
            'Precision (NUMERICAL)': [None],
            'Hits (TEXTUAL)': [None],
            'Precision (TEXTUAL)': [None]
            })

        rows = []

        for method in method_types:
            row = [
                method,
                f'{cos_sum_desc[method]/count_desc[method]:.3f}' if count_desc[method] > 0 else None,
                f'{hits[method]/(count_num + count_non_num):.3f}',
                f'{sum_correct[method]/(sum_all_num[method] + sum_all_non_num[method]):.3f}',
                f'{hits_num[method]/count_num:.3f}',
                f'{sum_correct_num[method]/sum_all_num[method]:.3f}',
                f'{hits_non_num[method]/count_non_num:.3f}',
                f'{sum_correct_non_num[method]/sum_all_non_num[method]:.3f}'
            ]
            rows.append(row)


        for index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                df_results.at[index, df_results.columns[col_index]] = value

        return df_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stratyper.yaml')
    parser.add_argument('--print', action='store_true', dest='print_progress', default=False, help='Whether to print progress information during execution.')
    parser.add_argument('--evaluate', action='store_true', dest='evaluate', help='Run LLM-Judge evaluation after completing StraTyper execution.')
    parser.add_argument('--restart-eval', action='store_true', dest='restart_eval', help='Whether to restart the evaluation process from scratch, ignoring any existing evaluation results.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    stratyper = Stratyper(dataset_path=config['dataset_path'], respath=config['results_path'], metadata_path=config.get('metadata_path', None),
                          other_annotations_paths=config.get('other_annotations_paths', None),
                          providers=config['providers'], models=config['models'], temperatures=config['temperatures'],
                          thresholds={'name_clusters': config['name_clusters'], 'value_clusters': config['value_clusters']}, number_of_samples_ctd=config['number_of_samples_ctd'],
                          number_of_samples_ccta=config['number_of_samples_ccta'], number_of_rows_ccta=config['number_of_rows_ccta'], 
                          number_of_rows_judge=config['number_of_samples_judge'], min_cluster_size=config['min_cluster_size'], 
                          sampling_type=config['sampling_type'], context=config['context'])
    
    if args.print_progress:
        print('Computing Metadata, Content and Seed Clusters...')

    stratyper.compute_clusters(embedding_path=config['embedding_path'], recompute=False) # recompute = True if we want to recompute embeddings

    if args.print_progress:
        print('Starting Semantic Type Discovery...')

    stratyper.column_type_discovery()

    if args.print_progress:
        print('Semantic Type Discovery Completed.')
        stratyper.get_number_of_types()
        print('Starting Closed-Set Column Type Annotation...')

    stratyper.closed_cta()

    stratyper.combine_annotations()


    if args.print_progress:
        print('Closed-Set Column Type Annotation Completed.')
        stratyper.get_token_usage(include_ccta=True)
        stratyper.get_coverage()
    
    if args.evaluate:
        if args.print_progress:
            print('Starting Evaluation with LLM-as-a-Judge...')

        stratyper.load_columns_descriptions()
        eval_df = stratyper.evaluate(restart=args.restart_eval, store_dict=True)

        if args.print_progress:
            print('Evaluation Completed.')
            print(eval_df)

        eval_df.to_csv(os.path.join(config['results_path'], 'evaluation_results.csv'), index=False)