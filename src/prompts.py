from utils import df_to_text, sample_values_to_text
import pandas as pd
from collections import Counter

def cluster_type_annotation_prompt(dfs:list[pd.DataFrame], cols:list[str], num_samples:int=5, semantic_types:list[str|tuple[str]]=None, 
                                   sampling:str='length', whole_cluster_samples:bool=False, multiple_types:bool=False, context:str=None,
                                   inverted_index:dict[str, set[str]]=None) -> tuple[str, str]:
    system_message = "You are an AI assistant that strictly follows instructions."
    plurality_user_message = "all distinct semantic types that accurately describe" if multiple_types else "the semantic type that accurately describes"

    if context:
        if context.startswith('most_frequent'):
            k = context.split('_')[-1]

            context_columns = []
            for i, df in enumerate(dfs):
                co_columns = [col for col in df.columns if col!= cols[i]]
                context_columns.extend(co_columns)

            freq_counter = Counter(context_columns)
            most_common_context = [f"{col} ({str(count)}/{len(dfs)} tables)" for col, count in freq_counter.most_common(int(k)) if count > 1]

            if not most_common_context:
                context = None
                context_content_message = ""
            else:
                context_content_message = f"\n\nContext Columns that frequently co-appear with the cluster columns: {' | '.join(most_common_context)}"

        
    if context: 
        context_message = " You are also given information about columns that co-appear with the ones in the cluster."
    else:
        context_message = ""
    
    if semantic_types:
        info_message = " Their column names, a representative sample of values, and a set of possible semantic types are provided."
    else:
        info_message = " Their column names and a representative sample of values are provided."
    user_message =f"""You are given a set of columns grouped together based on embedding similarity of the values they store and their column names.{info_message}{context_message}\n\n\
Your task is to identify {plurality_user_message} the data within a group of columns."""
    
    description_message = f"""Your analysis should follow these key rules:"""

    if multiple_types:
        description_message += "\n\n* **Handle mixed types**: A single cluster may contain a mix of multiple, distinct semantic concepts. Your primary goal is to identify ALL of them. Do not try to find a single \"average\" type that partially fits all values."

    standard_rules_message = """* **Be specific**: Semantic types must be more precise than generic data types (e.g., string, integer). They should capture the intended use of the data.

* **Use all available information**: Analyze both the column names and the sample values. Expand any abbreviations to inform your choice. If names and values conflict, prioritize what the values represent.

* **Reuse or Propose (Apply per-concept)**: For EACH distinct concept you identify in the cluster, you must follow this logic:
    1. **Check for Reuse**: Look at the `Possible Semantic Types` list. If an existing type exactly and unambiguously describes that concept, you MUST use it.
    2. **Propose New**: If no existing type from the list is an exact match for that specific concept, you MUST propose a new, specific semantic type for it.
    
* **Do not select a 'closest match'**: This is critical. If an existing type is only similar but not an exact fit for a concept, you must propose a new type instead"""
        
    description_message += "\n\n" + standard_rules_message
    if context:
        description_message += f"""\n\n* **Consider the broader context**: Columns that co-appear with the ones in the cluster can be valuable context to better understand semantic types. Use the context provided to inform your choice."""    
    if multiple_types:
        description_message += """\n\n* **Avoid redundancy**: Your final list of types should be distinct and not overlap."""

    user_message += f"\n\n{description_message}"

    if whole_cluster_samples:   
        user_message += f"\n\nColumn names in cluster: {' | '.join(list(set(cols)))}"
        user_message += f"\n\nSample Values: {sample_values_to_text(dfs, cols, sampling, inverted_index, num_samples)}"
    else:
        for i, df in enumerate(dfs):
            user_message += f"\n\n{i+1}. Column Name: {cols[i]}\n"
            user_message += f"Sample Values: {sample_values_to_text([df], [cols[i]], sampling, inverted_index,num_samples)}"            

    if context and context_content_message:
        user_message += context_content_message

    if semantic_types:
        user_message += f"\n\nPossible Semantic Types: {' | '.join(list(set(semantic_types)))}"
    
    
    if multiple_types:
        output_message = """\n\nYour output should ONLY consist of a JSON dict \n{\"answer\": [semantic_type_1, semantic_type_2, ...]}. You should NOT provide any explanation.\n\nOutput: \n"""
    else:
        output_message = """\n\nYour output should ONLY consist of a JSON dict \n{\"answer\": [semantic_type]}. You should NOT provide any explanation.\n\nOutput: \n"""

    user_message += output_message
    return system_message, user_message


def column_type_annotation(df:pd.DataFrame, coverage:str="single", num_rows:int=3, column:str=None, 
                           multiple_types:bool=False, semantic_types: list[str]=None, prompt_type:str='stratyper') -> tuple[str, str]:

    system_message = "You are an AI assistant that strictly follows instructions."

    plurality_user_message = "all distinct semantic types that accurately describe" if multiple_types else "the semantic type that accurately describes"


    if coverage == "single":
        if semantic_types:
            task_message = f"""You are given the following table with given column names, a sample of rows and a set of possible semantic types. \
Your task is to identify {plurality_user_message} the column with name: {column}."""
        else:
            task_message = f"""You are given the following table with given column names and a sample of rows. \
Your task is to identify {plurality_user_message} the column with name: {column}."""
    else:
        if semantic_types:
            task_message = f"""You are given the following table with given column names, a sample of rows and a set of possible semantic types. \
Your task is to identify {plurality_user_message} each of the columns."""
        else:
            task_message = f"""You are given the following table with given column names and a sample of rows. \
Your task is to identify {plurality_user_message} each of the columns."""

    if coverage == "single":
        if multiple_types:
            output_message = f"Your output should ONLY consist of a JSON dict {{{column}: [semantic_type_1, semantic_type_2, ...]}}."
        else:
            output_message = f"Your output should ONLY consist of a JSON dict {{{column}: [semantic_type]}}."
    else:    

        columns = df.columns.tolist()
        dict_builder = ""
        for column in columns:
            if multiple_types:
                dict_builder += f"{column}: [semantic_type_1, semantic_type_2, ...], "
            else:
                dict_builder += f"{column}: [semantic_type], "
        dict_builder = dict_builder.rstrip(", ")
        if multiple_types:
            output_message = f"Your output should ONLY consist of a JSON dict {{{dict_builder}}}"
        else:
            output_message = f"Your output should ONLY consist of a JSON dict {{{dict_builder}}}"

    if prompt_type == 'stratyper':

        description_message = """\n\nYour analysis should follow these key rules:

* **Be specific**: Semantic types must be more precise than generic data types (e.g., string, integer). They should capture the intended use of the data.

* **Use all available information**: Analyze both the column names and the sample values. Expand any abbreviations to inform your choice. If names and values conflict, prioritize what the values represent."""
    
        if semantic_types:
            description_message += """\n\n* **Reuse or Propose (Apply per-concept)**: For EACH distinct concept you identify in the cluster, you must follow this logic:
    1. **Check for Reuse**: Look at the `Possible Semantic Types` list. If an existing type exactly and unambiguously describes that concept, you MUST use it.
    2. **Propose New**: If no existing type from the list is an exact match for that specific concept, you MUST propose a new, specific semantic type for it.

* **Do not select a 'closest match'**: This is critical. If an existing type is only similar but not an exact fit for a concept, you must propose a new type instead."""
        if multiple_types:
            description_message += """\n\n* **Avoid redundancy**: Your final list of types should be distinct and not overlap."""
    elif prompt_type == 'naive':
        if semantic_types:
            description_message = "You may or may not use one or more semantic types from the `Possible Semantic Types` list. If no existing type from the list is an exact match for the column, you MUST propose a new, specific semantic type for it."
        else:
            description_message = ""
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


    user_message = task_message + description_message + "\n\n" + output_message + f"\n\nInput: \n\n"

    user_message += df_to_text(df, num_rows=num_rows)

    if semantic_types:
        user_message += f"\n\nPossible Semantic Types: {' | '.join(semantic_types)}"

    user_message += "\n\nOutput: \n"
    return system_message, user_message

def closed_column_type_annotation(df:pd.DataFrame, column:str, semantic_types:list[str], sampling:str='length', numerical:bool=False,
                                  inverted_index:dict[str, set[str]]=None,num_rows:int=3, 
                                  multiple_types:bool=False, num_samples:int=5) -> tuple[str, str]:

    system_message = "You are an AI assistant that strictly follows instructions."
    if multiple_types:
        edit_message = "one or more semantic types"
    else:
        edit_message = "a semantic type"

    user_message = f"""You are an expert data analyst specializing in semantic type detection. You will be given a data column, including its name, sample values, and surrounding context from a table. \
    Your task is to assign {edit_message} from a provided list.\n\n"""
    
    guidelines_message = "Evaluation Steps:\n\n"
    
    guidelines_message += "1. **Analyze All Evidence**: Look at the **Column Name**, the **Sample Values**, and the **Table Context**. Use *all* of this information to understand the different *kinds* of information present in the column.\n\n"

    if multiple_types:
        guidelines_message += "2. **Iterate and Test Each Type**: You must evaluate **every single type** in the `Possible Semantic Types` list, one by one.\n\n"
        
        if numerical:
            guidelines_message += "3. **Test Each Type**: For each type, ask: 'Based on the **Column Name** and **Table Context**, does this type accurately describe what the numerical values *measure* or *represent*?'\n\n"
        else:
            guidelines_message += "3. **Test Each Type**: For each type, ask: 'Based on **all the evidence**, are *any* of the **Sample Values** direct, unambiguous instances of this type?' (e.g., 'BROOKLYN' is an instance of 'NYC Borough').\n\n"
        
        guidelines_message += "4. **Collect All Matches**: Build a list of *all* types that passed this test. This list can be empty if no types match.\n\n"
        guidelines_number = 5
    else: 
        guidelines_message += "2. **Find the Single Best Match**: You must find the *one* type from the list that best describes the column.\n\n"
        
        if numerical:
            guidelines_message += "3. **Evaluate Match**: Based on the **Column Name** and **Table Context**, determine which *single* type from the list best describes what the numerical values *measure* or *represent*.\n\n"
        else:
            guidelines_message += "3. **Evaluate Match**: Find the *single* type from the list that the **Sample Values** are direct, unambiguous instances of. Use the Column Name and Context to resolve ambiguity.\n\n"
        guidelines_number = 4
    guidelines_message += f"{guidelines_number}. **Avoid Related Types**: Avoid selecting types that are only thematically associated or related.\n\n"
    guidelines_message += f"{guidelines_number + 1}. **Handle No Match**: If no types from the list are a good match, you MUST respond with 'None'. Do not select the closest option.\n\n"

    user_message += guidelines_message

    user_message += "Here is the data for your analysis:\n\n"
    
    user_message += f"**Table Context** (sample rows):\n"
    user_message += df_to_text(df, column_names=True, num_rows=num_rows)
    
    user_message += f"\n\n**Target Column Name**: {column}\n\n"
    
    user_message += f"**Additional Value Samples**: {sample_values_to_text([df], [column], sampling, inverted_index, num_samples)}\n\n"
    
    try:
        user_message += f"**Possible Semantic Types**: {' | '.join(semantic_types)}\n\n"
    except:
        print(semantic_types)
        print(column)
        raise Exception("Error in formatting closed column type annotation prompt.")
    
    
    if multiple_types:
        response_message = "[semantic_type_1, semantic_type_2, ...]"
    else:
        response_message = "semantic_type"

    output_message = f"\n\nYour output should ONLY consist of a JSON dict {{{column}: {response_message}}}. You should NOT provide any explanation."

    user_message += output_message + "\n\nOutput: \n"

    return system_message, user_message

def judge_column_type_prompt(df:pd.DataFrame, column:str, semantic_types:list[str], col_description:str=None, explanation:str=False, 
                             sampling:str='length', inverted_index:dict[str, set[str]]=None, num_samples:int=5, num_rows:int=3) -> tuple[str, str]:

    system_message = "You are a data expert evaluating the correctness of semantic type annotations for table columns. You strictly follow instructions."

    if col_description:
        description_instruction = """\n5. **Description:** Use the provided column description as a hint, but if it is generic (e.g., "Values", "Data"), prioritize the evidence in the actual data samples."""
    user_message = f"""You are an expert data evaluator. You are given a table snippet and a set of "Additional Samples" from a specific column. Your task is to assess the correctness of proposed semantic type annotations for that column.

### Evaluation Criteria (Read Carefully):
1. **Subset Match (CRITICAL):** The column likely contains **mixed types**. An annotation is **correct** if it accurately describes **any identifiable subset** of the values found in EITHER the "Table Rows" OR the "Additional Value Samples". It does not need to describe 100% of the values.
2. **Semantic Meaning:** The annotation must describe the *real-world concept*.
   - Generic data types (e.g., "string", "number", "numerical", "percentage") are **incorrect**.
3. **Handling Header Repetitions:**
   - If an annotation repeats the column header, it is **correct** as long as the header describes a specific real-world entity.
   - It is **incorrect** only if the header is generic, abstract, or purely structural.
4. **Context:** Use the sibling columns to infer the semantic meaning.{description_instruction if col_description else ""}


---

### Data to Evaluate:

**Table Schema and Sample Rows:**
"""

    user_message += df_to_text(df, num_rows=num_rows)

    try:
        user_message += f"\n\n**Column Under Review:** {column}\n\n"
        if col_description:
            user_message += f"**Column Description:** {col_description}\n\n"
        user_message += f"**Additional Value Samples (Crucial for mixed types):** {sample_values_to_text([df], [column], sampling, inverted_index, num_samples)}\n\n"
        user_message += f"**Annotated Semantic Types to Judge:** {' | '.join(semantic_types)}\n\n"
    except:
        print(df)
        print(column)
        print(semantic_types)
        raise Exception("Error in formatting column type judgment prompt.")

    judgment_string = ""
    for semantic_type in semantic_types:
        judgment_string += f""""{semantic_type}": "correct" or "incorrect", """
    judgment_string = judgment_string.rstrip(", ")
    judgment_output = f""""judgment": {{{judgment_string}}} """

    user_message += "---\n\n### Output Format:\n\n"

    if explanation:
        user_message += f"""Your output should ONLY consist of a JSON dict.
    {{  {judgment_output},
    "explanation": "Brief reasoning for the judgment, including any relevant details from the data or context."}}\n\nOutput: \n"""
    else:
        user_message += f"""Your output should ONLY consist of a JSON dict

    {{{judgment_output}}}\nYou should NOT provide any explanation.\n\nOutput: \n"""
    return system_message, user_message