'''
Author: Saurabh Arun Yadgire
LLM used: Llama-2-13b-chat-hf
Embedding Model Used: sentence-transformers/all-MiniLM-L6-v2
'''

# Import Libraries
import torch
from torch import cuda, bfloat16
import sentence_transformers
import transformers
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
# -----------------------------------------------------------------------

# 1. Define LLM
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model_id = 'meta-llama/Llama-2-13b-chat-hf'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    bnb_4bit_quant_type='nf4',
    load_in_8bit_fp32_cpu_offload=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# initialize HF token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

# tokentizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

# pipeline
llm_pipeline = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=1024,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# 2. Define Embedding Model
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

# 3. Read Data
# considering the file is in the same directory as the code
path = "take_home_dataset.csv"

# 4. Preprocess Data
df = pd.read_csv(path, delimiter=";")
columns_to_use = ['Date', 'Order_ID', 'Product_Category', 'Delivery_distance']
df = df[columns_to_use]
df.reset_index(inplace=True)
df.rename(columns={'index': 'Row_ID'}, inplace=True)

# metadata
metadata = {
    'Row_ID' : 'numeric',
    'Date' : "datetime",
    'Order_ID' : 'numeric',
    'Product_Category' : 'categorical',
    'Delivery_distance' : 'numeric'
}

# helper functions
# -----------------------------------------------------------------------
def get_column_name(query: str, embeddings) -> str:
    """
    Retrieves the column name corresponding to the provided query.

    Parameters:
        query (str): The query string for similarity search.
        embeddings (Embeddings): The embeddings object containing precomputed embeddings.

    Returns:
        str: The column name most similar to embedding columns.
    """
    return embeddings.similarity_search(query, k=1)[0].page_content


def get_column_type(column: str, metadata: dict):
    """
    Retrieves the type of the specified column from the metadata.

    Parameters:
        column (str): The name of the column.
        metadata (dict): The metadata containing column information.

    Returns:
        Any: The type of the specified column.
    """
    return metadata[column]

def extract_similar_value(query: str, data: pd.DataFrame, column: str, embed_model) -> str:
    '''
    Extracts a value similar to the given query from the specified categorical column.

    Parameters:
        query (str): The query string for similarity search.
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the categorical column.
        embed_model (Embeddings): The embeddings model for similarity search.

    Returns:
        str: The extracted value from the column that is most similar to the query.
    '''
    col_values_embedding = FAISS.from_texts(data[column].unique().astype(str), embed_model)
    extracted_value = col_values_embedding.similarity_search(query, k=1)[0].page_content
    return extracted_value

def get_final_query(query: str, embed_model, column_embeddings, data: pd.DataFrame, metadata: dict) -> str:
    """
    Constructs the final query based on the provided query, embeddings, column embeddings, data, and metadata.

    Parameters:
        query (str): The original query string.
        embed_model (Embeddings): The embeddings model for similarity search.
        column_embeddings (Embeddings): The embeddings model for column names.
        data (pd.DataFrame): The DataFrame containing the data.
        metadata (dict): The metadata containing column information.

    Returns:
        str: The final query constructed based on the provided parameters.
    """
    similar_column = get_column_name(query, column_embeddings)
    column_type = get_column_type(similar_column, metadata)
    if column_type == 'categorical':
        extracted_column_value = extract_similar_value(query, data, similar_column, embed_model)
        final_query = f"{similar_column} {extracted_column_value}"
        return final_query
    else:
        final_query = query
        return final_query

def extract_query_from_response(answer: str) -> str:
    """
    Extracts the pandas query from the provided response.

    Parameters:
        answer (str): The response containing the pandas query.

    Returns:
        str: The extracted pandas query.
    """
    pandas_query = answer.split("\n")[-1]
    return pandas_query

def get_row_id(df: pd.DataFrame, answer: str):
    """
    Retrieves the row indices based on the provided DataFrame and pandas query extracted from the response.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        answer (str): The response containing the pandas query.

    Returns:
        List[int]: The list of row indices.
    """
    pandas_query = extract_query_from_response(answer)
    result_df = eval(pandas_query)
    if isinstance(result_df, pd.DataFrame):
        indices = result_df.index.tolist()
    else:
        indices = df.index[result_df].tolist()
    return indices

def extract_col_and_value(extraction_chain, answer: str):
    """
    Extracts the column name and corresponding value from the provided response.

    Parameters:
        answer (str): The response containing the extraction chain response.

    Returns:
        Tuple[str, str]: A tuple containing the extracted column name and value.
    """
    pandas_query = extract_query_from_response(answer)
    extraction_chain_response = extraction_chain.invoke(pandas_query)
    column_name = extraction_chain_response.split("END")[1].strip().split("column_name:")[1].split("\n")[0].strip()
    value = extraction_chain_response.split("END")[1].strip().split("value:")[1].split("\n")[0].strip()
    return column_name, value

def generate_output(answer: str, df: pd.DataFrame) -> dict:
    """
    Generates output containing column name, value, and row indices based on the provided response and DataFrame.

    Parameters:
        answer (str): The response containing the extraction chain response.
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        dict: A dictionary containing the column name, value, and row indices.
    """
    column_name, value = extract_col_and_value(extraction_chain, answer)
    indices = get_row_id(df, answer)
    return {"column_name": column_name, "value": value, "row_ids": indices}

def run(query: str):
    """
    Executes a series of steps to process a query and generate output.

    Parameters:
        query (str): The query string to be processed.
    """
    final_query = get_final_query(query, embed_model, column_embeddings, df, metadata)
    answer = chain.invoke(final_query)
    output = generate_output(answer, df)
    print(output)

# -----------------------------------------------------------------------
    
# Implementation
    
column_embeddings = FAISS.from_texts(columns_to_use, embed_model)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# prompts
pandas_prompt_template = """<<SYS>>
    You are a data analyst working on a pandas datframe 'df'.\n Use df.head for your reference.Your job is to return a Pandas expression that answers the query.<</SYS>>
    [INST]
    Instructions: {instructions}
    df: {df_head}
    Query: {query}
    PRINT ONLY THE PANDAS EXPRESSION after Expression--> AT THE END OF RESPONSE.
    [/INST]
    END
"""

extraction_prompt = """<<SYS>>
    You are a data analyst working on a pandas datframe 'df'.\nYour job is to extract 'column_name' and 'value' from the pandas expression.\n</SYS>>
    [INST]
    PRINT ONLY THE `column_name` AND `value` AFTER  SOLUTION--> at the end of response.
    expression: {expression}
    [/INST]
    END
"""

instructions_prompt = """
1. RETURN THE PANDAS EXPRESSION ONLY.
Here are the possible columns:
```
[{columns}]
```
"""

prompt = ChatPromptTemplate.from_template(pandas_prompt_template)
final_prompt = prompt.partial(instructions = instructions_prompt)
extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt)

# chains
chain = (
    {"query": RunnablePassthrough(), "columns": lambda x: list(df.columns), "df_head": lambda x:df.head(5).to_dict()}
    | final_prompt
    | llm
    | StrOutputParser()
)

extraction_chain =  (
    {"expression": RunnablePassthrough()}
    | extraction_prompt
    | llm
    | StrOutputParser()
)

query = input("Enter your query: ")
run(query)