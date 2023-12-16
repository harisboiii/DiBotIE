import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
import gdown
from sentence_transformers import SentenceTransformer

# Download the file
file_id = '1P3Nz6f3KG0m0kO_2pEfnVIhgP8Bvkl4v'
url = f'https://drive.google.com/uc?id={file_id}'
excel_file_path = os.path.join(os.path.expanduser("~"), 'medical_data.csv')

gdown.download(url, excel_file_path, quiet=False)

# Read the CSV file into a DataFrame using 'latin1' encoding
try:
    medical_df = pd.read_csv(excel_file_path, encoding='utf-8')
except UnicodeDecodeError:
    medical_df = pd.read_csv(excel_file_path, encoding='latin1')

def remove_digits_with_dot(input_string):
    # Define a regex pattern to match digits with a dot at the beginning of the string
    pattern = re.compile(r'^\d+\.')

    # Use sub() method to replace the matched pattern with an empty string
    result_string = re.sub(pattern, '', input_string)

    return result_string

medical_df["Questions"] = medical_df["Questions"].apply(remove_digits_with_dot)

medical_df = medical_df[medical_df["Answers"].notna()]

from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR("hkunlp/instructor-large")
corpus = medical_df["Answers"].apply(lambda x:[x]).tolist()
answer_embeddings = []
for answer in corpus:
    answer_embeddings.append(model.encode(answer))
    
answer_embeddings = np.array(answer_embeddings)
answer_embeddings = answer_embeddings.reshape(148, 768)

def get_answer(query):
    
    query = [['Represent the Wikipedia question for retrieving supporting documents: ', query]]

    query_embedding = model.encode(query)

    similarities = cosine_similarity(query_embedding, answer_embeddings)

    retrieved_doc_id = np.argmax(similarities)

    q = medical_df.iloc[retrieved_doc_id]["Questions"]
    a = medical_df.iloc[retrieved_doc_id]["Answers"]
    r = medical_df.iloc[retrieved_doc_id]["References"]
    
    return (q, a, r)

# Streamlit app
st.title("Medical QA System")

user_input = st.text_input("Ask a medical question:")
if user_input:
    result = get_answer(user_input)
    st.subheader("Question:")
    st.write(result[0])
    st.subheader("Answer:")
    st.write(result[1])
    st.subheader("References:")
    st.write(result[2])
