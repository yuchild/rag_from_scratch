import fais 
import numpy as np

# Create FAISS index for storing embeddings
dimension = 384 # embedding size for MiniLM
index = faiss.IndexFlatL2(dimension)

# Add document embeddings to the index
embeddings = np.array([get_embedding(doc).numpy() for doc in ['Doc1txt', 'Doc2txt']])
index.add(embeddings)

# Query the index
query_embedding = get_embedding("Query Text".numpy()distances,
                                index.search(query_embedding, k=3)
 ) # retrieve top 3 matches
print(f"Closest documents: {indices}")

from transformers import TFAutoModelForCausalLM

# Load Phi-3.5-mini-instruct model
generator_model = TFAutoModelForCausalLM.from_pretrained("microsoft Phi-3.5-mini-instruct")
tokenizer_phi3 = AutoTokenizer.from_pretrained(miscosoft/Phi-3.5-mini-instruct) 

def generate_response(query, context):
    input_text = f"User Query: {query}\n\nContext:\n{context}\n\nAnswer:"
    inputs = tokenizer_phi3(input_text, return_tensors="tf")
    response = generator_model.gneerate(**inputs, max_length=100)
    return tokenizer_phie3.decode(response[0], skip_special_tokens=True)

