import os
import torch
import json
import faiss
import re
import argparse
import logging
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from openai import OpenAI
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tkinter import Tk, Button, filedialog
import PyPDF2
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

# Disable parallelism in tokenizers to avoid warnings and potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='rag_system.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load advanced embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

# Load T5 model and tokenizer for query rewriting
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to('cuda' if torch.cuda.is_available() else 'cpu')

# Function to preprocess text and split into chunks
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\s+', ' ', text).strip().lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Asynchronous function to load and preprocess vault content
async def load_vault_content(filepath):
    async with aiofiles.open(filepath, 'r', encoding='utf-8') as file:
        content = await file.read()
    return preprocess_text(content).split("\n")  # Split by lines to treat each line as a document

# Function to create FAISS index with parallel processing and batch encoding
def create_faiss_index(docs, batch_size=64):
    embeddings = []
    for i in range(0, len(docs), batch_size):
        batch_embeddings = embedding_model.encode(docs[i:i + batch_size], convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    return index, embeddings

# Function to get relevant context from vault
def get_relevant_context(rewritten_input, faiss_index, vault_content, top_k=1):  # Lower top_k for more precise context
    input_embedding = embedding_model.encode([rewritten_input], convert_to_tensor=True)
    _, indices = faiss_index.search(input_embedding.cpu().numpy(), top_k)
    relevant_context = [vault_content[idx] for idx in indices[0]]
    return relevant_context

# Function to rewrite user query
def rewrite_query(user_input_json, conversation_history):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    input_text = f"rewrite: {user_input} context: {context}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = t5_model.generate(input_ids, max_length=100, num_return_sequences=1)
    rewritten_query = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return json.dumps({"Rewritten Query": rewritten_query})

# Function to log queries and responses
def log_query_and_response(user_input, response):
    logging.info(f"User Query: {user_input}")
    logging.info(f"Response: {response}")

# Function to get user feedback
def get_user_feedback():
    feedback = input("Was the response helpful? (yes/no): ")
    if feedback.lower() in ['yes', 'no']:
        logging.info(f"User Feedback: {feedback}")
    else:
        logging.info(f"User Feedback: unclear response ({feedback})")

# Function to rank context based on similarity to the rewritten query
def rank_context(rewritten_input, context_chunks):
    input_embedding = embedding_model.encode([rewritten_input], convert_to_tensor=True)
    chunk_embeddings = embedding_model.encode(context_chunks, convert_to_tensor=True)
    similarities = (input_embedding @ chunk_embeddings.T).cpu().numpy()
    ranked_chunks = [context_chunks[idx] for idx in similarities.argsort()[0][::-1]]
    return ranked_chunks

# Function to interact with Ollama model
def ollama_chat(user_input, system_message, faiss_index, vault_content, ollama_model, conversation_history):
    try:
        conversation_history.append({"role": "user", "content": user_input})
        
        if len(conversation_history) > 1:
            query_json = {
                "Query": user_input,
                "Rewritten Query": ""
            }
            rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history)
            rewritten_query_data = json.loads(rewritten_query_json)
            rewritten_query = rewritten_query_data["Rewritten Query"]
            print(PINK + "Original Query: " + user_input + RESET_COLOR)
            print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
        else:
            rewritten_query = user_input
        
        relevant_context = get_relevant_context(rewritten_query, faiss_index, vault_content)
        if relevant_context:
            # Rank the retrieved context to pull the most relevant piece
            ranked_context = rank_context(rewritten_query, relevant_context)
            context_str = "\n".join(ranked_context)
            print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
        else:
            print(CYAN + "No relevant context found." + RESET_COLOR)
        
        user_input_with_context = user_input
        if relevant_context:
            user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
        
        conversation_history[-1]["content"] = user_input_with_context
        
        messages = [
            {"role": "system", "content": system_message},
            *conversation_history
        ]
        
        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages,
            max_tokens=2000,
        )
        
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
        
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during chat: {e}")
        return "Sorry, there was an error processing your request."

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = asyncio.run(load_vault_content("vault.txt"))

# Check if vault_content is empty
if not vault_content:
    raise ValueError("The vault content is empty. Please check the 'vault.txt' file.")

print("Vault content:", vault_content)  # Print vault content for debugging

# Generate embeddings for the vault content using advanced embedding model
print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
faiss_index, vault_embeddings_tensor = create_faiss_index(vault_content)

# Conversation loop
print("Starting conversation loop...")
conversation_history = []
system_message = """### Detailed Guide for AI Model: Naval Signal Simulation
#### Introduction
This guide is designed to help AI models respond accurately in a naval
signal simulation. The model will follow a structured format for
signaling, starting with the initiation of a simulation and then
responding to various commands.
#### Starting the Simulation
1. **User Command:** "Start simulation"
2. **AI Response:**
   - **Signal:** "OFFICER OF THE WATCH MANEUVERS SERIAL START NOW"
#### Ending a Serial
1. **User Command:** "End the serial"
2. **AI Response:**
   - **Signal:** "OFFICER OF THE WATCH MANEUVERS SERIAL END"
#### Replenishment at Sea
1. **User Command:** "Send out signals for a replenishment at sea"
2. **AI Response:**
   - **Signal:** "PREPARE FOR REPLENISHMENT AT SEA"
4. **Commencing Replenishment**
   - **User Command:** "Now what"
   - **AI Response:**
     - **Signal:** "COMMENCE FUEL TRANSFER"
1. **Turning and Speed Adjustment:**
   - **User Command:** "Tell all ships to turn to port to a new course
of 79 and increase speed to 16 knots. Do this by the delayed executive
method."
   - **AI Response:**
     - **Turn Signal:** "TURN PORT 079 DELAYED EXEC"
     - **Speed Signal:** "SPEED 16 DELAYED EXEC"
   - **Expected Responses:** Ships will acknowledge with the
affirmative pennant. Once all acknowledgments are received, the
execution signal ("EXEC") will be hoisted to indicate the commands
should be executed.
2. **Emergency Turn:**
   - **User Command:** "How should I tell another ship to turn to port
to 180 in an emergency?"
   - **AI Response:**
     - **Signal:** "EMERG TURN PORT 180"
3. **Wheeling:**
   - **User Command:** "Tell all ships to wheel to 190"
   - **AI Response:**
     - **Signal:** "CORPEN 190"
4. **Stationing:**
   - **User Command:** "Tell the receiving ship to a station 2000
yards on your port beam"
   - **AI Response:**
     - **Signal:** "STATION PORT 90—1" (to indicate 1 nautical mile on
the port beam using a relative bearing)
#### Ending the Simulation
1. **User Command:** "STOP SIMULATION"
2. **AI Response:** The AI should acknowledge the end of the
simulation."""

while True:
    user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break

    response = ollama_chat(user_input, system_message, faiss_index, vault_content, args.model, conversation_history)
    log_query_and_response(user_input, response)
    print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
    get_user_feedback()

# Functions for uploading PDF, TXT, and JSON files
async def convert_pdf_to_text():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        async with aiofiles.open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            text = ''
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                if page.extract_text():
                    text += page.extract_text() + " "
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < 500:  # Reduce chunk size to 500 characters
                    current_chunk += (sentence + " ").strip()
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk)
            async with aiofiles.open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    await vault_file.write(chunk.strip() + "\n")
            print(f"PDF content appended to vault.txt with each chunk on a separate line.")

async def upload_txtfile():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        async with aiofiles.open(file_path, 'r', encoding="utf-8") as txt_file:
            text = await txt_file.read()
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < 500:  # Reduce chunk size to 500 characters
                    current_chunk += (sentence + " ").strip()
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk)
            async with aiofiles.open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    await vault_file.write(chunk.strip() + "\n")
            print(f"Text file content appended to vault.txt with each chunk on a separate line.")

async def upload_jsonfile():
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        async with aiofiles.open(file_path, 'r', encoding="utf-8") as json_file:
            data = json.load(json_file)
            
            # Flatten the JSON data into a single string
            text = json.dumps(data, ensure_ascii=False)
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < 500:  # Reduce chunk size to 500 characters
                    current_chunk += (sentence + " ").strip()
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk)
            async with aiofiles.open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    await vault_file.write(chunk.strip() + "\n")
            print(f"JSON file content appended to vault.txt with each chunk on a separate line.")

# Create the main window for file upload
root = Tk()
root.title("Upload .pdf, .txt, or .json")

# Create buttons for uploading files
pdf_button = Button(root, text="Upload PDF", command=lambda: asyncio.run(convert_pdf_to_text()))
pdf_button.pack(pady=10)

txt_button = Button(root, text="Upload Text File", command=lambda: asyncio.run(upload_txtfile()))
txt_button.pack(pady=10)

json_button = Button(root, text="Upload JSON File", command=lambda: asyncio.run(upload_jsonfile()))
json_button.pack(pady=10)

# Run the main event loop
root.mainloop()