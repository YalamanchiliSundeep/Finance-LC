import streamlit as st
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDFs
import pytesseract  # For OCR of scanned images
from PIL import Image
import os
import pandas as pd  # To display the dataset
import pickle
import re
import time
import io  # To handle file as byte stream
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Sentence Transformer for embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index file (for embedding storage)
INDEX_FILE = "faiss_rag_index.index"
METADATA_FILE = "metadata_rag.pickle"
DATA_FILE = "documents_metadata.csv"

# Initialize FAISS index and metadata
if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    st.success("FAISS index and metadata loaded from disk.")
else:
    index = faiss.IndexFlatL2(384)  # 384-dim vector size from MiniLM
    metadata = {}
    st.info("New FAISS index created.")

# Function to create embeddings using Sentence Transformer
def create_embeddings(text):
    return embedder.encode([text], convert_to_tensor=False)[0]

# Function to add document embeddings to FAISS index
def add_document_to_faiss(doc_text, doc_name):
    embedding = create_embeddings(doc_text)
    index.add(np.array([embedding], dtype=np.float32))
    metadata[len(metadata)] = {"document_name": doc_name, "content": doc_text}
    st.success(f"Document '{doc_name}' added to FAISS index.")

# Initialize an empty DataFrame to store document metadata
if os.path.exists(DATA_FILE):
    df_metadata = pd.read_csv(DATA_FILE)
else:
    df_metadata = pd.DataFrame(columns=["Document Name", "Issuer", "Beneficiary", "Amount", "Expiration Date", "Contract Number", "Project Name", "Purpose", "Cancellation", "Renewal", "Extracted Text"])

# Save metadata to CSV
def save_metadata_to_csv():
    df_metadata.to_csv(DATA_FILE, index=False)
    st.success("Metadata saved to CSV file.")

# Function to extract text from scanned PDFs using OCR
def extract_text_from_scanned_pdf(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        if not pdf_bytes:
            return "No PDF data found."

        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        full_text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            gray = np.array(img.convert('L'))
            text = pytesseract.image_to_string(gray)
            full_text += f"--- Page {page_num + 1} ---\n{text}\n"

        return full_text

    except Exception as e:
        return f"Error extracting text from scanned PDF: {str(e)}"

# Extract text from regular (text-based) PDFs
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return ""

# Extract text from DOCX
def extract_text_from_docx(docx_file):
    try:
        return docx2txt.process(docx_file)
    except Exception as e:
        st.error(f"Error extracting DOCX: {str(e)}")
        return ""

# Extract text from scanned images
def extract_text_from_image(image_file):
    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

# Handle various file types and extract text
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
        if len(extracted_text.strip()) < 100:  # If text is too short, assume it's a scanned PDF
            extracted_text = extract_text_from_scanned_pdf(uploaded_file)
        return extracted_text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        return extract_text_from_image(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode('utf-8')
    else:
        st.error("Unsupported file type.")
        return ""

# Use GPT to extract refined metadata, including Purpose
def extract_metadata_with_gpt(document_text):
    prompt = f"""
The following is a legal document. Extract the following details if present:

1. Issuer (the bank that issued the letter of credit)
2. Beneficiary (the party who receives the benefit)
3. Amount in USD
4. Expiration Date
5. Contract Number
6. Project Name
7. Purpose (the reason or intent behind the document)
8. Cancellation policy
9. Renewal policy

Be as specific as possible, and return the details in JSON format with the following keys: issuer, beneficiary, amount, expiration_date, contract_number, project_name, purpose, cancellation, renewal.

Document text:
\"\"\"{document_text}\"\"\"
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        st.error(f"Error extracting metadata with GPT: {str(e)}")
        return ""

# Parse the metadata response with fallback for missing fields
def parse_refined_metadata(metadata_response, document_text):
    try:
        metadata_dict = eval(metadata_response.replace("null", "None"))
        metadata_dict["issuer"] = metadata_dict.get("issuer", "Unknown Issuer")
        metadata_dict["beneficiary"] = metadata_dict.get("beneficiary", "Unknown Beneficiary")
        metadata_dict["amount"] = metadata_dict.get("amount", "Unknown Amount")
        metadata_dict["expiration_date"] = metadata_dict.get("expiration_date", "Unknown Expiration Date")
        metadata_dict["contract_number"] = metadata_dict.get("contract_number", "Unknown Contract Number")
        metadata_dict["project_name"] = metadata_dict.get("project_name", "Unknown Project Name")
        metadata_dict["purpose"] = metadata_dict.get("purpose", "Unknown Purpose")

        cancellation_text = metadata_dict.get("cancellation", "Not mentioned in the document")
        renewal_text = metadata_dict.get("renewal", "Not mentioned in the document")
        metadata_dict["cancellation"] = cancellation_text if cancellation_text != "Not mentioned in the document" else "Not mentioned in the document"
        metadata_dict["renewal"] = renewal_text if renewal_text != "Not mentioned in the document" else "Not mentioned in the document"

        return metadata_dict

    except Exception as e:
        st.error(f"Error parsing metadata: {str(e)}")
        return {
            "issuer": "Unknown Issuer",
            "beneficiary": "Unknown Beneficiary",
            "amount": "Unknown Amount",
            "expiration_date": "Unknown Expiration Date",
            "contract_number": "Unknown Contract Number",
            "project_name": "Unknown Project Name",
            "purpose": "Unknown Purpose",
            "cancellation": "Not mentioned in the document",
            "renewal": "Not mentioned in the document"
        }

# Save extracted text and metadata to the in-memory DataFrame
def save_to_dataframe(doc_name, extracted_text, issuer, beneficiary, amount, expiration_date, contract_number, project_name, purpose, cancellation, renewal):
    global df_metadata
    new_data = pd.DataFrame({
        "Document Name": [doc_name],
        "Issuer": [issuer],
        "Beneficiary": [beneficiary],
        "Amount": [amount],
        "Expiration Date": [expiration_date],
        "Contract Number": [contract_number],
        "Project Name": [project_name],
        "Purpose": [purpose],
        "Cancellation": [cancellation],
        "Renewal": [renewal],
        "Extracted Text": [extracted_text]  # Store the full extracted text here
    })
    df_metadata = pd.concat([df_metadata, new_data], ignore_index=True)
    st.success(f"Metadata for '{doc_name}' added to dataset.")

# Function to display the dataset of stored documents
def display_dataset():
    st.subheader("Documents Metadata in the Dataset")
    st.dataframe(df_metadata.drop(columns=["Extracted Text"]))  # Exclude 'Extracted Text' column when showing the table

# Search FAISS index with a query
def search_faiss(query, k=3):
    if len(metadata) == 0:
        st.warning("The FAISS index is empty. Please upload documents first.")
        return []
    query_embedding = create_embeddings(query)
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
    results = []
    for idx in indices[0]:
        if idx != -1 and idx in metadata:
            results.append(metadata[idx])
    return results

# Generate an answer using GPT-4 based on the retrieved documents (RAG approach)
def generate_answer_with_gpt(documents, query, retries=3, delay=5):
    context = "\n\n".join([f"Document Name: {doc['document_name']}\nContent: {doc['content'][:1000]}" for doc in documents])
    
    prompt = f"""
    You are an assistant that answers questions based on provided documents.

    Documents:
    \"\"\"{context}\"\"\"

    Question:
    {query}

    Provide a concise and accurate answer based on the documents above. If the answer is not found, say "The information is not available in the provided documents."
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        return response.choices[0].message["content"].strip()

    except openai.error.RateLimitError as e:
        if retries > 0:
            st.warning(f"Rate limit reached. Retrying in {delay} seconds...")
            time.sleep(delay)
            return generate_answer_with_gpt(documents, query, retries=retries-1, delay=delay)
        else:
            st.error("Rate limit reached. Unable to process the request after multiple attempts.")
            return "An error occurred while generating the answer."
    except Exception as e:
        st.error(f"Error generating answer with GPT: {str(e)}")
        return "An error occurred while generating the answer."

# Delete all data (FAISS index, metadata, and CSV file)
def delete_all_data():
    global metadata, df_metadata, index
    index.reset()
    metadata = {}
    df_metadata = pd.DataFrame(columns=["Document Name", "Issuer", "Beneficiary", "Amount", "Expiration Date", "Contract Number", "Project Name", "Purpose", "Cancellation", "Renewal", "Extracted Text"])
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    st.success("All data has been deleted.")

# Delete specific row
def delete_specific_row(doc_name):
    global df_metadata
    df_metadata = df_metadata[df_metadata["Document Name"] != doc_name]
    save_metadata_to_csv()
    st.success(f"Document '{doc_name}' deleted.")

# Streamlit UI
st.title("Document Metadata and RAG Model")

# Display the dataset of documents before any processing
display_dataset()

# File uploader for documents
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT, or scanned image)", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    extracted_text = extract_text_from_file(uploaded_file)
    if extracted_text:
        doc_name = uploaded_file.name

        # Use GPT to extract metadata
        refined_metadata_response = extract_metadata_with_gpt(extracted_text)
        st.write("Extracted Metadata (raw response):")
        st.code(refined_metadata_response)

        # Parse the metadata
        refined_metadata = parse_refined_metadata(refined_metadata_response, extracted_text)

        # Display the extracted metadata
        st.write("Parsed Metadata:")
        st.json(refined_metadata)

        # Save the document and its metadata to the in-memory DataFrame
        save_to_dataframe(
            doc_name,
            extracted_text,
            issuer=refined_metadata.get("issuer", "Unknown Issuer"),
            beneficiary=refined_metadata.get("beneficiary", "Unknown Beneficiary"),
            amount=refined_metadata.get("amount", "Unknown Amount"),
            expiration_date=refined_metadata.get("expiration_date", "Unknown Expiration Date"),
            contract_number=refined_metadata.get("contract_number", "Unknown Contract Number"),
            project_name=refined_metadata.get("project_name", "Unknown Project Name"),
            purpose=refined_metadata.get("purpose", "Unknown Purpose"),
            cancellation=refined_metadata.get("cancellation", "Not mentioned in the document"),
            renewal=refined_metadata.get("renewal", "Not mentioned in the document")
        )

        # Add the document to the FAISS index
        add_document_to_faiss(extracted_text, doc_name)

        # Display updated dataset
        display_dataset()

# Save FAISS index and metadata
if st.button("Save FAISS Index and Metadata"):
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    st.success("FAISS index and metadata saved successfully.")

# Button to save metadata to CSV
if st.button("Save Metadata to CSV"):
    save_metadata_to_csv()

# Button to delete all data
if st.button("Delete All Data"):
    delete_all_data()

# Streamlit UI for asking questions based on the documents
st.subheader("Ask a Question Based on the Documents")

user_query = st.text_input("Enter your question:")

if st.button("Search and Generate Answer"):
    if user_query:
        # Search FAISS for relevant metadata
        search_results = search_faiss(user_query)
        
        if search_results:
            # Generate an answer using the retrieved metadata and GPT-4
            answer = generate_answer_with_gpt(search_results, user_query)
            st.subheader("Generated Answer")
            st.write(answer)
        else:
            st.write("No relevant documents found.")
    else:
        st.warning("Please enter a question.")

# Manual Entry of Metadata
st.subheader("Manual Entry of Metadata")

doc_name_manual = st.text_input("Document Name")
issuer_manual = st.text_input("Issuer")
beneficiary_manual = st.text_input("Beneficiary")
amount_manual = st.text_input("Amount")
expiration_date_manual = st.text_input("Expiration Date")
contract_number_manual = st.text_input("Contract Number")
project_name_manual = st.text_input("Project Name")
purpose_manual = st.text_area("Purpose")
cancellation_manual = st.text_area("Cancellation")
renewal_manual = st.text_area("Renewal")
extracted_text_manual = st.text_area("Extracted Text")

if st.button("Save Manual Entry"):
    save_to_dataframe(
        doc_name_manual,
        extracted_text_manual,
        issuer=issuer_manual,
        beneficiary=beneficiary_manual,
        amount=amount_manual,
        expiration_date=expiration_date_manual,
        contract_number=contract_number_manual,
        project_name=project_name_manual,
        purpose=purpose_manual,
        cancellation=cancellation_manual,
        renewal=renewal_manual
    )
    st.success(f"Manual entry for '{doc_name_manual}' added.")

# Delete specific document
st.subheader("Delete Specific Document Metadata")

doc_names = df_metadata["Document Name"].unique().tolist()
doc_to_delete = st.selectbox("Select a document to delete", doc_names)

if st.button("Delete Selected Document"):
    delete_specific_row(doc_to_delete)
