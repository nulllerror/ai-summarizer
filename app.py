import streamlit as st
from transformers import pipeline
from pypdf import PdfReader


summarizer = None


@st.cache_resource
def load_summarizer(token=None):
    return pipeline("summarization", model="facebook/bart-large-cnn", token=token)

def generate_summary(article, token=None):

    # Load summarizer using cached function
    summarizer = load_summarizer(token)

    chunk_size = 500

    chunks = [article[i:i + chunk_size] for i in range(0, len(article), chunk_size)]

   
    summary_texts = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary_texts.append(summary[0]['summary_text'])

    
    final_summary = " ".join(summary_texts)
    return final_summary

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

st.title("Text Summarizer App")

st.sidebar.header("Settings")
hf_token = st.sidebar.text_input("Hugging Face Token (Optional)", type="password")

input_type = st.radio("Input type:", ["Paste text", "Upload PDF"])

article_input = ""

if input_type == "Paste text":
    article_input = st.text_area("Enter the article:")
else:
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf_file is not None:
        article_input = extract_text_from_pdf(pdf_file)


if st.button("Generate Summary"):
    if article_input:
        summary_result = generate_summary(article_input, token=hf_token)
        st.subheader("Summary:")
        st.write(summary_result)
    else:
        st.warning("Please enter an article to summarize.")