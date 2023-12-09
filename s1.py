import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
from PyPDF2 import PdfReader

st.set_page_config(layout="wide")

@st.cache_data
def text_summary(text, model_name="facebook/bart-large-cnn", maxlength=None):
    # Load pre-trained model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenize input text
    inputs = tokenizer(text, max_length=maxlength, return_tensors="pt", truncation=True)
    
    # Generate summary using model
    summary_ids = model.generate(**inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Convert summary IDs to text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary_text


def extract_text_from_pdf(file_path):
    # Open the PDF file using PyPDF2
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text





choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize Text")
    st.write("Summarize long texts with Advo")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = text_summary(input_text)
                st.success(result)

elif choice == "Summarize Document":
    st.subheader("Summarize Document")
    st.write("Summarize your docs with Advo")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1,1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Extracted Text is Below:**")
                st.info(extracted_text)
            with col2:
                st.markdown("**Summary Result**")
                text = extract_text_from_pdf("doc_file.pdf")
                doc_summary = text_summary(text)
                st.success(doc_summary)
