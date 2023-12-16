import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import io

def extract_text_from_image(uploaded_file):
    # Check if the input is a BytesIO object
    if isinstance(uploaded_file, io.BytesIO):
        # Convert BytesIO to numpy array
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        # Perform OCR on the image
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_np)
    else:
        # Perform OCR on the image path
        reader = easyocr.Reader(['en'])
        result = reader.readtext(uploaded_file)

    extracted_text = ' '.join([text[1] for text in result])
    return extracted_text

def main():
    st.title("Text Extraction from Image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the selected image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform text extraction on the uploaded image
        extracted_text = extract_text_from_image(uploaded_file)
        
        # Display the extracted text
        st.header("Extracted Text:")
        st.write(extracted_text)

if __name__ == "__main__":
    main()
