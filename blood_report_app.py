import streamlit as st
st.set_page_config(page_title="Blood Report Analyzer", layout="wide")

# Then import all other modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import json
import argparse
import sys
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Global variables to track dependency availability
TESSERACT_AVAILABLE = False
PDF_IMAGE_AVAILABLE = False 
TRANSFORMERS_AVAILABLE = False
summarizer = None

def setup_dependencies():
    """Set up dependencies and return availability flags"""
    global TESSERACT_AVAILABLE, PDF_IMAGE_AVAILABLE, TRANSFORMERS_AVAILABLE, summarizer

    # Download necessary NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')

    # Optional imports with fallbacks
    try:
        import pytesseract
        TESSERACT_AVAILABLE = True
    except ImportError:
        TESSERACT_AVAILABLE = False

    try:
        import pdf2image
        PDF_IMAGE_AVAILABLE = True
    except ImportError:
        PDF_IMAGE_AVAILABLE = False

    # Try to import transformers, with proper error handling
    try:
        from transformers import pipeline
        # Explicitly specify model to avoid defaults and use CPU to avoid meta tensor issues
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device="cpu")
        TRANSFORMERS_AVAILABLE = True
    except (ImportError, RuntimeError, Exception) as e:
        st.error(f"Transformers error: {str(e)}")
        TRANSFORMERS_AVAILABLE = False

# The rest of the functions from blood_report_og.py
# (Only main() and imports are modified to guarantee set_page_config is first)

# Replace spaCy with NLTK for basic NLP tasks
def simple_nlp_processing(text):
    """Basic NLP processing using NLTK instead of spaCy"""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Basic POS tagging
    pos_tags = nltk.pos_tag(filtered_words)
    
    # Extract medical terms and numbers
    medical_terms = []
    numbers = []
    
    for word, tag in pos_tags:
        if tag.startswith('N') and len(word) > 3:  # Nouns that are longer than 3 chars
            medical_terms.append(word)
        if tag == 'CD':  # Cardinal numbers
            try:
                numbers.append(float(word))
            except ValueError:
                pass
    
    return {
        'sentences': sentences,
        'medical_terms': medical_terms,
        'numbers': numbers
    }

# Copy all other functions from blood_report_og.py here...
# (For brevity, we'll focus on just showing the key structure changes)

def main():
    """Main Streamlit app function"""
    # Call setup after st.set_page_config but before any other Streamlit commands
    setup_dependencies()
    
    # Display warnings about missing dependencies
    if not TESSERACT_AVAILABLE:
        st.warning("pytesseract not available. OCR functionality will be limited.")
    if not PDF_IMAGE_AVAILABLE:
        st.warning("pdf2image not available. PDF processing will be disabled.")
    if not TRANSFORMERS_AVAILABLE:
        st.warning("Advanced summarization disabled. Using basic summarization instead.")
    
    st.title("Blood Report Analyzer and Summarizer")
    st.write("Upload your blood test report (PDF or image) to get a comprehensive summary and analysis")
    
    uploaded_file = st.file_uploader("Choose a blood report file", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Add your file processing logic here from blood_report_og.py main() function
        st.write("This file is now being processed...")
        
        # Example placeholder output
        st.info("In the complete app, this is where your blood report analysis would appear.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pass # main_cli() - implement if needed
    else:
        main()  # Run the Streamlit app 