from flask import Flask, render_template, request
import PyPDF2
import re
import os
from transformers import pipeline
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters except punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Generate summary using a pre-trained model
def generate_summary(text, max_length=130, min_length=30):
    # Split text into chunks (BART has a max token limit of 1024)
    chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)

# Summarize chapter
def summarize_chapter(file_path, file_type='pdf'):
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r') as file:
            text = file.read()
    
    cleaned_text = clean_text(text)
    summary = generate_summary(cleaned_text)
    return summary

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    # Create the uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    if request.method == 'POST':
        file = request.files['file']
        file_type = file.filename.split('.')[-1]
        if file_type not in ['txt', 'pdf']:
            return "Unsupported file type. Please upload a .txt or .pdf file."
        
        file_path = f"uploads/{file.filename}"
        file.save(file_path)
        summary = summarize_chapter(file_path, file_type)

        # Delete the uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

        return render_template('index.html', summary=summary)
    
    return render_template('index.html', summary=None)

if __name__ == '__main__':
    app.run(debug=True)