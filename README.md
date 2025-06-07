# Text Insight
Text Insight is a Flask-based web application that provides a suite of text analysis tools including spell checking, Wikipedia summarization, word cloud generation, keyword extraction, text similarity, topic modeling, language detection, and translation.

## Features

Spell Checker: Corrects spelling mistakes in input text.

Wikipedia Summarizer: Summarizes any Wikipedia article or section.

Word Cloud Generator: Visualizes important words in text.

Keyword Extraction: Extracts top keywords using TF-IDF.

Text Similarity: Calculates similarity score between two texts.

Topic Modeling: Identifies main topics from multiple documents using LDA.

Language Detection: Detects the language of input text.

Translation: Translates text to a target language.

## Installation

### Clone the repository:
git clone https://github.com/Praj-2808/text-insight.git
cd text-insight

### Create a virtual environment:
python -m venv venv
source venv/bin/activate (On Windows: venv\Scripts\activate)

### Install required packages:
pip install -r requirements.txt

### Run the Flask application:
python app.py

### Open your web browser and go to:
http://127.0.0.1:5000/

Navigate through the different features via the homepage links.

## Project Structure

app.py # Main Flask application with routes

templates/ # HTML templates for pages and results

static/ # CSS, JS, images (if any)

requirements.txt # Python dependencies

README.md # This file

## Dependencies

Flask

scikit-learn

nltk

langdetect

googletrans

wordcloud

matplotlib

wikipedia

transformers

torch

happytransformer

(For exact versions, see requirements.txt)

## Notes

For Wikipedia summarization, an internet connection is required.

Language detection and translation rely on third-party libraries which may have rate limits.

Word cloud generation saves an image file temporarily to display.

Some features require additional setup (e.g., PyTorch for transformer models).

## License

This project is open-source and free to use.

