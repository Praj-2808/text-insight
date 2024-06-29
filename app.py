from flask import Flask, request, render_template, url_for, send_file, make_response, jsonify, session
from wordcloud import WordCloud, STOPWORDS
import re
import io
import base64
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import wikipediaapi
import pandas as pd
from transformers import pipeline
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from happytransformer import HappyTextToText, TTSettings
from flask_caching import Cache
import os
import JamSpell



app = Flask(__name__)
app.secret_key = 'Prajakta_Mishra'  


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_words)

def generate_wordcloud(text):
    cleaned_text = preprocess_text(text)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, colormap='viridis', max_words=200).generate(cleaned_text)
    img_buffer = io.BytesIO()
    wordcloud.to_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)  # Rewind the buffer
    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return img_str

@app.route('/word-cloud', methods=['GET', 'POST'])
def word_cloud_page():
    if request.method == 'POST':
        text = request.form['text']
        img_str = generate_wordcloud(text)
        return render_template('word_cloud_result.html', result=img_str, text=text)
    return render_template('word_cloud.html')

@app.route('/download-wordcloud', methods=['POST'])
def download_wordcloud():
    img_str = request.form['img_str']
    img_data = base64.b64decode(img_str)
    response = make_response(img_data)
    response.headers['Content-Type'] = 'image/png'
    response.headers['Content-Disposition'] = 'attachment; filename=wordcloud.png'
    return response

def categorize_sentiment(polarity):
    if polarity > 0.5:
        return "Very Positive"
    elif polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    elif polarity > -0.5:
        return "Negative"
    else:
        return "Very Negative"

@app.route('/sentiment-analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            try:
                data = pd.read_csv(file)
            except pd.errors.EmptyDataError:
                return render_template('sentiment_analysis.html', error="The uploaded file is empty. Please upload a valid CSV file.")
            except UnicodeDecodeError:
                try:
                    data = pd.read_csv(file, encoding='latin1')
                except UnicodeDecodeError:
                    return render_template('sentiment_analysis.html', error="Could not decode the file. Please ensure it is encoded in UTF-8 or Latin-1.")
            except Exception as e:
                return render_template('sentiment_analysis.html', error=f"An error occurred while reading the file: {str(e)}")

            if 'text' not in data.columns:
                return render_template('sentiment_analysis.html', error="The CSV file must contain a 'text' column.")

            data['Polarity'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
            data['Subjectivity'] = data['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
            data['Sentiment'] = data['Polarity'].apply(categorize_sentiment)

            # Save the processed DataFrame as a CSV file in a temporary location
            temp_csv_path = 'static/sentiment_analysis_result.csv'
            data.to_csv(temp_csv_path, index=False)

            data_html = data.head(10).to_html(classes='data', header="true")
            return render_template('sentiment_analysis_result.html', tables=[data_html], titles=data.columns.values, csv_path=temp_csv_path, choice="file")

        if 'text' in request.form and request.form['text'] != '':
            text = request.form['text']
            polarity = TextBlob(text).sentiment.polarity
            subjectivity = TextBlob(text).sentiment.subjectivity
            sentiment = categorize_sentiment(polarity)
            return render_template('sentiment_analysis_result.html', text=text, polarity=polarity, subjectivity=subjectivity, sentiment=sentiment, choice="text")
    
    return render_template('sentiment_analysis.html')

@app.route('/download', methods=['GET'])
def download_file():
    csv_path = request.args.get('csv_path')
    if csv_path:
        return send_file(csv_path, mimetype='text/csv', download_name='sentiment_analysis.csv', as_attachment=True)
    return render_template('error.html', error="No data to download.")

def spell_checker(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

@app.route('/spell-checker', methods=['GET', 'POST'])
def spell_checker_page():
    if request.method == 'POST':
        text = request.form['text']
        result = spell_checker(text)
        return render_template('spell_checker_result.html', input_text=text, corrected_text=result)
    return render_template('spell_checker.html')


# Load pre-trained model and tokenizer for abstractive summarization
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
abstractive_summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Function for extractive summarization using sumy
def extractive_summarizer(text, method='lsa'):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    if method == 'lsa':
        summarizer = LsaSummarizer()
    elif method == 'text_rank':
        summarizer = TextRankSummarizer()
    else:
        raise ValueError("Invalid summarization method")

    summary = summarizer(parser.document, sentences_count=3)  # Adjust sentences_count as needed
    return ' '.join([str(sentence) for sentence in summary])


@app.route('/summarizer', methods=['GET', 'POST'])
def summarizer_page():
    if request.method == 'POST':
        text = request.form['text']
        summarization_method = request.form['summarization_method']
        max_ratio = float(request.form.get('max_ratio', 0.2))  # Default max_ratio is 0.2

        if summarization_method == 'extractive':
            result = extractive_summarizer(text)
            return render_template('extractive_summarizer_result.html', text=text, result=result)
        
        elif summarization_method == 'abstractive':
            # Determine max_length based on input text length and max_ratio
            max_length = int(len(tokenizer.encode(text)) * max_ratio)
            result = abstractive_summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            summary = result[0]['summary_text']
            return render_template('abstractive_summarizer_result.html', text=text, summary=summary)
        
        else:
            return "Invalid summarization method selected"

    return render_template('summarizer.html')

    
    
def wikipedia_summarizer(query, section_title=None, num_sentences=3):
    user_agent= "Chrome/126.0.0.0(prajaktamishra16@gmail.com)"
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=user_agent
    )
    page = wiki_wiki.page(query)
    
    if not page.exists():
        return "Page not found."
    
    content = ""
    if section_title:
        section = next((sec for sec in page.sections if sec.title.lower() == section_title.lower()), None)
        if section:
            content = section.text
        else:
            return f"Section '{section_title}' not found in the page."
    else:
        content = page.text

    # Summarize the content using sumy
    parser = PlaintextParser.from_string(content, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, num_sentences)
    
    summary = ' '.join(str(sentence) for sentence in summary_sentences)
    return summary

@app.route('/wikipedia-summarizer', methods=['GET', 'POST'])
def wikipedia_summarizer_page():
    if request.method == 'POST':
        query = request.form['query']
        section_title = request.form.get('section_title')
        num_sentences = int(request.form.get('num_sentences', 3))
        result = wikipedia_summarizer(query, section_title, num_sentences)
        return render_template('wikipedia_summarizer_result.html', result=result, query=query, section_title=section_title, num_sentences=num_sentences)
    
    return render_template('wikipedia_summarizer.html')


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
