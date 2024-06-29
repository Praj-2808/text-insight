# Ensure the stopwords and VADER lexicon are downloaded
#import nltk
#nltk.download('stopwords')
#nltk.download('vader_lexicon')'

#pip3 install gensim==3.8.2

#from transformers import BartForConditionalGeneration, BartTokenizer

#model_name = 'facebook/bart-large-cnn'
#tokenizer = BartTokenizer.from_pretrained(model_name)
#model = BartForConditionalGeneration.from_pretrained(model_name)

#def spell_checker(text):
    #blob = TextBlob(text)
    #return corrected_text
#@app.route('/spell-checker', methods=['GET', 'POST'])
#def spell_checker_page():
    #if request.method == 'POST':
        #text = request.form['text']
        #result = spell_checker(text)
        #return render_template('spell_checker_result.html', result=result)
    #return render_template('spell_checker.html')
#cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Set cache timeout (in seconds)
#CACHE_TIMEOUT = 3600  # Cache for 1 hour (adjust as needed)

#MODEL_NAME = "prithivida/grammar_error_correcter_v1"

# Initialize HappyTextToText with your model name
#happy_tt = HappyTextToText("T5", MODEL_NAME)

# Define default settings for text generation
#default_settings = TTSettings(do_sample=True, top_k=10, temperature=0.5, min_length=1, max_length=100)
#@app.route('/spell-checker', methods=['GET', 'POST'])
#@cache.cached(timeout=CACHE_TIMEOUT, key_prefix='correct_text')
    
#generated text from result
#corrected_text = result.text
      #  return render_template('spell_checker_result.html', input_text=text, corrected_text=corrected_text)
    #return render_template('spell_checker.html')
