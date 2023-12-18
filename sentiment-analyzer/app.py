from flask import Flask,flash, render_template, request,redirect,send_file
import joblib
import nltk
from nltk import word_tokenize
nltk.download('punkt')
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]
from sentiment_analysis_project import data_cleaning
classifier = joblib.load(open('LogesticRegression-2.joblib', 'rb'))
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    output=""
    # for text    
    if request.form['message']:
          message = request.form['message']
          data = message
          if data:
            print(type(data))
            preprocessed_data = data_cleaning(data)
            prediction = classifier.predict(preprocessed_data)
            print(prediction)
            # print(prediction)
            
            if prediction[0]=='0':
                output="Negative Review" 
            elif prediction[0]=='1':
                output= "Positive Review"


    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == '__main__':
	app.run(debug=True)
