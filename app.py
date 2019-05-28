import pickle
from flask import Flask, render_template, request
from flasgger import Swagger
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
swagger = Swagger(app)


#opening artifacts
with open('./Artefacts/sentiment_model.pkl','rb') as model_file:
	model = pickle.load(model_file)

with open('./Artefacts/vectorizer.pkl','rb') as model_file:
	sentance_vectorizer = pickle.load(model_file)


@app.route("/")
def main():
	return render_template('index.html')


@app.route('/service')
def showSignUp():
    return render_template('sentiment_service.html')


@app.route('/sentiment', methods=['POST'])
def sentiment():
	"""
	Endpoint to predict sentiment for text
	---
	parameters:
	- name: Enter text
	  in: query
	  type: string
	  description: Predict sentiment
	  required: true
	"""

	if request.method == 'POST':
		try:
			texttoanalyze = request.form['texttoanalyze']
			pred = model.predict(sentance_vectorizer.transform([texttoanalyze]))
			confidence_score = model.predict_proba(sentance_vectorizer.transform([texttoanalyze]))
			if pred == 0:
				model_output_sentiment = "Negative" 
				confidence_score = float("{0:.2f}".format(confidence_score[:,0][0]))
			else: 
				model_output_sentiment = "Positive"
				confidence_score = float("{0:.2f}".format(confidence_score[:,1][0]))	
		except:
			texttoanalyze = ""
			model_output_sentiment = ""
			confidence_score = ""
    

	return render_template('sentiment_service.html', sentiment=model_output_sentiment, score=confidence_score, texttoanalyze=texttoanalyze), model_output_sentiment




if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000, debug=True)
