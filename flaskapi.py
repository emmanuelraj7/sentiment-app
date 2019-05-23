import pickle
from flask import Flask,request
import numpy as numpy

with open('/Users/jnandikonda/Desktop/Arcada/ASP_Final_Project/sentiment_model.pkl','rb') as model_file:
	model = pickle.load(model_file)

app = Flask(__name__)
@app.route('/predict')
def predict():
	prediction = model.predict(np.array[[]])

	return str(prediction)

if __name__ == '__main__':
	app.run()