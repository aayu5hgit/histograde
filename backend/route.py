# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
import pickle
from extract_features import extract_features
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
	return 'Hello World'

@app.route('/predict', methods=['GET','POST'])
@cross_origin()
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Call the feature_predict function to extract features from the image
    features = extract_features(file)
    
    loaded_model = pickle.load(open('./model.sav', 'rb'))
    result = loaded_model.predict([[
        features['Mean Intensity'],
        features['Mean Size'],
        features['Cytoplasmic ratio'],
        features['Lower'],
        features['Mid'],
        features['Upper'],
        ]])
    print(result[0])
    labels = ["Mild", "Moderate", "Severe"]
    # Do something with the features, e.g. return them as JSON
    return jsonify({
        'prediction': labels[result[0]],
        'features': features
    })

# main driver function
if __name__ == '__main__':
	# run() method of Flask class runs the application
	# on the local development server.
	app.run(host='0.0.0.0', debug=True, port=9000)
