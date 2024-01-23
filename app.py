# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import cv2
import numpy as np
import urllib.request
import base64
import json
from cellsize import cell_size
from nucleussize import process_nucleus_image

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the app


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/api/cell_size', methods=['POST'])
def cell_size_route():
    try:
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})

        # Read image file
        image_bytes = image_file.read()

        return cell_size(image_bytes)

    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/api/nucleus_size', methods=['POST'])
def nucleus_size():
    try:
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})

        # Read image file
        image_bytes = image_file.read()

        return process_nucleus_image(image_bytes)


    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
