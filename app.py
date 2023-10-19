import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os

def getPrediction(filename):
   
    dic = {0 : 'Covid', 1 : 'Healthy', 2 : 'Lung Tumor', 3 : 'Common Pneumonia'}
    
    #Load model
    my_model=load_model("chest_model_deploy.h5")
    
    SIZE = 64 #Resize to same size as training images
    img_path = 'static/'+filename
    img= Image.open(img_path).resize((SIZE,SIZE))
    img= img.convert('RGB')
    img= np.asarray(img)

    #Scale pixel values
    img = img/255.      
    
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = my_model.predict(img) #Predict                    
    pred_class = dic[np.argmax(pred)]
    
    print("Diagnosis is:", pred_class)
    return pred_class


app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

UPLOAD_FOLDER = 'static'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/static', methods=['POST', 'GET'])
def submit_file():
  if request.method == 'POST':
    if 'file' not in request.files:
      return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
      return jsonify({'message': 'No selected file'}), 400
    if file:
      filename = (file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      label = getPrediction(filename)
      return jsonify({'message': label , 'image' : "/"+filename}), 200

@app.route('/', methods=[ 'GET'])
def load():
      return 'Hello world', 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)