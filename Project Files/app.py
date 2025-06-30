from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')


labels = [
    'Apple_Healthy', 'Apple_Rotten', 'Banana_Healthy', 'Banana_Rotten',
    'Bellpepper_Healthy', 'Bellpepper_Green', 'Carrot_Healthy', 'Carrot_Rotten',
    'Cucumber_Healthy', 'Cucumber_Rotten', 'Grape_Healthy', 'Grape_Rotten',
    'Guava_Healthy', 'Guava_Rotten', 'Jujube_Healthy', 'Jujube_Rotten',
    'Mango_Healthy', 'Mango_Rotten', 'Orange_Healthy', 'Orange_Rotten',
    'Pomegranate_Healthy', 'Pomegranate_Rotten', 'Potato_Healthy',
    'Potato_Rotten', 'Strawberry_Healthy', 'Strawberry_Rotten',
    'Tomato_Healthy', 'Tomato_Rotten'
]


try:
    model = load_model('healthy_vs_rotten.h5')
    print(f"Model loaded successfully. Output shape: {model.output_shape}")
    
   
    if model.output_shape[-1] != len(labels):
        print(f"WARNING: Model outputs {model.output_shape[-1]} classes but you have {len(labels)} labels")
        print("You need to retrain your model or use the correct model file")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)
        
     
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction_probs = model.predict(img_array)[0]
        
       
        if len(prediction_probs) != len(labels):
            return jsonify({
                'error': f'Model output length {len(prediction_probs)} does not match label count {len(labels)}. Please use the correct model file.'
            }), 500
        
        predicted_index = np.argmax(prediction_probs)
        predicted_label = labels[predicted_index]
        confidence = round(float(prediction_probs[predicted_index]) * 100, 2)
        
     
        os.remove(file_path)
        
        return jsonify({
            'predicted_class': predicted_label,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=2222)