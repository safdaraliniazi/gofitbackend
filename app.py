from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('mobilenet_v2.h5')

def process_image(image):
    """Preprocess the image to match the MobileNetV2 input requirements."""
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert image to RGB if necessary
    
    image = image.resize((224, 224))  # Resize to 224x224
    image = np.array(image) / 255.0   # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = process_image(image)
        
        # Make prediction using the model
        prediction = model.predict(processed_image)
        
        # Convert prediction to list for JSON response
        response = {
            'prediction': prediction.tolist()
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200

if __name__ == '__main__':
    app.run(debug=True)
