from flask import Flask, render_template, request, jsonify
import os
import random
import time
from model_simulator import PneumoniaModelSimulator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the simulator
model_simulator = PneumoniaModelSimulator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Simulate processing time
        time.sleep(1.5)
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Simulate AI prediction
        result = model_simulator.predict(file.filename)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_sample', methods=['POST'])
def analyze_sample():
    """Analyze pre-loaded sample images"""
    try:
        data = request.get_json()
        sample_type = data.get('sample_type', 'normal')
        
        # Simulate processing time
        time.sleep(1.2)
        
        # Generate realistic probabilities based on sample type
        if sample_type == 'pneumonia':
            pneumonia_prob = random.uniform(0.75, 0.95)
            normal_prob = 1 - pneumonia_prob
        elif sample_type == 'normal':
            normal_prob = random.uniform(0.80, 0.98)
            pneumonia_prob = 1 - normal_prob
        else:  # uncertain
            pneumonia_prob = random.uniform(0.4, 0.6)
            normal_prob = 1 - pneumonia_prob
        
        result = {
            'top_label': 'Pneumonia' if pneumonia_prob > normal_prob else 'Normal',
            'top_prob': max(pneumonia_prob, normal_prob),
            'probs': {
                'Pneumonia': round(pneumonia_prob, 3),
                'Normal': round(normal_prob, 3)
            },
            'confidence': 'high' if max(pneumonia_prob, normal_prob) > 0.8 else 'medium'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)



