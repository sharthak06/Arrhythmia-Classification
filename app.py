# Flask ECG Arrhythmia Classification App

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random string

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and configuration
model = None
class_indices = None
config = None
class_names = None


def load_model_and_config():
    """Load the trained model and configuration files"""
    global model, class_indices, config, class_names

    try:
        # Load model
        print("Loading model...")
        model = tf.keras.models.load_model('ecg_arrhythmia_model.h5')
        print("Model loaded successfully!")

        # Load class indices
        print("Loading class indices...")
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)

        # Create reverse mapping (index to class name)
        class_names = {v: k for k, v in class_indices.items()}
        print(f"Classes loaded: {list(class_indices.keys())}")

        # Load model configuration
        print("Loading model configuration...")
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        print(f"Model configuration loaded: {config}")

        return True

    except Exception as e:
        print(f"Error loading model or configuration: {str(e)}")
        return False


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Get image size from config
        img_size = tuple(config['IMG_SIZE'])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")

        # Resize image
        image = cv2.resize(image, img_size)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None


def predict_arrhythmia(image_path):
    """Make prediction on ECG image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None, None, None

        # Make prediction
        predictions = model.predict(processed_image)

        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        # Get all class probabilities
        all_predictions = {}
        for idx, prob in enumerate(predictions[0]):
            class_name = class_names[idx]
            all_predictions[class_name] = float(prob)

        return predicted_class, confidence, all_predictions

    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None, None


def create_prediction_chart(all_predictions):
    """Create a bar chart of all predictions"""
    try:
        # Create figure
        plt.figure(figsize=(12, 8))

        # Sort predictions by probability
        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        classes = [item[0] for item in sorted_predictions]
        probabilities = [item[1] for item in sorted_predictions]

        # Create bar chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        bars = plt.bar(range(len(classes)), probabilities, color=colors)

        # Customize chart
        plt.xlabel('Arrhythmia Types', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('ECG Arrhythmia Classification Results', fontsize=14, fontweight='bold')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{prob:.2%}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # Save to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return chart_data

    except Exception as e:
        print(f"Error creating chart: {str(e)}")
        return None


# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Upload and analyze ECG image"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # Check file and process
        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Make prediction
                predicted_class, confidence, all_predictions = predict_arrhythmia(file_path)

                if predicted_class is None:
                    flash('Error processing image. Please try again.')
                    return redirect(request.url)

                # Create prediction chart
                chart_data = create_prediction_chart(all_predictions)

                # Clean up uploaded file
                os.remove(file_path)

                # Prepare results
                results = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_predictions': all_predictions,
                    'chart_data': chart_data
                }

                return render_template('results.html', results=results)

            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)

    return render_template('upload.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            # Save file temporarily
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Make prediction
            predicted_class, confidence, all_predictions = predict_arrhythmia(file_path)

            # Clean up
            os.remove(file_path)

            if predicted_class is None:
                return jsonify({'error': 'Error processing image'}), 500

            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(class_indices) if class_indices else 0
    })


if __name__ == '__main__':
    print("Starting ECG Arrhythmia Classification App...")
    print("=" * 50)

    # Load model and configuration
    if load_model_and_config():
        print("✅ Model and configuration loaded successfully!")
        print(f"✅ Ready to classify {len(class_indices)} types of arrhythmia")
        print("✅ Starting Flask server...")
        print("=" * 50)

        # Run the app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model or configuration files!")
        print("❌ Please ensure all required files are present:")
        print("   - ecg_arrhythmia_model.h5")
        print("   - class_indices.json")
        print("   - model_config.json")