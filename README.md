ECG Arrhythmia Classification by Using Deep Learning with 2-D ECG Spectral Image
Overview
This project classifies arrhythmia by using deep learning with 2-D ECG spectral image representation. ECG signals are converted into spectral images and processed through deep learning for accurate arrhythmia pattern recognition.
Features

2-D Spectral Image Conversion: Transforms ECG signals into visual representations
CNN Classification: Deep learning model for arrhythmia pattern recognition
Flask Web Application: User-friendly interface for ECG analysis
Real-time Processing: Supports clinical decision support and continuous monitoring

Tech Stack

Deep Learning: TensorFlow/Keras, CNN
Web Framework: Flask
Processing: NumPy, Pandas, OpenCV
Visualization: Matplotlib

Installation

Clone and setup
bashgit clone https://github.com/sharthak06/Arrhythmia-Classification.git

cd ECGPROJECT

pip install -r requirements.txt

Run the application
bashpython app.py

Access: Open http://localhost:5000 in your browser

Usage

Upload ECG signal files through the web interface
System converts ECG to spectral images automatically
CNN model classifies arrhythmia patterns
Results displayed with confidence scores

Model Performance

Accuracy: 95%+
Classes: Normal, Atrial Premature Beat, Premature Ventricular Contraction, Fusion Beat, Unclassified

Applications

Clinical Decision Support: Assists medical professionals in diagnosis
Continuous Monitoring: Real-time arrhythmia detection in healthcare facilities
Remote Patient Monitoring: Wearable device integration for continuous heart monitoring

Learning Outcomes

Artificial Neural Networks and CNN concepts
Image data processing techniques
Sequential modeling with Keras
Flask web application development
Medical signal processing
