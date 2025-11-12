import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# ----------------------------------------------------------
# Flask App Initialization
# ----------------------------------------------------------
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# ----------------------------------------------------------
# Load the trained CNN model
# ----------------------------------------------------------
MODEL_PATH = 'model/best_model.keras'
model = load_model(MODEL_PATH)

# ----------------------------------------------------------
# Folder to store uploaded images
# ----------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ----------------------------------------------------------
# Home Page
# ----------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


# ----------------------------------------------------------
# Route to Serve Uploaded Files
# ----------------------------------------------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ----------------------------------------------------------
# Prediction Route
# ----------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="Please upload an image.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")

    # Save uploaded image to uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Interpretation
    if prediction > 0.5:
        result = "♻️ This is Recyclable Waste"
    else:
        result = "🌿 This is Organic Waste"

    # Pass image path (served through Flask route)
    image_url = f"/uploads/{file.filename}"

    return render_template('index.html', prediction=result, image_path=image_url)


# ----------------------------------------------------------
# Run Flask App
# ----------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
