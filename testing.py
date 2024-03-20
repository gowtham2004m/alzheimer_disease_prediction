from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import os

app = Flask(__name__)

# Load the trained model
model = load_model('alzheimer_model.h5')

# Function to preprocess input images
def preprocess_input_data(image_path, target_size=(150, 150)):
    img = image.load_img(image_path, target_size=target_size, grayscale=True)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect('/')
    
    file = request.files['image']
    if file.filename == '':
        return redirect('/')
    
    # Save the uploaded image to a temporary file
    temp_image_path = 'temp_image.jpg'
    file.save(temp_image_path)
    
    img_array = preprocess_input_data(temp_image_path)
    
    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = "Demented" if predictions[0][0] > 0.5 else "Non-Demented"
    
    # Encode uploaded image to base64
    with open(temp_image_path, 'rb') as img_file:
        uploaded_image = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Remove the temporary image file
    os.remove(temp_image_path)
    
    return render_template('result.html', uploaded_image=uploaded_image, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)


