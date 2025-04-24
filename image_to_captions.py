import image_models
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import test
from pymongo import MongoClient
from bson import ObjectId
from deepface import DeepFace
import cv2




app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # If user does not select file, browser submits an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image through your functions
            try:
                image = image_models.load_image(filepath)
                
                # Extract various features
                x1 = image_models.captioning_model(image)          # Image caption
                x2 = image_models.objects_with_count(filepath)     # Object detection with counts
                x3 = image_models.color(filepath)                 # Color analysis
                x4 = image_models.analyze_face(filepath)           # Face analysis
                
                # Compile all features into a dictionary
                result = {
                    "image_path": filepath,
                    "caption": x1,
                    "objects_count": x2,
                    "objects_colors": x3,
                    "actions": x4
                }
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'error': str(e)})
    
    return render_template('image_to_captions.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007, threaded=True)