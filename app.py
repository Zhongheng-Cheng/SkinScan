from flask import Flask, render_template, request, jsonify
import os, shutil
from llm import generate_img_response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/tmp_uploads'

@app.route('/')
def home():
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'])
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    response = generate_response(user_message)
    return jsonify({'message': response})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        app.config["UPLOADED_FILE_PATH"] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(app.config["UPLOADED_FILE_PATH"])
        return jsonify({'url': app.config["UPLOADED_FILE_PATH"]})

@app.route('/img_analyze', methods=['POST'])
def img_analyze():
    return jsonify({'message': generate_img_response(app.config["UPLOADED_FILE_PATH"])})

def generate_response(user_message):
    return f"You said: {user_message}"

if __name__ == '__main__':
    app.run(debug=True)
