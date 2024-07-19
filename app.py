from flask import Flask, render_template, request, jsonify
import os, shutil
from llm import generate_img_response, generate_query_engine, generate_text_response
from multimodal_gemini import process_file, prompt_diagnose

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
    if app.config['query_engine']:
        response = generate_text_response(app.config['query_engine'], user_message)
    else:
        response = "Please upload a image to start"
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
    response = process_file(prompt_diagnose, app.config["UPLOADED_FILE_PATH"])
    response_text = "<br><br>".join([f"<b>- {key}</b>: {value}" for key, value in response.items()])
    return jsonify({'message': response_text})

if __name__ == '__main__':
    app.run(debug=True)
