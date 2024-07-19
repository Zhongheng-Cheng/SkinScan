from flask import Flask, render_template, request, jsonify
import os, shutil
# from llm import generate_img_response, generate_query_engine, generate_text_response
from multimodal_gemini import process_file, prompt_diagnose, generate_response
from markdown import markdown

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
    return jsonify({'message': markdown(response)})

@app.route('/upload_media', methods=['POST'])
def upload_media():
    if 'media' not in request.files:
        return jsonify({'error': 'No media part'}), 400
    file = request.files['media']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        app.config["UPLOADED_FILE_PATH"] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(app.config["UPLOADED_FILE_PATH"])
        return jsonify({'url': app.config["UPLOADED_FILE_PATH"]})

@app.route('/media_analyze', methods=['POST'])
def media_analyze():
    response = process_file(prompt_diagnose, app.config["UPLOADED_FILE_PATH"])
    diagnose = "# Diagnose\n\n" + "\n\n".join([f"## {key.replace('_', ' ')}\n\n{value}" for key, value in response.items()])
    return jsonify({'message': markdown(diagnose)})

if __name__ == '__main__':
    app.run(debug=True)
