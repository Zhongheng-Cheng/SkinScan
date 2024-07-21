from flask import Flask, render_template, request, jsonify
import os, shutil
from multimodal_gemini import DermatologistBot
from markdown import markdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/tmp_uploads'

@app.route('/')
def home():
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'])
    app.config["BOT"] = DermatologistBot()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    response = app.config["BOT"].generate_response(user_message)
    return jsonify({'message': markdown(response)}), 200

@app.route('/transcript', methods=['POST'])
def transcript():
    if 'audio' not in request.files:
        return "No audio file in request", 400
    
    audio_file = request.files['audio']
    transcript = app.config["BOT"].get_transcript(audio_file.content_type, audio_file.read())
    return jsonify({'transcript': transcript}), 200

@app.route('/upload_media', methods=['POST'])
def upload_media():
    if 'media' not in request.files:
        return jsonify({'error': 'No media part'}), 400
    file = request.files['media']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        file_address = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_address)
        return jsonify({'url': file_address}), 200

@app.route('/media_analyze', methods=['POST'])
def media_analyze():
    response = app.config["BOT"].process_file(request.form['message'])
    diagnose = "# Diagnose\n\n" + "\n\n".join([f"## {key.replace('_', ' ')}\n\n{value}" for key, value in response.items()])
    return jsonify({'message': markdown(diagnose)})

if __name__ == '__main__':
    app.run(debug=True)
