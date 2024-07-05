from flask import Flask, render_template, request, jsonify
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    response = generate_response(user_message)
    return jsonify({'message': response})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
    image_data = data['image']
    return jsonify({'url': image_data})

def generate_response(user_message):
    return f"You said: {user_message}"

if __name__ == '__main__':
    app.run(debug=True)
