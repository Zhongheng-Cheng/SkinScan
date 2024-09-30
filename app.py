from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os, shutil
from multimodal_gemini import DermatologistBot
from markdown import markdown
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/tmp_uploads'
app.config['SECRET_KEY'] = 'SkinScan_secret_key'  # Change this to a random secret key

# Initialize PostgreSQL connection
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="skin_scan_db",
        user="healthiai",
        password="healthiai"
    )
    return conn

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    global bot
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'])
    bot = DermatologistBot()
    return render_template('index.html', username=session['username'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            return "Username already exists"
        
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        cur.close()
        conn.close()

        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT username, password FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user_message = request.form['message']
    response = bot.generate_response(user_message)
    return jsonify({'message': markdown(response)}), 200

@app.route('/transcript', methods=['POST'])
def transcript():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if 'audio' not in request.files:
        return "No audio file in request", 400
    
    audio_file = request.files['audio']
    transcript = bot.get_transcript(audio_file.content_type, audio_file.read())
    return jsonify({'transcript': transcript}), 200

@app.route('/upload_media', methods=['POST'])
def upload_media():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
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
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    response = bot.process_file(request.form['message'])
    diagnose = "# Diagnose\n\n" + "\n\n".join([f"## {key.replace('_', ' ')}\n\n{value}" for key, value in response.items()])
    return jsonify({'message': markdown(diagnose)})

@app.route("/get-recommand-question", methods=["GET"])
def get_recommand_question():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    response = bot.recommand_question()
    return jsonify({'recommand_question': response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

# CREATE TABLE
# CREATE TABLE users (
#     id SERIAL PRIMARY KEY,
#     username VARCHAR(255) UNIQUE NOT NULL,
#     password VARCHAR(255) NOT NULL
# );
