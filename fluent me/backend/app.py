import os
import random
import io
from datetime import date, datetime
import tempfile
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from gtts import gTTS
import whisper
from jiwer import wer
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
DB_CONFIG = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME')
}
PASS_THRESHOLD = 0.80
WHISPER_MODEL = "tiny.en"
GAME_TASKS = {
    "Beginner": ["cat", "dog", "ball", "apple", "book"],
    "Intermediate": ["The brown fox jumped over the gate", "I love reading books", "Please pass the salt"],
    "Advanced": ["She sells seashells by the seashore", "How can a clam cram in a clean cream can?", "Peter Piper picked a peck of pickled peppers"]
}
WORD_PUZZLES = ["banana", "elephant", "giraffe", "computer", "python", "streamlit", "butterfly", "sunflower", "window", "mountain"]

# --- Database Connection ---
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# --- Whisper Model ---
def load_whisper_model(model_name):
    return whisper.load_model(model_name)

# Lazy load model
model_cache = {}
def get_model(model_name):
    if model_name not in model_cache:
        model_cache[model_name] = load_whisper_model(model_name)
    return model_cache[model_name]

# --- Utilities ---
def synthesize_tts(text, lang="en"):
    tts = gTTS(text, lang=lang)
    audio_stream = io.BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)
    return audio_stream

def transcribe_audio(file_path, language=None, model_name="tiny.en"):
    model = get_model(model_name)
    options = {}
    if language:
        options["language"] = language
    res = model.transcribe(file_path, **options)
    return res.get("text", "").strip()

def compute_accuracy(reference, hypothesis):
    try:
        w = wer(reference.lower(), hypothesis.lower())
        accuracy = max(0.0, 1.0 - w)
        return accuracy
    except Exception:
        return 0.0

def shuffle_word(word):
    if len(word) <= 2:
        return word
    word_list = list(word)
    random.shuffle(word_list)
    return "".join(word_list)

# --- Auth Routes ---
@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    full_name = data.get('full_name', '')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password_hash, full_name) VALUES (%s, %s, %s)",
                       (username, password_hash, full_name))
        conn.commit()
        return jsonify({"success": True, "message": "User registered successfully."}), 201
    except mysql.connector.Error as err:
        if err.errno == 1062: # Duplicate entry for unique key
            return jsonify({"success": False, "message": "Username already exists."}), 409
        return jsonify({"success": False, "message": "Registration failed."}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT id, password_hash, full_name FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        return jsonify({
            "success": True,
            "message": "Logged in successfully.",
            "user": {
                "id": user['id'],
                "username": username,
                "full_name": user['full_name']
            }
        }), 200
    else:
        return jsonify({"success": False, "message": "Invalid username or password."}), 401

# --- Practice and Game Routes ---
@app.route('/api/practice', methods=['POST'])
def process_practice_audio():
    if 'audio' not in request.files:
        return jsonify({"success": False, "message": "No audio file provided."}), 400
    
    audio_file = request.files['audio']
    ref_text = request.form.get('referenceText')
    user_id = request.form.get('userId')
    language = request.form.get('language', 'en')
    model = request.form.get('model', 'tiny.en')

    if not all([audio_file, ref_text, user_id]):
        return jsonify({"success": False, "message": "Missing required data."}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        transcript = transcribe_audio(tmp_path, language, model)
        accuracy = compute_accuracy(ref_text, transcript)
        os.unlink(tmp_path)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attempts (user_id, reference_text, transcript, accuracy, language) VALUES (%s, %s, %s, %s, %s)",
                       (user_id, ref_text, transcript, accuracy, language))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "transcript": transcript,
            "accuracy": accuracy,
            "message": "Attempt saved successfully."
        }), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/record-progress', methods=['POST'])
def record_progress_route():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO progress (user_id, level, task, passed, accuracy) VALUES (%s, %s, %s, %s, %s)",
                   (data['userId'], data['level'], data['task'], data['passed'], data['accuracy']))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"success": True})

# --- Other Data Routes ---
@app.route('/api/daily-puzzle', methods=['GET'])
def get_daily_puzzle():
    user_id = request.args.get('userId')
    today = date.today().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT puzzle_word, solved FROM puzzles WHERE user_id = %s AND assigned_date = %s", (user_id, today))
    puzzle_data = cursor.fetchone()
    
    if not puzzle_data:
        new_puzzle = random.choice(WORD_PUZZLES)
        try:
            cursor.execute("INSERT INTO puzzles (user_id, puzzle_word, assigned_date) VALUES (%s, %s, %s)",
                           (user_id, new_puzzle, today))
            conn.commit()
            puzzle_data = {'puzzle_word': new_puzzle, 'solved': False}
        except mysql.connector.Error as err:
            if err.errno == 1062: # Duplicate entry from race condition
                cursor.execute("SELECT puzzle_word, solved FROM puzzles WHERE user_id = %s AND assigned_date = %s", (user_id, today))
                puzzle_data = cursor.fetchone()
            else:
                cursor.close()
                conn.close()
                return jsonify({"success": False, "message": "Database error"}), 500
    
    cursor.close()
    conn.close()
    
    unscrambled_word = puzzle_data['puzzle_word']
    scrambled_word = shuffle_word(unscrambled_word)
    
    return jsonify({
        "success": True,
        "puzzle": scrambled_word,
        "unscrambled": unscrambled_word,
        "solved": puzzle_data['solved']
    }), 200

@app.route('/api/solve-puzzle', methods=['POST'])
def solve_puzzle_route():
    data = request.json
    user_id = data.get('userId')
    today = date.today().isoformat()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("UPDATE puzzles SET solved=1 WHERE user_id = %s AND assigned_date = %s", (user_id, today))
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({"success": True, "message": "Puzzle marked as solved."}), 200

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard_route():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT u.username, AVG(a.accuracy) as avg_acc FROM attempts a JOIN users u ON a.user_id = u.id GROUP BY u.id ORDER BY avg_acc DESC LIMIT 10")
    leaderboard = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(leaderboard), 200

@app.route('/api/progress/<int:user_id>', methods=['GET'])
def get_progress(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT accuracy, created_at FROM attempts WHERE user_id = %s ORDER BY created_at", (user_id,))
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(history), 200

@app.route('/api/stats/<int:user_id>', methods=['GET'])
def get_stats(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM attempts WHERE user_id = %s", (user_id,))
    attempts_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT accuracy FROM attempts WHERE user_id = %s ORDER BY created_at DESC LIMIT 1", (user_id,))
    latest_acc_row = cursor.fetchone()
    latest_acc = latest_acc_row[0] if latest_acc_row else 0
    
    cursor.execute('SELECT COUNT(DISTINCT DATE(created_at)) FROM attempts WHERE user_id=%s AND created_at >= CURDATE() - INTERVAL 7 DAY', (user_id,))
    streak = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return jsonify({
        "attempts_count": attempts_count,
        "latest_acc": latest_acc,
        "streak": streak
    }), 200

@app.route('/api/tts', methods=['POST'])
def tts_route():
    data = request.json
    text = data.get('text')
    lang = data.get('lang', 'en')
    
    if not text:
        return jsonify({"success": False, "message": "No text provided."}), 400
    
    audio_stream = synthesize_tts(text, lang)
    return audio_stream.getvalue(), 200, {'Content-Type': 'audio/mpeg'}

@app.route('/api/game-tasks', methods=['GET'])
def get_game_tasks():
    return jsonify(GAME_TASKS), 200

if __name__ == '__main__':
    app.run(debug=True)