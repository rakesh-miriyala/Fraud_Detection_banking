import os
import shutil
import joblib
import pandas as pd
from datetime import timedelta, datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

from blockchain import Blockchain

# Flask setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your-secret-key'
app.permanent_session_lifetime = timedelta(minutes=30)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Blockchain setup
fraud_chain = Blockchain()

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

with app.app_context():
    db.create_all()

# Load ML model and scaler
MODEL_PATH = os.path.join('model', 'fraud_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Choose another.', 'danger')
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash('Signup successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session.permanent = True
            session['user'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    full_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
    default_values = pd.read_csv('data/creditcard.csv')[full_columns].mean()

    if request.method == 'POST':
        try:
            input_data = request.form
            values = default_values.copy()
            for col in full_columns:
                values[col] = float(input_data.get(col))

            df_input = pd.DataFrame([values])
            df_input['Amount'] = scaler.transform(df_input['Amount'].values.reshape(-1, 1))
            pred = model.predict(df_input)[0]
            prediction = "Fraudulent" if pred == 1 else "Legitimate"

            log_entry = {
                "username": session.get("user"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": prediction
            }
            for col in full_columns:
                log_entry[col] = values[col]

            os.makedirs('logs', exist_ok=True)
            pd.DataFrame([log_entry]).to_csv(
                'logs/prediction_logs.csv',
                mode='a', index=False,
                header=not os.path.exists('logs/prediction_logs.csv')
            )

            fraud_chain.add_block(log_entry)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction, full_columns=full_columns)

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return 'No file uploaded.', 400

        try:
            df = pd.read_csv(file)
            required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
            if not all(col in df.columns for col in required_cols):
                return f'Missing columns: {required_cols}', 400

            X = df[required_cols].copy()
            X['Amount'] = scaler.transform(X['Amount'].values.reshape(-1, 1))
            df['Prediction'] = model.predict(X)

            # Filter only fraud transactions
            fraud_df = df[df['Prediction'] == 1]

            os.makedirs('logs', exist_ok=True)
            output_file = 'logs/batch_output.csv'
            fraud_df.to_csv(output_file, index=False)
            static_path = os.path.join('static')
            os.makedirs(static_path, exist_ok=True)
            shutil.copy(output_file, os.path.join(static_path, 'batch_output.csv'))

            if fraud_df.empty:
                table_html = "<div class='alert alert-success'>No fraudulent transactions detected 🎉</div>"
            else:
                table_html = fraud_df.to_html(classes='table table-bordered', index=False)

            return render_template('batch_results.html', table_html=table_html)

        except Exception as e:
            return f"Prediction error: {e}", 500

    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/blockchain')
def view_blockchain():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('blockchain.html', chain=[block.to_dict() for block in fraud_chain.chain])

if __name__ == '__main__':
    app.run(debug=True)
