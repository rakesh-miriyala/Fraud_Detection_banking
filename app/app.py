import os
import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, send_file,
    session, redirect, url_for, flash
)
from src.predict import predict_transaction
from datetime import timedelta

app = Flask(__name__, template_folder='templates', static_folder='static')

app.secret_key = 'your-secret-key'  # Use a secure key in production
app.permanent_session_lifetime = timedelta(minutes=30)

# Dummy users (use hashed passwords & DB in production)
users = {
    "rakesh": "4754"
}

# Load model and scaler
MODEL_PATH = os.path.join('model', 'fraud_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session.permanent = True
            session['user'] = username
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
    important_cols = ['V4', 'V11', 'V14', 'V17', 'V12', 'V10', 'V16', 'Amount']
    full_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    default_values = pd.read_csv('data/creditcard.csv')[full_cols].mean()

    if request.method == 'POST':
        try:
            input_data = request.form
            values = default_values.copy()
            for col in important_cols:
                values[col] = float(input_data.get(col))
            df_input = pd.DataFrame([values])
            df_input['Amount'] = scaler.transform(df_input['Amount'].values.reshape(-1, 1))
            pred = model.predict(df_input)[0]
            prediction = "Fraudulent" if pred == 1 else "Legitimate"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction, important_columns=important_cols)

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']
        if file.filename == '':
            return 'Empty filename', 400

        try:
            df = pd.read_csv(file)
            required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
            if not all(col in df.columns for col in required_cols):
                return f'Missing columns. Expected: {required_cols}', 400

            X = df[required_cols]
            X['Amount'] = scaler.transform(X['Amount'].values.reshape(-1, 1))
            df['Prediction'] = model.predict(X)

            table_html = df.head(50).to_html(classes='table table-bordered', index=False)
            return render_template('batch_results.html', table_html=table_html)

        except Exception as e:
            return f"Prediction error: {e}", 500

    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
