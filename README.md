# 🛡️ Fraud Detection in Banking using Machine Learning & Blockchain

This project detects fraudulent credit card transactions using a Machine Learning model and securely logs the predictions on a Blockchain to ensure tamper-proof audit trails.

## 🚀 Features

- 🔐 User authentication system (signup/login/logout)
- 🧠 Predicts transaction fraud using a trained ML model (Random Forest / etc.)
- 📂 Supports both individual and batch predictions
- 🔗 Blockchain-based logging of each prediction
- 📊 Dashboard to visualize results
- 🗃️ Secure user data management with SQLite

## 📁 Project Structure
![image](https://github.com/user-attachments/assets/c3f96f07-36e1-4d91-923c-54b78f927007)


## 🛠️ Setup Instructions

1. Clone the repository:
git clone https://github.com/rakesh-miriyala/Fraud_Detection_banking.git
cd Fraud_Detection_banking

Create a virtual environment and activate it:
python -m venv .venv
.venv\Scripts\activate  # Windows

Install dependencies:
pip install -r requirements.txt

Run the application:
python run.py
Then open your browser and visit: http://127.0.0.1:5000

🔗 Blockchain
Every individual prediction is added as a block to a blockchain.
Navigate to /blockchain to view all prediction logs with their hashes.

📷 Screenshots
![image](https://github.com/user-attachments/assets/ea3f68ad-12aa-43db-9d5e-af53f5ccbab5)
![image](https://github.com/user-attachments/assets/d21393e2-96cf-472e-9573-6535b694520b)
![image](https://github.com/user-attachments/assets/79b75b18-abc2-428b-8ad9-74c41d9a1fda)

📜 License
This project is for academic/demo purposes only.

