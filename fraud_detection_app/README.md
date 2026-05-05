# ShieldPay AI - Fraud Detection Website

A Flask-based fraud detection project that looks and behaves like a realistic banking demo.

## Features
- Real-time transaction checker with risk score and confidence level
- Random Forest and XGBoost model training
- Explainable AI reasons using SHAP feature contributions
- Fraud dashboard with bar, pie, and trend charts
- Transaction history stored in SQLite
- Login system so each user sees only their own data
- Alerts when fraud probability crosses a threshold
- Model performance page with accuracy, precision, recall, F1 score, and confusion matrix
- Live API endpoint at `/api/predict`

## Inputs collected from the user
- Transaction amount
- Transaction type
- Location
- Transaction mode
- Device type
- Merchant category
- Age group
- Transaction time

The backend auto-generates smart features such as:
- Number of transactions in the last 24 hours
- User average transaction amount
- Previous location distance
- Time since previous transaction
- Geo-velocity check
- New-location flag

## Project structure
- `app.py` - Flask app and routes
- `train_model.py` - trains Random Forest and XGBoost models
- `model_service.py` - prediction, feature engineering, SHAP explanations
- `db.py` - SQLite tables and CRUD helpers
- `templates/` - website HTML pages
- `static/` - CSS, JavaScript, and confusion matrix images
- `models/` - saved model bundle and metrics JSON
- `data/` - SQLite database

## How to run
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open the local URL shown by Flask in your browser.

## Demo login
- Username: `demo`
- Password: `demo123`

## Important note
This project uses a synthetic training dataset generated from realistic fraud rules. It is excellent for demos, viva, and project submissions. For production-grade accuracy, you should retrain the models on a real labeled banking transaction dataset.
