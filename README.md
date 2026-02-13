# 🔐 Password Strength Checker

A machine learning-powered application that assesses password strength using an XGBoost model with **94% accuracy**. This project includes a FastAPI backend for predictions and a Streamlit frontend for an interactive user experience.

## 🚀 Features

- **Real-time Analysis**: Instant feedback on password strength (Weak, Medium, Strong).
- **Machine Learning Model**: Uses a pre-trained XGBoost classifier.
- **Batch Processing**: Upload CSV or TXT files to analyze multiple passwords at once.
- **Visualizations**: Interactive charts and probability distributions using Plotly.
- **REST API**: Fully documented API endpoints via FastAPI.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit, Plotly
- **Machine Learning**: XGBoost, Scikit-learn, Pandas, NumPy
- **Data Processing**: SQL (SQLite)

## 📂 Project Structure

- `app.py`: FastAPI application serving the ML model.
- `frontend.py`: Streamlit dashboard for user interaction.
- `NLP_password_strength.ipynb`: Jupyter notebook used for data analysis, feature engineering, and model training.
- `requirements.txt`: List of Python dependencies.
- `password_strength_XGBoost_0.9405.pkl`: Serialized XGBoost model.
- `password_data.sqlite`: Database containing password datasets.

## 📦 Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚦 Usage

The backend API is deployed on AWS at `http://54.226.206.2:8000`. The frontend is configured to use this deployed API by default.

### 1. Start the Frontend (Streamlit)
Run the following command to start the application interface:
```bash
streamlit run frontend.py
```
The application will open in your default web browser (usually at `http://localhost:8501`).

### Optional: Run Backend Locally (For Development)
If you wish to run the backend server locally instead of using the deployed AWS instance, you can start it with:

```bash
uvicorn app:app --reload
```
(Note: You will need to update `API_URL` in `frontend.py` to point to localhost)

## 🧠 Model Details

The model is trained on a dataset of passwords labeled as 0 (Weak), 1 (Medium), and 2 (Strong). Key features used for classification include:
- **TF-IDF Vectorization**: Character-level textual features.
- **Structural Features**: Length, lowercase frequency, uppercase presence, digit presence, and special character presence.

Accuracy: **94.05%**

## 📝 License

[Add your license here, e.g., MIT]
