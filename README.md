# 🛡️ Cloud Intrusion Detection System (IDS)

A machine learning-based intrusion detection system that identifies malicious network traffic using **Random Forest** and **SVM**, with an interactive **Streamlit dashboard**.
🚀 Features
 🔍 Detects network intrusions in real-time
 🤖 Dual-model comparison (Random Forest vs SVM)
 📊 Interactive Streamlit dashboard
 ⚙️ Automated data preprocessing
 📈 Performance metrics (Accuracy, Precision, Recall, F1-score)
 
🧠 Models Used
Random Forest Classifier
Support Vector Machine (SVM)

📂 Project Structure


Cloud-Intrusion-Detection-System/
│── app.py
│── train_model.py
│── utils.py
│── dataset.csv
│── rf_model.pkl
│── svm_model.pkl
│── preprocessing.pkl
│── metrics.pkl

 ▶️ Run the Project
Step 1: Train the model
python train_model.py

Step 2: Launch dashboard
streamlit run app.py

📊 How to Use

1. Upload a CSV dataset
2. Select model (Random Forest / SVM / Compare)
3. View predictions and analysis
4. Analyze graphs and performance

 📈 Output

* Intrusion vs Normal traffic detection
* Accuracy, Precision, Recall, F1-score
* Confusion matrix visualization
* Feature importance (Random Forest)

🧩 Key Components

🔹 Preprocessing (`utils.py`)
* Handles missing values
* Encodes categorical data
* Scales features

🔹 Model Training (`train_model.py`)
* Trains Random Forest & SVM
* Evaluates models
* Saves trained models

🔹 Dashboard (`app.py`)
* Interactive UI using Streamlit
* Real-time predictions
* Data visualization with Plotly

📌 Future Improvements
* Deep learning models (CNN/LSTM)
* Cloud deployment (AWS / Render)
* Real-time network packet monitoring


