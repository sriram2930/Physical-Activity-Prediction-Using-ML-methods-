# Physical-Activity-Prediction-Using-ML-methods-
Machine Learning-based human activity recognition system using PAMAP2 wearable sensor dataset. Features a Streamlit web app with 94.2% accuracy for real-time activity classification from IMU sensors.


# PAMAP2 Activity Recognition 🏃

A comprehensive machine learning system for human activity recognition using wearable sensor data from the PAMAP2 dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-94.2%25-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

This project implements a Random Forest classifier to recognize 12 different physical activities using data from IMU (Inertial Measurement Unit) sensors placed on the hand, chest, and ankle. The system achieves **94.2% accuracy** on test data.

### Recognized Activities
- 🛏️ Lying
- 🪑 Sitting  
- 🧍 Standing
- 🚶 Walking
- 🏃 Running
- 🚴 Cycling
- 🥾 Nordic Walking
- ⬆️ Ascending Stairs
- ⬇️ Descending Stairs
- 🧹 Vacuum Cleaning
- 👕 Ironing
- ⭐ Rope Jumping

## 🚀 Features

- **Web Interface**: Interactive Streamlit dashboard for real-time predictions
- **Multiple Models**: Random Forest, Gradient Boosting, Logistic Regression
- **Feature Engineering**: 135 engineered features from raw sensor data
- **Class Balancing**: SMOTE implementation to handle imbalanced data
- **Batch Testing**: Test model performance across multiple samples
- **Visualizations**: 30+ professional charts for data analysis

## 📊 Dataset

**Source**: PAMAP2 Physical Activity Monitoring Dataset (UCI ML Repository)

- **Subjects**: 9 participants
- **Sensors**: 3 IMU units (hand, chest, ankle)
- **Sampling Rate**: 100 Hz
- **Raw Samples**: 2.8M+ data points
- **Activities**: 12 different physical activities

## 🛠️ Tech Stack

- **ML/Data**: scikit-learn, pandas, numpy, imbalanced-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: Streamlit
- **Models**: Random Forest (primary), Gradient Boosting, Logistic Regression

## 📁 Project Structure
```
PAMAP2-Activity-Recognition/
├── pamap2_data/
│   ├── subject101.dat - subject109.dat
│   ├── pamap2_cleaned.csv
│   └── pamap2_engineered.csv
├── pamap2_models/
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── logistic_regression_model.pkl
│   └── results_summary.pkl
├── pamap2_visualizations/
│   └── (30+ PNG charts)
├── app.py                                  # Streamlit web application
├── Cleaning_and_preprocessing_pipeline.py  # Data cleaning
├── feature_engineering.py                  # Feature creation
├── train_ml_models.py                      # Model training
├── Visualisations.py                       # Generate charts
├── requirements.txt
└── README.md
```

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/PAMAP2-Activity-Recognition.git
cd PAMAP2-Activity-Recognition
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download PAMAP2 dataset**
- Download from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring)
- Place `.dat` files in `pamap2_data/` folder

4. **Run preprocessing** (if starting from raw data)
```bash
python Cleaning_and_preprocessing_pipeline.py
python feature_engineering.py
python train_ml_models.py
```

## 🎮 Usage

### Web Application
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

### Features:
- **🏠 Home**: Project overview and statistics
- **🎯 Predict**: Manual sensor input for predictions
- **🧪 Test Real Data**: Test with actual dataset samples
- **📊 Model Info**: Compare model performance
- **👥 Team**: Project team information

### Python API
```python
import joblib
import numpy as np

# Load model
model = joblib.load('pamap2_models/random_forest_model.pkl')

# Your sensor data (50 features)
sensor_data = np.array([...])  # Shape: (1, 50)

# Predict
prediction = model.predict(sensor_data)
confidence = model.predict_proba(sensor_data).max()

print(f"Predicted Activity: {prediction[0]}")
print(f"Confidence: {confidence:.2%}")
```

## 📈 Model Performance

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| **Random Forest** | **94.2%** | **0.938** | 45s |
| Gradient Boosting | 92.8% | 0.921 | 180s |
| Logistic Regression | 87.3% | 0.865 | 12s |

### Key Features (by importance):
1. Ankle acceleration magnitude (18%)
2. Chest acceleration magnitude (15%)
3. Hand acceleration magnitude (12%)
4. Heart rate (11%)
5. Overall activity intensity (9%)

## 🧪 Experiments

To retrain models with different configurations:
```bash
# Retrain with class balancing
python fix_model_balance.py

# Generate all visualizations
python Visualisations.py

# Generate ML-specific charts
python ML_Visualisations.py
```

## 📊 Visualizations

The project includes 30+ professional visualizations covering:
- Dataset overview and distribution
- Data quality analysis
- Sensor patterns across activities
- Feature engineering impact
- Model performance comparisons
- Confusion matrices and error analysis


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PAMAP2 Dataset: Reiss, A. and Stricker, D. (2012)
- UCI Machine Learning Repository

## 📧 Contact

For questions or collaboration:
- Email: sreeramachutuni@gmail.com

## 🔮 Future Improvements

- [ ] Real-time sensor data streaming
- [ ] Mobile app integration
- [ ] Deep learning models (LSTM, CNN)
- [ ] Transfer learning for new activities
- [ ] Edge deployment (Raspberry Pi)

---


---

## 📋 Files to Include in .gitignore

Create a `.gitignore` file:
```
# Data files (large)
*.dat
pamap2_data/*.csv
pamap2_data/*.dat

# Models (optional - include if small enough)
# pamap2_models/*.pkl

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
