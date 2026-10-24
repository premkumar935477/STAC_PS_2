# ☀️ Solar Radiation Analysis

## 📌 Project Overview

This project analyzes solar radiation data using Python, data visualization, and machine learning. It explores patterns in solar radiation, identifies environmental factors associated with radiation levels, and predicts solar radiation using a Random Forest regression model.

## 🎯 Objectives

* Analyze solar radiation patterns over time.
* Identify the relationship between temperature and solar radiation.
* Visualize trends and correlations using graphs.
* Build a machine learning model to predict solar radiation.
* Evaluate model performance using regression metrics.

## 📂 Project Structure

```text
STAC_PS_2/
├── solar_analysis.py
├── SolarPrediction.csv
├── solar_radiation.pptx
├── README.md
└── plots/
    ├── 01_....png
    └── ...
```

*Note: The PowerPoint filename and plot filenames above are examples. Update them to match your actual files.*

## 🛠️ Technologies Used

* **Python** — programming language
* **Pandas** — data handling and analysis
* **NumPy** — numerical operations
* **Matplotlib** — data visualization
* **Seaborn** — statistical visualization
* **Scikit-learn** — machine learning and model evaluation

## 📊 Key Findings

* The highest average solar radiation occurs around midday.
* Temperature shows a strong positive relationship with solar radiation in the analyzed dataset.
* A Random Forest regression model is used to predict solar radiation.
* Model performance is evaluated using RMSE and R².

## ⚙️ Installation

Install Python, then open a terminal in the project directory and run:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn
```

## ▶️ How to Run

Make sure `SolarPrediction.csv` and `solar_analysis.py` are in the same directory.

Run:

```bash
python solar_analysis.py
```

The script analyzes the dataset, trains the model, and saves visualization images in the `plots/` directory.

## 📈 Model Evaluation

The project evaluates its Random Forest regression model using:

* **RMSE (Root Mean Squared Error):** Measures the typical scale of prediction errors.
* **R² Score:** Measures how well the model explains variation in the target variable.

Actual results may vary depending on the data and evaluation method.

## 📁 Results

The generated plots help visualize solar radiation patterns, environmental relationships, and model predictions.

## 👨‍💻 Author

**sujal gupta**

## 📜 Disclaimer

This project is intended for educational and analytical purposes. Predictions depend on the quality of the dataset and the assumptions used during model training.
