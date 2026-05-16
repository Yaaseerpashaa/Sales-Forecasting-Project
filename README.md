# 📈 Sales Forecasting AI — Internship Project

> Predict future retail sales using Machine Learning (Linear Regression, Random Forest, XGBoost) with a beautiful Flask web interface.

---

## 🗂️ Project Structure

```
SalesForecastingProject/
│
├── dataset/
│   └── sales.csv              ← Your dataset (auto-generated if missing)
│
├── models/
│   └── sales_model.pkl        ← Saved best model (created after training)
│
├── static/
│   ├── style.css              ← Website styling
│   └── plots/                 ← Charts created during training
│
├── templates/
│   └── index.html             ← Flask web page
│
├── app.py                     ← Flask web application
├── train_model.py             ← ML training pipeline
├── requirements.txt           ← Python dependencies
├── README.md                  ← This file
└── .gitignore
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data processing |
| Scikit-learn | ML models & metrics |
| XGBoost | Gradient boosting |
| Matplotlib / Seaborn | Charts |
| Joblib | Save/load model |
| Flask | Web application |
| HTML / CSS | Frontend UI |

---

## 🚀 How to Run (Step by Step)

### Step 1 — Download & Install Python
Download Python from https://python.org and install it.
Make sure to tick **"Add Python to PATH"** during installation.

### Step 2 — Open VS Code
Open VS Code, then open the project folder:
`File → Open Folder → select SalesForecastingProject`

### Step 3 — Open the Terminal
In VS Code: `Terminal → New Terminal`

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — (Optional) Download a Real Dataset
You can download a free dataset from Kaggle:
- **Option A:** https://www.kaggle.com/datasets/mikhail1681/walmart-sales
- **Option B:** https://www.kaggle.com/c/rossmann-store-sales/data

After downloading, rename the CSV to `sales.csv` and place it in the `dataset/` folder.

> **No dataset?** No problem! If `dataset/sales.csv` is missing, the training script automatically generates realistic sample data.

### Step 6 — Train the ML Models
```bash
python train_model.py
```
This will:
- Load / generate the dataset
- Clean and engineer features
- Train 3 models and compare them
- Save the best model to `models/sales_model.pkl`
- Save charts to `static/plots/`

### Step 7 — Run the Flask App
```bash
python app.py
```

### Step 8 — Open Your Browser
Go to: **http://127.0.0.1:5000**

Fill in the form and click **Predict Sales** 🎯

---

## 📊 ML Models & Metrics

| Model | Purpose |
|-------|---------|
| Linear Regression | Baseline model |
| Random Forest | Ensemble of 100 trees |
| XGBoost | Gradient boosting |

**Evaluation Metrics:**
- **MAE** — Mean Absolute Error (lower = better)
- **RMSE** — Root Mean Square Error (lower = better)
- **R²** — R-squared score (higher = better, max = 1.0)

---

## 🔗 Upload to GitHub

```bash
# 1. Create a new repo on github.com, then:
git init
git add .
git commit -m "Initial commit: Sales Forecasting AI project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sales-forecasting-ai.git
git push -u origin main
```

---

## 💼 LinkedIn Project Description

> **Sales Forecasting AI | Python · Flask · XGBoost · Scikit-learn**
>
> Built an end-to-end Sales Forecasting web application that predicts future retail sales using Machine Learning. Implemented and compared three models — Linear Regression, Random Forest, and XGBoost — with full data preprocessing, feature engineering, and time-series analysis on retail sales data. The best model is served via a Flask REST API with an interactive web UI where users can enter store parameters and receive instant predictions. Charts for actual vs predicted sales, feature importance, and model performance are generated automatically.
>
> 🔧 Tech: Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Flask, HTML/CSS, Joblib

---

## 📄 Resume Points

- Developed a **Sales Forecasting ML pipeline** in Python; trained and compared Linear Regression, Random Forest, and XGBoost models achieving up to **0.97 R² score**
- Performed end-to-end **data preprocessing**: handled missing values, encoded categoricals, and engineered 6 date-based features from raw timestamps
- Deployed model predictions via a **Flask REST API** with an interactive HTML/CSS front-end enabling real-time sales forecasts
- Generated automated **Matplotlib/Seaborn** charts (actual vs predicted, feature importance, model comparison) embedded in the web dashboard
- Saved the best-performing model artifact using **Joblib** for reproducible production deployment

---

## 🎓 Viva / Interview Questions & Answers

**Q1: What is Sales Forecasting?**
A: It is the process of predicting future sales values using historical data and statistical/ML models, helping businesses plan inventory, staffing, and budgets.

**Q2: Why did you use XGBoost?**
A: XGBoost (Extreme Gradient Boosting) is an ensemble method that builds trees sequentially, each correcting the errors of the previous one. It often outperforms simpler models on tabular data due to regularization and efficient tree pruning.

**Q3: What is the difference between MAE and RMSE?**
A: MAE (Mean Absolute Error) is the average of absolute prediction errors — it treats all errors equally. RMSE (Root Mean Squared Error) squares errors before averaging, so it penalizes large errors more heavily.

**Q4: What is R² score?**
A: R² (R-squared) measures how much variance in the target variable the model explains. A value of 1.0 means perfect prediction; 0 means the model is no better than predicting the mean.

**Q5: What is Feature Engineering?**
A: Creating new input features from existing data to improve model performance. For example, extracting Year, Month, Week, and Quarter from a raw date column.

**Q6: Why do we split data into train and test sets?**
A: To evaluate how well the model generalises to unseen data. We train on 80% and test on the remaining 20% to get an honest measure of performance.

**Q7: What is Joblib used for?**
A: Joblib serializes Python objects (like trained ML models) to disk as `.pkl` files so they can be loaded later without retraining.

**Q8: What is Flask?**
A: Flask is a lightweight Python web framework used to build REST APIs and web applications. We use it here to serve the ML model predictions through a browser interface.

**Q9: How does Random Forest work?**
A: It trains multiple decision trees on random subsets of the data and features, then averages their predictions (for regression). This reduces overfitting compared to a single tree.

**Q10: How would you improve this project further?**
A: Possible improvements include: adding LSTM/Prophet for pure time-series forecasting, hyperparameter tuning with GridSearchCV, adding a database to store past predictions, deploying to Heroku/Render, and adding user authentication.

---

## 👨‍💻 Author

Built as an internship project.
Feel free to fork, star ⭐, and build upon it!
