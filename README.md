

# 🚴‍♂️ Bike Rental Demand Prediction

## **Project Overview**

Predict the number of bikes rented based on **time, weather, and daily features**. This project helps bike-sharing operators optimize fleet allocation, prepare for peak demand, and improve operational efficiency.

---

## **Dataset**

**Source:** [Add dataset link here]

**Columns / Features:**

| Feature               | Type  | Description                            |
| --------------------- | ----- | -------------------------------------- |
| Hour                  | int   | Hour of the day (0–23)                 |
| Temperature           | float | °C                                     |
| Humidity              | float | %                                      |
| Wind_speed            | float | m/s                                    |
| Visibility            | float | 10m units                              |
| Dew_point_temperature | float | °C                                     |
| Solar_Radiation       | float | MJ/m²                                  |
| Rainfall              | float | mm                                     |
| Snowfall              | float | cm                                     |
| Seasons               | int   | 0=Spring, 1=Summer, 2=Autumn, 3=Winter |
| Holiday               | int   | 0=No Holiday, 1=Holiday                |
| Functioning_Day       | int   | 0=No, 1=Yes                            |
| is_Holiday_WorkingDay | int   | Binary flag                            |
| is_clear_weather      | int   | Binary flag                            |
| is_rainy_weather      | int   | Binary flag                            |
| is_snowy_weather      | int   | Binary flag                            |
| Month                 | int   | Month extracted from date              |
| Day                   | int   | Day of month                           |
| Weekday               | int   | Day of week (0=Monday)                 |
| DayOfYear             | int   | Day of year                            |

**Target:** `Rented_Bike_Count`

---

## **Environment Setup**

### **1️⃣ Clone the repository**

```bash
https://github.com/lotfibenabdelaziz/bike-prediction.git
cd Notebooks
```

### **2️⃣ Install Python dependencies**

```bash
pip install -r requirements.txt
```

---

## **Running Locally**

### **Streamlit App**

```bash
streamlit run app.js
```

* Opens an interactive web app.
* Select date, hour, weather, and categorical features.
* Click **Predict** to see predicted bike rentals.

### **MLflow Tracking Server**

```bash
mlflow ui
```

* Default runs at `http://localhost:5000`.
* View logged models, parameters, metrics, and artifacts.

### **Docker**

```bash
docker build -t bike-rental-app .
docker run -p 8501:8501 bike-rental-app
```

* Streamlit app will be accessible at `http://localhost:8501`.

---

## **Recommended 3-Terminal Workflow**

For **robust testing and development**, run these three terminals simultaneously:

| Terminal | Command                   | Purpose                                        |
| -------- | ------------------------- | ---------------------------------------------- |
| 1        | `streamlit run app.js`    | Interactive web app for predictions            |
| 2        | `mlflow ui`               | View experiment metrics, models, and artifacts |
| 3        | `pytest test_training.py` | Automated tests for model training pipeline    |

This ensures your **app, tracking, and tests** are always synchronized.

---

## **Training Pipeline**

1. Preprocess data (feature extraction, encoding).
2. Split into `X` (features) and `y` (target).
3. Train **Random Forest Regressor**.
4. Log metrics and artifacts with **MLflow**.
5. Save trained model with **joblib**.
6. Generate **feature importance Plots**.

---

## **CI / GitHub Actions**

* Automatically runs tests on every push  to `main`.
* `.github/workflows/ci.yml`:

  * Installs dependencies
  * Runs `pytest`
  * Optionally builds Docker image

**Local CI Testing:**

```bash
cd Scripts
pytest test_training.py -v
```

---

## **Future Improvements**

* Real-time weather API integration.
* Geospatial station data for local demand prediction.
* Rolling time-series forecasting (LSTM/Prophet).
* Automated Docker deployment.

---

## **Git Commands**

```bash
git status       
git add .        
git commit -m "Add feature / fix bug"
git push origin main
git pull origin main
```

---

## **References**

* [Pandas Documentation](https://pandas.pydata.org/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Scikit-learn Documentation](https://scikit-learn.org/)
* MLflow Documentation: [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)