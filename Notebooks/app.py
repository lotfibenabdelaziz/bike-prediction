import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")


st.set_page_config(page_title="🚴‍♂️ Bike Rental Prediction App", layout="centered")
st.title("🚴‍♀️ Bike Rental Demand Prediction")
st.markdown("### Predict the number of rented bikes based on weather, time, and date conditions 🌤️")


#  Date Picker for daily features
selected_date = st.date_input("📅 Select Date")
selected_date = pd.to_datetime(selected_date)

# Extract daily features
day = selected_date.day
month = selected_date.month
weekday = selected_date.weekday()
dayofyear = selected_date.dayofyear


#  Input Features

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("🕒 Hour of the Day", 0, 23, 12)
    temp = st.number_input("🌡️ Temperature (°C)", min_value=-20.0, max_value=40.0, value=20.0)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    wind = st.number_input("🌬️ Wind Speed (m/s)", min_value=0.0, max_value=10.0, value=2.0)
    visibility = st.number_input("👀 Visibility (10m)", min_value=0.0, max_value=2000.0, value=1000.0)
    dew_point = st.number_input("🌫️ Dew Point (°C)", min_value=-20.0, max_value=30.0, value=10.0)

with col2:
    solar = st.number_input("☀️ Solar Radiation (MJ/m2)", min_value=0.0, max_value=5.0, value=1.5)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.0)
    snowfall = st.number_input("❄️ Snowfall (cm)", min_value=0.0, max_value=10.0, value=0.0)
    
    season = st.selectbox("🍂 Seasons", ["Spring", "Summer", "Autumn", "Winter"])
    holiday = st.selectbox("🏖️ Holiday", ["No Holiday", "Holiday"])
    functioning_day = st.selectbox("🏢 Functioning Day", ["Yes", "No"])
    
    is_holiday_workingday = st.selectbox("📅 Is Holiday & Working Day Combined?", ["No", "Yes"])
    is_clear_weather = st.selectbox("🌤️ Clear Weather?", ["No", "Yes"])
    is_rainy_weather = st.selectbox("🌧️ Rainy Weather?", ["No", "Yes"])
    is_snowy_weather = st.selectbox("❄️ Snowy Weather?", ["No", "Yes"])


#  Encode categorical data

season_map = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
binary_map = {"No": 0, "Yes": 1}
holiday_map = {"No Holiday": 0, "Holiday": 1}
func_map = {"Yes": 1, "No": 0}

season_val = season_map[season]
holiday_val = holiday_map[holiday]
function_val = func_map[functioning_day]
holiday_working_val = binary_map[is_holiday_workingday]
clear_weather_val = binary_map[is_clear_weather]
rainy_weather_val = binary_map[is_rainy_weather]
snowy_weather_val = binary_map[is_snowy_weather]


#  Prepare input dataframe

input_data = pd.DataFrame([[ 
    hour, temp, humidity, wind, visibility, dew_point, solar, rainfall, snowfall,
    season_val, holiday_val, function_val,
    holiday_working_val, clear_weather_val, rainy_weather_val, snowy_weather_val,
    month, day, weekday, dayofyear
]], columns=[
    'Hour', 'Temperature', 'Humidity', 'Wind_speed', 'Visibility', 'Dew_point_temperature',
    'Solar_Radiation', 'Rainfall', 'Snowfall',
    'Seasons', 'Holiday', 'Functioning_Day',
    'is_Holiday_WorkingDay', 'is_clear_weather', 'is_rainy_weather', 'is_snowy_weather',
    'Month', 'Day', 'Weekday', 'DayOfYear'
])


#  Prediction

if st.button("🚀 Predict Bike Rentals"):
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Rented Bike Count: **{int(prediction)}**")


#  Display data summary

st.markdown("---")
st.subheader("📊 Input Summary")
st.dataframe(input_data)
