import streamlit as st
import requests

def main():
    st.title("River Water Quality Prediction Tool")

    module = st.sidebar.selectbox("Select Module", ["River Water Temperature", "Saturated Dissolved Oxygen"])

    if module == "River Water Temperature":
        run_river_water_temperature_module()

# Module for River Water Temperature
def run_river_water_temperature_module():
    st.header("River Water Temperature Prediction")

    air_temp = st.selectbox("Select Air Temperature", air_temp_data)
    observed_river_temp = st.selectbox("Select Observed River Temperature", observed_river_temp_data)
    model_name = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])
    metric = st.selectbox("Select Performance Metric", ["RMSE", "MAE", "MSE"])

    if st.button("Submit"):
        data = {
            "air_temperature": air_temp,
            "river_water_temperature": observed_river_temp,
            "model_type": model_name,
            "metric": metric
        }

        response = requests.post("http://localhost:8000/predict/", json=data)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Metric Value: {result['metric_value']}")
            st.write(f"Predicted River Temperature: {result['predicted_value']}")
        else:
            st.write("An error occurred during prediction.")

if __name__ == "__main__":
    main()
