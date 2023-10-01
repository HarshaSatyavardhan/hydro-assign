import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Main function
def main():
    st.title("River Water Quality Prediction Tool")

    module = st.sidebar.selectbox("Select Module", ["River Water Temperature", "Saturated Dissolved Oxygen"])

    if module == "River Water Temperature":
        run_river_water_temperature_module()
    else:
        run_saturated_dissolved_oxygen_module()

# Module for River Water Temperature
def run_river_water_temperature_module():
    st.header("River Water Temperature Prediction")

    # Dummy data
    air_temp_data = pd.DataFrame({"air_temperature": [20, 25, 22, 30, 35]})
    observed_river_temp_data = pd.DataFrame({"river_temperature": [19, 24, 21, 29, 34]})

    # Dropdown for Air Temperature and Observed River Temperature
    air_temp = st.selectbox("Select Air Temperature", air_temp_data["air_temperature"])
    observed_river_temp = st.selectbox("Select Observed River Temperature", observed_river_temp_data["river_temperature"])

    # Model selection
    model_name = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])

    # Performance metric
    metric = st.selectbox("Select Performance Metric", ["RMSE", "MAE", "MSE"])

    # Submit button
    if st.button("Submit"):
        model = None
        if model_name == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor()

        X = air_temp_data.values.reshape(-1, 1)
        y = observed_river_temp_data.values.reshape(-1, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if metric == "RMSE":
            st.write(f"Root Mean Squared Error: {mean_squared_error(y_test, y_pred, squared=False)}")
        elif metric == "MAE":
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
        else:
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Module for Saturated Dissolved Oxygen
def run_saturated_dissolved_oxygen_module():
    st.header("Saturated Dissolved Oxygen")

    # Your implementation here

# Run the main function
if __name__ == "__main__":
    main()
