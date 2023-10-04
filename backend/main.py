from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

app = FastAPI()

# Sample data
# Sample water temperature data
water_temp_data = {
    "Year": [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015],
    "Jan": [27, 26, 26, 25, 23, 25, 26, 24, 26],
    # ... (other months)
    "Dec": [None, 27, 26, 26, 25, 28, 25, 27, 21]
}

# Model training
X = air_temp_data
y = water_temp_df["Jan"].values
# Model training for Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)

# Model training for Decision Tree
tree_model = DecisionTreeRegressor()
tree_model.fit(X, y)

class PredictionInput(BaseModel):
    air_temperature: float
    model_type: str
    metric: str

@app.post("/predict/")
def predict(input_data: PredictionInput):
    if input_data.model_type == 'Linear Regression':
        selected_model = linear_model
    elif input_data.model_type == 'Decision Tree':
        selected_model = tree_model
    else:
        return {"error": "Invalid model type"}

    y_pred = selected_model.predict(np.array([[input_data.air_temperature]]))
    
    # Dummy observed river temperature for metric calculation
    observed_river_temp = 27  
    
    if input_data.metric == 'RMSE':
        metric_value = mean_squared_error([observed_river_temp], [y_pred], squared=False)
    elif input_data.metric == 'MAE':
        metric_value = mean_absolute_error([observed_river_temp], [y_pred])
    elif input_data.metric == 'MSE':
        metric_value = mean_squared_error([observed_river_temp], [y_pred])
    
    return {"metric_value": metric_value, "predicted_value": y_pred[0]}
