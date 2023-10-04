from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = FastAPI()

# Load the saved models
best_linear_model = joblib.load('./best_linear_model.pkl')
best_rf_model = joblib.load('./best_rf_model.pkl')

class PredictionInput(BaseModel):
    air_temperature: float
    river_temperature: float
    model_type: str
    metric: str

@app.post("/predict/")
def predict(input_data: PredictionInput):
    if input_data.model_type == 'Linear Regression':
        selected_model = best_linear_model
    elif input_data.model_type == 'Random Forest':
        selected_model = best_rf_model
    else:
        return {"error": "Invalid model type"}

    # Prepare the input as numpy array
    # Here I'm assuming that the model uses air_temperature for prediction. 
    # Adjust this as needed for your specific use case.
    input_array = np.array([[input_data.air_temperature]])

    # Perform prediction
    y_pred = selected_model.predict(input_array)

    # Use the provided observed river temperature for metric calculation
    observed_river_temp = input_data.river_temperature
    
    if input_data.metric == 'RMSE':
        metric_value = np.sqrt(mean_squared_error([observed_river_temp], [y_pred]))
    elif input_data.metric == 'MAE':
        metric_value = mean_absolute_error([observed_river_temp], [y_pred])
    elif input_data.metric == 'MSE':
        metric_value = mean_squared_error([observed_river_temp], [y_pred])
    else:
        return {"error": "Invalid metric type"}

    return {"metric_value": metric_value, "predicted_value": y_pred[0]}

