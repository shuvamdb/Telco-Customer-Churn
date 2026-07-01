# =============FASTAPI================
# FastApi serves as the serving layer, it turns our ML model into an API that can receive data and return preductions
# web framework for building APIs with Python based on standard Python type hints

#Pydantic validates and structures the data
# uvicorn is the server that runs fastapi applications

'''
FASTAPI + GRADIO SERVING APPLICATION - Production Ready ML Model Serving
========================================================================
This application provides a complete solution for the Telco Customer Churn model with both programmatic API access
and a user-friendly web interface.

Architecture :
1. FastAPI
2. Gradio
3. Pydantic

'''


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.serving.inference import predict
import gradio as gr

# Initialize fastapi application
app = FastAPI(title="Telco-Customer-Churn-Prediction-API", 
              description="ML API for predicting customer churn in telecom industry")

# HEALTH CHECK
@app.get("/")
def root():
    return {"status":"ok"}

# Pydantic model for automatic validation
class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents:str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity:str
    OnlineBackup: str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:float
    tenure:int

@app.post("/predict")
def get_prediction(data : CustomerData):
    '''
    Main prediction endpoint for customer churn
    
    1. Receives validated customer data via pydantic model.
    2. Calls the inference pipeline to transform features and predict.
    3. Returns churn prediction in json format.
    
    '''
    try:
        customer_dict = data.model_dump()
        result = predict(customer_dict)
        return {"prediction": result}
    except Exception as ee:
        return {"error": str(ee)}
    
# ===============================GRADIO UI INTERFACE================================

def gradio_app(gender,Partner,Dependents,PhoneService,MultipleLines,InternetService,OnlineSecurity,
              OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,
              PaymentMethod,MonthlyCharges,TotalCharges,tenure):
    try:
        
        payload = {
            "gender":gender,
            "Partner":Partner,
            "Dependents":Dependents,
            "PhoneService":PhoneService,
            "MultipleLines":MultipleLines,
            "InternetService":InternetService,
            "OnlineSecurity":OnlineSecurity,
            "OnlineBackup":OnlineBackup,
            "DeviceProtection":DeviceProtection,
            "TechSupport":TechSupport,
            "StreamingTV":StreamingTV,
            "StreamingMovies":StreamingMovies,
            "Contract":Contract,
            "PaperlessBilling":PaperlessBilling,
            "PaymentMethod":PaymentMethod,
            "MonthlyCharges":float(MonthlyCharges),
            "TotalCharges":float(TotalCharges),
            "tenure":int(tenure)
        }
        out = predict(payload)
        return str(out)
    
    except Exception as e:
        return f"Error : {str(e)}"

demo = gr.Interface(fn=gradio_app, inputs=[gr.Dropdown(["Male","Female"], label="Gender"),
                                          gr.Dropdown(["Yes","No"], label="Partner"),
                                          gr.Dropdown(["Yes","No"], label="Dependents"),
                                          gr.Dropdown(["Yes","No"], label="PhoneService"),
                                          gr.Dropdown(["Yes","No","No phone service"], label="MultipleLines"),
                                          gr.Dropdown(["DSL","Fiber optic", "No"], label="InternetService"),
                                          gr.Dropdown(["Yes","No","No internet service"], label="OnlineSecurity"),
                                          gr.Dropdown(["Yes","No","No internet service"], label="OnlineBackup"),
                                          gr.Dropdown(["Yes","No","No internet service"], label="DeviceProtection"),
                                          gr.Dropdown(["Yes","No","No internet service"], label="TechSupport"),
                                          gr.Dropdown(["Yes","No","No internet service"], label="StreamingTV"),
                                          gr.Dropdown(["Yes","No","No internet service"], label="StreamingMovies"),
                                          gr.Dropdown(["Month-to-month","One year","Two year"], label="Contract"),
                                          gr.Dropdown(["Yes","No"], label="PaperlessBilling"),
                                          gr.Dropdown(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], label="PaymentMethod"),
                                          gr.Number(label="MonthlyCharges"),
                                          gr.Number(label="TotalCharges"),
                                          gr.Number(label="tenure (months)")], 
                    outputs="text", 
                    title="Telco Churn Predictor", description="Fill in the customer details to get prediction")

app = gr.mount_gradio_app(app,                        # FastAPI application instance
                          demo,                 # Gradio interface
                          path="/ui")                 # url where gradio will be accessible