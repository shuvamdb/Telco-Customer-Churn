import gradio as gr
from src.serving.inference import predict

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

demo.launch()
              
