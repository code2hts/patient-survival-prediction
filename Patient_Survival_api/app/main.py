import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import pandas as pd
#from titanic_model.processing.data_manager import load_dataset, load_pipeline
#from titanic_model import __version__ as _version
#from titanic_model.config.core import config
from sklearn.model_selection import train_test_split
#from titanic_model.predict import make_prediction

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score




# FastAPI object
app = FastAPI()


# UI - Input components
in_age = gradio.Slider(1, 100, value=4, step = int, label = "Age", info='Age of the patient in yrs between 1 & 100')
in_anaemia = gradio.Radio(["no", "yes"], type="value", label = "anaemia :1 for yes and 0 for no")
in_creatine_ph = gradio.Slider(20, 1300, value=4, step = float, label = "Creatine Ph", info='creatinine phosphokinase between 20 & 1300')
in_diabetes = gradio.Radio(["no", "yes"], type="value", label = "diabetes :1 for yes and 0 for no")
in_ejection_frac = gradio.Slider(10, 70, value=4, step = float, label = "Ejection Fraction", info='ejection number between 10 & 70')
in_high_bp = gradio.Radio(["no", "yes"], type="value", label = "High BP :1 for yes and 0 for no")
in_platelets = gradio.Slider(10000, 500000, value=100000, step = int, label = "Platelets", info='platelets number between 10000 & 500000')
in_serum_creat = gradio.Slider(0, 3, value=0, step = float, label = "Serum Creatine", info='serum creatine fraction between 0 & 3')
in_serum_sodium = gradio.Slider(100, 150, value=0, step = float, label = "Serum Sodium", info='serum creatine fraction between 100 & 150')
in_sex = gradio.Radio(["Female", "Male"], type="value", label = "Sex :1 for Male and 0 for Female")
in_smoking = gradio.Radio(["no", "yes"], type="value", label = "Smoking :1 for Yes and 0 for No")
in_time = gradio.Slider(1, 300, value=0, step = int, label = "Time", info='Time value between 1 & 300')
#in_cp = gradio.Radio([0, 1, 2, 3, 4], type="value", label='Chest pain type')


# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")

# Label prediction function
def predict_death_event(in_age, in_anaemia, in_creatine_ph, in_diabetes, in_ejection_frac, in_high_bp, in_platelets, in_serum_creat, in_serum_sodium, in_sex, in_smoking, in_time):
#def predict_death_event(in_age, in_anaemia):

    save_file_name = "./../xgboost-model.pkl"
    
    with open(save_file_name, 'rb') as file:
    model = pickle.load(file)

    input_df = pd.DataFrame({"age": [in_age],
                             "anaemia": [in_anaemia],
                             "creatinine_phosphokinase": [in_creatine_ph],
                             "diabetes": [in_diabetes],
                             "ejection_fraction": [in_ejection_frac],
                             "high_blood_pressure": [in_high_bp],
                             "platelets": [in_platelets],
                             "serum_creatinine": [in_serum_creat],
                             "serum_sodium": [in_serum_sodium],
                             "sex": [in_sex],
                             "smoking": [in_smoking],
                             "time": [in_time]
    })

    yes_no_mapping = {'yes': 1, 'no': 0}
    input_df['anaemia'] = df['anaemia'].map(yes_no_mapping)
    input_df['diabetes'] = df['diabetes'].map(yes_no_mapping)
    input_df['high_blood_pressure'] = df['high_blood_pressure'].map(yes_no_mapping)
    input_df['smoking'] = df['smoking'].map(yes_no_mapping)

    sex_mapping = {'male':1, 'female':0}
    input_df['sex'] = input_df['sex'].map(sex_mapping)

    output = model.predict(input_df)
    label = "Survive" if output[0]==1 else "Not Survive"
    return label


# Create Gradio interface object
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = [in_age, in_anaemia, in_creatine_ph, in_diabetes, in_ejection_frac, in_high_bp, in_platelets, in_serum_creat, in_serum_sodium, in_sex, in_smoking, in_time],
                         #inputs = [in_age, in_anaemia],
                         outputs = [out_label],
                         title = title,
                         description = description,
                         allow_flagging='never')

#iface.launch(share = True, debug = True)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface
# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
