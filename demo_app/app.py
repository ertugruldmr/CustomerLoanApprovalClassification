import pickle
import json 
import gradio as gr
import numpy as np
import pandas as pd
import lightgbm
from lightgbm import LGBMClassifier

# Pre-Declearations of the params
# File Paths
model_path = 'lgbm_model.sav'
endoing_path = "cat_encods.json"
component_config_path = "component_configs.json"

# predefined
indexes = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Credit_History', 'Property_Area', 'ApplicantIncome_Log',
       'LoanAmount_Log', 'Loan_Amount_Term_Log', 'Total_Income_Log']
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Loan_Status']
num_cols = ['Credit_History', 'ApplicantIncome_Log', 'LoanAmount_Log','Loan_Amount_Term_Log', 'Total_Income_Log']

target = "Loan_Status"

feature_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Credit_History', 'Property_Area', 'ApplicantIncome_Log',
       'LoanAmount_Log', 'Loan_Amount_Term_Log', 'Total_Income_Log']

# Loading the files
model = pickle.load(open(model_path, 'rb'))

classes = json.load(open(endoing_path, "r"))
inverse_class = {col:{val:key for key, val in clss.items()}  for col, clss in classes.items()}

labels = classes["Loan_Status"].values()

feature_limitations = json.load(open(component_config_path, "r"))
import pandas as pd


# Util Functions
def decode(col, data):
  return classes[col][data]

def encode(col, str_data):
  return inverse_class[col][str_data]

def feature_decode(df):

  # exclude the target var
  cat_cols = list(classes.keys())
  cat_cols.remove("Loan_Status")

  for col in cat_cols:
     df[col] = decode(col, df[col])

  return df

def feature_encode(df):
  
  # exclude the target var
  cat_cols = list(classes.keys())
  cat_cols.remove("Loan_Status")
  
  for col in cat_cols:
     df[col] = encode(col, df[col])
  
  return df

def predict(*args):

  # preparing the input into convenient form
  features = pd.Series([*args], index=indexes)
  features = feature_encode(features)
  features = np.array(features).reshape(-1,11)

  # prediction
  probabilities = model.predict_proba(features) #.predict(features)
  probs = probabilities.flatten()

  # output form
  results = {l : np.round(p, 3) for l, p in zip(labels, probs)}

  return results

import gradio as gr

# Creating the web components
inputs = list()
for col in feature_order:
  if col in feature_limitations["cat"].keys():
    
    # extracting the params
    vals = feature_limitations["cat"][col]["values"]
    def_val = feature_limitations["cat"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Dropdown(vals, default=def_val, label=col))
  else:
    
    # extracting the params
    min = feature_limitations["num"][col]["min"]
    max = feature_limitations["num"][col]["max"]
    def_val = feature_limitations["num"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Slider(minimum=min, maximum=max, default=def_val, label=col) )

demo_app = gr.Interface(predict, inputs, "label")

# Launching the demo
if __name__ == "__main__":
    demo_app.launch()
