import streamlit as st
import numpy as np
import pandas as pd
import os
from joblib import dump, load
import base64
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
#https://stackoverflow.com/questions/46284107/feature-importance-using-imbalanced-learn-library

st.title('Backorder Prediction')


columns = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month',
       'forecast_6_month', 'forecast_9_month', 'sales_1_month',
       'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank',
       'potential_issue', 'pieces_past_due', 'perf_6_month_avg',
       'perf_12_month_avg', 'local_bo_qty', 'deck_risk', 'oe_constraint',
       'ppap_risk', 'stop_auto_buy', 'rev_stop']

log_columns = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]

def log_transform(a):

  sign = np.sign(a[log_columns])
  a[log_columns] =  np.log(1.0+abs(a[log_columns]))*sign
  return a


def get_table_download_link(df):

  """Generates a link allowing the data in a given panda dataframe to be downloaded
     in:  dataframe
    out: href string
  """
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
  return f'<a href="data:file/csv;base64,{b64}"  download="pred.csv">Download prediction file</a>'

def load_objects():
  imputer = load('imputer.joblib')
  scaler = load('scaler.joblib')
  model = load('model.joblib')
  return imputer,scaler,model


  
  

file = st.file_uploader("Upload input file ", type=["csv"])


def predict_label(raw_data):
    
    
    
    """
    This function input of one array 
    raw_data : Input data
    returns model predictions
    """
    imputer,scaler,model= load_objects()
    
    #Replace -99.0 with np.nan in performance columns
    raw_data[:,13][raw_data[:,13] == -99.0] = np.NaN
    raw_data[:,14][raw_data[:,14] == -99.0] = np.NaN
   
  
    #One-hotencodig categorical columns
    raw_data[:][raw_data[:]=='Yes'] = 1
    raw_data[:][raw_data[:]=='No'] = 0
            
    
    #Imputing missing values
    raw_data = imputer.transform(raw_data)

    #Applying logtransform
    raw_data = np.apply_along_axis(log_transform, 1, raw_data)
    
    
    #Scaling the input
    raw_data = scaler.transform(raw_data)
    predictions = model.predict(raw_data)

    
    predictions = predictions.astype(str)
    predictions[:][predictions[:]=='0'] = 'No'
    predictions[:][predictions[:]=='1'] = 'Yes'
    
    upload_data['Back_order_Prediction'] = predictions.reshape(-1,1)
    st.write('Download the model predictions using link below')
    

    st.markdown(get_table_download_link(upload_data), unsafe_allow_html=True)
    st.write('Predicted data with Model prediction in the last column : ')
    st.write(upload_data)
    importances = np.mean([est.steps[1][1].feature_importances_ for est in model.estimators_], axis=0)
    indices = np.argsort(importances)



    plt.title('Feature Importance in prediction of the output')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [columns[i] for i in indices])
    plt.xlabel('Relative Importance of each feature')
    st.pyplot()

    


if file is not None:
    upload_data = pd.read_csv(file)
    st.write('Uploaded data')
    st.write(upload_data)
    raw_data = upload_data.drop(['sku'], axis = 1).to_numpy()
    predict_label(raw_data)
     
  




