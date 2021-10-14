import streamlit
import joblib
from joblib import dump,load
import numpy as np

loaded = load('lr_model.pkl')

def lr_prediction(loaded,var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8):
      pred_arr=np.array([var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8])
      preds=pred_arr.reshape(1,-1)
      #preds=preds.astype(int)
      model_prediction=loaded.predict(preds)
      
      return model_prediction


def run():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h5 style ="color:black;text-align:center;">Housing Value Predictions</h5> 
    </div> 
    """
      
    # display the front end aspect
    streamlit.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    var_1=streamlit.text_input("Variable 1")
    var_2=streamlit.text_input("Variable 2")
    var_3=streamlit.text_input("Variable 3")
    var_4=streamlit.text_input("Variable 4")
    var_5=streamlit.text_input("Variable 5")
    var_6=streamlit.text_input("Variable 6")
    var_7=streamlit.text_input("Variable 7")
    var_8=streamlit.text_input("Variable 8")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if streamlit.button("Predict"): 
        prediction=lr_prediction(loaded,var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8)
        streamlit.success("The prediction by Model : {}".format(prediction))
        
if __name__=='__main__':
      run()
