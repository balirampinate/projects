import pickle
import numpy as np
import os
import joblib
from joblib import dump,load
import numpy as np



from flask import Flask, request
port = int(os.environ.get("PORT", 5000))


app = Flask(__name__)

def lr_prediction(loaded,var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8):
      pred_arr=np.array([var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8])
      preds=pred_arr.reshape(1,-1)
      #preds=preds.astype(int)
      model_prediction=loaded.predict(preds)
      
      return model_prediction

@app.route('/')
def hello():
    return "hellow world"

@app.route('/predict')
def predict_iris():
    v1 = request.args.get('v1')
    v2 = request.args.get('v2')
    v3 = request.args.get('v3')
    v4 = request.args.get('v4')
    v5 = request.args.get('v5')
    v6 = request.args.get('v6')
    v7 = request.args.get('v7')
    v8 = request.args.get('v8')

    new_record = np.array([[v1,v2,v3,v4,v5,v6,v7,v8]])
    
    loaded = load('lr_model.pkl')
    predict_result = lr_prediction(loaded,v1,v2,v3,v4,v5,v6,v7,v8)

    # return the result back

    return 'Result:'+ str(predict_result)

if __name__ == '__main__':

    app.run()