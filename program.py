from flask import Flask,request,jsonify

import pickle
import numpy as np 

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))



@app.route('/')
def home():
    return "helloooo"


@app.route('/pred',methods=['POST'])
def Pred():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')


    input_query = np.array([[cgpa,iq,profile_score]])
    result = model.predict(input_query)[0]

    # result = {'cgpa':cgpa,'iq':iq,'profile_score':profile_score}
    
    return jsonify({'placement':str(result)})


if __name__ == '__main__':
    app.run(debug=True)