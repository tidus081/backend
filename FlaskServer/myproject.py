from flask import Flask, request, jsonify, render_template
import pandas as pd
import ins_controller as ic

class check:
    user=False

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data', methods=['POST'])
def data():
    if request.method == 'POST':
        try:
            input_json = request.get_json(force=True) #Receive data from google app script
            src = 'google_sheet'
            user, pname, sname, data = input_json
        except:
            return jsonify("Can't read the data from google sheet.")

        print('---request update data') #Send parameter to store instance
        try:
            res = ic.client_store(type="pass_data", src=src, pname=pname, sname=sname,
                                  user=user, data=str(data), extension='csv')
        except:
            return jsonify('Unable to process the request; please try again later.')

        print("data received: " + res)
        return jsonify(res)

@app.route('/model', methods=['POST'])
def model():
    if request.method == 'POST':
        try:
            input_json = request.get_json(force=True)
            src = 'google_sheet'
            user, pname, sname = input_json
            print(user, pname, sname)
        except:
            return jsonify("Can't read the data from google sheet.")

        print ('---request build model') #Send parameter to model instance
        try:
            res= ic.client_model(type="model", src=src, pname=pname, sname=sname, user=user)
        except:
            return jsonify('Unable to process the request; please try again later.')

        print("model received: " + res)
        return jsonify(res)

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        try:
            input_json = request.get_json(force=True)
            src = 'google_sheet'
            user, pname, sname, data = input_json
            print(user, pname, sname, data)
        except:
            return jsonify("Can't read the data from google sheet.")

        print ('---request prediction') #Send parameter to model instance
        try:
            res = ic.client_model(type="predict", src=src, pname=pname,
                                    sname=sname, user=user, data=str(data))
        except:
            return jsonify('Unable to process the request; please try again later.')

        print("predict received: " + res)
        return jsonify(res)

@app.route('/anomaly', methods=['POST'])
def anomaly():
    if request.method == 'POST':
        try:
            input_json = request.get_json(force=True)
            src = 'google_sheet'
            user, pname, sname, data, real = input_json
            print(user, pname, sname, data, real)
        except:
            return jsonify("Can't read the data from google sheet.")

        print ('---request anomaly detection') #Send parameter to model instance
        # force=True, above, is necessary if another developer
        # forgot to set the MIME type to 'application/json'
        res = ic.client_model(type='anomaly', src=src, pname=pname, sname=sname,
                                user=user, data="%s"%(data), real="%s"%(real))
        try:
            res = ic.client_model(type='anomaly', src=src, pname=pname, sname=sname,
                                    user=user, data="%s"%(data), real="%s"%(real))
        except:
            return jsonify('Unable to process the request; please try again later.')

        return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
