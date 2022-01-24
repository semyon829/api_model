#! /usr/bin/env python3

from flask import Flask, jsonify, abort, make_response, request
import model as M
import numpy as np

app = Flask(__name__)

model = M.load_model()
targets = ['setosa', 'versicolor', 'virginica']

def get_pred(data):
    result = model.predict_proba(data)
    pred_target = targets[np.argmax(result)]
    prob_target = np.max(result)
    return pred_target, prob_target

@app.route('/iris/api/v1.0/form_tabular/pred', methods=['GET', 'POST'])
def get_form_tabular_task():
    if request.method == 'POST':
        sepal_length = float(request.form.get('sepal length'))
        sepal_width = float(request.form.get('sepal width'))
        petal_length = float(request.form.get('petal length'))
        petal_width = float(request.form.get('petal width'))
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred_target, prob_target = get_pred(data)
        return '''
                  <h1>With a probability of {}%, this flower is iris {}</h1>
               '''.format(prob_target * 100, pred_target)
    return '''
              <form method="POST">
                  <h1>Please enter iris parameters:</h1>
                  <div><label>sepal length: <input type="number" step=0.01 name="sepal length"></label></div>
                  <div><label>sepal width: <input type="number" step=0.01 name="sepal width"></label></div>
                  <div><label>petal length: <input type="number" step=0.01 name="petal length"></label></div>
                  <div><label>petal width: <input type="number" step=0.01 name="petal width"></label></div>
                  <input type="submit" value="Submit">
              </form>'''

@app.route('/iris/api/v1.0/json_tabular/pred', methods=['POST'])
def get_json_tabular_task():
    if request.method == 'POST':
        request_data = request.get_json()
        sepal_length = float(request_data.get('sepal length'))
        sepal_width = float(request_data.get('sepal width'))
        petal_length = float(request_data.get('petal length'))
        petal_width = float(request_data.get('petal width'))
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred_target, prob_target = get_pred(data)
        return '''
                  <h1>With a probability of {}%, this flower is iris {}</h1>
               '''.format(prob_target * 100, pred_target)

@app.route('/iris/api/v1.0/tabular/pred', methods=['GET'])
def get_tabular_task():
    data = np.array(
        [[request.args.get('sepal_length'),
          request.args.get('sepal_width'),
          request.args.get('petal_length'),
          request.args.get('petal_width')]]
    )
    pred_target, prob_target = get_pred(data)
    return '''
              <h1>With a probability of {}%, this flower is iris {}</h1>
           '''.format(prob_target * 100, pred_target)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'code': 'PAGE_NOT_FOUND'}), 404)

@app.errorhandler(500)
def server_error(error):
    return make_response(jsonify({'code': 'INTERNAL_SERVER_ERROR'}), 500)


if __name__ == '__main__':
    app.run(port=5000, debug=True)