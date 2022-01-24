#! /usr/bin/env python3

from flask import Flask, jsonify, abort, make_response, request
import model as M
import numpy as np
import yaml

with open("configs/model_config.yaml") as file:
    config = yaml.load(file, Loader=yaml.Loader)

model = M.load_model(config["model_name"])
targets = ['setosa', 'versicolor', 'virginica']
feature_names = config["feature_names"]

app = Flask(__name__)

def get_pred(data):
    result = model.predict_proba(data)
    pred_target = targets[np.argmax(result)]
    prob_target = np.max(result)
    return pred_target, prob_target

@app.route('/iris/api/v1.0/form_tabular/pred', methods=['GET', 'POST'])
def get_form_tabular_task():
    if request.method == 'POST':
        data = []
        for feature in feature_names:
            data.append(float(request.form.get(feature)))
        data = np.array([data])
        pred_target, prob_target = get_pred(data)
        return '''
                  <h1>With a probability of {}%, this flower is iris {}</h1>
               '''.format(prob_target * 100, pred_target)
    else:
        get_request_1 = '''
            <form method="POST">
                <h1>Please enter iris parameters:</h1>
        '''
        get_request_2 = "".join(
            [f'<div><label>{feature}: <input type="number" step=0.01 name="{feature}"></label></div>'
             for feature in feature_names]
        )
        get_request_3 = '''
            <input type="submit" value="Submit">
            </form>
        '''
        return "".join([get_request_1, get_request_2, get_request_3])

@app.route('/iris/api/v1.0/json_tabular/pred', methods=['POST'])
def get_json_tabular_task():
    request_data = request.get_json()
    data = []
    for feature in feature_names:
        data.append(float(request_data.get(feature)))
    data = np.array([data])
    pred_target, prob_target = get_pred(data)
    return '''
              <h1>With a probability of {}%, this flower is iris {}</h1>
           '''.format(prob_target * 100, pred_target)

@app.route('/iris/api/v1.0/tabular/pred', methods=['GET'])
def get_tabular_task():
    data = []
    for feature in feature_names:
        data.append(request.args.get(feature))
    data = np.array([data])
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
    app.run(port=config["port"], debug=True)