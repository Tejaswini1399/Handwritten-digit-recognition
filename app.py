from flask import Flask
from flask import request
from flask import jsonify

import test
import json
import numpy as np

app = Flask(__name__)

model=test.loading_model()
@app.route('/result/',methods=['GET'])
def prediction():
    url = request.args.get('d')
    out= test.predcit(url,model)
    print(out)
    return jsonify(out)

if __name__ == '__main__':
 app.run(port=5000,host='0.0.0.0')

 
