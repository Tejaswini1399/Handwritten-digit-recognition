import os
import sys
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
graph =tf.io.get_default_graph()

def loading_model():
    model=load_model('model.h5')
    print('model loaded')
    return model

def predcit(url,model):
    response=requests.get(url)
    img=Image.open(BytesIO(response.content))
    img=img.convert('L')
    img=np.array(img.resize((28,28)))
    img=img/255
    img=img.reshape(1,28,28,1)
   # print(img)
    with graph.as_default():
        b=model.predict(img)
    label=np.argmax(b)
    print(label)
    if label==0:
        label_name='0'
    elif label ==1:
        label_name='1'
    elif label==2:
        label_name='2'
    elif label==3:
        label_name='3'
    elif label ==4:
        label_name='4'
    elif label ==5:
        label_name='5'
    elif label ==6:
        label_name='6'
    elif label ==7:
        label_name='7'
    elif label ==8:
        label_name='8'
    elif label ==9:
        label_name='9'    
    return label_name
        

