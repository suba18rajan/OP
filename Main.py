from flask import Flask, render_template, flash, request, session
from flask import render_template, redirect, url_for, request
#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename

#import mysql.connector
import smtplib
#from PIL import Image
import pickle

import numpy as np



app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

app.config['DEBUG']


@app.route("/")
def homepage():
    return render_template('index.html')
@app.route("/home")
def home():
    return render_template('index1.html')
@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')

@app.route("/AdminHome")
def AdminHome():
    return render_template('AdminHome.html')



@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':
       if request.form['uname'] == 'admin' or request.form['password'] == 'admin':

           return render_template('AdminHome.html')

       else:
        return render_template('index.html', error=error)

@app.route("/pre", methods=['GET', 'POST'])
def pre():
    error = None
    if request.method == 'POST':
        file=request.files['file']
        file.save("static/upload/" + secure_filename(file.filename))
        import warnings
        warnings.filterwarnings('ignore')
        import numpy
        import tensorflow as tf
        classifierLoad = tf.keras.models.load_model('my_trained_model.h5', compile=False)

        import numpy as np
        from tensorflow.keras.preprocessing import image
        test_image = image.load_img("static/upload/"+file.filename, target_size=(224, 224))
        # test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image)
        result = classifierLoad.predict(test_image)

        print(result)
        print(type(result))
        result = np.round(result).astype(int)
        print(result[0][0])
        if result[0][0] == 1:
            label = "Normal"
            tdata='age.txt'
        elif result[0][1] == 1:
            label = "Cataract"
            tdata = 'cataract.txt'
        elif result[0][2] == 1:
            label = "Diabetes"
            tdata = 'Diabetes.txt'
        elif result[0][3] == 1:
            label = "Glaucoma"
            tdata = 'Glaucoma.txt'
        elif result[0][4] == 1:
            label = "Hypertension"
            tdata = 'Hypertension.txt'
        elif result[0][5] == 1:
            label = "Myopia"
            tdata = 'Myopia.txt'
        elif result[0][6] == 1:
            label = "Age Issues"
            tdata = 'age.txt'
        else:
            label = "Other"
            tdata = 'age.txt'
        print(label)
        return render_template('result.html', data=label,image=file.filename,tdata=tdata)



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
