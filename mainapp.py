from flask import Flask,request,jsonify,redirect,render_template,\
    abort,make_response,session,g,render_template,url_for,flash
from MLA import predict_LSTM,predict_LGBM,predict_RandomForest,predict_Adaboost
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import json
import os


app = Flask(__name__)
app.config['UPLOAD_PATH'] = os.path.join(app.root_path, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
app.config['SECRET_KEY']="snahjhcbbajbay78"
allowed_file = {'csv'}
data=pd.DataFrame()

Algorithm={
    "LSTM":{
        "name":"LSTM_predict"
    },
    "LGBM":{
        "name":"LGBM_predict"
    }
}

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.datetime):
            return obj.__str__()
        else:
            return "error"

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename, 'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)

def request_parse(req_data):
    if req_data.method == 'POST':
        data = req_data.json
    elif req_data.method == 'GET':
        data = req_data.args
    return data

# @app.route("/", methods=['GET', "POST"])
def transmit(request):
    if request.method =="POST":
        #postdata = request.values.get('name')
        postbody= request.data
        j_data = json.loads(postbody)
        df = pd.DataFrame.from_dict(j_data)
        # global data
        # data=df
        # if postdata in Algorithm:
        #     name=Algorithm[postdata]["name"]
        #     return redirect(url_for(name))
    # elif request.method =="GET":
    #     if regressor_name in Algorithm:
    #         name=Algorithm[regressor_name]["name"]
    #         return redirect(url_for(name))
    return df

class UploadForm (FlaskForm):
    csvFile = FileField(u'Upload file', validators=[FileRequired(), FileAllowed(['csv'])])
    submit = SubmitField(label=u"提交")

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    form = UploadForm() #CombinedMultiDict([request.form, request.files]))
    if form.validate_on_submit():
        f = form.csvFile.data
        if f.filename== None:
            flash("no select file",category='danger')
            print("danger")
            return render_template("upload.html",form=form)
        if f :
            filename =secure_filename(f.filename)
            f.save(os.path.join(app.root_path,'uploads', filename))
            session['filenames'] = [filename]
        return redirect(url_for('index'))
    else:
        return "验证未通过"
    return render_template('upload.html', form = form)

@app.route("/set_cookie")
def set_cookie():
    resp=make_response("success")#g构造响应对象
    #设置cookie，临时cookie，浏览器关闭失效
    resp.set_cookie("itcast","python")#cookie名字，值
    resp.set_cookie("itcast1","python1")
    resp.set_cookie("itcast2","python2",max_age=3600)
    return resp

@app.route("/get_cookie")
def get_cookie():
    c=request.cookies.get("itcast")
    return c

@app.route("/delete_cookie")
def delete_cookie():
    resp=make_response("del success")
    resp.delete_cookie("itcast1")
    return resp

@app.route('/regression/LSTM', methods=['GET','POST'])
def LSTM_predict():
    df =transmit(request)
    pred,act= predict_LSTM(df)
    dic = {}
    dic['predict'] = pred.tolist()
    dic['actual'] = act.tolist()
    dicjson = json.dumps(dic)
    return dicjson

@app.route('/regression/LGBM', methods=['GET','POST'])
def LGBM_predict():
    df= transmit(request)
    pred,act= predict_LGBM(df)
    dic = {}
    dic['predict'] = pred.tolist()
    dic['actual'] = act.tolist()
    dicjson = json.dumps(dic)
    return dicjson

@app.route('/regression/RF', methods=['GET','POST'])
def RF_predict():
    df= transmit(request)
    pred,act= predict_RandomForest(df)
    dic = {}
    dic['predict'] = pred.tolist()
    dic['actual'] = act.tolist()
    dicjson = json.dumps(dic)
    return dicjson

@app.route('/regression/ADA', methods=['GET','POST'])
def Ada_predict():
    df= transmit(request)
    pred,act= predict_Adaboost(df)
    dic = {}
    dic['predict'] = pred.tolist()
    dic['actual'] = act.tolist()
    dicjson = json.dumps(dic)
    return dicjson

app.config.from_pyfile("config.cfg")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
