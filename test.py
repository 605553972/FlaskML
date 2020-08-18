import numpy
from MLA import predict_LSTM
import json
pred,act= predict_LSTM()
dic={}
dic['predict'] = pred.tolist()
dic['actual'] = act.tolist()
dicJson = json.dumps(dic)


print(dicJson)