
import xgboost as xgb
from global_var import *
import pickle
import math

# loc_data  = os.path.join(g_loc_tmp, "data.pkl")

def load_data(loc):
    with open(loc) as f:
        (d_data, lst_label) = pickle.load(f)
    return d_data, lst_label

def train():
    d_train, l_train = load_data(g_loc_train)#g_loc_train
    l_train_reg = []
    for i in l_train:
        l_train_reg.append(1.0/(1+math.exp(-5*i)))
    dTrain = xgb.DMatrix(d_train, label=l_train_reg)
    
    d_test, l_test = load_data(g_loc_test)#g_loc_test
    l_test_reg = []
    for i in l_test:
        l_test_reg.append(1.0/(1+math.exp(-5*i)))
    dTest = xgb.DMatrix(d_test, label=l_test_reg)
    
    param = {'objective': 'binary:logistic', 'eval_metric':'error', 'bst:eta':0.05, 'subsample':0.85, 'bst:max_depth':5, 'bst:silent':1}
    paramList = param.items()
    
    num_round = 5000
    evalList = [(dTrain, 'train'), (dTest, 'test')]
    bst = xgb.train(paramList, dTrain, num_round, evalList, early_stopping_rounds=100)
    
    loc_prediction = "pred_test.pkl"
    res = bst.predict(dTest)
    with open(loc_prediction, "w") as f:
        pickle.dump(res, f)
    
if __name__ == '__main__':
    train()
    