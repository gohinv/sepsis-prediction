from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from hyperopt import STATUS_OK, hp, fmin, tpe
import pandas as pd
import numpy as np
import preprocessing as pre

def downsample(dataset):
    x, y = pre.process_data(dataset)
    index_0 = np.where(y == 0)[0]
    index_1 = np.where(y == 1)[0]
    
    index = index_0[len(index_1): -1]
    x_del = np.delete(x, index, 0)
    y_del = np.delete(y, index, 0)
    index = [i for i in range(len(y_del))]
    np.random.shuffle(index)
    x_del = x_del[index]
    y_del = y_del[index]
    
    return x_del, y_del

def hp_opt(x_train, y_train, x_val, y_val):
    train = xgb.DMatrix(x_train, label=y_train)
    val = xgb.DMatrix(x_val, label=y_val)
    x_val_D = xgb.DMatrix(x_val)
        
    def objective(params):
        xgb_model = xgb.train(params, dtrain=train, num_boost_round=1000, evals=[(val, 'eval')],
                              verbose_eval=False, early_stopping_rounds=80)
        y_vd_pred = xgb_model.predict(x_val_D, iteration_range=(0, xgb_model.best_iteration+1))
        y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]

        acc = accuracy_score(y_val, y_val_class)
        loss = 1 - acc
        return {'loss' : loss, 'params' : params, 'status' : STATUS_OK}

    max_depths = [3, 4]
    learning_rates = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
    subsamples = [0.5, 0.6, 0.7, 0.8, 0.9]
    colsample_bytrees = [0.5, 0.6, 0.7, 0.8, 0.9]
    reg_alphas = [0.0, 0.005, 0.01, 0.05, 0.1]
    reg_lambdas = [0.8, 1, 1.5, 2, 4]

    space = {
        'max_depth': hp.choice('max_depth', max_depths),
        'learning_rate': hp.choice('learning_rate', learning_rates),
        'subsample': hp.choice('subsample', subsamples),
        'colsample_bytree': hp.choice('colsample_bytree', colsample_bytrees),
        'reg_alpha': hp.choice('reg_alpha', reg_alphas),
        'reg_lambda': hp.choice('reg_lambda', reg_lambdas),
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)

    best_param = {'max_depth': max_depths[(best['max_depth'])],
                  'learning_rate': learning_rates[(best['learning_rate'])],
                  'subsample': subsamples[(best['subsample'])],
                  'colsample_bytree': colsample_bytrees[(best['colsample_bytree'])],
                  'reg_alpha': reg_alphas[(best['reg_alpha'])],
                  'reg_lambda': reg_lambdas[(best['reg_lambda'])]
                  }

    return best_param

def train_model(k, x_train, y_train, x_val, y_val, save_dir):
    print('*************************************************************')
    print('{}th training ..............'.format(k + 1))
    print('Hyperparameters optimization')
    best_param = hp_opt(x_train, y_train, x_val, y_val)
    xgb_model = xgb.XGBClassifier(max_depth = best_param['max_depth'],
                                  eta = best_param['learning_rate'],
                                  n_estimators = 1000,
                                  subsample = best_param['subsample'],
                                  colsample_bytree = best_param['colsample_bytree'],
                                  reg_alpha =  best_param['reg_alpha'],
                                  reg_lambda = best_param['reg_lambda'],
                                  objective = "binary:logistic"
                                   )
    xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='error',
                  early_stopping_rounds=80, verbose=False)
    
    y_tr_pred = (xgb_model.predict_proba(x_train, iteration_range=(0, xgb_model.best_iteration+1)))[:,1]
    train_auc = roc_auc_score(y_train, y_tr_pred)
    print('training dataset AUC: ' + str(train_auc))
    y_tr_class = [0 if prob <= 0.5 else 1 for prob in y_tr_pred]
    acc = accuracy_score(y_train, y_tr_class)
    print('training dataset accuracy: ' + str(acc))
    
    y_vd_pred = (xgb_model.predict_proba(x_val, iteration_range=(0, xgb_model.best_iteration+1)))[:, 1]
    valid_auc = roc_auc_score(y_val, y_vd_pred)
    print('validation dataset AUC: ' + str(valid_auc))
    y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]
    acc = accuracy_score(y_val, y_val_class)
    print('validation dataset accuracy: ' + str(acc))
    print('************************************************************')
    
    # save the model
    save_model_path = save_dir + 'model{}.mdl'.format(k + 1)
    xgb_model.get_booster().save_model(fname=save_model_path)
    

