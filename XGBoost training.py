import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils import class_weight
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from skopt import BayesSearchCV
print('start...')
# Ignition absence
nofire_dt = ['Hypersampling', 'sss', 'tss4', 'tss15']
ign_ab_X_train = []
for i in nofire_dt:
    print('Processing', i, 'training dataset...')
    temp_X_train = pd.read_csv(filepath_or_buffer=f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Data/X_train_{i}.csv',
                                 sep=',',
                                 low_memory=False)
    ign_ab_X_train.append(temp_X_train)

print('\nConcatenation starts...')
ign_ab_train = pd.concat(ign_ab_X_train, ignore_index=True)
ign_ab_train['Ignition'] = 0

ign_ab_X_test = []
for i in nofire_dt:
    print('Processing', i, 'test dataset...')
    temp_X_test = pd.read_csv(filepath_or_buffer=f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Data/X_test_{i}.csv',
                              sep=',',
                              low_memory=False)
    ign_ab_X_test.append(temp_X_test)

print('\nConcatenation starts...')
ign_ab_test = pd.concat(ign_ab_X_test, ignore_index=True)
ign_ab_test['Ignition'] = 0

# Ignition Presence and training loop
causes =  ['Debris', 'Fireworks', 'Natural', 'Arson', 'Recreation', 'Missing', 'Smoking',
           'Equipment', 'Power', 'Misuse by minor', 'Other causes', 'Firearms', 'Railroad']
size_train_list = []
size_test_list = []
for i, cause in enumerate(iterable=causes):
    print('\n',i+1,': Processing for', cause, 'ignition source...')
    fire_train = pd.read_csv(filepath_or_buffer=f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Data/X_train_{cause}.csv',
                                    sep=',')
    fire_test = pd.read_csv(filepath_or_buffer=f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Data/X_test_{cause}.csv',
                                    sep=',')
    data_size = fire_train.copy(deep=True)
    test_size = fire_test.copy(deep=True)
    fire_train['Ignition'] = 1
    fire_test['Ignition'] = 1

    data_igni = pd.concat(objs = [fire_train, ign_ab_train])
    test_igni = pd.concat(objs = [fire_test, ign_ab_test])

    data_igni = data_igni.astype(np.float64)
    data_size = data_size.astype(np.float64)
    test_igni = test_igni.astype(np.float64)
    test_size = test_size.astype(np.float64)

    data_igni.sort_values(by=['FIRE_YEAR', 'DISCOVERY_DOY'], ascending=[True, True], inplace=True)
    data_size.sort_values(by=['FIRE_YEAR', 'DISCOVERY_DOY'], ascending=[True, True], inplace=True)
    test_igni.sort_values(by=['FIRE_YEAR', 'DISCOVERY_DOY'], ascending=[True, True], inplace=True)
    test_size.sort_values(by=['FIRE_YEAR', 'DISCOVERY_DOY'], ascending=[True, True], inplace=True)

    data_size.loc[data_size['FIRE_SIZE'] >= 99.99, 'big_fire'] = 1
    test_size.loc[test_size['FIRE_SIZE'] >= 99.99, 'big_fire'] = 1
    data_size.loc[data_size['FIRE_SIZE'] < 99.99, 'big_fire'] = 0
    test_size.loc[test_size['FIRE_SIZE'] < 99.99, 'big_fire'] = 0

    data_igni = data_igni.drop(columns=['FIRE_SIZE', 'RPL_THEMES'])
    data_size = data_size.drop(columns=['FIRE_SIZE', 'RPL_THEMES'])
    test_igni = test_igni.drop(columns=['FIRE_SIZE', 'RPL_THEMES'])
    test_size = test_size.drop(columns=['FIRE_SIZE', 'RPL_THEMES'])

    data_igni = data_igni[(data_igni['FIRE_YEAR'] > 1999) & (data_igni['FIRE_YEAR'] < 2019)]
    data_size = data_size[(data_size['FIRE_YEAR'] > 1999) & (data_size['FIRE_YEAR'] < 2019)]
    test_igni = test_igni[(test_igni['FIRE_YEAR'] > 1999) & (test_igni['FIRE_YEAR'] < 2019)]
    test_size = test_size[(test_size['FIRE_YEAR'] > 1999) & (test_size['FIRE_YEAR'] < 2019)]

    X_train_igni = data_igni.loc[:, data_igni.columns != 'Ignition']
    X_train_size = data_size.loc[:, data_size.columns != 'big_fire']
    X_test_igni = test_igni.loc[:, test_igni.columns != 'Ignition']
    X_test_size = test_size.loc[:, test_size.columns != 'big_fire']
    
    y_train_igni = data_igni.loc[:, 'Ignition']
    y_train_size = data_size.loc[:, 'big_fire']
    y_test_igni = test_igni.loc[:, 'Ignition']
    y_test_size = test_size.loc[:, 'big_fire']

    param = {'tree_method': ['hist'], # gpu_hist, hist
             'base_score': (0.2, 0.8),
             'booster': ['gbtree'], # 'gbtree', 'gblinear', 'dart'
             'objective': ['binary:logistic'], # 'binary:logistic', 'binary:logitraw'
             'max_depth': (2, 10),
             'alpha': (0, 2),
             'gamma':(0, 10),
             'subsample': (0.5, 1),
             'learning_rate': (0.01, 0.3),
             'n_estimators': (350, 450),
             'min_child_weight': (1, 10),
             }
    # Without weights
    clf_XGb_igni = xgb.XGBClassifier(seed=42)
    clf_XGb_igni = BayesSearchCV(estimator=clf_XGb_igni,
                                 search_spaces=param,
                                 cv=5,
                                 n_iter=10,
                                 verbose=1)

    clf_XGb_igni.fit(X_train_igni, y_train_igni)
    
    joblib.dump(value=clf_XGb_igni,
                filename=f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Final_models/{cause}_Ignition.pkl')

    y_pred_igni = clf_XGb_igni.predict(X_test_igni)

    precision = precision_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    recall = recall_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    f1 = f1_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    accuracy = accuracy_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    print(f'\n--- Evaluation of Best {cause} Model on Test Set ---')
    print(f'Ignition Precision: {precision:.4f}%')
    print(f'Ignition Recall: {recall:.4f}%')
    print(f'Ignition F1 Score: {f1:.4f}%')
    print(f'Ignition Accuracy: {accuracy:.4f}%')

    # With weights
    cls_wght_igni = class_weight.compute_sample_weight(class_weight='balanced', y=y_train_igni)

    clf_XGb_igni_ww = xgb.XGBClassifier(seed=42)

    clf_XGb_igni_ww = BayesSearchCV(estimator=clf_XGb_igni_ww,
                                    search_spaces=param,
                                    cv=5,
                                    n_iter=10,
                                    verbose=1)
    
    clf_XGb_igni_ww.fit(X_train_igni, y_train_igni, sample_weight=cls_wght_igni)

    joblib.dump(value=clf_XGb_igni_ww,
                filename=f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Final_models/{cause}_Ignition_ww.pkl')

    y_pred_igni = clf_XGb_igni_ww.predict(X_test_igni)

    precision = precision_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    recall = recall_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    f1 = f1_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    accuracy = accuracy_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    print(f'\n--- Evaluation of Best {cause} Model with weight on Test Set ---')
    print(f'Ignition Precision: {precision:.4f}%')
    print(f'Ignition Recall: {recall:.4f}%')
    print(f'Ignition F1 Score: {f1:.4f}%')
    print(f'Ignition Accuracy: {accuracy:.4f}%')