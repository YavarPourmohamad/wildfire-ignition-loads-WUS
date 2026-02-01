import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

nofire_dt = ['Hypersampling', 'sss', 'tss4', 'tss15']
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

feat_dict = {
             'Popo_1km': 'Pop_1km',
             '11': 'Open Water',
             '12': 'Snow/Ice',
             '13': 'Developed-Upland Deciduous Forest',
             '14': 'Developed-Upland Evergreen Forest',
             '15': 'Developed-Upland Mixed Forest',
             '16': 'Developed-Upland Herbaceous',
             '17': 'Developed-Upland Shrubland',
             '22': 'Developed - Low Intensity',
             '23': 'Developed - Medium Intensity',
             '24': 'Developed - High Intensity',
             '25': 'Developed-Roads',
             '31': 'Barren',
             '32': 'Quarries-Strip Mines-Gravel Pits-Well and Wind Pads',
             '61': 'NASS-Vineyard',
             '63': 'NASS-Row Crop-Close Grown Crop',
             '64': 'NASS-Row Crop',
             '65': 'NASS-Close Grown Crop',
             '68': 'NASS-Wheat',
             '69': 'NASS-Aquaculture',
             '100': 'Sparse Vegetation Canopy',
             '200': 'Shrub Cover',
             '300': 'Herb Cover',
            }

causes =  ['Debris', 'Fireworks', 'Natural', 'Arson', 'Recreation', 'Smoking',
           'Equipment', 'Power', 'Misuse by minor', 'Firearms', 'Railroad']
for i, cause in enumerate(iterable=causes):
    print('\n',i+1,': Processing for', cause, 'ignition source...')
    fire_test = pd.read_csv(filepath_or_buffer=f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Data/X_test_{cause}.csv',
                                    sep=',')
    
    test_igni = pd.concat(objs = [fire_test, ign_ab_test])

    test_igni = test_igni.astype(np.float64)
    
    test_igni.sort_values(by=['FIRE_YEAR', 'DISCOVERY_DOY'], ascending=[True, True], inplace=True)
        
    test_igni = test_igni.drop(columns=['FIRE_SIZE', 'RPL_THEMES'])
    
    test_igni = test_igni[(test_igni['FIRE_YEAR'] > 1999) & (test_igni['FIRE_YEAR'] < 2019)]
    
    X_test_igni = test_igni.loc[:, test_igni.columns != 'Ignition']
    
    y_test_igni = test_igni.loc[:, 'Ignition']
    
    igni = joblib.load(filename=f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Final_models/{cause}_Ignition.pkl')
    
    y_pred_igni = igni.predict(X_test_igni)
    
    precision = precision_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    recall = recall_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    f1 = f1_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    accuracy = accuracy_score(y_pred=y_pred_igni, y_true=y_test_igni)*100
    print(f'\n--- Evaluation of Best {cause} Model on Test Set ---')
    print(f'Ignition Precision: {precision:.4f}%')
    print(f'Ignition Recall: {recall:.4f}%')
    print(f'Ignition F1 Score: {f1:.4f}%')
    print(f'Ignition Accuracy: {accuracy:.4f}%')

    print(f'\n--- Calculating SHAP vslues for {cause} Model on Test Set ---')
    shap.initjs()
    explainer = shap.TreeExplainer(igni.best_estimator_, approximate = True)
    shps = X_test_igni.sample(frac = 0.5)
    shap_values = explainer.shap_values(shps)
    print('--- Calculating Finihsed ---')

    feat_name = X_test_igni.columns.to_list()
    index_to_replace = feat_name.index('Annual_tempreture')
    feat_name[index_to_replace] = 'Annual_temperature'
    for idx, item in enumerate(X_test_igni.columns):
        if item in feat_dict:
            feat_name[idx] = feat_dict[item]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt_shap = shap.summary_plot(shap_values,                       # Use Shap values array
                                features = shps,                    # Use training set features
                                feature_names = feat_name,          # Use column names
                                show = False,                       # Set to false to output to folder
                                color_bar_label = 'Feature value',
                                plot_size = (15,5),                 # Change plot size
                                #  class_inds = 'original',         # It will always keep the class labels in the same order
                                )
    plt.title(label=f'{cause}',
              loc='center',
              fontdict={'fontsize': 18,
                        'fontweight':'bold'},
              )
    plt.savefig(f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Final_models/SHAP_{cause}.png')

    sh_val = np.zeros(shape = shap_values[0].shape)

    for cls in range(len(shap_values)):
        sh_val = sh_val + shap_values[cls]
    rf_resultX = pd.DataFrame(sh_val.reshape((1, 27)), columns = feat_name)
    vals = np.abs(rf_resultX.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feat_name, vals)),
                                    columns = ['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by = ['feature_importance_vals'],
                                ascending = False,
                                inplace = True)
    shap_importance.to_csv(path_or_buf = f'/bsuhome/yavarpourmohamad/scratch/Dissertation/Cause_specific/modeling/Final_models/feat_import_{cause}.csv',
                        sep = ',',
                        index = False)

    plt.close()