import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
from multiprocessing import Pool
import lightgbm as lgb

processed_data_dir = 'data/preprocessed/'
log_dir = 'log/'
model_dir = 'model/'
submission_dir = 'submission/'

ver, KKK = 'priv', 0
STORES = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
CATS = ['HOBBIES','HOUSEHOLD', 'FOODS']

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def get_data_by_store(store, dept):
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    df = df[df['d']>=START_TRAIN]
    
    df = df[(df['store_id']==store) & (df['cat_id']==dept)]

    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
    
    df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2
    
    df = pd.concat([df, df3], axis=1)
    del df3
    
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]
    
    df = df.reset_index(drop=True)
    
    return df, features

def get_base_test():
    base_test = pd.DataFrame()

    for store_id in STORES:
        for state_id in CATS:
            temp_df = pd.read_pickle(processed_data_dir+'test_'+store_id+'_'+state_id+'.pkl')
            temp_df['store_id'] = store_id
            temp_df['cat_id'] = state_id
            base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    
    return base_test


def make_lag(LAG_DAY):
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'sales_lag_'+str(LAG_DAY)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)
    return lag_df[[col_name]]


def make_lag_roll(LAG_DAY):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.015,
                    'num_leaves': 2**8-1,
                    'min_data_in_leaf': 2**8-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 3000,
                    'boost_from_average': False,
                    'verbose': -1
                } 


VER = 1                          
SEED = 42                        
seed_everything(SEED)            
lgb_params['seed'] = SEED        
N_CORES = psutil.cpu_count()     


TARGET      = 'sales'            
START_TRAIN = 700                
END_TRAIN   = 1941 - 28*KKK      
P_HORIZON   = 28                 
USE_AUX     = False             

remove_features = ['id','cat_id', 'state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
mean_features   = ['enc_store_id_dept_id_mean','enc_store_id_dept_id_std',
                   'enc_item_id_store_id_mean','enc_item_id_store_id_std'] 

ORIGINAL = raw_data_dir
BASE     = processed_data_dir+'grid_part_1.pkl'
PRICE    = processed_data_dir+'grid_part_2.pkl'
CALENDAR = processed_data_dir+'grid_part_3.pkl'
LAGS     = processed_data_dir+'lags_df_28.pkl'
MEAN_ENC = processed_data_dir+'mean_encoding_df.pkl'


SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])



_, MODEL_FEATURES = get_data_by_store(STORES[-1], CATS[-1])
del _; gc.collect()


all_preds = pd.DataFrame()
base_test = get_base_test()

main_time = time.time()

for PREDICT_DAY in range(1,29):    
    print('Predict | Day:', PREDICT_DAY)
    start_time = time.time()

    grid_df = base_test.copy()

    temp = []
    for a in ROLS_SPLIT:
        temp.append(make_lag_roll(a))
    temp = pd.concat(temp, axis=1)
    grid_df = pd.concat([grid_df, temp], axis=1)
    del temp; gc.collect()

        
    for store_id in STORES:
        for state_id in CATS:
        
            model_path = model_dir+'lgb_model_'+store_id+'_'+state_id+'_v'+str(VER)+'.bin'
            if USE_AUX:
                model_path = AUX_MODELS + model_path

            estimator = pickle.load(open(model_path, 'rb'))

            day_mask = base_test['d']==(END_TRAIN+PREDICT_DAY)
            store_mask = base_test['store_id']==store_id
            state_mask = base_test['cat_id']==state_id

            mask = (day_mask)&(store_mask)&(state_mask)
            base_test[TARGET][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])
    
    temp_df = base_test[day_mask][['id',TARGET]]
    temp_df.columns = ['id','F'+str(PREDICT_DAY)]
    if 'id' in list(all_preds):
        all_preds = all_preds.merge(temp_df, on=['id'], how='left')
    else:
        all_preds = temp_df.copy()
        
    print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
    del temp_df
    
all_preds = all_preds.reset_index(drop=True)
all_preds


submission = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]
submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
submission.to_csv(submission_dir+'submission.csv', index=False)