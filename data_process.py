#Importing python packages
import numpy as np
import pandas as pd
import os
import argparse
import time
from sklearn.preprocessing import LabelEncoder

def downSample(dataset,down_size):

    user = dataset[:,6]
    phone = dataset[:,7]
    phone_model = dataset[:,8]    

    user_enc = LabelEncoder().fit(user)
    phone_enc = LabelEncoder().fit(phone)
    phone_mod_enc = LabelEncoder().fit(phone_model)

    t_user = user_enc.transform(user)
    t_phone = phone_enc.transform(phone)
    t_phone_model = phone_mod_enc.transform(phone_model)

    x_grad = np.array(np.gradient(dataset[:,3]))
    y_grad = np.array(np.gradient(dataset[:,4]))
    z_grad = np.array(np.gradient(dataset[:,5]))

    target=dataset[:,9]
    features=dataset[:,3:6]
    features=np.concatenate((features,x_grad.reshape(-1,1),y_grad.reshape(-1,1),z_grad.reshape(-1,1),t_user.reshape(-1,1),t_phone.reshape(-1,1),t_phone_model.reshape(-1,1)),axis=1)

    dataset = np.concatenate((features,target.reshape(-1,1)),axis=1)

    l_col = features.shape[1]

    del user,phone,phone_model,user_enc,phone_enc,phone_mod_enc,t_user,t_phone,t_phone_model,x_grad,y_grad,z_grad,target,features

    class_bike = np.where(dataset[:,l_col]=='bike')[0]
    class_sit = np.where(dataset[:,l_col]=='sit')[0]
    class_sd = np.where(dataset[:,l_col]=='stairsdown')[0]
    class_su = np.where(dataset[:,l_col]=='stairsup')[0]
    class_stand = np.where(dataset[:,l_col]=='stand')[0]
    class_walk = np.where(dataset[:,l_col]=='walk')[0]
    class_nan = np.where(str(dataset[:,l_col])=='nan')[0]
    
    bike_down = np.random.choice(class_bike, size=down_size, replace=False)
    sit_down = np.random.choice(class_sit, size=down_size, replace=False)
    sd_down = np.random.choice(class_sd, size=down_size, replace=False)
    su_down = np.random.choice(class_su, size=down_size, replace=False)
    stand_down = np.random.choice(class_stand, size=down_size, replace=False)
    walk_down = np.random.choice(class_walk, size=down_size, replace=False)

    bike_data   = dataset[bike_down]
    sit_data    = dataset[sit_down]
    sd_data     = dataset[sd_down]
    su_data     = dataset[su_down]
    stand_data  = dataset[stand_down]
    walk_data   = dataset[walk_down]

    data_down = np.concatenate((bike_data,sit_data,sd_data,su_data,stand_data,walk_data),axis=0)
    del class_bike, class_sit, class_sd, class_su, class_stand, class_walk, class_nan,bike_down,sit_down,sd_down,su_down,stand_down,walk_down,bike_data,sit_data,sd_data,su_data,stand_data,walk_data


    targetx=data_down[:,l_col]
    featuresx=data_down[:,0:l_col]

    enc = LabelEncoder().fit(targetx)
    targ = enc.transform(targetx)

    
    datasetx=np.concatenate((featuresx,targ[:,None]),axis=1)
    del targetx,featuresx,dataset
    
    return datasetx

