#!/usr/bin/env python
# coding: utf-8

####################  LSTM Crytocurrency Trading Bot  ###################


# Primary Imports

import pandas as pd
import numpy as np
import os
from random import shuffle
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from model import model


# Variables

SEQ_WINDOW = 60 #unit minute, how long of a preceding sequence to collect for RNN
FUTURE_PRED_WINDOW = 3 #unit minute, how far in the future are we predicting into
PAIR_TO_PRED = "BTC-USD"

BASE = r"~\repos\LSTM-Crypto-Trading-Bot-v1"


# Create Dataset

main_df = pd.DataFrame()
crypto_pairs = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
for pair in crypto_pairs:
    
    dataset = f"{BASE}/crypto_data/{pair}.csv"
    sub_df = pd.read_csv(dataset, names = ['time', 'low', 'high', "open", f"{pair}_close", f"{pair}_volume"])
    sub_df.set_index('time', inplace=True)
    sub_df = sub_df[[f"{pair}_close", f"{pair}_volume"]]
    
    if len(main_df)==0:
        main_df = sub_df
    else:
        main_df = main_df.join(sub_df)
        

main_df.fillna(method="ffill", inplace=True) #fill gaps with nan's
main_df.dropna(inplace=True) #drop all nan's
#main_df.head()
        

#Buy/sell logic
def buysell_logic(current, future):
    
    if float(future) > float(current):
        return 1 #buy condition
    else:
        return 0 #sell condition


#Compute future values for exisiting data (just a shift based on FUTURE_PRED_WINDOW)
main_df[f'{PAIR_TO_PRED}_future'] = main_df[f"{PAIR_TO_PRED}_close"].shift(-FUTURE_PRED_WINDOW)

#Compute target values for existing data (based on buy/sell logic)
main_df[f'{PAIR_TO_PRED}_target'] = list(map(buysell_logic, main_df[f"{PAIR_TO_PRED}_close"], main_df[f"{PAIR_TO_PRED}_future"]))

#main_df.head()


# Partion Main Dataset into Train & Test Datasets

times = sorted(main_df.index.values)
last_5pct = times[-int(len(times)*0.05)]
last_5pct_len = int(len(times)*0.05)

train_df = main_df[(main_df.index >= last_5pct)]
test_df = main_df[(main_df.index < last_5pct)]

# Partion Train Dataset into Train & Validation Datasets

times = sorted(train_df.index.values)
last_5pct = times[-last_5pct_len]

train_df = train_df[(train_df.index >= last_5pct)]
val_df = train_df[(train_df.index < last_5pct)]


# Normalise & Scale, Create Sequences and Balance Datasets

from sklearn import preprocessing
from collections import deque

#Create a preprocessing function that normalises the dataset
def preprocess_df(df, PAIR_TO_PRED):
    
    #Normalise and Scale
    
    df.drop(f'{PAIR_TO_PRED}_future', 1)
    
    for col in df.columns:
        if col != f'{PAIR_TO_PRED}_target':
            
            df[col] = df[col].pct_change() #percent change - normalises
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values) #scale between [0,1]
    
    df.dropna(inplace=True) #jic
    
    #Create Sequence
    
    sequential_data = []
    prev_window = deque(maxlen=SEQ_WINDOW) #keeps max. length by popping out older values as new ones are added

    for i in df.values:
        prev_window.append([n for n in i[:-1]]) #store all but target [[],[],[],[],[]x9]
        
        if len(prev_window)==SEQ_WINDOW:
            sequential_data.append([np.array(prev_window), i[-1]]) #append the sequences [features, label]

    shuffle(sequential_data) #shuffle for good measure
    
    #Balance
    
    buys = [] #store buy sequences
    sells = [] #store sell sequences
    
    for seq, target in sequential_data:
        
        if target==0:
            sells.append([seq, target])
        else:
            sells.append([seq, target])
            
    
    shuffle(sells)
    shuffle(buys)
    
    least = min(len(sells), len(buys))
    
    buys = buys[:least] #balance to least
    sells = sells[:least] #balance to least
    
    #Recombine and get dataset into feature, labels (and appt.data type)
    
    sequential_data = buys+sells
    shuffle(sequential_data) #shuffle to get rid of consec. 1s and 0s
    
    labels = []
    features = []
    for seq, target in sequential_data:
        
        features.append(seq)
        labels.append(target)
    
    
    return np.array(features), labels #features must always be a numpy array
        

train_features, train_labels = preprocess_df(train_df, PAIR_TO_PRED)
val_features, val_labels = preprocess_df(val_df, PAIR_TO_PRED)
test_features, test_labels = preprocess_df(test_df, PAIR_TO_PRED)

train_len = len(train_labels)
test_len = len(test_labels)
val_len = len(val_labels)

print(f"Train data: {train_len}, Validation data: {val_len} Test data: {test_len}")
print(f"Train Dont buys: {train_labels.count(0)}, Buys: {train_labels.count(1)}")
print(f"Validation Dont buys: {val_labels.count(0)}, Buys: {val_labels.count(1)}")
print(f"Test Dont buys: {test_labels.count(0)}, Buys: {test_labels.count(1)}")


# Create and fit model

MODEL_NAME = "lstm_m1"
VERSION = "v1"

if not os.path.exists(f"{BASE}/logs/{MODEL_NAME}/{VERSION}"):
    os.mkdir(f"{BASE}/logs/{MODEL_NAME}/{VERSION}")
if not os.path.exists(f"{BASE}/weights/{MODEL_NAME}/{VERSION}/"):
    os.mkdir(f"{BASE}/weights/{MODEL_NAME}/{VERSION}/")


BATCH_SIZE = 64 #reduce if OOM
EPOCHS = 10
lstm = model(train_len)

#Create callbacks
tf_callback = TensorBoard(log_dir=f"logs/{MODEL_NAME}/{VERSION}")
cp_callback = ModelCheckpoint(f"weights/{MODEL_NAME}/{VERSION}/{epoch:02d}-{val_acc:.3f}",
                                     monitor='val_acc',
                                     verbose=0, 
                                     save_best_only=False,
                                     save_weights_only=True,
                                     mode='max')

# Train model
history = lstm.fit(
    train_features, train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_features, val_labels),
    callbacks=[tf_callback, cp_callback])




