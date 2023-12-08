import enum
import torch
import numpy as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data_loader import *
from sklearn.model_selection import train_test_split



class BTCDaily_Dataset(Dataset):

    def __init__(self):

        raw_data = get_DF_from_DB()

        sample_number = int(len(raw_data)*0.7)

        train_df=raw_data.iloc[:sample_number, :]
        test_val_df=raw_data.iloc[sample_number:, :]
        
        test_val_number = int(len(test_val_df)/2)

        val_df = test_val_df.iloc[:test_val_number, :]
        test_df = test_val_df.iloc[test_val_number:, :]

        print(val_df[:5])
        print(test_df[:5])


        train_df_norm = normalize_dataframe(train_df)
        val_df_norm = normalize_dataframe(val_df)
        test_df_norm = normalize_dataframe(test_df)

        self.train, self.labels_train = create_sections(train_df_norm, 14)
        print(f"TRAIN SET STAT -> 0: {self.labels_train.count(0)} 1: {self.labels_train.count(1)} 2: {self.labels_train.count(2)}")
        self.val, self.labels_val = create_sections(val_df_norm, 14)
        print(f"VALIDATION SET STAT -> 0: {self.labels_val.count(0)} 1: {self.labels_val.count(1)} 2: {self.labels_val.count(2)}")
        self.test, self.labels_test = create_sections(test_df_norm, 14) 
        print(f"TEST SET STAT -> 0: {self.labels_test.count(0)} 1: {self.labels_test.count(1)} 2: {self.labels_test.count(2)}")

        print(f"Number of training sections: {len(self.train)} ")
        print(f"Number of validation sections: {len(self.val)} ")
        print(f"Number of testing sections: {len(self.test)} ")




    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx],self.labels[idx]
    
    def set_fold(self,set_type):
        if set_type=="TRAIN":
            self.dataset = torch.tensor(self.train).float() 
            self.labels = self.labels_train
        if set_type=="TEST":
            self.dataset,self.labels=torch.tensor(self.test).float(),self.labels_test
        if set_type=="VALIDATION":
            self.dataset,self.labels = torch.tensor(self.val).float(), self.labels_val

        return self

ds = BTCDaily_Dataset()

def get_data_loader(BATCH, mode):
    torch.manual_seed(42)
    if mode=="TEST":
        return DataLoader(ds.set_fold("TEST"),
                          batch_size=BATCH,
                          shuffle=False
                          )
    elif mode=="TRAIN":
        return DataLoader(ds.set_fold("TRAIN"),
                          batch_size=BATCH,
                          shuffle=True
                          )
    
    elif mode=="VALIDATION":
        return DataLoader(ds.set_fold("VALIDATION"),
                          batch_size=BATCH,
                          shuffle=False)

