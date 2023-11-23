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

        train_df=raw_data.sample(frac=0.8,random_state=42)
        test_df=raw_data.drop(train_df.index)

        train_df_norm = normalize_dataframe(train_df)
        test_df_norm = normalize_dataframe(test_df)

        self.train, self.labels = create_sections(train_df_norm, 14)
        self.test, self.labels_test = create_sections(test_df_norm, 14) 



    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx],self.labels[idx]
    
    def set_fold(self,set_type):
        if set_type=="TRAIN":
            self.dataset = torch.tensor(self.train).float() 
            self.labels = self.labels
        if set_type=="TEST":
            self.dataset,self.labels=torch.tensor(self.test).float(),self.labels_test
        
        return self

ds = BTCDaily_Dataset()

def get_data_loader(BATCH, mode):
    if mode=="TEST":
        ds.set_fold("TEST")
        return DataLoader(ds,
                          batch_size=BATCH,
                          shuffle=False
                          )
    elif mode=="TRAIN":
        ds.set_fold("TRAIN")
        return DataLoader(ds,
                          batch_size=BATCH,
                          shuffle=True
                          )



new_dl = get_data_loader(BATCH=8, mode="TRAIN")
