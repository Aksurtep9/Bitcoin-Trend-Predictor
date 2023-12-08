import torch
#from torch import nn
from models import model_vgg
from data_loader import *
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset_class import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


device = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:::::>>>>>>>>>>>>>>" + device)

loss_fn = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.SGD(params= model_vgg.model.parameters() , lr=0.001, weight_decay=0.01) 



def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.train()
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred_logit = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred_logit, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred_logit, dim=1), dim=1)
        train_acc += accuracy_fn(y_true=y,
                y_pred=y_pred_class 
            )


    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval() 
    with torch.inference_mode(): 
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            

            test_pred_logits = model(X)

            test_pred = torch.softmax(test_pred_logits, dim=1).argmax(dim=1)

            loss = loss_fn(test_pred_logits, y).item()
            test_loss += loss
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred 
            )
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        return test_loss, test_acc

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
          epochs: int = 500):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": []
    }
    temp_model = model
    patience = 15
    temp_loss, bad_series_counter = 100000, 0

    for epoch in tqdm(range(epochs)):
        
        if bad_series_counter == patience:
            print("EarlyStop")
            break
        
        train_loss, train_acc = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        

        test_loss, test_acc = test_step(model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn)
        
        if test_loss <= temp_loss:
            bad_series_counter = 0
            temp_model = model
        else:
            bad_series_counter += 1
            print(f"\nBADSERIES:{bad_series_counter}")
        
        temp_loss = test_loss
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {test_loss:.4f} | "
            f"valid: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(test_loss)
        results["valid_acc"].append(test_acc)

    timestamp_string = datetime.now().strftime('%Y%m%d%H%M%S')
    torch.save(temp_model.state_dict(),f"saved_models/vgg7_Leaky_e{epochs}_b4_wd0_01_lr0_01_t{timestamp_string}.pt")
    return results

def testing_model(model: torch.nn.Module, 
          test_dataloader: torch.utils.data.DataLoader, 
          loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()):
    
    test_loss, test_acc, index = 0, 0, 0
    count_zero, count_pred_zero,true_zero, false_zero = 0,0,0,0
    count_one, count_pred_one,true_one, false_one = 0,0,0,0
    count_two, count_pred_two,true_two, false_two = 0,0,0,0
    
    model.to(device)
    model.eval()
    with torch.inference_mode(): 
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            index += 1

            test_pred_logits = model(X)

            print(f"Test pred logits:\n {torch.softmax(test_pred_logits, dim=1)} ")

            test_pred = torch.softmax(test_pred_logits, dim=1).argmax(dim=1)

            if test_pred == 0:
                count_pred_zero +=1
                if y == 0:
                    true_zero +=1
                else:
                    false_zero +=1

            if(y == 0):
                count_zero += 1

            if test_pred == 1:
                count_pred_one +=1
                if y == 1:
                    true_one +=1
                else:
                    false_one +=1

            if(y == 1):
                count_one += 1

            if test_pred == 2:
                count_pred_two +=1
                if y == 2:
                    true_two +=1
                else:
                    false_two +=1

            if(y == 2):
                count_two += 1
            #loss += loss_fn(test_pred_logits, y)
            test_loss += loss_fn(test_pred_logits, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred 
            ) 

            print(f"Prediction: {test_pred}")
            print(f"True: {y}")
            print(f" index: {index} | "
            f"test_loss: {(test_loss / index):.4f} | "
            f"test_acc: {(test_acc / index):.4f}"
            )
        

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)


        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        print(f"All prediction 0s {count_pred_zero} | True 0s: {true_zero} | False 0s: {false_zero} | All 0s: {count_zero}\n")
        print(f"All prediction 1s {count_pred_one} | True 1s: {true_one} | False 1s: {false_one} | All 1s: {count_one}\n")
        print(f"All prediction 2s {count_pred_two} | True 2s: {true_two} | False 2s: {false_two} | All 2s: {count_two}\n")
        return test_loss.cpu().detach().item(), test_acc

    

torch.manual_seed(42)

model_results = train(model=model_vgg.model, train_dataloader=get_data_loader(BATCH=4, mode="TRAIN"), test_dataloader=get_data_loader(BATCH=4, mode="VALIDATION"), optimizer=optimizer, loss_fn=loss_fn, epochs=1000)

##modelVGG = model_vgg.model
##modelVGG.load_state_dict(torch.load(f"./saved_models/vgg7_Leaky_e1000_b4_wd0_01_lr0_01_t20231207204627.pt"))
##
##model_testing_result = testing_model(model=modelVGG,
##                                     test_dataloader=get_data_loader(BATCH=1, mode="TEST"),
##                                     loss_fn=loss_fn)

def plot_loss_curves(results):
    loss = results['train_loss']
    test_loss = results['valid_loss']


    accuracy = results['train_acc']
    test_accuracy = results['valid_acc']

    epochs = range(len(results['train_loss']))


    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='valid_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='valid_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show(block=True)



plot_loss_curves(model_results)

