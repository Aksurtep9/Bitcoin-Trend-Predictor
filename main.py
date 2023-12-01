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


device = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:::::>>>>>>>>>>>>>>" + device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params= model_vgg.model.parameters() , lr=0.001) 



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
        #y_pred = torch.softmax(y_pred_logit, dim=1).argmax()
        loss = loss_fn(y_pred_logit, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred_logit, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred_logit)

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    #print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    #return train_loss.cpu().detach().item(), train_acc
    return train_loss, train_acc

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred_logits = model(X)

            test_pred = torch.softmax(test_pred_logits, dim=1).argmax(dim=1)
            
            # 2. Calculate loss and accuracy
            #loss += loss_fn(test_pred_logits, y)
            test_loss += loss_fn(test_pred_logits, y).item()
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        #print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        #return test_loss.cpu().detach().item(), test_acc
        return test_loss, test_acc

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):


        train_loss, train_acc = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        

        test_loss, test_acc = test_step(model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {test_loss:.4f} | "
            f"valid: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(test_loss)
        results["valid_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

def testing_model(model: torch.nn.Module, 
          test_dataloader: torch.utils.data.DataLoader, 
          loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()):
    
    test_loss, test_acc = 0, 0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            test_pred = torch.softmax(test_pred_logits, dim=1).argmax(dim=1)
            
            # 2. Calculate loss and accuracy
            #loss += loss_fn(test_pred_logits, y)
            test_loss += loss_fn(test_pred_logits, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred # Go from logits -> pred labels
            )
            print(
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            )
        
        # Adjust metrics and print out
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        #print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        return test_loss.cpu().detach().item(), test_acc

    



model_results = train(model=model_vgg.model,
                          train_dataloader=get_data_loader(BATCH=4, mode="TRAIN"),
                          test_dataloader=get_data_loader(BATCH=4, mode="VALIDATION"),
                          optimizer=optimizer,
                          loss_fn=loss_fn,
                          epochs=800)

model_testing_result = testing_model(model=model_vgg.model,
                                     test_dataloader=get_data_loader(BATCH=4, mode="TEST"),
                                     loss_fn=loss_fn)


#print(X[:5], y[:5])

#torch.save(model_vgg.model, 'saved_models/vgg_model.pt')
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['valid_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['valid_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    #print(loss)
    #test_loss.cpu()
    #accuracy.cpu()
    #test_accuracy.cpu()

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

#torch.manual_seed(42)

print(model_results)

plot_loss_curves(model_results)

#loss = model_results['train_loss']
#print(loss)
#print(type(loss[0]))