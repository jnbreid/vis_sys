import numpy as np
import torch
import matplotlib.pyplot as plt

# changed eval to possibly include confusion matrix
def eval_model_conf(model, data_loader, criterion, device):
  with torch.no_grad():
      model.eval()
      
      correct = 0
      total = 0
      loss_list = []

      # lists to save labels for confusion matrix
      correct_labels = []
      predicted_labels = []
      
      for images, labels in data_loader:
          images = images.to(device)
          labels = labels.to(device)

          outputs = model(images)
                  
          loss = criterion(outputs, labels)
          loss_list.append(loss.item())
              
          preds = torch.argmax(outputs, dim=1)
          correct += len( torch.where(preds==labels)[0] )
          total += len(labels)

          correct_labels = correct_labels + list(labels.to('cpu').numpy())
          predicted_labels= predicted_labels + list(preds.to('cpu').numpy())

      accuracy = correct / total * 100
      loss = np.mean(loss_list)

      model.train()
      return accuracy, loss, correct_labels, predicted_labels

# function to count model parameters that are adjusted during training
def model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# evaluation function for specified dataset
def eval_model(model, data_loader, criterion, device):
  with torch.no_grad():
      model.eval()
      
      correct = 0
      total = 0
      loss_list = []
      
      for images, labels in data_loader:
          images = images.to(device)
          labels = labels.to(device)

          outputs = model(images)
                  
          loss = criterion(outputs, labels)
          loss_list.append(loss.item())
              
          preds = torch.argmax(outputs, dim=1)
          correct += len( torch.where(preds==labels)[0] )
          total += len(labels)

      accuracy = correct / total * 100
      loss = np.mean(loss_list)

      model.train()
      return accuracy, loss


# evaluation function for specified dataset

# evaluation function for specified dataset

# in the other notebooks this function is used from utils.py
from tqdm import tqdm

def train(model, NUM_EPOCHS, train_loader,train_loader1, test_loader, optimizer, criterion, EVAL_FREQ, SAVE_FREQ, savepath, device, scheduler, twriter = None, loadpath = None, stats = None):
  
  if loadpath != None:
    model, optimizer, init_epoch, stats = load_model(model, optimizer, loadpath)
    print('loaded model: ' + loadpath)
  elif stats != None:
     stats = stats
     init_epoch = 0
  else:
    stats = {
    "epoch": [],
    "full_train_loss": [],
    "train_loss": [],
    "valid_loss": [],
    "train_accuracy": [],
    "valid_accuracy": [],
    "per_batch_loss": [],
    }
    init_epoch = 0

  model.to(device)
  model.train()

  loss_hist = []
  best_acc = 0
  current_lr = scheduler.get_last_lr()[0]

  # evaluate once before start of training
  #train_accuracy, train_loss = eval_model(model, train_loader, criterion, device)
  valid_accuracy, valid_loss = eval_model(model, test_loader, criterion, device)
  #stats["train_loss"].append(train_loss)
  stats["valid_loss"].append(valid_loss)
  #stats["train_accuracy"].append(train_accuracy)
  stats["valid_accuracy"].append(valid_accuracy)
  if len(stats["epoch"]) == 0:
      stats["epoch"].append(0)
  else:
      stats["epoch"].append(stats["epoch"][-1]+1)
  print(f"Epoch -1: valid_loss {valid_loss:.5f} valid_acc {valid_accuracy:.2f} | lr {current_lr}")

  for epoch in range(init_epoch, NUM_EPOCHS):
      loss_list = []
      progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
      for i, (images, labels) in progress_bar:
          images = images.to(device)
          labels = labels.to(device)
          optimizer.zero_grad()
          
          outputs = model(images)

          loss = criterion(outputs, labels)
          loss_list.append(loss.item())
          stats["per_batch_loss"].append(loss.cpu().detach().numpy())

          loss.backward()

          optimizer.step()

          #progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: loss {loss.item():.5f}. ")
              
      loss_hist.append(np.mean(loss_list))
      stats["epoch"].append(stats["epoch"][-1]+1)
      stats["full_train_loss"].append(loss_hist[-1])
      if twriter != None:
         twriter.add_scalar(f'Full Loss', stats["full_train_loss"][-1], global_step=stats["epoch"][-1])

      if epoch % EVAL_FREQ == 0:
          #train_accuracy, train_loss = eval_model(model, train_loader, criterion, device)
          valid_accuracy, valid_loss = eval_model(model, test_loader, criterion, device)
          #stats["train_loss"].append(train_loss)
          stats["valid_loss"].append(valid_loss)
          #stats["train_accuracy"].append(train_accuracy)
          stats["valid_accuracy"].append(valid_accuracy)
          if twriter != None:
            twriter.add_scalar(f'Accuracy/Valid', valid_accuracy, global_step=stats["epoch"][-1])
            twriter.add_scalar(f'Loss/Valid', valid_loss, global_step=stats["epoch"][-1])
            #twriter.add_scalar(f'Accuracy/Train', train_accuracy, global_step=stats["epoch"][-1])
            #twriter.add_scalar(f'Loss/Train', train_loss, global_step=stats["epoch"][-1])
          
          print(f"Epoch {epoch+1}: train_loss {np.mean(loss_list):.5f} | valid_loss {valid_loss:.5f} valid_acc {valid_accuracy:.2f}| lr {current_lr}")

          # save model if it has the best accuracy
          if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            save_model(model=model, optimizer=optimizer, epoch=epoch, stats=stats, path=savepath, best=True)

      if epoch % SAVE_FREQ == 0:
          save_model(model=model, optimizer=optimizer, epoch=epoch, stats=stats, path=savepath, best=False)

      scheduler.step()
      current_lr = scheduler.get_last_lr()[0]


  return stats, model
  
# function to make loss smooth
def smooth(f, K=5):
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]]) 
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  
    return smooth_f


# functions to save model/stats and load them
import os

def save_model(model, optimizer, epoch, stats, path, best=False):    
    if(not os.path.exists(path + "/models")):
        os.makedirs(path + "/models")
    if best:
        savepath = path + '/' + f"models/best_model.pth"
    else:
        savepath = path + '/' + f"models/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath):
   
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats


# function to evaluate cnn model and plot results
def eval_plots(model, test_loader, criterion, device, stats):
  eval_accuracy, eval_loss = eval_model(model, test_loader, criterion, device)
  print(f"Classification accuracy: {round(eval_accuracy, 2)}%")

  epochs = np.array(stats["epoch"])-1
  train_loss = np.array(stats["full_train_loss"])
  valid_loss = np.array(stats["valid_loss"])
  #train_acc = np.array(stats["train_accuracy"])
  valid_acc = np.array(stats["valid_accuracy"])

  # show loss curves and accuracy
  fig, ax = plt.subplots(1, 2, figsize=(15, 5))

  ax[0].plot(train_loss, label="train")
  ax[0].plot(epochs, valid_loss, label="valid")
  ax[0].set_xlabel("Epoch")
  ax[0].set_ylabel("Loss")
  ax[0].set_title("Loss curves")
  ax[0].legend()
  ax[0].grid()

  #ax[1].plot(epochs, train_acc, label="train")
  ax[1].plot(epochs, valid_acc, label="valid")
  ax[1].set_xlabel("Epoch")
  ax[1].set_ylabel("Accuracy")
  ax[1].set_title("Accuracy curves")
  ax[1].legend()
  ax[1].grid()

  plt.show()



def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 13
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return