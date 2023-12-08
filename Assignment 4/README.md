
VisionSystems_Assignment04_convLSTM.ipymb contains the code for the LSTM implemented from scratch, and the training of it.

VisionSystems_Assignment04_extra.ipynb contains the code for the 3dCNN (extra task) and its training.


VisionSystems_Assignment04_cellGru.ipynb and VisionSystems_Assignment04_cellLstm.ipynb contains the code for the model with the GRUCell and the LSTMCell and its training. Unfortunately there is a bug/error in the code, such that the model does not learn. The code can run a training and an evaluation, but the loss does not decrease/the accuracy does not increase. Therefore the results of the training can not be interpreted (except that maybe a local optimum was found, which is very unlikely).

Because of this we added the notebook VisionSystems_Assignment04_GRU_LSTM.ipynb. This notebook contains code for models using LSTM and GRU directly. Here the training works fine (which makes a local optimum in the other cases highly unlikely). We commented on the results of this notebook (and comparing it to the other results.)
