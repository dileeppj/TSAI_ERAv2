# ERA V2 S5

### Abstract:
Train a model to classify handwritten digits. 

Training is done on MNIST dataset using PyTorch.
The MNIST dataset consists of grayscale images of handwritten digits (0 to 9) with a size of 28x28 pixels.

### Files:
**model.py**: Contains the definition of the neural network architecture used for training.

        The neural network contains 4 convolutional layers and 2 fully connected layers.
        The Summary of the model is as follows:
        ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
        ================================================================
                Conv2d-1           [-1, 32, 26, 26]             320
                Conv2d-2           [-1, 64, 24, 24]          18,496
                Conv2d-3          [-1, 128, 10, 10]          73,856
                Conv2d-4            [-1, 256, 8, 8]         295,168
                Linear-5                   [-1, 50]         204,850
                Linear-6                   [-1, 10]             510
        ================================================================
        Total params: 593,200
        Trainable params: 593,200
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.00
        Forward/backward pass size (MB): 0.67
        Params size (MB): 2.26
        Estimated Total Size (MB): 2.94
        ----------------------------------------------------------------

**utils.py**: Contains utility functions for data preprocessing, evaluation, plotting etc.

        The utils.py file contains the utility functions that can be used in the notebook.
        Functions in the utils.py:
            mountDrive():
            selectDevice():
            download_MNIST(train, transform):
            view_dataset(data_loader, title):
            GetCorrectPredCount(pPrediction, pLabels):
            train_test_model(network, device, train, train_loader, test, test_loader):
            viewAnalysis(train_losses, train_acc, test_losses, test_acc):

**S5.ipynb**: Notebook for training the model on the MNIST dataset.

        A common neural network pipeline looks like this:
            1. Prepare the data
            2. Build the model
            3. Train the model
            4. Analyze the model

        In this assignment we have 6 steps
            1. Importing the required libraries
            2. Preparing the Training and Testing Data
            3. Viewing the Data
            4. Initializing the model
            5. Training the Testing the model
            6. Analyze Training, Testing Loss and Accuracy
   
        
        Results
        After training the model, it achieves an accuracy of over 99.27% on the test set.