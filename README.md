Implemented  Neural Network with one hidden layer and one output layer, trained the neural network using backpropagation. 

Training and testing files are in ARFF format

Code is intended for binary classification problems.

All of the attributes are numeric.

The number of units in the hidden layer is equal to the number of input units.

For training the neural network, used n-fold stratified cross validation.

Used sigmoid activation function and trained using stochastic gradient descent.

Randomly set initial weights for all units including bias in the range (-1,1).

Used a threshold value of 0.5. If the sigmoidal output was less than 0.5, took the prediction to be the class listed first in the ARFF file in the class attributes section; else took the prediction to be the class listed second in the ARFF file

The program is callable from command line as follows: 
"neuralnet trainfile num_folds learning_rate num_epochs" 
