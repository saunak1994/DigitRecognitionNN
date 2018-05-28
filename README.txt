############################################
REQUIREMENTS
############################################

The python files are provided. To run these, the keras library along with tensorflow must be installed in the system. If anaconda is installed in the system, it can be executed from an anaconda prompt. I have also included the ipython notebook files(.ipynb) which can be run through jupyter notebook easily. 

Also required: Pandas library, numpy library, scikit_learn library. 

#############################################
COMMAND LINE
#############################################

To train the fully connected neural network:

python FCNNTrain.py 

This will train the network on the "optdigits.tra" file and keep training until the early stopping is triggered 
Also, it will save the trained parameters/weights in a file "FCNNWeights.h5"

To test the performance of the network:

python FCNNTest.py

This will load both the "optdigits.tra" and "optdigits.tes" file and the weights from "FCNNWeights.h5" and evaluate the network. 
This file will output the relevant data to the terminal. 


Similarly, for the convolutional neural network, to train:

python CNNTrain.py

This will train the network on the "optdigits.tra" file and keep training until the early stopping is triggered 
Also, it will save the trained parameters/weights in a file "CNNWeights.h5"

To test the performance of the network:

python CNNTest.py

This will load both the "optdigits.tra" and "optdigits.tes" file and the weights from "CNNWeights.h5" and evaluate the network. 
This file will output the relevant data to the terminal.



#############################################
EXPERIMENT WITH HYPERPARAMETERS              
#############################################

1. The layers can be added/removed or modified(units increased/activations changed/filter size changed etc.) in the model section of the train file between the *****MODEL BEGINS***** and *****END OF MODEL***** comments. This is for both the FCNNTrain.py and CNNTrain.py files. 

[ IMPORTANT NOTE: Any change made in the Model section of the Train file must be reflected in the model section of the corresponding Test file or the routine will throw an exception while testing. For example, If anything is changed in the model section of the CNNTrain.py, THAT SAME MODEL HAS TO BE COPIED TO THE MODEL SECTION OF THE CNNTest.py FILE. ]

2. To change the loss function, you have to change the loss attribute of the model.compile() statement immediately following the *****END OF MODEL***** line in both the files. Here, the optimizer is set to Stochastic Gradient descent. It can be changed by changing the corresponding attribute of the model.compile() function. 

3. models.optimizers.SGD() attributes can be changed if one wishes to change the learning rate, momentum rate etc. 