# importing all libraries

import numpy as np
import time
from scipy.special import expit




class MyNeuralNetwork():
    """
        Class for the multilayer perceptron

        self._N_layers                        >  no. of layers in neural network
        self._Layer_sizes                     > individual neuron count for each layer

        self.weight_updates = 0               > no. of weight updates needed
        self._activation = activation         > activation function to apply 
        self._learning_rate = learning_rate   > learning rate or step size(neta)
        self._weight_init = weight_init       > method to initialize weights  
        self._batch_size = batch_size         > batch size for training
        self._num_epochs = num_epochs         > no. of iteration for which the training will be done  
        self._optimizer = optimizers          > optimizer to use for training, default = Gradient descent  
       
        self._params = {}                     > weights for neural network 
        self._biases = {}                     > biasses for neural network  
       
        self._moments = {}                    > momentum values for weights   
        self._learning_rate_constant = {}     > scale factor of adaptive learning rates for weights 
        self._moments_bias = {}               > weights for bias parameters 
        self._learning_rate_constant_bias = {}   > scale factor of adaptive learning rates for biasses 
       
        self._beta = 0.9                      > beta parameter for momentum and NAG calculations 
        self._eps = 0.0000001                 > epsilon parameter for AdaGrad and RMS calculations 
        self._gamma = 0.9                     > gamma parameter for Adam and RMS calculations  
    """

    def __init__(self, N_layers, Layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs,optimizers="GD"):
        self._N_layers = N_layers
        self._Layer_sizes = Layer_sizes

        self.weight_updates = 0               
        self._activation = activation
        self._learning_rate = learning_rate
        self._weight_init = weight_init
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._optimizer = optimizers
       
        self._params = {}                     
        self._biases = {}                       
       
        self._moments = {}
        self._learning_rate_constant = {}
        self._moments_bias = {}
        self._learning_rate_constant_bias = {}  
       
        self._beta = 0.9                       
        self._eps = 0.0000001                  
        self._gamma = 0.9                       
        
        # intializing the weights
        self.initialization()

    
    # initilization
    def initialization(self):
        for i in range(1, self._N_layers):
            name = "W" + str(i)
            layer_weights = self.random_initialization(self._Layer_sizes[i], self._Layer_sizes[i-1])
            layer_bias =self.random_initialization(1,self._Layer_sizes[i])
            
            #  initializing weights
            self._params[name] = layer_weights

            #  initializing bias
            self._biases[name] = layer_bias

            #  initializing momentums and learning rate
            #  scale factor for all parameters of net
            self._moments[name] = np.zeros((self._Layer_sizes[i], self._Layer_sizes[i-1]))
            self._learning_rate_constant[name] = np.zeros((self._Layer_sizes[i], self._Layer_sizes[i-1]))

            self._moments_bias[name] = np.zeros((1,self._Layer_sizes[i] ))
            self._learning_rate_constant_bias[name] = np.zeros((1,self._Layer_sizes[i]))


    # helper function for random intialization
    def random_initialization(self, w_rows, w_cols, alpha = 0.01):
        np.random.seed(1)
        return np.random.randn(w_rows, w_cols) * alpha


    #  relu activation
    def relu(self, X):
        return X * (X > 0)


    #  sigmoid activation
    def sigmoid(self, X):
        return expit(X)


    #  softmax activation
    def softmax(self, X):
        f = np.exp(X - np.max(X))  # shift values
        return f / f.sum(axis=0)


    #  tanh activation
    def tanh(self, X):
        return np.tanh(X)


    # derivatives of activation functions
    def reluPrime(self, X):
        return np.matmul(np.identity(X.shape[0]), (X > 0))

    def sigmoidPrime(self, X):
        return self.sigmoid(X)*(1-self.sigmoid(X))

    def softmaxPrime(self, X):
        exp_element = np.exp(X - X.max())
        return exp_element/np.sum(exp_element, axis = 0)*(1 - exp_element/np.sum(exp_element, axis = 0))

    def tanhPrime(self, X):
        return (1 - self.tanh(X)**2)



    # wrapper function for applying Activation functions 
    def activation(self, X, alpha = 0.01, final_ = False):

        if (final_):
            # if the layer is ouptut layer directly apply softmax 
            return (self.softmax(X))

        if self._activation == "relu":
            return (self.relu(X))
        
        elif self._activation == "sigmoid":
            return (self.sigmoid(X))
        
        elif self._activation == "softmax":
            return (self.softmax(X))

        else:
            return (self.tanh(X))
 

    # wrapper function for applying derivative of Activation functions
    def derivative(self, X, alpha = 0.01, final_ = False):

        if (final_):
            return (self.softmaxPrime(X))

        if self._activation == "relu":
            return (self.reluPrime(X))

        elif self._activation == "sigmoid":
            return (self.sigmoidPrime(X))
        
        elif self._activation == "softmax":
            return (self.softmaxPrime(X))

        elif self._activation == "tanh":
            return (self.tanhPrime(X))

        return self.relu(X)    

    
    # function to return the accuracy score 
    def score(self, x_val, y_val):
        predictions = []

        output = self.forward(x_val)
        pred = np.argmax(output, axis = 0)
        y_val_new = np.argmax(y_val, axis = 1)
        
        # accuracy
        acc = 0
        for i in range(len(pred)):
            if (pred[i] == y_val_new[i]):
                acc += 1
        
        return acc/len(pred)


    # function for making prediction on x_val
    def predict(self, x_val):
        output = self.forward(x_val)
        return np.argmax(output, axis = 0)
    

    # function to get classwise probability for inputted samples
    def predict_proba(self, x_val):
        return self.forward(x_val)


    # function for fitting the train data and 
    # learning the paramaters of the network
    def fit(self, x_train, y_train, x_val, y_val):

        val_epoch = []        # list to store validation accuracy for each epoch
        train_epoch = []      # list to store train accuracy for each epoch  
        start_time = time.time()

        # total no. of batches 
        num_batches = int(np.ceil(np.divide(x_train.shape[0], self._batch_size)))

        # iteratiing for epoch no. of times
        for iteration in range(self._num_epochs):

            # iterating for every batch
            for b in range(num_batches):
                 
                batch_inds = np.arange(b*self._batch_size, min((b+1)*self._batch_size, x_train.shape[0]))
                
                # forward propagation
                output = self.forward(x_train[batch_inds])
                
                # backward propagation
                changes_to_w,changes_to_bias = self.backward(y_train[batch_inds], output)
                
                # updating weights and bias using the gradients calculated
                self.update_network_parameters(changes_to_w,changes_to_bias)

            # validation accuracy for current epoch 
            accuracy = self.score(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Validation Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy*100
            ))

            val_epoch.append(accuracy)
            
            # train accuracy for current epoch
            train_epoch.append(self.score(x_train, y_train))

        # returning lists of epochwise validation and train accuracies 
        return val_epoch, train_epoch


    # function for forward propagation
    def forward(self, x):
        params = self._params
        bias = self._biases

        params["Y0"] = x.T
      
        # updating parameters if optimizer is NAG
        # by adding momentum before finding gradient
        if self._optimizer=="NAG":
          for key in self._moments.keys():
            params[key] = params[key] + self._beta*self._moments[key] 
            bias[key]   = bias[key] + self._beta*self._moments_bias[key] 
 

        # fowarding output of previous layer to next layer
        for i in range(1, self._N_layers - 1):
            name1 = "V" + str(i)
            # print(np.dot(params["W" + str(i)], params["Y" + str(i-1)]).shape," bias shape : " , bias["W" + str(i)].shape)
            params[name1] = (np.dot(params["W" + str(i)], params["Y" + str(i-1)]).T + bias["W" + str(i)]).T
            name2 = "Y" + str(i)
            params[name2] = self.activation(params[name1])
            
        params["V" + str(self._N_layers - 1)] = (np.dot(params["W" + str(self._N_layers - 1)], params["Y" + str(self._N_layers - 2)]).T  + bias["W" + str(self._N_layers - 1)]).T
        params["Y" + str(self._N_layers - 1)] = self.activation(params["V" + str(self._N_layers - 1)], final_=True)
        
        # returning output layer's value
        return params["Y" + str(self._N_layers - 1)]


    # function for backward propagation
    def backward(self, y_train, output):
              
        params = self._params
        update_w = {}

        bias = self._biases
        update_bias = {}  

        # error and derivative
        error = 2 * (output - y_train.T) * self.derivative(params['V' +  str(self._N_layers - 1)], final_ = True )
        update_w['W' + str(self._N_layers - 1)] = np.dot(error, params["Y" + str(self._N_layers - 2)].T)
        update_bias['W' + str(self._N_layers - 1)] = np.mean(error,axis = 1)
          
        # propagating the loss backward till input layer
        for i in range(self._N_layers - 2, 0, -1):

            error = np.dot(params['W' + str(i+1)].T, error) * self.derivative(params['V' + str(i)])
            update_w['W' + str(i)] = np.dot(error, params["Y" + str(i-1)].T)   
            update_bias['W' + str(i)] = np.mean(error ,axis = 1)
         
        return update_w,update_bias


    # function for updating all weights and bias of network
    def update_network_parameters(self, changes_to_w,changes_to_bias):
        """
            key gradient descent equation-
                  w(t+1) = w(t) - η * ∇J(w(t)
        """

        # updating no. of weight update required
        self.weight_updates += 1

        for key, value in changes_to_w.items(): 
            # updating the momentum values          
            self._moments[key] *= self._beta
            self._moments[key] -= self._learning_rate * value 
            
            self._moments_bias[key] *= self._beta
            self._moments_bias[key] -= self._learning_rate * changes_to_bias[key] 
            
            
            # weight updates depending on the optimizers type
            if self._optimizer=='GD':
              self._params[key] -= self._learning_rate * value 
              self._biases[key] -= self._learning_rate * changes_to_bias[key] 
                  
                  
            elif self._optimizer=='momentum' or self._optimizer=='NAG':
              self._params[key] += self._moments[key]
              self._biases[key] -= self._moments_bias[key] 
                

            elif self._optimizer=='AdaGrad':
              # updating learing rate scale factor for weights
              self._learning_rate_constant[key] += value * value
              
              new_learning_rate = self._learning_rate  
              new_learning_rate *=  np.power( self._eps + self._learning_rate_constant[key] , -0.5) 
                          
              # updating learing rate scale factor for weights
              self._learning_rate_constant_bias[key] += changes_to_bias[key] * changes_to_bias[key]
              
              new_learning_rate_bias = self._learning_rate  
              new_learning_rate_bias *=  np.power( self._eps + self._learning_rate_constant_bias[key] , -0.5) 
                
              # final update with adapted learning rate  
              self._params[key] -=   new_learning_rate * value
              self._biases[key] -=   new_learning_rate_bias * changes_to_bias[key]


            elif self._optimizer=='RMS':
              # updating learing rate scale factor for weights
              self._learning_rate_constant[key] *= self._gamma
              self._learning_rate_constant[key] += (1-self._gamma)*value*value
              
              new_learning_rate = self._learning_rate 
              new_learning_rate *=  np.power( self._eps + self._learning_rate_constant[key] , -0.5) 

              # updating learing rate scale factor for weights  
              self._learning_rate_constant_bias[key] *= self._gamma
              self._learning_rate_constant_bias[key] += (1-self._gamma)*changes_to_bias[key] * changes_to_bias[key]
              
              new_learning_rate_bias = self._learning_rate 
              new_learning_rate_bias *=  np.power( self._eps + self._learning_rate_constant_bias[key] , -0.5) 

              # final update with adapted learning rate  
              self._params[key] -=   new_learning_rate * value 
              self._biases[key] -=   new_learning_rate_bias * changes_to_bias[key] 

            
            elif self._optimizer=='Adam': 
              unbiased_moment = self._moments[key] / (self._eps + 1 - self._beta**self.weight_updates)
              
              # updating learing rate scale factor for weights  
              self._learning_rate_constant[key] *= self._gamma
              self._learning_rate_constant[key] += (1-self._gamma)*value*value
                  
              new_learning_rate =  self._learning_rate_constant[key]/( self._eps + 1 - self._gamma**self.weight_updates)
             
              new_learning_rate =  np.power(self._eps + new_learning_rate , -0.5)
              new_learning_rate *= self._learning_rate
              
              #########
              # final weight update with adapted learning rate and momuntum
              self._params[key] +=   new_learning_rate * unbiased_moment 
              #########


              unbiased_moment_bias = self._moments_bias[key] / (self._eps + 1 - self._beta**self.weight_updates)
    
              # updating learing rate scale factor for weights  
              self._learning_rate_constant_bias[key] *= self._gamma
              self._learning_rate_constant_bias[key] += (1-self._gamma)*changes_to_bias[key] * changes_to_bias[key]
                            
              new_learning_rate_bias =  self._learning_rate_constant_bias[key]/( self._eps + 1 - self._gamma**self.weight_updates)
             
              new_learning_rate_bias =  np.power(self._eps + new_learning_rate_bias , -0.5)
              new_learning_rate_bias *= self._learning_rate
              
              #########
              # final bias update with adapted learning rate and momentum
              self._biases[key] +=   new_learning_rate_bias * unbiased_moment_bias 
              #########

            
