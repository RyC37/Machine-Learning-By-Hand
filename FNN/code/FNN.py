import numpy as np

class Network(object):
    """ 
    This is a class for neural network (NN). 
      
    Attributes: 
        num_layers (int): The number of layers of NN. 
        sizes (array): The structure of NN, represent number of neuron in each layer. 
        biases: The bias of NN.
        weights: The weights of NN.
    """
    def __init__(self, sizes):
        '''Initialize attributes of the neural network based on network structure.
        
        Parameters:
        sizes (array): The structure of neural network.
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Input layer has no bias, so start with the 2nd layer (sizes[1:])
        # The shape of bias coef in each layer is a vertical vector
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # The shape of weights between each two layers are:
        # The row number is the size of latter layer
        # The column number is the size of previous number
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1],sizes[1:])]
        
    def sigmoid(self,z):
        '''Sigmoid transformation
        
        Parameters:
        z (array): The vector to be converted, elementwise.
        '''
        return 1.0/(1.0 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        '''The derivative of sigmoid function
        
        Parameters:
        z (array): The vector to be converted, elementwise.
        '''
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''This function implements stochastic gradient descent
        
        Parameters:
        training_data (zip/list): Training data, containing x and y.
        epochs (int): The number of training iterations.
        mini_batch_size (int): The sample size for each batch in SGD.
        eta (float): The learning rate.
        test_data (zip/list): Test data, containing x and y.
        '''
        training_data = list(training_data)
        test_data = list(test_data)
        n = len(training_data)
        for i in range(epochs):
            # Shuffle to make the processes 'stochastic'
            np.random.shuffle(training_data)
            # Split data into mini batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            # Update weights and biases by mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # Test prediction performance if test data available
            if test_data:
                eval_res = self.evaluate(test_data)
                print('Epotch %d: %f' % (i+1, eval_res))
            else:
                print('Epotch %d complete' % i+1)
    
    def update_mini_batch(self, mini_batch, eta):
        '''Update model's weights and biases using the results of 
        backpropagate algorithm, for each mini batch of data
        
        Parameters:
        mini_batch (list): One batch of training data.
        eta (float): The learning rate.
        '''
        # Initiate nabla weights and nabla biases, 
        # in a way they have the same shape with weights and biases
        # fill with 0
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Update nabla weights and nabla biases
        for x, y in mini_batch:
            # Use backpropagate to calculate the derivative of cost function of bias,
            # and the derivative of the cost function of weight. 
            # And take the average.
            delta_nabla_b, delta_nabla_w = self.back_prop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        mini_batch_size = len(mini_batch)
        # Average the sum of derivatives and update weights and biases
        self.biases = [b - eta/mini_batch_size * db for b, db in zip(self.biases, nabla_b)]
        self.weights = [w - eta/mini_batch_size * dw for w, dw in zip(self.weights, nabla_w)]
    
    def feed_forward(self, x):
        '''Calculate the activiations and zs for the NN
        
        Parameters:
        x (array): Input value.
        
        Output:
        activations (array): The activiation value of each neuron.
        zs (array): The z value of each neuron.
        '''
        activations = [x]
        zs = []
        a = np.array(x)
            
        for i in range(self.num_layers-1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = self.sigmoid(z)
            zs.append(z)
            activations.append(a)
        return activations, zs
    
    def back_prop(self, x, y):
        '''Calculate the derivative of the cost function of weights
        and the derivative of the cost function of bias
        
        Parameters:
        x (array): Features
        y (array): Labels
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feed forward to calculate z and activiation
        activations, zs = self.feed_forward(x)
        # Updata weight and bias of last layer
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Update weight and bias of previous layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-(l-1)].transpose(), delta) * sp
            # Update weights and bias
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-(l+1)].transpose())
        return nabla_b, nabla_w
        
    def cost_derivative(self, activiation_output, y):
        '''Calculate the derivative of cost function
        
        Parameters:
        activiation_output (array): The predicted result
        y (array): The label.
        
        Output:
        Derivative of cost function in array.
        '''
        return activiation_output - y
    
    def evaluate(self, test_data):
        '''Evaluate the precition accuracy
        
        Parameters:
        test_data (array)
        '''
        count = 0
        n = len(test_data)
        for x, y in test_data:
            activations, zs = self.feed_forward(x)
            y_pred = np.argmax(activations[-1])
            if y_pred == y:
                count += 1
        return 1.0*count/n