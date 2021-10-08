import pandas as pd
import numpy as np
import matplotlib
import sys
import time
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)

class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        #raise NotImplementedError
        self.scaling_type = "minmaxnorm" 

    def __call__(self,features, is_train=False):
        """
          TODO
        """
        if self.scaling_type == "minmaxnorm":
            return np.apply_along_axis(self.minmaxnorm,0,features)
        else:
            raise NotImplementedError
              
    def minmaxnorm(self,data_in):
        """
          data_in: numpy array of features
        """
        minimum = np.min(data_in)
        maximum = np.max(data_in)
        return (data_in-minimum)/(maximum-minimum)        


def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''

    #raise NotImplementedError
    data = pd.read_csv(csv_path)
    data[['Year','Month','Day']] = data.acq_date.str.split("-",expand=True)
    data = data.drop(columns=['Unnamed: 0','Month','Day','acq_date','instrument', 'version','frp'])
    data['satellite'] =  data['satellite'].str.replace('Terra','0')
    data['satellite'] =  data['satellite'].str.replace('Aqua','1')
    data['satellite'] =  pd.to_numeric(data['satellite'])

    data['daynight'] =  data['daynight'].str.replace('D','0')
    data['daynight'] =  data['daynight'].str.replace('N','1')
    data['daynight'] = pd.to_numeric(data['daynight'])
    
    data['Year'] = data['Year'].str.replace('2019','19')
    data['Year'] = data['Year'].str.replace('2020','20')
    data['Year'] = pd.to_numeric(data['Year'])

    #scaling
    #scaler = Scaler()
    if scaler:
        np_data = scaler(data.to_numpy())
    else:
        np_data = data.to_numpy()
        
    #add bias 
    np_data =  np.hstack((np_data, np.ones((np_data.shape[0], 1), dtype=np_data.dtype)))
    #NOTE: Uncomment following line and comment basis code to run defult version wothout basis
    #return np_data
    
    #>>>>>>>>>>>>>>>>RADIAL BASIS CODE START >>>>>>>>>>>>>
    #num_means = 50
    #means = np.stack([np.random.uniform(low=-1, high=1, size=11) for i in np.arange(num_means)])
    #radial_data = np.zeros((train_features.shape[0],means.shape[0]))
    #for i in range(radial_data.shape[0]):
    #    for j in range(radial_data.shape[1]):
    #        radial_data[i,j] = np.exp(-np.linalg.norm(train_features[i] -means[j])**2)
    #return radial_data    
    #>>>>>>>>>>>>>>>>RADIAL BASIS CODE END >>>>>>>>>>>>>

    #>>>>>>>>>>>>>>>>POLYNOMIAL BASIS CODE START >>>>>>>>>>>>>
    #poly_data = np_data
    #Second degree polynomial computation 
    #poly_data = np.einsum('ki,kj->kij',np_data , np_data)
    #Thirs degree polynomial computation 
    poly_data = np.einsum('ki,kj,kl->kijl',np_data , np_data,np_data)
    poly_data = poly_data.reshape(poly_data.shape[0],-1)
    #>>>>>>>>>>>>>>>>POLYNOMIAL BASIS CODE END >>>>>>>>>>>>>
    return poly_data
   



def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    #raise NotImplementedError
    data = pd.read_csv(csv_path)
    return data['frp'].to_numpy()
     

def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 5d
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''
    soln = np.matmul(np.linalg.inv(np.matmul(feature_matrix.T,feature_matrix) +\
                                   np.eye(feature_matrix.shape[1])*C ), 
                     np.matmul(feature_matrix.T,targets) )
    #raise NotImplementedError
    return soln

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''
    return np.matmul(feature_matrix,weights)
    #raise NotImplementedError

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    preds = get_predictions(feature_matrix, weights)
    loss = np.matmul(preds-targets,preds-targets)
    return loss/feature_matrix.shape[0]
    #raise NotImplementedError

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    return np.matmul(weights,weights)
    #raise NotImplementedError

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''
    return mse_loss(feature_matrix, weights, targets) +\
           C*l2_regularizer(weights)
    #raise NotImplementedError

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    grad = 2*C*weights -(2/feature_matrix.shape[0]) * (np.matmul(feature_matrix.T,\
                              targets - np.matmul(feature_matrix,weights)))
    return grad
    #raise NotImplementedError

def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''    
    idx = np.random.randint(0,feature_matrix.shape[0],batch_size)
    return feature_matrix[idx], targets[idx]
    #raise NotImplementedError
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    return np.ones(n)
    #raise NotImplementedError

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    
    return weights- lr*gradients
    #raise NotImplementedError

def early_stopping(arg_1=None, arg_2=None, arg_3=None, arg_n=None):
    # allowed to modify argument list as per your need
    # return True or False
    raise NotImplementedError
    

def plot_trainsize_losses():
    '''
    Description:
    plot losses on the development set instances as a function of training set size 
    '''

    '''
    Arguments:
    # you are allowed to change the argument list any way you like 
    '''  
    scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('data/train.csv',True,scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',False,scaler), get_targets('data/dev.csv')
    
    training_sizes = [5000, 10000, 15000, 20000, 25001]
    dev_losses = []
    
    for train_size in training_sizes:
        a_solution = analytical_solution(train_features[:train_size], train_targets[:train_size], C=1e-8)
        print('evaluating analytical_solution...')
        dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
        dev_losses.append(dev_loss)
        train_loss=do_evaluation(train_features[:train_size], train_targets[:train_size], a_solution)
        print('analytical_solution train size: {} \t train loss: {}, dev_loss: {} '.format(train_size,train_loss, dev_loss))

    plt.plot(training_sizes,dev_losses)
    plt. xticks(training_sizes)
    plt.ylabel("dev set loss",fontsize=20)
    plt.xlabel("training data size",fontsize=20)
    plt.show()
    return training_sizes,dev_losses
    #raise NotImplementedError


def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights

    a sample code is as follows -- 
    '''
    n = train_feature_matrix.shape[-1]
    weights = initialize_weights(n)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    
    #>>>>>>>>>>>>>EARLY STOPPING CODE >>>>>>>>>>>
    best_dev_loss = dev_loss
    best_weights = weights
    num_bad_epochs = 0
    patience_count = 2000
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))

        '''
        implement early stopping etc. to improve performance.
        '''
        #>>>>>>>>>>>>>MORE EARLY STOPPING CODE >>>>>>>>>>>
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            num_bad_epochs = 0
            best_weights = weights
        else:
            num_bad_epochs = num_bad_epochs+1
            #print(num_bad_epochs)
            if num_bad_epochs>patience_count:
                return best_weights
            
    weights = best_weights
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

if __name__ == '__main__':
    scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('data/train.csv',True,scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',False,scaler), get_targets('data/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=1e-8)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.01,#1.0,
                        C=1e-8,#0.0,
                        batch_size=256,#32,
                        max_steps=400000,#2000000,
                        eval_steps=5)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    #plot_trainsize_losses()       
