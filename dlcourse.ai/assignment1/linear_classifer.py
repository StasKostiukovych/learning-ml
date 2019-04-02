import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    
    predictions -= np.max(predictions)
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
    
    return probabilities
    
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    
    # TODO implement cross-entropy
    errorF = 0
    
    errorF = - np.log(probs[target_index])
    
    return errorF
    
    raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
   
    
    # TODO implement softmax with cross-entropy
    
    
    #nums = predictions.shape[0]
    dim = predictions.ndim
        
    
    if dim == 1:
        new_predictions = predictions - np.max(predictions)
        probs = np.exp(new_predictions) / np.sum(np.exp(new_predictions))
        loss = cross_entropy_loss(probs , target_index)
        dprediction = probs.copy()
        dprediction[target_index] -=1
        return loss, dprediction
        
    else:
        dprediction = np.zeros_like(predictions)
        num = predictions.shape[0]
        loss = 0
        
        for i in range(num):
            
            new_predictions = predictions[i] - np.max(predictions[i])
            probs = np.exp(new_predictions) / np.sum(np.exp(new_predictions))
            
            loss += cross_entropy_loss(probs , target_index[i])
            
            
            dprediction[i] = probs
            
            dprediction[i][target_index[i]] -=1
            
        dprediction /= num
        return loss/num, dprediction
            
    
    raise Exception("Not implemented!")
    

    
def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    #raise Exception("Not implemented!")

    grad = 2 * reg_strength * W
    
    loss = reg_strength * np.sum(W ** 2)

    return loss,  grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''

    predictions = np.dot(X, W)
    loss, dscores = softmax_with_cross_entropy(predictions, target_index)
    dW = X.T.dot(dscores) 
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        #if self.W is None:
        self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)

            batches_indices = np.array(np.array_split(shuffled_indices, sections))


            index = np.random.choice(batches_indices.shape[0])

            loss_softmax ,dW = linear_softmax(X[batches_indices[index]], self.W, y)
            loss_reg, grad_reg = l2_regularization(self.W, reg)


            self.W -= learning_rate*dW
            self.W -= learning_rate*grad_reg
            loss = loss_softmax + loss_reg
            loss_history.append(loss_softmax)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!

            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        #y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        #raise Exception("Not implemented!")

        y_pred = np.dot(X, self.W)
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred



                
                                                          

            

                
