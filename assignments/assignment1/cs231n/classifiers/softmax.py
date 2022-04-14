from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    test_num = X.shape[0]
    class_num = W.shape[1]

    res = np.dot(X,W)
    sub_max = np.max(res,axis=1)
    res -= np.reshape(sub_max,(sub_max.shape[0],-1))
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(test_num):
      sofm = np.exp(res[i][y[i]])/np.sum(np.exp(res[i]))
      loss += -np.sum(np.log(sofm))
      dW += np.reshape(X[i].T,(X.shape[1],-1))*np.reshape(np.exp(res[i])/np.sum(np.exp(res[i])),(-1,class_num))
      dW[:,y[i]] -= X[i,:].T
    
    loss /= test_num
    loss += reg * np.sum(W * W)
    dW = dW / test_num + 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    test_num = X.shape[0]
    class_num = W.shape[1]
    dem = X.shape[1]
  
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    f = np.dot(X,W)
    ef = np.exp(f)
    E = 1/np.sum(ef,axis=1)

    loss = -1.0/test_num * np.sum(np.log(E * ef[np.arange(test_num),y]))
    loss += reg * np.sum(W * W)

    E = np.reshape(E,(-1,E.shape[0]))
    dW = np.dot(E * X.T,ef)

    YN = (np.reshape(y,(test_num, -1)) == np.reshape(np.arange(class_num), (-1,class_num)))

    dW -= np.dot(X.T,YN)
    dW /= test_num
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW += 2 * reg * W

    return loss, dW
