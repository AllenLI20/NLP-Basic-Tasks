from keras.utils import to_categorical
import copy
import math
import numpy as np


def gen_y_matrix(labels):
    return to_categorical(labels)

def soft_compute_loss(pred, y):
    
    pred, label=np.array(pred),np.array(y)
    
    N=pred.shape[0]
    C=pred.shape[1]
    
    loss=np.mean([-np.log(pred[i,:]).dot(y[i,:]) for i in range(N)])
    accuracy = np.mean(np.argmax(pred,1)==np.argmax(y,1))
    
    return loss, accuracy

def soft_predict(W,x):
    logits= x.dot(W) # N*C
    preds=np.exp(logits)
    for i in range(len(preds)):
        preds[i,:]=preds[i,:]/preds[i,:].sum()
    return preds

def soft_valid(W,x,y):
    
    return soft_compute_loss(soft_predict(W,x), y)


def soft_GD_one_step(W0, x, y,lr=0.01):
    
    W0,x,y= np.array(W0),np.array(x),np.array(y)
    preds=soft_predict(W0,x)

    app=x.T.dot(y-preds)/x.shape[0]
    W=W0+lr*app
    loss,accuracy=soft_valid(W,x,y)

    return loss, accuracy, W


def soft_SGD_one_step(W0, x, y,lr=0.01):
    
    W0,x,y= np.array(W0),np.array(x),np.array(y)
    preds=soft_predict(W0,x)
    n=len(x)
    rand=np.random.randint(n)
    
    app=np.expand_dims(x[rand,:],axis=1).dot(np.expand_dims((y-preds)[rand,:],axis=0))
    W=W0+lr*app
    loss,accuracy=soft_valid(W,x,y)

    return loss, accuracy, W



def soft_mini_SGD_one_step(W0, x, y,lr=0.01,alpha=0.2):
    
    W0,x,y= np.array(W0),np.array(x),np.array(y)
    preds=soft_predict(W0,x)
    N=len(x)
    n=math.ceil(alpha*N)
    
    rand=np.random.randint(N,size=n)
    app=x[rand,:].T.dot((y-preds)[rand,:])/n  
    W=W0+lr*app
    loss,accuracy=soft_valid(W,x,y)

    return loss, accuracy, W


def soft_train(W0,x,y,shuffle='mini',lr=0.01,alpha=0.2,iter=5000,epsilon=1e-4):
    
    W0,x,y=np.array(W0),np.array(x),np.array(y)
    
    valid_ind=np.random.randint(len(x),size=math.ceil(0.3*len(x)))
    valid_x=x[valid_ind,:]
    valid_y=y[valid_ind,:]
    
    train_x=np.delete(x,valid_ind,axis=0)
    train_y=np.delete(y,valid_ind,axis=0)
    
    W=copy.deepcopy(W0)
    t=0
    loss_0=soft_valid(W0,valid_x,valid_y)[0]
    
    if shuffle=='mini':
        while t<iter:
            loss_tr, accuracy_tr, W = soft_mini_SGD_one_step(W, train_x, train_y,lr,alpha)
            loss_va,accuracy_va=soft_valid(W,valid_x,valid_y)
            if abs(loss_0-loss_va)>epsilon:
                loss_0=loss_va
                t+=1
            else:
                break
        
    elif shuffle=='full':
        while t<iter:
            loss_tr, accuracy_tr, W = soft_GD_one_step(W, train_x, train_y,lr)
            loss_va,accuracy_va=soft_valid(W,valid_x,valid_y)
            if abs(loss_0-loss_va)>epsilon:
                loss_0=loss_va
                t+=1
            else:
                break
    
    elif shuffle=='stochastic':
         while t<iter:
            loss_tr, accuracy_tr, W = soft_SGD_one_step(W, train_x, train_y,lr)
            loss_va,accuracy_va=soft_valid(W,valid_x,valid_y)
            if abs(loss_0-loss_va)>epsilon:
                loss_0=loss_va
                t+=1
            else:
                break
                
    else:
        print("Not Found! Shuffle MUST be one of 'mini','full' and 'stochastic'!")
        loss_va,accuracy_va,W,t=None,None,None,None
        
    return loss_va,accuracy_va,W,t
