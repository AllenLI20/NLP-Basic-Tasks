import pandas as pd
import numpy as np
import math
import random


def logistic_compute_loss(pred, label,threshold=0.5):
    
    "compute loss and accuracy by given predicted prob and the true labels"
    
    pred, label=np.array(pred),np.array(label)
    
    loss = -sum(label*np.log(pred)+(1-label)*np.log(1-pred))/len(label)
    pred = (pred>threshold)*1
    accuracy = np.mean((pred>0.5)*1==label)
    
    return loss, accuracy


def logistic_valid(w,x,y,threshold=0.5):
    
    "compute loss and accuracy by given param, x and the true labels"
    
    logits= x.dot(w)
    preds=1/(1+np.exp(-logits))
    return logistic_compute_loss(preds, y,threshold)


def logistic_GD_one_step(w0, x, y,lr=0.01,threshold=0.5):
    
    "one step gredient decent function"
    
    """
    w0: the initial values for param
    x: data matrix, can include intercept
    y: observed labels
    lr: learning rate
    threshold: threshold to do classification
    """
    
    w0,x,y= np.array(w0),np.array(x),np.array(y)
    
    preds=1/(1+np.exp(-x.dot(w0)))

    app=[]
    for j in range(np.shape(x)[1]):
        app.append(np.mean(x[:,j]*(y-preds)))
    w=w0+lr*np.array(app)
    loss,accuracy=logistic_valid(w,x,y,threshold)

    return loss, accuracy, w


def logistic_SGD_one_step(w0, x, y,lr=0.01,threshold=0.5):
    
    "Stochastic Gredient Decent(one step)"
    
    w0,x,y= np.array(w0),np.array(x),np.array(y)
    preds=1/(1+np.exp(-x.dot(w0)))
    n=len(x)
    
    app=[]
    rand=np.random.randint(n)
    
    for j in range(np.shape(x)[1]):
        app.append(x[rand,j]*(y-preds)[rand])
    
    w=w0+lr*np.array(app)
    loss,accuracy=logistic_valid(w,x,y,threshold)
   
    return loss, accuracy, w


def logistic_mini_SGD_one_step(w0, x, y,lr=0.01,threshold=0.5,alpha=0.2):
    
    "Mini-Batch Gredient Decent(one step)"
    
    w0,x,y= np.array(w0),np.array(x),np.array(y)
    preds=1/(1+np.exp(-x.dot(w0)))
    n=len(x)
    
    app=[]
    rand=np.random.randint(n,size=math.ceil(alpha*n))
    
    for j in range(np.shape(x)[1]):
        app.append(np.mean(x[rand,j]*(y-preds)[rand]))
    
    w=w0+lr*np.array(app)
    loss,accuracy=logistic_valid(w,x,y,threshold)
    
    return loss, accuracy, w





def logistic_train(w0,x,y,shuffle='mini',lr=0.01,threshold=0.5,alpha=0.2,iter=1000,epsilon=1e-3):
    
    "logistic regression model training"
    
    w0,x,y=np.array(w0),np.array(x),np.array(y)
    
    valid_ind=np.random.randint(len(x),size=math.ceil(0.3*len(x)))
    valid_x=x[valid_ind,:]
    valid_y=y[valid_ind]
    
    train_x=np.delete(x,valid_ind,axis=0)
    train_y=np.delete(y,valid_ind,axis=0)
    
    w=w0[:]
    t=0
    loss_0=logistic_valid(w0,valid_x,valid_y,threshold)[0]
    if shuffle=='mini':
        while t<iter:
            loss_tr, accuracy_tr, w = logistic_mini_SGD_one_step(w, train_x, train_y,lr,threshold,alpha)
            loss_va,accuracy_va=logistic_valid(w,valid_x,valid_y,threshold)
            if abs(loss_0-loss_va)>epsilon:
                loss_0=loss_va
                t+=1
            else:
                break
        
    elif shuffle=='full':
        while t<iter:
            loss_tr, accuracy_tr, w = logistic_GD_one_step(w, train_x, train_y,lr,threshold)
            loss_va,accuracy_va=logistic_valid(w,valid_x,valid_y,threshold)
            if abs(loss_0-loss_va)>epsilon:
                loss_0=loss_va
                t+=1
            else:
                break
    
    elif shuffle=='stochastic':
         while t<iter:
            loss_tr, accuracy_tr, w = logistic_SGD_one_step(w, train_x, train_y,lr,threshold)
            loss_va,accuracy_va=logistic_valid(w,valid_x,valid_y,threshold)
            if abs(loss_0-loss_va)>epsilon:
                loss_0=loss_va
                t+=1
            else:
                break
                
    else:
        print("Not Found! Shuffle MUST be one of 'mini','full' and 'stochastic'!")
        loss_va,accuracy_va,w,t=None,None,None,None
        
    return loss_va,accuracy_va,w,t

data_mat=np.array(pd.read_csv("data_mat2.csv"))
train=pd.read_table('train.tsv',sep='\t')

y=train['Sentiment'].tolist()
y=np.array(y)
y0=(y==2)*1

x = np.concatenate((np.expand_dims(np.ones_like(data_mat[:,0],dtype=int),1),data_mat),axis=1)
del(data_mat)

logi=np.zeros([10,3])
i=0

while i<10:
    
    train_inds=random.sample(range(len(x)),np.int(np.floor(len(x)/2)))
    test_inds=list(set(range(len(x)))-set(train_inds))

    x_train=x[train_inds,:]
    x_test=x[test_inds,:]
    y0_train=y0[train_inds]
    y0_test=y0[test_inds]
    
    w0=np.random.randn(x.shape[1])
    log_stoch_param=logistic_train(w0,x_train,y0_train,'stochastic',lr=0.1,iter=100,epsilon=1e-5)
    log_mini_param=logistic_train(w0,x_train,y0_train,'mini',lr=0.1,iter=100,epsilon=1e-5)
    log_full_param=logistic_train(w0,x_train,y0_train,'full',lr=0.1,iter=100,epsilon=1e-5)
    
    logi[i,0]=logistic_valid(log_stoch_param[2],x_test,y0_test)[1]
    logi[i,1]=logistic_valid(log_mini_param[2],x_test,y0_test)[1]
    logi[i,2]=logistic_valid(log_full_param[2],x_test,y0_test)[1]
    
    print("已完成%d" % (i+1),end='\r')
    
    i+=1

with open("logistic2.out","a+") as f:
    for i in range(len(logi)):
        f.write(str(logi[i,0])+"\t"+str(logi[i,1])+"\t"+str(logi[i,2])+"\n")
