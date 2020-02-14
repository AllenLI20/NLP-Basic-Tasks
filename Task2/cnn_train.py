# library

import gensim
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

data=pd.read_table('train.tsv',sep='\t')
# texts=data['Phrase'].tolist()
data_y=data["Sentiment"]
data_y=np.array(data_y)
N=len(data_y)

del(data)

words_ls=pd.read_table('words_ls.txt',header=None)[0]
words_ls=[eval(words) for words in words_ls]

word_maxlen=0
for words in words_ls:
    word_maxlen=max(word_maxlen,len(words))

word2vec = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

# Hyper Parameters
LEN_SEN = word_maxlen
VEC_LEN = 300

# CNN Architecture
class CNN(nn.Module):
    def __init__(self,n_window = 3,vec_len=300):
        self.window = n_window
        self.vec_len = vec_len
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=(self.window,self.vec_len),              # filter size
                stride=(1,self.vec_len),                   # filter movement/step
                
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=(LEN_SEN-self.window+1,1),stride=(LEN_SEN-self.window+1,1)),   
        )

        self.out = nn.Linear(16, 5)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization
    
# Word Vectors Generation (to tensor)
def wv_to_tensor(inds,t_height=word_maxlen,v_length=VEC_LEN,WV=word2vec):
    l=len(inds)
    wordvec=np.zeros([l,1,t_height,v_length]) # dimension 
    for i in range(l):
        words=words_ls[inds[i]]
        n=len(words)
        if n>0: 
            try:
                wordvec[i,0,:n,:]=WV[words]
            except KeyError:
                for h in range(n):
                    try:
                        wordvec[i,0,h,:]=WV[words[h]].reshape(1,v_length)
                    except KeyError:
                        wordvec[i,0,h,:]=np.random.randn(1,v_length)/10
                        # 到此 wordvec的type还是np.array, need to convert to torch.tensor
    return torch.from_numpy(wordvec).to(torch.float32)



def train(EPOCH = 2 ,BATCH_SIZE = 200,LR = 0.01,n_window = 3,wv=word2vec):
    
    cnn = CNN(n_window)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

    train_inds=random.sample(range(N),np.int(np.floor(N*0.8)))
    test_inds=list(set(range(N))-set(train_inds))

    mat=np.concatenate((np.arange(N).reshape(N,1),data_y.reshape(N,1)),axis=1)
    train_loader = Data.DataLoader(dataset=mat[train_inds,:], batch_size=BATCH_SIZE, shuffle=True)
    test_x=wv_to_tensor(inds=mat[test_inds,0],WV=wv)
    test_y=torch.from_numpy(mat[test_inds,1])

    for epoch in range(EPOCH):
        for step, batch_data in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            b_x_ind = batch_data[:,0]  # batch x
            b_y = batch_data[:,1]   # batch y
            b_x = wv_to_tensor(inds=b_x_ind,WV=wv)

            output = cnn(b_x)[0]               # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if step % 100 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))
                print('Epoch: ', epoch, '| Step: ', step, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)
    return (cnn,accuracy)


cnn10,acc10=train(n_window=10,wv=word2vec)
print('test accuracy: %.2f' % accuracy)
torch.save(cnn10, 'cnn-w10.pkl')  # save entire net