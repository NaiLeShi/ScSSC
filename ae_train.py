import torch
from torch import optim,nn
import visdom
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from ae_geneset import Geneset
from ae import AE

batchsz = 10
lr = 9e-5
epochs = 100

#device = torch.device('cuda')
torch.manual_seed(1234)
np.random.seed(1234)

train_gene = Geneset('data',100,mode='train')
all_gene = Geneset('data',100,mode='all')
test_gene = Geneset('data',100,mode='test')
train_loader = DataLoader(train_gene,batch_size=batchsz,shuffle=True)
all_loader = DataLoader(all_gene,batch_size=batchsz)
test_loader = DataLoader(test_gene,batch_size=batchsz)

encoder_gene = Geneset('data',100,mode='all')
encoder_loader = DataLoader(all_gene,batch_size=301)


def main():

    model = AE()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criteon = nn.MSELoss()
    #criteon = nn.CrossEntropyLoss()
    print(model)

    #for epoch in range(epochs):
    for epoch in range(100):
        for step,(x,y) in enumerate(all_loader):
            # x: [b,1,100,100]
            x_hat = model(x,False)
            #print('x shape:',x.shape,'x_hat shape:',x_hat.shape)
            loss = criteon(x_hat,x)
            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch,'loss',loss.item())


        # 使用encode部分
        # encoder = []
        # label = []
        encoder = torch.randn(301,100)
        label = torch.randn(301,2)
        #tmp = torch.randn(2, 1, 100, 100)
        for x,y in encoder_loader:
            #x,y = iter(all_loader).next()
            with torch.no_grad():
                x_encoder = model(x,True)
                # label.append(y)
                # encoder.append(x_encoder)
                label = y
                encoder = x_encoder
        encoder =encoder.numpy()
        label = label.numpy()
        #print(encoder.shape,label.shape)

        #  谱聚类
        if epoch==0:
            pred = torch.zeros(301, 101)
            pred = pred.numpy()
        pred.T[0][:] = label.T[0]
        print(pred)

        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import adjusted_rand_score
        from sklearn.metrics import normalized_mutual_info_score
        from sklearn.metrics.pairwise import cosine_similarity
        simatrix = 0.5 * cosine_similarity(encoder) + 0.5
        SC = SpectralClustering(affinity='precomputed', assign_labels='discretize', random_state=100)
        label1 = SC.fit_predict(simatrix)

        pred.T[epoch+1][:] = label1[:]


        print('pred:',pred.shape)
        print(pred)
        if epoch == 99:
            pd.DataFrame(pred).to_csv("pred.csv", index=False, sep=',')





if __name__ == '__main__':
    main()