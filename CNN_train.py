import torch
from torch import optim,nn
import visdom
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from CNN_geneset import CNN_Geneset
from geneset import Geneset
from CNN import Lenet
from  ae import AE

batchsz = 1
lr = 9e-5
epochs = 100

#device = torch.device('cuda')
torch.manual_seed(1234)

#train_gene = CNN_Geneset('data',100,mode='train')
all_gene = CNN_Geneset('data',100,mode='all')
#test_gene = Geneset('data',100,mode='test')
#train_loader = DataLoader(train_gene,batch_size=batchsz,shuffle=True)
all_loader = DataLoader(all_gene,batch_size=batchsz)
#test_loader = DataLoader(test_gene,batch_size=batchsz)

encoder_gene = Geneset('data',100,mode='all')
encoder_loader = DataLoader(encoder_gene,batch_size=301)


def main():
    x,label = iter(all_loader).next()
    print('x:',x.shape, 'label:',label.shape)

    model = Lenet()
    criteon1 = nn.CrossEntropyLoss()
    criteon2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    print(model)
    for epoch in range(500):
        model.train()
        for batchidx, (x, label) in enumerate(all_loader):
            # x: [b,1,100,100]
            # label: [b]
            #print(label)
            if int(label) == -1:
                print(label)
                logits = model(x,-1,True)
                print(logits.shape)
                print(x.shape)
                loss = 0.00001*criteon2(logits, x)
            else:
                logits = model(x,label, True)
                loss = criteon1(logits, label)

            # logits: [b, 10]
            # label: [b]
            # loss: tensor scalar

            # print(logits)
            # print(label)


            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, loss.item())

        model.eval()
        with torch.no_grad():
            # test

            # 使用conv与fc_unit1部分
            encoder = torch.randn(301, 184)
            label = torch.randn(301)
            for x,y in encoder_loader:
                #x,y = iter(all_loader).next()
                with torch.no_grad():
                    x_encoder = model(x,y,False)
                    # label.append(y)
                    # encoder.append(x_encoder)
                    label = y
                    encoder = x_encoder
            encoder = encoder.numpy()
            label = label.numpy()
            #print(encoder.shape,label.shape)

            from sklearn.cluster import SpectralClustering
            from sklearn.metrics import adjusted_rand_score
            from sklearn.metrics import normalized_mutual_info_score
            from sklearn.metrics.pairwise import cosine_similarity
            #simatrix = np.arange(len(encoder) ** 2, dtype=float).reshape(len(encoder), -1)
            simatrix = 0.5 * cosine_similarity(encoder) + 0.5
            SC = SpectralClustering(affinity='precomputed', assign_labels='discretize')#, random_state=100)
            label1 = SC.fit_predict(simatrix)
            print('epoch:',epoch)
            print('label:',label.shape)
            ARI = adjusted_rand_score(label, label1)
            NMI = normalized_mutual_info_score(label, label1)
            # if ARI > 0.9:
            #     print("谱聚类：ARI", ARI)
            #
            # if NMI > 0.9:
            #     print("谱聚类：NMI", NMI)
            print("谱聚类：ARI", ARI)
            print("谱聚类：NMI", NMI)


            # k-means 聚类
            from sklearn.metrics import adjusted_rand_score
            from sklearn.metrics import normalized_mutual_info_score
            from sklearn.cluster import KMeans
            from sklearn import metrics

            label1 = KMeans(n_clusters=11).fit_predict(encoder)
            ARI = adjusted_rand_score(label, label1)
            NMI = normalized_mutual_info_score(label, label1)
            # if ARI > 0.9:
            #     print("k-means：ARI", ARI)
            #
            # if NMI > 0.9:
            #     print("k-means：NMI", NMI)
            print("k-means：ARI", ARI)
            print("k-means：NMI", NMI)





if __name__ == '__main__':
    main()