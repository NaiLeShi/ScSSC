import torch
import os,glob
import random,csv

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

class AEGeneset(Dataset):
    def __init__(self,root,resize,mode):
        super(Dataset,self).__init__()
        self.root = root
        self.resize = resize

        self.name2label = {} # ”sq...“ : 0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        #print(self.name2label)
        # image, label
        self.images,self.labels = self.load_csv('iamges.csv')

        if mode == 'train':
            self.images = self.images[:int(0.8 * len(self.images))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        elif mode == 'all':
            self.images = self.images[:]
            self.labels = self.labels[:]
        else:
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root,filename)):

            images = []
            for name in self.name2label.keys():
                # 'data\\0\\0.jpg'
                images += glob.glob(os.path.join(self.root,name,'*jpg'))
            # 301, 'data\\1\\36.jpg'
            print(len(images),images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w',newline='') as f:
                writer = csv.writer(f)
                for img in images: # 'data\\0\\0.jpg'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'data\\0\\0.jpg', 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)


        # read from csv file
        images,labels = [],[]
        with open(os.path.join(self.root,filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'data\\0\\0.jpg', 0
                img,label = row
                #name = img.split(os.sep)[-1][0]
                label = int(label)

                #names.append(name)
                images.append(img)
                labels.append(label)
        assert  len(images) == len(labels)

        return  images,labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'data\\0\\0.jpg'
        # label； 0
        img, label = self.images[idx],self.labels[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x),
            #lambda x: Image.open(x).convert('RGB'), # string path =>  image data
            transforms.Resize((self.resize,self.resize)),
            transforms.ToTensor()
        ])
        name = torch.tensor(int(img.split(os.sep)[-1][:-4]))
        img = tf(img)
        all_label = torch.randn(2)
        all_label[0] = int(name)
        all_label[1] = label



        return img,all_label



def main():

    import visdom
    import time

    viz = visdom.Visdom()
    db = Geneset('data',100,'train')
    #
    x,y = next(iter(db))
    print('sample:',x.shape,y.shape,y[0],y[1])
    viz.images(x,win='sample_x',opts=dict(title='sample_x'))

    loader = DataLoader(db,batch_size=32,shuffle=True,)

    for x,y in loader:
        viz.images(x,nrow=8,win='batch',opts=dict(title='batch'))
        viz.text(str(y.numpy()),win='label',opts=dict(title='batch_y'))
        time.sleep(10)

if __name__ == '__main__':
    main()