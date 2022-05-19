#testing one class neural network

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

from matplotlib import pyplot as plt

import pdb


def main():
    train_model()


class OCNN(nn.Module):
    def __init__(self):
        super().__init__()

        #encode into class decision
        self.embed = nn.Linear(784*2, 784) #embed pair down to size of single input
        self.V = nn.Linear(784, 256)
        self.w = nn.Linear(256, 2) #second output is for deciding if the pair are clones or random

    def forward(self, x, paired=False):
        """x and x2 are batches of 784 vectors representing a pair. 
        The first one receives the class decision
        an identical-ness decision is made for if the network thinks they are the same input or not
        """
        if not paired:
            assert x.shape[-1] == 784
            x = torch.cat([x,x], dim=-1)
        else:
            assert x.shape[-1] == 2 * 784

        x = F.relu(self.embed(x))
        x = F.relu(self.V(x))
        x = torch.tanh(self.w(x)) #TBD if y_hat shouldn't receive tanh here

        y_hat, same_hat = x[:,0], x[:,1]

        return y_hat, same_hat



def get_data(normal_class=6, device='cuda'):
    """get mnist data split into train and test for the normal class, and all others are anomalies"""

    mnist = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # collect the normal and anomaly data
    normal = [img for img, label in mnist if label == normal_class]
    anomaly = [img for img, label in mnist if label != normal_class]

    # create a train test split for the normal data
    num_train = int(len(normal) * 0.5)
    num_test = len(normal) - num_train
    normal_train, normal_test = torch.utils.data.random_split(normal, [num_train, num_test])

    # stack all of the training/testing/anomaly data into single tensors
    normal_train = torch.stack((*normal_train,))
    normal_test = torch.stack((*normal_test,))
    anomaly = torch.stack(anomaly)

    # reshape each image into a vector
    normal_train = normal_train.reshape(len(normal_train), -1)
    normal_test = normal_test.reshape(len(normal_test), -1)
    anomaly = anomaly.reshape(len(anomaly), -1)

    # normalize the data from 0 to 1
    normal_train /= 255.0
    normal_test /= 255.0
    anomaly /= 255.0

    # move the data to the specified device
    normal_train = normal_train.to(device)
    normal_test = normal_test.to(device)
    anomaly = anomaly.to(device)

    return normal_train, normal_test, anomaly


def vec2img(vector: torch.Tensor) -> torch.Tensor:
    """convert a vector to an image"""
    return vector.view((28, 28))

def vecshow(vector: torch.Tensor) -> None:
    """plot a vector as an image"""
    img = vec2img(vector)
    plt.imshow(img, cmap='gray')
    plt.show()


def train_model():
    #get data and move to the GPU
    normal_train, normal_test, anomaly = get_data(device='cuda')

    #create the model and move to the GPU
    model = OCNN().cuda()
    # r = 1.0 #radius from the origin
    nu = 0.1 #tradeoff between points crossing hyperplane and hyperplane distance from origin
    N = normal_train.shape[0]

    #create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #training loop
    for epoch in range(100):
        print(f'----------------- epoch {epoch} -----------------')
        
        for i in range(100):
            optimizer.zero_grad()
            
            #shuffle normal_train (instead of a proper derangement for next step)
            perm = torch.randperm(N)
            normal_train = normal_train[perm]

            #create a shifted copy of the data for paired sameness classification
            #roll instead of permute to guarantee that no element is paired with itself
            #TODO: look into if there is a way to generate derangements, which are more random
            random_shift = torch.randint(1,N,(1,)).item()
            normal_train_permuted = torch.roll(normal_train, random_shift, dims=0)
            paired_normal_train = torch.cat([normal_train, normal_train_permuted], dim=-1)

            y_hat, same_hat = model(normal_train) #pair data with itself
            _, diff_hat = model(paired_normal_train, paired=True) #pair data with its rolled counterpart

            #update r for this iteration    
            with torch.no_grad():
                r = torch.quantile(y_hat, nu).item()
            
            #one class classification loss
            model_loss = F.relu(r - y_hat).sum() / N / nu

            #param loss is l2 norm of all model weights
            param_loss = (nn.utils.parameters_to_vector(model.parameters()) ** 2).sum() / 2
            
            #separability loss, i.e. how well the network can tell different points apart
            sep_loss = (((same_hat - 1)**2).sum()) + ((diff_hat - -1)**2).sum()

            loss = model_loss + param_loss + sep_loss
            loss.backward()
            optimizer.step()

            #set r to the nu'th quantile of y_hat

            print(f'{loss.item():.4f}')

        if (epoch+1) % 20 == 0:
            pdb.set_trace()



        # pdb.set_trace()


if __name__ == '__main__':
    main()