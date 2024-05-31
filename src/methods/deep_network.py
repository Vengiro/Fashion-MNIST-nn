import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.utils import update_progress_bar

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.l1 = nn.Linear(input_size, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##


        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, img_width=28, img_height=28, filters=(32, 64, 128), kernel_size=3):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        
        # The two next lines are here to make sure the dimension of the layer before and after the convolutions stay the same
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        same_padding = kernel_size // 2
        
        self.conv2d1 = nn.Conv2d(input_channels, filters[0], kernel_size, padding=same_padding)
        self.conv2d2 = nn.Conv2d(filters[0], filters[1], kernel_size, padding=same_padding)
        self.conv2d3 = nn.Conv2d(filters[1], filters[2], kernel_size, padding=same_padding)
        self.fc1 = nn.Linear(128*(img_height//8)*(img_width//8), 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        preds = F.relu(self.conv2d1(x))
        preds = F.relu(self.conv2d2(self.pool(preds)))
        preds = F.relu(self.conv2d3(self.pool(preds)))
        preds = self.pool(preds)
        preds = preds.reshape((preds.shape[0], -1))
        preds = F.relu(self.fc1(preds))
        return self.fc2(preds)

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d) 
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    
    """
    A Transformer-based neural network
    """
    
    def get_positional_embeddings(sequence_length, d):
        """
        Generate positional embeddings.

        Arguments:
            sequence_length (int): Length of the sequence
            d (int): Dimension of the positional embeddings

        Returns:
            positional_embeddings (torch.Tensor): Positional embeddings of shape (sequence_length, d)
        """
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
    
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.

        Arguments:
            chw (tuple): (channels, height, width) of the input images
            n_patches (int): number of patches to split the image into
            n_blocks (int): number of transformer blocks
            hidden_d (int): dimension of the hidden layer
            n_heads (int): number of attention heads
            out_d (int): output dimension (number of classes)
        """
        super().__init__()

        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_patches = n_patches
        self.out_d = out_d
        self.chw = chw
        # Patch
        assert chw[1] == chw[2], "be squared!"
        assert chw[1] % n_patches == 0, "input divisible by n_patches"
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1]//n_patches, chw[2]//n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear = nn.Linear(self.input_d, hidden_d)

        # Token
        self.token = nn.Parameter(torch.randn(1, n_patches*n_patches, hidden_d))
        
        # Positional embedding
        self.positional_embeddings = MyViT.get_positional_embeddings(n_patches ** 2 + 1, hidden_d) ### WRITE YOUR CODE HERE

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )
    


    def patchify(images, n_patches):
        """
        Patchify the input images into patches for the Vision Transformer.
        
        Arguments:
            images (torch.Tensor): Input images of shape (N, C, H, W)
            n_patches (int): Number of patches to split the image into
        
        Returns:
            patches (torch.Tensor): Patches of the input images of shape (N, n_patches ** 2, patch_size ** 2 * C)
        """
        n, c, h, w = images.shape

        assert h == w  # We assume square image.

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches  ### WRITE YOUR CODE HERE

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):

                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size] ### WRITE YOUR CODE HERE

                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

    assert patchify(torch.rand(2, 1, 28, 28), 2).shape == (2, 4, 196)
    assert patchify(torch.rand(16, 3, 56, 56), 7).shape == (16, 49, 192)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, opti="SGD"):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        ### WRITE YOUR CODE HERE

        self.criterion = nn.CrossEntropyLoss()
        if(opti == "SGD"):
            self.optimizer = torch.optim.SGD(model.parameters(), lr)
        elif(opti == "ADAM"):
            self.optimizer = torch.optim.Adam(model.parameters(), lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.model.train()

        eploss = 0
        s = time.time()
        for it, data in enumerate(dataloader):
            x, y = data

            # foward pass through network from image to one hot vector
            y_pred = self.model(x)
            # calculate loss
            loss = self.criterion(y_pred, y)
            # reset gradient and backpropagate loss
            self.optimizer.zero_grad()
            loss.backward()
            # update weights
            self.optimizer.step()
            
            update_progress_bar(it+1, len(dataloader), prefix=f"Epoch {ep+1}/{self.epochs}", suffix=f"Loss: {loss.item()}", length=50)
            eploss += loss.item()

        eploss /= len(dataloader)
        str_s = str(datetime.timedelta(seconds=time.time()-s))
        print(f"Epoch {ep+1}/{self.epochs} done in {str_s}, average loss in epoch: {eploss}\n")



    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.model.eval()
        pred_labels = torch.tensor([])
        with torch.no_grad():
            for it, data in enumerate(dataloader):
                for x in data[0]:
                    y_pred = nn.Softmax(dim=0)(self.model(x.unsqueeze(0)).squeeze(0))
                    pred_labels = torch.cat((pred_labels, torch.argmax(y_pred).unsqueeze(0)))
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        s = time.time()
        self.train_all(train_dataloader)
        print("Total raining time: ", str(datetime.timedelta(seconds=time.time()-s)).split(".")[0])
        
        s = time.time()
        pred_labels = self.predict(training_data)
        print("Prediction time: ", str(datetime.timedelta(seconds=time.time()-s)).split(".")[0])
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()