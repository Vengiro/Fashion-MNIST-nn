import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    print(f"Train data: {xtrain.shape} - Test data: {xtest.shape} - Train labels: {ytrain.shape}")

    num_train_samples = ytrain.shape[0]


    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
        print("Using a validation set")
        
        fraction_validation_test = 0.3
        
        num_train_samples = ytrain.shape[0]
        num_train_samples = ytrain.shape[0]
        rinds = np.random.permutation(num_train_samples)
        n_validation = int(num_train_samples * fraction_validation_test)
        xtest = xtrain[rinds[:n_validation]]
        ytest = ytrain[rinds[:n_validation]] 
        xtrain = xtrain[rinds[n_validation:]]
        ytrain = ytrain[rinds[n_validation:]]
        num_train_samples = ytrain.shape[0]
        num_test_samples = ytest.shape[0]

    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    in_size = xtrain.shape[1]
    if args.nn_type == "mlp":
        model = MLP(in_size, n_classes)
    elif args.nn_type == "cnn":
        xtrain = np.reshape(xtrain, (num_train_samples, 1, 28, 28))
        xtest = np.reshape(xtest, (num_test_samples, 1, 28, 28))
        model = CNN(1, n_classes)
        
    elif args.nn_type == "transformer":
        # A toi de jouer Kake
        ...
        positional_embeddings = MyViT.get_positional_embeddings(100, 300, plot=args.show_heatmap)
    else:
        raise ValueError(f"Unknown network type: {args.nn_type}")
        
    summary(model)

    # Trainer object

    if args.ADAM:
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, opti="ADAM")
    else:
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
        

    # TODO: pour pas test tout le dataset pcq cest long
    size = 1000
    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain[:len(preds_train)])
    macrof1 = macrof1_fn(preds_train, ytrain[:len(preds_train)])
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    
    if not args.test:
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    # Additional arguments by the students
    parser.add_argument('--ADAM', action="store_true", help="Use ADAM optimizer instead of SGD")
    # New argument for showing or not showing the heatmap
    parser.add_argument('--show_heatmap', action="store_true", help="Show the positional embeddings heatmap for the Transformer")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)