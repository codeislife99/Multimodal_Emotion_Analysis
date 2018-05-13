from mosei_dataloader import mosei
from models.text_encoders import TextOnlyModel
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np


def preprocess(options):
    # parse the input args
    dataset = options['dataset']
    epochs = options['epochs']
    model_path = options['model_path']
    vid_or_seg_based = options['vid_or_seg_based']
    if vid_or_seg_based == 'seg':
        segment=True
    elif vid_or_seg_based == 'vid':
        segment=False
    else:
        raise ValueError("illegal string value {} for vid_or_seg_based arg".format(vid_or_seg_based))

    # prepare the paths for storing models
    model_path = os.path.join(
        model_path, "text_only.pt")
    print("Temp location for saving model: {}".format(model_path))

    # prepare the datasets
    print("Currently using {} dataset.".format(dataset))
    train_loader = mosei('train', segment)
    valid_loader = mosei('val', segment)

    ######### FIX THIS ##########

    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    # input_dims = (audio_dim, visual_dim, text_dim)

    # text_train_set = train_set[:][2]
    # text_valid_set = valid_set[:][2]

    return text_train_set, text_valid_set, text_dim

def display(test_loss, test_binacc, test_precision, test_recall, test_f1, test_septacc, test_corr):
    print("MAE on test set is {}".format(test_loss))
    print("Binary accuracy on test set is {}".format(test_binacc))
    print("Precision on test set is {}".format(test_precision))
    print("Recall on test set is {}".format(test_recall))
    print("F1 score on test set is {}".format(test_f1))
    print("Seven-class accuracy on test set is {}".format(test_septacc))
    print("Correlation w.r.t human evaluation on test set is {}".format(test_corr))

def main(options):
    DTYPE = torch.FloatTensor
    train_set, valid_set, input_dim = preprocess(options)

    hid_size, out_size, num_layers=1, rnn_dropout=0.2, post_dropout=0.2, bidirectional=False
    text_hid_size = 64
    model = TextOnlyModel(input_dim, text_hid_size, 6, rnn_dropout=0.2, post_dropout=0.2, bidirectional=False)
    if options['cuda']:
        model = model.cuda()
        DTYPE = torch.cuda.FloatTensor
    print("Model initialized")

    criterion = nn.L1Loss(size_average=False)
    optimizer = optim.Adam(list(model.parameters())[2:]) # don't optimize the first 2 params, they should be fixed (output_scale and shift)

    # setup training
    complete = True
    min_valid_loss = float('Inf')
    batch_sz = options['batch_size']
    patience = options['patience']
    epochs = options['epochs']
    model_path = options['model_path']
    train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=4, shuffle=True)
    valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
    test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
    curr_patience = patience
    for e in range(epochs):
        model.train()
        model.zero_grad()
        train_loss = 0.0
        for batch in train_iterator:
            model.zero_grad()

            # the provided data has format [batch_size, seq_len, feature_dim] or [batch_size, 1, feature_dim]
            x_avt = batch[:-1]

            x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
            y = Variable(batch[-1].view(-1, 1).float().type(DTYPE), requires_grad=False)
            output = model(x_t)
            loss = criterion(output, y)
            loss.backward()
            train_loss += loss.data[0] / len(train_set)
            optimizer.step()

        print("Epoch {} complete! Average Training loss: {}".format(e, train_loss))

        # Terminate the training process if run into NaN
        if np.isnan(train_loss):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        # On validation set we don't have to compute metrics other than MAE and accuracy
        model.eval()
        for batch in valid_iterator:
            x_avt = batch[:-1]
            x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
            y = Variable(batch[-1].view(-1, 1).float().type(DTYPE), requires_grad=False)
            output = model(x_t)
            valid_loss = criterion(output, y)
        output_valid = output.cpu().data.numpy().reshape(-1)
        y = y.cpu().data.numpy().reshape(-1)

        if np.isnan(valid_loss.data[0]):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        valid_binacc = accuracy_score(output_valid>=0, y>=0)

        print("Validation loss is: {}".format(valid_loss.data[0] / len(valid_set)))
        print("Validation binary accuracy is: {}".format(valid_binacc))

        if (valid_loss.data[0] < min_valid_loss):
            curr_patience = patience
            min_valid_loss = valid_loss.data[0]
            torch.save(model, model_path)
            print("Found new best model, saving to disk...")
        else:
            curr_patience -= 1

        if curr_patience <= 0:
            break
        print("\n\n")

    # if complete:

    #     best_model = torch.load(model_path)
    #     best_model.eval()
    #     for batch in test_iterator:
    #         x_avt = batch[:-1]
    #         x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
    #         y = Variable(batch[-1].view(-1, 1).float().type(DTYPE), requires_grad=False)
    #         output_test = model(x_t)
    #         loss_test = criterion(output_test, y)
    #         test_loss = loss_test.data[0]
    #     output_test = output_test.cpu().data.numpy().reshape(-1)
    #     y = y.cpu().data.numpy().reshape(-1)

    #     test_binacc = accuracy_score(output_test>=0, y>=0)
    #     test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y>=0, output_test>=0, average='binary')
    #     test_septacc = (output_test.round() == y.round()).mean()

    #     # compute the correlation between true and predicted scores
    #     test_corr = np.corrcoef([output_test, y])[0][1]  # corrcoef returns a matrix
    #     test_loss = test_loss / len(test_set)

    #     display(test_loss, test_binacc, test_precision, test_recall, test_f1, test_septacc, test_corr)
    return

if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--dataset', dest='dataset',
                         type=str, default='MOSEI')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=50)
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--vidorseg', dest='vid_or_seg_based', type=str, default='seg')

    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
