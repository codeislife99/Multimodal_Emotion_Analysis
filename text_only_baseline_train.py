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
    text_train_set = mosei('train', segment)
    text_valid_set = mosei('val', segment)
    text_test_set = mosei('test', segment)

    return text_train_set, text_valid_set, text_test_set

def display(test_loss, test_binacc, test_precision, test_recall, test_f1, test_septacc, test_corr):
    print("MAE on test set is {}".format(test_loss))
    print("Binary accuracy on test set is {}".format(test_binacc))
    print("Precision on test set is {}".format(test_precision))
    print("Recall on test set is {}".format(test_recall))
    print("F1 score on test set is {}".format(test_f1))
    print("Seven-class accuracy on test set is {}".format(test_septacc))
    print("Correlation w.r.t human evaluation on test set is {}".format(test_corr))

def save_checkpoint(state, is_final, filename='text_only'):
    filename = filename +'_'+str(state['epoch'])+'.pth.tar'
    os.system("mkdir -p text_only") 
    torch.save(state, './text_only/'+filename)
    if is_final:
        shutil.copyfile(filename, './text_only/model_final.pth.tar')

def main(options):
    DTYPE = torch.FloatTensor
    train_set, valid_set, test_set = preprocess(options)

    batch_size = options['batch_size']
    num_workers = options['num_workers']
    patience = options['patience']
    epochs = options['epochs']
    model_path = options['model_path']
    curr_patience = patience

    train_iterator = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_iterator = DataLoader(valid_set, batch_size=1, num_workers=num_workers, shuffle=True)
    test_iterator = DataLoader(test_set, batch_size=1, num_workers=num_workers, shuffle=True)

    input_dim = 300
    batch_size = options['batch_size']
    bidirectional = options['bidirectional']
    num_layers = options['num_layers']
    text_hid_size = options['hidden_size']
    batch_size = options['batch_size']
    model = TextOnlyModel(input_dim, text_hid_size, 6, batch_size, rnn_dropout=0.2, post_dropout=0.2, bidirectional=bidirectional)
    if options['cuda']:
        model = model.cuda()
        DTYPE = torch.cuda.FloatTensor
    print("Model initialized")

    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(list(model.parameters())[2:]) # don't optimize the first 2 params, they should be fixed (output_scale and shift)

    # setup training
    complete = True
    min_valid_loss = float('Inf')
    use_pretrained = False
    e = 0
    if use_pretrained:
        # pretrained_file = './TAN/triple_attention_net_iter_8000_0.pth.tar'
        pretrained_file = './text_only/text_only_net__0.pth.tar'
        checkpoint = torch.load(pretrained_file)
        model.load_state_dict(checkpoint['text_model'])
        use_pretrained = False
        e = checkpoint['epoch']+1
        optimizer.load_state_dict(checkpoint['optimizer'])

    while e<epochs:
        model.train()
        model.zero_grad()
        train_loss = 0.0
        K = 0
        for _, _, x_t, gt in train_iterator: # iterate over batches of text and gt labels (x_t is unpadded)
            # model.zero_grad()

            # the provided data has format [batch_size, seq_len, feature_dim] or [batch_size, 1, feature_dim]

            # x_t = Variable(x_t.float().type(DTYPE), requires_grad=False) # unpadded
            gt = Variable(gt.float().type(DTYPE), requires_grad=False)

            if batch_size > 1:

                # need to pad the batch according to longest sequence within it
                seq_lengths = torch.LongTensor([x_t[i, :].size()[0] for i in range(x_t.size()[0])])

                # NOTE: typically padding is performed at word idx level i.e. before embedding projection
                # but we begin with embeddings, so *hopefully* it's ok to embed pad tkn as [0]*300
                seq_tensor = torch.zeros((x_t.size()[0], seq_lengths.max(), x_t.size()[2]))
                for idx, (seq, seqlen) in enumerate(zip(x_t.long(), seq_lengths)):
                    seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
                # sort tensors by length
                seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
                seq_tensor = seq_tensor[perm_idx]
                seq_tensor = Variable(seq_tensor.float().type(DTYPE), requires_grad=False)

                output = model(seq_tensor, seq_lengths.cpu().numpy)
            else:
                x_t = Variable(x_t.float().type(DTYPE), requires_grad=False)
                output = model(x_t)

            loss = criterion(output, gt)
            if K%options['mega_batch_size'] == 0:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
            
            
            train_loss += loss.data[0] 
            K+=1
            average_loss = train_loss/K
            if K%20 == 0:
                print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f'
                % (e+1, K, average_loss))
            if K%4000 == 0:
                save_checkpoint({
                    'epoch': e,
                    'loss' : average_loss,
                    'text_model' : model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, False,'text_only_net_iter_'+str(K))

        print("Epoch {} complete! Average Training loss: {}".format(e, average_loss))
        save_checkpoint({
            'epoch': e,
            'loss' : average_loss,
            'text_model' : model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False,'text_only_net_')
        # Terminate the training process if run into NaN
        # On validation set we don't have to compute metrics other than MAE and accuracy
        model.zero_grad()
        model.eval()
        K = 0
        valid_loss = 0.0
        for _, _, x_t, gt in valid_iterator:

            # x_t = Variable(x_t.float().type(DTYPE), requires_grad=False)
            gt = Variable(gt.float().type(DTYPE), requires_grad=False)
            if batch_size > 1:

                # need to pad the batch according to longest sequence within it
                seq_lengths = torch.LongTensor([x_t[i, :].size()[0] for i in range(x_t.size()[0])])

                # NOTE: typically padding is performed at word idx level i.e. before embedding projection
                # but we begin with embeddings, so *hopefully* it's ok to embed pad tkn as [0]*300
                seq_tensor = torch.zeros((x_t.size()[0], seq_lengths.max(), x_t.size()[2]))
                for idx, (seq, seqlen) in enumerate(zip(x_t.long(), seq_lengths)):
                    seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
                # sort tensors by length
                seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
                seq_tensor = seq_tensor[perm_idx]
                seq_tensor = Variable(seq_tensor.float().type(DTYPE), requires_grad=False)

                output = model(seq_tensor, seq_lengths.cpu().numpy)
            else:
                x_t = Variable(x_t.float().type(DTYPE), requires_grad=False)
                output = model(x_t)

            loss = criterion(output, gt)
            valid_loss += loss.data[0]
            K+=1
            average_valid_loss = valid_loss/K
            if K%20 == 0:
                print('Validating -- Epoch [%d], Sample [%d], Average Loss: %.4f'
                % (e+1, K, average_valid_loss))

        print("Validation loss is: {}".format(average_valid_loss))

        if (valid_loss.data[0] < min_valid_loss):
            curr_patience = patience
            min_valid_loss = valid_loss.data[0]
            save_checkpoint({
                'epoch': e,
                'loss' : min_valid_loss,
                'text_model' : model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, True,'text_only_net_')
            print("Found new best model, saving to disk...")
        else:
            curr_patience -= 1

        if curr_patience <= 0:
            break
        print("\n\n")

    # if complete:

    #     best_model = torch.load(model_path)
    #     best_model.eval()
    #     for _, _, x_t, gt in test_iterator:
    #         x_t = Variable(x_avt[2].float().type(DTYPE), requires_grad=False)
    #         gt = Variable(gt.float().type(DTYPE), requires_grad=False)
    #         output_test = model(x_t)
    #         loss_test = criterion(output_test, gt)
    #         test_loss = loss_test.data[0]
    #     output_test = output_test.cpu().data.numpy().reshape(-1)
    #     gt = gt.cpu().data.numpy().reshape(-1)

    #     test_binacc = accuracy_score(output_test>=0, gt>=0)
    #     test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(gt>=0, output_test>=0, average='binary')
    #     test_septacc = (output_test.round() == gt.round()).mean()

    #     # compute the correlation between true and predicted scores
    #     test_corr = np.corrcoef([output_test, gt])[0][1]  # corrcoef returns a matrix
    #     test_loss = test_loss / len(test_set)

    #     display(test_loss, test_binacc, test_precision, test_recall, test_f1, test_septacc, test_corr)
    return

if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--dataset', dest='dataset',
                         type=str, default='MOSEI')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=50)
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=1)
    OPTIONS.add_argument('--mega_batch_size', dest='mega_batch_size', type=int, default=16)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--vidorseg', dest='vid_or_seg_based', type=str, default='seg')
    OPTIONS.add_argument('--num_workers', dest='num_workers', type=int, default=20)
    OPTIONS.add_argument('--num_layers', dest='num_layers', type=int, default=1)
    OPTIONS.add_argument('--hidden_size', dest='hidden_size', type=int, default=64)
    OPTIONS.add_argument('--bidirectional', dest='bidirectional', action='store_true', default=False)



    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
