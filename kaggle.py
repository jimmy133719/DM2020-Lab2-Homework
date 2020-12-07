# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


identification_file = 'dm2020-hw2-nthu/data_identification.csv'
emotion_file = 'dm2020-hw2-nthu/emotion.csv'
json_file = 'dm2020-hw2-nthu/tweets_DM.json'
samplesubmission_file = 'dm2020-hw2-nthu/sampleSubmission.csv'

def data_preprocessing(identification_file, emtion_file, json_file):
    # load dataset
    data_identification = pd.read_csv(identification_file)
    emotion = pd.read_csv(emotion_file).set_index('tweet_id').to_dict()['emotion']
    
    tweets_dict = {}
    with open(json_file, 'r') as f:
        lines = f.readlines()    
        for line in lines:
            tweets_DM = json.loads(line)
            tweets_dict[tweets_DM['_source']['tweet']['tweet_id']] = tweets_DM['_source']['tweet']['text']
    
    # generate train, test data
    x_train = []
    y_train = []
    x_test = []
    for index, row in data_identification.iterrows():
        if row['identification'] == 'train':
            x_train.append(tweets_dict[row['tweet_id']])
            y_train.append(emotion[row['tweet_id']])
        else:
            x_test.append(tweets_dict[row['tweet_id']])


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    """
    
    m = X.shape[0]  # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):  # loop over training examples
        
        # Convert the ith sentence in lower case and split into a list of words
        sentence_words = X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
            else:
                X_indices[i, j] = len(word_to_index) + 1
            # Increment j to j + 1
            j = j + 1
    
    return X_indices

class BiRNN_atten(nn.Module):
    def __init__(self, input_size, embedding, hidden_size, num_layers, num_classes):
        super(BiRNN_atten, self).__init__()
        self.word_embeddings = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            dropout = 0.5, batch_first=True, bidirectional=True) # original does not have dropout
        self.fc = nn.Linear(hidden_size, num_classes)  # 2 for bidirection 
        
        # # not original
        # self.softmax = nn.Softmax(dim=-1)

    def attention_net(self, lstm_output, final_state):
        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                    new hidden state.
                    
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
                        
        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state


    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        embeds = self.word_embeddings(x)
        # Forward propagate RNN
        out, (h_final, c_final) = self.lstm(embeds, (h0, c0))

        # attention
        out_atten_input = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        h_final_atten_input = (h_final[-2,:,:] + h_final[-1,:,:]).unsqueeze(0) 
        out = self.attention_net(out_atten_input, h_final_atten_input)

        # Decode hidden state of last time step
        out = self.fc(out)        


        return out


# BiRNN Model (Many-to-One)
class BiRNN(nn.Module):
    def __init__(self, input_size, embedding, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.word_embeddings = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            dropout = 0.5, batch_first=True, bidirectional=True) # original does not have dropout
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection 

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        embeds = self.word_embeddings(x)
        # Forward propagate RNN
        out, _ = self.lstm(embeds, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        
        # out = self.softmax(out)

        return out

def pretrained_embedding_layer(word_to_vec_map, word_to_index, non_trainable=True):
    num_embeddings = len(word_to_index) + 2   # first for padding, last for unk                
    embedding_dim = word_to_vec_map["cucumber"].shape[0]  #  dimensionality of GloVe word vectors (= 50)

    # Initialize the embedding matrix as a numpy array of zeros of shape (num_embeddings, embedding_dim)
    weights_matrix = np.zeros((num_embeddings, embedding_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        if len(word_to_vec_map[word]) == embedding_dim: # temporary method to solve dimension mismatch problem
            weights_matrix[index, :] = word_to_vec_map[word]
        else:
            weights_matrix[index, :] = np.random.rand(embedding_dim)
    weights_matrix[-1, :] = np.random.rand(embedding_dim) # for unk
    
    embed = nn.Embedding.from_pretrained(torch.from_numpy(weights_matrix).type(torch.FloatTensor), freeze=non_trainable)

    return embed, num_embeddings, embedding_dim


def train(model, trainloader, criterion, optimizer, epochs=10):
    
    model.to(device)
    running_loss = 0
    
    train_losses, test_losses, accuracies = [], [], []
    for e in range(epochs):

        running_loss = 0
        
        model.train()
        
        for (idx, (sentences, labels)) in enumerate(trainloader):

            sentences, labels = sentences.to(device), labels.to(device)

            # 1) erase previous gradients (if they exist)
            optimizer.zero_grad()

            # 2) make a prediction
            # pred = model.forward(sentences)
            pred = model(sentences)

            # 3) calculate how much we missed
            loss = criterion(pred, labels)
            
            # 4) figure out which weights caused us to miss
            loss.backward()

            # 5) change those weights
            optimizer.step()
    
            # 6) log our progress
            running_loss += loss.item()
        
        
        # else:

        model.eval()

        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for sentences, labels in test_loader:
                sentences, labels = sentences.to(device), labels.to(device)
                log_ps = model(sentences)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        accuracies.append(accuracy / len(test_loader) * 100)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
            "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

        if e==0:
            best_accuracy = accuracy
            torch.save(model, 'bilstm_256_atten_glove_200d')
        elif accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model, 'bilstm_256_atten_glove_200d')        
    # Plot
    plt.figure(figsize=(20, 5))
    plt.plot(train_losses, c='b', label='Training loss')
    plt.plot(test_losses, c='r', label='Testing loss')
    plt.xticks(np.arange(0, epochs))
    plt.title('Losses')
    plt.legend(loc='upper right')
    plt.show()
    plt.figure(figsize=(20, 5))
    plt.plot(accuracies)
    plt.xticks(np.arange(0, epochs))
    plt.title('Accuracy')
    plt.show()

def predict(model, testloader):
    model.to(device)
    model.eval()

    predictions = []
    for sentences in test_loader:
        sentences = sentences[0].to(device)
        prediction = model(sentences)
        predictions += np.argmax(prediction.detach().cpu().numpy(), axis=1).tolist()

    return predictions

def ensemble_predict(model_1, model_2, model_3, model_4, testloader):
    model_1.to(device)
    model_2.to(device)
    model_3.to(device)
    model_4.to(device)
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()

    predictions = []
    for sentences in test_loader:
        sentences = sentences[0].to(device)
        prediction_1 = model_1(sentences)
        prediction_2 = model_2(sentences)
        prediction_3 = model_3(sentences)
        prediction_4 = model_4(sentences)
        prediction = prediction_1 + prediction_2 + prediction_3 + prediction_4
        predictions += np.argmax(prediction.detach().cpu().numpy(), axis=1).tolist()
    
    return predictions

def submission(sample_id, predictions, json_file, idx2class):
    tweets_dict = {}
    with open(json_file, 'r') as f:
        lines = f.readlines()    
        for line in lines:
            tweets_DM = json.loads(line)
            tweets_dict[tweets_DM['_source']['tweet']['text']] = tweets_DM['_source']['tweet']['tweet_id'] 
    
    id = sample_id # [tweets_dict[item] for item in test_list]
    emotion = [idx2class[item] for item in predictions]
    submission_df = pd.DataFrame()
    submission_df['id'] = id
    submission_df['emotion'] = emotion
    submission_df.to_csv('xlnet_0.csv', index=False)


if __name__  == '__main__':
    is_train = False
    
    class2idx = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3, 'joy': 4, 'sadness': 5, 'surprise': 6, 'trust': 7}
    idx2class = {v: k for k, v in class2idx.items()}

    if is_train:
        train_df = pd.read_csv('train.csv')

        X = np.array(train_df.x.tolist()[1:])
        y = np.array([class2idx[item] for item in train_df.y.tolist()[1:]])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020, stratify=y)
        
        # pretrained glove embedding can be downloaded from https://nlp.stanford.edu/projects/glove/ 
        word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.twitter.27B.200d.txt') # glove.6B.50d.txt
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        maxLen = max([len(item.split()) for item in X])
        X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
        X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
        y_train_oh = convert_to_one_hot(y_train, C = len(class2idx))
        y_test_oh = convert_to_one_hot(y_test, C = len(class2idx))

        
        embedding, vocab_size, embedding_dim = pretrained_embedding_layer(word_to_vec_map, word_to_index, non_trainable=True)
        
        hidden_dim = 256
        output_size = len(class2idx)
        batch_size = 32
        num_layers = 2
        
        
        model = BiRNN_atten(embedding_dim, embedding, hidden_dim, num_layers, output_size)
        print(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001) # 0.002
        epochs = 50
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_indices).type(torch.LongTensor), torch.tensor(y_train).type(torch.LongTensor))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_indices).type(torch.LongTensor), torch.tensor(y_test).type(torch.LongTensor))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
        train(model, train_loader, criterion, optimizer, epochs)
    
    else:
        is_ensemble = True # whether to use model ensemble

        # load dataset
        data_identification = pd.read_csv(identification_file)
        emotion = pd.read_csv(emotion_file).set_index('tweet_id').to_dict()['emotion']
        
        tweets_dict = {}
        with open(json_file, 'r') as f:
            lines = f.readlines()    
            for line in lines:
                tweets_DM = json.loads(line)
                tweets_dict[tweets_DM['_source']['tweet']['tweet_id']] = tweets_DM['_source']['tweet']['text']
        
        sample_id = pd.read_csv(samplesubmission_file).id.tolist()
        test_list = [tweets_dict[item] for item in sample_id]

        X_test = np.array(test_list)
        
        
        word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.twitter.27B.200d.txt') # glove.6B.50d.txt

        maxLen = max([len(item.split()) for item in X_test])
        X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)


        hidden_dim = 256
        output_size = len(class2idx)
        batch_size = 32
        num_layers = 2

        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_indices).type(torch.LongTensor))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not is_ensemble:
            model = torch.load('bilstm_256_atten_glove_200d')
            predictions = predict(model, test_loader)
        else:
            model_1 = torch.load('bilstm_glove_100d')
            model_2 = torch.load('bilstm_glove_200d')
            model_3 = torch.load('bilstm_atten_glove_200d')
            model_4 = torch.load('bilstm_256_atten_glove_200d')
            predictions = ensemble_predict(model_1, model_2, model_3, model_4, test_loader)        
        
        
        submission(sample_id, predictions, json_file, idx2class)
