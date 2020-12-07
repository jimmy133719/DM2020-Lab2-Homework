import pdb
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification, AdamW, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from kaggle import submission

class tweetsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    class2idx = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3, 'joy': 4, 'sadness': 5, 'surprise': 6, 'trust': 7}
    idx2class = {v: k for k, v in class2idx.items()}

    is_train = False
    if is_train:
        train_df = pd.read_csv('train.csv')
        texts = train_df.x.tolist()
        labels = [class2idx[item] for item in train_df.y.tolist()]

        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=2020, stratify=labels)

        # split texts into tokens
        # tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        # convert into Dataset
        train_dataset = tweetsDataset(train_encodings, train_labels)
        val_dataset = tweetsDataset(val_encodings, val_labels)


        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # train with trainer
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
        )

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model.classifier = nn.Linear(768, len(class2idx))

        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset             # evaluation dataset
        )

        trainer.train()
        trainer.save_model('bert_trainer_0')


        # # train with pytorch standard flow
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
        # model.logits_proj = nn.Linear(768, len(class2idx))
        # # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        # # model.classifier = nn.Linear(768, len(class2idx))
        # model.to(device)

        # optim = AdamW(model.parameters(), lr=5e-5)

        # global_step = 0
        # for epoch in range(3):
        #     model.train()
        #     for iteration, batch in enumerate(train_loader):
        #         global_step += 1
        #         optim.zero_grad()
        #         input_ids = batch['input_ids'].to(device)
        #         attention_mask = batch['attention_mask'].to(device)
        #         labels = batch['labels'].to(device)
        #         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        #         loss = outputs[0]
        #         if global_step % 1000 == 0:
        #             print('global step = {}, loss = {}'.format(global_step, loss))
        #         loss.backward()
        #         optim.step()

        #     model.eval()
        #     with torch.no_grad():
        #         best_accuracy = 0
        #         labels_list = []
        #         predictions_list = []
        #         for batch in val_loader:
        #             input_ids = batch['input_ids'].to(device)
        #             attention_mask = batch['attention_mask'].to(device)
        #             labels = batch['labels'].to(device)
        #             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        #             predictions_list += np.argmax(outputs.logits.detach().cpu().numpy(), axis=1).tolist()
        #             labels_list += labels.detach().cpu().numpy().tolist()
        #         accuracy = accuracy_score(labels_list, predictions_list)
        #         if accuracy > best_accuracy:
        #             torch.save(model, 'xlnet_0')
        #             best_accuracy = accuracy
        #     print('epoch = {}, accuracy = {}'.format(epoch, accuracy))
        #     print('---------------------------------')
    else:
        identification_file = 'dm2020-hw2-nthu/data_identification.csv'
        emotion_file = 'dm2020-hw2-nthu/emotion.csv'
        json_file = 'dm2020-hw2-nthu/tweets_DM.json'
        samplesubmission_file = 'dm2020-hw2-nthu/sampleSubmission.csv'

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
        test_texts = [tweets_dict[item] for item in sample_id]
        test_labels = [0 for i in range(len(test_texts))]

        # split texts into tokens
        # tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        # convert into Dataset
        test_dataset = tweetsDataset(test_encodings, test_labels)
        
        # Dataloader
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # model = torch.load('bert_0')
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model.classifier = nn.Linear(768, len(class2idx))
        checkpoint = torch.load('results/checkpoint-75500/pytorch_model.bin')
        model.load_state_dict(checkpoint)
        model.to(device)

        # inference
        model.eval()
        predictions = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            predictions += np.argmax(outputs.logits.detach().cpu().numpy(), axis=1).tolist()

        submission(sample_id, predictions, json_file, idx2class)