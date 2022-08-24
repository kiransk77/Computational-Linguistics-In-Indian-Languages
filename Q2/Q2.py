#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import re


# In[96]:


file =  open(r"hi_train.conll",encoding="utf-8")
text = file.read()
text = re.sub(r'[^\w\s]', '', text)


# In[97]:


words,tags = [],[]
for line in text.split('\n'):
  if(line == '' or line[0] == '#'):
    continue
  else:
      line = line.split()
      if line[0] !='id':
        words.append(line[0])
        tags.append(line[-1])


# In[98]:


df = pd.DataFrame({"sentence":words,"word_labels":tags})
df


# In[99]:


labels_to_ids = {k: v for v, k in enumerate(df.word_labels.unique())}
ids_to_labels = {v: k for v, k in enumerate(df.word_labels.unique())}
labels_to_ids
ids_to_labels


# In[100]:


df = df[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
df.head()


# In[101]:


df['word_labels'].unique()


# In[102]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install sentencepiece')


# In[103]:


MAX_LEN = 20
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 6*1e-03
MAX_GRAD_NORM = 10
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')


# In[104]:


class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        #sentence and word labels for the given index
        sentence = self.data.sentence[index].strip()
        word_labels = self.data.word_labels[index].split(",") 

        #using tokenizer to encode sentence
        encoding = self.tokenizer(sentence,
                                  return_offsets_mapping=True, 
                                  padding='max_length',
                                  truncation=True, 
                                  max_length=self.max_len)
        
        #create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels] 

        #turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels[0])
        
        return item

  def __len__(self):
        return self.len


# In[105]:


train_size = 0.8
#val_size = 0.5

train_dataset = df.sample(frac=train_size,random_state=200)
val_dataset = df.drop(train_dataset.index)

train_dataset = train_dataset.reset_index(drop=True)
val_dataset = val_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
validation_set = dataset(val_dataset, tokenizer, MAX_LEN)


# In[106]:


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)


# In[107]:


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)


# In[108]:


import torch
import torch.nn as nn


# 

# In[36]:


class add_layer(nn.Module):
  
  def __init__(self,freeze_bert=False):
     super(add_layer, self).__init__()
     self.auto =  AutoModel.from_pretrained('ai4bharat/indic-bert')
     self.classifier =  nn.Sequential(nn.Linear(768, 150), nn.ReLU(),nn.Linear(150, 80),nn.ReLU(),nn.Dropout(0.15),nn.Linear(80, 45),nn.ReLU(),nn.Dropout(0.15),nn.Linear(45,13))
     if freeze_bert:
            for param in self.auto.parameters():
                param.requires_grad = False
  def forward(self,ids,mask):
    output = self.auto(input_ids=ids,attention_mask=mask)
    hidden_stat = output[0][:, 0, :]
    
    logits = self.classifier(hidden_stat)
    return logits



# In[37]:


loss_fn = nn.CrossEntropyLoss()
bertt = add_layer(True)
bertt.to(device)


# In[38]:


def train(epochs):

  tr_loss, tr_accuracy = 0, 0
  nb_tr_examples, nb_tr_steps = 0, 0
  tr_preds, tr_labels = [], []
  
  optimizer = torch.optim.Adam(params=bertt.parameters(), lr=LEARNING_RATE)
  bertt.train()
  for idx, batch in enumerate(training_loader):
    ids = batch['input_ids'].to(device, dtype = torch.long)
    mask = batch['attention_mask'].to(device, dtype = torch.long)
    labels = batch['labels'].to(device, dtype = torch.long)
    
    logits = bertt(ids,mask)
    loss = loss_fn(logits, labels)
    tr_loss += loss.item()
  
    nb_tr_steps += 1
    nb_tr_examples += labels.size(0)
    
    if idx % 100==0:
        loss_step = tr_loss/nb_tr_steps
        print(f"Training loss per 100 training steps: {loss_step}")

    flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
    active_logits = logits.view(-1,13) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len)

    active_accuracy = labels.view(-1) != -100

    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    tr_labels.extend(labels)
    tr_preds.extend(predictions)

    tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    tr_accuracy += tmp_tr_accuracy
  
    torch.nn.utils.clip_grad_norm_(
        parameters=bertt.parameters(), max_norm=MAX_GRAD_NORM
    )
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  epoch_loss = tr_loss / nb_tr_steps
  tr_accuracy= tr_accuracy / nb_tr_steps
  print(f"Training loss epoch: {epoch_loss}")
  print(f"Training accuracy epoch: {tr_accuracy}")


# In[39]:


for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)


# In[56]:


def evaluate(bertt,loader):
   
    # For each batch in our validation set...
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
  
    
    bertt.eval()
    with torch.no_grad():
       for idx, batch in enumerate(loader):
             ids = batch['input_ids'].to(device, dtype = torch.long)
             mask = batch['attention_mask'].to(device, dtype = torch.long)
             labels = batch['labels'].to(device, dtype = torch.long)
    
             logits = bertt(ids,mask)
             loss = loss_fn(logits, labels)
             tr_loss += loss.item()
  
             nb_tr_steps += 1
             nb_tr_examples += labels.size(0)
    
             if idx % 100==0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

             flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
             active_logits = logits.view(-1,13) # shape (batch_size * seq_len, num_labels)
             flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len)

             active_accuracy = labels.view(-1) != -100

             labels = torch.masked_select(flattened_targets, active_accuracy)
             predictions = torch.masked_select(flattened_predictions, active_accuracy)
          
             tr_labels.extend(labels)
             tr_preds.extend(predictions)
            
             tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
             tr_accuracy += tmp_eval_accuracy
  
    labels = [ids_to_labels[id.item()] for id in tr_labels]
    predictions = [ids_to_labels[id.item()] for id in tr_preds]
    
    eval_loss = tr_loss / nb_tr_steps 
    eval_accuracy = tr_accuracy / nb_tr_steps 
    print(f"Validation Loss: { eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy }")

    return labels, predictions


# In[41]:


x = evaluate(bertt,validation_loader)


# Testing the model

# In[75]:


file_folder = r"hi_dev.conll"
file =  open(file_folder,encoding="utf-8")
text = file.read()


# In[76]:


import re
text = re.sub(r'[^\w\s]', '', text)


# In[77]:


s_cout = 0
l1 = []
l2 = []
l3 = []
for word in text.split('\n'):
  if(word == ''):
    s_cout =  s_cout + 1
  elif(word[0] == '#'):
    continue
    
  else:
      word = word.split(" ")
      k  = "sent_"+ str(s_cout)
      l1.append(k)
      l2.append(word[0])
      l3.append(word[-1])


# In[78]:


df.head()


# In[79]:


df = pd.DataFrame({"sent":l1,"word":l2,"tag":l3})
df.head()


# In[80]:


lis = []
for i in range(df.shape[0]):
  if(df.iloc[i,1] != ''):
    lis.append(i)
df = df.iloc[lis,:]


# In[81]:


lis = []
for i in range(df.shape[0]):
  if(df.iloc[i,2] != ''):
    lis.append(i)
df = df.iloc[lis,:]


# In[82]:


df['sentence'] =  df['word']
df['word_labels']  = df['tag']
df.head()


# In[83]:


labels_to_ids = {k: v for v, k in enumerate(df.word_labels.unique())}
ids_to_labels = {v: k for v, k in enumerate(df.word_labels.unique())}
labels_to_ids
ids_to_labels


# In[90]:


from collections import Counter
Counter(df.word_labels)


# In[51]:


df = df[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
df.head()


# In[52]:


testing_set = dataset(df, tokenizer, MAX_LEN)


# In[53]:


test_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }


testing_loader = DataLoader(testing_set, **test_params)


# In[63]:


x = evaluate(bertt,testing_loader)


# In[94]:


from sklearn.metrics import f1_score
print("F1 score for testing set:")
f1_score(x[0], x[1], average='micro')

