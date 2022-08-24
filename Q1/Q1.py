#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import norm
from gensim.models import Word2Vec
from gensim.models import FastText
import warnings
warnings.filterwarnings('ignore')


# In[2]:


thresholds=[4,5,6,7,8]


# In[3]:


#calculation of cosine similarity
def cosine_sim(vec1,vec2):
    return 10*(vec1.dot(vec2)/(norm(vec1)*norm(vec2)))


# # Preprosessing

# 1. Word similarity dataset

# In[4]:


#load word similarity file along with ground truth values
file=pd.read_csv('hindi_word_similarity.txt',names=["word1","word2","similarity"],sep=',',header=None)
words = file[["word1","word2"]]
#ground_truth = file[["similarity"]]


# In[6]:


for threshold in thresholds:
    pred_column = "Ground_truth_"+str(threshold)
    file[pred_column] = np.where(file["similarity"] >= threshold, 1, 0)


# In[7]:


file


# # GloVe

# In[8]:


#loading glove word embedding for 50 and 100 dimensions
g50 = pd.read_csv("./hi/50/glove/hi-d50-glove.txt",sep=" ",header=None)
g100= pd.read_csv("./hi/100/glove/hi-d100-glove.txt",sep=" ",header=None)
#g200=pd.read_csv("./hi/200/glove/hi-d200-glove.txt",sep=" ",header=None)
#g300=pd.read_csv("./hi/300/glove/hi-d300-glove.txt",sep=" ",header=None)


# In[9]:


#extracting glove word embeddings for given words
def glove_data(df,word1,word2):
    temp1 = df.loc[df[0] == word1]
    temp1 =  temp1.T
    temp1 = list(temp1.iloc[1:,0])
   
    temp2 = df.loc[df[0] == word2]
    temp2 =  temp2.T
    temp2 = list(temp2.iloc[1:,0])
    
    vec1 =np.array(temp1)
    vec2 =np.array(temp2)
    return vec1,vec2


# In[10]:


g50_sim,g100_sim,g200_sim,g300_sim =[],[],[],[]

for index, row in words.iterrows(): #iterating over each word pair
    
    #taking two words to find their similarity
    word1 = row["word1"].strip() 
    word2 = row["word2"].strip()
    
    #extracting glove-50 and glove-100embeddings for both words
    vec1,vec2=glove_data(g50,word1,word2)   
    cos_sim = cosine_sim(vec1,vec2)         #cosine similarity
    g50_sim.append(cos_sim)                 
    
    vec1,vec2=glove_data(g100,word1,word2)  
    cos_sim = cosine_sim(vec1,vec2)         #cosine similarity
    g100_sim.append(cos_sim)                
    
    
    #vec1,vec2=glove_data(g200,word1,word2)
    #cos_sim = cosine_sim(vec1,vec2)
    #g200_sim.append(cos_sim)
    
    
    #vec1,vec2=glove_data(g300,word1,word2)
    #cos_sim = cosine_sim(vec1,vec2)
    #g300_sim.append(cos_sim)


# In[11]:


#saving similarity value in dataframe

glove_file = file[["word1","word2"]]
glove_file["GloVe_sim_50"] = g50_sim
glove_file["GloVe_sim_100"]= g100_sim
#glove_file["g200_sim"]= g200_sim
#glove_file["g300_sim"]= g300_sim
glove_file


# In[12]:


import pickle
with open('glove_sim.pkl','wb') as f:
    pickle.dump(file,f)
    f.close()


# # CBow

# In[13]:


#loading cbow50 and cbow100 models using Word2Vec

cbow50  = Word2Vec.load("./hi/50/cbow/hi-d50-m2-cbow.model")
cbow100 = Word2Vec.load("./hi/100/cbow/hi-d100-m2-cbow.model")
#cbow200 = Word2Vec.load("./hi/200/cbow/hi-d200-m2-cbow.model")
#cbow300 = Word2Vec.load("./hi/300/cbow/hi-d300-m2-cbow.model")


# In[14]:


cbow50_sim,cbow100_sim,cbow200_sim,cbow300_sim =[],[],[],[]
for index, row in words.iterrows():    #iterating over each word pair
    
    #taking two words to find their similarity
    word1 = row["word1"].strip()
    word2 = row["word2"].strip()
    
    #extracting cbow50 and cbow100 embeddings for both words
    vec1 = cbow50.wv[word1]
    vec2 = cbow50.wv[word2]
    cos_sim = cosine_sim(vec1,vec2)   #cosine similarity
    cbow50_sim.append(cos_sim)        

    vec1 = cbow100.wv[word1]
    vec2 = cbow100.wv[word2]
    cos_sim = cosine_sim(vec1,vec2)   #cosine similarity
    cbow100_sim.append(cos_sim)
    
    #vec1 = cbow200.wv[word1]
    #vec2 = cbow200.wv[word2]
    #cos_sim = cosine_sim(vec1,vec2)
    #cbow200_sim.append(cos_sim)
    
    #vec1 = cbow300.wv[word1]
    #vec2 = cbow300.wv[word2]
    #cos_sim = cosine_sim(vec1,vec2)
    #cbow300_sim.append(cos_sim)


# In[15]:


#saving similarity values in dataframe
cbow_file = file[["word1","word2"]]
cbow_file["Cbow_sim_50"] = cbow50_sim
cbow_file["Cbow_sim_100"]= cbow100_sim

#cbow_file["cbow200_sim"]= cbow200_sim
#cbow_file["cbow300_sim"]= cbow300_sim

cbow_file


# # Fasttext

# In[16]:


#loading fasttext50 and fasttext100 models
ft50  = FastText.load("./hi/50/fasttext/hi-d50-m2-fasttext.model")
ft100 = FastText.load("./hi/100/fasttext/hi-d100-m2-fasttext.model")
#ft200 = FastText.load("./hi/200/fasttext/hi-d200-m2-fasttext.model")
#ft300 = FastText.load("./hi/300/fasttext/hi-d300-m2-fasttext.model")


# In[17]:


ft50_sim,ft100_sim,ft200_sim,ft300_sim =[],[],[],[]
for index, row in words.iterrows(): #iterating over each word pair
    
    #taking two words to find their similarity
    word1 = row["word1"].strip()
    word2 = row["word2"].strip()
    
    #extracting fasttext50 and 100 embeddings 
    vec1 = ft50.wv[word1]
    vec2 = ft50.wv[word2]
    cos_sim = cosine_sim(vec1,vec2) #cosine similarity
    ft50_sim.append(cos_sim)
    
    vec1 = ft100.wv[word1]
    vec2 = ft100.wv[word2]
    cos_sim = cosine_sim(vec1,vec2) #cosine similarity
    ft100_sim.append(cos_sim)
    
    #vec1 = ft200.wv[word1]
    #vec2 = ft200.wv[word2]
    #cos_sim = cosine_sim(vec1,vec2)
    #ft200_sim.append(cos_sim)
    
    #vec1 = ft300.wv[word1]
    #vec2 = ft300.wv[word2]
    #cos_sim = cosine_sim(vec1,vec2)
    #ft300_sim.append(cos_sim)


# In[18]:


#saving similarities in the dataframe

ft_file = file[["word1","word2"]]
ft_file["FastText_sim_50"] = ft50_sim
ft_file["FastText_sim_100"]= ft100_sim
#ft_file["ft200_sim"]= ft200_sim
#ft_file["ft300_sim"]= ft300_sim
ft_file


# # Skip grams

# In[19]:


#loading skip grams 50 and 100 models
sg50  = Word2Vec.load("./hi/50/sg/hi-d50-m2-sg.model")
sg100 = Word2Vec.load("./hi/100/sg/hi-d100-m2-sg.model")
#sg200 = Word2Vec.load("./hi/200/cbow/hi-d200-m2-cbow.model")
#sg300 = Word2Vec.load("./hi/300/cbow/hi-d300-m2-cbow.model")


# In[20]:


sg50_sim,sg100_sim,sg200_sim,sg300_sim =[],[],[],[]
for index, row in words.iterrows(): #iterating over each word pair
    
    #taking both words to find similarity
    word1 = row["word1"].strip()
    word2 = row["word2"].strip()
    
    #extracting skip grams 50 and 100 embeddings
    vec1 = sg50.wv[word1]
    vec2 = sg50.wv[word2]
    cos_sim = cosine_sim(vec1,vec2) #cosine similarity
    sg50_sim.append(cos_sim)

    vec1 = sg100.wv[word1]
    vec2 = sg100.wv[word2]
    cos_sim = cosine_sim(vec1,vec2)  #cosine similarity
    sg100_sim.append(cos_sim)
    
    #vec1 = sg200.wv[word1]
    #vec2 = sg200.wv[word2]
    #cos_sim = cosine_sim(vec1,vec2)
    #sg200_sim.append(cos_sim)
    
    #vec1 = sg300.wv[word1]
    #vec2 = sg300.wv[word2]
    #cos_sim = cosine_sim(vec1,vec2)
    #sg300_sim.append(cos_sim)


# In[21]:


#saving similarity to dataframe
sg_file = file[["word1","word2"]]
sg_file["SkipGram_sim_50"] = sg50_sim
sg_file["SkipGram_sim_100"]= sg100_sim

#sg_file["sg200_sim"]= sg200_sim
#sg_file["sg300_sim"]= sg300_sim

sg_file


# In[22]:


#calculating accuracy
def accuracy_func(embed_df,pred_sim_name,threshold):
    ground_truth = "Ground_truth_"+str(threshold)
    true_similarity = file[ground_truth].to_list() #manually written true similarities
    pred_similarity = embed_df[pred_sim_name].to_list() #predicted similarity based on threshold values
    result=np.invert(np.logical_xor(true_similarity,pred_similarity)) #XNOR to find the common similarities
    count = np.count_nonzero(result) #counting ones in the result
    accuracy_value = count/len(words) #accuracy finding
    return accuracy_value


# In[23]:


#determining similarities based on threshold value

#giving name to each type of embedding
glove_file.name ='GloVe'
cbow_file.name ='Cbow'
sg_file.name ='SkipGram'
ft_file.name ='FastText'
#embed_mapping={'glove':"GloVe",'cbow':}
embedding_types=[]
accuracies =[]
embedding_dfs=[glove_file,cbow_file,sg_file,ft_file]
dimensions = [50,100]

#this loop compares all threshold values with all cosine similarity values for all given embeddings and dimensions
for embed in embedding_dfs: 
    for threshold in thresholds:
        for dim in dimensions:
            emd_type = embed.name
            pred_column = emd_type+"pred_sim_"+str(threshold)+"_"+str(dim)
            target_column = emd_type+"_sim_"+str(dim)
            #saving predicted similarity labels in dataframe
            embed[pred_column] = np.where(embed[target_column] >= threshold, 1, 0)
            
            semi_df = pd.DataFrame(columns=["Word1","Word2","Similarity Score","Ground Truth similarity score","Label"])
            semi_df["Word1"]=file["word1"]
            semi_df["Word2"]=file["word2"]
            semi_df["Similarity Score"]=embed[target_column]
            semi_df["Ground Truth similarity score"]=file["similarity"]
            semi_df["Label"]=embed[pred_column]
            
            #accuracy
            accuracy_value=accuracy_func(embed,pred_column,threshold)
            embedding_types.append(emd_type+"_"+str(threshold)+"_"+str(dim))
            accuracies.append(accuracy_value)
            
            #save in csv file
            acc_df=pd.DataFrame(data=[['','Accuracy','=',accuracy_value,'']],columns=list(semi_df))
            final_df = pd.concat([semi_df,acc_df],ignore_index=True)
            final_df.to_csv("./Outputs/Q1/"+"Q1_"+emd_type+str(dim)+"_similarity_"+str(threshold)+".csv")


# In[24]:


#save accuracy in dataframe
accuracy_df=pd.DataFrame()
accuracy_df["Embedding_type"]=embedding_types
accuracy_df["accuracy"]=accuracies
accuracy_df

