#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
pd.set_option('display.max_rows', None)


# In[1]:


alphabets=['ऄ','अ','आ','इ','ई','उ','ऊ','ऋ','ऌ','ऍ','ऎ','ए','ऐ','ऑ','ऒ','ओ','औ','क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट',
'ठ','ड','ढ','ण','त','थ','द','ध','न','ऩ','प','फ','ब','भ','म','य','र','ऱ','ल','ळ','ऴ','व','श','ष','स','ह']

vowels=['ऄ','अ','आ','इ','ई','उ','ऊ','ऋ','ऌ','ऍ','ऎ','ए','ऐ','ऑ','ऒ','ओ','औ','अ:']

consonant=['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट',
'ठ','ड','ढ','ण','त','थ','द','ध','न','ऩ','प','फ','ब','भ','म','य','र','ऱ','ल','ळ','ऴ','व','श','ष','स','ह']

matra=['ऀ','ँ','ं','ः','ऺ','ऻ','़','ा','ि','ी','ु','ू','ृ','ॄ','ॅ','ॆ','े','ै','ॉ','ॊ','ो','ौ','्','ॎ','ॏ','ॕ','ॖ','ॗ']


# In[3]:


def tokenizing(li):
    i=0
    new_li=[]
    while i< (len(li)-1):
        if li[i]=='्':
            i+=1
            continue
        if li[i] in consonant  and li[i+1] in alphabets:
            new_li.append(li[i]+'्')
            new_li.append('अ')
        else:
            if li[i] in consonant:
                new_li.append(li[i]+'्')
            else:
                new_li.append(li[i])
        i=i+1
    new_li.append(li[i])
    if li[i] in consonant:
        new_li[-1]+='्'
        new_li.append('अ')
    return new_li


# In[5]:


def regex(line):
    string=''
    for i in line:
        if i in alphabets+matra:
            string=string+i
        else:
            string=string+" "
    return " ".join(string.split())


# In[6]:


def ngram_list(element_list,ngram=1,ngram_type='char'):
    ans_lst=[]
    ngram_word=''
    try:
        for i in range(len(element_list)-ngram+1):
            for j in range(ngram):
                ngram_word+=element_list[i]
                if ngram_type=='word':
                    ngram_word+=" "
                i+=1
            i -=ngram-1
            #print(eng_word)
            ans_lst.append(ngram_word.strip())
            ngram_word=''
        #lst.append(letters)
    except:
        print("out of bound",i)
    return ans_lst


# In[7]:


def ngram_dict(ngram_elements,ngram_list):
    for pk in ngram_list:
        ngram_elements[pk] = ngram_elements.get(pk,0)+1
    return ngram_elements


# In[8]:


#syllable
def syllable_list(word):  
    syl_list=[]
    oddtoken=[i for i in word]
    eventoken=[]
    for i in range(len(oddtoken)):
        if oddtoken[i] not in matra:
            eventoken.append(oddtoken[i])
        else:
            eventoken[-1]=eventoken[-1]+oddtoken[i]
    li=eventoken
    i=0
    while i<len(li):
        if li[i]!=li[i][0]+'्':
            #print(li[i])
            syl_list.append(li[i])
            i+=1
        else:
            flag=True
            s=''
            j=0
            while flag:
                if li[i]!=li[i][0]+'्':
                    flag=False
                    i=i+1
                else:
                    if j>=1:
                        s+=li[i+1]
                        j+=1
                    else:
                        j+=1
                        s+=li[i]+li[i+1]
                    i+=1
            syl_list.append(s)
            #print(s)
    return syl_list


# In[4]:


def top100(ngram_dict,ngram_type,df):
    top100_ngram=[]
    top100_freq=[]
    i=0
    for key,val in ngram_dict.items():
        if i==100:
            break
        i+=1
        top100_ngram.append(key)
        top100_freq.append(val)
    df[ngram_type]=top100_ngram
    df[ngram_type+"freq"]=top100_freq
    return df


# # Q.3a. Uni,Bi,Tri,Quad - grams for characters

# In[9]:


import os


# In[10]:


unigram_chars,bigram_chars,trigram_chars,quadgram_chars={},{},{},{}
for f in os.listdir("./New folder/"):
    print(f)
    file=open('./New folder/'+f,encoding='utf-8')
    for line in file:
        line=regex(line)
        line=line.split()
        for word in line:
            #print(word)
            word=tokenizing(word)
            
            #unigram characters
            unigrams_list=ngram_list(word)
            unigram_chars = ngram_dict(unigram_chars,unigrams_list)
            #unigram_chars = dict(Counter(unigrams_list)+Counter(unigram_chars))
                
            #bigram characters
            bigrams_list=ngram_list(word,2)
            bigram_chars = ngram_dict(bigram_chars,bigrams_list)
            #bigram_chars = dict(Counter(bigrams_list)+Counter(bigram_chars))
            
            #trigram characters
            trigrams_list=ngram_list(word,3)
            trigram_chars = ngram_dict(trigram_chars,trigrams_list)
            #trigram_chars = dict(Counter(trigrams_list)+Counter(trigram_chars))
            
            #quadgram characters
            quadgrams_list=ngram_list(word,4)
            quadgram_chars = ngram_dict(quadgram_chars,quadgrams_list)
            #quadgram_chars = dict(Counter(quadgrams_list)+Counter(quadgram_chars))
    file.close() 


# In[30]:


#sorting and taking top 100 unigrams
unigram_chars = dict(sorted(unigram_chars.items(), key=lambda item: item[1],reverse=True))
ngram_char_df = top100(unigram_chars,"unigram",pd.DataFrame())
ngram_char_df


# In[31]:


#padding empty rows
lis = ['-']*20
df = pd.DataFrame({'unigram':lis,'unigramfreq':lis})
ngram_char_df = pd.concat([ngram_char_df,df],axis =0,ignore_index=True)
ngram_char_df


# In[23]:


#sorting and taking top 100 bigrams
bigram_chars = dict(sorted(bigram_chars.items(), key=lambda item: item[1],reverse=True))
print(len(bigram_chars))
ngram_char_df = top100(bigram_chars,"bigram",ngram_char_df)
ngram_char_df


# In[24]:


#sorting and taking top 100 trigrams
trigram_chars = dict(sorted(trigram_chars.items(), key=lambda item: item[1],reverse=True))
print(len(trigram_chars))
ngram_char_df = top100(trigram_chars,"trigram",ngram_char_df)
ngram_char_df


# In[25]:


#sorting and taking top 100 quadgrams
quadgram_chars = dict(sorted(quadgram_chars.items(), key=lambda item: item[1],reverse=True))
print(len(quadgram_chars))
ngram_char_df = top100(quadgram_chars,"quadgram",ngram_char_df)
ngram_char_df


# In[28]:


ngram_char_df.to_csv("top100_ngram_char.csv")
ngram_char_df


# # Q.3b.unigrams, bigrams and trigrams for words

# In[15]:


unigram_words,bigram_words,trigram_words,quadgram_words={},{},{},{}
for ik in os.listdir("./New folder/"):
    file=open('./New folder/'+ik,encoding='utf-8')
    for line in file:
        #print(line)
        
        #preprocessing
        line=regex(line)
        line=line.split()
        
        #unigram words
        unigram_list=ngram_list(line,1,'word')
        unigram_words=ngram_dict(unigram_words,unigram_list)
        
        #bigrams and trigrams not computed as my laptop is not supporting such intensive tasks
        """#bigram words
        bigram_list=ngram_list(line,2,'word')
        bigram_words=ngram_dict(bigram_words,bigram_list)
        
        #trigram words
        trigram_list=ngram_list(line,3,'word')
        trigram_words=ngram_dict(trigram_words,trigram_list)"""

        #print(line_num)
    file.close()


# In[32]:


unigram_words=dict(sorted(unigram_words.items(), key=lambda item: item[1],reverse=True))
ngram_word_df = top100(unigram_words,"unigram",pd.DataFrame())
ngram_word_df


# In[33]:


ngram_word_df.to_csv("top100_ngram_word.csv")


# # Q.3c.unigrams, bigrams and trigrams for syllables.

# In[19]:


unigram_syl,bigram_syl,trigram_syl,quadgram_syl={},{},{},{}
for ik in os.listdir("./New folder/"):
    file=open('./New folder/'+ik,encoding='utf-8')
    for line in file:
        
        line=regex(line)
        line=line.split()
        #print(line)
        for word in line:
            word_syl=syllable_list(word)
            #print(word_syl)
            
        #unigram characters
            unigrams_list=ngram_list(word_syl)
            #print(unigrams_list)
            unigram_syl = ngram_dict(unigram_syl,unigrams_list)
            #print(unigram_syl)
                
        #bigram characters
            bigrams_list=ngram_list(word_syl,2)
            #print(bigrams_list)
            bigram_syl = ngram_dict(bigram_syl,bigrams_list)
            #print(bigram_syl)
            
        #trigram characters
            trigrams_list=ngram_list(word_syl,3)
            #print(trigrams_list)
            trigram_syl = ngram_dict(trigram_syl,trigrams_list)
            #print(trigram_syl)
        
    file.close()


# In[35]:


print(len(unigram_syl))
unigram_syl=dict(sorted(unigram_syl.items(), key=lambda item: item[1],reverse=True))
ngram_syl_df = top100(unigram_syl,"unigram",pd.DataFrame())
ngram_syl_df


# In[36]:


print(len(bigram_syl))
bigram_syl=dict(sorted(bigram_syl.items(), key=lambda item: item[1],reverse=True))
ngram_syl_df = top100(bigram_syl,"bigram",ngram_syl_df)
ngram_syl_df


# In[37]:


print(len(trigram_syl))
trigram_syl=dict(sorted(trigram_syl.items(), key=lambda item: item[1],reverse=True))
ngram_syl_df = top100(trigram_syl,"trigram",ngram_syl_df)
ngram_syl_df


# In[38]:


ngram_syl_df.to_csv("top100_ngram_syl.csv")


# # Q.3.d  Zipfian distribution

# For character unigrams

# In[57]:


#character distribution
import matplotlib.pyplot as plt 
import numpy as np
x=np.array(list(range(1, 81)))
y=list(ngram_word_df["unigramfreq"])[:80]
y=np.array(list(map(int, y)))
plt.plot(x,y)
plt.show()


# This frequency distribution for char unigram is similar to Zipfian distribution. So this follows Zips law

# For Word Unigrams

# In[50]:


#word distribution
x=np.array(list(range(1, 101)))
y = list(map(int, list(ngram_word_df["unigramfreq"])))
y=np.array(y)
plt.plot(x,y)
plt.show()


# This frequency distribution for word unigram is similar to Zipfian distribution. So this follows Zips law

# For syllable unigrams

# In[51]:


#Syllable distribution
x=np.array(list(range(1, 101)))
y = list(map(int, list(ngram_syl_df["unigramfreq"])))
y=np.array(y)
plt.plot(x,y)
plt.show()


# This frequency distribution for syllable unigram is similar to Zipfian distribution. So this follows Zips law
