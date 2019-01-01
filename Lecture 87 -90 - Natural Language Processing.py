
# coding: utf-8

# In[ ]:


get_ipython().system(' conda install nltk')


# In[ ]:


import nltk


# In[ ]:


pwd


# In[ ]:


messages = [line.rstrip() for line in open ('SMSSpamcollection')]


# In[ ]:


print (len(messages))


# In[ ]:


for num, message in enumerate(messages[:10]):
    print (num,message)
    print('\n')


# In[6]:


import pandas


# In[ ]:


messages = pandas.read_csv('SMSSpamcollection',
                           sep='\t', names = ['lables','message'])


# In[ ]:


messages.head()


# In[ ]:


messages.describe()


# In[12]:


messages.info()


# In[13]:


messages.groupby('lables').describe()


# In[14]:


messages['length'] = messages['message'].apply(len)
messages.head()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


messages['length'].plot(bins=10,kind='hist')


# In[16]:


messages['length'].describe()


# In[17]:


# Here we are trying to identify the sentence who's length is 910. Called the message
messages[messages['length']==910]


# In[49]:


messages[messages['length']==910]['message'].iloc[0]


# In[12]:


messages.hist(column='length', by ='lables',bins=50,figsize=(10,4))


# In[18]:


# Part 3

import string


# In[19]:


mess = 'Sample message! Notice: it has punctuation'


# In[20]:


string.punctuation


# In[22]:


nopunc = [char for char in mess if char not in string.punctuation]


# In[24]:


nopunc = ''.join(nopunc)


# In[25]:


nopunc


# In[26]:


from nltk.corpus import stopwords


# In[27]:


stopwords.words('english')[0:10]


# In[63]:


nopunc.split()


# In[64]:


clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[65]:


clean_mess


# In[28]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[29]:


messages.head()


# In[30]:


# Tokenization - Process of converting normal text strings into tokens (which are the words we actually want)

messages['message'].head(5).apply(text_process)


# In[36]:


# Part 4 Vectorization

from sklearn.feature_extraction.text import CountVectorizer


# In[39]:


bow_transformer = CountVectorizer(analyzer=text_process)


# In[40]:


bow_transformer.fit(messages['message'])


# In[41]:


message4 = messages['message'][3]


# In[42]:


print(message4)


# In[43]:


bow4 = bow_transformer.transform([message4])


# In[44]:


print(bow4)


# In[49]:


print(bow_transformer.get_feature_names()[4068])


# In[1]:


messages_bow = bow_transformer.transform(messages['message'])


# In[ ]:


print ('Shape of Sparse Matrix: ', messages_bow.shape)
print ('Amount of Non-Zero occurences: ', messages_bow.nnz)
print ('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

