#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


movies1 = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies1.head()


# In[4]:


credits.head()


# In[5]:


movies1.shape


# In[6]:


credits.shape


# In[7]:


movies2 = movies1.merge(credits,on='title')


# In[8]:


movies2.head()


# In[9]:


movies2.shape


# In[10]:


movies2.describe()


# In[11]:


movies2.info()


# In[12]:


# genres, id, keywords, title, overview, cast, crew
movies = movies2[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[13]:


movies.head()


# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace=True)


# In[16]:


movies.isnull().sum()


# In[17]:


movies.duplicated().sum()


# In[18]:


movies.iloc[0].genres


# In[19]:


import ast
def convert(obj):
    L =[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[20]:


movies['genres'] = movies['genres'].apply(convert)


# In[21]:


movies.head()


# In[22]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[23]:


movies.head()


# In[24]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[25]:


movies['cast'] = movies['cast'].apply(convert3)


# In[26]:


movies.head()


# In[27]:


movies['crew'][0]


# In[28]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[29]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[30]:


movies.head(1)


# In[31]:


movies['overview'][0]


# In[32]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[33]:


movies.head()


# In[34]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[35]:


movies.head()


# In[36]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[37]:


movies.head()


# In[38]:


new_df = movies[['movie_id', 'title','tags']]


# In[39]:


new_df


# In[40]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[41]:


new_df.head()


# In[42]:


new_df['tags'][0]


# In[43]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[44]:


new_df.head()


# In[45]:


import nltk


# In[46]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[47]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[48]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[49]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[50]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[51]:


vectors


# In[52]:


vectors[0]


# In[53]:


cv.get_feature_names()


# In[54]:


from sklearn.metrics.pairwise import cosine_similarity


# In[55]:


similarity = cosine_similarity(vectors)


# In[56]:


similarity


# In[62]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[64]:


recommend('Batman Begins')


# In[ ]:




