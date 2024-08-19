import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import mysql.connector
import pickle

conn = mysql.connector.connect(
 
    host="finalproject.c52a4kiimm6r.ap-south-1.rds.amazonaws.com",
 
    user="admin",
 
    port="3306",
 
    password="******",
 
    database="finalproject"
 
)
curser=conn.cursor()
Data=curser.execute('select * from youtube')
Data=pd.DataFrame(Data)
data=curser.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'youtube' AND TABLE_SCHEMA = 'finalproject' ORDER BY ORDINAL_POSITION")
x=[item[0] for item in data]
Data.columns=x
Data['tags_combined'] = Data['tags'].apply(lambda x: ''.join(x))
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Data['tags_combined'])

kmeans = KMeans(n_clusters=35, random_state=42)
Data['cluster'] = kmeans.fit_predict(tfidf_matrix)

model_package = {
    'Data': Data,
    'tfidf': tfidf,
    'kmeans': kmeans,
    'cosine_similarity':cosine_similarity
    
}

import pickle
with open('model_package.pkl', 'wb') as f:
    pickle.dump(model_package, f)

