import streamlit as st
import pandas as pd
import pickle

with open('model_package.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

Data = loaded_pipeline['Data']
tfidf = loaded_pipeline['tfidf']
kmeans = loaded_pipeline['kmeans']
cosine_similarity=loaded_pipeline['cosine_similarity']

def recommendations(new, Data, tfidf, kmeans, top_n=10):
    
    new_vec = tfidf.transform([new])
    
    new_cluster = kmeans.predict(new_vec)[0]
    
    similar_videos = Data[Data['cluster'] == new_cluster].copy()

    similar_videos.loc[:, 'similarity'] = cosine_similarity(new_vec, tfidf.transform(similar_videos['tags_combined'])).flatten()

    similar_videos = similar_videos.sort_values(by='similarity', ascending=False)

    return similar_videos[['channelTitle','comment_count','title','thumbnail_url','view_count']].head(top_n)


st.image(r"C:\Users\yoges\Downloads\download.png")
new = st.text_input('Search: ')

st.sidebar.image(r"C:\Users\yoges\Downloads\download (1).png")
st.sidebar.title('Channels')
channels = Data['channelTitle'].unique()
for channel in channels:
    st.sidebar.write(channel)


if st.button('Click'):
    if new:
        recommendations_df = recommendations(new, Data,tfidf,kmeans)
        if not recommendations_df.empty:
            for index, row in recommendations_df.iterrows():
                col_thumb, col_text = st.columns([1, 3])
                with col_thumb:
                    st.image(row['thumbnail_url'], width=150)


                with col_text:
                    st.subheader(f"**{row['title']}**")
                    st.write(f"**Channel**: {row['channelTitle']}")
                    st.write(f"**Comments**: {row['comment_count']}")
                    st.write(f"**Views**: {row['view_count']}")
                    st.write("---")
        else:
            st.write("No Vidios found ")
    else:
        st.write("Please Search again")