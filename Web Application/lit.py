import numpy as np
import pandas as pd
import string
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
nltk.download(
    ["stopwords", "punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "universal_tagset"]
)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image


# Create a custom plotly theme and set it as default
pio.templates["custom"] = pio.templates["plotly_white"]
pio.templates["custom"].layout.margin = {"b": 25, "l": 25, "r": 25, "t": 50}
pio.templates["custom"].layout.width = 600
pio.templates["custom"].layout.height = 450
pio.templates["custom"].layout.autosize = False
pio.templates["custom"].layout.font.update(
    {"family": "Arial", "size": 12, "color": "#707070"}
)
pio.templates["custom"].layout.title.update(
    {
        "xref": "container",
        "yref": "container",
        "x": 0.5,
        "yanchor": "top",
        "font_size": 16,
        "y": 0.95,
        "font_color": "#353535",
    }
)
pio.templates["custom"].layout.xaxis.update(
    {"showline": True, "linecolor": "lightgray", "title_font_size": 14}
)
pio.templates["custom"].layout.yaxis.update(
    {"showline": True, "linecolor": "lightgray", "title_font_size": 14}
)
pio.templates["custom"].layout.colorway = [
    "#1F77B4",
    "#FF7F0E",
    "#54A24B",
    "#D62728",
    "#C355FA",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#FFE323",
    "#17BECF",
]
pio.templates.default = "custom"


def convert_to_dataframe(uploaded_file, file_type):
    if file_type == "csv":
        data = pd.read_csv(uploaded_file)
    elif file_type == "json":
        data = pd.read_json(uploaded_file)
    else:
        data = None
    return data


def text_preprocessing(text):
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = r'@[^\s]+'
    stopword = set(stopwords.words('english'))
    # Lower Casing
    tweet = text.lower()
    tweet=tweet[1:]
    # Removing all URls 
    tweet = re.sub(urlPattern,'',tweet, flags=re.S)
    # Removing all @username.
    tweet = re.sub(userPattern,'', tweet, flags=re.S) 
    #Remove punctuations
    tweet = tweet.translate(str.maketrans("","",string.punctuation))
    #tokenizing words
    tokens = word_tokenize(tweet)
    #Removing Stop Words
    final_tokens = [w for w in tokens if w not in stopword]
    #reducing a word to its word stem 
    wordLemm = WordNetLemmatizer()
    finalwords=[]
    for w in final_tokens:
    	if len(w)>1:
           word = wordLemm.lemmatize(w)
           finalwords.append(word)
    return ' '.join(finalwords)

def predict_sentiment(tweet_df):
    # Load the pre-trained model
    model = load_model('rnn_model.hdf5')
    
    max_words = 5000 
    tokenizer = Tokenizer(num_words=max_words)
    # Preprocess the input text in the DataFrame
    temp_df = tweet_df.copy()
    temp_df["Cleaned Tweet"] = temp_df["text"].apply(text_preprocessing) 
    # Remove empty and NaN rows
    temp_df = temp_df.dropna(subset=["Cleaned Tweet"]).reset_index(drop=True)
    max_words = 5000 
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(temp_df["Cleaned Tweet"])
    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences(temp_df["Cleaned Tweet"])
    sequences = pad_sequences(sequences, maxlen=200)
    # Make sentiment predictions
    scores = model.predict(sequences)
    temp_df["Score"] = scores
    
    # Assign sentiment labels
    temp_df["Sentiment"] = np.where(temp_df["Score"] >= 0.50, "Positive", "Negative")
    
    return temp_df


def plot_sentiment(tweet_df):
    sentiment_count = tweet_df["Sentiment"].value_counts()
    fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
        hole=0.3,
        title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        color_discrete_map={"Positive": "#1F77B4", "Negative": "#FF7F0E"},
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_wordcloud(tweet_df, colormap="Greens"):
    stopwords = []
    with open("en_stopwords_viz.txt", "r") as file:
        for word in file:
            stopwords.append(word.rstrip("\n"))
    cmap = mpl.cm.get_cmap(colormap)(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[10:15])
    mask = np.array(Image.open("twitter_mask.png"))
    font = "quartzo.ttf"
    text = " ".join(tweet_df["Cleaned Tweet"])
    wc = WordCloud(
        background_color="white",
        font_path=font,
        stopwords=stopwords,
        max_words=90,
        colormap=cmap,
        mask=mask,
        random_state=42,
        collocations=False,
        min_word_length=2,
        max_font_size=200,
    )
    wc.generate(text)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud", fontdict={"fontsize": 16}, fontweight="heavy", pad=20, y=1.0)
    return fig


def get_top_n_gram(tweet_df, ngram_range, n=10):
    stopwords = []
    with open("en_stopwords_viz.txt", "r") as file:
        for word in file:
            stopwords.append(word.rstrip("\n"))
    corpus = tweet_df["Cleaned Tweet"]
    vectorizer = CountVectorizer(
        analyzer="word", ngram_range=ngram_range, stop_words=stopwords
    )
    X = vectorizer.fit_transform(corpus.astype(str).values)
    words = vectorizer.get_feature_names_out()
    words_count = np.ravel(X.sum(axis=0))
    df = pd.DataFrame(zip(words, words_count))
    df.columns = ["words", "counts"]
    df = df.sort_values(by="counts", ascending=False).head(n)
    df["words"] = df["words"].str.title()
    return df


def plot_n_gram(n_gram_df, title, color="#54A24B"):
    fig = px.bar(
        x=n_gram_df.counts,
        y=n_gram_df.words,
        title="<b>{}</b>".format(title),
        text_auto=True,
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    return fig
