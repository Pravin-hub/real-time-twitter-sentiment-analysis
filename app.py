import streamlit as st
import pandas as pd
import lit as hf


st.set_page_config(
    page_title="Twitter Sentiment Analyzer", page_icon="📊", layout="wide"
)


adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)


# Whenever the search button is clicked, the search_callback function is called
def analyze_data(up_file, file_type):
    data = hf.convert_to_dataframe(up_file, file_type)
    st.session_state.df=data
    if data is not None:
        st.session_state.df = hf.predict_sentiment(data)
        st.write(data.head())
    else:
        st.write("Unsupported file type. Please upload a CSV or JSON file.")

    


with st.sidebar:
    st.title("Twitter Sentiment Analyzer")

    st.markdown(
        """
        <div style="text-align: justify;">
            This app performs sentiment analysis on the tweets based on 
            the given data. Since the app can only predict positive or 
            negative sentiment, it is more suitable towards analyzing the 
            sentiment of brand, product, service, company, or person. 
            Only English tweets are supported.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Upload Data File")
    uploaded_file = st.sidebar.file_uploader("Upload a data file", type=["csv", "json"])

    if uploaded_file:
        file_type = st.sidebar.radio("Select file type", ["csv", "json"])
        st.sidebar.button("Analyze", on_click=lambda: analyze_data(uploaded_file, file_type))
        st.markdown(
            "Note: it may take a while to load the results, especially with large number of tweets"
        )

    st.markdown("[Github link](https://github.com/tmtsmrsl/TwitterSentimentAnalyzer)")
    st.markdown("Created by Mayank Goyal\nAditya Kasar\nBrajendra Pathak\nPravin Kumar")


if "df" in st.session_state:

    def make_dashboard(tweet_df, bar_color, wc_color):
        # first row
        col1, col2, col3 = st.columns([28, 34, 38])
        with col1:
            sentiment_plot = hf.plot_sentiment(tweet_df)
            sentiment_plot.update_layout(height=350, title_x=0.5)
            st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)
        with col2:
            top_unigram = hf.get_top_n_gram(tweet_df, ngram_range=(1, 1), n=10)
            unigram_plot = hf.plot_n_gram(
                top_unigram, title="Top 10 Occuring Words", color=bar_color
            )
            unigram_plot.update_layout(height=350)
            st.plotly_chart(unigram_plot, theme=None, use_container_width=True)
        with col3:
            top_bigram = hf.get_top_n_gram(tweet_df, ngram_range=(2, 2), n=10)
            bigram_plot = hf.plot_n_gram(
                top_bigram, title="Top 10 Occuring Bigrams", color=bar_color
            )
            bigram_plot.update_layout(height=350)
            st.plotly_chart(bigram_plot, theme=None, use_container_width=True)

        # second row
        col1, col2 = st.columns([60, 40])
        with col1:

            def sentiment_color(sentiment):
                if sentiment == "Positive":
                    return "background-color: #1F77B4; color: white"
                else:
                    return "background-color: #FF7F0E"

            st.dataframe(
                tweet_df[["Sentiment", "text"]].style.applymap(
                    sentiment_color, subset=["Sentiment"]
                ),
                height=350,
            )
        with col2:
            wordcloud = hf.plot_wordcloud(tweet_df, colormap=wc_color)
            st.pyplot(wordcloud)

    adjust_tab_font = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
    </style>
    """

    st.write(adjust_tab_font, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["All", "Positive 😊", "Negative ☹️"])
    with tab1:
        tweet_df = st.session_state.df
        make_dashboard(tweet_df, bar_color="#54A24B", wc_color="Greens")
    with tab2:
        tweet_df = st.session_state.df.query("Sentiment == 'Positive'")
        make_dashboard(tweet_df, bar_color="#1F77B4", wc_color="Blues")
    with tab3:
        tweet_df = st.session_state.df.query("Sentiment == 'Negative'")
        make_dashboard(tweet_df, bar_color="#FF7F0E", wc_color="Oranges")
