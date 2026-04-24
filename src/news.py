import feedparser
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# FinBERT model loaded lazily inside main() to avoid slow download on module import.
finbert_model = None
finbert_tokenizer = None

labels = ['Positive', 'Negative', 'Neutral']

def fetch_news(query, num_articles=10, max_age_minutes=120, fetch_content=True):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

    articles = []
    for item in feed.entries[:num_articles]:
        title = item.title
        link = item.link
        published = item.published

        try:
            published_dt = datetime(*item.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            continue

        if published_dt < cutoff:
            continue

        content = fetch_article_content(link) if fetch_content else ""
        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "published_dt": published_dt,
            "content": content
        })

    return articles

def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content.strip()
    except requests.RequestException:
        return "Content not retrieved."

def analyze_sentiment(text):

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    polarity = scores['compound']

    # analysis = TextBlob(text)
    # polarity = analysis.sentiment.polarity

    if polarity > 0.05:
        sentiment = 'Positive'
    elif polarity < -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return polarity, sentiment

# def analyze_sentiment(text):
#     if not text.strip():
#         return 0.0, 'Neutral'

#     inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = finbert_model(**inputs)

#     logits = outputs.logits
#     probabilities = torch.softmax(logits, dim=1).numpy()[0]
#     max_index = np.argmax(probabilities)
#     sentiment = labels[max_index]
#     confidence = probabilities[max_index]

#     return confidence, sentiment


def summarize_sentiments(articles):
    summary = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    for article in articles:
        # print("-"*25)
        # print(f"\n--- Analyzing Article: {article['title']} ---")
        # print(f"Published: {article['published']}")
        # print(article['content'])
        _, sentiment = analyze_sentiment(article['title']) # + " " + article['content'])
        summary[sentiment] += 1

    total = len(articles)
    print("\n--- Market Sentiment Summary ---")
    print(f"Total articles analyzed: {total}")
    for sentiment, count in summary.items():
        percent = (count / total) * 100
        print(f"{sentiment}: {count} ({percent:.2f}%)")

def main():
    global finbert_model, finbert_tokenizer
    finbert_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

    queries = [
    "EURUSD news now",
    "EUR USD breaking news",
    "EURUSD live updates",
    "EURUSD price move news",
    "EURUSD reaction news",
    "EURUSD market moving news",
    "EURUSD volatility news now",
    "EURUSD spike news",
    "EURUSD drop news",
    "euro dollar breaking news",
    "USD breaking news forex",
    "euro breaking news forex",
    "forex market breaking news EUR USD",
    "EURUSD central bank comments live",
    "ECB comments live euro impact",
    "Fed comments live USD impact",
    "US economic data release EURUSD reaction",
    "Eurozone data release EURUSD reaction",
    "EURUSD headlines now",
    "forex live headlines EUR USD"
  ]
    num_articles_per_query = 10
    all_articles = []

    for query in queries:
        print(f"Fetching news articles for '{query}'...\n")
        articles = fetch_news(query, num_articles_per_query)
        all_articles.extend(articles)

    for idx, article in enumerate(all_articles, 1):
        print(f"Article {idx}: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published']}")

        polarity, sentiment = analyze_sentiment(article['title'])  # or article['content']
        print(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})\n")

    summarize_sentiments(all_articles)

if __name__ == "__main__":
    main()