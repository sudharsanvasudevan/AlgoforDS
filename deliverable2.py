# -*- coding: utf-8 -*-
"""Deliverable2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gVWwygGNpxqzodXAzPqPMJaripyF3wHs
"""

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import csv

class URLValidator:
    def __init__(self):
        self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.fake_news_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
        self.sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

    def fetch_page_content(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return " ".join([p.text for p in soup.find_all("p")])
        except requests.RequestException:
            return ""

    def compute_similarity_score(self, user_query: str, content: str) -> int:
        if not content:
            return 0
        return int(util.pytorch_cos_sim(self.similarity_model.encode(user_query), self.similarity_model.encode(content)).item() * 100)

    def detect_bias(self, content: str) -> int:
        if not content:
            return 50
        sentiment_result = self.sentiment_analyzer(content[:512])[0]
        return 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    def rate_url_validity(self, user_query: str, url: str) -> dict:
        content = self.fetch_page_content(url)
        similarity_score = self.compute_similarity_score(user_query, content)
        bias_score = self.detect_bias(content)
        final_score = (0.7 * similarity_score) + (0.3 * bias_score)
        return {"func_rating": int(final_score)}