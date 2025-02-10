import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

def rate_url_validity(user_query: str, url: str) -> dict:
    """
    Evaluates the validity of a given URL by computing various metrics including
    domain trust, content relevance, fact-checking, bias, and citation scores.

    Args:
        user_query (str): The user's original query.
        url (str): The URL to analyze.

    Returns:
        dict: A dictionary containing scores for different validity aspects.
    """

    # === Step 1: Fetch Page Content ===
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = " ".join([p.text for p in soup.find_all("p")])  # Extract paragraph text
    except Exception as e:
        return {"error": f"Failed to fetch content: {str(e)}"}

    # === Step 2: Domain Authority Check (Moz API) ===
    # Replace with actual Moz API call
    domain_trust = 60  # Placeholder value (Scale: 0-100)

    # === Step 3: Content Relevance (Semantic Similarity using Hugging Face) ===
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Changed model
    similarity_score = util.pytorch_cos_sim(model.encode(user_query), model.encode(page_text)).item() * 100

    # === Step 4: Fact-Checking (Google Fact Check API) ===
    fact_check_score = check_facts(page_text)

    # === Step 5: Bias Detection (NLP Sentiment Analysis) ===
    sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
    sentiment_result = sentiment_pipeline(page_text[:512])[0]  # Process first 512 characters
    bias_score = 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    # === Step 6: Citation Check (Google Scholar via SerpAPI) ===
    citation_count = check_google_scholar(url)
    citation_score = min(citation_count * 10, 100)  # Normalize

    # === Step 7: Compute Final Validity Score ===
    final_score = (
        (0.3 * domain_trust) +
        (0.3 * similarity_score) +
        (0.2 * fact_check_score) +
        (0.1 * bias_score) +
        (0.1 * citation_score)
    )

    return {
        "Domain Trust": domain_trust,
        "Content Relevance": similarity_score,
        "Fact-Check Score": fact_check_score,
        "Bias Score": bias_score,
        "Citation Score": citation_score,
        "Final Validity Score": final_score
    }
