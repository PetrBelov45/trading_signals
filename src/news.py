"""
News Sentiment Module - Fetch and analyze news sentiment
Supports: NewsAPI, Finnhub, Alpha Vantage, GDELT
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time

import requests
import numpy as np


logger = logging.getLogger(__name__)


def get_sentiment_score(text: str) -> float:
    """
    Get sentiment polarity from text using TextBlob

    Args:
        text: Text to analyze

    Returns:
        Sentiment score (-1 to 1)
    """
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except ImportError:
        logger.warning("TextBlob not available, returning neutral sentiment")
        return 0.0
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return 0.0


class NewsProvider:
    """Base class for news providers"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def fetch_news(
        self,
        ticker: str,
        days: int = 3,
        max_articles: int = 25,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Fetch news articles"""
        raise NotImplementedError


class NewsAPIProvider(NewsProvider):
    """NewsAPI.org provider"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(
        self,
        ticker: str,
        days: int = 3,
        max_articles: int = 25,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI"""
        if not self.api_key:
            logger.warning("NewsAPI key not provided")
            return []

        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            params = {
                'q': ticker,
                'from': from_date,
                'language': language,
                'sortBy': 'relevancy',
                'pageSize': max_articles,
                'apiKey': self.api_key
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'ok':
                logger.warning(f"NewsAPI error: {data.get('message', 'Unknown')}")
                return []

            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', ''),
                    'url': article.get('url', '')
                })

            logger.info(f"Fetched {len(articles)} articles from NewsAPI for {ticker}")
            return articles

        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return []


class FinnhubProvider(NewsProvider):
    """Finnhub provider"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://finnhub.io/api/v1/company-news"

    def fetch_news(
        self,
        ticker: str,
        days: int = 3,
        max_articles: int = 25,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Fetch news from Finnhub"""
        if not self.api_key:
            logger.warning("Finnhub key not provided")
            return []

        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            params = {
                'symbol': ticker,
                'from': from_date,
                'to': to_date,
                'token': self.api_key
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data[:max_articles]:
                articles.append({
                    'title': item.get('headline', ''),
                    'description': item.get('summary', ''),
                    'source': item.get('source', 'Finnhub'),
                    'published_at': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                    'url': item.get('url', '')
                })

            logger.info(f"Fetched {len(articles)} articles from Finnhub for {ticker}")
            return articles

        except Exception as e:
            logger.error(f"Finnhub fetch error: {e}")
            return []


class AlphaVantageNewsProvider(NewsProvider):
    """Alpha Vantage news provider"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_news(
        self,
        ticker: str,
        days: int = 3,
        max_articles: int = 25,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage"""
        if not self.api_key:
            logger.warning("Alpha Vantage key not provided")
            return []

        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.api_key,
                'limit': max_articles
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'feed' not in data:
                logger.warning(f"No news in Alpha Vantage response for {ticker}")
                return []

            cutoff_date = datetime.now() - timedelta(days=days)

            articles = []
            for item in data['feed']:
                pub_date = datetime.strptime(item['time_published'][:8], '%Y%m%d')
                if pub_date >= cutoff_date:
                    articles.append({
                        'title': item.get('title', ''),
                        'description': item.get('summary', ''),
                        'source': item.get('source', 'AlphaVantage'),
                        'published_at': item.get('time_published', ''),
                        'url': item.get('url', '')
                    })

            logger.info(f"Fetched {len(articles)} articles from Alpha Vantage for {ticker}")
            return articles

        except Exception as e:
            logger.error(f"Alpha Vantage news fetch error: {e}")
            return []


class GDELTProvider(NewsProvider):
    """GDELT provider (free, no API key needed)"""

    def __init__(self):
        super().__init__(None)
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch_news(
        self,
        ticker: str,
        days: int = 3,
        max_articles: int = 25,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Fetch news from GDELT"""
        try:
            params = {
                'query': ticker,
                'mode': 'artlist',
                'maxrecords': max_articles,
                'format': 'json',
                'timespan': f'{days}d'
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get('articles', []):
                articles.append({
                    'title': item.get('title', ''),
                    'description': '',  # GDELT doesn't provide descriptions
                    'source': item.get('domain', 'GDELT'),
                    'published_at': item.get('seendate', ''),
                    'url': item.get('url', '')
                })

            logger.info(f"Fetched {len(articles)} articles from GDELT for {ticker}")
            return articles

        except Exception as e:
            logger.error(f"GDELT fetch error: {e}")
            return []


class NewsSentimentAnalyzer:
    """Aggregate news sentiment from multiple sources"""

    def __init__(
        self,
        provider: str = "newsapi",
        api_key: Optional[str] = None
    ):
        """
        Initialize news analyzer

        Args:
            provider: 'newsapi', 'finnhub', 'alphavantage', or 'gdelt'
            api_key: API key for the provider (not needed for GDELT)
        """
        self.provider_name = provider

        if provider == "newsapi":
            self.provider = NewsAPIProvider(api_key)
        elif provider == "finnhub":
            self.provider = FinnhubProvider(api_key)
        elif provider == "alphavantage":
            self.provider = AlphaVantageNewsProvider(api_key)
        elif provider == "gdelt":
            self.provider = GDELTProvider()
        else:
            logger.warning(f"Unknown provider: {provider}, using GDELT")
            self.provider = GDELTProvider()

    def analyze_sentiment(
        self,
        ticker: str,
        days: int = 3,
        max_articles: int = 25,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Analyze news sentiment for ticker

        Args:
            ticker: Stock ticker
            days: Number of days to look back
            max_articles: Maximum articles to fetch
            language: Language code

        Returns:
            Dictionary with sentiment analysis
        """
        logger.info(f"Analyzing news sentiment for {ticker}")

        # Fetch articles
        articles = self.provider.fetch_news(ticker, days, max_articles, language)

        if not articles:
            logger.warning(f"No news articles found for {ticker}")
            return {
                'enabled': False,
                'provider': self.provider_name,
                'window': f'{days}d',
                'score': 0.0,
                'sources': 0,
                'articles_count': 0,
                'top_headlines': [],
                'explanation': 'No news available'
            }

        # Analyze sentiment for each article
        article_sentiments = []
        for article in articles:
            text = article['title'] + ' ' + article.get('description', '')
            polarity = get_sentiment_score(text)

            # Calculate relevance (simple heuristic)
            relevance = 0.9 if ticker.upper() in text.upper() else 0.5

            # Recency factor (more recent = higher impact)
            try:
                pub_date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                age_days = (datetime.now(pub_date.tzinfo) - pub_date).days
                recency = np.exp(-age_days / days)
            except:
                recency = 0.5

            # Source credibility (simplified)
            source_weight = 1.0  # Could be enhanced with source ratings

            # Impact score
            impact = recency * source_weight

            article_sentiments.append({
                'title': article['title'],
                'polarity': polarity,
                'relevance': relevance,
                'impact': impact,
                'score': polarity * relevance * impact,
                'source': article['source']
            })

        # Aggregate sentiment
        if article_sentiments:
            # Weighted average
            total_weight = sum(a['relevance'] * a['impact'] for a in article_sentiments)
            if total_weight > 0:
                aggregate_score = sum(a['score'] for a in article_sentiments) / total_weight
            else:
                aggregate_score = 0.0
        else:
            aggregate_score = 0.0

        # Clip to [-1, 1]
        aggregate_score = np.clip(aggregate_score, -1, 1)

        # Count unique sources
        unique_sources = len(set(a['source'] for a in article_sentiments))

        # Top headlines
        top_headlines = sorted(
            article_sentiments,
            key=lambda x: abs(x['score']),
            reverse=True
        )[:5]

        # Generate explanation
        if abs(aggregate_score) >= 0.5:
            sentiment_label = "strongly positive" if aggregate_score > 0 else "strongly negative"
        elif abs(aggregate_score) >= 0.25:
            sentiment_label = "moderately positive" if aggregate_score > 0 else "moderately negative"
        else:
            sentiment_label = "neutral"

        explanation = f"{sentiment_label.capitalize()} sentiment from {len(article_sentiments)} articles across {unique_sources} sources"

        return {
            'enabled': True,
            'provider': self.provider_name,
            'window': f'{days}d',
            'score': aggregate_score,
            'sources': unique_sources,
            'articles_count': len(article_sentiments),
            'top_headlines': [
                {
                    'title': h['title'],
                    'polarity': h['polarity'],
                    'relevance': h['relevance']
                }
                for h in top_headlines
            ],
            'explanation': explanation
        }
