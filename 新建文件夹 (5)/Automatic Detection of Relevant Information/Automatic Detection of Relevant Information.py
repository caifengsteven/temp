import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import spacy
import random
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download necessary NLTK data
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Generate simulated financial news data
def generate_simulated_news(n=100):
    assets = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Tesla', 'Facebook', 'Netflix', 'Bitcoin']
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'FB', 'NFLX', 'BTC']
    
    news_list = []
    labels_relevant = []
    labels_forecast = []
    
    for i in range(n):
        asset = random.choice(assets)
        ticker = tickers[assets.index(asset)]
        
        # Generate different types of sentences
        context_sentences = [
            f"The market has been volatile lately due to economic uncertainty.",
            f"Investors have been watching {asset} closely in recent weeks.",
            f"The tech sector experienced some turbulence last month.",
            f"Many analysts have been discussing the future of {asset}.",
            f"The earnings season has begun with mixed results across the board."
        ]
        
        relevant_sentences = [
            f"{asset} ({ticker}) reported quarterly earnings of ${random.randint(1, 10)}.{random.randint(10, 99)} per share.",
            f"The company's revenue grew by {random.randint(5, 30)}% year-over-year.",
            f"{ticker} stock is trading at ${random.randint(100, 1000)}.{random.randint(10, 99)} as of market close.",
            f"{asset}'s market cap currently stands at ${random.randint(100, 1000)} billion.",
            f"The company announced a dividend of ${random.randint(1, 5)}.{random.randint(10, 99)} per share."
        ]
        
        forecast_sentences = [
            f"Analysts expect {asset} to grow by {random.randint(10, 50)}% in the next fiscal year.",
            f"{ticker} is projected to reach ${random.randint(1000, 2000)} by the end of the year.",
            f"The company will likely increase its dividend by {random.randint(5, 20)}% next quarter.",
            f"Industry experts predict that {asset} will expand into new markets in the coming months.",
            f"{asset} is expected to announce a new product line that could boost revenues significantly."
        ]
        
        # Randomly select sentences to form a news article
        n_context = random.randint(2, 4)
        n_relevant = random.randint(2, 4)
        n_forecast = random.randint(1, 3)
        
        selected_context = random.sample(context_sentences, n_context)
        selected_relevant = random.sample(relevant_sentences, n_relevant)
        selected_forecast = random.sample(forecast_sentences, n_forecast)
        
        # Create the full news article
        all_sentences = selected_context + selected_relevant + selected_forecast
        random.shuffle(all_sentences)
        news_text = ' '.join(all_sentences)
        
        # Create labels for each sentence
        sentences = sent_tokenize(news_text)
        article_relevant_labels = []
        article_forecast_labels = []
        
        for sentence in sentences:
            if sentence in selected_relevant:
                article_relevant_labels.append(1)
            else:
                article_relevant_labels.append(0)
                
            if sentence in selected_forecast:
                article_forecast_labels.append(1)
            else:
                article_forecast_labels.append(0)
        
        news_list.append(news_text)
        labels_relevant.append(article_relevant_labels)
        labels_forecast.append(article_forecast_labels)
    
    return news_list, labels_relevant, labels_forecast

# Function to perform tag processing (financial term detection, homogenization, and replacement)
def process_tags(text):
    # Replace tickers with TICKER tag
    text = re.sub(r'\b[A-Z]{1,5}\b', 'TICKER', text)
    
    # Replace dollar amounts with NUM tag
    text = re.sub(r'\$\d+(\.\d+)?', 'NUM', text)
    
    # Replace percentages with NUM tag
    text = re.sub(r'\d+(\.\d+)?%', 'NUM', text)
    
    # Replace company names with TICKER tag (simplified)
    companies = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Tesla', 'Facebook', 'Netflix', 'Bitcoin']
    for company in companies:
        text = text.replace(company, 'TICKER')
    
    return text

# Function to perform LDA-based relevance detection
def detect_relevant_text(news_list, threshold=0.6):
    # Process tags in each news article
    processed_news = [process_tags(news) for news in news_list]
    
    # Split into sentences
    all_sentences = []
    news_sentence_mapping = []
    
    for i, news in enumerate(processed_news):
        sentences = sent_tokenize(news)
        all_sentences.extend(sentences)
        news_sentence_mapping.extend([i] * len(sentences))
    
    # Apply CountVectorizer
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(all_sentences)
    
    # Apply LDA
    lda = LatentDirichletAllocation(n_components=2, random_state=0)
    lda.fit(X)
    
    # Get topic distributions for each sentence
    topic_distributions = lda.transform(X)
    
    # Identify the relevant topic (the one with more financial terms)
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords_idx = topic.argsort()[:-10-1:-1]
        top_keywords = [feature_names[i] for i in top_keywords_idx]
        topic_keywords.append(top_keywords)
    
    # The topic with more financial terms (TICKER, NUM) is considered the relevant one
    financial_term_counts = [0, 0]
    for topic_idx, keywords in enumerate(topic_keywords):
        for keyword in keywords:
            if keyword in ['ticker', 'num']:
                financial_term_counts[topic_idx] += 1
    
    relevant_topic = financial_term_counts.index(max(financial_term_counts))
    
    # Classify sentences as relevant or not
    relevant_sentences = []
    for i, dist in enumerate(topic_distributions):
        if dist[relevant_topic] > threshold:
            relevant_sentences.append(all_sentences[i])
    
    # Create a binary relevance classification for each sentence
    relevance_classification = []
    for i, dist in enumerate(topic_distributions):
        if dist[relevant_topic] > threshold:
            relevance_classification.append(1)
        else:
            relevance_classification.append(0)
    
    return relevance_classification, relevant_sentences, news_sentence_mapping

# Function to extract temporal features for forecast detection
def extract_temporal_features(sentences):
    features = []
    
    for sentence in sentences:
        # Parse the sentence with spaCy
        doc = nlp(sentence)
        
        # Initialize features
        feature_dict = {
            'has_future_modal': 0,
            'has_prediction_verb': 0,
            'has_future_time': 0,
            'sentence_length': len(doc),
            'has_ticker': 1 if 'TICKER' in sentence else 0,
            'has_num': 1 if 'NUM' in sentence else 0
        }
        
        # Check for future modals
        future_modals = ['will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could']
        for token in doc:
            if token.text.lower() in future_modals:
                feature_dict['has_future_modal'] = 1
        
        # Check for prediction verbs
        prediction_verbs = ['expect', 'predict', 'forecast', 'project', 'estimate', 'anticipate']
        for token in doc:
            if token.lemma_.lower() in prediction_verbs:
                feature_dict['has_prediction_verb'] = 1
        
        # Check for future time expressions
        future_time = ['next', 'upcoming', 'future', 'soon', 'tomorrow', 'coming']
        for token in doc:
            if token.text.lower() in future_time:
                feature_dict['has_future_time'] = 1
        
        features.append(list(feature_dict.values()))
    
    return np.array(features)

# Generate simulated data
n_samples = 200
news_list, labels_relevant, labels_forecast = generate_simulated_news(n_samples)

# Detect relevant text using LDA
relevance_classification, relevant_sentences, news_sentence_mapping = detect_relevant_text(news_list)

# Flatten the ground truth labels
flat_labels_relevant = []
flat_labels_forecast = []

for i, news in enumerate(news_list):
    sentences = sent_tokenize(news)
    flat_labels_relevant.extend(labels_relevant[i])
    flat_labels_forecast.extend(labels_forecast[i])

# Print relevance detection performance
print("Relevance Detection Performance:")
print(classification_report(flat_labels_relevant, relevance_classification))

# Train forecast detection model on relevant sentences
# Identify relevant sentences with their corresponding forecast labels
relevant_indices = [i for i, r in enumerate(relevance_classification) if r == 1]
relevant_forecast_labels = [flat_labels_forecast[i] for i in relevant_indices]

# Extract features for forecast detection
X = extract_temporal_features([sent_tokenize(news_list[news_sentence_mapping[i]])[i % len(sent_tokenize(news_list[news_sentence_mapping[i]]))] for i in relevant_indices])
y = relevant_forecast_labels

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Linear SVC model for forecast detection
forecast_model = LinearSVC(C=0.001, class_weight='balanced', max_iter=1500)
forecast_model.fit(X_train, y_train)

# Evaluate forecast detection
y_pred = forecast_model.predict(X_test)
print("\nForecast Detection Performance:")
print(classification_report(y_test, y_pred))

# Function to apply the full pipeline to a new financial news article
def analyze_financial_news(news_text):
    # Process the news article
    processed_news = process_tags(news_text)
    sentences = sent_tokenize(processed_news)
    
    # Apply relevance detection
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(sentences)
    
    lda = LatentDirichletAllocation(n_components=2, random_state=0)
    lda.fit(X)
    
    topic_distributions = lda.transform(X)
    
    # Determine the relevant topic (similar to above, but simplified)
    relevant_topic = 0  # Assume the first topic is relevant for this example
    
    # Classify sentences
    relevant_indices = []
    for i, dist in enumerate(topic_distributions):
        if dist[relevant_topic] > 0.6:
            relevant_indices.append(i)
    
    # Extract relevant sentences
    relevant_sentences = [sentences[i] for i in relevant_indices]
    
    # Apply forecast detection to relevant sentences
    features = extract_temporal_features(relevant_sentences)
    forecast_predictions = forecast_model.predict(features)
    
    # Create the final output
    result = {
        'original_text': news_text,
        'relevant_sentences': [],
        'forecast_sentences': []
    }
    
    for i, sentence in enumerate(relevant_sentences):
        result['relevant_sentences'].append(sentence)
        if forecast_predictions[i] == 1:
            result['forecast_sentences'].append(sentence)
    
    return result

# Test the full pipeline on a new example
test_news = """
Tesla (TSLA) announced impressive quarterly results yesterday. The company reported earnings of $2.45 per share, beating analyst expectations of $2.10. Revenue came in at $24.6 billion, representing a 28% increase year-over-year. 
The electric vehicle manufacturer's gross margin decreased slightly to 18.2% from 19.5% in the previous quarter. Industry analysts expect Tesla to deliver 1.8 million vehicles in the next fiscal year. The company is likely to face increased competition as traditional automakers expand their electric vehicle offerings.
The stock market has been volatile in recent weeks due to concerns about inflation and interest rates. Many investors are cautious about tech stocks in the current economic environment.
Tesla is projected to launch new models next year that could significantly boost its market share in the premium segment. The company will also likely expand its energy business, which currently represents only 5% of total revenue.
"""

analysis_result = analyze_financial_news(test_news)

print("\nTest Example Analysis:")
print("Relevant sentences:")
for i, sentence in enumerate(analysis_result['relevant_sentences']):
    print(f"{i+1}. {sentence}")

print("\nForecast sentences:")
for i, sentence in enumerate(analysis_result['forecast_sentences']):
    print(f"{i+1}. {sentence}")

    