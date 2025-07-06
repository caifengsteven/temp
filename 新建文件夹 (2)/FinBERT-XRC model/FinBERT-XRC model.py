import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TF_IDF_Vectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.tokenize import sent_tokenize
import re
import random
from tqdm import tqdm
import warnings
from wordcloud import WordCloud
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')
nltk.download('punkt')

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
set_seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simulate 10-K MD&A sections data
def generate_financial_sentences(risk_level, num_sentences=50):
    """
    Generate simulated financial sentences for 10-K MD&A sections
    
    Parameters:
    risk_level (str): 'risky' or 'non-risky'
    num_sentences (int): Number of sentences to generate
    
    Returns:
    list: List of simulated sentences
    """
    # Common financial terms
    common_terms = [
        "revenue", "profit", "loss", "growth", "decline", "market", "customers", 
        "business", "operations", "strategy", "financial", "performance", "industry",
        "competition", "expenses", "cash flow", "assets", "liabilities", "debt",
        "investment", "dividend", "shareholders", "regulatory", "compliance"
    ]
    
    # Risk-specific terms (more likely to appear in risky companies)
    risky_terms = [
        "uncertainty", "decline", "challenging", "difficult", "risk", "litigation",
        "lawsuit", "default", "debt", "loss", "volatility", "decrease", "failure",
        "concern", "investigation", "competitive pressure", "reduced", "disruption",
        "threat", "plaintiffs", "aggressive", "downturn", "restructuring", "deficit",
        "liquidity concerns", "material weakness", "insufficient"
    ]
    
    # Non-risky terms (more likely to appear in non-risky companies)
    non_risky_terms = [
        "growth", "increase", "improvement", "strong", "success", "opportunity",
        "innovation", "leadership", "stable", "expansion", "progress", "advantage",
        "efficient", "profit", "forward", "meaningful", "reliable", "consistent",
        "secure", "sustainable", "positive", "robust", "healthy", "beneficial"
    ]
    
    # Sentence templates
    templates = [
        "Our {} has shown {} during the fiscal year.",
        "We experienced {} in our {} compared to the previous year.",
        "The company's {} remains {} despite market conditions.",
        "Management expects {} for the upcoming {} in the next fiscal year.",
        "Our {} strategy resulted in {} for our shareholders.",
        "The {} market conditions have led to {} in our business operations.",
        "We continue to focus on {} to ensure {} for our business.",
        "The company has implemented {} measures to address {} in our operations.",
        "Our investment in {} has resulted in {} for our company.",
        "The board of directors has approved {} to enhance our {} position."
    ]
    
    sentences = []
    
    # Probability of using risk-specific terms
    p_risky = 0.7 if risk_level == 'risky' else 0.2
    
    for _ in range(num_sentences):
        template = random.choice(templates)
        
        # Select terms based on risk level
        if random.random() < p_risky:
            term1 = random.choice(risky_terms if risk_level == 'risky' else non_risky_terms)
            term2 = random.choice(common_terms + (risky_terms if risk_level == 'risky' else non_risky_terms))
        else:
            term1 = random.choice(common_terms)
            term2 = random.choice(common_terms)
        
        sentence = template.format(term1, term2)
        sentences.append(sentence)
    
    return sentences

def generate_mda_section(risk_level):
    """Generate a complete MD&A section"""
    num_sentences = random.randint(30, 50)
    sentences = generate_financial_sentences(risk_level, num_sentences)
    return " ".join(sentences)

# Generate synthetic dataset
def generate_dataset(num_samples=1000):
    """Generate synthetic 10-K dataset with risk labels"""
    data = []
    
    for year in range(2008, 2014):
        # Generate non-risky samples (approx. 75%)
        for i in range(int(num_samples * 0.75)):
            mda_text = generate_mda_section('non-risky')
            data.append({
                'year': year,
                'mda_text': mda_text,
                'risk_label': 0  # non-risky
            })
        
        # Generate risky samples (approx. 25%)
        for i in range(int(num_samples * 0.25)):
            mda_text = generate_mda_section('risky')
            data.append({
                'year': year,
                'mda_text': mda_text,
                'risk_label': 1  # risky
            })
    
    return pd.DataFrame(data)

# Create synthetic dataset
print("Generating synthetic dataset...")
df = generate_dataset(num_samples=200)  # Reduced for demo purposes
print(f"Dataset shape: {df.shape}")
print(f"Risk distribution: {df['risk_label'].value_counts()}")

# Split data by year as described in the paper
def prepare_dataset_by_year(df, test_year):
    train_df = df[df['year'] < test_year]
    test_df = df[df['year'] == test_year]
    return train_df, test_df

# FinBERT-XRC model implementation
class FinBERTXRC(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', freeze_bert=False):
        super(FinBERTXRC, self).__init__()
        
        # Load pre-trained BERT model (in a real scenario, we would use FinBERT)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Bidirectional GRU for sentence encoding
        self.hidden_size = self.bert.config.hidden_size
        self.bi_gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            bidirectional=True,
            batch_first=True
        )
        
        # Sentence-level attention mechanism
        self.attention_dim = 100
        self.attention_weight = nn.Linear(self.hidden_size * 2, self.attention_dim)
        self.attention_vector = nn.Parameter(torch.Tensor(self.attention_dim, 1))
        nn.init.xavier_uniform_(self.attention_vector)
        
        # Output layer
        self.fc = nn.Linear(self.hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, get_attention=False):
        batch_size = input_ids.shape[0]
        max_sentences = input_ids.shape[1]
        max_tokens = input_ids.shape[2]
        
        # Reshape for BERT processing
        input_ids = input_ids.view(-1, max_tokens)
        attention_mask = attention_mask.view(-1, max_tokens)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, max_tokens)
        
        # Get BERT embeddings and attention
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
        
        # Extract CLS token embeddings (sentence embeddings)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        sentence_embeddings = sentence_embeddings.view(batch_size, max_sentences, -1)
        
        # Get attention weights (for word-level explanation)
        if get_attention:
            # Last layer attention
            word_attention = outputs.attentions[-1]
            # Reshape and sum across attention heads
            word_attention = word_attention.mean(dim=1)  # Average across attention heads
            word_attention = word_attention.view(batch_size, max_sentences, max_tokens, max_tokens)
            # Sum attention values for each word (summed over columns)
            summed_word_attention = word_attention.sum(dim=3)
        
        # Apply bidirectional GRU to get context-aware sentence encoding
        self.bi_gru.flatten_parameters()
        gru_out, _ = self.bi_gru(sentence_embeddings)
        
        # Sentence-level attention mechanism
        u = torch.tanh(self.attention_weight(gru_out))
        att = torch.matmul(u, self.attention_vector).squeeze(-1)
        att_score = F.softmax(att, dim=1)
        
        # Compute document embedding using attention weights
        doc_embedding = torch.bmm(att_score.unsqueeze(1), gru_out).squeeze(1)
        
        # Final prediction
        logits = self.fc(doc_embedding)
        output = self.sigmoid(logits)
        
        if get_attention:
            return output, att_score, summed_word_attention
        else:
            return output, att_score, None

# Custom dataset for handling 10-K documents
class FinancialDocumentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_sentences=50, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_sentences = max_sentences
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Split document into sentences
        sentences = sent_tokenize(text)
        
        # Truncate or pad to max_sentences
        if len(sentences) > self.max_sentences:
            sentences = sentences[:self.max_sentences]
        else:
            sentences += [""] * (self.max_sentences - len(sentences))
        
        # Tokenize each sentence
        encodings = [self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ) for sentence in sentences]
        
        # Stack tensors
        input_ids = torch.stack([encoding['input_ids'].squeeze(0) for encoding in encodings])
        attention_mask = torch.stack([encoding['attention_mask'].squeeze(0) for encoding in encodings])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.float)
        }

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=5):
    best_val_f1 = 0
    train_losses = []
    val_losses = []
    val_f1_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs, _, _ = model(input_ids, attention_mask)
            outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs, _, _ = model(input_ids, attention_mask)
                outputs = outputs.squeeze()
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_preds.extend((outputs > 0.5).float().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_f1 = f1_score(val_labels, val_preds)
        val_f1_scores.append(val_f1)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'finbert_xrc_best.pt')
    
    return train_losses, val_losses, val_f1_scores

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs, _, _ = model(input_ids, attention_mask)
            outputs = outputs.squeeze()
            
            all_preds.extend((outputs > 0.5).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return f1, report, conf_matrix

# Generate explainable outputs at word, sentence, and corpus levels
def generate_explanations(model, test_loader, tokenizer, test_texts):
    model.eval()
    document_predictions = []
    sentence_attentions = []
    word_attentions = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Generating explanations")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs, sent_attention, word_attention = model(input_ids, attention_mask, get_attention=True)
            outputs = outputs.squeeze()
            
            document_predictions.extend((outputs > 0.5).float().cpu().numpy())
            sentence_attentions.append(sent_attention.cpu().numpy())
            word_attentions.append(word_attention.cpu().numpy())
    
    # Process a sample document for explanation
    sample_idx = 0  # First test document
    sample_text = test_texts[sample_idx]
    sentences = sent_tokenize(sample_text)
    
    # Get predictions and attentions for this document
    pred = document_predictions[sample_idx]
    sent_attn = sentence_attentions[0][sample_idx]
    word_attn = word_attentions[0][sample_idx]
    
    # 1. Word-level explanation
    # Get sentences with highest attention
    top_sent_indices = np.argsort(sent_attn)[-5:]  # Top 5 sentences
    
    print("\n--- Word-Level Explanation ---")
    print(f"Document predicted as: {'Risky' if pred == 1 else 'Non-risky'}")
    
    # Get attention for words in top sentences
    for sent_idx in top_sent_indices:
        if sent_idx < len(sentences):
            sent = sentences[sent_idx]
            tokens = tokenizer.tokenize(sent)
            
            # Get attention for this sentence
            if sent_idx < word_attn.shape[0]:
                # Truncate to actual tokens length
                token_attn = word_attn[sent_idx][:len(tokens)+2]  # +2 for [CLS] and [SEP]
                
                # Ignore [CLS] and [SEP] tokens
                token_attn = token_attn[1:-1]
                token_attn = token_attn[:len(tokens)]  # Ensure lengths match
                
                # Highlight words with high attention
                highlighted_sent = []
                for i, token in enumerate(tokens):
                    if i < len(token_attn):
                        if token_attn[i] > 1.1:  # Threshold as mentioned in the paper
                            highlighted_sent.append(f"**{token}**")
                        else:
                            highlighted_sent.append(token)
                
                print(f"\nSentence {sent_idx} (Attention: {sent_attn[sent_idx]:.4f}):")
                print("Original:", sent)
                print("Highlighted:", " ".join(highlighted_sent).replace(" ##", ""))
    
    # 2. Sentence-level explanation
    print("\n--- Sentence-Level Explanation ---")
    for sent_idx in top_sent_indices:
        if sent_idx < len(sentences):
            print(f"Sentence {sent_idx} (Attention: {sent_attn[sent_idx]:.4f}):")
            print(sentences[sent_idx])
    
    # 3. Corpus-level explanation (Word Cloud)
    # Collect important words from all documents
    risky_words = []
    non_risky_words = []
    
    for doc_idx in range(len(document_predictions)):
        pred = document_predictions[doc_idx]
        doc_text = test_texts[doc_idx]
        doc_sentences = sent_tokenize(doc_text)
        
        # Skip if index out of range
        if doc_idx >= len(sentence_attentions[0]):
            continue
            
        sent_attn = sentence_attentions[0][doc_idx]
        word_attn = word_attentions[0][doc_idx]
        
        for sent_idx, sent in enumerate(doc_sentences):
            # Skip if index out of range
            if sent_idx >= sent_attn.shape[0] or sent_idx >= word_attn.shape[0]:
                continue
                
            if sent_attn[sent_idx] > 0.025:  # Sentence attention threshold from the paper
                tokens = tokenizer.tokenize(sent)
                
                # Get attention for this sentence
                token_attn = word_attn[sent_idx]
                
                # Ignore [CLS] and [SEP] tokens
                if len(token_attn) > 2:
                    token_attn = token_attn[1:-1]
                    token_attn = token_attn[:len(tokens)]  # Ensure lengths match
                    
                    # Collect words with high attention
                    for i, token in enumerate(tokens):
                        if i < len(token_attn) and token_attn[i] > 1.1:  # Word attention threshold
                            if pred == 1:
                                risky_words.append(token.replace("##", ""))
                            else:
                                non_risky_words.append(token.replace("##", ""))
    
    # Generate word clouds
    print("\n--- Corpus-Level Explanation ---")
    print(f"Collected {len(risky_words)} risky words and {len(non_risky_words)} non-risky words")
    
    # Count word frequencies
    risky_word_counts = {}
    for word in risky_words:
        if word in risky_word_counts:
            risky_word_counts[word] += 1
        else:
            risky_word_counts[word] = 1
    
    non_risky_word_counts = {}
    for word in non_risky_words:
        if word in non_risky_word_counts:
            non_risky_word_counts[word] += 1
        else:
            non_risky_word_counts[word] = 1
    
    # Filter out common stopwords and punctuation
    stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'with', 'as', 'be', 'or'])
    risky_word_counts = {k: v for k, v in risky_word_counts.items() if k not in stopwords and len(k) > 1}
    non_risky_word_counts = {k: v for k, v in non_risky_word_counts.items() if k not in stopwords and len(k) > 1}
    
    # Create word clouds
    if risky_word_counts:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Risky word cloud
        wc_risky = WordCloud(width=800, height=400, 
                           background_color='white', 
                           colormap='Reds', 
                           max_words=100).generate_from_frequencies(risky_word_counts)
        ax1.imshow(wc_risky, interpolation='bilinear')
        ax1.set_title('Risky Words Word Cloud')
        ax1.axis('off')
        
        # Non-risky word cloud
        wc_non_risky = WordCloud(width=800, height=400, 
                               background_color='white', 
                               colormap='Blues', 
                               max_words=100).generate_from_frequencies(non_risky_word_counts)
        ax2.imshow(wc_non_risky, interpolation='bilinear')
        ax2.set_title('Non-Risky Words Word Cloud')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('word_clouds.png')
        plt.close()
        print("Word clouds saved to 'word_clouds.png'")

# Implement baseline model: TF-IDF with Logistic Regression
def train_tfidf_baseline(train_texts, train_labels, test_texts, test_labels):
    vectorizer = TF_IDF_Vectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)
    
    y_pred = model.predict(X_test)
    f1 = f1_score(test_labels, y_pred)
    report = classification_report(test_labels, y_pred)
    
    return f1, report

# Main execution
def main():
    test_year = 2013  # Example test year
    
    # Prepare dataset
    train_df, test_df = prepare_dataset_by_year(df, test_year)
    print(f"Training data: {train_df.shape}, Test data: {test_df.shape}")
    
    # TF-IDF baseline
    print("\n--- TF-IDF Baseline Model ---")
    tfidf_f1, tfidf_report = train_tfidf_baseline(
        train_df['mda_text'].tolist(), 
        train_df['risk_label'].tolist(),
        test_df['mda_text'].tolist(),
        test_df['risk_label'].tolist()
    )
    print(f"TF-IDF Baseline F1 Score: {tfidf_f1:.4f}")
    print(tfidf_report)
    
    # Initialize FinBERT-XRC model
    model = FinBERTXRC().to(device)
    tokenizer = model.tokenizer
    
    # Prepare datasets for PyTorch
    train_dataset = FinancialDocumentDataset(
        train_df['mda_text'].tolist(),
        train_df['risk_label'].tolist(),
        tokenizer
    )
    
    # Split train data into train and validation
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=train_df['risk_label']
    )
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Small batch size for demonstration
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=val_sampler
    )
    
    test_dataset = FinancialDocumentDataset(
        test_df['mda_text'].tolist(),
        test_df['risk_label'].tolist(),
        tokenizer
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4
    )
    
    # Training parameters
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-6},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert')], 'lr': 1e-3}
    ])
    criterion = nn.BCELoss()
    
    # Train model
    print("\n--- Training FinBERT-XRC Model ---")
    train_losses, val_losses, val_f1_scores = train_model(
        model, train_loader, val_loader, optimizer, criterion, num_epochs=3  # Reduced for demonstration
    )
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()
    
    # Load best model
    model.load_state_dict(torch.load('finbert_xrc_best.pt'))
    
    # Evaluate model
    print("\n--- Evaluating FinBERT-XRC Model ---")
    f1, report, conf_matrix = evaluate_model(model, test_loader)
    print(f"FinBERT-XRC F1 Score: {f1:.4f}")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Risky', 'Risky'], 
                yticklabels=['Non-Risky', 'Risky'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Generate explanations
    print("\n--- Generating Explanations ---")
    generate_explanations(model, test_loader, tokenizer, test_df['mda_text'].tolist())
    
    # Compare with baseline
    print("\n--- Model Comparison ---")
    print(f"TF-IDF Baseline F1 Score: {tfidf_f1:.4f}")
    print(f"FinBERT-XRC F1 Score: {f1:.4f}")
    print(f"Improvement: {(f1 - tfidf_f1) * 100:.2f}%")

if __name__ == "__main__":
    main()