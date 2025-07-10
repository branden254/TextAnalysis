import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk

try:
    # Download  NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except Exception as e:
    print(f"Warning: Unable to download NLTK resources automatically: {e}")
    print("You may need to download them manually using nltk.download()")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from wordcloud import WordCloud

try:
    from transformers import pipeline

    transformers_available = True
except ImportError:
    print("Warning: transformers package not available. Summarization will use fallback method.")
    transformers_available = False
from sklearn.cluster import KMeans
import os
import warnings

warnings.filterwarnings('ignore')

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except Exception as e:
    print(f"Warning: Error downloading NLTK resources: {e}")
    print("Please run the following commands manually:")
    print(
        "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')")

# Load spacy model for advanced NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Installing spaCy model...")
    import os

    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# 1. Define Entity Recognition
def define_entity_recognition():
    """
    Define and explain entity recognition
    """
    definition = """
    Named Entity Recognition (NER) is a natural language processing technique that identifies and classifies named entities in text into predefined categories such as:

    - Person names
    - Organizations
    - Locations
    - Date and time expressions
    - Quantities and monetary values
    - Percentages

    NER helps extract structured information from unstructured text, enabling applications to understand who, what, where, when, and how much is being discussed. It uses various techniques including:

    - Rule-based approaches using pattern matching and gazetteers
    - Statistical models like Conditional Random Fields (CRF)
    - Neural network architectures, particularly Bidirectional LSTMs with CRF layers
    - Transformer models like BERT, RoBERTa, and others fine-tuned for NER tasks

    Entity recognition is fundamental to many NLP applications including information extraction, question answering, text summarization, content recommendation, and knowledge graph construction.
    """

    print("\n", "-" * 50)
    print("1. ENTITY RECOGNITION DEFINITION")
    print("-" * 50)
    print(definition)
    return definition


# 2. Extract Entities from Text
def extract_entities(dataframe):
    """
    Extract entities from text data using spaCy
    """
    print("\n", "-" * 50)
    print("2. ENTITY EXTRACTION RESULTS")
    print("-" * 50)

    # Combine all text for analysis
    all_text = " ".join(dataframe['story'].dropna().tolist())

    # Process with spacy
    doc = nlp(all_text)

    # Extract entities by category
    locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
    organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    people = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
    amounts = [ent.text for ent in doc.ents if ent.label_ in ['MONEY', 'QUANTITY', 'PERCENT']]

    # Count frequencies
    location_counts = Counter(locations)
    org_counts = Counter(organizations)
    people_counts = Counter(people)
    date_counts = Counter(dates)
    amount_counts = Counter(amounts)

    # Print results
    print("\nTop Locations:")
    for loc, count in location_counts.most_common(10):
        print(f"- {loc}: {count}")

    print("\nTop Organizations:")
    for org, count in org_counts.most_common(10):
        print(f"- {org}: {count}")

    print("\nTop People:")
    for person, count in people_counts.most_common(10):
        print(f"- {person}: {count}")

    print("\nTop Date References:")
    for date, count in date_counts.most_common(10):
        print(f"- {date}: {count}")

    print("\nTop Amounts/Quantities:")
    for amount, count in amount_counts.most_common(10):
        print(f"- {amount}: {count}")

    # Visualize entity distribution
    entity_types = ['Locations', 'Organizations', 'People', 'Dates', 'Amounts']
    entity_counts = [len(locations), len(organizations), len(people), len(dates), len(amounts)]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=entity_types, y=entity_counts)
    plt.title('Distribution of Named Entities in Text Data')
    plt.ylabel('Count')
    plt.savefig('entity_distribution.png')
    plt.close()

    # Visualize top entities of each type
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()

    # Locations
    top_locations = dict(location_counts.most_common(10))
    sns.barplot(x=list(top_locations.keys()), y=list(top_locations.values()), ax=axes[0])
    axes[0].set_title('Top 10 Locations')
    axes[0].tick_params(axis='x', rotation=45)

    # Organizations
    top_orgs = dict(org_counts.most_common(10))
    sns.barplot(x=list(top_orgs.keys()), y=list(top_orgs.values()), ax=axes[1])
    axes[1].set_title('Top 10 Organizations')
    axes[1].tick_params(axis='x', rotation=45)

    # People
    top_people = dict(people_counts.most_common(10))
    sns.barplot(x=list(top_people.keys()), y=list(top_people.values()), ax=axes[2])
    axes[2].set_title('Top 10 People')
    axes[2].tick_params(axis='x', rotation=45)

    # Dates
    top_dates = dict(date_counts.most_common(10))
    sns.barplot(x=list(top_dates.keys()), y=list(top_dates.values()), ax=axes[3])
    axes[3].set_title('Top 10 Date References')
    axes[3].tick_params(axis='x', rotation=45)

    # Amounts
    top_amounts = dict(amount_counts.most_common(10))
    sns.barplot(x=list(top_amounts.keys()), y=list(top_amounts.values()), ax=axes[4])
    axes[4].set_title('Top 10 Amounts/Quantities')
    axes[4].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('entity_details.png')
    plt.close()

    return {
        'locations': location_counts,
        'organizations': org_counts,
        'people': people_counts,
        'dates': date_counts,
        'amounts': amount_counts
    }


# 3. Perform Topic Modeling
def perform_topic_modeling(dataframe, num_topics=5):
    """
    Perform topic modeling using LDA and NMF
    """
    print("\n", "-" * 50)
    print("3. TOPIC MODELING RESULTS")
    print("-" * 50)

    # Preprocessing
    try:
        stop_words = set(stopwords.words('english'))
    except:
        print("Warning: stopwords not available. Using a basic stopword list.")
        # Basic English stopwords if NLTK's not available
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                      'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                      'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                      'to', 'from', 'in', 'on', 'by', 'at', 'with', 'not', 'be', 'have', 'had',
                      'has', 'do', 'does', 'did', 'doing', 'can', 'would', 'should', 'could',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}

    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        try:
            # Lowercase
            text = text.lower()
            # Remove punctuation and numbers
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)

            # Safe tokenization
            try:
                tokens = word_tokenize(text)
            except LookupError:
                # Fallback simple tokenization if NLTK tokenizer fails
                tokens = text.split()

            # Remove stopwords and short words
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return ""

    # Apply preprocessing
    print("Preprocessing text data...")
    dataframe['processed_text'] = dataframe['story'].apply(preprocess_text)

    # Create document-term matrix
    print("Creating document-term matrix...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
    tfidf = tfidf_vectorizer.fit_transform(dataframe['processed_text'].values.astype('U'))

    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # LDA Model
    print("Fitting LDA model...")
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(tfidf)

    # NMF Model
    print("Fitting NMF model...")
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(tfidf)

    # Print LDA topics
    print("\nLDA Topics:")
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic {topic_idx + 1}:")
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        print(f"- {', '.join(top_features)}")

    # Print NMF topics
    print("\nNMF Topics:")
    for topic_idx, topic in enumerate(nmf_model.components_):
        print(f"Topic {topic_idx + 1}:")
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        print(f"- {', '.join(top_features)}")

    # Visualize topics
    try:
        print("Generating topic visualizations...")
        fig, axes = plt.subplots(num_topics, 2, figsize=(16, 4 * num_topics))

        for i, (lda_topic, nmf_topic) in enumerate(zip(lda_model.components_, nmf_model.components_)):
            # LDA topic
            lda_top_features_ind = lda_topic.argsort()[:-11:-1]
            lda_top_features = [feature_names[j] for j in lda_top_features_ind]
            lda_weights = lda_topic[lda_top_features_ind]

            # NMF topic
            nmf_top_features_ind = nmf_topic.argsort()[:-11:-1]
            nmf_top_features = [feature_names[j] for j in nmf_top_features_ind]
            nmf_weights = nmf_topic[nmf_top_features_ind]

            # Plot LDA
            axes[i, 0].barh(lda_top_features, lda_weights)
            axes[i, 0].set_title(f'LDA Topic {i + 1}')
            axes[i, 0].set_xlabel('Weight')

            # Plot NMF
            axes[i, 1].barh(nmf_top_features, nmf_weights)
            axes[i, 1].set_title(f'NMF Topic {i + 1}')
            axes[i, 1].set_xlabel('Weight')

        plt.tight_layout()
        plt.savefig('topic_modeling.png')
        plt.close()

        # Generate wordclouds for LDA topics
        try:
            print("Generating topic wordclouds...")
            fig, axes = plt.subplots(1, num_topics, figsize=(20, 4))

            for i, topic in enumerate(lda_model.components_):
                # Get top 50 words for wordcloud
                top_features_ind = topic.argsort()[:-51:-1]
                top_features = {feature_names[j]: topic[j] for j in top_features_ind}

                # Generate wordcloud
                wordcloud = WordCloud(width=400, height=400, background_color='white',
                                      prefer_horizontal=1.0, colormap='viridis')
                wordcloud.generate_from_frequencies(top_features)

                # Plot
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'Topic {i + 1}')
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig('topic_wordclouds.png')
            plt.close()
        except Exception as e:
            print(f"Error generating wordclouds: {e}")
    except Exception as e:
        print(f"Error generating topic visualizations: {e}")

    return {
        'lda_model': lda_model,
        'nmf_model': nmf_model,
        'feature_names': feature_names,
        'tfidf': tfidf
    }


# 4. Summarize text using transformer models
def summarize_text(dataframe):
    """
    Summarize text using transformers
    """
    print("\n", "-" * 50)
    print("4. TEXT SUMMARIZATION")
    print("-" * 50)

    # Check if transformers can be imported
    try:
        from transformers import pipeline
        # Initialize summarization pipeline with explicit model
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        using_transformers = True
        print("Using sshleifer/distilbart-cnn-12-6 model for summarization")
    except Exception as e:
        print(f"Transformer models not available: {str(e)}")
        print("Using extractive summarization as fallback method.")
        using_transformers = False

    # Process each document
    summaries = []

    for idx, row in dataframe.iterrows():
        if not isinstance(row['story'], str) or len(row['story'].strip()) == 0:
            summaries.append("No text available for summarization.")
            continue

        text = row['story']

        if using_transformers:
            # Handle potential token limit by chunking
            if len(text) > 1000:
                text = text[:1000]  # Simplistic approach - in production would use better chunking

            try:
                summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                print(f"Falling back to extractive summarization for document {idx} due to: {str(e)}")
                # Fallback to basic extractive summarization
                summary = extractive_summarize(text)
                summaries.append(summary)
        else:
            # Fallback to basic extractive summarization
            summary = extractive_summarize(text)
            summaries.append(summary)

    # Add summaries to dataframe
    dataframe['summary'] = summaries

    # Print some example summaries
    print("\nExample summaries:")
    for i in range(min(5, len(dataframe))):
        print(f"\nTitle: {dataframe.iloc[i]['title']}")
        print(f"Summary: {dataframe.iloc[i]['summary']}")

    return summaries
def extractive_summarize(text, sentences=3):
    """
    Basic extractive summarization as a fallback when transformers aren't available
    """
    # Tokenize into sentences
    sentences = sent_tokenize(text)

    # Clean and tokenize text
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = word_tokenize(clean_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Calculate word frequencies
    word_freq = Counter(filtered_words)

    # Score sentences based on word frequency
    sent_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if i in sent_scores:
                    sent_scores[i] += word_freq[word]
                else:
                    sent_scores[i] = word_freq[word]

    # Get top n sentences
    n = min(sentences, len(sentences))
    top_sentences = sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    top_sentences = sorted(top_sentences, key=lambda x: x[0]) 

    # Combine sentences into summary
    summary = ' '.join([sentences[i] for i, _ in top_sentences])
    return summary


# 5. Apply unsupervised machine learning for classification
def classify_texts(dataframe):
    """
    Classify texts into predefined categories using unsupervised techniques
    """
    print("\n", "-" * 50)
    print("5. UNSUPERVISED TEXT CLASSIFICATION")
    print("-" * 50)

    # Define the categories
    categories = [
        'Advanced Gene Editing',
        'mRNA Technology',
        'CAR-T Cell Therapy',
        'Organoids and Tissue Engineering',
        'Single-Cell Genomics',
        'Synthetic Biology',
        'Biological Computing',
        'Wearable Biosensors',
        'Microbiome Therapeutics',
        'Nanomedicine'
    ]

    # Create keyword dictionaries for each category
    category_keywords = {
        'Advanced Gene Editing': ['crispr', 'cas9', 'gene editing', 'genetic engineering', 'genome editing',
                                  'gene therapy', 'dna editing', 'genetic modification'],
        'mRNA Technology': ['mrna', 'messenger rna', 'rna vaccine', 'ribosomal', 'translation',
                            'transcription', 'nucleotide', 'lipid nanoparticle'],
        'CAR-T Cell Therapy': ['car-t', 'cart', 'chimeric antigen receptor', 't cell', 'immunotherapy',
                               'lymphocyte', 'leukemia', 'receptor'],
        'Organoids and Tissue Engineering': ['organoid', 'tissue engineering', 'bioprinting', '3d printing',
                                             'scaffold', 'culture', 'artificial organ', 'stem cell'],
        'Single-Cell Genomics': ['single cell', 'genomics', 'sequencing', 'rna seq', 'transcriptomics',
                                 'cellular heterogeneity', 'cell atlas'],
        'Synthetic Biology': ['synthetic biology', 'biohacking', 'genetic circuit', 'metabolic engineering',
                              'biosynthesis', 'artificial gene', 'biodesign'],
        'Biological Computing': ['biocomputing', 'dna computing', 'molecular computing', 'neural interface',
                                 'bioinformatics', 'protein folding', 'computational biology'],
        'Wearable Biosensors': ['biosensor', 'wearable', 'implantable', 'glucose monitor', 'patch',
                                'continuous monitoring', 'health tracking'],
        'Microbiome Therapeutics': ['microbiome', 'gut bacteria', 'probiotic', 'prebiotic', 'microbiota',
                                    'bacterial therapy', 'fecal transplant'],
        'Nanomedicine': ['nanomedicine', 'nanoparticle', 'drug delivery', 'nanotechnology',
                         'targeted therapy', 'quantum dot', 'nanorobot']
    }

    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataframe['story'].fillna('').values.astype('U'))
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Calculate similarity scores for each document to each category
    category_scores = {}

    for category, keywords in category_keywords.items():
        # Create a category document by joining keywords
        category_doc = ' '.join(keywords)
        category_vec = tfidf_vectorizer.transform([category_doc])

        # Calculate cosine similarity between each document and category
        similarity_scores = cosine_similarity(tfidf_matrix, category_vec).flatten()
        category_scores[category] = similarity_scores

    # Create dataframe with scores
    scores_df = pd.DataFrame(category_scores)

    # Normalize scores for each document (row)
    normalized_scores = scores_df.copy()
    row_sums = normalized_scores.sum(axis=1)
    for col in normalized_scores.columns:
        normalized_scores[col] = normalized_scores[col] / row_sums

    # Calculate overall category ratios
    category_ratios = normalized_scores.mean()

    # Print category ratios
    print("\nCategory Distribution Ratios:")
    for category, ratio in category_ratios.items():
        print(f"- {category}: {ratio:.4f}")

    # Visualize category distribution
    plt.figure(figsize=(14, 6))
    sns.barplot(x=category_ratios.index, y=category_ratios.values)
    plt.title('Distribution of Categories in the Text Data')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('category_distribution.png')
    plt.close()

    # Assign primary category to each document
    dataframe['primary_category'] = normalized_scores.idxmax(axis=1)

    # Count documents by primary category
    category_counts = dataframe['primary_category'].value_counts()

    # Visualize document distribution by primary category
    plt.figure(figsize=(14, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Document Count by Primary Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('document_categories.png')
    plt.close()

    # Add scores to original dataframe
    for category in categories:
        dataframe[f'score_{category}'] = normalized_scores[category]

    print("\nDocument count by primary category:")
    print(category_counts)

    return {
        'category_ratios': category_ratios,
        'document_categories': category_counts,
        'category_scores': normalized_scores
    }


# 6. Extract and explain emerging trends
def extract_trends(dataframe, topic_results):
    """
    Extract emerging trends from the text data
    """
    print("\n", "-" * 50)
    print("6. EMERGING TRENDS ANALYSIS")
    print("-" * 50)

    # Get time information if available
    if 'date' in dataframe.columns:
        dataframe['date'] = pd.to_datetime(dataframe['date'], errors='coerce')
        dataframe['year'] = dataframe['date'].dt.year
        dataframe['month'] = dataframe['date'].dt.month

        # Group by time and analyze category evolution
        if 'primary_category' in dataframe.columns:
            time_trends = dataframe.groupby(['year', 'month'])['primary_category'].value_counts().unstack().fillna(0)

            # Normalize to show percentage
            time_trends_pct = time_trends.div(time_trends.sum(axis=1), axis=0) * 100

            # Plot trend over time
            plt.figure(figsize=(16, 8))

            # Check if we have enough time points
            if len(time_trends_pct) > 1:
                time_trends_pct.plot(kind='line', marker='o')
                plt.title('Category Trends Over Time')
                plt.ylabel('Percentage (%)')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                plt.savefig('category_trends.png')
                plt.close()

                print("\nCategory trends over time detected:")
                # Identify growing and declining trends
                if len(time_trends_pct) >= 3:  # Need at least 3 time points for trend
                    for category in time_trends_pct.columns:
                        if category in time_trends_pct.columns:
                            values = time_trends_pct[category].values
                            if len(values) >= 3:
                                first_half = values[:len(values) // 2].mean()
                                second_half = values[len(values) // 2:].mean()
                                change = second_half - first_half

                                if change > 5:  # 5% increase
                                    print(f"- Rising trend: {category} (+{change:.1f}%)")
                                elif change < -5:  # 5% decrease
                                    print(f"- Declining trend: {category} ({change:.1f}%)")
            else:
                print("Not enough time points to detect trends")

    # Analyze emerging keywords
    feature_names = topic_results['feature_names']
    lda_model = topic_results['lda_model']

    # Extract top keywords from topics
    emerging_keywords = set()
    for topic in lda_model.components_:
        top_features_ind = topic.argsort()[:-21:-1]  # Top 20 words
        top_features = [feature_names[i] for i in top_features_ind]
        emerging_keywords.update(top_features)

    # Map keywords to potential trends
    trend_keywords = {
        'Personalized Medicine': ['personalized', 'precision', 'individual', 'custom', 'tailored'],
        'AI in Biotechnology': ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'algorithm'],
        'Sustainable Biotechnology': ['sustainable', 'green', 'environment', 'eco', 'renewable'],
        'Digital Health': ['digital', 'app', 'mobile', 'remote', 'telehealth', 'virtual'],
        'Blockchain in Healthcare': ['blockchain', 'ledger', 'crypto', 'token', 'secure'],
        'Point-of-Care Diagnostics': ['point', 'care', 'diagnostic', 'rapid', 'quick', 'bedside'],
        'Regenerative Medicine': ['regenerative', 'stem', 'cell', 'tissue', 'organ', 'repair'],
        'Antibody Therapeutics': ['antibody', 'monoclonal', 'immunotherapy', 'target'],
        'Brain-Computer Interfaces': ['brain', 'neural', 'interface', 'mind']
    }

    # Score each trend
    trend_scores = {}
    for trend, keywords in trend_keywords.items():
        score = sum(1 for keyword in emerging_keywords if any(k in keyword for k in keywords))
        trend_scores[trend] = score

    # Sort and print trends
    sorted_trends = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)

    print("\nTop emerging trends based on keyword analysis:")
    for trend, score in sorted_trends:
        if score > 0:
            print(f"- {trend} (score: {score})")

    # Visualize top trends
    top_trends = {k: v for k, v in sorted_trends if v > 0}
    if top_trends:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(top_trends.keys()), y=list(top_trends.values()))
        plt.title('Emerging Trends in Biotechnology')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('emerging_trends.png')
        plt.close()

    return {
        'top_trends': dict(sorted_trends),
        'emerging_keywords': list(emerging_keywords)
    }


# 7. Draw conclusions
def draw_conclusions(entity_results, topic_results, classification_results, trend_results):
    """
    Draw conclusions from the analysis
    """
    print("\n", "-" * 50)
    print("7. CONCLUSIONS")
    print("-" * 50)

    # Most mentioned entities
    top_locations = list(entity_results['locations'].most_common(3))
    top_organizations = list(entity_results['organizations'].most_common(3))
    top_people = list(entity_results['people'].most_common(3))

    # Top categories
    top_categories = classification_results['category_ratios'].nlargest(3)

    # Top emerging trends
    top_trends = {k: v for k, v in trend_results['top_trends'].items() if v > 0}
    top_3_trends = list(top_trends.items())[:3]

    # Draw conclusions
    conclusions = [
        f"1. The analysis reveals a strong focus on {', '.join([cat for cat in top_categories.index])} "
        f"with {top_categories.index[0]} being the dominant category.",

        f"2. Key geographical focus is on {', '.join([loc[0] for loc in top_locations])} "
        f"suggesting these regions are at the forefront of biotechnology innovation.",

        f"3. Leading organizations in the field include {', '.join([org[0] for org in top_organizations])}.",

        f"4. Emerging trends indicate growing interest in {', '.join([trend[0] for trend in top_3_trends])} "
        f"which suggests future directions for biotechnology research and development.",

        "5. The topic modeling results reveal interconnected research themes spanning multiple biotechnology subfields, "
        "indicating the multidisciplinary nature of current biotechnology research."
    ]

    print("\nKey conclusions from the analysis:")
    for conclusion in conclusions:
        print(conclusion)
        print()

    return conclusions


# Main function
def main():
    # Path to the Excel file
    file_path = "C:\\Users\\Data Analyst\\Downloads\\text_data.xlsx"

    # Load the data
    print("Loading data from:", file_path)
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with {len(df)} rows")
        print("Columns:", df.columns.tolist())

        # Show sample
        print("\nSample data:")
        print(df.head())

    except Exception as e:
        print(f"Error loading the file: {str(e)}")
        return

    # 1. Define entity recognition
    definition = define_entity_recognition()

    # 2. Extract entities
    entity_results = extract_entities(df)

    # 3. Perform topic modeling
    topic_results = perform_topic_modeling(df)

    # 4. Summarize text
    summary_results = summarize_text(df)

    # 5. Classify texts
    classification_results = classify_texts(df)

    # 6. Extract trends
    trend_results = extract_trends(df, topic_results)

    # 7. Draw conclusions
    conclusions = draw_conclusions(entity_results, topic_results, classification_results, trend_results)

    print("\nAnalysis complete! Output images saved.")


if __name__ == "__main__":
    main()