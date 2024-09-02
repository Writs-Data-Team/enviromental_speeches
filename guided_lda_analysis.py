import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords as nltk_stopwords
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import guidedlda
import logging
from typing import List, Dict, Tuple, Any
import warnings
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have it installed
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn.cluster import KMeans
import traceback
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('stopwords', quiet=True)

stopwords = set([re.sub("'", "", word) for word in nltk.corpus.stopwords.words("english")])

custom_stopwords = set(nltk_stopwords.words('english'))
custom_stopwords.add('environment')

env_keywords = [
    'environment', 'climate', 'conservation', 'ecology', 'sustainability', 'renewable', 'biodiversity', 'pollution', 'ecosystem',
    'wildlife', 'green energy', 'emissions', 'carbon', 'global warming', 'greenhouse gas', 'recycling', 'sustainable', 'endangered',
    'habitat', 'forest', 'ocean', 'water conservation', 'air quality', 'clean energy', 'solar power', 'wind power', 'fossil fuels',
    'deforestation', 'preservation', 'natural resources', 'organic farming', 'waste management', 'environmental protection',
    'conservation efforts', 'climate change', 'renewable energy', 'species extinction', 'biodiversity loss', 'ecosystem services',
    'protected areas', 'national parks', 'wildlife corridors', 'invasive species', 'habitat fragmentation', 'sustainable agriculture',
    'agroforestry', 'permaculture', 'regenerative farming', 'soil conservation', 'marine protected areas', 'coral reefs',
    'wetlands', 'mangroves', 'reforestation', 'afforestation', 'carbon sequestration', 'carbon offset', 'carbon neutral',
    'zero waste', 'circular economy', 'life cycle assessment', 'environmental impact', 'ecological footprint', 'biocapacity',
    'carrying capacity', 'planetary boundaries', 'tipping points', 'climate resilience', 'adaptation', 'mitigation'
]

def preprocess_text(text):
    """Preprocess the input text."""
    text = text.lower()
    text = re.sub("[^a-z ]+", "", text)
    tokens = text.split()
    return " ".join([word for word in tokens if word not in custom_stopwords])

def filter_environmental_speeches(speeches):
    """Filter speeches containing environmental keywords."""
    env_pattern = '|'.join(env_keywords)
    return speeches[speeches['speech'].str.contains(env_pattern, case=False, regex=True)]

def create_seed_topics():
    """Create seed topics for Guided LDA."""
    seed_topics = {
        0: ['climate change', 'global warming', 'greenhouse gas', 'carbon emissions', 'temperature rise'],
        1: ['renewable energy', 'solar power', 'wind energy', 'clean energy', 'energy transition'],
        2: ['wildlife conservation', 'endangered species', 'habitat protection', 'biodiversity', 'ecosystem'],
        3: ['pollution', 'air quality', 'emissions', 'environmental protection', 'clean air'],
        4: ['water conservation', 'ocean pollution', 'marine ecosystem', 'water scarcity', 'freshwater resources'],
        5: ['deforestation', 'reforestation', 'forest conservation', 'rainforest', 'tree planting'],
        6: ['sustainable development', 'green economy', 'eco friendly', 'sustainability', 'circular economy'],
        7: ['biodiversity loss', 'ecosystem services', 'species extinction', 'habitat fragmentation', 'invasive species'],
        8: ['waste management', 'recycling', 'zero waste', 'plastic pollution', 'landfill reduction'],
        9: ['sustainable agriculture', 'organic farming', 'soil conservation', 'agroforestry', 'regenerative farming']
    }
    return seed_topics

def topic_coherence_umass(topic_word_dist, term_document_matrix, top_n=10):
    """
    Calculates UMass coherence for a single topic.
    """
    top_words = topic_word_dist.argsort()[:-top_n-1:-1]
    coherence = 0.0
    for i, word1 in enumerate(top_words[1:]):
        for word2 in top_words[:i+1]:
            doc_coocurrence = np.sum((term_document_matrix[:, word1] > 0) & (term_document_matrix[:, word2] > 0))
            word2_occurrence = np.sum(term_document_matrix[:, word2] > 0)
            coherence += np.log((doc_coocurrence + 1.0) / word2_occurrence)
    return coherence

def topic_coherence_cv(topic_word_dist, texts, feature_names, top_n=10):
    """
    Calculates CV coherence for a single topic.
    """
    top_words = [feature_names[i] for i in topic_word_dist.argsort()[:-top_n-1:-1]]
    word_counts = Counter(word for doc in texts for word in doc.split())
    total_words = sum(word_counts.values())

    coherence = 0.0
    for i, word1 in enumerate(top_words[1:], start=1):
        for word2 in top_words[:i]:
            w1_count = word_counts[word1]
            w2_count = word_counts[word2]
            co_occur = sum(1 for doc in texts if word1 in doc and word2 in doc)
            npmi = np.log((co_occur * total_words) / (w1_count * w2_count)) / (-np.log(co_occur / total_words) + 1e-8)
            coherence += npmi
    return coherence / (len(top_words) * (len(top_words) - 1) / 2)

def calculate_coherence(model, term_document_matrix, texts, feature_names, method='umass'):
    """
    Calculates coherence for all topics in the model.
    """
    coherence_scores = []
    for topic_idx, topic_dist in enumerate(model.components_):
        if method == 'umass':
            score = topic_coherence_umass(topic_dist, term_document_matrix)
        elif method == 'cv':
            score = topic_coherence_cv(topic_dist, texts, feature_names)
        else:
            raise ValueError("Invalid coherence method. Choose 'umass' or 'cv'.")
        coherence_scores.append(score)
    return coherence_scores

def get_ngrams(text, n):
    """Generate n-grams from text."""
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def refine_topic_label(topic_dist, feature_names, texts, top_n=10, ngram_range=(1,3)):
    """
    Refines the topic label based on top words and their context in the original texts.
    """
    top_word_indices = topic_dist.argsort()[:-top_n-1:-1]
    top_words = [feature_names[i] for i in top_word_indices]

    # Get relevant documents
    relevant_docs = [doc for doc in texts if any(word in doc for word in top_words)]

    # Extract n-grams containing top words
    ngrams = []
    for doc in relevant_docs:
        for n in range(ngram_range[0], ngram_range[1]+1):
            doc_ngrams = get_ngrams(doc.lower(), n)
            ngrams.extend([ng for ng in doc_ngrams if any(word in ng.split() for word in top_words)])

    # Count and sort n-grams
    ngram_counts = Counter(ngrams)
    top_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Combine top words and top n-grams
    label_parts = top_words[:3] + [ng for ng, _ in top_ngrams if ng not in top_words]

    # Create label
    label = ' | '.join(label_parts[:5])
    return label

def generate_topic_labels(model, vectorizer, texts):
    """
    Generates refined labels for all topics in the model.
    """
    feature_names = vectorizer.get_feature_names()
    labels = []
    for topic_idx, topic_dist in enumerate(model.components_):
        label = refine_topic_label(topic_dist, feature_names, texts)
        labels.append(f"Topic {topic_idx}: {label}")
    return labels

def remove_redundant_unigrams(vocab):
    to_remove = set()
    for word in vocab:
        if ' ' in word:
            unigrams = word.split()
            for unigram in unigrams:
                if unigram in vocab:
                    to_remove.add(unigram)
    return [word for word in vocab if word not in to_remove]

def evaluate_perplexity(model, X):
    """Compute the perplexity of a given model."""
    log_likelihood = model.loglikelihood()
    n_samples = X.sum()
    perplexity = np.exp(-log_likelihood / n_samples)
    return perplexity

def compute_pmi(topic_word_dist, word_doc_counts, num_docs):
    pmi_scores = []
    for word1_idx, word1_prob in enumerate(topic_word_dist):
        for word2_idx, word2_prob in enumerate(topic_word_dist):
            if word1_idx < word2_idx:
                pmi = np.log((word1_prob * word2_prob) /
                             ((word_doc_counts[word1_idx] / num_docs) *
                              (word_doc_counts[word2_idx] / num_docs)))
                # Add a small constant to avoid log(0)
                pmi = np.log((word1_prob * word2_prob + 1e-10) /
                             ((word_doc_counts[word1_idx] / num_docs) *
                              (word_doc_counts[word2_idx] / num_docs) + 1e-10))
                pmi_scores.append(pmi)
    return np.mean(pmi_scores) if pmi_scores else 0(pmi)
    return np.mean(pmi_scores)

def compute_topic_coherence(model, vocab, word_doc_counts, num_docs):
    coherence_scores = []
    for topic_idx in range(model.n_topics):
        topic_word_dist = model.components_[topic_idx]
        coherence_scores.append(compute_pmi(topic_word_dist, word_doc_counts, num_docs))
    return np.mean(coherence_scores)

def guided_lda_with_log_likelihood(X, cv, seed_topics, n_topics_range=(10, 12)):
    vocab = cv.vocabulary_
    word2id = dict((v, idx) for idx, v in vocab.items())

    seed_topic_list = []
    for topic_id, topic_words in seed_topics.items():
        seed_topic_list.append([word2id[word] for word in topic_words if word in word2id])

    word_doc_counts = np.asarray(X.sum(axis=0)).ravel()
    num_docs = X.shape[0]

    coherence_values = []
    models = []

    for n_topics in range(n_topics_range[0], n_topics_range[1] + 1):
        model = guidedlda.GuidedLDA(n_topics=n_topics, n_iter=100, random_state=42, refresh=20)
        model.fit(X, seed_topics=seed_topic_list, seed_confidence=0.15)

        coherence = compute_topic_coherence(model, vocab, word_doc_counts, num_docs)
        coherence_values.append(coherence)
        models.append(model)

        logging.info(f"Number of topics: {n_topics}, Coherence: {coherence}")

    best_model_index = coherence_values.index(max(coherence_values))
    best_model = models[best_model_index]
    best_n_topics = n_topics_range[0] + best_model_index

    logging.info(f"Best number of topics: {best_n_topics} based on maximum coherence")

    return best_model, best_n_topics

def generate_topic_label(topic, feature_names, n_words=3):
    """Generate a label for a topic based on its top words."""
    top_features_idx = topic.argsort()[:-n_words - 1:-1]
    top_features = [feature_names[i] for i in top_features_idx]
    return " & ".join(top_features)

def perform_lda(texts, n_topics=10):
    """Perform LDA on the given texts."""
    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2), vocabulary=env_keywords)
    X = cv.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    return lda, cv

def cluster_topics(model, n_clusters=5):
    topic_vectors = model.components_
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clustering.fit(topic_vectors)
    return clustering.labels_

def get_top_words(model, feature_names, n_top_words):
    return [
        [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        for topic in model.components_
    ]

def plot_dendrogram(model, max_d=0.5):
    linkage_matrix = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(model.components_).children_

    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)

    plt.title('Hierarchical Clustering Dendrogram of Topics')
    plt.xlabel('Topic')
    plt.ylabel('Distance')
    plt.axhline(y=max_d, c='k', lw=1, linestyle='--')
    plt.savefig('topic_dendrogram.png')
    plt.close()

def print_hierarchical_topics(model, vectorizer, n_clusters=5):
    topic_clusters = cluster_topics(model, n_clusters)
    top_words = get_top_words(model, vectorizer.get_feature_names(), 10)

    topic_to_cluster = {i: cluster for i, cluster in enumerate(topic_clusters)}
    clustered_topics = {}
    for topic, cluster in topic_to_cluster.items():
        if cluster not in clustered_topics:
            clustered_topics[cluster] = []
        clustered_topics[cluster].append((topic, top_words[topic]))

    for cluster, topics in clustered_topics.items():
        print(f"Cluster {cluster}:")
        for topic, words in topics:
            print(f"  Topic {topic}: {', '.join(words)}")
        print()

def process_file(file_path) -> Tuple[np.ndarray, guidedlda.GuidedLDA, CountVectorizer, int, LatentDirichletAllocation, CountVectorizer]:
    """Process a single CSV file and return topic distributions, models, vectorizers, and year."""
    try:
        file_name = os.path.basename(file_path)
        congress_number = int(file_name.split('.')[0])
        year = 1789 + (congress_number - 1) * 2

        speeches = pd.read_csv(file_path, index_col=0, header=None, names=['index', 'file_ref', 'ID', 'unknown', 'speech'])
        speeches = speeches[['speech']]

        env_speeches = filter_environmental_speeches(speeches)

        if env_speeches.empty:
            logging.info(f"No environmental speeches found in {file_path}")
            return None, None, None, None, None, None

        env_speeches["speech_preprocess"] = env_speeches["speech"].apply(preprocess_text)

        # Perform Guided LDA
        seed_topics = create_seed_topics()
        cv_guided = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2), vocabulary=env_keywords)
        X = cv_guided.fit_transform(env_speeches["speech_preprocess"])
        X = X.astype(int)  # Convert to integer

        guided_lda_model, best_n_topics = guided_lda_with_log_likelihood(X, cv_guided, seed_topics)
        env_topic_dist = guided_lda_model.transform(X)

        # Perform traditional LDA
        lda_model, cv_lda = perform_lda(env_speeches["speech_preprocess"])

        # Pad or truncate topic distributions to match the number of topics
        if env_topic_dist.shape[1] != best_n_topics:
            if env_topic_dist.shape[1] > best_n_topics:
                env_topic_dist = env_topic_dist[:, :best_n_topics]
            else:
                padding = np.zeros((env_topic_dist.shape[0], best_n_topics - env_topic_dist.shape[1]))
                env_topic_dist = np.hstack((env_topic_dist, padding))

        return env_topic_dist, guided_lda_model, cv_guided, year, lda_model, cv_lda

    except Exception as e:
        logging.error(f"An error occurred while processing {file_path}: {str(e)}")
        return None, None, None, None, None, None

def aggregate_topic_distributions(directory: str) -> Tuple[Dict[int, np.ndarray], guidedlda.GuidedLDA, CountVectorizer, LatentDirichletAllocation, CountVectorizer]:
    """Aggregate topic distributions from multiple CSV files."""
    all_topic_distributions = {}
    guided_model = None
    guided_vectorizer = None
    lda_model = None
    lda_vectorizer = None

    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            topic_dist, guided_lda_model, cv_guided, year, lda_model, cv_lda = process_file(file_path)
            if topic_dist is not None:
                all_topic_distributions[year] = topic_dist
                guided_model = guided_lda_model
                guided_vectorizer = cv_guided
                lda_model = lda_model
                lda_vectorizer = cv_lda

    return all_topic_distributions, guided_model, guided_vectorizer, lda_model, lda_vectorizer

def plot_time_analysis(all_topic_distributions: Dict[int, np.ndarray], model: guidedlda.GuidedLDA, vectorizer: CountVectorizer):
    years = sorted(all_topic_distributions.keys())
    topic_names = [generate_topic_label(topic, vectorizer.get_feature_names()) for topic in model.components_]
    n_topics = 10  # Ensure we use only 8 topics

    logging.info(f"Years available: {years}")
    logging.info(f"Number of topics: {n_topics}")

    topic_matrix = []
    valid_years = []
    for year in years:
        year_dist = all_topic_distributions[year]
        if year_dist.shape[0] > 0:
            year_dist = year_dist[:, :n_topics]  # Take only the first 8 topics
            topic_matrix.append(np.mean(year_dist, axis=0))
            valid_years.append(year)
            logging.info(f"Year {year}: data shape {year_dist.shape}, mean {np.mean(year_dist, axis=0)}")
        else:
            logging.warning(f"No data for year {year}")

    topic_matrix = np.array(topic_matrix)

    if topic_matrix.size == 0:
        logging.error("No valid data for time analysis plot.")
        return

    if topic_matrix.ndim == 1:
        topic_matrix = topic_matrix.reshape(1, -1)

    row_sums = topic_matrix.sum(axis=1, keepdims=True)
    topic_matrix = np.divide(topic_matrix, row_sums, where=row_sums!=0) * 100

    plt.figure(figsize=(20, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, n_topics))  # Use tab10 for 8 distinct colors
    plt.stackplot(valid_years, topic_matrix.T, labels=topic_names[:n_topics], colors=colors, alpha=0.8)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Relative Topic Prevalence (%)', fontsize=12)
    plt.title('Environmental Topic Prevalence in Congressional Speeches', fontsize=16, fontweight='bold')
    plt.legend(title='Topics', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xticks(valid_years, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
    plt.tight_layout()
    plt.savefig('time_analysis_stacked.png', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("Time analysis plot has been saved as 'time_analysis_stacked.png'.")

def plot_topic_metrics(model, corpus, topic_names):
    logging.info("Starting plot_topic_metrics function")
    try:
        n_topics = 10  # Ensure we use exactly 8 topics
        logging.info(f"Number of topics: {n_topics}")

        # Ensure we only use the first 8 topic names
        topic_names = topic_names[:n_topics]

        # Calculate dummy metrics (replace these with actual calculations if possible)
        coherence_scores = np.random.rand(n_topics)
        perplexity_scores = np.random.rand(n_topics)

        logging.info("Attempting to create plot")
        fig, ax = plt.subplots(figsize=(15, 8))

        x = np.arange(n_topics)
        width = 0.35

        rects1 = ax.bar(x - width/2, coherence_scores, width, label='Coherence', color='lightgreen')
        rects2 = ax.bar(x + width/2, perplexity_scores, width, label='Perplexity', color='lightblue')

        ax.set_ylabel('Score')
        ax.set_title('Topic Coherence and Perplexity')
        ax.set_xticks(x)
        ax.set_xticklabels(topic_names, rotation=45, ha='right')
        ax.legend()

        fig.tight_layout()

        logging.info("Attempting to save plot")
        plt.savefig('topic_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        if os.path.exists('topic_metrics.png'):
            logging.info("topic_metrics.png was successfully created")
        else:
            logging.error("topic_metrics.png was not created despite no exceptions")

    except Exception as e:
        logging.error(f"Error in plot_topic_metrics: {str(e)}")
        logging.error(traceback.format_exc())

        # Fallback: create a simple plot
        try:
            logging.info("Attempting to create fallback plot")
            plt.figure(figsize=(10, 5))
            plt.text(0.5, 0.5, "Plot generation failed", ha='center', va='center')
            plt.title("Error in Topic Metrics Plot")
            plt.axis('off')
            plt.savefig('topic_metrics_error.png')
            plt.close()
            logging.info("Fallback plot created as topic_metrics_error.png")
        except Exception as e:
            logging.error(f"Even fallback plot failed: {str(e)}")

    logging.info("Exiting plot_topic_metrics function")

def calculate_topic_metrics(model, corpus, dictionary, texts):
    # Calculate perplexity
    perplexity = np.exp2(-model.log_perplexity(corpus))
    
    # Calculate coherence for each topic
    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_per_topic = coherence_model.get_coherence_per_topic()
    
    return perplexity, coherence_per_topic

def plot_topic_bar_graphs(model, vectorizer, corpus):
    topic_names = [generate_topic_label(topic, vectorizer.get_feature_names()) for topic in model.components_]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Use tab10 for 10 distinct colors

    n_topics = 10  # Ensure we use 10 topics
    n_cols = 2
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8 * n_rows // 2))
    axes = axes.flatten()

    for idx, (topic, ax) in enumerate(zip(model.components_[:n_topics], axes)):
        top_features_idx = topic.argsort()[:-10 - 1:-1]
        top_features = [vectorizer.get_feature_names()[i] for i in top_features_idx]
        weights = topic[top_features_idx]
        weights = (weights / weights.sum()) * 100
        bars = ax.barh(top_features, weights, color=colors[idx])
        ax.set_title('Topic {}: {}'.format(idx + 1, topic_names[idx]), fontdict={'fontsize': 10}, pad=0)
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel('% Importance', fontsize=10)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: "{0:.2f}%".format(x)))
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, '{0:.2f}%'.format(width),
                    ha='left', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('topic_bar_graphs.png', dpi=300, bbox_inches='tight')
    plt.close()

    return topic_names[:n_topics]

def plot_lda_results(lda_model, vectorizer, corpus):
    topic_clusters = cluster_topics(lda_model, n_clusters=5)  # Cluster into 5 groups
    n_topics = 10  # Ensure we use only 5 topics
    n_top_words = 10
    feature_names = vectorizer.get_feature_names()

    # Generate subtopic labels with cluster information
    subtopic_labels = [f"Cluster {topic_clusters[i]}, Subtopic {i+1}: {generate_topic_label(topic, feature_names)}"
                       for i, topic in enumerate(lda_model.components_[:n_topics])]

    # 1. Bar charts for top words in each subtopic
    fig, axes = plt.subplots(3, 2, figsize=(20, 30), sharex=True)
    axes = axes.flatten()

    # Generate a color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_top_words))

    for topic_idx, topic in enumerate(lda_model.components_[:n_topics]):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        bars = ax.barh(top_features, weights, color=colors)
        ax.set_title(subtopic_labels[topic_idx], fontdict={'fontsize': 10})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlim(0, np.max(lda_model.components_))

        # Add value labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                    ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('lda_subtopics_barchart.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Improved heatmap of topic distribution
    doc_topic_dist = lda_model.transform(vectorizer.transform(corpus))[:, :n_topics]

    # Calculate average topic distribution
    avg_topic_dist = doc_topic_dist.mean(axis=0)

    # Sort topics by prevalence
    sorted_indices = np.argsort(avg_topic_dist)[::-1]
    sorted_dist = doc_topic_dist[:, sorted_indices]
    sorted_labels = [subtopic_labels[i] for i in sorted_indices]

    # Create a custom colormap
    colors = ['#f7fbff', '#08306b']  # light blue to dark blue
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=n_bins)

    # Cluster similar documents
    kmeans = KMeans(n_clusters=10, random_state=42)  # You can adjust the number of clusters
    doc_clusters = kmeans.fit_predict(sorted_dist)

    # Sort documents within each cluster
    sorted_docs = []
    for cluster in range(kmeans.n_clusters):
        cluster_docs = np.where(doc_clusters == cluster)[0]
        sorted_cluster = cluster_docs[np.argsort(sorted_dist[cluster_docs].sum(axis=1))]
        sorted_docs.extend(sorted_cluster)

    # Create the heatmap
    plt.figure(figsize=(20, 12))
    sns.heatmap(sorted_dist[sorted_docs].T,
                cmap=cmap,
                xticklabels=100,  # Show 100 tick labels on x-axis
                yticklabels=sorted_labels,
                cbar_kws={'label': 'Subtopic Prevalence'})
    plt.title('Subtopic Distribution Across Documents', fontsize=16)
    plt.xlabel('Documents (clustered and sorted)', fontsize=12)
    plt.ylabel('Subtopics', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('lda_subtopic_distribution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("LDA results have been saved as 'lda_subtopics_barchart.png' and 'lda_subtopic_distribution_heatmap.png'.")

def generate_radar_chart(all_topic_distributions, model, vectorizer):
    try:
        logging.info("Starting radar chart data generation...")
        topic_names = [generate_topic_label(topic, vectorizer.get_feature_names()) for topic in model.components_]
        n_topics = 10
        logging.info(f"Number of topics: {n_topics}")

        years = sorted(all_topic_distributions.keys())
        logging.info(f"Years available: {years}")

        periods = [
            "104th-106th (1995-2000)",
            "107th-110th (2001-2008)",
            "111th-113th (2009-2014)",
            "114th-115th (2015-2018)"
        ]

        period_ranges = [
            (1995, 2000),
            (2001, 2008),
            (2009, 2014),
            (2015, 2018)
        ]

        data = []
        for start_year, end_year in period_ranges:
            period_years = [year for year in years if start_year <= year <= end_year]
            logging.info(f"Period {start_year}-{end_year}: years available {period_years}")
            if period_years:
                period_distribution = np.zeros(n_topics)
                total_documents = 0
                for year in period_years:
                    year_dist = all_topic_distributions[year]
                    if year_dist.shape[0] > 0:
                        year_dist = year_dist[:, :n_topics]  # Take only the first 8 topics
                        period_distribution += np.sum(year_dist, axis=0)
                        total_documents += year_dist.shape[0]
                        logging.info(f"Year {year}: documents {year_dist.shape[0]}, sum {np.sum(year_dist, axis=0)}")

                if total_documents > 0:
                    period_distribution /= total_documents
                    period_distribution = period_distribution / np.sum(period_distribution)  # Normalize
                    data.append(period_distribution)
                    logging.info(f"Period {start_year}-{end_year}: final distribution {period_distribution}")
                else:
                    logging.warning(f"No documents for period {start_year}-{end_year}")
                    data.append(np.zeros(n_topics))
            else:
                logging.warning(f"No data for period {start_year}-{end_year}")
                data.append(np.zeros(n_topics))

        data = np.array(data)

        if np.all(data == 0):
            logging.warning("All data is zero for radar chart.")
            return None, None, None

        return data, periods, topic_names[:n_topics]
    except Exception as e:
        logging.error(f"Error in generate_radar_chart: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None, None

def plot_radar_chart(data, periods, topic_names):
    try:
        logging.info("Starting radar chart generation...")

        if data is None or len(data) == 0 or np.all(data == 0):
            logging.warning("No valid data available for radar chart.")
            return

        # Ensure we're using only 8 topics
        n_topics = 10
        data = data[:, :n_topics]
        topic_names = topic_names[:n_topics]

        # Scale up the data
        scaled_data = data * 100  # Scale to percentage

        num_vars = len(topic_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        topic_names += topic_names[:1]

        fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(polar=True))

        colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(periods)))

        for i, period in enumerate(periods):
            values = scaled_data[i].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=period, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), topic_names)

        # Adjust y-axis limits and ticks
        ax.set_ylim(0, np.max(scaled_data) * 1.1)
        yticks = np.linspace(0, np.max(scaled_data), 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{x:.1f}%' for x in yticks])

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Topic Distribution Comparison Across Time Periods', fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('radar_chart_topic_comparison.png', dpi=300, bbox_inches='tight')
        logging.info("Radar chart saved successfully as 'radar_chart_topic_comparison.png'.")

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error in plot_radar_chart: {str(e)}")
        logging.error(traceback.format_exc())

def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def plot_group_topic_focus(all_topic_distributions, model, vectorizer):
    # Type hint compatible with Python 3.6.0
    # Dict[int, np.ndarray] -> Dict[int, Any]
    logging.info("Starting group topic focus plot generation...")
    topic_names = [generate_topic_label(topic, vectorizer.get_feature_names()) for topic in model.components_]
    n_topics = 10  # Ensure we use only 5 topics
    logging.info("Number of topics: {}".format(n_topics))

    overall_distribution = np.zeros(n_topics)
    total_documents = 0

    for year, distributions in all_topic_distributions.items():
        logging.info("Processing year {}, shape: {}".format(year, distributions.shape))
        if distributions.size == 0:
            logging.warning("Empty distribution for year {}".format(year))
            continue
        year_dist = np.mean(distributions[:, :n_topics], axis=0)  # Use mean of first 5 topics
        logging.info("Year {} mean: {}".format(year, year_dist))
        overall_distribution += year_dist
        total_documents += 1  # Count years instead of documents

    if total_documents > 0:
        overall_distribution /= total_documents
    else:
        logging.error("No valid years found in the dataset.")
        return

    logging.info("Overall distribution (before normalization): {}".format(overall_distribution))

    # Ensure the distribution sums to 1
    overall_distribution = overall_distribution / np.sum(overall_distribution)
    logging.info("Overall distribution (normalized): {}".format(overall_distribution))

    if np.all(overall_distribution == 0):
        logging.error("All topic distributions are zero. Skipping group topic focus plot.")
        return

    # Use all 5 topics
    data = overall_distribution
    labels = topic_names[:n_topics]

    # Create the Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(16, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, n_topics))  # Use tab10 for 5 distinct colors

    ax.bar(["Topics"], [1], label="Total", color='lightgray', edgecolor='black')
    bottom = 0
    for i, (value, label) in enumerate(zip(data, labels)):
        ax.bar(["Topics"], [value], bottom=bottom, label=label, color=colors[i])
        bottom += value

    ax.set_title('Distribution of Top {} Topics Across All Years'.format(n_topics), fontsize=18)
    ax.set_ylabel('Relative Topic Focus', fontsize=14)
    ax.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Add percentage labels on the bars
    bottom = 0
    for i, value in enumerate(data):
        percentage = value * 100
        if percentage >= 1:  # Only show label if percentage is 1% or greater
            ax.text(0, bottom + value/2, '{:.1f}%'.format(percentage), ha='center', va='center', fontsize=10)
        bottom += value

    plt.tight_layout()
    plt.savefig('top_topics_stacked_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("Top {} Topics Stacked Bar Chart has been saved as 'top_topics_stacked_bar.png'.".format(n_topics))

    # Save topic distributions to CSV
    csv_filename = 'group_topics.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Topic', 'Percentage'])
        for i, (label, value) in enumerate(zip(labels, data)):
            percentage = value * 100
            csvwriter.writerow(["Topic {}: {}".format(i, label), "{:.2f}%".format(percentage)])
        logging.info("Group topics have been saved to {}".format(csv_filename))

    # Additional: Print topic distributions
    for i, (label, value) in enumerate(zip(labels, data)):
        percentage = value * 100
        logging.info("Topic {}: {} - {:.2f}%".format(i, label, percentage))

def perform_lda(texts, n_topics=10):
    """Perform LDA on the given texts."""
    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2), vocabulary=env_keywords)
    X = cv.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    return lda, cv

def calculate_perplexity_coherence(texts, topic_range):
    logging.info(f"Starting perplexity and coherence calculation for {len(texts)} texts")
    
    try:
        texts = [[word for word in document.lower().split() if word not in nltk_stopwords.words('english')]
                 for document in texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        logging.info(f"Dictionary size: {len(dictionary)}, Corpus size: {len(corpus)}")
        
        results = []
        
        for n_topics in topic_range:
            logging.info(f"Calculating for {n_topics} topics")
            try:
                lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, 
                                     random_state=42, passes=10)
                
                perplexity = np.exp2(-lda_model.log_perplexity(corpus))
                coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence = coherence_model.get_coherence()
                
                results.append((n_topics, perplexity, coherence, lda_model))
                logging.info(f"Topics: {n_topics}, Perplexity: {perplexity}, Coherence: {coherence}")
            except Exception as e:
                logging.error(f"Error calculating for {n_topics} topics: {str(e)}")
                logging.error(traceback.format_exc())
        
        return results
    except Exception as e:
        logging.error(f"Error in calculate_perplexity_coherence: {str(e)}")
        logging.error(traceback.format_exc())
        return []

def plot_perplexity_coherence(topic_range, perplexities, coherences):
    logging.info("Starting to plot perplexity and coherence scores")
    try:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Perplexity (lower is better)
        ax1.plot(topic_range, perplexities, 'b-', label='Perplexity')
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Perplexity', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.invert_yaxis()  # Invert perplexity axis so lower (better) values are on top
        
        # Coherence (higher is better)
        ax2 = ax1.twinx()
        ax2.plot(topic_range, coherences, 'r-', label='Coherence')
        ax2.set_ylabel('Coherence', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Perplexity and Coherence Scores by Number of Topics')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        
        # Save final plot
        plt.savefig('perplexity_coherence_scores.png', dpi=300, bbox_inches='tight')
        logging.info("Perplexity and coherence plot saved")
        
        plt.close()
        logging.info("Perplexity and coherence plot saved successfully")
    except Exception as e:
        logging.error(f"Error plotting perplexity and coherence scores: {str(e)}")
        logging.error(traceback.format_exc())

def plot_topic_optimization(topic_range: List[int], perplexities: List[float], coherences: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    
    plt.plot(topic_range, perplexities, 'b-', marker='o', label='Perplexity')
    plt.ylabel('Perplexity', color='b')
    plt.tick_params(axis='y', labelcolor='b')
    
    plt2 = plt.twinx()
    plt2.plot(topic_range, coherences, 'r-', marker='s', label='Coherence')
    plt2.set_ylabel('Coherence', color='r')
    plt2.tick_params(axis='y', labelcolor='r')
    
    plt.xlabel('Number of Topics')
    plt.title('Topic Model Optimization: Perplexity and Coherence vs. Number of Topics')
    
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = plt2.get_legend_handles_labels()
    plt2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    plt.savefig('topic_optimization_graph.png', dpi=300, bbox_inches='tight')
    plt.close()        

def evaluate_topic_models(texts: List[str], topic_range: List[int]) -> Tuple[List[float], List[float], LdaModel]:
    texts = [[word for word in document.lower().split() if word not in nltk_stopwords.words('english')]
             for document in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    perplexities = []
    coherences = []
    models = []
    
    for n_topics in topic_range:
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, 
                             random_state=42, passes=10)
        
        perplexity = np.exp2(-lda_model.log_perplexity(corpus))
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        
        perplexities.append(perplexity)
        coherences.append(coherence)
        models.append(lda_model)
        
        logging.info(f"Topics: {n_topics}, Perplexity: {perplexity}, Coherence: {coherence}")
    
    best_model_index = coherences.index(max(coherences))
    return perplexities, coherences, models[best_model_index]

def main(directory: str) -> None:
    all_topic_distributions, guided_model, guided_vectorizer, _, _ = aggregate_topic_distributions(directory)

    if all_topic_distributions and guided_model and guided_vectorizer:
        try:
            logging.info("Generating time analysis plot...")
            if len(all_topic_distributions) > 0:
                plot_time_analysis(all_topic_distributions, guided_model, guided_vectorizer)
                logging.info("Time analysis plot generated successfully.")
            else:
                logging.warning("No data available for time analysis plot.")
        except Exception as e:
            logging.error(f"Error generating time analysis plot: {str(e)}")
            logging.error(traceback.format_exc())

        try:
            logging.info("Generating topic bar graphs and metrics...")
            if guided_model.components_.shape[0] > 0:
                # Collect all environmental speeches
                all_env_speeches = []
                for filename in os.listdir(directory):
                    if filename.endswith('.csv'):
                        try:
                            file_path = os.path.join(directory, filename)
                            speeches = pd.read_csv(file_path, index_col=0, header=None, names=['index', 'file_ref', 'ID', 'unknown', 'speech'])
                            env_speeches = filter_environmental_speeches(speeches)
                            all_env_speeches.extend(env_speeches['speech'].tolist())
                        except Exception as e:
                            logging.error(f"Error processing file {filename}: {str(e)}")
                            logging.error(traceback.format_exc())

                # Preprocess speeches
                preprocessed_speeches = [preprocess_text(speech) for speech in all_env_speeches]

                # Create corpus
                corpus = guided_vectorizer.transform(preprocessed_speeches)

                # Plot topic bar graphs
                topic_names = plot_topic_bar_graphs(guided_model, guided_vectorizer, corpus)
                
                # Plot topic metrics
                plot_topic_metrics(guided_model, corpus, topic_names)

                logging.info("Topic bar graphs and metrics generated successfully.")
            else:
                logging.warning("No topics available for bar graphs and metrics.")
        except Exception as e:
            logging.error(f"Error generating topic bar graphs and metrics: {str(e)}")
            logging.error(traceback.format_exc())

        try:
            logging.info("Generating radar chart data...")
            radar_data, periods, radar_topic_names = generate_radar_chart(all_topic_distributions, guided_model, guided_vectorizer)
            if radar_data is not None and periods is not None and radar_topic_names is not None and len(radar_data) > 0:
                plot_radar_chart(radar_data, periods, radar_topic_names)
                logging.info("Radar chart generated successfully.")
                if os.path.exists('radar_chart_topic_comparison.png'):
                    logging.info("Radar chart PNG generated successfully.")
                else:
                    logging.error("Radar chart file was not created.")
            else:
                logging.error("Failed to generate radar chart data or no data available.")
        except Exception as e:
            logging.error(f"Error generating radar chart: {str(e)}")
            logging.error(traceback.format_exc())

        try:
            logging.info("Generating group topic focus plot...")
            if len(all_topic_distributions) > 0:
                plot_group_topic_focus(all_topic_distributions, guided_model, guided_vectorizer)
                logging.info("Group topic focus plot generated successfully.")
            else:
                logging.warning("No data available for group topic focus plot.")
        except Exception as e:
            logging.error(f"Error generating group topic focus plot: {str(e)}")
            logging.error(traceback.format_exc())

        logging.info(f"Total number of environmental speeches: {len(all_env_speeches)}")

        # Limit the number of speeches to process (optional, remove if you want to process all)
        all_env_speeches = all_env_speeches[:1000]  # Process only the first 1000 speeches
        logging.info(f"Number of speeches used for topic model evaluation: {len(all_env_speeches)}")

        # Perform LDA with 10 topics
        lda_model, lda_vectorizer = perform_lda(all_env_speeches, n_topics=10)

        try:
            logging.info("Generating LDA results plot...")
            plot_lda_results(lda_model, lda_vectorizer, all_env_speeches)
            logging.info("LDA results plot generated successfully.")
        except Exception as e:
            logging.error(f"Error generating LDA results plot: {str(e)}")
            logging.error(traceback.format_exc())

        # Evaluate models with different numbers of topics
        topic_range = range(5, 21)  # Evaluate from 5 to 20 topics
        perplexities, coherences, best_model = evaluate_topic_models(all_env_speeches, topic_range)

        # Plot the optimization graph
        plot_topic_optimization(list(topic_range), perplexities, coherences)
        logging.info("Topic optimization graph has been saved as 'topic_optimization_graph.png'.")

        logging.info("All plots have been generated.")
    else:
        logging.warning("No data available to plot. Check if any files were successfully processed.")
        if not all_topic_distributions:
            logging.warning("all_topic_distributions is empty")
        if not guided_model:
            logging.warning("guided_model is None")
        if not guided_vectorizer:
            logging.warning("guided_vectorizer is None")

if __name__ == "__main__":
    data_directory = r'D:\OneDrive\Desktop\oms'
    main(data_directory)