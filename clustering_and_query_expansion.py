from flask import Flask, request, jsonify
import json
import os
import pickle
from collections import defaultdict
from itertools import chain
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import fastcluster
from scipy.cluster.hierarchy import fcluster
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
import re
import numpy as np
from nltk.corpus import stopwords
from nltk import PorterStemmer
from collections import Counter

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
csrf = CSRFProtect(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def tokenize_text(text):
    tokens = []
    text = re.sub(r'[\n]', ' ', text)
    text = re.sub(r'[,-]', ' ', text)
    text = re.sub('[0-9]', '', text)
    text = re.sub(r'[^A-Za-z_\s]', '', text)
    text = text.lower()
    tkns = text.split()
    tokens = [token for token in tkns if token not in stop_words and token != '' and not token.isdigit()]
    return tokens

def make_stem_map(vocab):
    token_2_stem = {}
    stem_2_tokens = {}

    for token in vocab:
        stem = porter_stemmer.stem(token)
        if stem not in stem_2_tokens:
            stem_2_tokens[stem] = set()
        stem_2_tokens[stem].add(token)
        token_2_stem[token] = stem

    return token_2_stem, stem_2_tokens 

def build_association(doc_tokens, token_2_stem, stem_2_tokens, query):
    stems = stem_2_tokens.keys()
    stems = list(sorted(stems))
    stem_2_idx = {s:i for i, s in enumerate(stems)}

    f = np.zeros((len(doc_tokens), len(stems)), dtype=int)
    for doc_id, tokens in enumerate(doc_tokens):
        for token in tokens:
            if token in token_2_stem:
                stem = token_2_stem[token]
                stem_idx = stem_2_idx[stem]
                f[doc_id, stem_idx] += 1

    c = np.dot(f.T, f)
    c_diag = np.diag(c)

    query_expands_id = []
    for token in query:
        stem = token_2_stem[token]
        stem_id = stem_2_idx[stem]

        c_token = c[stem_id, :]
        s_token = c_token / (c_token[stem_id] + c_diag + c_token)

        idx_sort = np.argsort(s_token)[::-1]
        idx_sort = idx_sort[1:2]
        query_expands_id.extend(idx_sort.tolist())

    query_expands = []
    for stem_idx in query_expands_id:
        query_expands.append(stems[stem_idx])

    return query_expands

def get_scalar_cluster(doc_tokens, token_2_stem, stem_2_tokens, query):
    stems = stem_2_tokens.keys()
    stems = list(sorted(stems))
    stem_2_idx = {s:i for i, s in enumerate(stems)}

    f = np.zeros((len(doc_tokens), len(stems)), dtype=int)
    for doc_id, tokens in enumerate(doc_tokens):
        for token in tokens:
            if token in token_2_stem:
                stem = token_2_stem[token]
                stem_idx = stem_2_idx[stem]
                f[doc_id, stem_idx] += 1

    c = np.dot(f.T, f)
    c_diag = np.expand_dims(np.diag(c), axis=0)

    s = c / (c + c_diag + c_diag.T)
    s_norm = np.linalg.norm(s, axis=1)

    query_expands_id = []
    for token in query:
        stem = token_2_stem[token]
        stem_id = stem_2_idx[stem]

        stem_vec = np.expand_dims(s[stem_id, :], axis=0)
        stem_norm = np.linalg.norm(stem_vec)
        s_stem = np.dot(stem_vec, s.T).squeeze()
        s_stem = (s_stem / stem_norm) / s_norm

        idx_sort = np.argsort(s_stem)[::-1]
        idx_sort = idx_sort[1:2]
        query_expands_id.extend(idx_sort.tolist())

    query_expands = []
    for stem_idx in query_expands_id:
        query_expands.append(stems[stem_idx])

    return query_expands

def get_metric_clusters(doc_tokens, token_2_stem, stem_2_tokens, query):
    stems = stem_2_tokens.keys()
    stems = list(sorted(stems))
    stem_2_idx = {s:i for i, s in enumerate(stems)}

    stem_len = [len(stem_2_tokens[s]) for s in stems]
    stem_len = np.array(stem_len)

    c = np.zeros((len(stem_2_idx), len(stem_2_idx)), dtype=int)
    for doc_id, tokens in enumerate(doc_tokens):
        tokens_count = Counter(tokens)
        for token_1, count_1 in tokens_count.items():
            stem_1 = token_2_stem[token_1]
            stem_1_id = stem_2_idx[stem_1]
            for token_2, count_2 in tokens_count.items():
                stem_2 = token_2_stem[token_2]
                stem_2_id = stem_2_idx[stem_2]
                if stem_1 == stem_2:
                    continue
                if count_1 != count_2:
                    c[stem_1_id, stem_2_id] += 1. / abs(count_1 - count_2)

    query_expands_id = []
    for token in query:
        stem = token_2_stem[token]
        stem_id = stem_2_idx[stem]

        s_stem = c[stem_id, :] / (stem_len[stem_id] * stem_len)

        s_stem = np.argsort(s_stem)[::-1]
        s_stem = s_stem[1:2]
        query_expands_id.extend(s_stem.tolist())

    query_expands = []
    for stem_idx in query_expands_id:
        query_expands.append(stems[stem_idx])

    return query_expands

def cluster_main(query, solr_results, cluster):
    vocab = set()
    doc_tokens = []

    for result in solr_results:
        content = result.get("content", "")
        if isinstance(content, list):
            content = ' '.join(content)
        tokens = tokenize_text(content)
        vocab.update(tokens)
        doc_tokens.append(tokens)
        
    query_tokens = tokenize_text(query)
    query_tokens1 = query_tokens
    vocab.update(query_tokens)
    vocab = list(sorted(vocab))
    token_2_stem, stem_2_tokens = make_stem_map(vocab)

    if cluster == "association":
        query_expands_stem = build_association(doc_tokens, token_2_stem, stem_2_tokens, query_tokens)
    elif cluster == "scalar":
        query_expands_stem = get_scalar_cluster(doc_tokens, token_2_stem, stem_2_tokens, query_tokens)
    elif cluster == "metric":
        query_expands_stem = get_metric_clusters(doc_tokens, token_2_stem, stem_2_tokens, query_tokens)

    query_expands = []
    for stem in query_expands_stem:
        temp = list(stem_2_tokens[stem])
        query_expands.append(temp[0])
    query_expands = set(query_expands)
    
    for token in query_tokens:
        query_expands.discard(token)
    temp_query = list(query_expands)
    query_tokens1.extend(temp_query)
    query = ' '.join(query_tokens1)

    print('Expanded query:', query)

    return query

class Clustering:
    def __init__(self, json_data):
        self.content_list = []
        self.url_list = []
        self.title_list = []
        self.summary_list = []
        for obj in json_data["response"]["docs"]:
            url = obj.get("url", "")
            title = obj.get("title", [""])[0]
            if "content" in obj:
                content = obj.get("content", [""])[0]
                if url not in self.url_list:
                    self.url_list.append(url)
                    self.content_list.append(content)
                    self.summary_list.append(self.summarize_content(content))
                    self.title_list.append(title)

        vec = pickle.load(open("new_vectorizer.pkl", "rb"))
        self.query_vector = vec.transform(self.content_list)

    def summarize_content(self, content, max_sentences=3):
        sentences = content.split(". ")
        summary = ". ".join(sentences[:max_sentences])
        return summary

    def get_clusters_results(self, method):
        if method == "Flat_Clustering":
            return self.get_clusters_flat_clustering(self.query_vector)
        elif method == "Agglomerative_Average_Link_Clustering":
            return self.get_clusters_agglomerative_clustering(self.query_vector, "average")
        elif method == "Agglomerative_Complete_Link_Clustering":
            return self.get_clusters_agglomerative_clustering(self.query_vector, "complete")
        elif method == "Agglomerative_Ward_Link_Clustering":
            return self.get_clusters_agglomerative_clustering(self.query_vector, "ward")

    def get_clusters_flat_clustering(self, query_vector):
        kmeans_model = pickle.load(open("new_kmeans_model_7.pkl", "rb"))
        predicted_clusters = kmeans_model.predict(query_vector)
        p_c = list(predicted_clusters.flatten())
        return self.re_rank_cluster(p_c)

    def re_rank_cluster(self, clusters):
        cluster_order = {}
        for i in range(len(clusters)):
            cluster_id = clusters[i]
            entry = {"url": self.url_list[i], "title": self.title_list[i], "content": self.summary_list[i],
                     "cluster_id": str(cluster_id)}
            if cluster_id not in cluster_order:
                cluster_order[cluster_id] = [entry]
            else:
                cluster_order[cluster_id].append(entry)

        json_str = list(chain.from_iterable(list(cluster_order.values())))
        return json_str, len(json_str)

    def get_clusters_agglomerative_clustering(self, query_vector, link):
        agg_cluster = 0
        if link == "average":
            agg_cluster = pickle.load(open("new_agg_average.pkl", "rb"))
        elif link == "complete":
            agg_cluster = pickle.load(open("new_agg_complete.pkl", "rb"))
        elif link == "ward":
            agg_cluster = pickle.load(open("new_agg_ward.pkl", "rb"))
        selected_data = query_vector
        svd = TruncatedSVD(n_components=3)
        tfidf_reduced = svd.fit_transform(selected_data)
        p_c = agg_cluster.fit_predict(tfidf_reduced)
        json_str = self.re_rank_cluster(p_c)
        return json_str

@app.route("/", methods=['GET'])
def hello_world():
    return "Hello"

@app.route('/cluster', methods=['POST'])
def cluster():
    json_data = request.json
    documents = json_data["documents"]
    method = json_data["method"]
    clustering = Clustering(documents)
    result, count = clustering.get_clusters_results(method)
    return jsonify(result=result, count=count)

@app.route('/query_expansion', methods=['POST'])
def query_expansion():
    json_data = request.json
    query = json_data["query"]
    solr_results = json_data["solrResults"]
    print(solr_results)
    cluster_type = json_data["clustertype"]
    expanded_query = cluster_main(query, solr_results, cluster_type)
    return jsonify(expanded_query=expanded_query)

if __name__ == '__main__':
    app.run(port=3001, debug=True)
