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




app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
csrf = CSRFProtect(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


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
        sentences = content.split(". ")  # Split content into sentences
        summary = ". ".join(sentences[:max_sentences])  # Take the first few sentences as summary
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
        # num_samples = query_vector.shape[0]
        elif link == "ward":
            agg_cluster = pickle.load(open("new_agg_ward.pkl", "rb"))
        selected_data = query_vector
        # Feature selection using SelectKBest and chi-square scoring
        # k_best = SelectKBest(score_func=chi2, k=50)
        # tfidf_selected = k_best.fit_transform(selected_data, [0] * num_samples)  # Assuming binary labels for example
        svd = TruncatedSVD(n_components=3)  # Reduce to 50 dimensions
        tfidf_reduced = svd.fit_transform(selected_data)
        # agg_d = fastcluster.linkage(tfidf_reduced, metric='euclidean')
        # Check the shape of the reduced TF-IDF matrix
        print("Shape of reduced TF-IDF matrix:", tfidf_reduced.shape)
        # print(tfidf_reduced)
        p_c = agg_cluster.fit_predict(tfidf_reduced)
        json_str = self.re_rank_cluster(p_c)
        # p_c=fcluster(agg_d,t=1)
        # print(p_c.flatten())
        # [7  9 11  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
        #  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 13 12  5 10  4  8  3  6
        #  2  1]
        return json_str

        # [ 0  0  7 12  0  0  8  0  0  0  0  0  2  9  4  4  1  0  0  0  0  2 11  6 10  0  1  5  0  3  3  0]

        # return json_str
@app.route("/", methods=['GET'])
def hello_world():
  return "Hello"

@app.route('/cluster', methods=['POST'])
def cluster():
    json_data = request.json
    documents = json_data["documents"]
    method = json_data["method"]
    print("A")
    clustering = Clustering(documents)
   
    result, count = clustering.get_clusters_results(method)
    print(result)
    return jsonify(result=result, count=count)

if __name__ == '__main__':
    app.run(port=3001,debug=True)