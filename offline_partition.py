from sklearn.cluster import KMeans

import numpy as np
import time
from utils import read_csv_data, sample_rows
import pickle

"""
Author: Jiang Li (jiangli@umass.edu)
"""

def get_kmeans(k,data):
    d = np.array(data)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(d)
    return kmeans

def encoder(partitioned_result, name):
    pickle_file = open(name, 'wb')
    pickle.dump(partitioned_result, pickle_file)
    pickle_file.close()

def decoder(name):
    pkl_file = open(name, 'rb')
    partitioned_result = pickle.load(pkl_file)
    pkl_file.close()
    return partitioned_result

def load_partitions(name):
    partitioned_result = decoder(name)
    kmeans = partitioned_result
    part = []
    new_data = data.tolist()
    kmeans_labels = kmeans.labels_.tolist()
    for i in range(len(kmeans_labels)):
        part.append(set())
    for m, n in zip(kmeans_labels, new_data):
        part[m].add(tuple(n))
    return list(zip(part, kmeans.cluster_centers_.tolist()))


if __name__ == '__main__':

    data, attribute_list = read_csv_data("./tpch.csv")
    print("data read")


    '''total cases'''
    sample_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    n_clusters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000]

    for s in sample_size:
        dada = sample_rows(data, s)
        for n in n_clusters:
            print("------------------------------------------")
            print(f"running sample_size:{s}, n_clusters:{n}")

            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)

            kmeans = get_kmeans(n, dada)

            encoder(partitioned_result=kmeans, name=f"{s}_{n}.pkl")
            print(f"saved as {s}_{n}.pkl")


    """
    use load_partitions(f"{s}_{n}.pkl") to get the partitioned result
    """

    # sample_size = [1]
    # n_clusters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    # for s in sample_size:
    #     dada = sample_rows(data, s)
    #     for n in n_clusters:
    #         partition_list = load_partitions(f"{s}_{n}.pkl")
    #         package = sketch(query=query,
    #                          n=n_clusters,
    #                          attribute_list=attribute_list,
    #                          partition_list=partition_list
    #                          )
    #         print(record_time)
    #         print(package)