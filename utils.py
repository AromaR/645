from sklearn.cluster import KMeans
import numpy as np

import csv

"""
Author: Jiang Li (jiangli@umass.edu)
"""


def read_csv_data(path="./tpch.csv", sample=1):
    data = []
    attribute_list = None
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        tuple_count = -1
        for row in csv_reader:
            tuple_count += 1
            if tuple_count == 0:
                attribute_list = row
                continue
            data.append(row)
    data = np.array(data).astype(float)

    data = sample_rows(data, sample)
    return data, attribute_list

def sample_rows(data, sample):
    tuple_count = len(data)
    if sample != 1:
        sample_size = int(sample * tuple_count)
        sample_rows = np.random.randint(low=0, high=tuple_count, size=(sample_size,)).tolist()
        return data[sample_rows, :]
    else:
        return data

def partition(rows, n_clusters=4):
    X = np.array(rows)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)

    return kmeans.labels_, kmeans.cluster_centers_


