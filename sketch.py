import time
import re
from pulp import *
from parser import *

import csv
import numpy as np
from sklearn.cluster import KMeans

"""
Author: Limengxi Yue (lyue@umass.edu)
"""


def sketch(
        query: str,
        n: int,
        attribute_list: [],
        partition_list: []
):
    
    variables = paql_to_variables(query)

    table_name = variables['TABLE_NAME']
    objective = variables['OBJECTIVE']
    attribute_objective = variables['ATTRIBUTE_OBJECTIVE']
    constraints = variables['CONSTRAINTS']
    count_constraint = variables['COUNT_CONSTRAINT']

    print(f"\n----------------------------\nModeling LP problem")
    time1 = time.time()

    if objective.upper() == "MIN":
        prob = LpProblem("PaQL_Problem", LpMinimize)
    elif objective.upper() == "MAX":
        prob = LpProblem("PaQL_Problem", LpMaximize)

    tuples = []

    for i in range(len(partition_list)):
        tuples.append(partition_list[i][1])

    x_list = [LpVariable(name="t" + str(i), lowBound=0, cat=LpInteger) for i in range(1, len(tuples)+1)]

    for i in range(len(partition_list)):
        prob += x_list[i] <= len(partition_list[i][0])

    if attribute_objective is not None:
        prob += lpSum([tuples[i][attribute_list.index(attribute_objective)] * x_list[i] for i in
                       range(len(x_list))]), "attribute_objective_constraint"
    else:
        prob += lpSum(x_list), "attribute_objective_constraint_count"

    if count_constraint[0] is not None:
        prob += count_constraint[0] <= lpSum(x_list), "count_lower_constraint"
    if count_constraint[1] is not None:
        prob += lpSum(x_list) <= count_constraint[1], "count_upper_constraint"

    # prob += count_constraint[0] <= lpSum(x_list) <= count_constraint[1], "count constraint"
    for k in range(len(constraints)):
        attr = constraints[k][0]
        Lk = constraints[k][1][0]
        Uk = constraints[k][1][1]
        if Lk is not None:
            prob += Lk <= lpSum([tuples[i][attribute_list.index(attr)] * x_list[i] for i in range(len(x_list))]), f"attribute_{attr}_lower_constraints"
        if Uk is not None:
            prob += lpSum([tuples[i][attribute_list.index(attr)] * x_list[i] for i in range(len(x_list))]) <= Uk, f"attribute_{attr}_upper_constraints"


    time2 = time.time()
    print(f"Time for modeling: {time2 - time1}")
    print(f"\n----------------------------\nStart running LP solver")
    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with it's resolved optimum value
    optimal_set = []
    for v in prob.variables():
        if v.varValue is not None:
            if v.varValue > 0:
                print(v.name, "=", v.varValue)
                optimal_set.append(v.name)
        else:
            continue
    time3 = time.time()
    print(f"LP problem solve time: {time3 - time2}")
    print(f"\n--------------------------------------\nSummary: "
          f"The optimal rows selected under the requirements are: {optimal_set}. "
          "\n  (Note: 't1' stands for the 1st row)")
    print(f"\n----------------------------\nTotal time cost: {time3 - time1}")

    return optimal_set


def read_csv_data(path):
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
    return data, attribute_list


def get_partitions(k, data):
    d = np.array(data)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(d)
    part = []
    new_data = data.tolist()
    kmeans_labels = kmeans.labels_.tolist()
    for i in range(0, k):
        part.append(set())
    for m, n in zip(kmeans_labels, new_data):
        part[m].add(tuple(n))
    return list(zip(part, kmeans.cluster_centers_.tolist()))


if __name__ == '__main__':

    query = """
                     select package(*) as P
                     from tpch REPEAT 0 such that
                     sum(p_size) <= 8 and
                     count(*) >= 1
                     minimize
                     sum(ps_min_supplycost);

                  """
    
    n_clusters = 5
    
    data, attribute_list = read_csv_data("./tpch.csv")

    partition_list = get_partitions(n_clusters, data)
    
    package = sketch(query=query,
                     n=n_clusters,
                     attribute_list=attribute_list,
                     partition_list=partition_list
    )

    print(package)



