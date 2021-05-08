import time
from pulp import *
from queries import *
from utils import *
import csv
import numpy as np
from sklearn.cluster import KMeans

"""
Author: Limengxi Yue (lyue@umass.edu)
"""

def hybrid_sketch(
    query: str,
    n: int,
    attribute_list: [],
    partition_list: []
):
    optimal_set = []
    left = 0
    right = 0
    for i in range(len(partition_list)):  
        data = []  
        for j in range(len(partition_list)):  
            if j == i:  
                for k in list(partition_list[j][0]):  
                    data.append(list(k))
            else:
                data.append(partition_list[j][1])  
        new_data = np.array(data)
       
        right = left + len(partition_list[i][0])
        cur_optimal_set = hybrid(new_data, query, attribute_list, n)

        for k in range(len(cur_optimal_set)): 
            str = cur_optimal_set[k]
            
            if int(str.lstrip('t')) < left or int(str.lstrip('t')) >= right:
                del cur_optimal_set[k]     

        if len(cur_optimal_set) != 0: 
            optimal_set.append(cur_optimal_set)

        left = left+1  

    return optimal_set

def hybrid(
        data: [],
        query: str,
        attribute_list: list,
        n: int,
):
    tuples = data
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

    x_list = [LpVariable(name="t" + str(i), lowBound=0, upBound=1, cat=LpInteger) for i in range(1, len(tuples)+1)]



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
    print("prob.variables():")
    #print(prob.variables())
    #if prob.variables() is not None:
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


def get_partitions(k,data):

    d = np.array(data)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(d)
    part = []
    for i in range(0,k):
        part.append(set())
    kmeans_labels = kmeans.labels_.tolist()
    for m,n in zip(kmeans_labels,data):
        part[m].add(tuple(n))
    return list(zip(part,kmeans.cluster_centers_.tolist()))


def sample_rows(data, sample):
    tuple_count = len(data)
    if sample != 1:
        sample_size = int(sample * tuple_count)
        sample_rows = np.random.randint(low=0, high=tuple_count, size=(sample_size,)).tolist()
        return data[sample_rows, :]
    else:
        return data


if __name__ == '__main__':

    '''
    query = """select package(*) as P
               from tpch REPEAT 0
               such that
               sum(sum_base_price) <= 15469853.7043 and
               sum(sum_disc_price) <= 45279795.0584 and
               sum(sum_charge) <= 95250227.7918 and
               sum(avg_qty) <= 50.353948653 and
               sum(avg_price) <= 68677.5852459 and
               sum(avg_disc) <= 0.110243522496 and
               sum(sum_qty) <= 77782.028739 and
               count(*) >= 1
               maximize sum(count_order);

             """
    '''
    '''
    query = """select package(*) as P
               from tpch REPEAT 0
               such that
               sum(o_totalprice) <= 453998.242103 and
               sum(o_shippriority) >= 3 and
               count(*) >= 1
               minimize count(*);"""
    '''
    '''
    query = """select package(*) as P
               from tpch REPEAT 0
               such that
               sum(revenue) >= 413930.849506 and
               count(*) >= 1
               minimize count(*);
    """
    '''

    query = """
                         select package(*) as P
                         from tpch REPEAT 0 such that
                         sum(p_size) <= 8 and
                         count(*) >= 1
                         minimize
                         sum(ps_min_supplycost);
              """



    n_clusters = 7
    data, attribute_list = read_csv_data("./tpch.csv")
    new_data = sample_rows(data, 0.1)
    partition_list = get_partitions(n_clusters, new_data)
    
    package = hybrid_sketch(query, n_clusters, attribute_list, partition_list)

    print(package)



