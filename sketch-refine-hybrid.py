"""People who worked on this code: primarily: Aroma Rodrigues, aprodrigues@umass.edu, utils and functions like direct and hybrid sketch from Jiang and Limengxi"""

import time
from pulp import *

from queries import *
import sys
sys.setrecursionlimit(1500)

def direct(
        data: [],
        query: str,
        past: None
):
    print("in direct")
    attribute_list = data[0]
    tuples = data[1:]
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
        prob += lpSum([tuples[i][attribute_list.index(attribute_objective)] * x_list[i] for i in range(len(x_list))]), "attribute_objective_constraint"
    else:
        print("setting count constraint")
        prob += lpSum(len([x_list[i]]) for i in range(len(x_list))), "count_constraint"
        
    if count_constraint[0] is not None:
        prob += count_constraint[0] <= lpSum(x_list), "count_lower_constraint"
    if count_constraint[1] is not None:
        print("count_constraint ",count_constraint[1],c)
        if past:
            c = len(past)
        else:
            c = 0
        if count_constraint[1]-c<=0:
            return set()
        prob += lpSum(x_list) <= count_constraint[1]-c, "count_upper_constraint"

    # prob += count_constraint[0] <= lpSum(x_list) <= count_constraint[1], "count constraint"
    for k in range(len(constraints)):
        attr = constraints[k][0]
        Lk = constraints[k][1][0]
        Uk = constraints[k][1][1]
        #print(past)
        if Uk:
            for p in list(past):
                print("Uk",Uk)
                print("sub",p[k])
                Uk = Uk - p[k]
                if Uk<0:
                    Uk=0
                    break
                print("Uk adjusted for past answers")
        if Lk:
            for p in list(past):
                print("Lk",Lk)
                print("sub",p[k])
                Lk = Lk - p[k]
                if Lk<0:
                    Lk=0
                    break
                print("Uk adjusted for past answers")
        if Lk==0 and Uk==0:
            return set()
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
        if v.varValue and v.varValue > 0:
            #print(v.name, "=", v.varValue)
            optimal_set.append(v.name)

    time3 = time.time()
    print(f"LP problem solve time: {time3 - time2}")
    # print(f"\n--------------------------------------\nSummary: "
          # f"The optimal rows selected under the requirements are: {optimal_set}. "
          # "\n  (Note: 't1' stands for the 1st row)")
    #print(f"Total value of objective function based on this set: {value(prob.objective)}")
    print(f"\n----------------------------\nTotal time cost: {time3 - time1}")

    return optimal_set
    
def sketch_refine(query,
                     n,
                     attribute_list,
                     centers,partition_list):
    """
    q: package query 
    p: partitioning
    
    """
    y,z = hybrid_sketch(query,n,attribute_list,list(zip(partition_list,centers)))
    #y = sketch(query,n,attribute_list,list(zip(partition_list,centers)) )
    #t1,t2,t3
    #t534
    #t511
    #print(y[0])
    #print(z[0])
    #return
    ps = []
    for i in y:
        #print(int(i[0].replace("t",""))-1)
        ps.append(tuple(z[int(i[0].replace("t",""))-1]))
    print("sketch results ",ps)
    if len(ps) == 0:
        return []
    else:
        print("starting refine")
        pp, f = greedy_backtracking_refine(query,list(zip(partition_list,centers)),set(ps),list(zip(partition_list,centers)),attribute_list)
        #print("returning results ",pp)
        print("failed ",f)
        if f:
            return 0
        else:
            return pp
          

def sample_rows(data, sample):
    tuple_count = len(data)
    if sample != 1:
        sample_size = int(sample * tuple_count)
        sample_rows = np.random.randint(low=0, high=tuple_count, size=(sample_size,)).tolist()
        return data[sample_rows, :]
    else:
        return data

def paql_to_variables(paql):
    """
    PaQL parser:
    return TABLE_NAME, OBJECTIVE, ATTRIBUTE_OBJECTIVE, CONSTRAINTS, COUNT_CONSTRAINT
    """
    print("\n--------------------------------------\nOriginal PaQL input:")
    print(paql)

    s = paql.lower()
    s = s.strip()  # delete indentations in the beginning and end
    s = s.replace('\n', '')
    s = s.replace(';', '')

    '''Determine PaQL OBJECTIVE'''
    maximize_str_index = s.lower().find("maximize")
    minimize_str_index = s.lower().find("minimize")
    if maximize_str_index >= 0:
        OBJECTIVE = "MAX"
        attribute_objective_val = s[maximize_str_index + 8:]
        s = s[:maximize_str_index]
    if minimize_str_index >= 0:
        OBJECTIVE = "MIN"
        attribute_objective_val = s[minimize_str_index + 8:]
        s = s[:minimize_str_index]

    '''Determine PaQL ATTRIBUTE_OBJECTIVE'''
    sum_pattern = re.compile(r'sum[(](.*?)[)]', re.S)
    sum_pattern_result = re.findall(sum_pattern, attribute_objective_val)
    if len(sum_pattern_result) > 0:
        ATTRIBUTE_OBJECTIVE = sum_pattern_result[0]
    elif "count(*)" in attribute_objective_val.lower():
        ATTRIBUTE_OBJECTIVE = None

    '''Parse TABLE_NAME, CONSTRAINTS, COUNT_CONSTRAINT'''
    matchObj = re.match(r'select (.*) from (.*?) such that (.*)', s, re.M | re.I)

    if matchObj:
        select_val = matchObj.group(1).strip()
        from_val = matchObj.group(2).strip()
        where_val = matchObj.group(3)
        condition_list = where_val.split("and")
        condition_list = [condition_list[i].strip() for i in range(len(condition_list))]

        TABLE_NAME = from_val[:from_val.strip().find(" ")]

        CONSTRAINTS = []
        for i in range(len(condition_list)):

            SUB_OPERATOR = re.findall(re.compile(r'<=|>=|=', re.S), condition_list[i])[0]
            SUB_CONDITION_VALUE = float(condition_list[i][condition_list[i].find(SUB_OPERATOR) + len(SUB_OPERATOR):])

            sum_pattern = re.compile(r'sum[(](.*?)[)]', re.S)
            count_pattern = re.compile(r'count[(](.*?)[)]', re.S)
            sum_pattern_result = re.findall(sum_pattern, condition_list[i])
            count_pattern_result = re.findall(count_pattern, condition_list[i])
            if len(sum_pattern_result) > 0:
                SUB_ATTRIBUTE = sum_pattern_result[0]
                if SUB_OPERATOR == '<=':
                    CONSTRAINTS.append([SUB_ATTRIBUTE, (None, SUB_CONDITION_VALUE)])
                elif SUB_OPERATOR == '>=':
                    CONSTRAINTS.append([SUB_ATTRIBUTE, (SUB_CONDITION_VALUE, None)])
            elif len(count_pattern_result) > 0:
                if SUB_OPERATOR == '<=':
                    COUNT_CONSTRAINT = (None, int(SUB_CONDITION_VALUE))
                elif SUB_OPERATOR == '>=':
                    COUNT_CONSTRAINT = (int(SUB_CONDITION_VALUE), None)
                elif SUB_OPERATOR == '=':
                    COUNT_CONSTRAINT = (int(SUB_CONDITION_VALUE), int(SUB_CONDITION_VALUE))

        print("--------------------------------------\nParsed variables:")
        print(f"""
        TABLE_NAME = {TABLE_NAME}, 
        OBJECTIVE = {OBJECTIVE}, 
        ATTRIBUTE_OBJECTIVE = {ATTRIBUTE_OBJECTIVE}, 
        CONSTRAINTS = {CONSTRAINTS}, 
        COUNT_CONSTRAINT = {COUNT_CONSTRAINT}""")

        return {
            "TABLE_NAME": TABLE_NAME,
            "OBJECTIVE": OBJECTIVE,
            "ATTRIBUTE_OBJECTIVE": ATTRIBUTE_OBJECTIVE,
            "CONSTRAINTS": CONSTRAINTS,
            "COUNT_CONSTRAINT": COUNT_CONSTRAINT
        }
    else:
        print("No match for select-from-where pattern")
    return None


def variables_to_paql(
        table_name: str,
        objective: str,
        attribute_objective: str,
        constraints: list,
        count_constraint: tuple
):
    """
    Transform the input variables into PaQL query form
    :param table_name:
    :param objective:
    :param attribute_objective:
    :param constraints:
    :param count_constraint:
    :return: String of PaQL query
    """
    package_alias = "P"

    such_that_list = []

    if count_constraint[0] != None and count_constraint[1] == None:
        such_that_list.append(f"COUNT({package_alias}.*) >= {count_constraint[0]}")
    elif count_constraint[0] == None and count_constraint[1] != None:
        such_that_list.append(f"COUNT({package_alias}.*) <= {count_constraint[1]}")
    elif count_constraint[0] == count_constraint[1] != None:
        such_that_list.append(f"COUNT({package_alias}.*) = {count_constraint[0]}")

    for constraint in constraints:
        attr = constraint[0]
        Lk = constraint[1][0]
        Uk = constraint[1][1]

        if Lk == None and Uk == None:
            raise RuntimeError(f"The bound of {attr} should be specified.")
        if Lk != None and Uk == None:
            such_that_list.append(f"SUM({package_alias}.{attr}) >= {Lk}")
        elif Lk == None and Uk != None:
            such_that_list.append(f"SUM({package_alias}.{attr}) <= {Uk}")
        elif Lk == Uk != None:
            such_that_list.append(f"SUM({package_alias}.{attr}) = {Lk}")
        elif Lk < Uk:
            such_that_list.append(f"{Lk} <= SUM({package_alias}.{attr}) <= {Uk}")

    if len(such_that_list) == 0:
        such_that_sql = ""
    elif len(such_that_list) == 1:
        such_that_sql = "SUCH THAT " + such_that_list[1]
    else:
        such_that_sql = "SUCH THAT\n        "
        for i in range(len(such_that_list)):
            if i > 0:
                such_that_sql += "\n        AND "
            such_that_sql += such_that_list[i]

    objective_sql = "MINIMIZE" if objective.upper() == "MIN" else "MAXIMIZE"

    if attribute_objective != None:
        attribute_objective_sql = f"SUM({package_alias}.{attribute_objective})"
    else:
        attribute_objective_sql = "COUNT(*)"



    return f"SELECT PACKAGE(*) AS {package_alias}" \
           f"\nFROM {table_name} REPEAT 0" \
           f"\n{such_that_sql}" \
           f"\n{objective_sql} {attribute_objective_sql}"

def sketch(
        tuples: [], # kmeans_cluster_centers
        query: str,
        attribute_list: list,
        n: int,
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

    x_list = [LpVariable(name="t" + str(i), lowBound=0, cat=LpInteger) for i in range(1, len(tuples)+1)]

    if attribute_objective is not None:
        prob += lpSum([tuples[i][attribute_list.index(attribute_objective)] * x_list[i] for i in range(len(x_list))]), "attribute_objective_constraint"

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
    print(prob.variables())
    for v in prob.variables():
        if v.varValue > 0:
            print(v.name, "=", v.varValue)
            optimal_set.append((v.name,v.varValue))

    time3 = time.time()
    print(f"LP problem solve time: {time3 - time2}")
    print(f"\n--------------------------------------\nSummary: "
          f"The optimal rows selected under the requirements are: {optimal_set}. "
          "\n  (Note: 't1' stands for the 1st row)")
    print(f"Total value of objective function based on this set: {value(prob.objective)}")
    print(f"\n----------------------------\nTotal time cost: {time3 - time1}")

    return optimal_set

import time
import re
from pulp import *
from parser import *

#from utils import *

import csv
import numpy as np
from sklearn.cluster import KMeans


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

    '''
    group = []
    for i in range(n):
        group.append([])

    for j in range(len(kmeans_labels)):
        group[kmeans_labels[j]].append(j + 1)
    '''
    if attribute_objective is not None:
        prob += lpSum([tuples[i][attribute_list.index(attribute_objective)] * x_list[i] for i in range(len(x_list))]), "attribute_objective_constraint"

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
        if v.varValue and v.varValue > 0:
            print(v.name, "=", v.varValue)
            optimal_set.append((v.name.replace("t",""),v.varValue))

    time3 = time.time()
    print(f"LP problem solve time: {time3 - time2}")
    print(f"\n--------------------------------------\nSummary: "
          f"The optimal rows selected under the requirements are: {optimal_set}. "
          "\n  (Note: 't1' stands for the 1st row)")
    #print(f"Total value of objective function based on this set: {value(prob.objective)}")
    print(f"\n----------------------------\nTotal time cost: {time3 - time1}")

    return optimal_set

def partition(data, n_clusters):
    
    X = np.array(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    #print(kmeans.labels_)
    #print(kmeans.cluster_centers_)

    return kmeans.labels_, kmeans.cluster_centers_


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
   for i in range(0, k):
       part.append(set())
   kmeans_labels = kmeans.labels_.tolist()
   for m, n in zip(kmeans_labels, new_data):
       part[m].add(tuple(n))
   return kmeans.cluster_centers_.tolist(),part

    
"""
Greedy Backtracking Refine
"""
#functions like direct, query
from queue import PriorityQueue
def greedy_backtracking_refine(q,p,ps,s,header,past=None):
  ''' q: package to be evaluated
      p: partitioning groups
      s: groups yet to be refined
      ps: result of sketch'''
  #print(type(s))#list
  #print(type(s[0]))#tuple
  #print(type(s[0][0]))#set
  #print(type(s[0][1]))#list
  if not past:
    past = set()
  failed_groups = set()
  if not s or len(s)==0:
    return (ps,failed_groups)
  u = list(s)
  l = len(u)-1
  #print(p)
  #print("ps:",ps)
  #print(s)
  while(u):
    g,t = u[l]
    #print("t:",t)
    #
    u.remove(u[l])
    l = l-1
    if not set(g).intersection(ps):
      continue
    a = []
    a.append(header)
    a.extend(g)
    p_i = direct(a,q,past)
    res = []
    for i in p_i:
        res.append(a[int(i.replace("t",""))])
    print(len(res))
    print("Refine direct results p_i ",len(p_i))
    if p_i:#If None then not feasible
      ps_dash = ps.difference(g.union(p_i))#doubt
      s.remove((g,t))
      print("sending to refine :",len(ps_dash),len(p_i))
      pp,f_dash = greedy_backtracking_refine(q,p,ps_dash,s,header,res)
      if len(f_dash)!=0:
        failed_groups = failed_groups.union(f_dash)
        prioritize(u,f)
      else:
        return (pp,failed_groups)
    else:
      if s!=p:
        failed_groups = failed_groups.union((g,t))
        return (None,failed_groups)
    
    #print("updated",l)
  return (None, failed_groups)

from sklearn.cluster import KMeans
import numpy as np
import time
from utils import read_csv_data, sample_rows
import pickle
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
    return kmeans.cluster_centers_.tolist(),part
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

    return optimal_set,data

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



    #(SELECT COUNT(∗) FROM pS WHERE gid = m) ≤ |Gm|
    '''
    group = []
    for i in range(n):
        group.append([])

    for j in range(len(kmeans_labels)):
        group[kmeans_labels[j]].append(j+1)

    for i in range(len(group)):

        prob += lpSum([x_list[group[i][j]] for j in range(len(group[i]))]) <= len(group[i]), "group_count_constraint"
    '''
        
    
    if attribute_objective is not None:
        prob += lpSum([tuples[i][attribute_list.index(attribute_objective)] * x_list[i] for i in range(len(x_list))]), "attribute_objective_constraint"

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
    #print(f"Total value of objective function based on this set: {value(prob.objective)}")
    print(f"\n----------------------------\nTotal time cost: {time3 - time1}")

    return optimal_set
if __name__ == '__main__':
    data, attribute_list = read_csv_data("./tpch.csv")
    print("data read")
# '''total cases'''
    # # sample_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # # n_clusters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500]
    #
    # '''prior cases'''
    # # sample_size = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    # sample_size = [1]
    # n_clusters = [10, 20, 40, 60, 80, 100, 200]
    # for s in sample_size:
    #     dada = sample_rows(data, s)
    #     for n in n_clusters:
    #         print("------------------------------------------")
    #         print(f"running sample_size:{s}, n_clusters:{n}")
    #
    #         t = time.localtime()
    #         current_time = time.strftime("%H:%M:%S", t)
    #         print(current_time)
    #
    #         kmeans = get_kmeans(n, dada)
    #
    #         encoder(partitioned_result=kmeans, name=f"{s}_{n}.pkl")
    #         print(f"saved as {s}_{n}.pkl")
    #
    #
    #
    # '''follow-up cases'''
    # sample_size = [1]
    # n_clusters = [500, 300, 400, 1000, 600, 700, 800, 900]
    # for s in sample_size:
    #     dada = sample_rows(data, s)
    #     for n in n_clusters:
    #         print("------------------------------------------")
    #         print(f"running sample_size:{s}, n_clusters:{n}")
    #
    #         t = time.localtime()
    #         current_time = time.strftime("%H:%M:%S", t)
    #         print(current_time)
    #
    #         kmeans = get_kmeans(n, dada)
    #
    #         encoder(partitioned_result=kmeans, name=f"{s}_{n}.pkl")
    #         print(f"saved as {s}_{n}.pkl")

    """
    use load_partitions(f"{s}_{n}.pkl") to get the partitioned result
    """
    query = """
    select
        package(*) as P
    from
        tpch REPEAT 0
    such that
        sum(o_totalprice) <= 453998.242103 and
        sum(o_shippriority) >= 3 and
        count(*) >= 1
    minimize
        count(*);
"""
    times = []
    sample_size = [1]
    n_clusters = [1]
    for s in sample_size:
        dada = sample_rows(data, s)
        for n in n_clusters:
            c,partition_list = load_partitions(f"{s}_{n}.pkl")
            centers =c
            print(len(partition_list))
            print(len(centers))
            #break
        #break
            #print(partition_list)
            sttime = time.time()
            package = sketch_refine(query=query,
                             n=n,
                             attribute_list=attribute_list,centers = centers
                            ,partition_list = partition_list
            )
            #print(package)
            endtime = time.time()
            print("time taken: ",endtime - sttime)
            times.append(endtime - sttime)
        print(times)