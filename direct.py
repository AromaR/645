import time
from pulp import *
from queries import *
from utils import *

"""
Author: Jiang Li (jiangli@umass.edu)
"""

def direct(
        data: list,
        attribute_list: list,
        query: str
):
    tuples = data
    variables = paql_to_variables(query)

    table_name = variables['TABLE_NAME']
    objective = variables['OBJECTIVE']
    attribute_objective = variables['ATTRIBUTE_OBJECTIVE']
    constraints = variables['CONSTRAINTS']
    count_constraint = variables['COUNT_CONSTRAINT']

    # tuples, attribute_list = attribute_filter(tuples, attribute_objective, constraints, attribute_list)

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
        prob += lpSum(x_list), "attribute_objective_constraint_count"

    if count_constraint[0] is not None:
        prob += count_constraint[0] <= lpSum(x_list), "count_lower_constraint"
    if count_constraint[1] is not None:
        prob += lpSum(x_list) <= count_constraint[1], "count_upper_constraint"

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
        if v.varValue == None:
            continue
        if v.varValue > 0:
            # print(v.name, "=", v.varValue)
            optimal_set.append(v.name)

    time3 = time.time()
    print(f"LP problem solve time: {time3 - time2}")
    # print(f"\n--------------------------------------\nSummary: "
    #       f"The optimal rows selected under the requirements are: {optimal_set}. "
    #       "\n  (Note: 't1' stands for the 1st row)")
    objective = value(prob.objective) if prob.objective is not None else len(optimal_set)
    # print(f"Total value of objective function based on this set: {objective}")
    print(f"\n----------------------------\nTotal time cost: {time3 - time1}")

    if prob.status == 1:
        return optimal_set
    else:
        return None


if __name__ == '__main__':

    '''
    Demonstration
    '''
    # package = direct(
    #     data=[
    #         (7.1, 0.45),
    #         (5.2, 0.55),
    #         (3.2, 0.25),
    #         (6.5, 0.15),
    #         (2.0, 1.20)
    #     ],
    #     attribute_list=("sat_fat", "kcal"),
    #     query="""
    #         SELECT PACKAGE(*) AS P
    #         FROM Recipes REPEAT 0
    #         SUCH THAT
    #             COUNT(*) = 3
    #             AND SUM(kcal) >= 2.0
    #             AND SUM(kcal) <= 2.5
    #         MINIMIZE SUM(sat_fat)
    #     """
    # )

    data, attribute_list = read_csv_data("./tpch.csv")

    sttime = time.time()
    package = direct(
        data=data,
        attribute_list=attribute_list,
        query="""
        select
            package(*) as P
        from
            tpch REPEAT 0
        such that
            sum(p_size) <= 8 and
            count(*) >= 1
        minimize
            sum(ps_min_supplycost);
        """
    )
    endtime = time.time()
    print("time taken: ", endtime - sttime)

    print(package)
