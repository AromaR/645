"""
PaQL Parser: Translate PaQL query to DIRECT input variables
"""

"""
Author: Jiang Li (jiangli@umass.edu)
"""

import re

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