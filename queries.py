from parser import *

"""
PaQL queries
"""

query_1 = """
    select
        package(*) as P
    from
        tpch REPEAT 0
    such that
        sum(sum_base_price) <= 15469853.7043 and
        sum(sum_disc_price) <= 45279795.0584 and
        sum(sum_charge) <= 95250227.7918 and
        sum(avg_qty) <= 50.353948653 and
        sum(avg_price) <= 68677.5852459 and
        sum(avg_disc) <= 0.110243522496 and
        sum(sum_qty) <= 77782.028739 and
        count(*) >= 1
    maximize
        sum(count_order);
    """

query_2 = """
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

query_3 = """
    select
        package(*) as P
    from
        tpch REPEAT 0
    such that
        sum(revenue) >= 413930.849506 and
        count(*) >= 1
    minimize
        count(*);
"""

query_4 = """
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

queries = [query_1, query_2, query_3, query_4]