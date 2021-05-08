# 645 Project Code

### Running Environments

```
Python 3.8
PuLP 2.4
scikit-learn 0.24.1
numpy 1.20.2
pickleshare 0.7.5
```

### How to Run

For Direct:

Put `tpch.csv` dataset in the same folder, and run `direct.py`, there's a sample code written in the main function of it. 

For PaQL parser:

There's a function called `paql_to_variables(paql)` in `parser.py`, which takes PaQL query as input and return variables that need.

For offline partition:

The main function of `offline_partition.py` contains the code of generating kmeans models as partition results.

For visualization:

Add our time results to `figure7.py` and `figure8.py`. Simply run the code of them , and we can get the figures.

For Sketch - refine:

Put the pkl files in the same folder, change query in the `sketch-refine-centers-aroma.py` and run it. You can change the sample size in the main function or the cluster numbers. This is the normal sketch refine

For Sketch - refine - min/max:

Put the pkl files in the same folder, change query in the `sketch-refine-maxcenter.py` and run it. You can change the sample size in the main function or the cluster numbers. This is a modified sketch that uses the min/max of the partitions instead of the k means centers.

For Sketch - refine - hybrid:

Put the pkl files in the same folder, change query in the `sketch-refine-hybrid.py` and run it. You can change the sample size in the main function or the cluster numbers. This is the hybrid sketch.

For Sketch-Refine-function:

Run `sketch.py`, there's a sample code written in the main function of it. 

The function `get_partition` in `sketch.py`, which takes data from .csv file as input and return partition_list which include every group's actual tuples and its representative.

For hybrid-sketch-function:

Run `hybrid-sketch.py`, there's a sample code written in the main function of it. 

The function `get_partition` in `hybrid-sketch.py`, which takes data from .csv file as input and return partition_list which include every group's actual tuples and its representative.

