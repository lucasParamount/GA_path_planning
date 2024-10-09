# GA_path_planning
a path planning method based on genetic algorithm
open main.py file, and run.
This is a pure GA, without any other algorithm mixed.

First, generate several nodes randomly, and connect them randomly, so as to form the first generation.
Then, start iteration until the maximum iteration number is met.
(each individual in a generation group is a line of a matrix, and each individual is consisted of nodes that the path goes through)
Last, modify the best individual by checking if some mid nodes could be skipped 
eg. given final path :(2, 2) (4, 5) (6, 4) .....(321, 452)
    check if we can skip (4, 5) and connect (2, 2) with (6, 4) directly.


