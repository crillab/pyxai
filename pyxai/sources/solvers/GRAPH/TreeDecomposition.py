import itertools
import os


class TreeDecomposition():

    def __init__(self):
        pass


    def create_instance(self, BTs):
        # BTs: the model
        trees = BTs.forest

        variables = set()

        combinations = set()
        for tree in trees:
            variables_set = BTs.get_set_of_variables(tree, tree.root)
            variables = variables.union(variables_set)
            combinations = combinations.union(set(itertools.combinations(variables_set, 2)))
        n_vertices = len(variables)
        n_edges = len(combinations)
        # print(combinations)
        # print(variables)

        instance_lines = []
        instance_lines.append("p tw " + str(n_vertices) + " " + str(n_edges) + "\n")
        for edge in combinations:
            instance_lines.append(str(edge[0]) + " " + str(edge[1]) + "\n")

        self.graph_file = "graph.txt"
        if os.path.exists(self.graph_file):
            os.remove(self.graph_file)
        f = open(self.graph_file, "a")
        for line in instance_lines:
            f.write(line)
        f.close()


    def solve(self):

        return None
