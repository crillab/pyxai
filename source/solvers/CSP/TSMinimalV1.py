""" from pycsp3 import *

protect()

class MinimalV1():


  def __init__(self):
    pass

  def data_formatting(self, explainer):
    # data_matrises is a list of matrises where each matrix represent a tree where each line is the values of a subset of the implicant. 
    # Moreover, at the end of the line, there is the associated worst or best weight. 
    data_matrises = []
    data_domains_weights = []
    data_literals_per_matrix = []
    data_classes = []
    for tree in explainer.boosted_trees.forest:
      data_classes.append(tree.target_class)
      variables = tree.get_variables(explainer.implicant)
      #print("variables:", len(variables))
      subsets = list(chain.from_iterable(combinations(variables, r) for r in range(len(variables)+1)))
      #print("size subsets:", len(subsets))
      values_per_subset = [tuple(1 if element in subset else 0 for element in variables) for subset in subsets]
      
      matrix = []
      data_domains_weights_tree = []
      for i, subset in enumerate(subsets):
        weights = explainer.compute_weights(tree, tree.root, subset)
        if explainer.boosted_trees.n_classes == 2:
          weight_value = explainer.weight_float_to_int(min(weights) if explainer.target_prediction == 1 else max(weights))
        else:
          weight_value = explainer.weight_float_to_int(min(weights) if explainer.target_prediction == tree.target_class else max(weights))
        data_domains_weights_tree.append(weight_value)
        line = tuple([element for element in values_per_subset[i]] + [weight_value])
        matrix.append(line)
      data_domains_weights.append(tuple(sorted(list(set(data_domains_weights_tree)))))
      data_matrises.append(matrix if variables != [] else None)
      data_literals_per_matrix.append([explainer.implicant.index(v) for v in variables])
    return data_matrises, data_domains_weights, data_literals_per_matrix, data_classes


  def create_model_minimal_abductive_BT(self, implicant, data_matrises, data_domains_weights, data_literals_per_matrix, data_classes, prediction):

    n_trees = len(data_matrises)

    # Say if a literal of the implicant is enabled or not 
    literals = VarArray(size=len(implicant), dom={0, 1})

    # The weight of each tree according to the variables 'literals' 
    weights = VarArray(size=[n_trees], dom=lambda x: data_domains_weights[x])
     
    #exit(0)
    satisfy(
      # Constrain table fixing the weight of each tree according to the variable 'literal'
      ([literals[lit] for lit in data_literals_per_matrix[id_tree]]+[weights[id_tree]] in data_matrises[id_tree] for id_tree in range(n_trees) if data_matrises[id_tree] is not None),
        
      # Take only these that classify always correctly the implicant
      [Sum(weights[i] for i in range(n_trees) if data_classes[i] == prediction) > Sum(weights[i] for i in range(n_trees) if data_classes[i] == other_class) for other_class in set(data_classes) if other_class != prediction]
    )

    # The goal is to have the most little subset :)
    minimize(Sum(literals))


  def solve(self):
    ace = solver(ACE)
    ace.setting("-ale=4 -p=SAC3 -so")
    instance = compile()
    result = ace.solve(instance, verbose=True)
    print(result)
    print(solution().values)
    if result == OPTIMUM:
      return [value for i, value in enumerate(solution().values) if 'literals' in solution().variables[i].id]
    return None """
