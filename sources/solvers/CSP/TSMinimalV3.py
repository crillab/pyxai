""" from re import M
from pycsp3 import *


protect()

from pyxai.core.tools.utils import flatten

class id_node(int):
  pass

max_weight = -float("inf")
min_weight = float("inf")

class MinimalV3():

  def __init__(self):
    pass

  

  def weight_float_to_int(self, weight):
    global max_weight
    global min_weight
    if weight > max_weight:
      max_weight = weight
    if weight < min_weight:
      min_weight = weight
      
    x = int(weight*pow(10,9))
    return x

  def compute_possible_weights(self, dict_nodes, i):
     output = []
     left = dict_nodes[i][1]
     right = dict_nodes[i][2]
     value = dict_nodes[i][3]

     if left is None and right is None:
       return value
     else:
       result = self.compute_possible_weights(dict_nodes, right)
       output.extend(result if isinstance(result, list) else [result])
       result = self.compute_possible_weights(dict_nodes, left)
       output.extend(result if isinstance(result, list) else [result])
     return output

  def compute_id_nodes(self, tree, node, dict_nodes, index):
    current_index = index
    
    if node.is_leaf():
      dict_nodes[current_index] = [None, None, None, self.weight_float_to_int(node.value)]
      return index

    id_variable = tree.get_id_variable(node) 
    
    index_left = index + 1
    index = self.compute_id_nodes(tree, node.left, dict_nodes, index_left)
   
    index_right = index + 1
    index = self.compute_id_nodes(tree, node.right, dict_nodes, index_right)
    
    dict_nodes[current_index] = [id_variable, index_left, index_right, None]

    return index

  def compute_data(self, trees):
    nodes = []
    for tree in trees: 
      dict_nodes = dict()
      #if not tree.root.is_leaf():
      self.compute_id_nodes(tree, tree.root, dict_nodes, 0)
      for i in dict_nodes.keys():
        weights = self.compute_possible_weights(dict_nodes, i)
        dict_nodes[i][3] = weights if isinstance(weights, list) else [weights]
      nodes.append(dict_nodes)
    return nodes

  def create_model_minimal_abductive_BT(self, implicant, BTs, prediction, n_classes, implicant_id_features):
    nVariables = len(implicant)
    trees = BTs.forest
    idVariableToLiteral = {abs(v):v for v in implicant}
    literalToPosition = {v:i for i, v in enumerate(implicant)}
    map_id_features = {feature:[i for i, v in enumerate(implicant) if implicant_id_features[i] == feature] for feature in implicant_id_features}
    
    nTrees = len(trees)

    # TODO : if implicant is not instance ??
    # keep trees in initial state
    # Reduce all trees
    
    BTs.reduce_trees(implicant, prediction)

    nodes = self.compute_data(trees)
    nNodes = [len(nodes[i]) for i in range(nTrees)]
    nNodesMax = max(nNodes)
    data_classes = [tree.target_class for tree in trees]

    # To clear a old model
    clear()
    
    # say if a literal of the implicant is activated (1) or not (0)
    s = VarArray(size=nVariables, dom={0,1})
    
    # one variable on each node, the domains are the possible weights
    n = VarArray(size=[nTrees, nNodesMax], dom=lambda t,n: set(nodes[t][n][3]) if n in nodes[t] else None)
    
    # auxiliaries variables to handle the minimum constraints
    aux = VarArray(size=[nTrees, nNodesMax], dom=lambda t,n: set(nodes[t][n][3]) if n in nodes[t] and nodes[t][n][0] is not None else None)

    def create_intensions(t, dict_nodes):
      intensions = []

      for key in dict_nodes.keys():
        id_variable = dict_nodes[key][0]
        if id_variable is not None: #Do not into account the leaves
          literal = idVariableToLiteral[id_variable]
          position = literalToPosition[literal]
          left = dict_nodes[key][1]
          right = dict_nodes[key][2]
          #the case where s[postion] is activated (equal to 1)
          if literal < 0: # equal to the left side
            intensions.append(imply(s[position] == 1, n[t][key] == n[t][left]))
          else: # equal to the right side
            intensions.append(imply(s[position] == 1, n[t][key] == n[t][right]))

          #the case where s[postion] is deactivated (equal to 0)

          if n_classes > 2: #multi-classes
            if data_classes[t] == prediction:
              intensions.append(aux[t][key] == Minimum(n[t][left], n[t][right]))
            else:
              intensions.append(aux[t][key] == Maximum(n[t][left], n[t][right]))
          else:#2-classes
            if prediction == 1:
              intensions.append(aux[t][key] == Minimum(n[t][left], n[t][right]))
            else:
              intensions.append(aux[t][key] == Maximum(n[t][left], n[t][right]))
          intensions.append(imply(s[position] == 0, n[t][key] == aux[t][key]))

      return intensions

    if implicant_id_features != []:
      #if we are in the ReasonExpressivity.Features mode
      satisfy(
        [AllEqual(s[i] for i in map_id_features[feature]) for feature in map_id_features.keys() if len(map_id_features[feature]) > 1]
      )

    satisfy(
      [create_intensions(t, nodes[t]) for t in range(nTrees)]
    )

    if n_classes > 2: #multi-classes case
      satisfy(
        # Take only these that classify always correctly the implicant
      [Sum(n[i][0] for i in range(nTrees) if data_classes[i] == prediction) > Sum(n[i][0] for i in range(nTrees) if data_classes[i] == other_class) for other_class in set(data_classes) if other_class != prediction]
      )
    else: #2-classes case
      if prediction == 1:
        satisfy(
          Sum(n[t][0] for t in range(nTrees)) > 0,
        )
      else:
        satisfy(
          Sum(n[t][0] for t in range(nTrees)) < 0,
        )
    if implicant_id_features != []:
      #if we are in the ReasonExpressivity.Features mode 
      minimize(Sum(s[map_id_features[feature][0]]  for feature in map_id_features.keys()))
    else:    
      minimize(Sum(s))

    # keep trees in initial state
    BTs.remove_reduce_trees()


  def solve(self, time_limit):
    
    ace = solver(ACE)
    t = " -t=" + str(time_limit) + "s" if time_limit != 0 else ""
    ace.setting("-ale=4 -di=0 -valh=Last" + t)
    instance = compile()  
    result = ace.solve(instance, verbose=True)


    if result == OPTIMUM or result == SAT:
      return result, [value for i, value in enumerate(solution().values) if solution().variables[i] is not None and 's' in solution().variables[i].id]
    return result, [] """
