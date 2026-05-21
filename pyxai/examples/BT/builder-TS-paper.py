
# As a matter of illustration, consider the following loan allocation problem viewed as a binary classification problem. The goal is to determine whether a loan of $\$100k$ must be granted
# to an applicant, described using three primitive attributes from $\mathcal{A} = \{A, I, S\}$: his/her age $A$ (a numerical attribute), his/her annual income $I$ (in $\$k$, a numerical attribute), and his/her professional status $S$ (a categorical attribute). 
# %In the available dataset $D$, 
# In $F$, three values for $S$ are encountered (``unemployed (U)'',
# ``temporary position (TP)'', or ``permanent position (PP)''). Therefore, the running domain of $S$ is $\{U, TP, PP\}$. 
# The boosted tree $F = \{T_1, T_2, T_3\}$ given at Figure \ref{fig:RF} was learned from 
# $D$. $F$ is based on $7$ Boolean conditions: $X = \{x_1, \ldots, x_7\}$, where $x_1 = (A > 25)$, $x_2 = (A > 60)$, $x_3 = (I > 30)$, $x_4 = (I > 50)$, 
# $x_5 = (S = U)$, $x_6 = (S = TP)$, and $x_7 = (S = PP)$. Those conditions are logically connected as given by the domain theory $Th_F = (x_2 \Rightarrow x_1) \wedge
# (x_4 \Rightarrow x_3) \wedge (x_5 \Rightarrow \overline{x_6}) \wedge (x_5 \Rightarrow \overline{x_7}) \wedge (x_6 \Rightarrow \overline{x_7})$.

# At the leaf of each branch $b$ of $T_i$ ($i \in [3]$), one can find the regression value $T_i(\vec x)$ that is obtained for every instance $\vec x$ that is compatible with $b$.

# The instance $(22, 34, TP)$ over $\mathcal{A}$ representing an applicant aged $22$, having $\$34k$ annual income, and holding a temporary position, 
# corresponds to the instance $\vec x = (0,$ $0,$ $1,$ $0,$ $0,$ $1,$ $0)$ from $\vec X$. The instance $\vec x = (0,$ $0,$ $1,$ $0,$ $0,$ $1,$ $0)$ over $X$
# is more general than the instance $(22, 34, TP)$ over $\mathcal{A}$ one started with, in the sense that other instances (e.g., $(20, 40, TP)$) correspond to the same instance $\vec x$ over $X$.

# We have $T_1(\vec x) = -1$, $T_2(\vec x) = -2$, and
# $T_3(\vec x) = -1$, so that $F(\vec x) = -4$.
# Since $F(\vec x) \leq 0$, $\vec x$ is classified by $F$ as a negative instance (hence, so are $(22, 34, TP)$ and $(20, 40, TP)$).

# Check V1.0: Ok
# Check V2.0: OK

from pyxai import Builder, Explaining, Learning

nodeI_x_3 = Builder.DecisionNode(2, operator="GT", threshold=30, left=-2, right=4)
nodeS_x_7 = Builder.DecisionNode(5, operator="EQ", threshold=1, left=-3, right=3)
nodeA_x_2 = Builder.DecisionNode(1, operator="GT", threshold=60, left=nodeI_x_3, right=nodeS_x_7)
nodeA_x_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=25, left=-1, right=nodeA_x_2)

tree1 = Builder.DecisionTree(5, nodeA_x_1)

nodeS_x_6 = Builder.DecisionNode(4, operator="EQ", threshold=1, left=-2.5, right=4)
nodeI_x_4 = Builder.DecisionNode(2, operator="GT", threshold=50, left=-2, right=nodeS_x_6)
nodeS_x_5 = Builder.DecisionNode(3, operator="EQ", threshold=1, left=nodeI_x_4, right=-2)

tree2 = Builder.DecisionTree(5, nodeS_x_5)

nodeI_x_4_bis = Builder.DecisionNode(2, operator="GT", threshold=50, left=-1, right=2)
nodeA_x_1_bis = Builder.DecisionNode(1, operator="GT", threshold=25, left=-2, right=3)
nodeS_x_7_bis = Builder.DecisionNode(5, operator="EQ", threshold=1, left=nodeI_x_4_bis, right=nodeA_x_1_bis)

tree3 = Builder.DecisionTree(5, nodeS_x_7_bis)

BTs = Builder.BoostedTrees([tree1, tree2, tree3], n_classes=2, feature_names=["A", "I", "S0", "S1", "S2"])

instance = (22, 34, 0, 1, 0)


print("instance:", instance)

explainer = Explaining.initialize(BTs, instance, features_type={"numerical": ["A","I"], "categorical": {'{S0,S1,S2}': ('U','TP','PP')}})
print("binary_representation:", explainer.binary_representation)

print("binary_representation:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

# binary_representation: (-1, -2, 3, -4, -5, -6, 7)
# (0, 0, 1, 0, 0, 1, 0)

print("predict:", explainer.predict(instance))
print(" Direct ---------------------------------------------------")
direct = explainer.direct_reason()
print("direct reason:", direct)
direct_features = explainer.to_features(direct)
print("to_features:", direct_features)
#assert direct_features == ('f1 > 2', 'f2 > 1', 'f3 == 1', 'f4 == 1'), "The direct reason is not correct."

print(" Tree Specific ---------------------------------------------------")
tree_specifics = explainer.minimal_tree_specific_reasons(n=1)
print("elapsed time:", explainer.elapsed_time)
print("tree specific reasons:", tree_specifics)
print("to_features:", explainer.to_features(tree_specifics, eliminate_redundant_features=False))

