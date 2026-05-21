
This program (in C++) is used to compute a k-anchored explanation for a given instance. 

# Dependencies

```
sudo apt install cmake libboost-all-dev
```

# Installation

```
./build.sh
```

## How to use

Options: 

- "-i": CNF file representing the model.
- "-p": The set of positive reference instances furnished by the user, in the form of a CNF file of binary variables.
- "-n": The set of negative reference instances furnished by the user, in the form of a CNF file of binary variables.
- "-x": The instance to be explained in the form of binary variables.
- "-k": The number of reference instances (positive and negative) that the explanation to be computed must cover. 
- "-m": The method used.
- "-c": To check whether a given implicant in the form of binary variables is k-anchored.   

if the instance is positive, put negF for the model and -p positive_anchors -n negative_anchors
if the instance is negative, put    F for the model and -p negative_anchors -n positive_anchors

Example:

```
./build/encore -i instances/paper_example/positive_f.cnf -p instances/paper_example/ac_moins.txt -n instances/paper_example/ac_plus.txt -x instances/paper_example/x.inst -k 2 -m toto
```

```
./build/encore -i instances/paper_example/negative_f.cnf -p instances/paper_example/ac_plus.txt -n instances/paper_example/ac_moins.txt -x instances/paper_example/x2.inst -k 2 -m toto -c instances/paper_example/implicant2.inst
```
