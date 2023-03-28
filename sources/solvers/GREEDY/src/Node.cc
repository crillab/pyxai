//
// Created by audemard on 22/04/2022.
//

#include "Node.h"
#include "bcp/ProblemTypes.h"
double pyxai::Node::compute_weight(std::vector<bool> &instance, std::vector<bool> &active_lits, bool get_min) {
    if (is_leaf())
        return leaf_value.weight;
    if (active_lits[lit]) { // Literal in implicant
        if (instance[lit]) // positive lit in instance
            return true_branch->compute_weight(instance, active_lits, get_min);
        else
            return false_branch->compute_weight(instance, active_lits, get_min);
    }

    // Literal not in implicant
    double wf = false_branch->compute_weight(instance, active_lits, get_min);
    // Do not traverse right branch // TODO CHeck
    //if(get_min && wf < true_min) return wf;
    //if(!get_min && wf > true_max) return wf;
    double wt = true_branch->compute_weight(instance, active_lits, get_min);
    if (get_min)
        return std::min(wf, wt);
    return std::max(wf, wt);
}


void pyxai::Node::performOnLeaf() {
    if(tree->_type == Classifier_RF) {
        tree->reachable_classes.insert(leaf_value.prediction);
        return;
    }

    if(tree->_type == Classifier_BT || tree->_type == Regression_BT) {
        if(tree->firstLeaf) {
            tree->current_weight = leaf_value.weight;
            tree->current_min_weight = leaf_value.weight;
            tree->current_max_weight = leaf_value.weight;
        } else {
            if(tree->get_min)
                tree->current_weight = std::min(tree->current_weight, leaf_value.weight);
            else
                tree->current_weight = std::max(tree->current_weight, leaf_value.weight);
            tree->current_min_weight =  std::min(tree->current_min_weight, leaf_value.weight);
            tree->current_max_weight =  std::max(tree->current_max_weight, leaf_value.weight);
        }
        tree->firstLeaf = false;
    }
}

void pyxai::Node::is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction) {
    if (is_leaf()) {
        performOnLeaf();
        return;
    }

    tree->used_lits.push_back(lit); // literal useful for prediction
    Lit normalLit = instance[lit] ? Lit::makeLitTrue(lit) : Lit::makeLitFalse(lit);
    if (active_lits[lit] || tree->propagator->value(normalLit) == l_True) { // Literal in implicant
        if (instance[lit]) // positive lit in instance
            true_branch->is_implicant(instance, active_lits, prediction);
        else
            false_branch->is_implicant(instance, active_lits, prediction);;
        return;
    }

    true_branch->is_implicant(instance, active_lits, prediction);
    false_branch->is_implicant(instance, active_lits, prediction);
}



void pyxai::Node::reduce_with_instance(std::vector<bool> &instance, bool get_min) {
    if(is_leaf()) return; // Nothing to do

    false_branch->reduce_with_instance(instance, get_min);
    true_branch->reduce_with_instance(instance, get_min);
    if(false_branch->is_leaf() && true_branch->is_leaf()) { // V1
        // TODO : si lit de base n'est pas dans l'instance ?????
        double instance_w = instance[lit] ? true_branch->leaf_value.weight : false_branch->leaf_value.weight;
        double not_instance_w = instance[lit] ? false_branch->leaf_value.weight : true_branch->leaf_value.weight;
        if((get_min && instance_w < not_instance_w) || (!get_min && instance_w > not_instance_w)) {
            artificial_leaf = true;
            leaf_value.weight = instance_w;
        }
    }
}

double pyxai::Node::extremum_true_branch(bool get_min) {
    if(is_leaf()) return leaf_value.weight;

    double wf = false_branch->extremum_true_branch(get_min);
    double tf = true_branch->extremum_true_branch(get_min);
    if(get_min)
        true_min = tf;
    else
        true_max = tf;
    return get_min ? std::min(wf, tf) : std::max(wf, tf);
}

int pyxai::Node::nb_nodes() {
    if(is_leaf()) return 1;
    return 1 + true_branch->nb_nodes() + false_branch->nb_nodes();
}