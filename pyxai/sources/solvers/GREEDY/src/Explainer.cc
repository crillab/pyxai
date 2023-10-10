//
// Created by audemard on 22/04/2022.
//

#include "Explainer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <random>
#include <iostream>
#include <chrono>
#include "Tree.h"
#include "bcp/ProblemTypes.h"

static bool abs_compare(int a, int b) { return (std::abs(a) < std::abs(b)); }

void pyxai::Explainer::addTree(PyObject *tree_obj) {
    Tree *tree = new Tree(tree_obj, _type);
    trees.push_back(tree);
}

void pyxai::Explainer::initializeBeforeOneRun(std::vector<bool> &polarity_instance, std::vector<bool> &active_lits,
                                             int prediction) {
    if (_type == Classifier_RF) {
        for (Tree *tree: trees) {
            if (tree->status != DEFINITIVELY_WRONG)
                tree->status = GOOD;
            if (tree->status == GOOD)
                tree->initialize_RF(polarity_instance, active_lits, prediction); // remember to reinit useful lits
        }
    } else {
        for (Tree *tree: trees)
            tree->status = GOOD;
    }
}

bool pyxai::Explainer::compute_reason_conditions(std::vector<int> &instance, int prediction, std::vector<int> &reason, long seed) {
    if(theory_propagator == nullptr) {// No theory exists. Create a fake propagator
        theory_propagator = new Propagator();
        for(pyxai::Tree *t : trees)
            t->propagator = theory_propagator;
    }

    int max = abs(*std::max_element(instance.begin(), instance.end(), abs_compare));
    reason.clear();
    int n_current_iterations = 0;
    pyxai::TimerHelper::initializeTime();
    std::vector<bool> polarity_instance(max + 1, true);
    std::vector<bool> active_lits(max + 1, false);

    std::vector<int> order;
    for (auto l: instance)
        if (is_specific(l))
            order.push_back(l);


    unsigned int best_size = instance.size() + 1, current_size = instance.size();
    for (auto l: instance)
        polarity_instance[std::abs(l)] = l > 0;

    //int nb = 0;
    //for(Tree *tree: trees) nb+=tree->nb_nodes();
    for (auto l: instance) active_lits[abs(l)] = true;

    // FOR Classifier_BT ONLY.
    // Before computing a reason, reduce the size of the tree wrt considered instance
    for (Tree *tree: trees)
        if (_type == pyxai::Classifier_BT)
            tree->initialize_BT(polarity_instance,
                                (n_classes == 2 ? prediction == 1 : int(tree->target_class) == prediction));
        else {

            tree->initialize_RF(polarity_instance, active_lits, prediction);
        }

    // Try to remove excluded features
    initializeBeforeOneRun(polarity_instance, active_lits, prediction);
    for (int l: excluded_features) {
        active_lits[abs(l)] = false;
        propagateActiveLits(excluded_features, polarity_instance, active_lits);
        try_to_remove = l;
        if (is_implicant(polarity_instance, active_lits, prediction) == false)
            return false; // It is not possible to remove excluded features
        theory_propagator->restart();
    }

    // Start to remove conditions
    std::default_random_engine rd = std::default_random_engine(
            seed == -1 ? std::chrono::steady_clock::now().time_since_epoch().count() : 1);
    while (true) {
        // Create an order
        std::shuffle(std::begin(order), std::end(order), rd);
        for (auto l: instance) active_lits[abs(l)] = true; // Init
        for (auto l: excluded_features) active_lits[abs(l)] = false; // Do not want them
        current_size = instance.size() - excluded_features.size();

        initializeBeforeOneRun(polarity_instance, active_lits, prediction);

        // Try to remove literals
        for (int l: order) {
            active_lits[abs(l)] = false;
            try_to_remove = l;
            // std::cout << "try " << l << " ";
            propagateActiveLits(order, polarity_instance, active_lits);

            if (is_implicant(polarity_instance, active_lits, prediction)) {
                current_size--;
                //std::cout << "ok\n";
            }
            else {
                active_lits[abs(l)] = true;
                //std::cout << "impossible\n";
            }
            theory_propagator->restart();
        }

        // We improve the best sol.
        if (current_size < best_size) {
            //std::cout << "current_size:" << current_size << std::endl;
            best_size = current_size;
            reason.clear();
            for (auto l: instance)
                if (active_lits[abs(l)])
                    reason.push_back(l);
        }
        n_current_iterations++;

        if ((time_limit != 0 && pyxai::TimerHelper::realTime() > time_limit)
            || (time_limit == 0 && n_current_iterations > n_iterations))
            return true;
    }
    return true;
}

void pyxai::Explainer::propagateActiveLits(std::vector<int> &order, std::vector<bool> &polarity_instance, std::vector<bool> &active_lits) {
    if(theory_propagator->getNbVar() == 0)
        return;
    for(int l : order) {
        Lit lit = l > 0 ? Lit::makeLitTrue(l) : Lit::makeLitFalse(-l);
        if(theory_propagator->value(lit) == l_False)
            throw std::runtime_error("An error occurs here. The instance seems not valid with the theory");
        if(active_lits[abs(l)] && theory_propagator->value(lit) != l_True) {
            theory_propagator->uncheckedEnqueue(lit);
            bool ret = theory_propagator->propagate();
            if(ret == false)
                throw std::runtime_error("An error occurs here. The instance seems not valid with the theory");

        }
    }
}

bool pyxai::Explainer::is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits,
                                   unsigned int prediction) {
    std::vector<unsigned int> new_wrong_trees;
    for (auto tree: trees) {
        // Init for Classifier_RF
        tree->reachable_classes.clear(); // Update for Classifier_RF
        // Init for Classifier_BT
        tree->get_min = n_classes == 2 ? (prediction == 1) : tree->target_class == prediction;
        tree->firstLeaf = true;

        if (tree->status != GOOD)
            continue;

        // TODO : Fix this trick to reduce the number of calls to is_implicant (only usefull for Classifier_RF with 2 classes).
        //if (tree->used_to_explain[abs(try_to_remove)])
            tree->is_implicant(instance, active_lits, prediction);
    }

    if (_type == Classifier_RF)
        return is_implicant_RF(instance, active_lits, prediction);
    if (_type == Classifier_BT)
        return is_implicant_BT(instance, active_lits, prediction);
    if (_type == Regression_BT)
        return is_implicant_regression_BT(instance, active_lits, prediction);

    return true; // Impossible But do not want a stupid warning. Keep in this form because new _type will arrive
}


bool pyxai::Explainer::is_implicant_BT(std::vector<bool> &instance, std::vector<bool> &active_lits,
                                      unsigned int prediction) {
    if (n_classes == 2) {
        double weight = 0;
        for (Tree *tree: trees)
            weight += tree->current_weight;
        return prediction == (weight > 0);
    }
    // Multi classes case
    std::vector<double> weights(n_classes, 0.0);
    for (Tree *tree: trees)
        weights[tree->target_class] += tree->current_weight;

    double target = weights[prediction];
    for (unsigned int i = 0; i < weights.size(); i++) {
        if (i != prediction && target < weights[i])
            return false;
    }
    return true;
}



bool pyxai::Explainer::is_implicant_regression_BT(std::vector<bool> &instance, std::vector<bool> &active_lits,
                                       unsigned int prediction) {
    double min_weight = base_score;
    double max_weight = base_score;
    for (Tree *tree: trees) {
        //std::cout << tree->current_weight << "\n";
        min_weight += tree->current_min_weight;
        max_weight += tree->current_max_weight;
    }

    //std::cout << lower_bound << " " << max_weight << " " << upper_bound << std::endl;
    return min_weight >= lower_bound && max_weight <= upper_bound;
}


bool pyxai::Explainer::is_implicant_RF(std::vector<bool> &instance, std::vector<bool> &active_lits,
                                      unsigned int prediction) {

    std::vector<unsigned int> new_wrong_trees;
    if (n_classes == 2) {
        unsigned int nb = 0,  i = 0;
        for (Tree *tree: trees) {
            if (tree->reachable_classes.size() == 1 && *(tree->reachable_classes.begin()) == prediction)
                nb++;
            else new_wrong_trees.push_back(i);
            i++;
        }

        if (nb > trees.size() / 2) {
            for (auto i: new_wrong_trees)
                trees[i]->status = CURRENTLY_WRONG;

            for (Tree *tree: trees)
                tree->update_used_lits();
            return true;
        }
        return false;
    }
    // Multiclasses
    std::vector<int> count_classes(n_classes, 0);

    std::fill(count_classes.begin(), count_classes.end(), 0);
    for (Tree *tree: trees)
        if (tree->reachable_classes.size() == 1 && *(tree->reachable_classes.begin()) == prediction) {
            count_classes[prediction]++;
        } else {
            for (auto c: tree->reachable_classes)
                if (c != prediction) count_classes[c]++;
        }


// Compute the best class
    unsigned int best_position = 0;
    for (unsigned int i = 0; i < count_classes.size(); i++) {
        if (count_classes[i] > count_classes[best_position])
            best_position = i;
    }

    if (best_position != prediction)
        return false;
    for (unsigned int i = 0; i < count_classes.size(); i++) {
        if (i != best_position && count_classes[i] == count_classes[best_position])
            return false;
    }
    return true;
}


bool pyxai::Explainer::compute_reason_features(std::vector<int> &instance, std::vector<int> &features, int prediction,
                                              std::vector<int> &reason) {
    assert(false);
    // TODO CHECK FOR FEATURES : V2......
    if (_type != pyxai::Classifier_BT)
        assert(false);
    int max = abs(*std::max_element(instance.begin(), instance.end(), abs_compare));
    int n_current_iterations = 0;
    pyxai::TimerHelper::initializeTime();
    std::vector<bool> polarity_instance(max + 1, true);
    std::vector<bool> active_lits(max + 1, false);

    std::set<int> possible_features(features.begin(), features.end());
    std::vector<int> order(possible_features.begin(), possible_features.end());
    std::map<int, std::vector<int> > features_to_lits;

    for (auto it_order = order.begin(); it_order != order.end(); it_order++) {
        std::vector<int> elements;
        int feature = *it_order;
        for (unsigned int i = 0; i < instance.size(); i++) {
            if (features[i] == feature) elements.push_back(instance[i]);
        }
        features_to_lits[feature] = elements;;
    }

    unsigned int best_size = order.size() + 1;
    unsigned int current_size = order.size();

    for (auto l: instance)
        polarity_instance[std::abs(l)] = l > 0;

    // FOR Classifier_BT ONLY.
    // Before computing a reason, reduce the size of the tree wrt considered instance
    if (_type == pyxai::Classifier_BT)
        for (Tree *tree: trees)
            tree->initialize_BT(polarity_instance,
                                (n_classes == 2 ? prediction == 1 : int(tree->target_class) == prediction));

    while (true) {
        std::shuffle(std::begin(order), std::end(order), std::default_random_engine());
        for (auto l: instance) active_lits[abs(l)] = true; // Init
        current_size = order.size();

        for (int feature: order) {
            std::vector<int> &lits = features_to_lits[feature];
            for (auto it_lits = lits.begin(); it_lits != lits.end(); it_lits++) {
                active_lits[abs(*it_lits)] = false;
            }
            if (is_implicant(polarity_instance, active_lits, prediction) == false) {
                // not a implicant
                for (auto it_lits = lits.begin(); it_lits != lits.end(); it_lits++) {
                    active_lits[abs(*it_lits)] = true;
                }
            } else {
                // alway a implicant
                current_size--;
            }
        }

        if (current_size < best_size) {
            //We are find a better reason :)
            best_size = current_size;
            //Save this new reason
            reason.clear();
            for (auto l: instance)
                if (active_lits[abs(l)])
                    reason.push_back(l);
        }
        n_current_iterations++;

        if ((time_limit != 0 && pyxai::TimerHelper::realTime() > time_limit)
            || (time_limit == 0 && n_current_iterations > n_iterations))
            return true;
    }
    return true;
}











