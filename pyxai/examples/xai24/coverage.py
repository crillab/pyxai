from pyxai.sources.solvers.COMPILER.D4Solver import D4Solver
import random

# -------------------------------------------------------------------------------------
# Compute coverage


class Coverage:

    def __init__(self, sigma, nb_v, time_limit, user):
        self.first_call = True
        self.nb_models_in_theory = None
        self.nb_variables = nb_v
        self.time_limit = time_limit
        self.sigma = sigma
        self.user = user

    def number_of_models_for_rules(self, rules):
        # Special cases
        if len(rules) == 0:
            return 0
        if len(rules) == 1 and len(rules[0]) == 0:
            return self.nb_models_in_theory


        compiler = D4Solver(filenames="/tmp/rules"+str(random.randint(1,100000)))
        cnf = self.sigma.copy()
        aux = self.nb_variables
        new_clause = []
        for rule in rules:
            if len(rule) == 1:
                new_clause.append(rule[0])
                continue
            aux += 1
            for lit in rule:
                cnf.append([-aux, lit])
            cnf.append([aux] + [-lit for lit in rule])
            new_clause.append(aux)

        cnf.append(new_clause)
        compiler.add_cnf(cnf, aux)

        return compiler.count(time_limit=self.time_limit)

    def coverage(self):
        if self.first_call:  # Compute once the number of model of the theory
            first_call = False
            models_in_sigma = D4Solver(filenames="/tmp/sigma-")
            models_in_sigma.add_cnf(self.sigma, self.nb_variables)
            self.nb_models_in_theory = models_in_sigma.count(time_limit=self.time_limit)
        if self.nb_models_in_theory is None:
            return None

        nb_pos = self.number_of_models_for_rules(self.user.positive_rules)
        if nb_pos is None:
            return None
        nb_neg = self.number_of_models_for_rules(self.user.negative_rules)
        if nb_neg is None:
            return None
        tmp = (nb_pos + nb_neg) / self.nb_models_in_theory
        if tmp > 1.0:
            print(nb_pos, nb_neg, self.nb_models_in_theory)
            sys.exit(1)
        assert(tmp <= 1.0)
        return tmp


    def test(self, sigma, positives, negatives, nb_variables):
        models_in_sigma = D4Solver(filenames="/tmp/sigma-")
        models_in_sigma.add_cnf(self.sigma, self.nb_variables)
        nb_sigma = models_in_sigma.count()
        nb_pos = self.number_of_models_for_rules(positives)
        nb_neg = self.number_of_models_for_rules(negatives)
        print(nb_sigma, nb_pos, nb_neg)






