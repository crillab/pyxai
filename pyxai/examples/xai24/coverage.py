from pyxai.sources.solvers.COMPILER.D4Solver import D4Solver


# -------------------------------------------------------------------------------------
# Compute coverage


class Coverage:

    def __init__(self, nb_v, time_limit):
        self.first_call = True
        self.nb_models_in_theory = None
        self.nb_variables = nb_v
        self.time_limit = time_limit

    def number_of_models_for_rules(self, sigma, rules):
        compiler = D4Solver()
        cnf = sigma.copy()
        aux = self.nb_variables
        for rule in rules:
            aux += 1
            for lit in rule:
                cnf.append([-aux, lit])
            cnf.append([aux] + [-lit for lit in rule])

        compiler.add_cnf(cnf, aux)
        return compiler.count(time_limit=self.time_limit)

    def coverage(self, sigma, positive_rules, negative_rules):
        if self.first_call:  # Compute once the number of model of the theory
            first_call = False
            models_in_sigma = D4Solver()
            models_in_sigma.add_cnf(sigma, self.nb_variables)
            self.nb_models_in_theory = models_in_sigma.count(time_limit=self.time_limit)
            first_call = False
        assert (self.nb_models_in_theory is not None)

        nb_pos = self.number_of_models_for_rules(sigma, positive_rules)
        nb_neg = self.number_of_models_for_rules(sigma, negative_rules)
        return (nb_pos + nb_neg) / self.nb_models_in_theory
