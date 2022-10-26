from collections.abc import Iterable


class DNF(list):
    pass


class CNF(list):
    pass


class PrimeImplicantCNF:

    def __init__(self, core):
        self.cnf = [clause for clause in core if len(clause) != 1]
        self.necessary = {clause[0] for clause in core if len(clause) == 1}
        if self.cnf == []:  # Special case when all clauses are of size 1.
            return
        self.mapping_original_to_new = None
        self.mapping_new_to_original = None
        self.n_literals_mapping = 1
        self.n_literals = 0
        self.new_mapping_id_variables()

        self.bijection_clauses = []
        self.compute_bijection()
        self.cnf += self.bijection_clauses


    def get_reason_from_model(self, model):
        reason = [lit for lit in self.necessary]
        reason += [self.mapping_new_to_original[lit] for lit in model if
                   lit > 0 and abs(lit) < self.n_literals_mapping and self.mapping_new_to_original[lit] not in reason]
        # TODO check self.mapping_new_to_original[l] not in sufficient is Ok or not ?
        return reason


    def get_blocking_clause(self, model):
        return [-lit for lit in model if lit > 0 and abs(lit) < self.n_literals_mapping]


    def compute_bijection(self):
        occurrences = [[] for _ in range(self.n_literals_mapping)]
        for i, clause in enumerate(self.cnf):
            for lit in clause:
                occurrences[lit].append(i)

        self.n_literals = self.n_literals_mapping
        for lit in range(1, self.n_literals_mapping):
            new_clause1 = [-lit]
            for j in occurrences[lit]:
                clause = self.cnf[j]
                if len(clause) == 2:
                    new_clause1.append(-clause[0] if clause[1] == lit else -clause[1])
                    continue
                new_clause1.append(self.n_literals)
                # nblit = -l1 and -l2 ... Add to bijection (tseitsin
                new_clause2 = [self.n_literals]
                for l2 in clause:
                    if l2 == lit:
                        continue
                    self.bijection_clauses.append([-self.n_literals, -l2])
                    new_clause2.append(l2)
                self.bijection_clauses.append(new_clause2)
                self.n_literals = self.n_literals + 1
            self.bijection_clauses.append(new_clause1)
        self.bijection_clauses = CNFencoding.format(self.bijection_clauses)


    def new_mapping_id_variables(self):
        """
        Create new id variables for the cnf in the form of list.
        Allow to clean the current missing id variables.
        Warning: do not take into account the signs of literals
        Example cnf = [[-1, 4, 8], [1, 10]] -> [[1, 2, 3], [1, 4]]
        Result list of the example: [0,1,0,0,2,0,0,0,3,0,4]
        """
        size = CNFencoding.compute_max_id_variable(self.cnf)
        mapping_original_to_new = [0 for _ in range(size + 1)]
        mapping_new_to_original = [0]
        for clause in self.cnf:
            for lit in clause:
                if mapping_original_to_new[abs(lit)] != 0:
                    continue
                mapping_original_to_new[abs(lit)] = self.n_literals_mapping
                mapping_new_to_original.append(lit)
                self.n_literals_mapping += 1
        self.mapping_original_to_new = mapping_original_to_new
        self.mapping_new_to_original = mapping_new_to_original

        # Apply the mapping
        for clause in self.cnf:
            for i, lit in enumerate(clause):
                clause[i] = self.mapping_original_to_new[abs(lit)]
        self.cnf = CNFencoding.format(self.cnf)


    def from_original_to_new(self, lit):
        return self.mapping_original_to_new[abs(lit)]


    def from_new_to_original(self, lit):
        assert lit > 0
        return self.mapping_new_to_original[lit]


class CNFencoding():

    @staticmethod
    def compute_set_variables(formula):
        set_variables = set()
        for cube in formula:
            set_variables.update(abs(lit) for lit in cube)
        return set_variables


    @staticmethod
    def compute_n_variables(formula):
        return len(CNFencoding.compute_set_variables(formula))


    @staticmethod
    def compute_max_id_variable(formula):
        if isinstance(formula, Iterable):
            if isinstance(formula[0], Iterable):
                return max(abs(lit) for c in formula for lit in c)
            else:
                return max(abs(lit) for lit in formula)
        else:
            assert False, "The formula have to be an Iterable but it is: " + type(formula) + " !"


    @staticmethod
    def tseitin(dnf):
        n_variables = CNFencoding.compute_n_variables(dnf)
        id_aux_var = n_variables + 1  # for new auxiliary variables
        clauses = CNF()

        for cube in dnf:
            clauses.append([-1 * cube[j] for j in range(len(cube))] + [id_aux_var])

            for lit in cube:
                clauses.append([lit, -1 * id_aux_var])
            id_aux_var += 1

        clauses.append([i for i in range(n_variables + 1, n_variables + len(dnf) + 1)])
        return CNF(clauses)


    @staticmethod
    def complementary(dnf):
        return CNF([[-1 * lit for lit in cube] for cube in dnf])


    @staticmethod
    def literals_in_implicant(cnf, implicant):
        return CNF([[lit for lit in clause if lit in implicant] for clause in cnf])


    @staticmethod
    def remove_subsumed(cnf):
        cnf = sorted(cnf, key=lambda clause: len(clause))
        subsumed = [False for _ in range(len(cnf) + 1)]
        flags = [False for _ in range(CNFencoding.compute_max_id_variable(cnf) + 1)]
        for i, clause in enumerate(cnf):
            if subsumed[i]:
                continue
            for lit in clause:
                flags[abs(lit)] = True
            for j in range(i + 1, len(cnf)):
                nLiteralsInside = tuple(flags[abs(lit)] for lit in cnf[j]).count(True)
                if nLiteralsInside == len(clause):
                    subsumed[j] = True
            for lit in clause:
                flags[abs(lit)] = False
        return CNF([clause for i, clause in enumerate(cnf) if not subsumed[i]])


    @staticmethod
    def format(cnf):
        if len(cnf) == 0:
            return tuple()
        if not isinstance(cnf[0], Iterable):
            return tuple(sorted(cnf, key=lambda l: abs(l)))
        return tuple(tuple(sorted(clause, key=lambda l: abs(l))) for clause in cnf)


    @staticmethod
    def extract_core(cnf, implicant):
        """
        A core is the CNF with only the literals of the implicant.
        """
        core = CNFencoding.literals_in_implicant(cnf, implicant)
        return CNFencoding.remove_subsumed(core)


    @staticmethod
    def apply_mapping_id_variables(cnf, mapping_id_variables):
        """
        Apply a mapping from new_mapping_id_variables(cnf)
        Warning: therefore remove the original signs of literals
        """
        return CNF([[mapping_id_variables[abs(lit)] for lit in clause] for clause in cnf])


    @staticmethod
    def to_prime_implicant_CNF(cnf, implicant):
        """
        Generate formula used to enumerate all prime implicants.
        Based on Saïd Jabbour, João Marques-Silva, Lakhdar Sais, Yakoub Salhi:
        Enumerating Prime Implicants of Propositional Formulae in Conjunctive Normal Form. JELIA 2014: 152-165
        """

        core = CNFencoding.extract_core(cnf, implicant)
        return PrimeImplicantCNF(core)
