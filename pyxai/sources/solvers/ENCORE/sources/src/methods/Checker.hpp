

#pragma once

#include <vector>

#include "src/problem/Cnf.hpp"

#include "src/problem/GlucoseCnf.hpp"

namespace encore
{
    enum class CheckerStatus { OK, NOT_INSTANCE, NOT_IMPLICANT, UNCOVER_X, LESS_THAN_K, COVER_NEG };
    class Checker
    {

    public:

        static bool cover(unsigned int nbVariables, const std::vector<int>& x, const std::vector<int>& implicant);
        static int nbCovered(unsigned int nbVariables, Cnf &positiveAnchors, const std::vector<int>& implicant);

        /**
         * @brief check if impicant  is an abductive explanation that cover at least k positive
         * examples (ap), and does not cover any negative examples (an).
         *
         * @param negFormula is the negation of the formula we search for an
         * implicant.
         * @param ap gives the set of positive examples.
         * @param an gives the set of negative examples.
         * @param k the number of positive exmaple we want to at least cover.
         * @param x is the given instance we search for an abductive explanation.
         * @param implicant. the implicant to be checked.
         *
         * @return true the implicant satisfies all constraints
         * false otherwise.
         */
        static CheckerStatus run(Cnf &negFormula, Cnf &ap, Cnf &an, unsigned k, std::vector<int> &x, std::vector<int> &implicant);
    };
}