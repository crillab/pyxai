
#pragma once

#include <vector>

#include "src/problem/Cnf.hpp"
#include "src/problem/GlucoseCnf.hpp"

namespace encore {

#define EXIT_NOT_FOUND 1
#define EXIT_UNSAT 10
#define EXIT_CANDIDATE 20
#define EXIT_FOUND 30

class KCoverMethod {
 private:
  unsigned m_maxVar, m_nbPosAnchored, m_k;
  unsigned m_nbCandidateTested;
  std::vector<std::vector<int>> m_occurrence;

  /**
   * @brief Create a solver for managing the first level.
   *
   * @param ap gives the set of positive examples.
   * @param an gives the set of negative examples.
   * @param k is the number of positive examples we want to at least cover.
   * @param x is the given instance.
   * @param[out] solver is the filled solver.
   * @param[out] selectorX are the variable used to select the literal in the
   * term x we select.
   */
  void createExistSolver(Cnf &ap, Cnf &an, unsigned k, std::vector<int> &x,
                         GlucoseCnf &solver, std::vector<int> &selectorX);

  /**
   * @brief Create a solver for handling the second level.
   *
   * @param negFormula is the formula.
   * @param an are the set of negative examples.
   * @param[out] solver is the filled solver.
   */
  void createForAllSolver(Cnf &negFormula, Cnf &an, GlucoseCnf &solver);

  /**
   * @brief Search for a candidate abductive explanation.
   *
   * @param existSolver the first level solver.
   * @param forAllSolver the second level solver.
   * @param x is the instance we want to reduce.
   * @param selectorX are the selector.
   * @param result is the candidate abductive explanation (can be emptied when
   * no candidate has been found).
   *
   * @return true if a candidate has been found, false otherwise.
   */
  bool searchKCoverCandidate(GlucoseCnf &existSolver, GlucoseCnf &forAllSolver,
                             std::vector<int> &x, std::vector<int> &selectorX,
                             std::vector<int> &results);

  /**
   * @brief Dig for a minimal explanation regarding the second level solver and
   * the negative example we want to expel.
   *
   * @param forAllSolver is the solver.
   * @param an are the negative example.
   * @param candidate is the current candidate solution.
   * @param[out] result is the resulting explanation.
   */
  void computeMinimalAx(GlucoseCnf &forAllSolver, Cnf &an,
                        std::vector<int> &candidate, std::vector<int> &results);

  /**
   * @brief Try to expend the given candidate.
   * @warning The variable m_occurence should be initialized (see the method
   * run for more details about the initialization). Idem for m_nbPosAnchored
   *
   * @param[out] candidate is the candidate we want to extend.
   * @param[in] k is the threshold.
   * @param[in] x is the given example.
   *
   * \return the number of positive anchor covered.
   */
  unsigned expandCandidate(std::vector<int> &candidate, unsigned k,
                           std::vector<int> &x);

 public:
  /**
   * @brief Dig for an abductive explanation that cover at least k positive
   * examples (ap), and does not cover any negative examples (an).
   *
   * @param negFormula is the negation of the formula we search for an
   * implicant.
   * @param ap gives the set of positive examples.
   * @param an gives the set of negative examples.
   * @param k the number of positive exmaple we want to at least cover.
   * @param x is the given instance we search for an abductive explanation.
   * @param results is the computed abductive expanation.
   * @param minimize is set to true if we want to minimize the candidate k
   * anchored abductive explanation.
   *
   * @return the status code.
   */
  int run(Cnf &negFormula, Cnf &ap, Cnf &an, unsigned k, std::vector<int> &x,
          std::vector<int> &results, bool minimize);
};
}  // namespace encore