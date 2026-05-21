

#pragma once

#include "3rdParty/glucose-3.0/core/Solver.hh"
#include "Cnf.hpp"
#include "core/SolverTypes.h"

namespace encore {
class GlucoseCnf : public Cnf {
 private:
  Glucose::Solver *m_satSolver = NULL;

 public:
  /**
   * @brief Construct a new Glucose Cnf object
   *
   */
  GlucoseCnf();

  /**
   * @brief Destroy the Glucose Cnf object
   *
   */
  ~GlucoseCnf();

  /**
   * @brief Get the Nb Clauses.
   *
   * @return the number of clauses.
   */
  unsigned getNbClauses() override;

  /**
   * @brief Add a new clause.
   *
   * @param cl is the clause we want to add.
   */
  void addClause(const std::vector<int> &cl) override;

  /**
   * @brief Increase the number of varaibles.
   *
   * @param nb is the number of variables we want to add.
   */
  void addNbVariable(unsigned nb) override;

  /**
   * @brief Increase the number of varaibles and store the result..
   *
   * @param nb is the number of variables we want to add.
   * @param vars are the added variables.
   */
  void addNbVariable(unsigned nb, std::vector<int> &vars) override;

  /**
   * @brief Display the formula following the clauses following the dimacs
   * format.
   *
   * @param out is the stream where are printed out the clauses.
   */
  void displayClauses(std::ostream &out) override;

  /**
   * @brief Get the Solver object
   *
   * @return the sat solver.
   */
  inline Glucose::Solver *getSolver() { return m_satSolver; }

  /**
   * @brief Solver the undelying problem.
   *
   * @return true if the problem is SAT, false otherwise. If the problem is set
   * it is possible to ask the model by calling the function 'valuation'.
   */
  bool solve();

  /**
   * @brief Solver the undelying problem under assumption.
   *
   * @param assums is the given assumption.
   *
   * @return true if the problem is SAT, false otherwise. If the problem is set
   * it is possible to ask the model by calling the function 'valuation'.
   */
  bool solve(std::vector<int> &assums);

  /**
   * @brief Ask for the core stored by the solver.
   *
   * @param[out] core is the computed core.
   */
  void getCore(std::vector<int> &core);

  /**
   * @brief Ask the valuation computed by the solver.
   *
   * @param var is the variable we are looking for its valuation.
   * @return true is the variable has been assigned to true, false otherwise.
   */
  inline bool valuation(unsigned var) {
    return m_satSolver->model[var] == l_True;
  }
};
}  // namespace encore