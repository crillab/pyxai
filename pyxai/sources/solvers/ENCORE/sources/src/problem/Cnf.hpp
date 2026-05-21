

#pragma once

#include <iostream>
#include <vector>

namespace encore {
class Cnf {
 protected:
  unsigned m_nbVariable = 0;

 public:
  virtual ~Cnf() {}

  /**
   * @brief Get the Nb Variable.
   *
   * @return the number of viariable.
   */
  inline unsigned getNbVariable() { return m_nbVariable; }

  /**
   * @brief Get the Nb Clauses.
   *
   * @return the number of clauses.
   */
  virtual unsigned getNbClauses() = 0;

  /**
   * @brief Add a new clause.
   *
   * @param cl is the clause we want to add.
   */
  virtual void addClause(const std::vector<int> &cl) = 0;

  /**
   * @brief Increase the number of varaibles.
   *
   * @param nb is the number of variables we want to add.
   */
  virtual void addNbVariable(unsigned nb) = 0;

  /**
   * @brief Increase the number of variables and store the result.
   *
   * @param nb is the number of variables we want to add.
   * @param vars are the added variables.
   */
  virtual void addNbVariable(unsigned nb, std::vector<int> &vars) = 0;

  /**
   * @brief Display the formula following the clauses following the dimacs
   * format.
   *
   * @param out is the stream where are printed out the clauses.
   */
  virtual void displayClauses(std::ostream &out) = 0;
};
}  // namespace encore