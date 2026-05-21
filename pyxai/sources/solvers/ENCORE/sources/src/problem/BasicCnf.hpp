
#include "Cnf.hpp"

namespace encore {
class BasicCnf : public Cnf {
 private:
  std::vector<std::vector<int>> clauses;

 public:
  ~BasicCnf() override {}

  std::vector<std::vector<int>> &getClauses() { return clauses; }  // getClauses

  /**
   * @brief Get the Nb Clauses.
   *
   * @return the number of clauses.
   */
  inline unsigned getNbClauses() override { return clauses.size(); }

  /**
   * @brief Add a new clause.
   *
   * @param cl is the clause we want to add.
   */
  inline void addClause(const std::vector<int> &cl) override {
    clauses.push_back(cl);
  }

  /**
   * @brief Increase the number of varaibles.
   *
   * @param nb is the number of variables we want to add.
   */
  inline void addNbVariable(unsigned nb) override { m_nbVariable += nb; }

  /**
   * @brief Increase the number of variables and store the result.
   *
   * @param nb is the number of variables we want to add.
   * @param vars are the added variables.
   */
  inline void addNbVariable(unsigned nb, std::vector<int> &vars) override {
    while (nb) {
      m_nbVariable++;
      vars.push_back(m_nbVariable);
    }
  }  // addNbVariable

  /**
   * @brief Display the formula following the clauses following the dimacs
   * format.
   *
   * @param out is the stream where are printed out the clauses.
   */
  inline void displayClauses(std::ostream &out) override {
    for (auto &cl : clauses) {
      for (auto &l : cl) out << l << " ";
      out << '\n';
    }
  }  // displayClauses
};
}  // namespace encore