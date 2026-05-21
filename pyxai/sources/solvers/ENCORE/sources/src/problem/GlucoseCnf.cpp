
#include "GlucoseCnf.hpp"

namespace encore {

/**
 * @brief GlucoseCnf::GlucoseCnf implementation.
 *
 */
GlucoseCnf::GlucoseCnf() {
  m_satSolver = new Glucose::Solver();
  m_satSolver->verbosity = 0;
  m_satSolver->newVar();  // we do not want the variable 0
  m_nbVariable = 0;
  m_satSolver->setIncrementalMode();
}  // constructor

/**
 * @brief GlucoseCnf::~GlucoseCnf implementation.
 *
 */
GlucoseCnf::~GlucoseCnf() {
  if (m_satSolver) delete m_satSolver;
  m_satSolver = nullptr;
}  // destructor

/**
 * @brief GlucoseCnf::getNbClauses implementation.
 */
unsigned GlucoseCnf::getNbClauses() {
  return m_satSolver->clauses.size();
}  // getNbClauses

/**
 * @brief GlucoseCnf::addClause implementation.
 */
void GlucoseCnf::addClause(const std::vector<int> &cl) {
  Glucose::vec<Glucose::Lit> newCl;
  for (auto l : cl) newCl.push(Glucose::mkLit(abs(l), l < 0));
  m_satSolver->addClause_(newCl);
}  // addClause

/**
 * @brief GlucoseCnf::addNbVariable implementation.
 */
void GlucoseCnf::addNbVariable(unsigned nb) {
  m_nbVariable += nb;
  while (nb--) m_satSolver->newVar();
}  // addNbVariable

/**
 * @brief GlucoseCnf::addNbVariable implementation.
 */
void GlucoseCnf::addNbVariable(unsigned nb, std::vector<int> &vars) {
  m_nbVariable += nb;
  while (nb--) vars.push_back(m_satSolver->newVar());
}  // addNbVariable

/**
 * @brief GlucoseCnf::displayClauses implementation.
 */
void GlucoseCnf::displayClauses(std::ostream &out) {
  for (int i = 0; i < m_satSolver->clauses.size(); i++) {
    Glucose::Clause &cl = m_satSolver->ca[m_satSolver->clauses[i]];
    for (int j = 0; j < cl.size(); j++)
      out << (Glucose::sign(cl[j]) ? "-" : "") << Glucose::var(cl[j]) << ' ';
    out << "0\n";
  }

  for (int i = 0; i < m_satSolver->trail.size(); i++) {
    out << (Glucose::sign(m_satSolver->trail[i]) ? "-" : "")
        << Glucose::var(m_satSolver->trail[i]) << ' ';
    out << "0\n";
  }
}  // displayClauses

/**
 * @brief GlucoseCnf::solve implementation.
 */
bool GlucoseCnf::solve() {
  Glucose::vec<Glucose::Lit> assums;
  return m_satSolver->solve(assums);
}  // solver

/**
 * @brief GlucoseCnf::solve implementation.
 */
bool GlucoseCnf::solve(std::vector<int> &assums) {
  Glucose::vec<Glucose::Lit> gassums;
  for (auto &l : assums) gassums.push(Glucose::mkLit(abs(l), l < 0));
  return m_satSolver->solve(gassums);
}  // solve

/**
 * @brief GlucoseCnf::getCore implementation.
 */
void GlucoseCnf::getCore(std::vector<int> &core) {
  for (int i = 0; i < m_satSolver->conflict.size(); i++) {
    Glucose::Lit l = m_satSolver->conflict[i];
    if (Glucose::sign(l))
      core.push_back(-Glucose::var(l));
    else
      core.push_back(-Glucose::var(l));
  }
}

}  // namespace encore