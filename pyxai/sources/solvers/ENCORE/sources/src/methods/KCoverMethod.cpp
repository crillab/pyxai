
#include "KCoverMethod.hpp"

#include <algorithm>
#include <csignal>
#include <iostream>
#include <random>

#include "Checker.hpp"
#include "src/card/CardinalityEncoder.hpp"
#include "src/problem/BasicCnf.hpp"

namespace encore {

typedef enum { NOT_FOUND, FOUND, UNSAT, CANDIDATE } Status;

struct Solution {
  Status status = NOT_FOUND;
  std::vector<int> solution;
};

Solution solutionFound;

void signalHandler(int signum) {
  std::cout << "c Interrupt signal (" << signum << ") received.\n";
  if (solutionFound.status == NOT_FOUND) {
    std::cout << "c No candidate has been found\n";
    exit(EXIT_NOT_FOUND);
  }

  if (solutionFound.status == UNSAT) {
    std::cout << "s UNSAT\n";
    exit(EXIT_UNSAT);
  }

  if (solutionFound.status == FOUND || solutionFound.status == CANDIDATE) {
    std::cout << "s "
              << ((solutionFound.status == FOUND) ? "MINIMAL" : "CANDIDATE")
              << '\n';
    std::cout << "v ";
    for (auto &l : solutionFound.solution) std::cout << l << " ";
    std::cout << '\n';
    exit((solutionFound.status == FOUND) ? EXIT_FOUND : EXIT_CANDIDATE);
  }

  assert(0);
  exit(signum);
}  // signalHandel

/**
 * @brief KCoverMethod::createExistSolver implementation.
 */
void KCoverMethod::createExistSolver(Cnf &ap, Cnf &an, unsigned k,
                                     std::vector<int> &x, GlucoseCnf &solver,
                                     std::vector<int> &selectorX) {
  std::vector<int> alignX, mapX;
  int cpt = 0;
  for (auto l : x) {
    while (mapX.size() <= abs(l)) mapX.push_back(0);
    while (alignX.size() <= abs(l)) alignX.push_back(0);
    alignX[abs(l)] = l;
    mapX[abs(l)] = cpt++;
  }

  solver.addNbVariable(x.size(), selectorX);

  // consider the positive example.
  std::vector<int> selectorAp;
  for (auto &t : static_cast<BasicCnf &>(ap).getClauses()) {
    solver.addNbVariable(1, selectorAp);
    for (auto l : t)
      if (l == -alignX[abs(l)]) {
        solver.addClause({-selectorX[mapX[abs(l)]], -selectorAp.back()});
        // std::cout << -selectorX[mapX[abs(l)]] << " " << -selectorAp.back() <<
        // std::endl;
      }
  }

  // consider the negative example.
  std::vector<int> selectorAn;
  for (auto &t : static_cast<BasicCnf &>(an).getClauses()) {
    std::vector<int> cl;
    for (auto l : t)
      if (l == -alignX[abs(l)]) cl.push_back(selectorX[mapX[abs(l)]]);

    solver.addClause(cl);
  }

  // add the cardinality constraint.
  CardinalityEncoder *cardEncoder =
      CardinalityEncoder::makeCardinalityEncoder(TOTALIZER, std::cout);
  int nbVar = solver.getNbVariable();
  cardEncoder->encodeAtLeastK(nbVar, selectorAp, k);
  solver.addNbVariable(nbVar - solver.getNbVariable());
  for (auto &cl : cardEncoder->getClauses()) {
    solver.addClause(cl);
  }
  delete cardEncoder;

}  // createExistSolver

/**
 * @brief KCoverMethod::createForAllSolver implementation.
 */
void KCoverMethod::createForAllSolver(Cnf &negFormula, Cnf &an,
                                      GlucoseCnf &solver) {
  solver.addNbVariable(negFormula.getNbVariable());
  for (auto &cl : static_cast<BasicCnf &>(negFormula).getClauses())
    solver.addClause(cl);
}  // createForAllSolver

/**
 * @brief KCoverMethod::expendCandidate implementation.
 */
unsigned KCoverMethod::expandCandidate(std::vector<int> &candidate, unsigned k,
                                       std::vector<int> &x) {
  // get the element we can add.
  std::vector<bool> marked(m_maxVar + 1, false);
  std::vector<int> toAdd;

  for (auto &l : candidate) marked[abs(l)] = true;
  for (auto &l : x)
    if (!marked[abs(l)]) toAdd.push_back(l);
  auto rng = std::default_random_engine{};
  srand(unsigned(time(NULL)));
  std::shuffle(toAdd.begin(), toAdd.end(), rng);

  // init the counter of falsified
  unsigned counter = 0;
  std::vector<bool> isFalsified(m_nbPosAnchored, false);

  for (auto &l : candidate) {
    for (auto &idx : m_occurrence[abs(l)])
      if (!isFalsified[idx]) {
        isFalsified[idx] = true;
        counter++;
      }
  }

  assert(m_nbPosAnchored - counter >= k);

  // try to expand.
  for (auto &l : toAdd) {
    unsigned countNewFalse = 0;
    for (auto &idx : m_occurrence[abs(l)])
      if (!isFalsified[idx]) countNewFalse++;

    if (m_nbPosAnchored - (counter + countNewFalse) >= k) {
      for (auto &idx : m_occurrence[abs(l)]) isFalsified[idx] = true;
      counter += countNewFalse;
      candidate.push_back(l);
    }
  }

  assert(m_nbPosAnchored >= counter);
  return m_nbPosAnchored - counter;
}  // expandCandidate

/**
 * @brief KCoverMethod::searchKCoverCandidate implementation.
 */
bool KCoverMethod::searchKCoverCandidate(GlucoseCnf &existSolver,
                                         GlucoseCnf &forAllSolver,
                                         std::vector<int> &x,
                                         std::vector<int> &selectorX,
                                         std::vector<int> &results) {
  assert(x.size() == selectorX.size());
  while (!results.size()) {
    if (!existSolver.solve()) break;
    /*for (unsigned i = 0; i < x.size(); i++){
      std::cout << existSolver.valuation(selectorX[i]) << " ";
    }
    std::cout << std::endl;*/

    // get the candidate term.
    std::vector<int> term, blockIdx;
    for (unsigned i = 0; i < x.size(); i++)
      if (existSolver.valuation(selectorX[i]))
        term.push_back(x[i]);
      else
        blockIdx.push_back(i);

#if 0
    std::cout << "c term: ";
    for (auto &l : term) std::cout << l << ' ';
    std::cout << '\n';
#endif
    m_nbCandidateTested++;
    if (!(m_nbCandidateTested & 127))
      std::cout << "c Number of candidates tested: " << m_nbCandidateTested
                << '\n';
    unsigned newk = expandCandidate(term, m_k, x);

    // test the term on the second level.
    if (!forAllSolver.solve(term)) {
      results = term;
      std::cout << "c The candidate term is anshored at level: " << newk
                << '\n';
      return true;
    } else {
      std::vector<int> block;
      for (auto &idx : blockIdx) {
        if ((forAllSolver.valuation(selectorX[idx]) && x[idx] > 0) ||
            (!forAllSolver.valuation(selectorX[idx]) && x[idx] < 0)) {
          continue;
        }

        block.push_back(selectorX[idx]);
      }
      existSolver.addClause(block);
    }
  }

  return false;
}  // searchKCoverCandidate

/**
 * @brief KCoverMethod::computeMinimalAx implementation.
 */
void KCoverMethod::computeMinimalAx(GlucoseCnf &forAllSolver, Cnf &an,
                                    std::vector<int> &candidate,
                                    std::vector<int> &results) {
  std::vector<u_int8_t> marked(forAllSolver.getNbVariable() + 1, 0);
  for (auto &l : candidate) marked[abs(l)] = (l > 0) ? 1 : 2;

  // construct the occurrence list.
  std::vector<int> counter(an.getNbClauses(), 0);
  std::vector<std::vector<int>> occurrence(forAllSolver.getNbVariable() + 1);

  unsigned cpt = 0;
  for (auto &t : static_cast<BasicCnf &>(an).getClauses()) {
    for (auto &l : t) {
      if (!marked[abs(l)] || (l > 0 && marked[abs(l)] == 1) ||
          (l < 0 && marked[abs(l)] == 2))
        continue;
      counter[cpt]++;
      occurrence[abs(l)].push_back(cpt);
    }

    cpt++;
  }

// cosmectic check.
#if DEBUG
  for (auto &c : counter) assert(c > 0);
#endif

  // check for the literals they are critical.
  std::vector<bool> critical(forAllSolver.getNbVariable() + 1, false);

  for (auto &l : candidate) {
    bool isCritical = false;
    for (auto &idx : occurrence[abs(l)])
      if (isCritical = counter[idx] == 1) break;

    critical[abs(l)] = isCritical;
  }

  unsigned nbSatCall = 0, nbUnsatCall = 0;

  // destructive approach.
  while (candidate.size()) {
    int l = candidate.back();
    candidate.pop_back();
    if (!marked[abs(l)]) continue;

    if (critical[abs(l)] || forAllSolver.solve(candidate)) {
      if (!critical[abs(l)]) nbSatCall++;
      results.push_back(l);
      forAllSolver.addClause({l});

      // copy the current solution.
      solutionFound.solution = results;
      for (auto &l : candidate) solutionFound.solution.push_back(l);
    } else {
      nbUnsatCall++;

      std::vector<int> core;
      forAllSolver.getCore(core);
      std::vector<bool> markCore(m_maxVar + 1, false);
      for (auto &m : core) markCore[abs(m)] = true;

      unsigned i = 0;
      do {
        marked[abs(l)] = 0;
        for (auto &idx : occurrence[abs(l)]) {
          counter[idx]--;
          assert(counter[idx] > 0);

          // one literal becomes critical
          if (counter[idx] == 1) {
            for (auto &l : static_cast<BasicCnf &>(an).getClauses()[idx]) {
              if (!marked[abs(l)] || (l > 0 && marked[abs(l)] == 1) ||
                  (l < 0 && marked[abs(l)] == 2))
                continue;

              critical[abs(l)] = true;
              break;
            }
          }
        }
        break; // uncomment for removing core.
        while (i < candidate.size() &&
               (critical[abs(candidate[i])] || markCore[abs(candidate[i])]))
          i++;
        if (i < candidate.size()) l = candidate[i];
        i++;
      } while (i <= candidate.size());
    }
  }

  std::cout << "c Minimisation SAT #calls:" << nbSatCall << '\n';
  std::cout << "c Minimisation UNSAT #calls:" << nbUnsatCall << '\n';
}  // computeMinialAx

/**
 * @brief KCoverMethod::run implementation.
 */
int KCoverMethod::run(Cnf &negFormula, Cnf &ap, Cnf &an, unsigned k,
                      std::vector<int> &x, std::vector<int> &results,
                      bool minimize) {
  std::cout << "c Statistics: \n";
  std::cout << "c Number of variables in the formula: "
            << negFormula.getNbVariable() << '\n'
            << "c Number of clauses in the formula: "
            << negFormula.getNbClauses() << '\n'
            << "c Number of positive examples: " << ap.getNbClauses() << '\n'
            << "c Number of negative examples: " << an.getNbClauses() << '\n';
  std::cout << "c x = ";
  m_maxVar = 0;
  for (auto l : x) {
    if (m_maxVar < abs(l)) m_maxVar = abs(l);
    std::cout << l << ' ';
  }
  std::cout << '\n';

  std::cout << "c Change the signal handler.\n";
  signal(SIGTERM, signalHandler);

  if (k > ap.getNbClauses()) {
    std::cout << "s UNSATISFIABLE\n";
    return EXIT_UNSAT;
  }

  // init m_occurrence regarding the positive anchored and the instance.
  m_occurrence.clear();
  m_occurrence.resize(m_maxVar + 1);

  std::vector<bool> value(m_maxVar + 1, 0);
  for (auto &l : x) value[abs(l)] = l > 0;
  std::vector<std::vector<int>> &posA =
      static_cast<BasicCnf &>(ap).getClauses();
  for (unsigned i = 0; i < posA.size(); i++) {
    for (auto &l : posA[i]) {
      if (value[abs(l)] != ((bool)(l > 0))) m_occurrence[abs(l)].push_back(i);
    }
  }
  m_nbPosAnchored = posA.size();
  m_k = k;
  m_nbCandidateTested = 0;

  // create the exist solver.
  GlucoseCnf existSolver;
  std::vector<int> selectorX;
  createExistSolver(ap, an, k, x, existSolver, selectorX);

  // create forall solver.
  GlucoseCnf forAllSolver;
  createForAllSolver(negFormula, an, forAllSolver);

  // CEGAR loop:
  std::vector<int> candidate;
  if (!searchKCoverCandidate(existSolver, forAllSolver, x, selectorX,
                             candidate)) {
    std::cout << "s UNSATISFIABLE\n";
    return EXIT_UNSAT;
  }

  solutionFound.status = CANDIDATE;
  solutionFound.solution = candidate;

  // print out the candidate.
  std::cout << "s SATISFIABLE\n";
  std::cout << "c Candidate size: " << candidate.size() << '\n';
  std::cout << "c Candidate: ";
  for (auto &l : candidate) std::cout << l << ' ';
  std::cout << '\n';
  std::cout << "o "
            << Checker::nbCovered(negFormula.getNbVariable(), ap, candidate)
            << std::endl;

  // make the explanation minimal subset minimal.
  if (minimize)
    computeMinimalAx(forAllSolver, an, candidate, results);
  else
    return EXIT_FOUND;

  // print out the solution.
  std::cout << "c Final number of tested candidate: " << m_nbCandidateTested
            << '\n';
  std::cout << "v ";
  std::sort(results.begin(), results.end(),
            [](int i, int j) { return abs(i) < abs(j); });
  for (auto &l : results) std::cout << l << ' ';
  std::cout << "0\n";
  std::cout << "o "
            << Checker::nbCovered(negFormula.getNbVariable(), ap, results)
            << std::endl;

  return EXIT_FOUND;
}  // run
}  // namespace encore