
#include "Checker.hpp"

#include <algorithm>
#include <iostream>

#include "src/card/CardinalityEncoder.hpp"
#include "src/problem/BasicCnf.hpp"

namespace encore
{
 bool Checker::cover(unsigned int nbVariables, const std::vector<int>& x, const std::vector<int>& implicant)
 {
  std::vector<int> instance;
  instance.assign(nbVariables, 0); // By default not present
  for(const int lit: x)
   instance[abs(lit)] = lit;

  // Check if implicant covers x
  for(int const lit: implicant)
   if(instance[abs(lit)] != lit)
    return false; // Aie

  return true;
 }

 int Checker::nbCovered(unsigned int nbVariables, Cnf &positiveAnchors, const std::vector<int>& implicant) {
     int nb = 0;
     for(auto &positive : static_cast<BasicCnf &>(positiveAnchors).getClauses())
         if(Checker::cover(nbVariables, positive, implicant)) // Aie
             nb++;
     return nb;
 }

    CheckerStatus Checker::run(Cnf &negFormula, Cnf &positiveAnchors, Cnf &negativeAnchors, unsigned k,
                       std::vector<int> &x, std::vector<int> &implicant) {

  // create the solver.
  GlucoseCnf solver;
  unsigned int nbVariables = negFormula.getNbVariable();
  solver.addNbVariable(nbVariables);
  for (auto &cl : static_cast<BasicCnf &>(negFormula).getClauses())
   solver.addClause(cl);

  // Check if the term is an implicant:
  if(solver.solve(x)) // Aie...
   return CheckerStatus::NOT_INSTANCE;

  if(solver.solve(implicant)) // Aie...
   return CheckerStatus::NOT_IMPLICANT;


  // Check if implicant covers x
  if(Checker::cover(nbVariables, x, implicant) == false)
   return CheckerStatus::UNCOVER_X;


  // Check if implicant does not cover a negative instance
  for(auto &negative : static_cast<BasicCnf &>(negativeAnchors).getClauses())
   if(Checker::cover(nbVariables, negative, implicant)) // Aie
    return CheckerStatus::COVER_NEG;

  // Check then umber of positive anchored covered
   if(Checker::nbCovered(nbVariables, positiveAnchors, implicant) < k) //  Aie
   return CheckerStatus::LESS_THAN_K;

  return CheckerStatus::OK;
 }  // run

}