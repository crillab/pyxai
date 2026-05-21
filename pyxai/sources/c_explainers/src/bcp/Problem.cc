/**
 * rfx
 *  Copyright (C) 2021  Lagniez Jean-Marie
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Affero General Public License for more details.
 *
 *  You should have received a copy of the GNU Affero General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "Problem.h"

#include "ParserDimacs.h"
#include "Problem.h"

namespace pyxai {

Problem::Problem(const std::string &nameFile, std::ostream &out, bool verbose) {
  if (verbose) out << "c [rfx PROBLEM] Constructor from file.\n";
  ParserDimacs parser;
  if (verbose) out << "c [rfx PROBLEM] Call the parser ... " << std::flush;
  m_nbVar = parser.parse_DIMACS(nameFile, this);
  if (verbose) out << "done\n";
  if (verbose) displayStat(out, "c [rfx PARSER] ");
}  // constructor

Problem::Problem() { m_nbVar = 0; }  // constructor

Problem::Problem(Problem &problem, std::ostream &out, bool verbose) {
  if (verbose) out << "c [rfx PROBLEM] Constructor from problem.\n";
  m_nbVar = problem.getNbVar();
  m_clauses = problem.getClauses();
  if (verbose) displayStat(out, "c [PARSER] ");
}  // constructor

Problem::Problem(std::vector<std::vector<Lit>> &clauses, unsigned nbVar,
                 std::ostream &out, bool verbose) {
  if (verbose) out << "c [rfx PROBLEM] Constructor from clauses.\n";
  m_nbVar = nbVar;
  m_clauses = clauses;
  if (verbose) displayStat(out, "c [rfx PARSER] ");
}  // constructor

Problem *Problem::getUnsatProblem() {
  Problem *ret = new Problem();
  ret->setNbVar(m_nbVar);

  std::vector<Lit> cl;
  Lit l = Lit::makeLit(1, false);

  cl.push_back(l);
  ret->getClauses().push_back(cl);

  cl[0] = l.neg();
  ret->getClauses().push_back(cl);

  return ret;
}  // getUnsatProblem

void Problem::display(std::ostream &out) {
  out << "p cnf " << m_nbVar << " " << m_clauses.size() << "\n";

  // print the comments
  for (auto &comment : m_comments) out << comment;

  // print the clauses.
  for (auto cl : m_clauses) {
    for (auto &l : cl) out << l << " ";
    out << "0\n";
  }
}  // diplay

void Problem::displayStat(std::ostream &out, std::string startLine) {
  unsigned nbLits = 0;
  unsigned nbBin = 0;
  unsigned nbTer = 0;
  unsigned nbMoreThree = 0;

  for (auto &c : m_clauses) {
    nbLits += c.size();
    if (c.size() == 2) nbBin++;
    if (c.size() == 3) nbTer++;
    if (c.size() > 3) nbMoreThree++;
  }

  out << startLine << "Number of variables: " << m_nbVar << "\n";
  out << startLine << "Number of clauses: " << m_clauses.size() << "\n";
  out << startLine << "Number of binary clauses: " << nbBin << "\n";
  out << startLine << "Number of ternary clauses: " << nbTer << "\n";
  out << startLine << "Number of clauses larger than 3: " << nbMoreThree
      << "\n";
  out << startLine << "Number of literals: " << nbLits << "\n";
}  // displaystat

}  // namespace rfx
