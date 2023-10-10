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
#pragma once

#include <vector>

#include "./ProblemTypes.h"

namespace pyxai {
class Problem {
 private:
  std::vector<std::vector<Lit>> m_clauses;
  std::vector<std::string> m_comments;
  unsigned m_nbVar;

 public:
  /**
   * @brief Construct a new Problem object
   */
  Problem();

  /**
   * @brief Construct a new Problem object
   *
   * @param nameFile is the file where we parse the CNF.
   * @param out is the stream where the logs are printed out.
   * @param verbose is set to true if we want to print out information.
   */
  Problem(const std::string &nameFile, std::ostream &out, bool verbose = false);

  /**
   * @brief Construct a new Problem object
   *
   * @param clause is the set of CNF clauses.
   * @param nbVar is the number of CNF variables.
   * @param out is the stream where are printed out the information.
   * @param verbose is set to true if we want to print out information.
   */
  Problem(std::vector<std::vector<Lit>> &clauses, unsigned nbVar,
          std::ostream &out, bool verbose = false);

  /**
   * @brief Construct a new Problem object
   *
   * @param problem is the problem we want to copy.
   * @param out is the stream where the logs are printed out.
   * @param verbose is set to true if we want to print out information.
   */
  Problem(Problem &problem, std::ostream &out, bool verbose = false);

  inline unsigned getNbVar() { return m_nbVar; }
  inline std::vector<std::vector<Lit>> &getClauses() { return m_clauses; }
  inline std::vector<std::string> &getComments() { return m_comments; }
  inline void setNbVar(unsigned nbVar) { m_nbVar = nbVar; }
  inline void setClauses(std::vector<std::vector<Lit>> &clauses) {
    m_clauses = clauses;
  }

  /**
   * @brief Get the Unsat ProblemManager object.
   *
   * @return an unsatisfiable problem.
   */
  Problem *getUnsatProblem();

  /**
     Display the problem.

     @param[out] out, the stream where the messages are redirected.
   */
  void display(std::ostream &out);

  /**
   Print out some statistic about the problem. Each line will start with the
   string startLine given in parameter.

   @param[in] out, the stream where the messages are redirected.
   @param[in] startLine, each line will start with this string.
 */
  void displayStat(std::ostream &out, std::string startLine);
};
}  // namespace rfx
