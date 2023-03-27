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

#include "ParserDimacs.h"

#include <algorithm>

#include "Problem.h"

namespace pyxai {

/**
 * @brief Read the next integer in the given stream while the value 0 is not
 * reached.
 *
 * @param in, the stream.
 * @param list, the list of integer we parsed.
 */
void ParserDimacs::readListIntTerminatedByZero(BufferRead &in,
                                               std::vector<int> &list) {
  int v = -1;
  do {
    v = in.nextInt();
    if (v) list.push_back(v);
  } while (v);
}  // readListIntTerminatedByZero

/**
 * @brief Parse a literal index and a weight and store the result in the given
 * vector.
 *
 * @param in, the stream buffer where we get the information.
 * @param weightLit, the place where is stored the data.
 */
void ParserDimacs::parseWeightedLit(BufferRead &in,
                                    std::vector<double> &weightLit) {
  int lit = in.nextInt();
  double w = in.nextDouble();

  if (lit > 0)
    weightLit[lit << 1] = w;
  else
    weightLit[((-lit) << 1) + 1] = w;
}  // parseWeightedLit

/**
 * @brief Parse the dimacs format in order to extract CNF formula and
 * different set of variables such projected variables, max variables, ...
 * but also variables' weights if it is the case that the problem is
 * weighted.
 *
 * @param in, the stream buffer where are read the information.
 * @param problemManager, the place where is store the result.
 * @return an integer that gives the problem's number of variables.
 */
int ParserDimacs::parse_DIMACS_main(BufferRead &in, Problem *problem) {
  std::vector<Lit> lits;
  std::string s;

  std::vector<std::vector<Lit>> &clauses = problem->getClauses();

  int nbVars = 0;
  int nbClauses = 0;

  for (;;) {
    in.skipSpace();

    if (in.eof()) break;

    if (in.currentChar() == 'p') {
      in.consumeChar();
      in.skipSpace();

      if (in.nextChar() != 'c' || in.nextChar() != 'n' || in.nextChar() != 'f')
        std::cerr << "PARSE ERROR! Unexpected char: " << in.currentChar()
                  << "\n",
            exit(3);

      nbVars = in.nextInt();
      nbClauses = in.nextInt();

      if (nbClauses < 0) printf("parse error\n"), exit(2);
    } else if (in.currentChar() == 'c') {
      std::string comment = "";
      while (in.currentChar() != '\n') {
        comment += in.currentChar();
        in.consumeChar();
      }
      comment += "\n";
      problem->getComments().push_back(comment);
    } else {
      lits.clear();
      int v = -1;
      do {
        v = in.nextInt();
        if ((v > 0 && nbVars < v) || (-v > 0 && nbVars < -v))
          std::cerr << "PARSE ERROR! Number of variables incorrect: " << v
                    << "\n",
              exit(3);

        if (v)
          lits.push_back((v > 0) ? Lit::makeLit(v, false)
                                 : Lit::makeLit(-v, true));
      } while (v);

      assert(lits.size());
      std::sort(lits.begin(), lits.end());

      // remove redundant literal and check for tautology.
      unsigned j = 1;
      bool isSat = false;
      for (unsigned i = 1; !isSat && i < lits.size(); i++) {
        if (lits[i] == lits[j - 1]) continue;
        isSat = lits[i] == ~lits[j - 1];
        lits[j++] = lits[i];
      }

      // add the clause only if not SAT.
      if (!isSat) {
        lits.resize(j);
        clauses.push_back(lits);
      }
    }
  }

  return nbVars;
}

int ParserDimacs::parse_DIMACS(std::string input_stream, Problem *problem) {
  BufferRead in(input_stream);
  return parse_DIMACS_main(in, problem);
}  // parse_DIMACS

}  // namespace rfx
