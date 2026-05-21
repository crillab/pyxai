

#include "ParserDimacs.hpp"

namespace encore {

/**
 * @brief Read the next integer in the given stream while the value 0 is not
 * reached.
 *
 * @param in, the stream.
 * @param list, the list of integer we parsed.
 */
void readListIntTerminatedByZero(BufferRead &in, std::vector<int> &list) {
  int v = -1;
  do {
    v = in.nextInt();
    if (v) list.push_back(v);
  } while (v);
}  // readListIntTerminatedByZero

/**
 * @brief Read a term on the stream given by an input file..
 *
 * @param input is the file where we get the term.
 * @param[out] term  is collected term.
 * @return 0 if a problem occurs.
 */
int ParserDimacs::parseTerm(std::string input, std::vector<int> &term) {
  BufferRead in(input);
  readListIntTerminatedByZero(in, term);
  return 1;
}  // parseTerm

/**
 * @brief ParserDimacs::parse_DIMACS_main implementation.
 */
int ParserDimacs::parse_DIMACS_main(BufferRead &in, Cnf &cnf) {
  std::string s;

  int nbVar = 0;
  [[maybe_unused]] int nbClause = 0;

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

      nbVar = in.nextInt();
      cnf.addNbVariable(nbVar);

      nbClause = in.nextInt();
    } else if (in.currentChar() == 'c') {
      in.skipLine();
    } else {
      std::vector<int> lits;
      readListIntTerminatedByZero(in, lits);
      cnf.addClause(lits);
    }
  }

  return nbVar;
}  // parse_DIMACS_main

int ParserDimacs::parse_DIMACS(std::string input_stream, Cnf &cnf) {
  BufferRead in(input_stream);
  return parse_DIMACS_main(in, cnf);
}  // parse_DIMACS

}  // namespace encore
