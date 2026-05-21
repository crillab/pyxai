

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "Cnf.hpp"
#include "src/utils/BufferRead.hpp"

namespace encore {
class ParserDimacs {
 private:
  int parse_DIMACS_main(BufferRead &in, Cnf &cnf);

 public:
  int parse_DIMACS(std::string input_stream, Cnf &cnf);
  int parseTerm(std::string input, std::vector<int> &term);
};
}  // namespace encore
