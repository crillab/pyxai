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

#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "BufferRead.h"
#include "Problem.h"
#include "ProblemTypes.h"

namespace pyxai {
class ParserDimacs {
 private:
  int parse_DIMACS_main(BufferRead &in, Problem *problemManager);

  void readListIntTerminatedByZero(BufferRead &in, std::vector<int> &list);
  void parseWeightedLit(BufferRead &in, std::vector<double> &weightLit);

 public:
  int parse_DIMACS(std::string input_stream, Problem *problemManager);
};
}  // namespace rfx
