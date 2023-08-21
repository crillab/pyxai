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

#include <iostream>
#include <string>

#define BUFFER_SIZE 65536

namespace pyxai {
class BufferRead {
  int pos;
  int size;
  char buffer[BUFFER_SIZE];
  FILE *f;

 public:
  BufferRead(std::string &name) {
    pos = 0;
    size = 0;

    f = fopen(name.c_str(), "r");
    if (!f)
      std::cerr << "ERROR! Could not open file: " << name << "\n", exit(1);

    // fill the buffer
    size = fread(buffer, sizeof(char), BUFFER_SIZE, f);
    if (!size && ferror(f))
      std::cerr << "Cannot read the file: " << name << "\n", exit(1);
  }

  ~BufferRead() {
    if (f) fclose(f);
  }

  inline char currentChar() { return buffer[pos]; }
  inline char nextChar() {
    char c = buffer[pos];
    consumeChar();
    return c;
  }

  inline void consumeChar() {
    pos++;
    if (pos >= size) {
      pos = 0;
      size = fread(buffer, sizeof(char), BUFFER_SIZE, f);
      if (!size && ferror(f))
        std::cerr << "Cannot read the reamaining\n", exit(1);
    }
  }

  inline bool eof() { return !size && feof(f); }
  inline void skipSpace() {
    while (!eof() && (currentChar() == ' ' || currentChar() == '\t' ||
                      currentChar() == '\n' || currentChar() == '\r'))
      consumeChar();
  }

  inline void skipSimpleSpace() {
    while (!eof() && (currentChar() == ' ' || currentChar() == '\t'))
      consumeChar();
  }

  inline void skipLine() {
    while (!eof() && currentChar() != '\n') consumeChar();
    consumeChar();
  }

  inline int nextInt() {
    int ret = 0;
    skipSpace();

    bool sign = currentChar() == '-';
    if (sign) consumeChar();
    while (!eof() && currentChar() >= '0' && currentChar() <= '9') {
      ret = ret * 10 + (nextChar() - '0');
    }
    return (sign) ? -ret : ret;
  }

  /**
   * @brief Check out if the given modif can be consumed.
   *
   * @param motif, the string we want to consume.
   * @return true if the modif can be consumed, false otherwise.
   */
  inline bool canConsume(std::string motif) {
    skipSimpleSpace();
    for (auto c : motif) {
      if (currentChar() != c)
        return false;
      else
        consumeChar();
    }
    return true;
  }  // canConsume

  inline double nextDouble() {
    skipSpace();

    bool sign = currentChar() == '-';
    if (sign) consumeChar();

    std::string cur = "";
    while (!eof() && ((currentChar() >= '0' && currentChar() <= '9') ||
                      currentChar() == '.' || currentChar() == 'e' ||
                      currentChar() == '-')) {
      cur += currentChar();
      nextChar();
    }

    std::string::size_type pos = 0;
    double ret = 0;
    ret = std::stod(cur, &pos);

    return (sign) ? -ret : ret;
  }
};
}  // namespace rfx
