
#pragma once

#include <iostream>

#define BUFFER_SIZE 65536

namespace encore {
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

  inline double nextDouble() {
    double vint = 0;
    skipSpace();

    bool sign = currentChar() == '-';
    if (sign) consumeChar();
    while (!eof() && currentChar() >= '0' && currentChar() <= '9') {
      vint = vint * 10 + (nextChar() - '0');
    }

    double vdec = 0.0;
    int nbDigit = 0;
    if (currentChar() == '.') {
      consumeChar();
      while (!eof() && currentChar() >= '0' && currentChar() <= '9') {
        vdec = vdec * 10 + (nextChar() - '0');
        nbDigit++;
      }
    }

    while (nbDigit--) vdec /= 10;

    double ret = vint + vdec;
    return (sign) ? -ret : ret;
  }
};

}  // namespace encore
