
#pragma once

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace encore {
class FactoryException : public std::exception {
 private:
  std::string m_error_message;
  const char *m_file;
  int m_line;

 public:
  FactoryException(const char *msg, const char *file_, int line_)
      : m_file(file_), m_line(line_) {
    std::ostringstream o;
    o << m_file << ":" << m_line << ": " << msg;
    m_error_message = o.str();
  }  // constructor

  /**
     Returns a pointer to the (constant) error description.

     \return A pointer to a const char*.
   */
  virtual const char *what() const throw() { return m_error_message.c_str(); }
};
}  // namespace encore
