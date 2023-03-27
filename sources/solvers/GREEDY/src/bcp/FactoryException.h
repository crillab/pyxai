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

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace pyxai {
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
}  // namespace rfx
