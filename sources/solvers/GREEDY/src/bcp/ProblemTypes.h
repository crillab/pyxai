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

#include <cassert>
#include <iostream>
#include <vector>

namespace pyxai {

typedef int Var;
typedef uint8_t lbool;

const Var var_Undef = -1;
const lbool l_True = 0;
const lbool l_False = 1;
const lbool l_Undef = 2;

struct Lit {
  int m_x;

  inline bool sign() const { return m_x & 1; }
  inline Var var() const { return m_x >> 1; }
  inline Lit neg() const { return {m_x ^ 1}; }
  inline unsigned intern() const { return m_x; }
  inline int human() const { return (m_x & 1) ? -var() : var(); }

  bool operator==(Lit p) const { return m_x == p.m_x; }
  bool operator!=(Lit p) const { return m_x != p.m_x; }
  bool operator<(Lit p) const {
    return m_x < p.m_x;
  }  // '<' makes p, ~p adjacent in the ordering.

  friend Lit operator~(Lit p);
  friend std::ostream &operator<<(std::ostream &os, Lit l);

  static inline Lit makeLit(Var v, bool sign) { return {(v << 1) + sign}; }
  static inline Lit makeLitFalse(Var v) { return {(v << 1) + 1}; }
  static inline Lit makeLitTrue(Var v) { return {v << 1}; }
};

const Lit lit_Undef = {-2};  // }- Useful special constants.
const Lit lit_Error = {-1};  // }

inline void showListLit(std::ostream &out, std::vector<Lit> &v) {
  for (auto &l : v) out << l << " ";
}  // showListLit

inline Lit operator~(Lit p) { return {p.m_x ^ 1}; }

struct Clause {
  unsigned size;
  Lit data[0];

  Lit &operator[](std::size_t idx) { return data[idx]; }

  inline void swap(unsigned i, unsigned j) {
    assert(i < size && j < size);
    Lit tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }

  inline void display(std::ostream &out) {
    for (unsigned i = 0; i < size; i++) out << data[i] << " ";
    std::cout << "\n";
  }
};

}  // namespace rfx
