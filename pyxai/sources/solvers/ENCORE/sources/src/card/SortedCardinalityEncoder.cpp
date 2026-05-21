

#include "SortedCardinalityEncoder.hpp"

#include <iostream>

namespace encore {
/**
 * @brief SortedCardinalityEncoder::getClauses implementation.
 */
std::vector<std::vector<int>> &SortedCardinalityEncoder::getClauses() {
  return m_clauses.get_clauses();
}  // getClauses

/**
 * @brief SortedCardinalityEncoder::encode implementation.
 */
void SortedCardinalityEncoder::encodeAtLeastK(int &indexLastVar,
                                              std::vector<int> &lits, int k) {
  // consider the negation.
  std::vector<int> negLit;
  int nrhs = -k;

  for (size_t i = 0; i < lits.size(); i++) {
    nrhs += 1;  // The weight in fact
    negLit.push_back(-lits[i]);
  }

  cardn_encode_atmostN(indexLastVar, m_clauses, negLit, nrhs);
}  // encode

}  // namespace encore
