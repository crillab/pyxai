

#include "TotalizerCardinalityEncoder.hpp"

#include <iostream>

#include "3rdParty/cardenc/mto.hh"

namespace encore {
/**
 * @brief TotalizerCardinalityEncoder::getClauses implementation.
 */
std::vector<std::vector<int>> &TotalizerCardinalityEncoder::getClauses() {
  return m_clauses.get_clauses();
}  // getClauses

/**
 * @brief TotalizerCardinalityEncoder::encode implementation.
 */
void TotalizerCardinalityEncoder::encodeAtLeastK(int &indexLastVar,
                                                 std::vector<int> &lits,
                                                 int k) {
  to_encode_atleastN(indexLastVar, m_clauses, lits, k);
}  // encode

}  // namespace encore
