
#pragma once

#include "3rdParty/cardenc/card.hh"
#include "CardinalityEncoder.hpp"

namespace encore {
class SortedCardinalityEncoder : public CardinalityEncoder {
  ClauseSet m_clauses;
  std::vector<int> m_cardVar;

 public:
  SortedCardinalityEncoder() {
    std::cout << "c Cardinality constraint: Sorted\n";
  }

  /**
   * @brief Ask for the set of clauses.
   */
  std::vector<std::vector<int>> &getClauses();

  /**
   * @brief Encode the cardinality constraint as a CNF.
   *
   * @param[out] indexLastVar is the next free var (at the end this variable
   * still contains the last fresh variable).
   * @param lits is the set of literals.
   * @param k is the limit.
   */
  void encodeAtLeastK(int &indexLastVar, std::vector<int> &lits, int k);
};
}  // namespace encore