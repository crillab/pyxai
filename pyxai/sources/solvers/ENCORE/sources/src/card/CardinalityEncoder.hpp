
#pragma once

#include <iostream>
#include <ostream>
#include <vector>

namespace encore {

enum CardValue { SORTED, TOTALIZER };

static CardValue getCardValue(std::string &s) {
  if (s == "sorted") return SORTED;
  if (s == "totalizer") return TOTALIZER;

  std::cerr << "c The cardinalty constraint selected is not known (" << s
            << ")\n";
  throw std::exception();
}  // getCardValue

static std::string fromCardValueToString(CardValue c) {
  if (c == SORTED) return "sorted";
  if (c == TOTALIZER) return "totalizer";

  std::cerr << "c The cardinalty constraint selected is not known (" << c
            << ")\n";
  throw std::exception();
}  // fromCardValueToString

class CardinalityEncoder {
 public:
  virtual ~CardinalityEncoder() {}

  /**
   * @brief Cardinality Constraint encoder.
   */
  static CardinalityEncoder *makeCardinalityEncoder(CardValue cardValue,
                                                    std::ostream &out);

  /**
   * @brief Ask for the set of clauses.
   */
  virtual std::vector<std::vector<int>> &getClauses() = 0;

  /**
   * @brief Encode the cardinality constraint as a CNF.
   *
   * @param[out] indexLastVar is the next free var (at the end this variable
   * still contains the last fresh variable).
   * @param lits is the set of literals.
   * @param k is the limit.
   */
  virtual void encodeAtLeastK(int &indexLastVar, std::vector<int> &lits,
                              int k) = 0;
};

}  // namespace encore