
#include "CardinalityEncoder.hpp"

#include <cstring>

#include "SortedCardinalityEncoder.hpp"
#include "TotalizerCardinalityEncoder.hpp"

namespace encore {
CardinalityEncoder *CardinalityEncoder::makeCardinalityEncoder(
    CardValue cardValue, std::ostream &out) {
  if (cardValue == SORTED) return new SortedCardinalityEncoder();
  if (cardValue == TOTALIZER) return new TotalizerCardinalityEncoder();
  return nullptr;
}  // makeCardinalityEncoder

}  // namespace encore