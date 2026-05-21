

#include <cassert>
#include <iostream>
#include <vector>

#include "3rdParty/glucose-3.0/utils/Options.hh"
#include "methods/Checker.hpp"
#include "src/methods/KCoverMethod.hpp"
#include "src/problem/BasicCnf.hpp"
#include "src/problem/ParserDimacs.hpp"

Glucose::StringOption input("main", "i", "input file", "NULL");
Glucose::StringOption pa(
    "main", "p", "(required) Path to get the file with the positive anchors.",
    "NULL");
Glucose::StringOption pn(
    "main", "n", "(required) Path to get the file with the positive anchors.",
    "NULL");
Glucose::StringOption instance("main", "x", "(required) Path to the instance",
                               "NULL");
Glucose::IntOption kk("main", "k", "value of k", 1,
                      Glucose::IntRange(0, INT_MAX));
Glucose::StringOption checkerfile("main", "c", "Check the implicant instance",
                                  "NULL");
Glucose::BoolOption minimize(
    "main", "minimize",
    "True if we want to get a subset minimal subset explanation.", true);

/**
   The main function!
*/
int main(int argc, char **argv) {
  Glucose::parseOptions(argc, argv, true);

  std::string inputFile = ((const char *)input);
  std::string posAnchorFile = ((const char *)pa);
  std::string negAnchorFile = ((const char *)pn);
  std::string xFile = ((const char *)instance);
  int k = kk;

  printf("%s %d\n", ((const char *)input), k);
  std::cout << "c Options parsed: "
            << "input-file(" << inputFile << ") "
            << "positive-anchor-file(" << posAnchorFile << ") "
            << "negative-anchor-file(" << negAnchorFile << ") "
            << "k(" << k << ") "
            << "minimize(" << minimize << ')' << "\n";

  std::vector<int> x, ax;
  encore::BasicCnf formula, posAnchors, negAnchors;
  encore::ParserDimacs parser;
  parser.parse_DIMACS(inputFile, formula);
  parser.parse_DIMACS(posAnchorFile, posAnchors);
  parser.parse_DIMACS(negAnchorFile, negAnchors);
  parser.parseTerm(xFile, x);

  if (strcmp(checkerfile, "NULL")) {  // Check the solution
    std::cout << "checker\n";
    std::string check = ((const char *)checkerfile);
    std::vector<int> implicant;
    std::cout << check << std::endl;
    parser.parseTerm(check, implicant);
    encore::Checker checker;
    switch (checker.run(formula, posAnchors, negAnchors, k, x, implicant)) {
      case encore::CheckerStatus::OK:
        std::cerr << "OK" << std::endl;
        exit(1);
      case encore::CheckerStatus::COVER_NEG:
        std::cerr << "ERROR: Cover negatve anchors" << std::endl;
        exit(-10);
      case encore::CheckerStatus::LESS_THAN_K:
        std::cerr << "ERROR: Cover less than k positive anchors" << std::endl;
        exit(-20);
      case encore::CheckerStatus::NOT_INSTANCE:
        std::cerr << "ERROR: Not instance of F" << std::endl;
        exit(-30);
      case encore::CheckerStatus::NOT_IMPLICANT:
        std::cerr << "ERROR: Not implicant of F" << std::endl;
        exit(-40);
      case encore::CheckerStatus::UNCOVER_X:
        std::cerr << "ERROR: Uncover instance" << std::endl;
        exit(-50);
    }
  }
  encore::KCoverMethod method;
  return method.run(formula, posAnchors, negAnchors, k, x, ax, minimize);
}  // main
