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

#include "Propagator.h"

#include <cassert>

namespace pyxai {

    // This propagator does nothing
    Propagator::Propagator() : m_out(std::cout), m_verbose(false) {
        m_nbVar = 0;
    }
/**
 * @brief constructor implementation.
 */
    Propagator::Propagator(Problem &p, std::ostream &out, bool verbose)
            : m_out(out), m_verbose(verbose) {
        if (m_verbose) m_out << "c [rfx PROPAGATOR] Construtor\n";

        // First: allocate the memory.
        // compute for each literal the possible number of binary clause.
        unsigned nbLitBin = 0, nbLitNotBin = 0, nbClauseBin = 0, nbClauseNotBin = 0;

        m_posClauseNotBin = 0;
        m_nbVar = p.getNbVar();
        std::vector<unsigned> counterBin((1 + m_nbVar) << 1, 0);
        std::vector<unsigned> counterNotBin((1 + m_nbVar) << 1, 0);

        m_isUnsat = false;
        m_trailLimUnit = 0;
        m_trailSize = 0;
        m_trailPos = 0;
        m_trail = new Lit[(m_nbVar + 1) << 1];
        m_assign = new lbool[(m_nbVar + 1) << 1];
        for (unsigned i = 0; i < (m_nbVar + 1) << 1; i++) m_assign[i] = l_Undef;

        for (auto &cl: p.getClauses()) {
            //for(Lit l : cl)
            //    std::cout << l << " " ;
            //std::cout << std::endl;
            if (cl.size() == 1)
                continue;
            else if (cl.size() == 2) {
                nbClauseBin++;
                nbLitBin += 2;
                for (auto &l: cl) counterBin[l.intern()]++;
            } else {
                nbLitNotBin += cl.size();
                nbClauseNotBin++;
                for (auto &l: cl) counterNotBin[l.intern()]++;
            }
        }
        // compute the needed memory in bytes.
        unsigned memoryNeeded =
                (nbClauseNotBin * sizeof(Clause)) +
                (nbLitNotBin * sizeof(Lit))  // clauses
                + ((1 + m_nbVar) << 1) * sizeof(Imply) +
                ((nbLitNotBin + nbLitBin) * sizeof(Lit))  // binary clauses
                + ((1 + m_nbVar) << 1) * sizeof(Watch) +
                nbLitNotBin * sizeof(CRef);  // watch list

        if (m_verbose)
            m_out << "c [rfx PROPAGATOR] Memory needed: " << memoryNeeded << "\n"
                  << "c [rfx PROPAGATOR] Binary clauses: " << nbClauseBin << "\n"
                  << "c [rfx PROPAGATOR] Not binary clauses: " << nbClauseNotBin << "\n"
                  << "c [rfx PROPAGATOR] Number of literals in not binary clauses: "
                  << nbLitNotBin << "\n";

        // reserve the data memory
        m_data = new uint8_t[memoryNeeded];

        // init the vectors regarding their size.
        m_notBinClauseRefs.reserve(nbClauseNotBin);
        m_binListRefs.resize((1 + m_nbVar) << 1, NULL);
        m_watchList.resize((1 + m_nbVar) << 1, NULL);

        // point the binary start for each literal.
        uint8_t *ptr =
                &m_data[nbClauseNotBin * sizeof(Clause) + nbLitNotBin * sizeof(Lit)];

        for (unsigned i = 0; i < counterBin.size(); i++) {
            assert((i ^ 1) < m_binListRefs.size());
            m_binListRefs[i ^ 1] = (Imply *) ptr;
            m_binListRefs[i ^ 1]->size = 0;
            ptr += sizeof(Imply) + (counterNotBin[i] + counterBin[i]) * sizeof(Lit);
        }

        // prepare the watch list.
        for (unsigned i = 0; i < counterNotBin.size(); i++) {
            assert((i ^ 1) < m_watchList.size());
            m_watchList[i] = (Watch *) ptr;
            m_watchList[i]->size = 0;
            ptr += sizeof(Watch) + counterNotBin[i] * sizeof(CRef);
        }
        assert(ptr <= &m_data[memoryNeeded]);

        // store the clauses.
        for (auto &cl: p.getClauses()) {
            addClause(cl);
            if (m_isUnsat) break;
        }

        if (!m_isUnsat) m_trailLimUnit = m_trailSize;
    }  // constructor

/**
 * @brief destructor implementation.
 */
    Propagator::~Propagator() {
        if(m_nbVar == 0)
            return;
        delete[] m_data;
        delete[] m_trail;
        delete[] m_assign;
    }  // destructor.

/**
 * @brief addBinary implementation.
 */
    void Propagator::addBinary(Lit a, Lit b) {
        Imply *imply = m_binListRefs[(~a).intern()];
        imply->lits[imply->size++] = b;

        imply = m_binListRefs[(~b).intern()];
        imply->lits[imply->size++] = a;
    }  // addBinary

/**
 * @brief addClause implementation.
 */
    void Propagator::addClause(std::vector<Lit> &clause) {
        // no need to add the clause if the problem is unsat.
        if (m_isUnsat) return;

        // reduce clause by unit:
        std::vector<Lit> clauseTmp = clause;

        unsigned i, j;
        bool isSat = false;
        for (i = j = 0; i < clauseTmp.size(); i++) {
            if (value(clauseTmp[i]) >= l_Undef)
                clauseTmp[j++] = clauseTmp[i];
            else if (value(clauseTmp[i]) == l_True) {  // SAT
                j = 0;
                isSat = true;
                break;
            }
        }
        clauseTmp.resize(j);

        if (clauseTmp.size() == 0) {
            m_isUnsat = !isSat;
            return;
        } else if (clauseTmp.size() == 1) {
            if (m_assign[clauseTmp[0].var()] == l_Undef) {
                uncheckedEnqueue(clauseTmp[0]);
                m_isUnsat = !propagate();
            } else
                m_isUnsat = value(clauseTmp[0]) == l_False;
        } else if (clauseTmp.size() == 2) {
            addBinary(clauseTmp[0], clauseTmp[1]);
        } else {
            // add the clause
            CRef cref = m_posClauseNotBin;
            Clause *cl = (Clause *) &m_data[cref];
            m_notBinClauseRefs.push_back(cref);
            cl->size = clauseTmp.size();
            i = 0;
            for (auto &l: clauseTmp) cl->data[i++] = l;

            // link the clause to the watch list.
            attachClause(cref);

            // move to next clause.
            m_posClauseNotBin += sizeof(Clause) + sizeof(Lit) * clauseTmp.size();
        }
    }  // addClause

/**
 * @brief uncheckedEnqueue implementation.
 */
    void Propagator::uncheckedEnqueue(Lit l) {
        if(m_nbVar == 0 || l.var() > ((int)getNbVar()))
            return;
        //if(m_verbose) m_out << "propagate" << l << "\n";
        //m_out << "propagate" << l << "\n";
        if(m_assign[l.var()] == l_True ||m_assign[l.var()] == l_False)
            throw std::runtime_error("An error occurs in uncheckenqueue");
        //assert(m_assign[l.var()] == l_Undef);
        m_trail[m_trailSize++] = l;
        m_assign[l.var()] = l.sign();
    }  // uncheckedUnqueue

/**
 * @brief propagate implementation.
 */
    bool Propagator::propagate() {
        if(m_nbVar == 0)
            return true;
        while (m_trailPos < m_trailSize) {
            Lit l = m_trail[m_trailPos++];
            //std::cout << m_trailPos << " => propagate " << l << std::endl;
            // propagate the binary clauses.
            Imply &imply = *m_binListRefs[l.intern()];

            for (unsigned i = 0; i < imply.size; i++)
                if (value(imply[i]) >= l_Undef)
                    uncheckedEnqueue(imply[i]);
                else if (value(imply[i]) == l_False) {
                    return false;
                }

            // propagate the non binary clauses.
            Watch *ws = m_watchList[(~l).intern()];
            unsigned j = 0;
            for (unsigned i = 0; i < ws->size; i++) {
                CRef cref = ws->watches[i];
                Clause &c = *((Clause *) &m_data[cref]);
                assert(c[0] == ~l || c[1] == ~l);

                if (c[1] == ~l) c.swap(0, 1);
                if (value(c[1]) == l_True)
                    ws->watches[j++] = ws->watches[i];
                else {
                    // search for another watch.
                    unsigned pi = 2;
                    while (pi < c.size && value(c[pi]) == l_False) pi++;

                    // UNSAT case:
                    if (pi == c.size) {
                        if (value(c[1]) == l_False) {
                            while (i < ws->size) ws->watches[j++] = ws->watches[i++];
                            ws->size = j;
                            return false;
                        } else {
                            ws->watches[j++] = ws->watches[i];
                            uncheckedEnqueue(c[1]);
                        }
                    } else {
                        // remaining:
                        if (value(c[pi]) == l_True)
                            ws->watches[j++] = ws->watches[i];
                        else {
                            c.swap(0, pi);
                            m_watchList[c[0].intern()]->push(cref);
                        }
                    }
                }
            }
            ws->size = j;
        }

        return true;
    }  // propagate

/**
 * @brief display implementation.
 */
    void Propagator::display(std::ostream &out) {
        std::vector<std::vector<Lit> > clauses;
        extractFormula(clauses);
        out << "p cnf " << m_nbVar << " " << clauses.size() << "\n";

        // print the clauses.
        for (auto c: clauses) {
            for (unsigned i = 0; i < c.size(); i++) out << c[i].human() << " ";
            out << "0\n";
        }
    }  // display

/**
 * @brief attachClause implementation.
 */
    void Propagator::attachClause(CRef cref) {
        Clause &c = *((Clause *) &m_data[cref]);
        Watch *ws = m_watchList[c[0].intern()];
        ws->watches[ws->size++] = cref;

        ws = m_watchList[c[1].intern()];
        ws->watches[ws->size++] = cref;
    }  // attachClause

/**
 * @brief detachClause implementation.
 */
    void Propagator::detachClause(CRef cref) {
        Clause &c = *((Clause *) &m_data[cref]);
        Watch *ws = m_watchList[c[0].intern()];
        unsigned pos = 0;
        while (pos < ws->size && ws->watches[pos] != cref) pos++;
        assert(pos < ws->size);
        ws->watches[pos] = ws->watches[--ws->size];

        ws = m_watchList[c[1].intern()];
        pos = 0;
        while (pos < ws->size && ws->watches[pos] != cref) pos++;
        assert(pos < ws->size);
        ws->watches[pos] = ws->watches[--ws->size];
    }  // detachClause

/**
 * @brief extractFormula implementation.
 */
    void Propagator::extractFormula(std::vector<std::vector<Lit> > &clauses) {
        // if the problem is unsat, then ... the problem is unsat :D
        if (m_isUnsat) return generateUnsat(clauses);

        // get the unit literals.
        for (unsigned i = 0; i < m_trailLimUnit; i++) clauses.push_back({m_trail[i]});

        // print the binary clauses.
        for (unsigned lit = 0; lit < m_binListRefs.size(); lit++) {
            Lit l1 = Lit::makeLit(lit >> 1, (lit ^ 1) & 1);
            if (value(l1) == l_True) continue;

            Imply *imply = m_binListRefs[lit];

            for (unsigned i = 0; i < imply->size; i++)
                if (imply->lits[i].intern() < lit) {
                    Lit l2 = imply->lits[i];
                    if (value(l2) == l_True) continue;

                    if (value(l1) >= l_Undef && value(l2) >= l_Undef)
                        clauses.push_back({l1, l2});
                    else if (value(l1) == l_False)
                        clauses.push_back({l2});
                    else if (value(l2) == l_False)
                        clauses.push_back({~l1});
                }
        }

        // print the longest clauses.
        for (auto cref: m_notBinClauseRefs) {
            Clause &c = getClause(cref);
            std::vector<Lit> cl;
            bool isSAT = false;

            for (unsigned i = 0; !isSAT && i < c.size; i++) {
                if (value(c[i]) >= l_Undef)
                    cl.push_back(c[i]);
                else
                    isSAT = value(c[i]) == l_True;
            }

            if (isSAT) continue;
            if (cl.size() == 0)
                return generateUnsat(clauses);
            else
                clauses.push_back(cl);
        }
    }  // extractFormula

    void Propagator::restart() {
        if(m_nbVar == 0)
            return;
        for (unsigned i = m_trailLimUnit; i < m_trailSize; i++)
            m_assign[m_trail[i].var()] = l_Undef;
        m_trailPos = m_trailSize = m_trailLimUnit;
    }  // restart

    void Propagator::cancelUntilPos(unsigned pos) {
        if(m_nbVar == 0)
            return;
        //std::cout << "cancel until " << pos << std::endl;
        while (m_trailSize > pos) m_assign[m_trail[--m_trailSize].var()] = l_Undef;
        if (m_trailPos > m_trailSize) m_trailPos = m_trailSize;
    }  // cancelUntilPos

}  // namespace pyxai