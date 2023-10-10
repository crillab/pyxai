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

#include <iostream>
#include <vector>

#include "./Problem.h"
#include "./ProblemTypes.h"



namespace pyxai {
    struct Imply {
        unsigned size;
        Lit lits[0];
        Lit &operator[](std::size_t idx) { return lits[idx]; }
    };

    typedef unsigned CRef;

    struct Watch {
        unsigned size;
        CRef watches[0];

        inline void push(CRef cref) { watches[size++] = cref; }
    };

    class Propagator {
    private:
        std::ostream &m_out;

        uint8_t *m_data;

        unsigned m_nbVar;
        unsigned m_posClauseNotBin;
        bool m_isUnsat;

        std::vector<CRef> m_notBinClauseRefs;
        std::vector<Imply *> m_binListRefs;
        std::vector<Watch *> m_watchList;

        Lit *m_trail;
        unsigned m_trailSize;
        unsigned m_trailLimUnit;
        unsigned m_trailPos;

        lbool *m_assign;
        bool m_verbose;

        /**
         * @brief Generate an unsat problem.
         *
         * @param clauses is where is strored the problem.
         */
        inline void generateUnsat(std::vector<std::vector<Lit> > &clauses) {
            clauses.clear();
            clauses.push_back({Lit::makeLitTrue(1)});
            clauses.push_back({Lit::makeLitFalse(1)});
        }  // generateUnsat

        /**
         * @brief Delete the defaut constructor.
         *
         */

    public:
        Propagator();

        inline std::vector<CRef> &getNotBinClauses() { return m_notBinClauseRefs; }

        inline Clause &getClause(CRef cref) { return *((Clause * ) & m_data[cref]); }

        inline void setIsUnsat(bool val) { m_isUnsat = val; }

        inline bool getIsUnsat() { return m_isUnsat; }

        inline unsigned getNbVar() { return m_nbVar; }

        inline Imply *litImplied(Lit l) { return m_binListRefs[l.intern()]; }

        inline unsigned getTrailSize() { return m_trailSize; }

        /**
         * @brief Construct a new Propagator object.
         *
         * @param p is the problem we want to load.
         */
        Propagator(Problem &p, std::ostream &out, bool verbose = false);

        Propagator(Problem &p, bool verbose = false)
                : Propagator(p, std::cout, verbose) {}

        /**
         * @brief Destroy the Propagator object.
         *
         */
        ~Propagator();

        /**
         * @brief Add a binary clause.
         *
         * @param a is the first literal of the clause.
         * @param b is the seconde literal of the clause.
         */
        void addBinary(Lit a, Lit b);

        /**
         * @brief Add a clause.
         *
         * @param clause is the clause we want to add.
         */
        void addClause(std::vector<Lit> &clause);

        /**
         * @brief Ask for the value.
         *
         * @param l is the literal we want to the value.
         * @return 0 if SAT, 1 if UNSAT and >1 otherwise.
         */
        inline lbool value(const Lit l) {
            if(l.var() > ((int)m_nbVar))
                return l_Undef;
            return ((uint8_t) l.sign()) ^ m_assign[l.var()];
        }

        /**
         * @brief Push a literal in the trail.
         *
         * @param l is the literal we push.
         */
        void uncheckedEnqueue(Lit l);

        /**
         * @brief Realize the unit propagation process.
         *
         * @return false if a conflict occurs, true otherwise.
         */
        bool propagate();

        /**
         * @brief Propagate the literals in the stack as they are unit literals.
         *
         */
        void propagateLevelZero() {
            m_isUnsat = !propagate();
            m_trailLimUnit = m_trailSize;
        }  // propagateLevelZero

        /**
         * @brief Display, following the DIMACS format, the problem contained in the
         * propagator.
         *
         */
        void display(std::ostream &out);

        void displayTrail() {
            if(getNbVar() == 0)
                return;
            std::cout << "  [";
            for(unsigned int i = 0; i < m_trailSize; i++)
                std::cout << m_trail[i] << " ";
            std::cout << "]" << std::endl;
        }
        /**
         * @brief Attach a clause regarding its CRef.
         *
         * @param cref is the clause we want to attach.
         */
        void attachClause(CRef cref);

        /**
         * @brief Detach a clause regarding its CRef.
         *
         * @param cref is the clause we want to detach.
         */
        void detachClause(CRef cref);

        /**
         * @brief Extract the clauses.
         *
         * @param clauses is where are stored the clauses.
         */
        void extractFormula(std::vector<std::vector<Lit> > &clauses);

        /**
         * @brief Restart
         */
        void restart();

        /**
         * @brief Remove the propagation until the trail size is equal to pos.
         *
         * @param pos is the size the trail should be at the end of the process
         * (except is its size is less than pos).
         */
        void cancelUntilPos(unsigned pos);
    };
}  // namespace rfx