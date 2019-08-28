//                       MFEM Example 5 - Parallel Version
//
// Compile with: make ex5p
//
// Sample runs:  mpirun -np 4 ex5p -m ../data/square-disc.mesh
//               mpirun -np 4 ex5p -m ../data/star.mesh
//               mpirun -np 4 ex5p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex5p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex5p -m ../data/escher.mesh
//               mpirun -np 4 ex5p -m ../data/fichera.mesh
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//                                 k*u + grad p = f
//                                 - div u      = g
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockMatrix class, as
//               well as the collective saving of several grid functions in a
//               VisIt (visit.llnl.gov) visualization format.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <memory>

using namespace std;
using namespace mfem;

void SetOptions(IterativeSolver& solver, int print_lvl, int max_it, double atol, double rtol);

SparseMatrix AggToIntDof(const SparseMatrix& agg_elem, const SparseMatrix& elem_dof);

Vector LocalSolution(const DenseMatrix& M,  const DenseMatrix& B, const Vector& F);

SparseMatrix ElemToDofs(const FiniteElementSpace& fes);

Vector div_part(const SparseMatrix& M_fine,
                const SparseMatrix& B_fine,
                const Vector& F_fine,
                const vector<SparseMatrix>& agg_elem,
                const vector<SparseMatrix>& elem_hdivdofs,
                const vector<SparseMatrix>& elem_l2dofs,
                const vector<SparseMatrix>& P_hdiv,
                const vector<SparseMatrix>& P_l2,
                const HypreParMatrix& coarse_hdiv_d_td,
                const HypreParMatrix& coarse_l2_d_td,
                const Array<int>& coarsest_ess_dofs);

class BBTSolver : public Solver
{
public:
    BBTSolver(HypreParMatrix& B, int print_level=0, int max_iter=500,
              double abs_tol=1e-12, double rel_tol=1e-9);

    virtual void Mult(const Vector &x, Vector &y) const;

    virtual void SetOperator(const Operator &op) { }
private:
    OperatorHandle BT_;
    OperatorHandle S_;
    HypreBoomerAMG invS_;
    CGSolver S_solver_;
    int verbose_;
};

class MLDivFreeSolver : public Solver
{
public:
    MLDivFreeSolver(ParFiniteElementSpace& hdiv_fes,
                    ParFiniteElementSpace& l2_fes,
                    int num_refine, const Array<int>& ess_bdr);

    void Collect();

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op);
private:
    ParFiniteElementSpace& hdiv_fes_;
    ParFiniteElementSpace& l2_fes_;
    ParFiniteElementSpace coarse_hdiv_fes_;
    ParFiniteElementSpace coarse_l2_fes_;

    L2_FECollection l2_coll_0;
    ParFiniteElementSpace l2_fes_0_;

    SparseMatrix M_fine_;
    SparseMatrix B_fine_;
    vector<SparseMatrix> agg_el_;
    vector<SparseMatrix> el_hdivdofs_;
    vector<SparseMatrix> el_l2dofs_;
    vector<SparseMatrix> P_hdiv_;
    vector<SparseMatrix> P_l2_;
    Array<int> coarse_ess_dofs_;

    int ref_count_;
};

class InterpolationCollector
{
public:
    InterpolationCollector(ParFiniteElementSpace& fes, int num_refine);

    void Collect();

    const Array<OperatorHandle>& GetP() const { return P_; }

private:
    Array<OperatorHandle> P_;
    ParFiniteElementSpace& fes_;
    ParFiniteElementSpace coarse_fes_;
    int ref_count_;
};

// Geometric Multigrid
class Multigrid : public Solver
{
public:
    Multigrid(HypreParMatrix& op, const Array<OperatorHandle>& P,
              OperatorHandle coarse_solver=OperatorHandle());

    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
private:
    void MG_Cycle(int level) const;

    const Array<OperatorHandle>& P_;

    Array<OperatorHandle> ops_;
    Array<OperatorHandle> smoothers_;
    OperatorHandle coarse_solver_;

    mutable Array<Vector> correct_;
    mutable Array<Vector> resid_;
    mutable Vector cor_cor_;
};

