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

struct IterSolveParameters
{
    int print_level = 0;
    int max_iter = 500;
    double abs_tol = 1e-12;
    double rel_tol = 1e-9;
    bool iter_mode = false;
};

struct MLDivFreeSolveParameters
{
    bool ml_part = true;
    bool verbose = false;
    IterSolveParameters CTMC_solve_param;
    IterSolveParameters BBT_solve_param;
};

void SetOptions(IterativeSolver& solver, int print_lvl, int max_it,
                double atol, double rtol, bool iter_mode=false);

void SetOptions(IterativeSolver& solver, const IterSolveParameters& param);

void PrintConvergence(const IterativeSolver& solver, bool verbose);

Vector MLDivPart(const HypreParMatrix& M,
                 const HypreParMatrix& B,
                 const Vector& F,
                 const Array<SparseMatrix> &agg_elem,
                 const Array<SparseMatrix> &elem_hdivdofs,
                 const Array<SparseMatrix> &elem_l2dofs,
                 const Array<OperatorHandle> &P_hdiv,
                 const Array<OperatorHandle> &P_l2,
                 const Array<int>& coarsest_ess_dofs);

class BBTSolver : public Solver
{
public:
    BBTSolver(HypreParMatrix& B, IterSolveParameters param = IterSolveParameters());

    virtual void Mult(const Vector &x, Vector &y) const;

    virtual void SetOperator(const Operator &op) { }
private:
    OperatorHandle BT_;
    OperatorHandle S_;
    HypreBoomerAMG invS_;
    CGSolver S_solver_;
    int verbose_;
};

class InterpolationCollector
{
public:
    InterpolationCollector(const ParFiniteElementSpace &fes, int num_refine);

    void CollectData(const ParFiniteElementSpace& fes);

    const Array<OperatorHandle>& GetP() const { return P_; }

private:
    Array<OperatorHandle> P_;
    unique_ptr<ParFiniteElementSpace> coarse_fes_;
    int refine_count_;
};

struct HdivL2Hierarchy
{
    Array<SparseMatrix> agg_el_;
    Array<SparseMatrix> el_hdivdofs_;
    Array<SparseMatrix> el_l2dofs_;
    Array<OperatorHandle> P_hdiv_;
    Array<OperatorHandle> P_l2_;
    Array<int> coarse_ess_dofs_;

    HdivL2Hierarchy(const ParFiniteElementSpace &hdiv_fes,
                    const ParFiniteElementSpace &l2_fes,
                    int num_refine, const Array<int>& ess_bdr);

    void CollectData(const ParFiniteElementSpace& hdiv_fes,
                     const ParFiniteElementSpace& l2_fes);
private:
    unique_ptr<ParFiniteElementSpace> coarse_hdiv_fes_;
    unique_ptr<ParFiniteElementSpace> coarse_l2_fes_;
    L2_FECollection l2_coll_0;
    unique_ptr<FiniteElementSpace> l2_fes_0_;
    int refine_count_;
};

class MLDivFreeSolver : public Solver
{
public:
    MLDivFreeSolver(HdivL2Hierarchy& hierarchy, HypreParMatrix& M,
                    HypreParMatrix& B, HypreParMatrix& C,
                    MLDivFreeSolveParameters param = MLDivFreeSolveParameters());

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op) { }

    void SetupMG(const InterpolationCollector& P);
    void SetupAMS(ParFiniteElementSpace& hcurl_fes);
private:
    // Find a particular solution for div sigma = f
    void SolveParticular(const Vector& rhs, Vector& sol) const;

    void SolveDivFree(const Vector& rhs, Vector& sol) const;

    void SolvePotential(const Vector &rhs, Vector& sol) const;

    SparseMatrix M_fine_;
    SparseMatrix B_fine_;
    HdivL2Hierarchy& h_;
    HypreParMatrix& M_;
    HypreParMatrix& B_;
    HypreParMatrix& C_;

    BBTSolver BBT_solver_;
    OperatorHandle CTMC_;
    OperatorHandle CTMC_prec_;
    CGSolver CTMC_solver_;

    MLDivFreeSolveParameters param_;
    Array<int> offsets_;
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

/* Block diagonal preconditioner of the form
 *               P = [ diag(M)         0         ]
 *                   [  0       B diag(M)^-1 B^T ]
 *
 *   Here we use Symmetric Gauss-Seidel to approximate the inverse of the
 *   pressure Schur Complement.
 */
class L2H1Preconditioner : public BlockDiagonalPreconditioner
{
public:
    L2H1Preconditioner(HypreParMatrix& M,
                       HypreParMatrix& B,
                       const Array<int>& offsets);
private:
    OperatorHandle S_;
};


