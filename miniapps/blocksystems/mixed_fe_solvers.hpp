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

void GetDiagBlockSubMatrix(const HypreParMatrix& A, const Array<int> rows,
                           const Array<int> cols, DenseMatrix& sub_A);

SparseMatrix AggToIntDof(const SparseMatrix& agg_elem, const SparseMatrix& elem_dof);

Vector LocalSolution(const DenseMatrix& M,  const DenseMatrix& B, const Vector& F);

SparseMatrix ElemToTrueDofs(const FiniteElementSpace& fes);

Vector MLDivPart(const HypreParMatrix& M,
                 const HypreParMatrix& B,
                 const Vector& F,
                 const vector<SparseMatrix>& agg_elem,
                 const vector<SparseMatrix>& elem_hdivdofs,
                 const vector<SparseMatrix>& elem_l2dofs,
                 const vector<OperatorHandle>& P_hdiv,
                 const vector<OperatorHandle>& P_l2,
                 const HypreParMatrix& coarse_hdiv_d_td,
                 const HypreParMatrix& coarse_l2_d_td,
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
    InterpolationCollector(ParFiniteElementSpace& fes, int num_refine);

    void CollectData(ParFiniteElementSpace& fes);

    const Array<OperatorHandle>& GetP() const { return P_; }

private:
    Array<OperatorHandle> P_;
    ParFiniteElementSpace coarse_fes_;
    int refine_count_;
};

struct HdivL2Hierarchy
{
    friend class MLDivFreeSolver;

    HdivL2Hierarchy(ParFiniteElementSpace& hdiv_fes,
                    ParFiniteElementSpace& l2_fes,
                    int num_refine, const Array<int>& ess_bdr);

    void CollectData();
private:
    ParFiniteElementSpace& hdiv_fes_;
    ParFiniteElementSpace& l2_fes_;
    ParFiniteElementSpace coarse_hdiv_fes_;
    ParFiniteElementSpace coarse_l2_fes_;

    L2_FECollection l2_coll_0;
    ParFiniteElementSpace l2_fes_0_;

    vector<SparseMatrix> agg_el_;
    vector<SparseMatrix> el_hdivdofs_;
    vector<SparseMatrix> el_l2dofs_;
    vector<OperatorHandle> P_hdiv_;
    vector<OperatorHandle> P_l2_;
    Array<int> coarse_ess_dofs_;

    int refine_count_;
};

class MLDivFreeSolver : public Solver
{
public:
    MLDivFreeSolver(HdivL2Hierarchy& hierarchy, HypreParMatrix& M,
                    HypreParMatrix& B, HypreParMatrix& BT, HypreParMatrix& C,
                    MLDivFreeSolveParameters param = MLDivFreeSolveParameters());

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op);

    void SetupMG(const InterpolationCollector& P);
    void SetupAMS(ParFiniteElementSpace& hcurl_fes);
private:
    // Find a particular solution for div sigma = f
    void SolveParticularSolution(const Vector& blk_rhs_1,
                                 Vector& true_flux_part) const;

    void SolveDivFreeSolution(const Vector& true_flux_part,
                              Vector& blk_rhs_0,
                              Vector& true_flux_divfree) const;

    void SolvePotential(const Vector& true_flux_divfree,
                        Vector& blk_rhs_0,
                        Vector& potential) const;

    SparseMatrix M_fine_;
    SparseMatrix B_fine_;
    HdivL2Hierarchy& h_;
    HypreParMatrix& M_;
    HypreParMatrix& B_;
    HypreParMatrix& BT_;
    HypreParMatrix& C_;
    OperatorHandle CT_;

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


