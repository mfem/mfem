#include "mfem.hpp"
#include <memory>

using namespace std;
using namespace mfem;

enum MG_Type { AlgebraicMG, GeometricMG };

struct IterSolveParameters
{
    int print_level = 0;
    int max_iter = 500;
    double abs_tol = 1e-12;
    double rel_tol = 1e-9;
    bool iter_mode = false;
};

struct DivFreeSolverParameters
{
    bool ml_particular = true;
    MG_Type MG_type = GeometricMG;
    bool verbose = false;
    IterSolveParameters CTMC_solve_param;
    IterSolveParameters BBT_solve_param;
};

struct DivFreeSolverData
{
    Array<SparseMatrix> agg_el;
    Array<SparseMatrix> el_hdivdofs;
    Array<SparseMatrix> el_l2dofs;
    Array<OperatorHandle> P_hdiv;
    Array<OperatorHandle> P_l2;
    Array<OperatorHandle> P_curl;
    Array<int> coarse_ess_dofs;

    ND_FECollection hcurl_fec;
    ParFiniteElementSpace hcurl_fes;
    OperatorHandle discrete_curl;

    DivFreeSolverParameters param;

    DivFreeSolverData(int order, ParMesh* mesh)
        : hcurl_fec(order, mesh->Dimension()), hcurl_fes(mesh, &hcurl_fec) { }
};

void SetOptions(IterativeSolver& solver, int print_lvl, int max_it,
                double atol, double rtol, bool iter_mode=false);

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
    BBTSolver(const HypreParMatrix &B, IterSolveParameters param=IterSolveParameters());

    virtual void Mult(const Vector &x, Vector &y) const;

    virtual void SetOperator(const Operator &op) { }
private:
    OperatorHandle BT_;
    OperatorHandle S_;
    HypreBoomerAMG invS_;
    CGSolver S_solver_;
    int verbose_;
};

class DivFreeSolverDataCollector
{
public:
    DivFreeSolverDataCollector(const ParFiniteElementSpace &hdiv_fes,
                               const ParFiniteElementSpace &l2_fes,
                               int num_refine, const Array<int>& ess_bdr,
                               const DivFreeSolverParameters& param);

    void CollectData(const ParFiniteElementSpace& hdiv_fes,
                     const ParFiniteElementSpace& l2_fes);

    const DivFreeSolverData& GetData() const { return data_; }
private:
    unique_ptr<ParFiniteElementSpace> coarse_hdiv_fes_;
    unique_ptr<ParFiniteElementSpace> coarse_l2_fes_;
    unique_ptr<ParFiniteElementSpace> coarse_hcurl_fes_;

    L2_FECollection l2_0_fec_;
    unique_ptr<FiniteElementSpace> l2_0_fes_;

    int level_;

    DivFreeSolverData data_;
};

class DivFreeSolver : public Solver
{
public:
    DivFreeSolver(const HypreParMatrix& M, const HypreParMatrix &B,
                  const DivFreeSolverData& data);

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op) { }

private:
    // Find a particular solution for div sigma_p = f
    void SolveParticular(const Vector& rhs, Vector& sol) const;

    void SolveDivFree(const Vector& rhs, Vector& sol) const;

    void SolvePotential(const Vector &rhs, Vector& sol) const;

    const HypreParMatrix& M_;
    const HypreParMatrix& B_;

    BBTSolver BBT_solver_;
    OperatorHandle CTMC_;
    OperatorHandle CTMC_prec_;
    CGSolver CTMC_solver_;

    Array<int> offsets_;

    const DivFreeSolverData& data_;
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


