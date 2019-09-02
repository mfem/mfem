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
    bool B_has_nullity_one = false;
    IterSolveParameters CTMC_solve_param;
    IterSolveParameters BBT_solve_param;
};

struct DivFreeSolverData
{
    Array<SparseMatrix> agg_el;
    Array<SparseMatrix> el_hdivdof;
    Array<SparseMatrix> el_l2dof;
    Array<OperatorPtr> P_hdiv;
    Array<OperatorPtr> P_l2;
    Array<OperatorPtr> P_curl;
    Array<Array<int> > bdr_hdivdofs; // processor bdr dofs (global bdr + shared)
    Array<int> coarsest_ess_hdivdofs;

    ND_FECollection hcurl_fec;
    ParFiniteElementSpace hcurl_fes;
    OperatorPtr discrete_curl;

    OperatorPtr mass_l2;

    DivFreeSolverParameters param;

    DivFreeSolverData(int order, ParMesh* mesh)
        : hcurl_fec(order, mesh->Dimension()), hcurl_fes(mesh, &hcurl_fec) { }
};

void SetOptions(IterativeSolver& solver, int print_lvl, int max_it,
                double atol, double rtol, bool iter_mode=true);

void PrintConvergence(const IterativeSolver& solver, bool verbose);

Vector MLDivPart(const HypreParMatrix& M,
                 const HypreParMatrix& B,
                 const HypreParMatrix& W,
                 const Vector& F,
                 const Array<SparseMatrix> &agg_elem,
                 const Array<SparseMatrix> &elem_hdivdof,
                 const Array<SparseMatrix> &elem_l2dof,
                 const Array<OperatorPtr> &P_hdiv,
                 const Array<OperatorPtr> &P_l2,
                 const Array<Array<int> > &bdr_hdivdofs,
                 const Array<int> &coarsest_ess_hdivdofs);

class BBTSolver : public Solver
{
public:
    BBTSolver(const HypreParMatrix &B, bool B_has_nullity_one=false,
              IterSolveParameters param=IterSolveParameters());

    virtual void Mult(const Vector &x, Vector &y) const;

    virtual void SetOperator(const Operator &op) { }
private:
    OperatorPtr BT_;
    OperatorPtr S_;
    OperatorPtr invS_;
    CGSolver S_solver_;
    bool B_has_nullity_one_;
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

    const Array<int>& ess_bdr_;
    Array<int> all_bdr_;

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
    OperatorPtr CTMC_;
    OperatorPtr CTMC_prec_;
    CGSolver CTMC_solver_;

    Array<int> offsets_;

    const DivFreeSolverData& data_;
};

// Geometric Multigrid
class Multigrid : public Solver
{
public:
    Multigrid(HypreParMatrix& op, const Array<OperatorPtr>& P,
              OperatorPtr coarse_solver=OperatorPtr());

    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
private:
    void MG_Cycle(int level) const;

    const Array<OperatorPtr>& P_;

    Array<OperatorPtr> ops_;
    Array<OperatorPtr> smoothers_;
    OperatorPtr coarse_solver_;

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
    OperatorPtr S_;
};


