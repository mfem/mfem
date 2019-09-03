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
    OperatorPtr C;                   // discrete curl: ND -> RT
    OperatorPtr W;                   // L2 FE space mass matrix
    DivFreeSolverParameters param;
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

class BBTSolver : public Solver // TODO: make it a template?
{
    OperatorPtr BBT_;
    OperatorPtr BBT_prec_;
    CGSolver BBT_solver_;
    bool B_has_nullity_one_;
    int verbose_;
public:
    BBTSolver(const HypreParMatrix &B, bool B_has_nullity_one=false,
              IterSolveParameters param=IterSolveParameters());

    virtual void Mult(const Vector &x, Vector &y) const;

    virtual void SetOperator(const Operator &op) { }
};

class LocalSolver : public Solver // TODO: make it a template?
{
    DenseMatrix BT_;
    DenseMatrix BBT_;
    DenseMatrixInverse BBT_solver_;
public:
    LocalSolver(const DenseMatrix &B);
    virtual void Mult(const Vector &x, Vector &y) const;
    virtual void SetOperator(const Operator &op) { }
};

class BlockDiagSolver : public Solver
{
    SparseMatrix& block_dof_ref_;
    Array<DenseMatrixInverse> block_solver_;
    mutable Array<int> local_dofs_;
    mutable Vector sub_rhs_;
    mutable Vector sub_sol_;
public:
    BlockDiagSolver(const OperatorPtr& A, const SparseMatrix& block_dof);
    virtual void Mult(const Vector &x, Vector &y) const;
    virtual void SetOperator(const Operator &op) { }
};

class DivFreeSolverDataCollector
{
    RT_FECollection hdiv_fec_;
    L2_FECollection l2_fec_;
    ND_FECollection hcurl_fec_;
    L2_FECollection l2_0_fec_;

    unique_ptr<ParFiniteElementSpace> coarse_hdiv_fes_;
    unique_ptr<ParFiniteElementSpace> coarse_l2_fes_;
    unique_ptr<ParFiniteElementSpace> coarse_hcurl_fes_;
    unique_ptr<FiniteElementSpace> l2_0_fes_;

    const Array<int>& ess_bdr_;
    Array<int> all_bdr_;

    int level_;
    int order_;
    DivFreeSolverData data_;
public:
    DivFreeSolverDataCollector(int order, int num_refine, ParMesh *mesh,
                               const Array<int>& ess_bdr,
                               const DivFreeSolverParameters& param);

    void CollectData(ParMesh *mesh);

    const DivFreeSolverData& GetData() const { return data_; }

    unique_ptr<ParFiniteElementSpace> hdiv_fes_;
    unique_ptr<ParFiniteElementSpace> l2_fes_;
    unique_ptr<ParFiniteElementSpace> hcurl_fes_;
};

class MLDivSolver : public Solver
{
    const DivFreeSolverData& data_;
    Array<OperatorPtr> agg_hdivdof_;
    Array<OperatorPtr> agg_l2dof_;
    Array<Array<OperatorPtr> > agg_solver_;
    Array<OperatorHandle> W_;
    Array<OperatorHandle> coarser_W_inv_;
    OperatorPtr coarsest_B_;
    OperatorPtr coarsest_solver_;
public:
    MLDivSolver(const HypreParMatrix& M, const HypreParMatrix &B,
                const OperatorPtr& W, const DivFreeSolverData& data);

    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
};

class DivFreeSolver : public Solver
{
    // Find a particular solution for div sigma_p = f
    void SolveParticular(const Vector& rhs, Vector& sol) const;
    void SolveDivFree(const Vector& rhs, Vector& sol) const;
    void SolvePotential(const Vector &rhs, Vector& sol) const;

    const HypreParMatrix& M_;
    const HypreParMatrix& B_;

    OperatorPtr particular_solver_;
    BBTSolver BBT_solver_;
    OperatorPtr CTMC_;
    OperatorPtr CTMC_prec_;
    CGSolver CTMC_solver_;

    Array<int> offsets_;

    const DivFreeSolverData& data_;
public:
    DivFreeSolver(const HypreParMatrix& M, const HypreParMatrix &B,
                  ParFiniteElementSpace* hcurl_fes,
                  const DivFreeSolverData& data);

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op) { }
};

class Multigrid : public Solver
{
    void MG_Cycle(int level) const;

    const Array<OperatorPtr>& P_;

    Array<OperatorPtr> ops_;
    Array<OperatorPtr> smoothers_;
    OperatorPtr coarse_solver_;

    mutable Array<Vector> correct_;
    mutable Array<Vector> resid_;
    mutable Vector cor_cor_;
public:
    Multigrid(HypreParMatrix& op, const Array<OperatorPtr>& P,
              OperatorPtr coarse_solver=OperatorPtr());

    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
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
    OperatorPtr S_;
public:
    L2H1Preconditioner(HypreParMatrix& M,
                       HypreParMatrix& B,
                       const Array<int>& offsets);
};


