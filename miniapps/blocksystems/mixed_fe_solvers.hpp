#include "mfem.hpp"
#include <memory>

using namespace std;
using namespace mfem;

enum MG_Type { AlgebraicMG, GeometricMG };

/// Parameters for iterative solver
struct IterSolveParameters
{
    int print_level = 0;
    int max_iter = 500;
    double abs_tol = 1e-12;
    double rel_tol = 1e-9;
};

/// Parameters for the divergence free solver
struct DFSParameters : IterSolveParameters
{
    MG_Type MG_type = GeometricMG;
    bool verbose = false;
    bool B_has_nullity_one = false;
    bool coupled_solve = false;
    IterSolveParameters BBT_solve_param;
};

/// Data for the divergence free solver
struct DFSData
{
    Array<OperatorPtr> agg_hdivdof;
    Array<OperatorPtr> agg_l2dof;
    Array<OperatorPtr> P_hdiv;
    Array<OperatorPtr> P_l2;
    Array<OperatorPtr> P_hcurl;
    Array<OperatorPtr> Q_l2;           // Q_l2[l] = (W_{l+1})^{-1} P_l2[l]^T W_l
    Array<int> coarsest_ess_hdivdofs;
    Array<OperatorPtr> C;              // discrete curl: ND -> RT
    DFSParameters param;
};

/// For collecting data needed for the divergence free solver
class DFSDataCollector
{
    RT_FECollection hdiv_fec_;
    L2_FECollection l2_fec_;
    ND_FECollection hcurl_fec_;
    L2_FECollection l2_0_fec_;

    unique_ptr<ParFiniteElementSpace> coarse_hdiv_fes_;
    unique_ptr<ParFiniteElementSpace> coarse_l2_fes_;
    unique_ptr<ParFiniteElementSpace> coarse_hcurl_fes_;
    unique_ptr<ParFiniteElementSpace> l2_0_fes_;

    Array<SparseMatrix> el_l2dof_;
    const Array<int>& ess_bdr_attr_;
    Array<int> all_bdr_attr_;

    int level_;
    int order_;
    DFSData data_;

    void MakeDofRelationTables(int level);
    void DataFinalize(ParMesh* mesh);
public:
    DFSDataCollector(int order, int num_refine, ParMesh *mesh,
                     const Array<int>& ess_attr, const DFSParameters& param);

    void CollectData(ParMesh *mesh);

    const DFSData& GetData() const { return data_; }

    unique_ptr<ParFiniteElementSpace> hdiv_fes_;
    unique_ptr<ParFiniteElementSpace> l2_fes_;
    unique_ptr<ParFiniteElementSpace> hcurl_fes_;
};

/// Abstract solver class for Darcy's flow
class DarcySolver : public Solver
{
protected:
    Array<int> offsets_;
public:
    DarcySolver(int size0, int size1) : Solver(size0 + size1), offsets_(3)
    { offsets_[0] = 0; offsets_[1] = size0; offsets_[2] = height; }
    virtual int GetNumIterations() const = 0;
};

/// Solver for B * B^T
class BBTSolver : public Solver
{
    OperatorPtr BBT_;
    OperatorPtr BBT_prec_;
    CGSolver BBT_solver_;
    bool B_has_nullity_one_;
public:
    BBTSolver(const HypreParMatrix &B, bool B_has_nullity_one=false,
              IterSolveParameters param=IterSolveParameters());
    virtual void Mult(const Vector &x, Vector &y) const;
    virtual void SetOperator(const Operator &op) { }
};

/// Block diagonal solver for A, each block is inverted by direct solver
class BlockDiagSolver : public Solver
{
    mutable SparseMatrix block_dof_;
    mutable Array<int> local_dofs_;
    mutable Vector sub_rhs_;
    mutable Vector sub_sol_;
    Array<DenseMatrixInverse> block_solver_;
public:
    BlockDiagSolver(const OperatorPtr& A, SparseMatrix block_dof);
    virtual void Mult(const Vector &x, Vector &y) const;
    virtual void MultTranspose(const Vector &x, Vector &y) const { Mult(x, y); }
    virtual void SetOperator(const Operator &op) { }
};

/// Solver for local problems in SchwarzSmoother
class LocalSolver : public Solver
{
    DenseMatrix local_system_;
    DenseMatrixInverse local_solver_;
    const int offset_;
public:
    LocalSolver(const DenseMatrix &M, const DenseMatrix &B);
    virtual void Mult(const Vector &x, Vector &y) const;
    virtual void SetOperator(const Operator &op) { }
};

/// non-overlapping additive Schwarz for saddle point problems
class SchwarzSmoother : public Solver
{
    const SparseMatrix& agg_hdivdof_;
    const SparseMatrix& agg_l2dof_;
    OperatorPtr coarse_l2_projector_;

    Array<int> offsets_;
    mutable Array<int> offsets_loc_;
    mutable Array<int> hdivdofs_loc_;
    mutable Array<int> l2dofs_loc_;
    Array<OperatorPtr> solvers_loc_;
public:
    SchwarzSmoother(const HypreParMatrix& M,
                    const HypreParMatrix& B,
                    const SparseMatrix& agg_hdivdof,
                    const SparseMatrix& agg_l2dof,
                    const HypreParMatrix& P_l2,
                    const HypreParMatrix& Q_l2);
    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
};

/// Relaxation on the kernel of divergence
class KernelSmoother : public Solver
{
    Array<int> offsets_;
    OperatorPtr blk_kernel_map_;
    OperatorPtr kernel_system_;
    OperatorPtr kernel_smoother_;
public:
    KernelSmoother(const BlockOperator& op, const HypreParMatrix& kernel_map_);
    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
};

/// Solver S such that I - A * S = (I - A * S1) * (I - A * S0).
/// A, S0, S1 are assumed to be symmetric.
class ProductSolver : public Solver
{
    OperatorPtr op_;
    Array<OperatorPtr> solvers_;
    void Mult(int i, int j, const Vector & x, Vector & y) const;
public:
    ProductSolver(Operator* A, Operator* S0, Operator* S1,
                  bool ownA, bool ownS0, bool ownS1)
        : Solver(A->NumRows()), op_(A, ownA), solvers_(2)
    {
        solvers_[0].Reset(S0, ownS0);
        solvers_[1].Reset(S1, ownS1);
    }
    virtual void Mult(const Vector & x, Vector & y) const { Mult(0, 1, x, y); }
    virtual void MultTranspose(const Vector & x, Vector & y) const { Mult(1, 0, x, y); }
    virtual void SetOperator(const Operator &op) { }
};

/// Divergence free solver, cf.
class DivFreeSolver : public DarcySolver
{
    const DFSData& data_;
    OperatorPtr BT_;
    BBTSolver BBT_solver_;
    OperatorPtr CTMC_;
    Array<Array<int>> ops_offsets_;
    Array<OperatorPtr> ops_;
    Array<OperatorPtr> blk_Ps_;
    Array<OperatorPtr> smoothers_;
    OperatorPtr prec_;
    OperatorPtr solver_;

    void SolveParticular(const Vector& rhs, Vector& sol) const;
    void SolveDivFree(const Vector& rhs, Vector& sol) const;
    void SolvePotential(const Vector &rhs, Vector& sol) const;
public:
    DivFreeSolver(const HypreParMatrix& M, const HypreParMatrix &B,
                  ParFiniteElementSpace* hcurl_fes, const DFSData& data);
    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
    virtual int GetNumIterations() const
    {
        return solver_.As<IterativeSolver>()->GetNumIterations();
    }
};

/// Multigrid class that uses HypreSmoother if smoothers are not given
class AbstractMultigrid : public Solver
{
    const Array<OperatorPtr>& Ps_;
    Array<OperatorPtr> ops_;
    Array<OperatorPtr> smoothers_;
    mutable Array<Vector> correct_;
    mutable Array<Vector> resid_;
    mutable Vector cor_cor_;
    bool smoothers_are_symmetric_;

    void MG_Cycle(int level) const;
public:
    AbstractMultigrid(HypreParMatrix& op, const Array<OperatorPtr>& Ps);
    AbstractMultigrid(const Array<OperatorPtr>& ops,
                      const Array<OperatorPtr>& Ps,
                      const Array<OperatorPtr>& smoothers,
                      bool smoother_is_symmetric = false);

    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
};

/// Wrapper for the block-diagonally preconditioned MINRES defined in ex5p.cpp
class BDPMinresSolver : public DarcySolver
{
    BlockOperator op_;
    BlockDiagonalPreconditioner prec_;
    OperatorPtr BT_;
    OperatorPtr S_;   // S_ = B diag(M)^{-1} B^T
    MINRESSolver solver_;
    Array<int> ess_zero_dofs_;
public:
    BDPMinresSolver(HypreParMatrix& M, HypreParMatrix& B,
                    bool own_input, IterSolveParameters param);
    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
    void SetEssZeroDofs(const Array<int>& dofs) { dofs.Copy(ess_zero_dofs_); }
    const BlockOperator& GetOperator() const { return op_; }
    virtual int GetNumIterations() const { return solver_.GetNumIterations(); }
};
