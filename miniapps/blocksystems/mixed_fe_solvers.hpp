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

struct DFSParameters
{
    bool ml_particular = true;
    MG_Type MG_type = GeometricMG;
    bool verbose = false;
    bool B_has_nullity_one = false;
    bool coupled_solve = false;
    IterSolveParameters CTMC_solve_param;
    IterSolveParameters BBT_solve_param;
};

struct DFSData
{
    Array<OperatorPtr> agg_hdivdof;
    Array<OperatorPtr> agg_l2dof;
    Array<OperatorPtr> P_hdiv;
    Array<OperatorPtr> P_l2;
    Array<OperatorPtr> P_hcurl;
    Array<OperatorPtr> Q_l2;            // Q_l2[l] = W_l P_l2[l] (W_{l+1})^{-1}
    Array<int> coarsest_ess_hdivdofs;
//    OperatorPtr C;                      // discrete curl: ND -> RT
    Array<OperatorPtr> C;
    DFSParameters param;
};

void SetOptions(IterativeSolver& solver, int print_lvl, int max_it,
                double atol, double rtol, bool iter_mode=true);

void PrintConvergence(const IterativeSolver& solver, bool verbose);

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
    DenseMatrix M_;
    DenseMatrix BT_;
    DenseMatrix local_system_;
    DenseMatrixInverse local_solver_;
    const int offset_;
public:
    LocalSolver(const DenseMatrix &B);
    LocalSolver(const DenseMatrix &M, const DenseMatrix &B);
    virtual void Mult(const Vector &x, Vector &y) const;
    virtual void SetOperator(const Operator &op) { }
};

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

class DarcySolver : public Solver
{
protected:
    Array<int> offsets_;
public:
    DarcySolver(int size0, int size1) : Solver(size0 + size1), offsets_(3)
    { offsets_[0] = 0; offsets_[1] = size0; offsets_[2] = height; }
    virtual int GetNumIterations() const = 0;
};

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

class MLDivSolver : public Solver
{
    const DFSData& data_;
    Array<Array<OperatorPtr> > agg_solvers_;
    OperatorPtr coarsest_M_;
    OperatorPtr coarsest_B_;
    OperatorPtr coarsest_solver_;
public:
    MLDivSolver(const HypreParMatrix& M, const HypreParMatrix &B, const DFSData& data);
    MLDivSolver(const Array<OperatorPtr>& M, const Array<OperatorPtr>& B, const DFSData& data);

    virtual void Mult(const Vector & x, Vector & y) const;
    void Mult(int level, const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
};

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
    SchwarzSmoother(const BlockOperator& op,
                    const SparseMatrix& agg_hdivdof,
                    const SparseMatrix& agg_l2dof,
                    const HypreParMatrix& P_l2,
                    const HypreParMatrix& Q_l2);
    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
};

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

class HiptmairSmoother : public Solver
{
    Array<int> C_col_offsets_;
    int level_;
    BlockOperator& op_;
    MLDivSolver& ml_div_solver_;
    OperatorPtr blk_C_;
    OperatorPtr div_kernel_system_;
    OperatorPtr div_kernel_smoother_;
public:
    HiptmairSmoother(BlockOperator& op,
                     MLDivSolver& ml_div_solver,
                     int level,
                     HypreParMatrix& C);

    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void MultTranspose(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
};

class DivFreeSolver : public DarcySolver
{
    const DFSData& data_;
    const HypreParMatrix& M_;
    const HypreParMatrix& B_;

    OperatorPtr particular_solver_;
    BBTSolver BBT_solver_;

    OperatorPtr CTMC_;
    OperatorPtr CTMC_prec_;
    CGSolver CTMC_solver_;

    OperatorPtr BT_;
    Array<Array<int>> coarse_offsets_;
    Array<OperatorPtr> ops_;
    Array<OperatorPtr> blk_Ps_;
    Array<OperatorPtr> smoothers_;

//    CGSolver block_solver_;
//    MINRESSolver block_solver_;
    GMRESSolver block_solver_;

    // Find a particular solution for div sigma_p = f
    void SolveParticular(const Vector& rhs, Vector& sol) const;
    void SolveDivFree(const Vector& rhs, Vector& sol) const;
    void SolvePotential(const Vector &rhs, Vector& sol) const;
public:
    DivFreeSolver(const HypreParMatrix& M, const HypreParMatrix &B,
                  ParFiniteElementSpace* hcurl_fes, const DFSData& data);
    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
    virtual int GetNumIterations() const { return block_solver_.GetNumIterations(); }
};

class AbstractMultigrid : public Solver
{
    const Array<OperatorPtr>& Ps_;

    Array<OperatorPtr> ops_;
    Array<OperatorPtr> smoothers_;

    mutable Array<Vector> correct_;
    mutable Array<Vector> resid_;
    mutable Vector cor_cor_;

    void MG_Cycle(int level) const;
public:
    AbstractMultigrid(HypreParMatrix& op, const Array<OperatorPtr>& Ps);
    AbstractMultigrid(const Array<OperatorPtr>& ops,
                      const Array<OperatorPtr>& Ps,
                      const Array<OperatorPtr>& smoothers);

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
public:
    BDPMinresSolver(HypreParMatrix& M, HypreParMatrix& B, IterSolveParameters param);
    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
    const Operator& GetOperator() const { return op_; }
    virtual int GetNumIterations() const { return solver_.GetNumIterations(); }
};


