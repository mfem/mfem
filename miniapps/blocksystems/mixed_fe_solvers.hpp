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
    bool use_schwarz_smoother = false;
    IterSolveParameters CTMC_solve_param;
    IterSolveParameters BBT_solve_param;
};

struct DFSData
{
    Array<OperatorPtr> agg_hdivdof;
    Array<OperatorPtr> agg_l2dof;
    Array<OperatorPtr> P_hdiv;
    Array<OperatorPtr> P_l2;
    Array<OperatorPtr> P_curl;
    Array<OperatorPtr> Q_l2;            // Q_l2[l] = W_l P_l2[l] (W_{l+1})^{-1}
    Array<int> coarsest_ess_hdivdofs;
    OperatorPtr C;                      // discrete curl: ND -> RT
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
    Array<Array<OperatorPtr> > agg_solver_;
    OperatorPtr coarsest_M_;
    OperatorPtr coarsest_B_;
    OperatorPtr coarsest_solver_;
public:
    MLDivSolver(const HypreParMatrix& M, const HypreParMatrix &B, const DFSData& data);

    virtual void Mult(const Vector & x, Vector & y) const;
    void Mult(int level, const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
};

class DivFreeSolver : public DarcySolver
{
    const HypreParMatrix& M_;
    const HypreParMatrix& B_;

    OperatorPtr particular_solver_;
    BBTSolver BBT_solver_;
    OperatorPtr CTMC_;
    OperatorPtr CTMC_prec_;
    CGSolver CTMC_solver_;

    const DFSData& data_;

    // Find a particular solution for div sigma_p = f
    void SolveParticular(const Vector& rhs, Vector& sol) const;
    void SolveDivFree(const Vector& rhs, Vector& sol) const;
    void SolvePotential(const Vector &rhs, Vector& sol) const;
public:
    DivFreeSolver(const HypreParMatrix& M, const HypreParMatrix &B,
                  ParFiniteElementSpace* hcurl_fes, const DFSData& data);
    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op) { }
    virtual int GetNumIterations() const { return CTMC_solver_.GetNumIterations(); }
};

class AbsMultigrid : public Solver
{
    const Array<OperatorPtr>& P_;

    Array<OperatorPtr> ops_;
    Array<OperatorPtr> smoothers_;
    OperatorPtr coarse_solver_;

    mutable Array<Vector> correct_;
    mutable Array<Vector> resid_;
    mutable Vector cor_cor_;

    void MG_Cycle(int level) const;
public:
    AbsMultigrid(HypreParMatrix& op, const Array<OperatorPtr>& P,
                 OperatorPtr coarse_solver=OperatorPtr());

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


