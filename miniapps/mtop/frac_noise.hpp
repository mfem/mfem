#ifndef FRACT_NOISE_HPP
#define FRACT_NOISE_HPP 

#include "mfem.hpp"

namespace mfem
{


/// Symmetrized Smoother for use in iterative methods.
/// Given a smoother S and operator A, the symmetrized smoother
/// is defined as  (S^-1)+(S^-T)-(S^-1) A (S^-T). For more details,
/// see "Multilevel Block Factorization Preconditioners" by P. S. Vassilevski,
/// 2008, Section 3.1.3.    
class SymmetrizedSmoother : public Solver
{
    public:
    SymmetrizedSmoother(Solver *smoother_, Operator *op_)
        : Solver(op_->Height()), smoother(smoother_), op(op_) {}

    void Mult(const Vector &x, Vector &y) const override
    {
        tmp1.SetSize(x.Size());
        tmp2.SetSize(x.Size());

        smoother->MultTranspose(x, y);
        op->Mult(y, tmp2);
        smoother->Mult(x, tmp1);
        y.Add(1.0, tmp1);

        smoother->Mult(tmp2, tmp1);
        y.Add(-1.0, tmp1);
    }

    void MultTranspose(const Vector &x, Vector &y) const override
    {
        Mult(x, y); // Symmetric
    }

    virtual
    void SetOperator(const Operator &op_) 
    {
        op = &op_;
    }

    void SetSmoother(const Solver &smoother_)
    {
        smoother = &smoother_;
    }

    private:
    const Solver *smoother;
    const Operator *op;

    mutable Vector tmp1;
    mutable Vector tmp2;

};


class FracRandomFieldGenerator: public Solver
{
public:
    FracRandomFieldGenerator(ParMesh &pmesh_,
                             const int par_ref_levels_, 
                             const int order_=1.0, real_t sigma_=1.0, real_t s=0.0);
        
    ~FracRandomFieldGenerator();


    void Mult(const Vector &x, Vector &y) const override;    
    

    /// not supported for FracRandomFieldGenerator
    virtual void SetOperator(const Operator &op) 
    {
        MFEM_ABORT("SetOperator is not supported in Multigrid!");
    }   

    ParFiniteElementSpace& GetFinestFESpace()
    {
        return fespaces->GetFinestFESpace();
    }

    ParFiniteElementSpaceHierarchy& GetFESpaceHierarchy()
    {
        return *fespaces;
    }

    ParFiniteElementSpace* GetCoarsestFESpace()
    {
        return &fespaces->GetFESpaceAtLevel(0);
    }

private:
    ParMesh &pmesh;
    const int par_ref_levels;
    const int order;
    real_t sigma;
    real_t s;

    std::unique_ptr<FiniteElementCollection> fec;
    std::unique_ptr<ParFiniteElementSpaceHierarchy> fespaces;

    mfem::Array<Operator*> prolongations;
    mfem::Array<Operator*> operators;
    mfem::Array<Solver*> smoothers;



};

};// namespace mfem


#endif // FRACT_NOISE_HPP