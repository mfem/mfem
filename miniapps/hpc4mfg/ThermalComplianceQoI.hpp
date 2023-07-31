#ifndef HPC4THERMALCOMPLIANCE_HPP
#define HPC4THERMALCOMPLIANCE_HPP

#include "mfem.hpp"
#include "hpc4solvers.hpp"
#include "hpc4DiffMat.hpp"


namespace mfem{


class ThermalComplianceIntegrator : public NonlinearFormIntegrator
{
public:

    ThermalComplianceIntegrator()
    {};

    void SetFieldsAndMicrostructure(
        BasicAdvDiffCoefficient * MicroModelCoeff_,
        ParGridFunction * tempfield_,
        ParGridFunction * preassure_,
        ParGridFunction * desfield_)
    {
        MicroModelCoeff=MicroModelCoeff_;
        tempGF         =tempfield_;
        preassureGF    =preassure_;
        designGF       =desfield_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun);

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat)
    {
        {
            mfem::mfem_error("ThermalComplianceIntegrator::AssembleElementGrad is not defined!");
        }
    }

private:
    mfem::ParGridFunction             * tempGF          = nullptr;
    mfem::ParGridFunction             * preassureGF     = nullptr;
    mfem::BasicAdvDiffCoefficient     * MicroModelCoeff = nullptr;
    mfem::ParGridFunction             * designGF        = nullptr;

};

class ThermalComplianceIntegrator_1 : public LinearFormIntegrator
{
public:

    ThermalComplianceIntegrator_1()
    {};

    void SetFieldsAndMicrostructure(
        BasicAdvDiffCoefficient * MicroModelCoeff_,
        ParGridFunction * tempfield_,
        ParGridFunction * preassure_,
        ParGridFunction * desfield_)
    {
        MicroModelCoeff=MicroModelCoeff_;
        tempGF         =tempfield_;
        preassureGF    =preassure_;
        designGF       =desfield_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun)
    {
        {
            mfem::mfem_error("ThermalComplianceIntegrator_1::AssembleElementGrad is not defined!");
        }
        return 0.0;
    }

    virtual
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat)
    {
        {
            mfem::mfem_error("ThermalComplianceIntegrator_1::AssembleElementGrad is not defined!");
        }
    }

private:
    mfem::ParGridFunction             * tempGF          = nullptr;
    mfem::ParGridFunction             * preassureGF     = nullptr;
    mfem::BasicAdvDiffCoefficient     * MicroModelCoeff = nullptr;
    mfem::ParGridFunction             * designGF        = nullptr;

};

class MeanTempIntegrator : public NonlinearFormIntegrator
{
public:

    MeanTempIntegrator()
    {};

    void SetFieldsAndMicrostructure(
        BasicAdvDiffCoefficient * MicroModelCoeff_,
        ParGridFunction * tempfield_,
        ParGridFunction * preassure_,
        ParGridFunction * desfield_)
    {
        MicroModelCoeff=MicroModelCoeff_;
        tempGF         =tempfield_;
        preassureGF    =preassure_;
        designGF       =desfield_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun);

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat)
    {
        {
            mfem::mfem_error("ThermalComplianceIntegrator::AssembleElementGrad is not defined!");
        }
    }

private:
    mfem::ParGridFunction             * tempGF          = nullptr;
    mfem::ParGridFunction             * preassureGF     = nullptr;
    mfem::BasicAdvDiffCoefficient     * MicroModelCoeff = nullptr;
    mfem::ParGridFunction             * designGF        = nullptr;

};

class MeanTempIntegrator_1 : public LinearFormIntegrator
{
public:

    MeanTempIntegrator_1()
    {};

    void SetFieldsAndMicrostructure(
        BasicAdvDiffCoefficient * MicroModelCoeff_,
        ParGridFunction * tempfield_,
        ParGridFunction * preassure_,
        ParGridFunction * desfield_)
    {
        MicroModelCoeff=MicroModelCoeff_;
        tempGF         =tempfield_;
        preassureGF    =preassure_;
        designGF       =desfield_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun)
    {
        {
            mfem::mfem_error("ThermalComplianceIntegrator_1::GetElementEnergy is not defined!");
        }
        return 0.0;
    }

    virtual
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat)
    {
        {
            mfem::mfem_error("ThermalComplianceIntegrator_1::AssembleElementGrad is not defined!");
        }
    }

private:
    mfem::ParGridFunction             * tempGF          = nullptr;
    mfem::ParGridFunction             * preassureGF     = nullptr;
    mfem::BasicAdvDiffCoefficient     * MicroModelCoeff = nullptr;
    mfem::ParGridFunction             * designGF        = nullptr;

};

class AdvDiffAdjointPostIntegrator : public LinearFormIntegrator
{
public:

    AdvDiffAdjointPostIntegrator()
    {};

    void SetAdjoint(ParGridFunction* Adjoint_)
    {
        AdjointGF=Adjoint_;
    };

    void SetCoeffsAndGF(
        BasicNLDiffusionCoefficient * MicroModelCoeff_,
        BasicAdvDiffCoefficient * AdvDiffMicroModelCoeff_,
        mfem::ParGridFunction * desfield_ ,
        mfem::ParGridFunction * preassure_,
        mfem::ParGridFunction * temp_ )
    {
        MicroModelCoeff        =MicroModelCoeff_;
        AdvDiffMicroModelCoeff = AdvDiffMicroModelCoeff_;
        desfield               =desfield_;
        preassureGF            =preassure_;
        tempGF                = temp_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun)
    {
        {
            mfem::mfem_error("ADVDiffAdjointPostIntegrator::GetElementEnergy is not defined!");
        }
    }

    virtual
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat)
    {
        {
            mfem::mfem_error("ADVDiffAdjointPostIntegrator::AssembleElementGrad is not defined!");
        }
    }

private:
    mfem::ParGridFunction             * AdjointGF       = nullptr;
    mfem::BasicNLDiffusionCoefficient * MicroModelCoeff = nullptr;
    mfem::BasicAdvDiffCoefficient     * AdvDiffMicroModelCoeff = nullptr;
    mfem::ParGridFunction             * desfield        = nullptr;
    mfem::ParGridFunction             * preassureGF     = nullptr;
    mfem::ParGridFunction             * tempGF          = nullptr;

};

class AdjointDResidualDGradPPostIntegrator : public LinearFormIntegrator
{
public:

    AdjointDResidualDGradPPostIntegrator()
    {};

    void SetAdjoint(ParGridFunction* Adjoint_)
    {
        AdjointGF=Adjoint_;
    };

    void SetCoeffsAndGF(
        BasicNLDiffusionCoefficient * MicroModelCoeff_,
        BasicAdvDiffCoefficient * AdvDiffMicroModelCoeff_,
        mfem::ParGridFunction * desfield_ ,
        mfem::ParGridFunction * preassure_,
        mfem::ParGridFunction * temp_ )
    {
        MicroModelCoeff        =MicroModelCoeff_;
        AdvDiffMicroModelCoeff = AdvDiffMicroModelCoeff_;
        desfield               =desfield_;
        preassureGF            =preassure_;
        tempGF                = temp_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun)
    {
        {
            mfem::mfem_error("ADVDiffAdjointPostIntegrator::GetElementEnergy is not defined!");
        }
    }

    virtual
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat)
    {
        {
            mfem::mfem_error("ADVDiffAdjointPostIntegrator::AssembleElementGrad is not defined!");
        }
    }

private:
    mfem::ParGridFunction             * AdjointGF       = nullptr;
    mfem::BasicNLDiffusionCoefficient * MicroModelCoeff = nullptr;
    mfem::BasicAdvDiffCoefficient     * AdvDiffMicroModelCoeff = nullptr;
    mfem::ParGridFunction             * desfield        = nullptr;
    mfem::ParGridFunction             * preassureGF     = nullptr;
    mfem::ParGridFunction             * tempGF          = nullptr;

};

class ThermalComplianceQoI
{
public:
    ThermalComplianceQoI()
    {};

    ~ThermalComplianceQoI()
    { delete nf;};

    void SetFESpaceFieldsAndMicromodel(
        ParFiniteElementSpace       * fes,
        ParGridFunction             * preassureGF_,
        ParGridFunction             * designGF_,
        ParGridFunction             * tempGF_,
        BasicNLDiffusionCoefficient * MicroModelCoeff_,
        BasicAdvDiffCoefficient     * AdvDiffMicroModelCoeff_)
    {
        dfes                   = fes;
        preassureGF            = preassureGF_;
        designGF               = designGF_;
        tempGF                 = tempGF_;
        MicroModelCoeff        = MicroModelCoeff_;
        AdvDiffMicroModelCoeff = AdvDiffMicroModelCoeff_;
    };


    double Eval();

    void Grad(Vector& grad);


private:

    NLDiffusion                 * NLDiffSolver    = nullptr;
    ParFiniteElementSpace       * dfes            = nullptr;
    ParNonlinearForm            * nf              = nullptr;
    ThermalComplianceIntegrator * intgr           = nullptr;
    BasicNLDiffusionCoefficient * MicroModelCoeff = nullptr;
    BasicAdvDiffCoefficient     * AdvDiffMicroModelCoeff = nullptr;
    ParGridFunction             * preassureGF     = nullptr;
    ParGridFunction             * designGF        = nullptr;
    ParGridFunction             * tempGF          = nullptr;
};

class MeanTempQoI
{
public:
    MeanTempQoI()
    {};

    ~MeanTempQoI()
    { delete nf;};

    void SetFESpaceFieldsAndMicromodel(
        ParFiniteElementSpace       * fes,
        ParGridFunction             * preassureGF_,
        ParGridFunction             * designGF_,
        ParGridFunction             * tempGF_,
        BasicNLDiffusionCoefficient * MicroModelCoeff_,
        BasicAdvDiffCoefficient     * AdvDiffMicroModelCoeff_)
    {
        dfes                   = fes;
        preassureGF            = preassureGF_;
        designGF               = designGF_;
        tempGF                 = tempGF_;
        MicroModelCoeff        = MicroModelCoeff_;
        AdvDiffMicroModelCoeff = AdvDiffMicroModelCoeff_;
    };


    double Eval();

    void Grad(Vector& grad);


private:

    NLDiffusion                 * NLDiffSolver    = nullptr;
    ParFiniteElementSpace       * dfes            = nullptr;
    ParNonlinearForm            * nf              = nullptr;
    MeanTempIntegrator          * intgr           = nullptr;
    BasicNLDiffusionCoefficient * MicroModelCoeff = nullptr;
    BasicAdvDiffCoefficient     * AdvDiffMicroModelCoeff = nullptr;
    ParGridFunction             * preassureGF     = nullptr;
    ParGridFunction             * designGF        = nullptr;
    ParGridFunction             * tempGF          = nullptr;
};

}



#endif

