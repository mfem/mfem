// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MTOPINTEGRATORS_HPP
#define MTOPINTEGRATORS_HPP

#include "mfem.hpp"
#include <map>

namespace mfem {


class BaseQFunction
{
public:
    virtual ~BaseQFunction(){}

    virtual std::string GetType()=0;

    //return the energy at a integration point
    virtual
    double QEnergy(ElementTransformation &T, const IntegrationPoint &ip,
                   mfem::Vector &dd, mfem::Vector &uu)
    {
        return 0.0;
    }

    //return the residual at a integration point
    virtual
    void QResidual(ElementTransformation &T, const IntegrationPoint &ip,
            mfem::Vector &dd, mfem::Vector &uu, mfem::Vector &rr)=0;

    //return the gradient of the redidual at a integration point
    virtual
    void QGradResidual(ElementTransformation &T, const IntegrationPoint &ip,
            mfem::Vector &dd, mfem::Vector &uu, mfem::DenseMatrix &hh)=0;

    //return the gradient of the residual with respect to the design parameters, multiplied by the adjoint
    virtual
    void AQResidual(ElementTransformation &T, const IntegrationPoint &ip,
                      mfem::Vector &dd, mfem::Vector &uu,
                      mfem::Vector &aa, mfem::Vector &rr)=0;

};


class QLinearDiffusion:public BaseQFunction
{
public:
    QLinearDiffusion(mfem::Coefficient& diffco, mfem::Coefficient& hsrco,
                     double pp=1.0, double minrho=1e-7, double betac=4.0, double etac=0.5):
                     diff(diffco),load(hsrco), powerc(pp), rhomin(minrho), beta(betac), eta(etac)
    {

    }

    virtual std::string GetType() override
    {
        return "QLinearDiffusion";
    }

    virtual
    double QEnergy(ElementTransformation &T, const IntegrationPoint &ip,
                   Vector &dd, Vector &uu) override
    {
        double di=diff.Eval(T,ip);
        double ll=load.Eval(T,ip);
        double rz=0.5+0.5*std::tanh(beta*(dd[0]-eta));
        double fd=di*(std::pow(rz,powerc)+rhomin);
        double rez = 0.5*(uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2])*fd-uu[3]*ll;
        return rez;
    }

    virtual
    void QResidual(ElementTransformation &T, const IntegrationPoint &ip,
                   Vector &dd, Vector &uu, Vector &rr) override
    {
        double di=diff.Eval(T,ip);
        double ll=load.Eval(T,ip);
        double rz=0.5+0.5*std::tanh(beta*(dd[0]-eta));
        double fd=di*(std::pow(rz,powerc)+rhomin);

        rr[0]=uu[0]*fd;
        rr[1]=uu[1]*fd;
        rr[2]=uu[2]*fd;
        rr[3]=-ll;
    }

    virtual
    void AQResidual(ElementTransformation &T, const IntegrationPoint &ip,
                   Vector &dd, Vector &uu, Vector &aa, Vector &rr) override
    {
        double di=diff.Eval(T,ip);
        double tt=std::tanh(beta*(dd[0]-eta));
        double rz=0.5+0.5*tt;
        double fd=di*powerc*std::pow(rz,powerc-1.0)*0.5*(1.0-tt*tt)*beta;

        rr[0] = -(aa[0]*uu[0]+aa[1]*uu[1]+aa[2]*uu[2])*fd;
    }

    virtual
    void QGradResidual(ElementTransformation &T, const IntegrationPoint &ip,
                       Vector &dd, Vector &uu, DenseMatrix &hh) override
    {
        double di=diff.Eval(T,ip);
        double tt=std::tanh(beta*(dd[0]-eta));
        double rz=0.5+0.5*tt;
        double fd=di*(std::pow(rz,powerc)+rhomin);
        hh=0.0;

        hh(0,0)=fd;
        hh(1,1)=fd;
        hh(2,2)=fd;
        hh(3,3)=0.0;
    }

private:
    mfem::Coefficient& diff; //diffusion coefficient
    mfem::Coefficient& load; //load coeficient
    double powerc;
    double rhomin;
    double beta;
    double eta;
};

class ParametricLinearDiffusion: public ParametricBNLFormIntegrator
{
public:
    ParametricLinearDiffusion(BaseQFunction& qfunm): qfun(qfunm)
    {

    }

    /// Compute the local energy
    virtual
    double GetElementEnergy(const Array<const FiniteElement *> &el,
                            const Array<const FiniteElement *> &pel,
                            ElementTransformation &Tr,
                            const Array<const Vector *> &elfun,
                            const Array<const Vector *> &pelfun) override;

    virtual
    void AssembleElementVector(const Array<const FiniteElement *> &el,
                               const Array<const FiniteElement *> &pel,
                               ElementTransformation &Tr,
                               const Array<const Vector *> &elfun,
                               const Array<const Vector *> &pelfun,
                               const Array<Vector *> &elvec) override;

    virtual
    void AssembleElementGrad(const Array<const FiniteElement *> &el,
                             const Array<const FiniteElement *> &pel,
                             ElementTransformation &Tr,
                             const Array<const Vector *> &elfun,
                             const Array<const Vector *> &pelfun,
                             const Array2D<DenseMatrix *> &elmats) override;

    virtual
    void AssemblePrmElementVector(const Array<const FiniteElement *> &el,
                                  const Array<const FiniteElement *> &pel,
                                  ElementTransformation &Tr,
                                  const Array<const Vector *> &elfun,
                                  const Array<const Vector *> &alfun,
                                  const Array<const Vector *> &pelfun,
                                  const Array<Vector *> &elvec) override;
private:
    BaseQFunction& qfun;
};


// Computes an example of nonlinear objective
// \int(field*field*weight)d\Omega_e
class DiffusionObjIntegrator:public BlockNonlinearFormIntegrator
{
public:

    DiffusionObjIntegrator()
    {

    }

    virtual
    double GetElementEnergy(const Array<const FiniteElement *> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector *> &elfun) override;

    virtual
    void AssembleElementVector(const Array<const FiniteElement *> &el,
                               ElementTransformation &Tr,
                               const Array<const Vector *> &elfun,
                               const Array<Vector *> &elvec) override;
};

}

#endif
