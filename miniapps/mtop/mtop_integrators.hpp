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

/* Base class for representing function at integration points.
 * */
class BaseQFunction
{
public:
    virtual ~BaseQFunction(){}

    /// Returns a user defined string identifying the function.
    virtual std::string GetType()=0;

    // Returns the energy at an integration point.
    virtual
    double QEnergy(ElementTransformation &T, const IntegrationPoint &ip,
                   mfem::Vector &dd, mfem::Vector &uu)
    {
        return 0.0;
    }

    // Returns the residual at an integration point.
    virtual
    void QResidual(ElementTransformation &T, const IntegrationPoint &ip,
            mfem::Vector &dd, mfem::Vector &uu, mfem::Vector &rr)=0;

    /// Returns the gradient of the redidual at a integration point.
    virtual
    void QGradResidual(ElementTransformation &T, const IntegrationPoint &ip,
            mfem::Vector &dd, mfem::Vector &uu, mfem::DenseMatrix &hh)=0;

    /// Returns the gradient of the residual with respect to the design
    /// parameters, multiplied by the adjoint.
    virtual
    void AQResidual(ElementTransformation &T, const IntegrationPoint &ip,
                      mfem::Vector &dd, mfem::Vector &uu,
                      mfem::Vector &aa, mfem::Vector &rr)=0;

};

/* QLinearDiffusion implements methods for computing the energy, the residual,
 * gradient of the residual and the product of the adjoint fields with the derivative
 * of the residual with respect to the parameters. All computations are performed
 * at a integration point. Therefore the vectors (vv,uu,aa,rr ..) hold the fields'
 * values and the fields' derivatives at the integration point. For example for a
 *  single scalar parametric field representing the density in topology optimization
 *  the vector dd will have size one and the element will be the density at the
 * integration point. The map between state and parameter is not fixed and depends
 * on the implementation of the QFunction class.
 * */
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
        // dd[0] - density
        // uu[0] - grad_x
        // uu[1] - grad_y
        // uu[2] - grad_z
        // uu[3] - temperature/scalar field

        double di=diff.Eval(T,ip);
        double ll=load.Eval(T,ip);
        //Computes the physical density using projection.
        double rz=0.5+0.5*std::tanh(beta*(dd[0]-eta)); //projection
        //Computes the diffusion coefficient at the integration point.
        double fd=di*(std::pow(rz,powerc)+rhomin);
        //Computes the sum of the energy and the product of the temperature and the
        //external input at the integration point.
        double rez = 0.5*(uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2])*fd-uu[3]*ll;
        return rez;
    }

    /// Returns the derivative of QEnergy with respect to the state vector uu.
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


    // Returns the derivative, with respect to the density, of the product
    // of the adjoint field with the residual at the integration point ip.
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

    // Returns the gradient of the residual with respect to the state vector
    // at the integration point ip.
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
    double powerc; //penalization coefficient
    double rhomin; //lower bound for the density
    double beta;   //controls the sharpness of the projection
    double eta;    //projection threshold for tanh
};

/// Provides implementation of an integrator for linear diffusion
/// with parametrization provided by a density field. The setup
/// is standard for topology optimization problems.
class ParametricLinearDiffusion: public ParametricBNLFormIntegrator
{
public:
    ParametricLinearDiffusion(BaseQFunction& qfunm): qfun(qfunm)
    {

    }

    /// Computes the local energy.
    virtual
    double GetElementEnergy(const Array<const FiniteElement *> &el,
                            const Array<const FiniteElement *> &pel,
                            ElementTransformation &Tr,
                            const Array<const Vector *> &elfun,
                            const Array<const Vector *> &pelfun) override;

    /// Computes the element's residual.
    virtual
    void AssembleElementVector(const Array<const FiniteElement *> &el,
                               const Array<const FiniteElement *> &pel,
                               ElementTransformation &Tr,
                               const Array<const Vector *> &elfun,
                               const Array<const Vector *> &pelfun,
                               const Array<Vector *> &elvec) override;

    /// Computes the stiffness/tangent matrix.
    virtual
    void AssembleElementGrad(const Array<const FiniteElement *> &el,
                             const Array<const FiniteElement *> &pel,
                             ElementTransformation &Tr,
                             const Array<const Vector *> &elfun,
                             const Array<const Vector *> &pelfun,
                             const Array2D<DenseMatrix *> &elmats) override;

    /// Computes the product of the adjoint solution and the derivative of
    /// the residual with respect to the parametric fields.
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


/// Computes an example of nonlinear objective
/// \int(field*field*weight)d\Omega_e.
class DiffusionObjIntegrator:public BlockNonlinearFormIntegrator
{
public:

    DiffusionObjIntegrator()
    {

    }

    /// Returns the objective contribution at element level.
    virtual
    double GetElementEnergy(const Array<const FiniteElement *> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector *> &elfun) override;

    /// Returns the gradient of the objective contribution at
    /// element level.
    virtual
    void AssembleElementVector(const Array<const FiniteElement *> &el,
                               ElementTransformation &Tr,
                               const Array<const Vector *> &elfun,
                               const Array<Vector *> &elvec) override;
};

}

#endif
