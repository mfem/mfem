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

namespace PointHeavisideProj
{

inline
double Project(double rho, double eta, double beta)
{
    // tanh projection - Wang&Lazarov&Sigmund2011
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double c=std::tanh(beta*(rho-eta));
    double rez=(a+c)/(a+b);
    return rez;
}

inline
double Grad(double rho, double eta, double beta)

{
    double c=std::tanh(beta*(rho-eta));
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double rez=beta*(1.0-c*c)/(a+b);
    return rez;
}

inline
double Hess(double rho,double eta, double beta)
{
    double c=std::tanh(beta*(rho-eta));
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double rez=-2.0*beta*beta*c*(1.0-c*c)/(a+b);
    return rez;
}

}

class QAdvectionDiffusionLSFEM:public BaseQFunction
{
public:
    QAdvectionDiffusionLSFEM(mfem::VectorCoefficient& difft, mfem::Coefficient& muco,
                        mfem::Coefficient& loadc, double pp=1.0, double minrho=1e-7,
                        double betac=4.0, double etac=0.5):dtensor(difft), muc(muco),
                        load(loadc), power_co(pp), rho_min(minrho), beta_co(betac),
                        eta_co(etac)
    {

    }

    virtual std::string GetType() override
    {
        return "QAdvectionDiffusionLSFEM";
    }

    //return the value of the Lagrangian at an integration point
    //the parameters are ordered as follows
    //dd[0] = density
    //dd[1] = velocity_x
    //dd[2] = velocity_y
    //dd[3] = velocity_z
    //the state variables are ordered as follows
    //uu[0] = u
    //uu[1] = grad_x(u)
    //uu[2] = grad_y(u)
    //uu[3] = grad_z(u)
    //uu[4] = flux_x [q_x]
    //uu[5] = flux_y [q_y]
    //uu[6] = flux_z [q_z]
    //uu[7] = div(q)
    virtual
    double QEnergy(ElementTransformation &T, const IntegrationPoint &ip,
                   Vector &dd, Vector &uu) override;

    virtual
    void QResidual(ElementTransformation &T, const IntegrationPoint &ip,
                   Vector &dd, Vector &uu, Vector &rr) override;

    virtual
    void AQResidual(ElementTransformation &T, const IntegrationPoint &ip,
                   Vector &dd, Vector &uu, Vector &aa, Vector &rr) override;

    virtual
    void QGradResidual(ElementTransformation &T, const IntegrationPoint &ip,
                       Vector &dd, Vector &uu, DenseMatrix &hh) override;

private:
    mfem::VectorCoefficient& dtensor;
    mfem::Coefficient& muc;
    mfem::Coefficient& load;
    double power_co;
    double rho_min;
    double beta_co;
    double eta_co;

};



class QAdvectionDiffusion:public BaseQFunction
{
public:
    QAdvectionDiffusion(mfem::VectorCoefficient& difft, mfem::Coefficient& muco,
                        mfem::Coefficient& loadc, double pp=1.0, double minrho=1e-7,
                        double betac=4.0, double etac=0.5):dtensor(difft), muc(muco),
                        load(loadc), power_co(pp), rho_min(minrho), beta_co(betac),
                        eta_co(etac)
    {

    }

    virtual std::string GetType() override
    {
        return "QAdvectionDiffusion";
    }

    //return the value of the Lagrangian at an integration point
    //the parameters are ordered as follows
    //dd[0] = density
    //dd[1] = velocity_x
    //dd[2] = velocity_y
    //dd[3] = velocity_z
    //the state variables are ordered as follows
    //uu[0] = u
    //uu[1] = grad_x(u)
    //uu[2] = grad_y(u)
    //uu[3] = grad_z(u)
    //uu[4] = flux_x [q_x]
    //uu[5] = flux_y [q_y]
    //uu[6] = flux_z [q_z]
    //uu[7] = div(q)
    //uu[8] = x (Lagrange multiplier)
    virtual
    double QEnergy(ElementTransformation &T, const IntegrationPoint &ip,
                   Vector &dd, Vector &uu) override;

    virtual
    void QResidual(ElementTransformation &T, const IntegrationPoint &ip,
                   Vector &dd, Vector &uu, Vector &rr) override;

    virtual
    void AQResidual(ElementTransformation &T, const IntegrationPoint &ip,
                   Vector &dd, Vector &uu, Vector &aa, Vector &rr) override;

    virtual
    void QGradResidual(ElementTransformation &T, const IntegrationPoint &ip,
                       Vector &dd, Vector &uu, DenseMatrix &hh) override;


private:
    mfem::VectorCoefficient& dtensor;
    mfem::Coefficient& muc;
    mfem::Coefficient& load;
    double power_co;
    double rho_min;
    double beta_co;
    double eta_co;

};


class ParametricAdvecDiffusLSFEM: public ParametricBNLFormIntegrator
{

public:

    ParametricAdvecDiffusLSFEM(BaseQFunction& qfunm): qfun(qfunm)
    {

    }

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


class ParametricAdvecDiffusIntegrator: public ParametricBNLFormIntegrator
{

public:
    ParametricAdvecDiffusIntegrator(BaseQFunction& qfunm): qfun(qfunm)
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

    DiffusionObjIntegrator(double p=2.0):ppc(p)
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

private:
    double ppc;
};


class ThermalComplianceIntegrator: public BlockNonlinearFormIntegrator
{
public:
     ThermalComplianceIntegrator(mfem::Coefficient& inp):load(inp)
     {}

     virtual
     double GetElementEnergy(const Array<const FiniteElement *> &el,
                             ElementTransformation &Tr,
                             const Array<const Vector *> &elfun) override;

     virtual
     void AssembleElementVector(const Array<const FiniteElement *> &el,
                                ElementTransformation &Tr,
                                const Array<const Vector *> &elfun,
                                const Array<Vector *> &elvec) override;

private:
     mfem::Coefficient& load;

};

// Formulation for the ScreenedPoisson equation integrator for filtering
// in topology optimization. The input is assumed to be between 0 and 1.
// The parameter rh is the radius of a linear cone filter which will
// deliver similar smoothing effect as the Screened Poisson equation.
// It determines the length scale of the smoothing.
class FScreenedPoisson: public NonlinearFormIntegrator
{
protected:
   double diffcoef;
   Coefficient *func;

public:
   FScreenedPoisson(Coefficient &nfunc, double rh):func(&nfunc)
   {
      double rd=rh/(2*std::sqrt(3.0));
      diffcoef= rd*rd;
   }

   ~FScreenedPoisson() { }

   void SetInput(Coefficient &nfunc) { func = &nfunc; }

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &trans,
                                   const Vector &elfun) override;

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect) override;

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat) override;
};


// Low-pass filter based on the Screened Poisson equation.
// B. S. Lazarov, O. Sigmund: "Filters in topology optimization based on
// Helmholtz-type differential equations", DOI:10.1002/nme.3072.
class PDEFilterTO
{
public:
   PDEFilterTO(ParMesh &mesh, double rh, int order = 2,
             int maxiter=100, double rtol=1e-7, double atol=1e-15, int print_lv=0)
      : rr(rh),
        fecp(order, mesh.Dimension()),
        fesp(&mesh, &fecp, 1),
        gf(&fesp)
   {
      sv = fesp.NewTrueDofVector();

      nf = new ParNonlinearForm(&fesp);
      prec = new HypreBoomerAMG();
      prec->SetPrintLevel(print_lv);

      gmres = new GMRESSolver(mesh.GetComm());

      gmres->SetAbsTol(atol);
      gmres->SetRelTol(rtol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(print_lv);
      gmres->SetPreconditioner(*prec);

      sint=nullptr;
   }

   ~PDEFilterTO()
   {
      delete gmres;
      delete prec;
      delete nf;
      delete sv;
   }

   void Filter(ParGridFunction &func, ParGridFunction &ffield)
   {
      GridFunctionCoefficient gfc(&func);
      Filter(gfc, ffield);
   }

   void Filter(Coefficient &func, ParGridFunction &ffield);

private:
   const double rr;
   H1_FECollection fecp;
   ParFiniteElementSpace fesp;
   ParGridFunction gf;

   ParNonlinearForm* nf;
   HypreBoomerAMG* prec;
   GMRESSolver *gmres;
   HypreParVector *sv;

   FScreenedPoisson* sint;
};




}

#endif
