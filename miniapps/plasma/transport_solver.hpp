// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TRANSPORT_SOLVER
#define MFEM_TRANSPORT_SOLVER

#include "../common/pfem_extras.hpp"
#include "plasma.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

namespace transport
{

/**
   Returns the mean Electron-Ion mean collision time in seconds (see
   equation 2.5e)
   Te is the electron temperature in eV
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
inline double tau_e(double Te, double zi, double ni, double lnLambda)
{
   // The factor of eV_ is included to convert Te from eV to Joules
   return 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(0.5 * me_kg_ * pow(Te * eV_, 3) / M_PI) /
          (lnLambda * pow(q_, 4) * zi * zi * ni);
}

/**
   Returns the mean Ion-Ion mean collision time in seconds (see equation 2.5i)
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
   lnLambda is the Coulomb Logarithm
*/
inline double tau_i(double mi, double zi, double ni, double Ti,
                    double lnLambda)
{
   // The factor of eV_ is included to convert Ti from eV to Joules
   return 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(mi * amu_ * pow(Ti * eV_, 3) / M_PI) /
          (lnLambda * pow(q_ * zi, 4) * ni);
}

/** Multispecies Electron-Ion Collision Time in seconds
 Te is the electron temperature in eV
 ns is the number of ion species
 ni is the density of ions (assuming ni=ne) in particles per meter^3
 zi is the charge number of the ion species
 lnLambda is the Coulomb Logarithm
*/
//double tau_e(double Te, int ns, double * ni, int * zi, double lnLambda);

/** Multispecies Ion-Ion Collision Time in seconds
   ma is the ion mass in a.m.u
   Ta is the ion temperature in eV
   ion is the ion species index for the desired collision time
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
//double tau_i(double ma, double Ta, int ion, int ns, double * ni, int * zi,
//             double lnLambda);

/**
  Thermal diffusion coefficient along B field for electrons
  Return value is in m^2/s.
   Te is the electron temperature in eV
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
*/
/*
inline double chi_e_para(double Te, int ns, double * ni, int * zi)
{
 // The factor of q_ is included to convert Te from eV to Joules
 return 3.16 * (q_ * Te / me_kg_) * tau_e(Te, ns, ni, zi, 17.0);
}
*/
/**
  Thermal diffusion coefficient perpendicular to B field for electrons
  Return value is in m^2/s.
*/
/*
inline double chi_e_perp()
{
 // The factor of q_ is included to convert Te from eV to Joules
 return 1.0;
}
*/
/**
  Thermal diffusion coefficient perpendicular to both B field and
  thermal gradient for electrons.
  Return value is in m^2/s.
   Te is the electron temperature in eV
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   z is the charge number of the ion species
*/
/*
inline double chi_e_cross()
{
 // The factor of q_ is included to convert Te from eV to Joules
 return 0.0;
}
*/
/**
  Thermal diffusion coefficient along B field for ions
  Return value is in m^2/s.
   ma is the ion mass in a.m.u.
   Ta is the ion temperature in eV
   ion is the ion species index for the desired coefficient
   ns is the number of ion species
   nb is the density of ions in particles per meter^3
   zb is the charge number of the ion species
*/
/*
inline double chi_i_para(double ma, double Ta,
                       int ion, int ns, double * nb, int * zb)
{
 // The factor of q_ is included to convert Ta from eV to Joules
 // The factor of u_ is included to convert ma from a.m.u to kg
 return 3.9 * (q_ * Ta / (ma * amu_ ) ) *
        tau_i(ma, Ta, ion, ns, nb, zb, 17.0);
}
*/
/**
  Thermal diffusion coefficient perpendicular to B field for ions
  Return value is in m^2/s.
*/
/*
inline double chi_i_perp()
{
 // The factor of q_ is included to convert Ti from eV to Joules
 // The factor of u_ is included to convert mi from a.m.u to kg
 return 1.0;
}
*/
/**
  Thermal diffusion coefficient perpendicular to both B field and
  thermal gradient for ions
  Return value is in m^2/s.
*/
/*
inline double chi_i_cross()
{
 // The factor of q_ is included to convert Ti from eV to Joules
 // The factor of u_ is included to convert mi from a.m.u to kg
 return 0.0;
}
*/
/**
  Viscosity coefficient along B field for electrons
  Return value is in (a.m.u)*m^2/s.
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
*/
/*
inline double eta_e_para(double ne, double Te, int ns, double * ni, int * zi)
{
 // The factor of q_ is included to convert Te from eV to Joules
 // The factor of u_ is included to convert from kg to a.m.u
 return 0.73 * ne * (q_ * Te / amu_) * tau_e(Te, ns, ni, zi, 17.0);
}
*/
/**
  Viscosity coefficient along B field for ions
  Return value is in (a.m.u)*m^2/s.
   ma is the ion mass in a.m.u.
   Ta is the ion temperature in eV
   ion is the ion species index for the desired coefficient
   ns is the number of ion species
   nb is the density of ions in particles per meter^3
   zb is the charge number of the ion species
*/
/*
inline double eta_i_para(double ma, double Ta,
                       int ion, int ns, double * nb, int * zb)
{
 // The factor of q_ is included to convert Ti from eV to Joules
 // The factor of u_ is included to convert from kg to a.m.u
 return 0.96 * nb[ion] * (q_ * Ta / amu_) *
        tau_i(ma, Ta, ion, ns, nb, zb, 17.0);
}
*/

class StateVariableFunc
{
public:
   enum DerivType {INVALID, NEUTRAL_DENSITY,
                   ION_DENSITY, ION_PARA_VELOCITY, ION_TEMPERATURE,
                   ELECTRON_TEMPERATURE
                  };

   virtual bool NonTrivialValue(DerivType deriv) const = 0;

   void SetDerivType(DerivType deriv) { derivType_ = deriv; }
   DerivType GetDerivType() const { return derivType_; }

protected:
   StateVariableFunc(DerivType deriv = INVALID) : derivType_(deriv) {}

   DerivType derivType_;
};


class StateVariableCoef : public StateVariableFunc, public Coefficient
{
public:
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      switch (derivType_)
      {
         case INVALID:
            return Eval_Func(T, ip);
         case NEUTRAL_DENSITY:
            return Eval_dNn(T, ip);
         case ION_DENSITY:
            return Eval_dNi(T, ip);
         case ION_PARA_VELOCITY:
            return Eval_dVi(T, ip);
         case ION_TEMPERATURE:
            return Eval_dTi(T, ip);
         case ELECTRON_TEMPERATURE:
            return Eval_dTe(T, ip);
         default:
            return 0.0;
      }
   }

   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip) { return 0.0; }

   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip) { return 0.0; }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip) { return 0.0; }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip) { return 0.0; }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip) { return 0.0; }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip) { return 0.0; }

protected:
   StateVariableCoef(DerivType deriv = INVALID) : StateVariableFunc(deriv) {}
};

class StateVariableMatCoef : public StateVariableFunc, public MatrixCoefficient
{
public:
   virtual void Eval(DenseMatrix &M,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      M.SetSize(height, width);

      switch (derivType_)
      {
         case INVALID:
            return Eval_Func(M, T, ip);
         case NEUTRAL_DENSITY:
            return Eval_dNn(M, T, ip);
         case ION_DENSITY:
            return Eval_dNi(M, T, ip);
         case ION_PARA_VELOCITY:
            return Eval_dVi(M, T, ip);
         case ION_TEMPERATURE:
            return Eval_dTi(M, T, ip);
         case ELECTRON_TEMPERATURE:
            return Eval_dTe(M, T, ip);
         default:
            M = 0.0;
            return;
      }
   }

   virtual void Eval_Func(DenseMatrix &M,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dNn(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dNi(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dVi(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dTi(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dTe(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

protected:
   StateVariableMatCoef(int dim, DerivType deriv = INVALID)
      : StateVariableFunc(deriv), MatrixCoefficient(dim) {}

   StateVariableMatCoef(int h, int w, DerivType deriv = INVALID)
      : StateVariableFunc(deriv), MatrixCoefficient(h, w) {}
};

/** Given the electron temperature in eV this coefficient returns an
    approzximation to the expected ionization rate in m^3/s.
*/
class ApproxIonizationRate : public StateVariableCoef
{
private:
   Coefficient *TeCoef_;
   // GridFunction *Te_;

   // StateVariable derivType_;

public:
   // ApproxIonizationRate(GridFunction &Te) : Te_(&Te) {}
   ApproxIonizationRate(Coefficient &TeCoef)
      : TeCoef_(&TeCoef) {}

   bool NonTrivialValue(DerivType deriv) const
   {
      return (deriv == INVALID || deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Te2 = pow(TeCoef_->Eval(T, ip), 2);

      return 3.0e-16 * Te2 / (3.0 + 0.01 * Te2);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = TeCoef_->Eval(T, ip);

      return 2.0 * 3.0 * 3.0e-16 * Te / pow(3.0 + 0.01 * Te * Te, 2);
   }

};

class NeutralDiffusionCoef : public StateVariableCoef
{
private:
   ProductCoefficient * ne_;
   Coefficient        * vn_;
   StateVariableCoef  * iz_;

public:
   NeutralDiffusionCoef(ProductCoefficient &neCoef, Coefficient &vnCoef,
                        StateVariableCoef &izCoef)
      : ne_(&neCoef), vn_(&vnCoef), iz_(&izCoef) {}

   bool NonTrivialValue(DerivType deriv) const
   {
      return (deriv == INVALID || deriv == ION_DENSITY ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ne = ne_->Eval(T, ip);
      double vn = vn_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);

      return vn * vn / (3.0 * ne * iz);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ne = ne_->Eval(T, ip);
      double vn = vn_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);

      double dNe_dNi = ne_->GetAConst();

      return -vn * vn * dNe_dNi / (3.0 * ne * ne * iz);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ne = ne_->Eval(T, ip);
      double vn = vn_->Eval(T, ip);
      double iz = iz_->Eval_Func(T, ip);

      double diz_dTe = iz_->Eval_dTe(T, ip);

      return -vn * vn * diz_dTe / (3.0 * ne * iz * iz);
   }

};

class IonDiffusionCoef : public StateVariableMatCoef
{
private:
   Coefficient       * Dperp_;
   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   IonDiffusionCoef(Coefficient &DperpCoef, VectorCoefficient &B3Coef)
      : StateVariableMatCoef(2), Dperp_(&DperpCoef), B3_(&B3Coef), B_(3) {}

   bool NonTrivialValue(DerivType deriv) const
   {
      return (deriv == INVALID);
   }

   void Eval_Func(DenseMatrix & M,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      M.SetSize(2);

      double Dperp = Dperp_->Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;

      M(0,0) = (B_[1] * B_[1] + B_[2] * B_[2]) * Dperp / Bmag2;
      M(0,1) = -B_[0] * B_[1] * Dperp / Bmag2;
      M(1,0) = M(0,1);
      M(1,1) = (B_[0] * B_[0] + B_[2] * B_[2]) * Dperp / Bmag2;
   }

};

struct DGParams
{
   double sigma;
   double kappa;
};

class ParGridFunctionArray : public Array<ParGridFunction*>
{
public:
   ParGridFunctionArray() {}
   ParGridFunctionArray(int size) : Array<ParGridFunction*>(size) {}
   ParGridFunctionArray(int size, ParFiniteElementSpace *pf)
      : Array<ParGridFunction*>(size)
   {
      for (int i=0; i<size; i++)
      {
         data[i] = new ParGridFunction(pf);
      }
   }

   void ProjectCoefficient(Array<Coefficient*> &coeff)
   {
      for (int i=0; i<size; i++)
      {
         if (coeff[i] != NULL)
         {
            data[i]->ProjectCoefficient(*coeff[i]);
         }
         else
         {
            *data[i] = 0.0;
         }
      }
   }

   void Update()
   {
      for (int i=0; i<size; i++)
      {
         data[i]->Update();
      }
   }

   void ExchangeFaceNbrData()
   {
      for (int i=0; i<size; i++)
      {
         data[i]->ExchangeFaceNbrData();
      }
   }
};

class DGAdvectionDiffusionTDO : public TimeDependentOperator
{
private:
   DGParams & dg_;

   bool imex_;
   int logging_;
   std::string log_prefix_;
   double dt_;

   ParFiniteElementSpace *fes_;
   ParGridFunctionArray  *pgf_;

   Coefficient       *CCoef_;    // Scalar coefficient in front of du/dt
   VectorCoefficient *VCoef_;    // Velocity coefficient
   Coefficient       *dCoef_;    // Scalar diffusion coefficient
   MatrixCoefficient *DCoef_;    // Tensor diffusion coefficient
   Coefficient       *SCoef_;    // Source coefficient

   ScalarVectorProductCoefficient *negVCoef_;   // -1  * VCoef
   ScalarVectorProductCoefficient *dtNegVCoef_; // -dt * VCoef
   ProductCoefficient             *dtdCoef_;    //  dt * dCoef
   ScalarMatrixProductCoefficient *dtDCoef_;    //  dt * DCoef

   Array<int>   dbcAttr_;
   Coefficient *dbcCoef_; // Dirichlet BC coefficient

   Array<int>   nbcAttr_;
   Coefficient *nbcCoef_; // Neumann BC coefficient

   ParBilinearForm  m_;
   ParBilinearForm *a_;
   ParBilinearForm *b_;
   ParBilinearForm *s_;
   ParBilinearForm *k_;
   ParLinearForm   *q_exp_;
   ParLinearForm   *q_imp_;

   HypreParMatrix * M_;
   HypreSmoother M_prec_;
   CGSolver M_solver_;

   // HypreParMatrix * B_;
   // HypreParMatrix * S_;

   mutable ParLinearForm rhs_;
   mutable Vector RHS_;
   mutable Vector X_;

   void initM();
   void initA();
   void initB();
   void initS();
   void initK();
   void initQ();

public:
   DGAdvectionDiffusionTDO(DGParams & dg,
                           ParFiniteElementSpace &fes,
                           ParGridFunctionArray &pgf,
                           Coefficient &CCoef, bool imex = true);

   ~DGAdvectionDiffusionTDO();

   void SetTime(const double _t);

   void SetLogging(int logging, const std::string & prefix = "");

   void SetAdvectionCoefficient(VectorCoefficient &VCoef);
   void SetDiffusionCoefficient(Coefficient &dCoef);
   void SetDiffusionCoefficient(MatrixCoefficient &DCoef);
   void SetSourceCoefficient(Coefficient &SCoef);

   void SetDirichletBC(Array<int> &dbc_attr, Coefficient &dbc);
   void SetNeumannBC(Array<int> &nbc_attr, Coefficient &nbc);

   virtual void ExplicitMult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const double dt, const Vector &u, Vector &dudt);

   void Update();
};

class TransportPrec : public BlockDiagonalPreconditioner
{
private:
   Array<Operator*> diag_prec_;

public:
   TransportPrec(const Array<int> &offsets);
   ~TransportPrec();

   virtual void SetOperator(const Operator &op);
};

class DGTransportTDO : public TimeDependentOperator
{
private:
   int MyRank_;
   int logging_;

   ParFiniteElementSpace *fes_;
   ParFiniteElementSpace *ffes_;
   ParGridFunctionArray  *pgf_;
   ParGridFunctionArray  *dpgf_;

   Array<int> &offsets_;

   // HypreSmoother newton_op_prec_;
   // Array<HypreSmoother*> newton_op_prec_blocks_;
   TransportPrec newton_op_prec_;
   GMRESSolver  newton_op_solver_;
   NewtonSolver  newton_solver_;

   class NLOperator : public Operator
   {
   protected:
      DGParams & dg_;

      int logging_;
      std::string log_prefix_;

      int index_;
      double dt_;
      ParFiniteElementSpace *fes_;
      ParMesh               *pmesh_;
      ParGridFunctionArray  *pgf_;
      ParGridFunctionArray  *dpgf_;

      // mutable Vector shape_;
      // mutable DenseMatrix dshape_;
      // mutable DenseMatrix dshapedxt_;
      mutable Array<int> vdofs_;
      mutable Array<int> vdofs2_;
      mutable DenseMatrix elmat_;
      mutable DenseMatrix elmat_k_;
      mutable Vector elvec_;
      mutable Vector locvec_;
      mutable Vector locdvec_;
      // mutable Vector vec_;

      Array<BilinearFormIntegrator*> dbfi_m_;  // Domain Integrators
      Array<BilinearFormIntegrator*> dbfi_;  // Domain Integrators
      Array<BilinearFormIntegrator*> fbfi_;  // Interior Face Integrators
      Array<BilinearFormIntegrator*> bfbfi_; // Boundary Face Integrators
      Array<Array<int>*>             bfbfi_marker_; ///< Entries are not owned.

      Array<LinearFormIntegrator*> dlfi_;  // Domain Integrators

      Array<ParBilinearForm*> blf_; // Bilinear Form Objects for Gradients

      NLOperator(DGParams & dg, int index,
                 ParGridFunctionArray & pgf,
                 ParGridFunctionArray & dpgf);

   public:

      virtual ~NLOperator();

      void SetLogging(int logging, const std::string & prefix = "");

      virtual void SetTimeStep(double dt) { dt_ = dt; }

      virtual void Mult(const Vector &k, Vector &y) const;

      virtual void Update();
      virtual Operator *GetGradientBlock(int i);
   };

   /** The NeutralDensityOp is an mfem::Operator designed to work with
       a NewtonSolver as one row in a block system of non-linear
       transport equations.  Specifically, this operator models the
       mass conservation equation for a neutral species.

          d n_n / dt = Div(D_n Grad(n_n)) + S_n

       Where the diffusion constant D_n is a function of n_e and T_e
       (the electron density and temperature respectively) and the
       source term S_n is a function of n_e, T_e, and n_n.  Note that n_e is
       not a state variable but is related to n_i by the simple relation
          n_e = z_i n_i
       where z_i is the charge of the ions and n_i is the ion density.

       To advance this equation in time we need to find k_nn = d n_n / dt
       which satisfies:
          k_nn - Div(D_n(n_e, T_e) Grad(n_n + dt k_nn))
               - S_n(n_e, T_e, n_n + dt k_nn) = 0
       Where n_e and T_e are also evaluated at the next time step.  This is
       done with a Newton solver which needs the Jacobian of this block of
       equations.

       The diagonal block is given by:
          1 - dt Div(D_n Grad) - dt d S_n / d n_n

       The other non-trivial blocks are:
          - dt Div(d D_n / d n_i Grad(n_n)) - dt d S_n / d n_i
          - dt Div(d D_n / d T_e Grad(n_n)) - dt d S_n / d T_e

       The blocks of the Jacobian will be assembled finite element matrices.
       For the diagonal block we need a mass integrator with coefficient
       (1 - dt d S_n / d n_n), and a set of integrators to model the DG
       diffusion operator with coefficient (dt D_n).

       The off-diagonal blocks will consist of a mass integrator with
       coefficient (-dt d S_n / d n_i) or (-dt d S_n / d T_e).
    */
   class NeutralDensityOp : public NLOperator
   {
   private:
      int    z_i_;
      double m_n_;
      double T_n_;

      mutable GridFunctionCoefficient nn0Coef_;
      mutable GridFunctionCoefficient ni0Coef_;
      mutable GridFunctionCoefficient Te0Coef_;

      GridFunctionCoefficient dnnCoef_;
      GridFunctionCoefficient dniCoef_;
      GridFunctionCoefficient dTeCoef_;

      mutable SumCoefficient  nn1Coef_;
      mutable SumCoefficient  ni1Coef_;
      mutable SumCoefficient  Te1Coef_;

      ProductCoefficient      ne0Coef_;
      ProductCoefficient      ne1Coef_;
      ConstantCoefficient      vnCoef_;
      ApproxIonizationRate     izCoef_;

      NeutralDiffusionCoef      DCoef_;
      ProductCoefficient      dtDCoef_;

      ProductCoefficient     nnizCoef_; // nn * iz
      ProductCoefficient     neizCoef_; // ne * iz
      ProductCoefficient dtdSndnnCoef_; // - dt * dSn/dnn
      ProductCoefficient dtdSndniCoef_; // - dt * dSn/dni

      // mutable DiffusionIntegrator   diff_;
      // mutable DGDiffusionIntegrator dg_diff_;

   public:
      NeutralDensityOp(DGParams & dg,
                       ParGridFunctionArray & pgf, ParGridFunctionArray & dpgf,
                       int ion_charge, double neutral_mass,
                       double neutral_temp);

      virtual void SetTimeStep(double dt);

      void Update();

      // void Mult(const Vector &k, Vector &y) const;

      // Operator *GetGradientBlock(int i);
   };

   class IonDensityOp : public NLOperator
   {
   private:
      int    z_i_;
      double DPerpConst_;

      GridFunctionCoefficient nn0Coef_;
      GridFunctionCoefficient ni0Coef_;
      GridFunctionCoefficient Te0Coef_;

      GridFunctionCoefficient dnnCoef_;
      GridFunctionCoefficient dniCoef_;
      GridFunctionCoefficient dTeCoef_;

      mutable SumCoefficient  nn1Coef_;
      mutable SumCoefficient  ni1Coef_;
      mutable SumCoefficient  Te1Coef_;

      ApproxIonizationRate     izCoef_;

      ConstantCoefficient DPerpCoef_;
      MatrixCoefficient * PerpCoef_;
      ScalarMatrixProductCoefficient DCoef_;
      ScalarMatrixProductCoefficient dtDCoef_;

      ProductCoefficient nnizCoef_;
      ProductCoefficient niizCoef_;
      ProductCoefficient dtdSndnnCoef_;
      ProductCoefficient dtdSndniCoef_;

      // mutable DiffusionIntegrator   diff_;
      // mutable DGDiffusionIntegrator dg_diff_;

   public:
      IonDensityOp(DGParams & dg,
                   ParGridFunctionArray & pgf, ParGridFunctionArray & dpgf,
                   int ion_charge, double DPerp, MatrixCoefficient & PerpCoef);

      virtual void SetTimeStep(double dt);

      void Update();

      // void Mult(const Vector &k, Vector &y) const;

      // Operator *GetGradientBlock(int i);
   };

   class DummyOp : public NLOperator
   {
   public:
      DummyOp(DGParams & dg,
              ParGridFunctionArray & pgf,
              ParGridFunctionArray & dpgf, int index);

      virtual void SetTimeStep(double dt)
      {
         std::cout << "Setting time step: " << dt << " in DummyOp" << std::endl;
         NLOperator::SetTimeStep(dt);
      }

      void Update();

      // void Mult(const Vector &k, Vector &y) const;
   };

   class CombinedOp : public Operator
   {
   private:
      int neq_;
      int MyRank_;
      int logging_;

      ParFiniteElementSpace *fes_;
      ParGridFunctionArray  *pgf_;
      ParGridFunctionArray  *dpgf_;
      /*
       NeutralDensityOp n_n_op_;
       IonDensityOp     n_i_op_;
       DummyOp          v_i_op_;
       DummyOp          t_i_op_;
       DummyOp          t_e_op_;
      */
      Array<NLOperator*> op_;

      Array<int> & offsets_;
      mutable BlockOperator *grad_;

      void updateOffsets();

   public:
      CombinedOp(DGParams & dg,
                 ParGridFunctionArray & pgf, ParGridFunctionArray & dpgf,
                 Array<int> & offsets,
                 int ion_charge, double neutral_mass, double neutral_temp,
                 double DiPerp, MatrixCoefficient & PerpCoef,
                 unsigned int op_flag = 31, int logging = 0);

      ~CombinedOp();

      void SetTimeStep(double dt);
      void SetLogging(int logging);

      void Update();

      void Mult(const Vector &k, Vector &y) const;

      void UpdateGradient(const Vector &x) const;

      Operator &GetGradient(const Vector &x) const
      { UpdateGradient(x); return *grad_; }
   };

   CombinedOp op_;

   // ConstantCoefficient oneCoef_;

   // DGAdvectionDiffusionTDO n_n_oper_; // Neutral Density
   // DGAdvectionDiffusionTDO n_i_oper_; // Ion Density
   // DGAdvectionDiffusionTDO v_i_oper_; // Ion Parallel Velocity
   // DGAdvectionDiffusionTDO T_i_oper_; // Ion Temperature
   // DGAdvectionDiffusionTDO T_e_oper_; // Electron Temperature

   mutable Vector x_;
   mutable Vector y_;
   Vector u_;
   Vector dudt_;

public:
   DGTransportTDO(DGParams & dg,
                  ParFiniteElementSpace &fes,
                  ParFiniteElementSpace &ffes,
                  Array<int> &offsets,
                  ParGridFunctionArray &pgf,
                  ParGridFunctionArray &dpgf,
                  int ion_charge,
                  double neutral_mass,
                  double neutral_temp,
                  double Di_perp,
                  MatrixCoefficient & perpCoef,
                  Coefficient &MomCCoef,
                  Coefficient &TiCCoef,
                  Coefficient &TeCCoef,
                  bool imex = true,
                  unsigned int op_flag = 31,
                  int logging = 0);

   ~DGTransportTDO();

   void SetTime(const double _t);
   void SetLogging(int logging);
   /*
    void SetNnDiffusionCoefficient(Coefficient &dCoef);
    void SetNnDiffusionCoefficient(MatrixCoefficient &DCoef);
    void SetNnSourceCoefficient(Coefficient &SCoef);

    void SetNnDirichletBC(Array<int> &dbc_attr, Coefficient &dbc);
    void SetNnNeumannBC(Array<int> &nbc_attr, Coefficient &nbc);

    void SetNiAdvectionCoefficient(VectorCoefficient &VCoef);
    void SetNiDiffusionCoefficient(Coefficient &dCoef);
    void SetNiDiffusionCoefficient(MatrixCoefficient &DCoef);
    void SetNiSourceCoefficient(Coefficient &SCoef);

    void SetNiDirichletBC(Array<int> &dbc_attr, Coefficient &dbc);
    void SetNiNeumannBC(Array<int> &nbc_attr, Coefficient &nbc);
   */
   /*
    void SetViAdvectionCoefficient(VectorCoefficient &VCoef);
    void SetViDiffusionCoefficient(Coefficient &dCoef);
    void SetViDiffusionCoefficient(MatrixCoefficient &DCoef);
    void SetViSourceCoefficient(Coefficient &SCoef);

    void SetViDirichletBC(Array<int> &dbc_attr, Coefficient &dbc);
    void SetViNeumannBC(Array<int> &nbc_attr, Coefficient &nbc);

    void SetTiAdvectionCoefficient(VectorCoefficient &VCoef);
    void SetTiDiffusionCoefficient(Coefficient &dCoef);
    void SetTiDiffusionCoefficient(MatrixCoefficient &DCoef);
    void SetTiSourceCoefficient(Coefficient &SCoef);

    void SetTiDirichletBC(Array<int> &dbc_attr, Coefficient &dbc);
    void SetTiNeumannBC(Array<int> &nbc_attr, Coefficient &nbc);

    void SetTeAdvectionCoefficient(VectorCoefficient &VCoef);
    void SetTeDiffusionCoefficient(Coefficient &dCoef);
    void SetTeDiffusionCoefficient(MatrixCoefficient &DCoef);
    void SetTeSourceCoefficient(Coefficient &SCoef);

    void SetTeDirichletBC(Array<int> &dbc_attr, Coefficient &dbc);
    void SetTeNeumannBC(Array<int> &nbc_attr, Coefficient &nbc);
   */
   // virtual void ExplicitMult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const double dt, const Vector &u, Vector &dudt);

   void Update();
};

class MultiSpeciesDiffusion;
class MultiSpeciesAdvection;

class TransportSolver : public ODESolver
{
private:
   ODESolver * impSolver_;
   ODESolver * expSolver_;

   ParFiniteElementSpace & sfes_; // Scalar fields
   ParFiniteElementSpace & vfes_; // Vector fields
   ParFiniteElementSpace & ffes_; // Full system

   BlockVector & nBV_;

   ParGridFunction & B_;

   Array<int> & charges_;
   Vector & masses_;

   MultiSpeciesDiffusion * msDiff_;

   void initDiffusion();

public:
   TransportSolver(ODESolver * implicitSolver, ODESolver * explicitSolver,
                   ParFiniteElementSpace & sfes,
                   ParFiniteElementSpace & vfes,
                   ParFiniteElementSpace & ffes,
                   BlockVector & nBV,
                   ParGridFunction & B,
                   Array<int> & charges,
                   Vector & masses);
   ~TransportSolver();

   void Update();

   void Step(Vector &x, double &t, double &dt);
};
/*
class ChiParaCoefficient : public Coefficient
{
private:
 BlockVector & nBV_;
 ParGridFunction nGF_;
 GridFunctionCoefficient nCoef_;
 GridFunctionCoefficient TCoef_;

 int ion_;
 Array<int> & z_;
 Vector     * m_;
 Vector       n_;

public:
 ChiParaCoefficient(BlockVector & nBV, Array<int> & charges);
 ChiParaCoefficient(BlockVector & nBV, int ion_species,
                    Array<int> & charges, Vector & masses);
 void SetT(ParGridFunction & T);

 double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiPerpCoefficient : public Coefficient
{
private:
 int ion_;

public:
 ChiPerpCoefficient(BlockVector & nBV, Array<int> & charges);
 ChiPerpCoefficient(BlockVector & nBV, int ion_species,
                    Array<int> & charges, Vector & masses);

 double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiCrossCoefficient : public Coefficient
{
private:
 int ion_;

public:
 ChiCrossCoefficient(BlockVector & nBV, Array<int> & charges);
 ChiCrossCoefficient(BlockVector & nBV, int ion_species,
                     Array<int> & charges, Vector & masses);

 double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiCoefficient : public MatrixCoefficient
{
private:
 ChiParaCoefficient  chiParaCoef_;
 ChiPerpCoefficient  chiPerpCoef_;
 ChiCrossCoefficient chiCrossCoef_;
 VectorGridFunctionCoefficient BCoef_;

 Vector bHat_;

public:
 ChiCoefficient(int dim, BlockVector & nBV, Array<int> & charges);
 ChiCoefficient(int dim, BlockVector & nBV, int ion_species,
                Array<int> & charges, Vector & masses);

 void SetT(ParGridFunction & T);
 void SetB(ParGridFunction & B);

 void Eval(DenseMatrix &K, ElementTransformation &T,
           const IntegrationPoint &ip);
};
*/
/*
class EtaParaCoefficient : public Coefficient
{
private:
 BlockVector & nBV_;
 ParGridFunction nGF_;
 GridFunctionCoefficient nCoef_;
 GridFunctionCoefficient TCoef_;

 int ion_;
 Array<int> & z_;
 Vector     * m_;
 Vector       n_;

public:
 EtaParaCoefficient(BlockVector & nBV, Array<int> & charges);
 EtaParaCoefficient(BlockVector & nBV, int ion_species,
                    Array<int> & charges, Vector & masses);

 void SetT(ParGridFunction & T);

 double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};
*/
class MultiSpeciesDiffusion : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &sfes_;
   ParFiniteElementSpace &vfes_;

   BlockVector & nBV_;

   Array<int> & charges_;
   Vector & masses_;

   void initCoefficients();
   void initBilinearForms();

public:
   MultiSpeciesDiffusion(ParFiniteElementSpace & sfes,
                         ParFiniteElementSpace & vfes,
                         BlockVector & nBV,
                         Array<int> & charges,
                         Vector & masses);

   ~MultiSpeciesDiffusion();

   void Assemble();

   void Update();

   void ImplicitSolve(const double dt, const Vector &x, Vector &y);
};


// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form for the diffusion term. (modified from ex14p)
class DiffusionTDO : public TimeDependentOperator
{
private:
   const int dim_;
   double dt_;
   double dg_sigma_;
   double dg_kappa_;

   ParFiniteElementSpace &fes_;
   ParFiniteElementSpace &dfes_;
   ParFiniteElementSpace &vfes_;

   ParBilinearForm m_;
   ParBilinearForm d_;

   ParLinearForm rhs_;
   ParGridFunction x_;

   HypreParMatrix * M_;
   HypreParMatrix * D_;

   Vector RHS_;
   Vector X_;

   HypreSolver * solver_;
   HypreSolver * amg_;

   MatrixCoefficient &nuCoef_;
   ScalarMatrixProductCoefficient dtNuCoef_;

   void initSolver(double dt);

public:
   DiffusionTDO(ParFiniteElementSpace &fes,
                ParFiniteElementSpace &dfes,
                ParFiniteElementSpace &_vfes,
                MatrixCoefficient & nuCoef,
                double dg_sigma,
                double dg_kappa);

   // virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~DiffusionTDO() { }
};

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form for the advection term.
class AdvectionTDO : public TimeDependentOperator
{
private:
   const int dim_;
   const int num_equation_;
   const double specific_heat_ratio_;

   mutable double max_char_speed_;

   ParFiniteElementSpace &vfes_;
   Operator &A_;
   SparseMatrix &Aflux_;
   DenseTensor Me_inv_;

   mutable Vector state_;
   mutable DenseMatrix f_;
   mutable DenseTensor flux_;
   mutable Vector z_;

   void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;

public:
   AdvectionTDO(ParFiniteElementSpace &_vfes,
                Operator &A, SparseMatrix &Aflux, int num_equation,
                double specific_heat_ratio);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~AdvectionTDO() { }
};

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
   int num_equation_;
   double specific_heat_ratio_;
   Vector flux1_;
   Vector flux2_;

public:
   RiemannSolver(int num_equation, double specific_heat_ratio);
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
};


// Constant (in time) mixed bilinear form multiplying the flux grid function.
// The form is (vec(v), grad(w)) where the trial space = vector L2 space (mesh
// dim) and test space = scalar L2 space.
class DomainIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape_;
   DenseMatrix flux_;
   DenseMatrix dshapedr_;
   DenseMatrix dshapedx_;

public:
   DomainIntegrator(const int dim, const int num_equation);

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   int num_equation_;
   double max_char_speed_;
   RiemannSolver rsolver_;
   Vector shape1_;
   Vector shape2_;
   Vector funval1_;
   Vector funval2_;
   Vector nor_;
   Vector fluxN_;
   IntegrationPoint eip1_;
   IntegrationPoint eip2_;

public:
   FaceIntegrator(RiemannSolver &rsolver_, const int dim,
                  const int num_equation);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

} // namespace transport

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_TRANSPORT_SOLVER
