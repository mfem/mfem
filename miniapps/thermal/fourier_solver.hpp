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

#ifndef MFEM_FOURIER_SOLVER
#define MFEM_FOURIER_SOLVER

#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

namespace thermal
{

struct DGParams
{
   double sigma;
   double kappa;
};

class AdvectionTDO : public TimeDependentOperator
{
public:
   AdvectionTDO(ParFiniteElementSpace &H1_FES, VectorCoefficient &velCoef);
   ~AdvectionTDO();

   /** @brief Perform the action of the operator: @a q = f(@a y, t), where
       q solves the algebraic equation F(@a y, q, t) = G(@a y, t) and t is the
       current time. */
   virtual void Mult(const Vector &y, Vector &q) const;

private:

   void initMult() const;

   ParFiniteElementSpace & H1_FESpace_;
   VectorCoefficient & velCoef_;

   Array<int> ess_bdr_tdofs_;

   mutable ParBilinearForm m1_;
   ParBilinearForm adv1_;

   mutable HypreParMatrix   M1_;
   mutable HyprePCG       * M1Inv_;
   mutable HypreDiagScale * M1Diag_;
   mutable ParGridFunction dydt_gf_;
   mutable Vector SOL_;
   mutable Vector RHS_;
   mutable Vector rhs_;
};

/**
   The thermal diffusion equation can be written:

      dcT/dt = Div (sigma Grad T) + Q_s

   where

     T     is the temperature.
     Div   is the divergence operator,
     grad  is the gradient operator,
     sigma is the thermal conductivity,
     c     is the heat capacity,
     Q_s   is the heat source

   Class DiffusionTDO represents the right-hand side of the above
   system of ODEs.

     f(t, T) = -M_0(c)^{-1}(S_0(sigma)T - M_0 Q_s)

   where

     M_0(c)     is an H_1 mass matrix
     S_0(sigma) is the diffusion operator

   The implicit solve method will solve

     (M_0(c)+dt S_0(sigma))k = -S_0(sigma)T + M_0 Q_s
*/
class DiffusionTDO : public TimeDependentOperator
{
public:
   DiffusionTDO(ParFiniteElementSpace &H1_FES,
                Coefficient & dTdtBdr,
                Array<int> & bdr_attr,
                Coefficient & c, bool td_c,
                Coefficient & k, bool td_k,
                Coefficient & Q, bool td_Q);
   DiffusionTDO(ParFiniteElementSpace &H1_FES,
                Coefficient & dTdtBdr,
                Array<int> & bdr_attr,
                Coefficient & c, bool td_c,
                MatrixCoefficient & K, bool td_k,
                Coefficient & Q, bool td_Q);

   void SetTime(const double time);
   /*
    void SetHeatSource(Coefficient & Q, bool time_dep = false);

    void SetConductivityCoefficient(Coefficient & k,
             bool time_dep = false);

    void SetConductivityCoefficient(MatrixCoefficient & K,
             bool time_dep = false);

    void SetSpecificHeatCoefficient(
             bool time_dep = false);
   */

   const ParBilinearForm & GetMassMatrix() { return *mC_; }

   /** @brief Perform the action of the operator: @a q = f(@a y, t), where
        q solves the algebraic equation F(@a y, q, t) = G(@a y, t) and t is the
        current time. */
   virtual void Mult(const Vector &y, Vector &q) const;

   /** @brief Solve the equation: @a q = f(@a y + @a dt @a q, t), for the
       unknown @a q at the current time t.

       For general F and G, the equation for @a q becomes:
       F(@a y + @a dt @a q, @a q, t) = G(@a y + @a dt @a q, t).

       The input vector @a y corresponds to time index (or cycle) n, while the
       currently set time, #t, and the result vector @a q correspond to time
       index n+1. The time step @a dt corresponds to the time interval between
       cycles n and n+1.

       This method allows for the abstract implementation of some time
       integration methods, including diagonal implicit Runge-Kutta (DIRK)
       methods and the backward Euler method in particular.

       If not re-implemented, this method simply generates an error. */
   virtual void ImplicitSolve(const double dt, const Vector &y, Vector &q);

   virtual ~DiffusionTDO();

private:

   void init();

   void initMult() const;
   void initA(double dt);
   void initImplicitSolve();

   int myid_;
   bool init_;
   // bool initA_;
   // bool initAInv_;
   bool newTime_;

   mutable int multCount_;
   int solveCount_;

   ParFiniteElementSpace * H1_FESpace_;

   ParBilinearForm * mC_;
   ParBilinearForm * sK_;
   ParBilinearForm * a_;

   ParGridFunction * dTdt_gf_;
   ParLinearForm * Qs_;

   mutable HypreParMatrix   MC_;
   mutable HyprePCG       * MCInv_;
   mutable HypreDiagScale * MCDiag_;

   HypreParMatrix    A_;
   HyprePCG        * AInv_;
   HypreBoomerAMG  * APrecond_;

   // HypreParVector * T_;
   mutable Vector dTdt_;
   mutable Vector RHS_;
   Vector * rhs_;

   Array<int> * bdr_attr_;
   Array<int>   ess_bdr_tdofs_;

   Coefficient       * dTdtBdrCoef_;

   bool tdQ_;
   bool tdC_;
   bool tdK_;
   /*
   bool ownsQ_;
   bool ownsC_;
   bool ownsK_;
   */
   Coefficient       * QCoef_;
   Coefficient       * CCoef_;
   Coefficient       * kCoef_;
   MatrixCoefficient * KCoef_;
   // Coefficient       * CInvCoef_;
   // Coefficient       * kInvCoef_;
   // MatrixCoefficient * KInvCoef_;
   ProductCoefficient       * dtkCoef_;
   ScalarMatrixProductCoefficient * dtKCoef_;
};

class AdvectionDiffusionTDO : public TimeDependentOperator
{
public:
   AdvectionDiffusionTDO(ParFiniteElementSpace &H1_FES,
                         Coefficient & dTdtBdr,
                         Array<int> & bdr_attr,
                         Coefficient & c, bool td_c,
                         Coefficient & k, bool td_k,
                         VectorCoefficient &velCoef, bool td_v, double nu,
                         Coefficient & Q, bool td_Q);
   AdvectionDiffusionTDO(ParFiniteElementSpace &H1_FES,
                         Coefficient & dTdtBdr,
                         Array<int> & bdr_attr,
                         Coefficient & c, bool td_c,
                         MatrixCoefficient & K, bool td_k,
                         VectorCoefficient &velCoef, bool td_v, double nu,
                         Coefficient & Q, bool td_Q);

   void SetTime(const double time);
   /*
    void SetHeatSource(Coefficient & Q, bool time_dep = false);

    void SetConductivityCoefficient(Coefficient & k,
             bool time_dep = false);

    void SetConductivityCoefficient(MatrixCoefficient & K,
             bool time_dep = false);

    void SetSpecificHeatCoefficient(
             bool time_dep = false);
   */

   const ParBilinearForm & GetMassMatrix() { return *mC_; }

   /** @brief Perform the action of the operator: @a q = f(@a y, t), where
        q solves the algebraic equation F(@a y, q, t) = G(@a y, t) and t is the
        current time. */
   virtual void Mult(const Vector &y, Vector &q) const;

   /** @brief Solve the equation: @a q = f(@a y + @a dt @a q, t), for the
       unknown @a q at the current time t.

       For general F and G, the equation for @a q becomes:
       F(@a y + @a dt @a q, @a q, t) = G(@a y + @a dt @a q, t).

       The input vector @a y corresponds to time index (or cycle) n, while the
       currently set time, #t, and the result vector @a q correspond to time
       index n+1. The time step @a dt corresponds to the time interval between
       cycles n and n+1.

       This method allows for the abstract implementation of some time
       integration methods, including diagonal implicit Runge-Kutta (DIRK)
       methods and the backward Euler method in particular.

       If not re-implemented, this method simply generates an error. */
   virtual void ImplicitSolve(const double dt, const Vector &y, Vector &q);

   virtual ~AdvectionDiffusionTDO();

private:

   void init();

   void initMult() const;
   void initA(double dt);
   void initImplicitSolve();

   int myid_;
   bool init_;
   // bool initA_;
   // bool initAInv_;
   bool newTime_;

   mutable int multCount_;
   int solveCount_;

   ParFiniteElementSpace * H1_FESpace_;

   ParBilinearForm * mC_;
   ParBilinearForm * sK_;
   ParBilinearForm * aV_;
   ParBilinearForm * a_;

   ParGridFunction * dTdt_gf_;
   ParLinearForm * Qs_;

   mutable HypreParMatrix   MC_;
   mutable HyprePCG       * MCInv_;
   mutable HypreDiagScale * MCDiag_;

   HypreParMatrix    A_;
   HyprePCG        * AInv_;
   HypreBoomerAMG  * APrecond_;

   // HypreParVector * T_;
   mutable Vector dTdt_;
   mutable Vector RHS_;
   Vector * rhs_;

   Array<int> * bdr_attr_;
   Array<int>   ess_bdr_tdofs_;

   Coefficient       * dTdtBdrCoef_;

   bool tdQ_;
   bool tdC_;
   bool tdK_;
   bool tdV_;

   double nu_;

   Coefficient       * QCoef_;
   Coefficient       * CCoef_;
   Coefficient       * kCoef_;
   MatrixCoefficient * KCoef_;
   VectorCoefficient * VCoef_;

   ProductCoefficient       * dtkCoef_;
   ScalarMatrixProductCoefficient * dtKCoef_;
   ScalarVectorProductCoefficient * dtnuVCoef_;
};

struct CoefficientByAttr
{
   Array<int> attr;
   Coefficient * coef;
   bool ownCoef;
};

struct CoefficientsByAttr
{
   Array<int> attr;
   Array<Coefficient*> coefs;
   Array<bool> ownCoefs;
};

//typedef Coefficient*(*CoefFactory)(std::istream&);
class CoefFactory
{
private:
   Array<Coefficient*> coefs;                ///< Owned
   Array<CoefFactory*> ext_fac;              ///< Not owned
   Array<GridFunction*> ext_gf;              ///< Not owned
   Array<double (*)(const Vector &)> ext_fn; ///< Not owned

public:
   CoefFactory() {}

   ~CoefFactory();

   void StealData(Array<Coefficient*> &c) { c = coefs; coefs.LoseData(); }
  
   int AddExternalFactory(CoefFactory &cf) { return ext_fac.Append(&cf); }

   int AddExternalGridFunction(GridFunction &gf) { return ext_gf.Append(&gf); }

   int AddExternalFunction(double (*fn)(const Vector &))
   { return ext_fn.Append(fn); }

   Coefficient *operator()(std::istream &input);
   Coefficient *operator()(std::string &coef_name, std::istream &input);
};

class AdvectionDiffusionBC
{
private:
   Array<CoefficientByAttr*>  dbc; // Dirichlet BC data
   Array<CoefficientByAttr*>  nbc; // Neumann BC data
   Array<CoefficientsByAttr*> rbc; // Robin BC data
   mutable Array<int>  hbc; // Homogeneous Neumann BC boundry attributes

   std::set<int> bc_attr;
   const Array<int> & bdr_attr;

   CoefFactory * coefFact;

   void ReadBCs(std::istream &input);

   void ReadAttr(std::istream &input,
                 const std::string &bctype,
                 Array<int> &attr);

   void ReadCoefByAttr(std::istream &input,
                       const std::string &bctype,
                       CoefficientByAttr &cba);

   void ReadCoefsByAttr(std::istream &input,
                        const std::string &bctype,
                        CoefficientsByAttr &cba);

public:
   AdvectionDiffusionBC(const Array<int> & bdr)
      : bdr_attr(bdr), coefFact(NULL) {}

   AdvectionDiffusionBC(const Array<int> & bdr,
                        CoefFactory &cf, std::istream &input)
      : bdr_attr(bdr), coefFact(&cf) { ReadBCs(input); }

   ~AdvectionDiffusionBC();

   void LoadBCs(CoefFactory &cf, std::istream &input)
   { coefFact = &cf; ReadBCs(input); }
  
   // Enforce u = val on boundaries with attributes in bdr
   void AddDirichletBC(const Array<int> & bdr, Coefficient &val);

   // Enforce du/dn = val on boundaries with attributes in bdr
   void AddNeumannBC(const Array<int> & bdr, Coefficient &val);

   // Enforce du/dn + a u = b on boundaries with attributes in bdr
   void AddRobinBC(const Array<int> & bdr, Coefficient &a, Coefficient &b);

   const Array<CoefficientByAttr*> & GetDirichletBCs() const { return dbc; }
   const Array<CoefficientByAttr*> & GetNeumannBCs() const { return nbc; }
   const Array<CoefficientsByAttr*> & GetRobinBCs() const { return rbc; }
   const Array<int> & GetHomogeneousNeumannBCs() const;
};

enum FieldType {INVALID = -1,
                DENSITY     = 0,
                TEMPERATURE = 1
               };

std::string FieldSymbol(FieldType t);

class StateVariableFunc
{
public:

   virtual bool NonTrivialValue(FieldType deriv) const = 0;

   void SetDerivType(FieldType deriv) { derivType_ = deriv; }
   FieldType GetDerivType() const { return derivType_; }

protected:
   StateVariableFunc(FieldType deriv = INVALID) : derivType_(deriv) {}

   FieldType derivType_;
};


class StateVariableCoef : public StateVariableFunc, public Coefficient
{
public:

   virtual StateVariableCoef * Clone() const = 0;

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      switch (derivType_)
      {
         case INVALID:
            return Eval_Func(T, ip);
         case DENSITY:
            return Eval_dRho(T, ip);
         case TEMPERATURE:
            return Eval_dT(T, ip);
         default:
            return 0.0;
      }
   }

   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip) { return 0.0; }

   virtual double Eval_dRho(ElementTransformation &T,
                            const IntegrationPoint &ip) { return 0.0; }

   virtual double Eval_dT(ElementTransformation &T,
                          const IntegrationPoint &ip) { return 0.0; }

protected:
   StateVariableCoef(FieldType deriv = INVALID) : StateVariableFunc(deriv) {}
};

class StateVariableVecCoef : public StateVariableFunc,
   public VectorCoefficient
{
public:
   virtual void Eval(Vector &V,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      V.SetSize(vdim);

      switch (derivType_)
      {
         case INVALID:
            return Eval_Func(V, T, ip);
         case DENSITY:
            return Eval_dRho(V, T, ip);
         case TEMPERATURE:
            return Eval_dT(V, T, ip);
         default:
            V = 0.0;
            return;
      }
   }

   virtual void Eval_Func(Vector &V,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dRho(Vector &V,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dT(Vector &V,
                        ElementTransformation &T,
                        const IntegrationPoint &ip) { V = 0.0; }

protected:
   StateVariableVecCoef(int dim, FieldType deriv = INVALID)
      : StateVariableFunc(deriv), VectorCoefficient(dim) {}
};

class StateVariableMatCoef : public StateVariableFunc,
   public MatrixCoefficient
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
         case DENSITY:
            return Eval_dRho(M, T, ip);
         case TEMPERATURE:
            return Eval_dT(M, T, ip);
         default:
            M = 0.0;
            return;
      }
   }

   virtual void Eval_Func(DenseMatrix &M,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dRho(DenseMatrix &M,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dT(DenseMatrix &M,
                        ElementTransformation &T,
                        const IntegrationPoint &ip) { M = 0.0; }

protected:
   StateVariableMatCoef(int dim, FieldType deriv = INVALID)
      : StateVariableFunc(deriv), MatrixCoefficient(dim) {}

   StateVariableMatCoef(int h, int w, FieldType deriv = INVALID)
      : StateVariableFunc(deriv), MatrixCoefficient(h, w) {}
};

/** @brief A TimeDependentOperator class for integrating the non-linear
    diffusion equation.

    The equation being solved is the non-linear heat equation:

       c_rho dT/dt = Div(kappa * (T/tau)^{p/2} Grad T) + Q

    Where T is the temperature, c_rho is the heat capacity,
    kappa * (T/tau)^{p/2} is the thermal conductivity coefficient,
    and Q is the heat source.

    To avoid time step restrictions we will solve this with an implicit
    SDIRK method which will require us to find k=dT/dt satisfying the
    following equation for a sequence of different T's but a single dt:

    c_rho k = Div(kappa * ((T + dt k)/tau)^{p/2} Grad(T + dt k)) + Q

    This nonlinear equation will be solved using Newton's method in the
    DGAdvectionDiffusionTDO::ImplicitSolve(dt, T, k) member function.

    The Newton solver will repeatedly evaluate this equation and its gradient
    for a fixed T and a sequence of k's.  The evaluation is performed by
    DGAdvectionDiffusionTDO::NLOperator::Mult(k, r) and the gradient is
    computed by DGAdvectionDiffusionTDO::NLOperator::GetGradient(k).

*/
class DGAdvectionDiffusionTDO : public TimeDependentOperator
{
private:
   const MPI_Session & mpi_;
   int logging_;

   ParFiniteElementSpace &fes_;
   ParGridFunction       &yGF_;
   // ParGridFunction       &kGF_;
   // SumCoefficient        &ykCoef_;

   class ADPrec : public Solver
   {
   private:
      Operator *prec_;

   public:
      ADPrec() : prec_(NULL) {}

      ~ADPrec() { delete prec_; }

      virtual void SetOperator(const Operator &op);
      virtual void Mult (const Vector & x, Vector & y) const
      { prec_->Mult(x, y); }
   };

   ADPrec        newton_op_prec_;
   // Array<HypreSmoother*> newton_op_prec_blocks_;
   // Operator    * newton_op_prec_;
   GMRESSolver   newton_op_solver_;
   NewtonSolver  newton_solver_;

   // Data collection used to write data files
   DataCollection * dc_;

   // Sockets used to communicate with GLVis
   std::map<std::string, socketstream*> socks_;

   class NLOperator : public Operator
   {
   protected:
      const MPI_Session &mpi_;
      const DGParams &dg_;

      int logging_;
      std::string log_prefix_;

      int index_;
      std::string field_name_;
      double dt_;
      ParFiniteElementSpace &fes_;
      ParMesh               &pmesh_;
      ParGridFunction       &yGF_;
      ParGridFunction       &kGF_;
      mutable ParLinearForm rLF_;

      GridFunctionCoefficient yCoef_;
      GridFunctionCoefficient kCoef_;

      SumCoefficient &ykCoef_;

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

      // Domain integrators for time derivatives of field variables
      BilinearFormIntegrator* dbfi_m_;  // Domain Integrators
      // Array<Array<StateVariableCoef*> >      dbfi_mc_; // Domain Integrators

      // Domain integrators for field variables at next time step
      Array<BilinearFormIntegrator*> dbfi_;  // Domain Integrators
      Array<BilinearFormIntegrator*> fbfi_;  // Interior Face Integrators
      Array<BilinearFormIntegrator*> bfbfi_; // Boundary Face Integrators
      Array<Array<int>*>             bfbfi_marker_; ///< Entries are owned.

      // Domain integrators for source terms
      Array<LinearFormIntegrator*> dlfi_;  // Domain Integrators
      Array<LinearFormIntegrator*> bflfi_; // Boundary Face Integrators
      Array<Array<int>*>           bflfi_marker_; ///< Entries are owned.

      ParBilinearForm* blf_; // Bilinear Form Object for the Gradient
      mutable Operator* blf_op_; // The gradient operator

      int term_flag_;
      int vis_flag_;

      // Data collection used to write data files
      DataCollection * dc_;

      // Sockets used to communicate with GLVis
      std::map<std::string, socketstream*> socks_;

      NLOperator(const MPI_Session & mpi, const DGParams & dg,
		 ParFiniteElementSpace &fes,
                 ParGridFunction & yGF,
                 ParGridFunction & kGF,
                 SumCoefficient &ykCoef,
                 int term_flag, int vis_flag, int logging = 0,
                 const std::string & log_prefix = "");


   public:

      virtual ~NLOperator();

      void SetLogging(int logging, const std::string & prefix = "");

      virtual void SetTimeStep(double dt);

      virtual void Mult(const Vector &k, Vector &r) const;

      virtual void Update();

      virtual Operator &GetGradient(const Vector &k) const;

      inline bool CheckTermFlag(int flag) { return (term_flag_>> flag) & 1; }

      inline bool CheckVisFlag(int flag) { return (vis_flag_>> flag) & 1; }

      virtual int GetDefaultVisFlag() { return 1; }

      virtual void RegisterDataFields(DataCollection & dc);

      virtual void PrepareDataFields();

      virtual void InitializeGLVis() = 0;

      virtual void DisplayToGLVis() = 0;
   };

   class AdvectionDiffusionOp : public NLOperator
   {
   private:
      Array<Coefficient*> coefs_;
      Array<ProductCoefficient*>             dtSCoefs_;
      Array<ProductCoefficient*>             negdtSCoefs_;
      Array<ScalarVectorProductCoefficient*> dtVCoefs_;
      Array<ScalarMatrixProductCoefficient*> dtMCoefs_;
      std::vector<socketstream*> sout_;
      ParGridFunction coefGF_;

      // GridFunctionCoefficient &y0Coef_;

      const AdvectionDiffusionBC & bcs_;

   public:

      AdvectionDiffusionOp(const MPI_Session & mpi, const DGParams & dg,
                           bool h1,
                           ParFiniteElementSpace &fes,
                           ParGridFunction & yGF,
                           ParGridFunction & kGF,
                           SumCoefficient &ykCoef,
                           const AdvectionDiffusionBC & bcs,
                           int term_flag, int vis_flag,
                           int logging = 0,
                           const std::string & log_prefix = "");

      ~AdvectionDiffusionOp();

      virtual void SetTimeStep(double dt);

      virtual void Update();

      /** Sets the time derivative on the left hand side of the equation to be:
             d MCoef / dt
      */
      void SetTimeDerivativeTerm(StateVariableCoef &MCoef);

      /** Sets the diffusion term on the right hand side of the equation
          to be:
             Div(DCoef Grad y[index])
          where index is the index of the equation.
       */
      void SetDiffusionTerm(StateVariableCoef &DCoef);
      void SetDiffusionTerm(StateVariableMatCoef &DCoef);

      /** Sets the advection term on the right hand side of the
      equation to be:
             Div(VCoef y[index])
          where index is the index of the equation.
       */
      void SetAdvectionTerm(StateVariableVecCoef &VCoef, bool bc = false);

      void SetSourceTerm(StateVariableCoef &SCoef);

      virtual void InitializeGLVis();

      virtual void DisplayToGLVis();
   };

   AdvectionDiffusionOp op_;

   mutable Vector x_;
   mutable Vector y_;
   Vector u_;
   Vector dudt_;

public:
   DGAdvectionDiffusionTDO(const MPI_Session & mpi,
                           const DGParams & dg,
                           bool h1,
                           ParFiniteElementSpace &fes,
                           ParGridFunction &yGF,
                           ParGridFunction &kGF,
                           SumCoefficient & ykCoef,
                           const AdvectionDiffusionBC & bcs,
                           int term_flag,
                           int vis_flag,
                           bool imex = true,
                           int logging = 0);

   ~DGAdvectionDiffusionTDO();

   void SetTime(const double _t);
   void SetLogging(int logging);

   /** Sets the time derivative on the left hand side of the equation to be:
              d MCoef / dt
       */
   void SetHeatCapacityCoef(StateVariableCoef &MCoef)
   { op_.SetTimeDerivativeTerm(MCoef); }

   /** Sets the diffusion term on the right hand side of the equation
       to be:
          Div(DCoef Grad y[index])
       where index is the index of the equation.
    */
   void SetConductivityCoef(StateVariableCoef &DCoef)
   { op_.SetDiffusionTerm(DCoef); }
   void SetDiffusionTerm(StateVariableMatCoef &DCoef)
   { op_.SetDiffusionTerm(DCoef); }

   /** Sets the advection term on the right hand side of the
   equation to be:
          Div(VCoef y[index])
       where index is the index of the equation.
    */
   void SetVelocityCoef(StateVariableVecCoef &VCoef)
   { op_.SetAdvectionTerm(VCoef); }

   void SetHeatSourceCoef(StateVariableCoef &SCoef)
   { op_.SetSourceTerm(SCoef); }

   void RegisterDataFields(DataCollection & dc);

   void PrepareDataFields();

   void InitializeGLVis();

   void DisplayToGLVis();

   // virtual void ExplicitMult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const double dt, const Vector &y, Vector &k);

   void Update();
};

} // namespace thermal

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_FOURIER_SOLVER
