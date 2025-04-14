#ifndef MFEM_NAVIER_SOLVER_GCN_HPP
#define MFEM_NAVIER_SOLVER_GCN_HPP

#define NAVIER_GCN_VERSION 0.1

#include "mfem.hpp"

namespace mfem {


class NavierSolverGCN
{
public:
   NavierSolverGCN(ParMesh* mesh, int order, std::shared_ptr<Coefficient> visc);


   ~NavierSolverGCN();

   /// Initialize forms, solvers and preconditioners.
   void SetupOperator(real_t t, real_t dt);
   void SetupRHS(real_t, real_t dt);

   /// Compute the solution at next time step t+dt
   void Step(real_t &time, real_t dt, int cur_step, bool provisional = false);

   /// Return the provisional velocity ParGridFunction.
   ParGridFunction* GetProvisionalVelocity() { return nvel.get(); }

   /// Return the provisional pressure ParGridFunction.
   ParGridFunction* GetProvisionalPressure() { return npres.get(); }

   void SetTheta(real_t t1=real_t(0.5),real_t t2=real_t(0.5),
                 real_t t3=real_t(0.5),real_t t4=real_t(0.5))
   {
      thet1=t1;
      thet2=t2;
      thet3=t3;
      thet4=t4;
   }

   void SetBrinkman(std::shared_ptr<Coefficient> brink_)
   {
      brink = brink_;
   }

   void SetViscosity(std::shared_ptr<Coefficient> visc_)
   {
      visc = visc_;
   }

   //velocity boundary conditions
   void AddVelocityBC(int id, std::shared_ptr<VectorCoefficient> val);

private:

   real_t thet1,thet2,thet3,thet4;


    /// Enable/disable debug output.
   bool debug = false;

    /// Enable/disable verbose output.
   bool verbose = true;

    /// Enable/disable partial assembly of forms.
   bool partial_assembly = false;

    /// The parallel mesh.
   ParMesh *pmesh = nullptr;

    /// The order of the velocity and pressure space.
   int order;

   std::shared_ptr<Coefficient> visc;
   std::shared_ptr<Coefficient> brink;

   std::unique_ptr<ParGridFunction> nvel; //next velocity
   std::unique_ptr<ParGridFunction> pvel; //previous velocity
   std::unique_ptr<ParGridFunction> cvel; //current velocity

   std::unique_ptr<ParGridFunction> ppres; //next pressure
   std::unique_ptr<ParGridFunction> npres; //previous pressure
   std::unique_ptr<ParGridFunction> cpres; //current pressure

   std::unique_ptr<H1_FECollection> vfec;
   std::unique_ptr<H1_FECollection> pfec;
   std::unique_ptr<ParFiniteElementSpace> vfes;
   std::unique_ptr<ParFiniteElementSpace> pfes;

   std::unique_ptr<ParBilinearForm> A11;
   std::unique_ptr<ParMixedBilinearForm> A12;
   std::unique_ptr<ParMixedBilinearForm> A21;

   OperatorHandle A11H;
   OperatorHandle A12H;
   OperatorHandle A21H;
   std::unique_ptr<BlockOperator> AB;
   Array<int> block_true_offsets;

   Vector rhs;

   VectorGridFunctionCoefficient nvelc;
   VectorGridFunctionCoefficient pvelc;
   VectorGridFunctionCoefficient cvelc;

   GridFunctionCoefficient ppresc;
   GridFunctionCoefficient npresc;
   GridFunctionCoefficient cpresc;

   ConstantCoefficient onecoeff;
   ConstantCoefficient zerocoef;

   std::unique_ptr<ProductCoefficient> nbrink;
   std::unique_ptr<ProductCoefficient> nvisc;
   ConstantCoefficient icoeff; //thet1*dt

   //boundary conditions
   std::map<int, std::shared_ptr<VectorCoefficient>> vel_bcs;

   // holds the velocity constrained DOFs
   mfem::Array<int> ess_tdofv;

   // holds the pressure constrained DOFs
   mfem::Array<int> ess_tdofp;

   void SetEssTDofs(real_t t, ParGridFunction& pgf, mfem::Array<int>& ess_dofs);
   void SetEssTDofs(mfem::Array<int>& ess_dofs);
   void SetEssTDofs(real_t t, ParGridFunction& pgf);

   std::unique_ptr<ParBilinearForm> K;

};//end NavierSolverGCN


// evaluates u+cc*(u \nabla u + brink*u)
class NSResCoeff: public VectorCoefficient
{
public:
   NSResCoeff(ParGridFunction &u, std::shared_ptr<Coefficient> brink_, real_t cc_) : VectorCoefficient(u.VectorDim())
   {
      gf=&u;
      grad.SetSize(gf->VectorDim());
      vel.SetSize(gf->VectorDim());
      brink=brink_;
      cc=cc_;
   }


   void Eval(Vector &v, ElementTransformation &Trans, const IntegrationPoint &ip) override
   {
      v=real_t(0.0);
      //get the velocity
      Trans.SetIntPoint(&ip);
      gf->GetVectorValue(Trans, ip, vel);
      gf->GetVectorGradient(Trans, grad);

      //u\nabla u
      grad.Mult(vel, v);
      v*=cc;

      real_t bp=0.0;
      if(brink!=nullptr)
      {
         bp=brink->Eval(Trans,ip);
      }

      v.Add(1.0+cc*bp,vel);
   }

private:
   DenseMatrix grad;
   Vector vel;
   ParGridFunction *gf;
   std::shared_ptr<Coefficient> brink;
   real_t cc;
};


class VectorConvectionIntegrator : public BilinearFormIntegrator
{
protected:
    VectorCoefficient *Q;
    real_t alpha;
    // PA extension
    Vector pa_data;
    const DofToQuad *maps;         ///< Not owned
    const GeometricFactors *geom;  ///< Not owned
    int dim, ne, nq, dofs1D, quad1D;

private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir, partelmat;
   Vector shape, vec2, BdFidxT;
#endif

public:
   VectorConvectionIntegrator(VectorCoefficient &q, real_t a = 1.0)
      : Q(&q) { alpha = a; }

   void AssembleElementMatrix(const FiniteElement &,
                              ElementTransformation &,
                              DenseMatrix &) override;

   static const IntegrationRule &GetRule(const FiniteElement &el,
                                         const ElementTransformation &Trans);

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         const ElementTransformation &Trans);

protected:
   const IntegrationRule* GetDefaultIntegrationRule(
      const FiniteElement& trial_fe,
      const FiniteElement& test_fe,
      const ElementTransformation& trans) const override
   {
      return &GetRule(trial_fe, test_fe, trans);
   }

};



}//end namespace mfem




#endif
