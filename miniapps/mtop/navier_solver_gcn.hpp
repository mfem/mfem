#ifndef MFEM_NAVIER_SOLVER_GCN_HPP
#define MFEM_NAVIER_SOLVER_GCN_HPP

#define NAVIER_GCN_VERSION 0.1

#include "mfem.hpp"

namespace mfem {


class NavierSolverGCN
{
public:
   NavierSolverGCN(ParMesh* mesh, int order, std::shared_ptr<Coefficient> visc,
                   bool partial_assembly_=true, bool verbose_=true);


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

   /// Set current velocity using true dofs
   void SetCVelocity(Vector& vvel)
   {
       cvel->SetFromTrueDofs(vvel);
   }

   /// Set previous velocity using true dofs
   void SetPVelocity(Vector& vvel)
   {
       pvel->SetFromTrueDofs(vvel);
   }

   /// Set current velocity using vector coefficient
   void SetCVelocity(VectorCoefficient& vc)
   {
       cvel->ProjectCoefficient(vc);
   }

   /// Set current pressure using true dofs
   void SetCPressure(Vector& vpres)
   {
       cpres->SetFromTrueDofs(vpres);
   }

   /// Set previous pressure using true dofs
   void SetPPressure(Vector& vpress)
   {
       ppres->SetFromTrueDofs(vpress);
   }

   /// Set current pressure using coefficient
   void SetCPressure(Coefficient& pc)
   {
       cpres->ProjectCoefficient(pc);
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
   void AddVelocityBC(int id, std::shared_ptr<VectorCoefficient> val)
   {
       vel_bcs[id]=val;
   }


   /// Set the Dirichlet BC on a given ParGridFunction.
   void SetEssTDofs(real_t t, ParGridFunction& pgf);

private:



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

   H1_FECollection* vfec;
   H1_FECollection* pfec;
   ParFiniteElementSpace* vfes;
   ParFiniteElementSpace* pfes;

   std::unique_ptr<ParBilinearForm> A11;
   std::unique_ptr<ParMixedBilinearForm> A12;
   std::unique_ptr<ParMixedBilinearForm> A21;

   OperatorHandle A11H;
   OperatorHandle A12H;
   OperatorHandle A21H;
   std::unique_ptr<BlockOperator> AB;
   Array<int> block_true_offsets;

   std::unique_ptr<ParLinearForm> lf;

   Vector rhs;

   VectorGridFunctionCoefficient nvelc;
   VectorGridFunctionCoefficient pvelc;
   VectorGridFunctionCoefficient cvelc;

   GridFunctionCoefficient ppresc;
   GridFunctionCoefficient npresc;
   GridFunctionCoefficient cpresc;

   ConstantCoefficient onecoeff;
   ConstantCoefficient zerocoef;

   std::unique_ptr<ProductCoefficient> nbrink; //next brinkman
   std::unique_ptr<ProductCoefficient> nvisc;  //next viscosity

   std::unique_ptr<ProductCoefficient> cbrink; //current brinkman
   std::unique_ptr<ProductCoefficient> cvisc;  //current viscosity
   std::unique_ptr<VectorCoefficient>  scvelc; //scaled current velocity
   GradientGridFunctionCoefficient     gradcp; //current pressure gradient
   std::unique_ptr<VectorCoefficient>  scgradcp;//scaled gradient of the pressure

   //boundary conditions
   std::map<int, std::shared_ptr<VectorCoefficient>> vel_bcs;

   // holds the velocity constrained DOFs
   mfem::Array<int> ess_tdofv;

   // holds the pressure constrained DOFs
   mfem::Array<int> ess_tdofp;

   void SetEssTDofs(real_t t, ParGridFunction& pgf, mfem::Array<int>& ess_dofs);
   void SetEssTDofs(mfem::Array<int>& ess_dofs);


   /// copy cvel->pvel, nvel->cvel, cpres->ppres, npres->cpres
   void UpdateHistory()
   {
       std::swap(cvel,pvel);
       std::swap(cvel,nvel);

       std::swap(cpres,ppres);
       std::swap(cpres,npres);
   }


};//end NavierSolverGCN

class ViscStressCoeff: public VectorCoefficient
{
public:
    ViscStressCoeff(Coefficient* visc_, GridFunction* gf_)
        :VectorCoefficient(gf_->VectorDim()*gf_->VectorDim())
    {
        visc=visc_;
        gf=gf_;
    }

    void Eval(Vector &v, ElementTransformation &Trans, const IntegrationPoint &ip) override
    {
        v.SetSize(gf->VectorDim()*gf->VectorDim());

        //evaluate velocity gradient
        DenseMatrix grad(v.GetData(),gf->VectorDim(),gf->VectorDim());
        Trans.SetIntPoint(&ip);
        gf->GetVectorGradient(Trans,grad);

        //evaluate viscosity
        real_t mu=visc->Eval(Trans,ip);

        //symmetrize the gradient and compute the viscous stress
        for(int i=0;i<gf->VectorDim();i++){
            for(int j=i+1;j<gf->VectorDim();j++){
                grad(i,j)=mu*(grad(i,j)+grad(j,i));
                grad(j,i)=grad(i,j);
            }
            grad(i,i)=2*mu*grad(i,i);
        }
    }

private:

    Coefficient* visc;
    GridFunction* gf;
};

class ProductScalarVectorCoeff: public VectorCoefficient
{
public:
    ProductScalarVectorCoeff(Coefficient& sc_, VectorCoefficient& vc_):VectorCoefficient(vc_.GetVDim())
    {
        sc=&sc_;
        vc=&vc_;
    }

    void Eval(Vector &v, ElementTransformation &Trans, const IntegrationPoint &ip) override
    {
        real_t s=sc->Eval(Trans,ip);
        vc->Eval(v,Trans,ip);
        v*=s;
    }

private:
    Coefficient* sc;
    VectorCoefficient* vc;
};

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
