#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <functional>

#include "mfem.hpp"
#include "fem/hyperbolic_conservation_laws.hpp"

namespace mfem
{
class ProxGalerkinHCL : public TimeDependentOperator
{
private:
   const int dim;
   const int num_equations;
   long fes_sequence;
   // Vector finite element space containing conserved variables
   FiniteElementSpace *vfes;
   FiniteElementSpace *fes;
   Array<FiniteElementSpace*> latent_fes;
   Array<std::function<double(double)>*> latent2primal;
   Array<std::function<double(double)>*> der_latent2primal;

   // Element integration form. Should contain ComputeFlux
   HyperbolicFormIntegrator &formIntegrator;
   // Base Nonlinear Form
   NonlinearForm *nonlinearForm;
   // element-wise inverse mass matrix
   BilinearForm *invMass;
   // global maximum characteristic speed. Updated by form integrators
   mutable double max_char_speed;
   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();

public:
   /**
    * @brief Construct a new ProxGalerkinHCL object
    *
    * @param vfes vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator (F(u,x), grad v) and (F̂(u±, x, n), [[v]])
    * @param num_equations the number of equations
    */
   ProxGalerkinHCL(
      FiniteElementSpace *fes,
      FiniteElementSpace *vfes,
      Array<FiniteElementSpace *> latent_fes,
      Array<std::function<double(double)>*> latent2primal,
      Array<std::function<double(double)>*> der_latent2primal,
      HyperbolicFormIntegrator &formIntegrator,
      const int num_equations);
   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   inline double getMaxCharSpeed()
   {
      return max_char_speed;
   }

   void Update()
   {
      nonlinearForm->Update();
      height = nonlinearForm->Height();
      width = height;
      z.SetSize(height);
      ComputeInvMass();
   }

   virtual ~ProxGalerkinHCL()
   {
      delete nonlinearForm;
      delete invMass;
   }
};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class ProxGalerkinHCL
ProxGalerkinHCL::ProxGalerkinHCL(
   FiniteElementSpace *fes, FiniteElementSpace *vfes,
   Array<FiniteElementSpace *>latent_fes,
   Array<std::function<double(double)>*> latent2primal,
   Array<std::function<double(double)>*> der_latent2primal,
   HyperbolicFormIntegrator &formIntegrator,
   const int num_equations)
   : TimeDependentOperator(vfes->GetNDofs() * num_equations),
     dim(vfes->GetFE(0)->GetDim()),
     num_equations(num_equations),
     fes_sequence(vfes->GetSequence()),
     fes(fes),
     vfes(vfes),
     latent_fes(latent_fes),
     latent2primal(latent2primal),
     der_latent2primal(der_latent2primal),
     formIntegrator(formIntegrator),
     invMass(nullptr),
     z(vfes->GetNDofs() * num_equations)
{
   MFEM_VERIFY(vfes->GetOrdering() == Ordering::byNODES,
               "Ordering should be by nodes.");
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();
#ifndef MFEM_USE_MPI
   nonlinearForm = new NonlinearForm(vfes);
#else
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(vfes);
   if (pvfes)
   {
      nonlinearForm = new ParNonlinearForm(pvfes);
   }
   else
   {
      nonlinearForm = new NonlinearForm(vfes);
   }
#endif
   formIntegrator.resetMaxCharSpeed();

   nonlinearForm->AddDomainIntegrator(&formIntegrator);
   nonlinearForm->AddInteriorFaceIntegrator(&formIntegrator);

   height = z.Size();
   width = z.Size();

}

void ProxGalerkinHCL::ComputeInvMass()
{
   if (invMass)
   {
      delete invMass;
   }
   invMass = new BilinearForm(fes);
   invMass->AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
   invMass->Assemble();
}

void ProxGalerkinHCL::Mult(const Vector &x, Vector &y) const
{
   y.SetSize(x.Size());

   // 0. Reset wavespeed computation before operator application.
   formIntegrator.resetMaxCharSpeed();
   // 1. Create the vector z with the face terms (F(u), grad v) - <F.n(u), [w]>.
   nonlinearForm->Mult(x, z);
   max_char_speed = formIntegrator.getMaxCharSpeed();

   // 2. Multiply element-wise by the inverse mass matrices.
   const int dof = x.Size() / num_equations;
   Vector z_comp(z.GetData(), dof);
   Vector y_comp(y.GetData(), dof);
   for (int i=0; i<num_equations; i++)
   {
      z_comp.SetData(z.GetData() + dof*i);
      y_comp.SetData(y.GetData() + dof*i);
      invMass->Mult(z_comp, y_comp);
   }
}

void ProxGalerkinHCL::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   k.SetSize(x.Size());

}

} // namespace mfem