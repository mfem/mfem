#include "mfem.hpp"

namespace mfem
{





/// @brief Time dependent DG operator for hyperbolic conservation laws
class DGHyperbolicConservationLaws : public TimeDependentOperator
{
private:
   const int dim; // domain dimension
   const int num_equations; // the number of equations
   FiniteElementSpace &vfes; // vector finite element space
   // Element integration form. Should contain ComputeFlux
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator;
   // Base Nonlinear Form
   std::unique_ptr<NonlinearForm> nonlinearForm;
   // element-wise inverse mass matrix
   std::vector<DenseMatrix> Me_inv;
   // global maximum characteristic speed. Updated by form integrators
   mutable double max_char_speed;
   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();

public:
   /**
    * @brief Construct a new DGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ integrator (F(u,x), grad v)
    */
   DGHyperbolicConservationLaws(
      FiniteElementSpace &vfes_,
      HyperbolicFormIntegrator *formIntegrator_);
   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   void Mult(const Vector &x, Vector &y) const override;
   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   double GetMaxCharSpeed() { return max_char_speed; }
   void Update();

};
//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
   FiniteElementSpace &vfes_, HyperbolicFormIntegrator *formIntegrator_)
   : TimeDependentOperator(vfes_.GetTrueVSize()),
     dim(vfes_.GetMesh()->Dimension()),
     num_equations(formIntegrator_->num_equations),
     vfes(vfes_),
     formIntegrator(formIntegrator_),
     z(vfes_.GetTrueVSize())
{
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();
#ifndef MFEM_USE_MPI
   nonlinearForm.reset(new NonlinearForm(&vfes));
#else
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(&vfes);
   if (pvfes)
   {
      nonlinearForm.reset(new ParNonlinearForm(pvfes));
   }
   else
   {
      nonlinearForm.reset(new NonlinearForm(&vfes));
   }
#endif
   formIntegrator->ResetMaxCharSpeed();

   nonlinearForm->AddDomainIntegrator(formIntegrator.get());
   nonlinearForm->AddInteriorFaceIntegrator(formIntegrator.get());
   nonlinearForm->UseExternalIntegrators();

}

void DGHyperbolicConservationLaws::ComputeInvMass()
{
   DenseMatrix Me;     // auxiliary local mass matrix
   MassIntegrator mi;  // mass integrator
   // resize it to the current number of elements
   Me_inv.resize(vfes.GetNE());
   for (int i = 0; i < vfes.GetNE(); i++)
   {
      Me.SetSize(vfes.GetFE(i)->GetDof());
      mi.AssembleElementMatrix(*vfes.GetFE(i),
                               *vfes.GetElementTransformation(i), Me);
      DenseMatrixInverse inv(&Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv[i]);
   }
}

void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   formIntegrator->ResetMaxCharSpeed();
   // 1. Create the vector z with the face terms (F(u), grad v) - <F.n(u), [w]>.
   nonlinearForm->Mult(x, z);
   max_char_speed = formIntegrator->GetMaxCharSpeed();

   // 2. Multiply element-wise by the inverse mass matrices.
   Vector zval;             // local dual vector storage
   Array<int> vdofs;        // local degrees of freedom storage
   DenseMatrix zmat, ymat;  // local dual vector storage

   for (int i = 0; i < vfes.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes.GetElementVDofs(i, vdofs);
      // get local dual vector
      z.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), vfes.GetFE(i)->GetDof(),
                           num_equations);
      ymat.SetSize(Me_inv[i].Height(), num_equations);
      // mass matrix inversion and pass it to global vector
      mfem::Mult(Me_inv[i], zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

void DGHyperbolicConservationLaws::Update()
{
   nonlinearForm->Update();
   height = nonlinearForm->Height();
   width = height;
   z.SetSize(height);

   ComputeInvMass();
}

std::__1::function<void(const Vector&, Vector&)> GetMovingVortexInit(
   const double Minf, const double radius, const double beta,
   const double gas_constant, const double specific_heat_ratio)
{
   return [specific_heat_ratio,
           gas_constant, Minf, radius, beta](const Vector &x, Vector &y)
   {
      MFEM_ASSERT(x.Size() == 2, "");

      const double xc = 0.0, yc = 0.0;

      // Nice units
      const double vel_inf = 1.;
      const double den_inf = 1.;

      // Derive remainder of background state from this and Minf
      const double pres_inf = (den_inf / specific_heat_ratio) *
                              (vel_inf / Minf) * (vel_inf / Minf);
      const double temp_inf = pres_inf / (den_inf * gas_constant);

      double r2rad = 0.0;
      r2rad += (x(0) - xc) * (x(0) - xc);
      r2rad += (x(1) - yc) * (x(1) - yc);
      r2rad /= (radius * radius);

      const double shrinv1 = 1.0 / (specific_heat_ratio - 1.);

      const double velX =
         vel_inf * (1 - beta * (x(1) - yc) / radius * exp(-0.5 * r2rad));
      const double velY =
         vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
      const double vel2 = velX * velX + velY * velY;

      const double specific_heat =
         gas_constant * specific_heat_ratio * shrinv1;
      const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                          (vel_inf * beta) / specific_heat *
                          exp(-r2rad);

      const double den = den_inf * pow(temp / temp_inf, shrinv1);
      const double pres = den * gas_constant * temp;
      const double energy = shrinv1 * pres / den + 0.5 * vel2;

      y(0) = den;
      y(1) = den * velX;
      y(2) = den * velY;
      y(3) = den * energy;
   };
}

}