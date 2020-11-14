#include "mfem.hpp"
#include "genericintegrator.hpp"

#pragma once

namespace mfem
{
class GenericForm : public Operator
{
   class Gradient : public Operator
   {
   public:
      Gradient(GenericForm &f) : form(f){};

      void Mult(const Vector &x, Vector &y) const override
      {
         form.GradientMult(x, y);
      }

   private:
      GenericForm &form;
   };

public:
   GenericForm(FiniteElementSpace *f);

   void Mult(const Vector &x, Vector &y) const override;

   void AddDomainIntegrator(GenericIntegrator *i)
   {
      domain_integrators.Append(i);
      i->Setup(*fes);
   }

   void AssumeLinear() { is_linear = true; }

   // Return an Operator that provides a Mult(x, y) which is the MatVec of the
   // gradient of the GenericForm wrt x. Acts as a "passthrough" to
   // ::GradientMult in order to satisfy mfem interfaces.
   Operator &GetGradient(const Vector &x) const override;

   // Return an assmbled parallel matrix which represents the gradient of the
   // GenericForm wrt to x.
   HypreParMatrix *GetGradientMatrix(const Vector &x);

protected:
   // y = F'(x_lin) * v
   void GradientMult(const Vector &v, Vector &y) const;

   bool is_linear = false;
   FiniteElementSpace *fes;
   Array<GenericIntegrator *> domain_integrators;
   Array<int> ess_tdof_list;

   // T -> L
   const Operator *P;

   // L -> E
   const Operator *G;

   mutable Vector x_local, y_local, v_local, px, py, pv;

   // State to build the Gradient on, single source of the true state.
   mutable Vector x_lin;

   mutable Gradient grad;

   HypreParMatrix *gradient_matrix = nullptr;
};
} // namespace mfem