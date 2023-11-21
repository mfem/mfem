#pragma once

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <functional>
#include "ex37.hpp"

namespace mfem
{
/** Mass integrator (u⋅d, v⋅d) restricted to the boundary of a domain */
class VectorBoundaryDirectionalMassIntegrator: public BilinearFormIntegrator
{
private:
   Coefficient &k;
   int vdim;
   VectorCoefficient &direction;
   int oa, ob;

public:
   /// Construct an integrator with coefficient 1.0
   VectorBoundaryDirectionalMassIntegrator(Coefficient &k,
                                           VectorCoefficient &direction,
                                           const int oa=1, const int ob=1)
      : k(k), vdim(direction.GetVDim()), direction(direction),
        oa(oa), ob(ob) { }

   using BilinearFormIntegrator::AssembleElementMatrix;
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      DenseMatrix &elmat)
   {
      int dof = el.GetDof();
      Vector shape(dof), vec(vdim);

      elmat.SetSize(dof*vdim);
      elmat = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = oa * el.GetOrder() + ob;    // <------ user control
         ir = &IntRules.Get(Tr.GetGeometryType(), intorder); // of integration order
      }

      DenseMatrix elmat_scalar(dof);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         // Set the integration point in the face and the neighboring element
         Tr.SetIntPoint(&ip);

         // Access the neighboring element's integration point
         direction.Eval(vec, Tr, ip);
         double val = k.Eval(Tr, ip)*Tr.Weight() * ip.weight;

         el.CalcShape(ip, shape);
         MultVVt(shape, elmat_scalar);
         for (int row = 0; row < vdim; row++)
         {
            for (int col = 0; col < vdim; col++)
            {
               elmat.AddMatrix(val*vec(row)*vec(col), elmat_scalar, dof*row, dof*col);
            }
         }
      }
   }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el,
                                   const FiniteElement &dummy,
                                   FaceElementTransformations &Tr,
                                   DenseMatrix &elmat)
   {
      int dof = el.GetDof();
      Vector shape(dof), vec(vdim);

      elmat.SetSize(dof*vdim);
      elmat = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = oa * el.GetOrder() + ob;    // <------ user control
         ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
      }

      DenseMatrix elmat_scalar(dof);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         // Set the integration point in the face and the neighboring element
         Tr.SetAllIntPoints(&ip);

         // Access the neighboring element's integration point
         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         direction.Eval(vec, *Tr.Face, ip);
         double val = k.Eval(*Tr.Face, ip)*Tr.Face->Weight() * ip.weight;

         el.CalcShape(eip, shape);
         MultVVt(shape, elmat_scalar);
         for (int row = 0; row < vdim; row++)
         {
            for (int col = 0; col < vdim; col++)
            {
               elmat.AddMatrix(val*vec(row)*vec(col), elmat_scalar, dof*row, dof*col);
            }
         }
      }
   }
};
/** Mass integrator (u⋅n, v⋅n) restricted to the boundary of a domain */
class VectorBoundaryDirectionalLFIntegrator : public LinearFormIntegrator
{
   VectorCoefficient &direction, &force;
   int oa, ob, vdim;
public:
   /** @brief Constructs a boundary integrator with a given Coefficient @a QG.
       Integration order will be @a a * basis_order + @a b. */
   VectorBoundaryDirectionalLFIntegrator(VectorCoefficient &direction,
                                         VectorCoefficient &force,
                                         int a = 1, int b = 1)
      : direction(direction), force(force), oa(a), ob(b), vdim(direction.GetVDim()) { }

   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(
      const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
   {
      int dof = el.GetDof();

      Vector shape(dof), vec(vdim), vecF(vdim);
      elvect.SetSize(dof*vdim);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = oa * el.GetOrder() + ob;    // <------ user control
         ir = &IntRules.Get(Tr.GetGeometryType(), intorder); // of integration order
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         direction.Eval(vec, Tr, ip);
         force.Eval(vecF, Tr, ip);
         double val = Tr.Weight() * ip.weight * (vec * vecF);

         el.CalcShape(ip, shape);
         for (int row = 0; row < vdim; row++)
         {
            elvect.Add(val*vec(row), shape, dof*row);
         }
      }
   }
   virtual void AssembleRHSElementVect(
      const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
   {
      int dof = el.GetDof();

      Vector shape(dof), vec(vdim), vecF(vdim);
      elvect.SetSize(dof*vdim);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = oa * el.GetOrder() + ob;    // <------ user control
         ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         // Set the integration point in the face and the neighboring element
         Tr.SetAllIntPoints(&ip);

         // Access the neighboring element's integration point
         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         direction.Eval(vec, Tr, ip);
         force.Eval(vecF, Tr, ip);
         double val = Tr.Face->Weight() * ip.weight * (vec * vecF);

         el.CalcShape(eip, shape);
         for (int row = 0; row < vdim; row++)
         {
            elvect.Add(val*vec(row), shape, dof*row);
         }
      }
   }
};
class CompliantMechanism : public ObjectiveFunction
{
public:
   CompliantMechanism(Coefficient *lambda, Coefficient *mu, double epsilon,
                      Coefficient *rho, const double target_volume,
                      Array2D<int> &ess_bdr,
                      Array<int> &input_bdr, Array<int> &output_bdr,
                      Coefficient &input_spring, Coefficient &output_spring,
                      VectorCoefficient &input_direction, VectorCoefficient &output_direction,
                      FiniteElementSpace *displacement_space,
                      FiniteElementSpace *filter_space, double exponent, double rho_min):
      SIMPlambda(*lambda, design_density), SIMPmu(*mu, design_density),
      eps2(epsilon*epsilon), ess_bdr(ess_bdr), input_bdr(input_bdr),
      output_bdr(output_bdr),
      input_spring(input_spring), output_spring(output_spring),
      input_direction(input_direction), output_direction(output_direction),
      target_volume(target_volume),
      u(nullptr), frho(nullptr),
      rho(rho), strainEnergy(lambda, mu, u, frho, rho_min, exponent)
   {
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes;

      pfes = dynamic_cast<ParFiniteElementSpace*>(displacement_space);
      if (pfes)
      {
         u = new ParGridFunction(pfes);
         adju = new ParGridFunction(pfes);
      }
      else
      {
         u = new GridFunction(displacement_space);
         adju = new GridFunction(displacement_space);
      }

      pfes = dynamic_cast<ParFiniteElementSpace*>(filter_space);
      if (pfes)
      {
         frho = new ParGridFunction(pfes);
      }
      else
      {
         frho = new GridFunction(filter_space);
      }
#else
      u = new GridFunction(displacement_space);
      adju = new GridFunction(displacement_space);
      frho = new GridFunction(filter_space);
#endif
      frho->ProjectCoefficient(*rho);
      *u = 0.0;

      design_density.SetFunction([exponent, rho_min](const double rho)
      {
         return simp(rho, rho_min, exponent);
      });
      design_density.SetGridFunction(frho);
      SIMPlambda.SetBCoef(design_density);
      SIMPmu.SetBCoef(design_density);
      strainEnergy.SetDisplacement(u, adju);
      strainEnergy.SetFilteredDensity(frho);
   }

   virtual double Eval()
   {
      BilinearForm *elasticity, *filter;
      LinearForm *load, *adjload, *filterRHS;
      FiniteElementSpace * fes;

#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes;


      fes = frho->FESpace();
      pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         filter = new ParBilinearForm(pfes);
         filterRHS = new ParLinearForm(pfes);
      }
      else
      {
         filter = new BilinearForm(fes);
         filterRHS = new LinearForm(fes);
      }

      fes = u->FESpace();
      pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         elasticity = new ParBilinearForm(pfes);
         load = new ParLinearForm(pfes);
         adjload = new ParLinearForm(pfes);
      }
      else
      {
         elasticity = new BilinearForm(fes);
         load = new LinearForm(fes);
         adjload = new LinearForm(fes);
      }
#else
      fes = frho->FESpace();
      filter = new BilinearForm(fes);
      filterRHS = new LinearForm(fes);

      fes = u->FESpace();
      elasticity = new BilinearForm(fes);
      load = new LinearForm(fes);
      adjload = new LinearForm(fes);
#endif
      // Step 1. Projection
      proj(1e-12, 500);

      // Step 2. Filter equation
      filter->AddDomainIntegrator(new DiffusionIntegrator(eps2));
      filter->AddDomainIntegrator(new MassIntegrator());
      filterRHS->AddDomainIntegrator(new DomainLFIntegrator(*rho));
      Array<int> ess_bdr_filter;
      if (filter->FESpace()->GetMesh()->bdr_attributes.Size())
      {
         ess_bdr_filter.SetSize(filter->FESpace()->GetMesh()->bdr_attributes.Max());
         ess_bdr_filter = 0;
      }
      EllipticSolver filterSolver(filter, filterRHS, ess_bdr_filter);
      filterSolver.Solve(frho);

      // Step 3. Linear Elasticity
      elasticity->AddDomainIntegrator(new ElasticityIntegrator(SIMPlambda, SIMPmu));
      elasticity->AddBdrFaceIntegrator(new VectorBoundaryDirectionalMassIntegrator(
                                          input_spring, input_direction), input_bdr);
      elasticity->AddBdrFaceIntegrator(new VectorBoundaryDirectionalMassIntegrator(
                                          output_spring, output_direction), output_bdr);
      load->AddBdrFaceIntegrator(new VectorBoundaryDirectionalLFIntegrator(
                                    input_direction, input_direction), input_bdr);
      adjload->AddBdrFaceIntegrator(new VectorBoundaryDirectionalLFIntegrator(
                                       output_direction, output_direction), output_bdr);
      EllipticSolver elasticitySolver(elasticity, load, ess_bdr);
      elasticitySolver.Solve(u);
      adjload->Assemble();
      current_val = -(*adjload)(*u);

      // if (!IsFinite(current_val))
      // {
      //    mfem_warning("current value is not finite.");
      // }

      delete elasticity;
      delete load;
      delete adjload;
      delete filter;
      delete filterRHS;

      return current_val;
   }

   virtual GridFunction *Gradient()
   {
      BilinearForm *filter, *invmass, *elasticity;
      LinearForm *filterRHS, *gradH1form, *adjload;
      FiniteElementSpace * fes;
      GridFunction *gradH1;
      if (!x_gf)
      {
         mfem_error("Gradient should be called after SetGridFunction(psi).");
      }

#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes;

      fes = frho->FESpace();
      pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         filter = new ParBilinearForm(pfes);
         filterRHS = new ParLinearForm(pfes);
         gradH1 = new ParGridFunction(pfes);
      }
      else
      {
         filter = new BilinearForm(fes);
         filterRHS = new LinearForm(fes);
         gradH1 = new GridFunction(fes);
      }

      fes = x_gf->FESpace();
      pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         invmass = new ParBilinearForm(pfes);
         gradH1form = new ParLinearForm(pfes);
      }
      else
      {
         invmass = new BilinearForm(fes);
         gradH1form = new LinearForm(fes);
      }

      fes = adju->FESpace();
      pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         elasticity = new BilinearForm(pfes);
         adjload = new LinearForm(pfes);
      }
      else
      {
         elasticity = new BilinearForm(fes);
         adjload = new LinearForm(fes);
      }

#else
      fes = frho->FESpace();
      filter = new BilinearForm(fes);
      filterRHS = new LinearForm(fes);
      gradH1 = new GridFunction(fes);
      fes = x_gf->FESpace();
      invmass = new BilinearForm(fes);
      gradH1form = new LinearForm(fes);
      fes = adju->FESpace();
      elasticity = new BilinearForm(fes);
      adjload = new LinearForm(fes);

#endif

      elasticity->AddDomainIntegrator(new ElasticityIntegrator(SIMPlambda, SIMPmu));
      elasticity->AddBdrFaceIntegrator(new VectorBoundaryDirectionalMassIntegrator(
                                          input_spring, input_direction), input_bdr);
      elasticity->AddBdrFaceIntegrator(new VectorBoundaryDirectionalMassIntegrator(
                                          output_spring, output_direction), output_bdr);
      adjload->AddBdrFaceIntegrator(new VectorBoundaryDirectionalLFIntegrator(
                                       output_direction, output_direction), output_bdr);
      EllipticSolver elasticitySolver(elasticity, adjload, ess_bdr);
      elasticitySolver.Solve(adju);
      adju->Neg();

      // Step 1. Dual Filter Equation with Strain Density Energy
      filter->AddDomainIntegrator(new DiffusionIntegrator(eps2));
      filter->AddDomainIntegrator(new MassIntegrator());
      filterRHS->AddDomainIntegrator(new DomainLFIntegrator(strainEnergy));
      Array<int> ess_bdr_filter(0);
      if (filter->FESpace()->GetMesh()->bdr_attributes.Size())
      {
         ess_bdr_filter.SetSize(filter->FESpace()->GetMesh()->bdr_attributes.Max());
         ess_bdr_filter = 0;
      }
      *gradH1 = 0.0;
      EllipticSolver filterSolver(filter, filterRHS, ess_bdr_filter);
      filterSolver.Solve(gradH1);


      // Step 2. Project gradient to Control space
      invmass->AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));
      invmass->Assemble();
      GridFunctionCoefficient gradH1cf(gradH1);
      gradH1form->AddDomainIntegrator(new DomainLFIntegrator(gradH1cf));
      gradH1form->Assemble();
      invmass->Mult(*gradH1form, *gradF_gf);

      delete filter;
      delete invmass;
      delete filterRHS;
      delete gradH1form;
      delete gradH1;
      return gradF_gf;
   }
   double GetVolume() {return current_volume;}

   GridFunction *GetDisplacement() { return u; }
   GridFunction *GetFilteredDensity() { return frho; }
   MappedGridFunctionCoefficient &GetDesignDensity() { return design_density; }

   ~CompliantMechanism()
   {
      delete u;
      delete frho;
   }

   void SetGridFunction(GridFunction* x)
   {
      ObjectiveFunction::SetGridFunction(x);
      FiniteElementSpace *fes = x->FESpace();
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         gradF_gf = new ParGridFunction(pfes);
      }
      else
      {
         gradF_gf = new GridFunction(fes);
      }
#else
      gradF_gf = new GridFunction(fes);
#endif
   }

protected:


   /**
    * @brief Bregman projection of ρ = sigmoid(ψ) onto the subspace
    *        ∫_Ω ρ dx = θ vol(Ω) as follows:
    *
    *        1. Compute the root of the R → R function
    *            f(c) = ∫_Ω sigmoid(ψ + c) dx - θ vol(Ω)
    *        2. Set ψ ← ψ + c.
    *
    * @param psi a GridFunction to be updated
    * @param target_volume θ vol(Ω)
    * @param tol Newton iteration tolerance
    * @param max_its Newton maximum iteration number
    * @return double Final volume, ∫_Ω sigmoid(ψ)
    */
   double proj(double tol=1e-12, int max_its=10)
   {
      double c = 0;
      MappedGridFunctionCoefficient rho(x_gf, [&c](const double x) {return sigmoid(x + c);});
      // MappedGridFunctionCoefficient proj_drho(x_gf, [&c](const double x) {return der_sigmoid(x + c);});
      GridFunction *zero_gf;
      FiniteElementSpace * fes = x_gf->FESpace();
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         zero_gf = new ParGridFunction(pfes);
      }
      else
      {
         zero_gf = new GridFunction(fes);
      }
#else
      zero_gf = new GridFunction(fes);
#endif
      *zero_gf = 0.0;

      double Vc = zero_gf->ComputeL1Error(rho);
      double dVc = Vc - std::pow(zero_gf->ComputeL2Error(rho), 2);
      if (fabs(Vc - target_volume) > tol)
      {
         double dc;
         if (dVc > tol) // if derivative is sufficiently large,
         {
            dc = -(Vc - target_volume) / dVc;
         }
         else
         {
            dc = -(Vc > target_volume ? x_gf->Max() : x_gf->Min());
         }
         c = dc;
         int k;
         // Find an interval (c, c+dc) that contains c⋆.
         for (k=0; k < max_its; k++)
         {
            double Vc_old = Vc;
            Vc = zero_gf->ComputeL1Error(rho);
            if ((Vc_old - target_volume)*(Vc - target_volume) < 0)
            {
               break;
            }
            c += dc;
         }
         if (k == max_its) // if failed to find the search interval
         {
            return infinity();
         }
         // Bisection
         dc = fabs(dc);
         while (fabs(dc) > 1e-08)
         {
            dc /= 2.0;
            c = Vc > target_volume ? c - dc : c + dc;
            Vc = zero_gf->ComputeL1Error(rho);
         }
         *x_gf += c;
         c = 0;
      }
      current_volume = zero_gf->ComputeL1Error(rho);

      delete zero_gf;
      return current_volume;
   }
   double current_volume;

protected:
   ProductCoefficient SIMPlambda, SIMPmu;
   ConstantCoefficient eps2;
   Array2D<int> ess_bdr;
   Array<int> &input_bdr, &output_bdr;
   Coefficient &input_spring, &output_spring;
   VectorCoefficient &input_direction, &output_direction;
   double target_volume;
   GridFunction *u, *adju, *frho;
   Coefficient *rho;
   StrainEnergyDensityCoefficient strainEnergy;
   MappedGridFunctionCoefficient design_density;
};
}