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
class MFEMNew
{
public:
   static NonlinearForm* newNonlinearForm(FiniteElementSpace *fes)
   {
#ifdef MFEM_USE_MPI
      auto *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         return new ParNonlinearForm(pfes);
      }
      else
      {
         return new NonlinearForm(fes);
      }
#else
      return new NonlinearForm(fes);
#endif
   }
   static BilinearForm* newBilinearForm(FiniteElementSpace* fes)
   {
#ifdef MFEM_USE_MPI
      auto *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         return new ParBilinearForm(pfes);
      }
      else
      {
         return new BilinearForm(fes);
      }
#else
      return new BilinearForm(fes);
#endif
   }
   static LinearForm* newLinearForm(FiniteElementSpace* fes)
   {
#ifdef MFEM_USE_MPI
      auto *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         return new ParLinearForm(pfes);
      }
      else
      {
         return new LinearForm(fes);
      }
#else
      return new LinearForm(fes);
#endif
   }
   static LinearForm* newLinearForm(FiniteElementSpace* fes, double *data)
   {
#ifdef MFEM_USE_MPI
      auto *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         return new ParLinearForm(pfes, data);
      }
      else
      {
         return new LinearForm(fes, data);
      }
#else
      return new LinearForm(fes, data);
#endif
   }
   static GridFunction* newGridFunction(FiniteElementSpace* fes)
   {
#ifdef MFEM_USE_MPI
      auto *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         return new ParGridFunction(pfes);
      }
      else
      {
         return new GridFunction(fes);
      }
#else
      return new GridFunction(fes);
#endif
   }
   static GridFunction* newGridFunction(FiniteElementSpace* fes, double *data)
   {
#ifdef MFEM_USE_MPI
      auto *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         return new ParGridFunction(pfes, data);
      }
      else
      {
         return new GridFunction(fes, data);
      }
#else
      return new GridFunction(fes, data);
#endif
   }
   static GridFunction* newGridFunction(FiniteElementSpace* fes, Vector &base,
                                        int offset=0)
   {
#ifdef MFEM_USE_MPI
      auto *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pfes)
      {
         return new ParGridFunction(pfes, base, offset);
      }
      else
      {
         return new GridFunction(fes, base, offset);
      }
#else
      return new GridFunction(fes, base, offset);
#endif
   }
};
class Scalar2ScalarMappedGF : public Coefficient
{
public:
   Scalar2ScalarMappedGF(GridFunction *gf,
                         std::function<double(double, ElementTransformation &, const IntegrationPoint &)>
                         fun,
                         int comp=1)
      :gfc(gf, comp), fun(fun) {}
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return fun(gfc.Eval(T, ip), T, ip);
   }
protected:
private:
   GridFunctionCoefficient gfc;
   std::function<double(double, ElementTransformation &, const IntegrationPoint &)>
   fun;
};
class Vector2ScalarMappedGF : public Coefficient
{
public:
   Vector2ScalarMappedGF(GridFunction *gf,
                         std::function<double(Vector&, ElementTransformation &, const IntegrationPoint &)>
                         fun)
      :gfc(gf), fun(fun), gf_vdim(gf->VectorDim())
   {
#ifndef MFEM_THREAD_SAFE
      val.SetSize(gf_vdim);
#endif
   }
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
#ifdef MFEM_THREAD_SAFE
      Vector val(gf_vdim);
#endif
      gfc.Eval(val, T, ip);
      return fun(val, T, ip);
   }
protected:
private:
#ifndef MFEM_THREAD_SAFE
   Vector val;
#endif
   VectorGridFunctionCoefficient gfc;
   const int gf_vdim;
   std::function<double(Vector&, ElementTransformation &, const IntegrationPoint &)>
   fun;
};



class Scalar2VectorMappedGF : public VectorCoefficient
{
public:
   Scalar2VectorMappedGF(GridFunction *gf,
                         std::function<void(Vector&, double, ElementTransformation &, const IntegrationPoint &)>
                         fun, int vd, int comp=1)
      :VectorCoefficient(vd), gfc(gf, comp), fun(fun) {}
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      V.SetSize(vdim);
      fun(V, gfc.Eval(T, ip), T, ip);
      return;
   }
protected:
private:
   GridFunctionCoefficient gfc;
   std::function<void(Vector&, double, ElementTransformation &, const IntegrationPoint &)>
   fun;
};
class Vector2VectorMappedGF : public VectorCoefficient
{
public:
   Vector2VectorMappedGF(GridFunction *gf,
                         std::function<void(Vector&, const Vector&, ElementTransformation &, const IntegrationPoint &)>
                         fun, int vd)
      :VectorCoefficient(vd), gfc(gf), fun(fun), gf_vdim(gf->VectorDim())
   {
#ifndef MFEM_THREAD_SAFE
      val.SetSize(gf_vdim);
#endif
   }
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      V.SetSize(vdim);
#ifdef MFEM_THREAD_SAFE
      Vector val(gf_vdim);
#endif
      gfc.Eval(val, T, ip);
      fun(V, val, T, ip);
      return;
   }
protected:
private:
#ifndef MFEM_THREAD_SAFE
   Vector val;
#endif
   VectorGridFunctionCoefficient gfc;
   const int gf_vdim;
   std::function<void(Vector&, const Vector&, ElementTransformation &, const IntegrationPoint &)>
   fun;
};

class Scalar2MatrixMappedGF : public MatrixCoefficient
{
public:
   Scalar2MatrixMappedGF(GridFunction *gf,
                         std::function<void(DenseMatrix&, double)> fun, int h, int w, int comp=1)
      :MatrixCoefficient(h,w), gfc(gf, comp), fun(fun) {}
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      M.SetSize(height, width);
      fun(M, gfc.Eval(T, ip));
      return;
   }
protected:
private:
   GridFunctionCoefficient gfc;
   std::function<void(DenseMatrix&, double)> fun;
};
class Vector2MatrixMappedGF : public MatrixCoefficient
{
public:
   Vector2MatrixMappedGF(GridFunction *gf,
                         std::function<void(DenseMatrix&, const Vector&, ElementTransformation &, const IntegrationPoint &)>
                         fun, int h, int w)
      :MatrixCoefficient(h, w), gfc(gf), fun(fun), gf_vdim(gf->VectorDim())
   {
#ifndef MFEM_THREAD_SAFE
      val.SetSize(gf_vdim);
#endif
   }
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      M.SetSize(height, width);
#ifdef MFEM_THREAD_SAFE
      Vector val(gf_vdim);
#endif
      gfc.Eval(val, T, ip);
      fun(M, val, T, ip);
      return;
   }
protected:
private:
#ifndef MFEM_THREAD_SAFE
   Vector val;
#endif
   VectorGridFunctionCoefficient gfc;
   const int gf_vdim;
   std::function<void(DenseMatrix&, const Vector&, ElementTransformation &, const IntegrationPoint &)>
   fun;
};

class ProxGalerkinAlphaMaker
{
public:
   ProxGalerkinAlphaMaker() {}
   virtual double GetAlpha(const int k) {mfem_error("Not implemented"); return 0.0;}
};

class ProxGalerkinPolynomialAlphaMaker : public ProxGalerkinAlphaMaker
{
public:
   ProxGalerkinPolynomialAlphaMaker(const double alpha0,
                                    const double degree):alpha0(alpha0), degree(degree),
      ProxGalerkinAlphaMaker() {}
   virtual double GetAlpha(const int k) {return alpha0*std::pow(k, degree);}
protected:
   const double alpha0;
   const double degree;
};

class ProxGalerkinExponentialAlphaMaker : public ProxGalerkinAlphaMaker
{
public:
   ProxGalerkinExponentialAlphaMaker(const double alpha0,
                                     const double base):alpha0(alpha0), base(base),
      ProxGalerkinAlphaMaker() {}
   virtual double GetAlpha(const int k) {return alpha0*std::pow(base, k);}
protected:
   const double alpha0;
   const double base;
};

class ProxGalerkinHCL : public TimeDependentOperator
{
private:
   // Vector finite element space containing conserved variables
   FiniteElementSpace *fes;
   FiniteElementSpace *vfes;
   // Element integration form. Should contain ComputeFlux
   HyperbolicFormIntegrator &formIntegrator;
   // Base Nonlinear Form
   NonlinearForm *nonlinearForm;
   // element-wise inverse mass matrix
   // std::vector<DenseMatrix> Me_inv;
   BilinearForm *M_inv;
   // global maximum characteristic speed. Updated by form integrators
   mutable double max_char_speed;
   // auxiliary variable used in Mult
   mutable Vector z;
   ProxGalerkinAlphaMaker alphamaker;
   GridFunction *latent_k, *latent, *delta_latent, *delta_dxdt;
   BilinearForm *M;
   NonlinearForm *latentMinv;
   Vector2VectorMappedGF expDiffLatent;
   const int dim;
   const int num_equations;
   const int maxit = 1e03;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();

public:
   /**
    * @brief Construct a new ProxGalerkinHCL object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ (F(u,x), grad v) and (F̂(u±, x, n), [[v]])
    * @param num_equations_ the number of equations
    */
   ProxGalerkinHCL(
      FiniteElementSpace *fes,
      FiniteElementSpace *vfes,
      HyperbolicFormIntegrator &formIntegrator,
      const int num_equations,
      ProxGalerkinAlphaMaker alphamaker);
   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   virtual void Mult(const Vector &x, Vector &y) const;
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
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &dxdt);

   virtual ~ProxGalerkinHCL() {}
};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class ProxGalerkinHCL
ProxGalerkinHCL::ProxGalerkinHCL(
   FiniteElementSpace *fes,
   FiniteElementSpace *vfes,
   HyperbolicFormIntegrator &formIntegrator,
   const int num_equations,
   ProxGalerkinAlphaMaker alphamaker)
   : fes(fes),
     vfes(vfes),
     formIntegrator(formIntegrator),
     M_inv(nullptr),
     z(vfes->GetVSize()),
     dim(vfes->GetFE(0)->GetDim()),
     num_equations(num_equations),
     alphamaker(alphamaker),
     latent_k(MFEMNew::newGridFunction(vfes)),
     latent(MFEMNew::newGridFunction(vfes)),
     delta_dxdt(MFEMNew::newGridFunction(vfes)),
     delta_latent(MFEMNew::newGridFunction(vfes)),
     expDiffLatent(latent, [](Vector &y, const Vector &x, ElementTransformation &T,
   const IntegrationPoint &ip) {y = x; y.ApplyMap([](double x) {return std::exp(x)*(1-x);} ); },
latent->VectorDim()),
TimeDependentOperator(vfes->GetVSize())
{
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();
   nonlinearForm = MFEMNew::newNonlinearForm(vfes);
   formIntegrator.resetMaxCharSpeed();

   nonlinearForm->AddDomainIntegrator(&formIntegrator);
   nonlinearForm->AddInteriorFaceIntegrator(&formIntegrator);
   latentMinv = MFEMNew::newNonlinearForm(vfes);
   latentMinv->AddDomainIntegrator(new InverseIntegrator(new
                                                         VectorMassIntegrator(expDiffLatent)));
}

void ProxGalerkinHCL::ComputeInvMass()
{
   if (M_inv) { delete M_inv; }
   if (M) {delete M; }
   M_inv = MFEMNew::newBilinearForm(vfes);
   M_inv->AddDomainIntegrator(new InverseIntegrator(new VectorMassIntegrator()));
   M_inv->Assemble();
   M = MFEMNew::newBilinearForm(vfes);
   M->AddDomainIntegrator(new VectorMassIntegrator());
   M->Assemble();
}

void ProxGalerkinHCL::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   formIntegrator.resetMaxCharSpeed();
   // 1. Create the vector z with the face terms (F(u), grad v) - <F.n(u), [w]>.
   nonlinearForm->Mult(x, z);
   max_char_speed = formIntegrator.getMaxCharSpeed();

   // 2. Multiply element-wise by the inverse mass matrices.
   int dof = M_inv->Size();
   Vector zval(z.GetData(), dof);
   Vector yval(y.GetData(), dof);
   for (int i=0; i<num_equations; i++)
   {
      zval.SetData(z.GetData() + i*dof);
      yval.SetData(y.GetData() + i*dof);
      M_inv->Mult(zval, yval);
   }
}

void ProxGalerkinHCL::ImplicitSolve(const double dt, const Vector &x,
                                    Vector &dxdt)
{
   int k;
   GridFunction *xnew = MFEMNew::newGridFunction(vfes);
   Mult(*xnew, dxdt);
   const double alpha = alphamaker.GetAlpha(k);
   dxdt.Add(alpha, *latent_k);
   dxdt.Add(-alpha, *latent);




}

} // namespace mfem