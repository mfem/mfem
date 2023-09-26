//                  MFEM Example 38
//
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <functional>

namespace mfem

{

double inv_sigmoid(double x, const double tol=1e-12)
{
   x = std::min(std::max(tol, x), 1.0 - tol);
   return std::log(x / (1.0 - x));
}

double sigmoid(double x)
{
   return x>=0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

double der_sigmoid(double x)
{
   double tmp = sigmoid(-x);
   return tmp - std::pow(tmp, 2);
}
double simp_rule(double x, const double min_val, const double max_val,
                 const double exponent)
{
   return min_val + pow(x, exponent)*(max_val - min_val);
}
double der_simp_rule(double x, const double min_val, const double max_val,
                     const double exponent)
{
   return exponent*pow(x, exponent - 1)*(max_val - min_val);
}


/**
 * @brief Returns f(u(x)) where u is a scalar GridFunction and f:R → R
 *
 */
class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   std::function<double(const double)> fun; // f:R → R
public:
   MappedGridFunctionCoefficient()
      :GridFunctionCoefficient(),
       fun([](double x) {return x;}) {}
   MappedGridFunctionCoefficient(const GridFunction *gf,
                                 std::function<double(const double)> fun_,
                                 int comp=1)
      :GridFunctionCoefficient(gf, comp),
       fun(fun_) {}


   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return fun(GridFunctionCoefficient::Eval(T, ip));
   }
   void SetFunction(std::function<double(const double)> fun_) { fun = fun_; }
};

MappedGridFunctionCoefficient SIMPCoefficient(GridFunction *gf,
                                              const double min_val=1e-06, const double max_val=1.0, const double exponent = 3)
{
   return MappedGridFunctionCoefficient(gf, [=](double x) {return simp_rule(x, min_val, max_val, exponent);});
}
class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient * lambda=nullptr;
   Coefficient * mu=nullptr;
   GridFunction *u = nullptr; // displacement
   GridFunction *rho_filter = nullptr; // filter density
   DenseMatrix grad; // auxiliary matrix, used in Eval
   double exponent;
   double rho_min;

public:
   StrainEnergyDensityCoefficient(Coefficient *lambda_, Coefficient *mu_,
                                  GridFunction * u_, GridFunction * rho_filter_, double rho_min_=1e-6,
                                  double exponent_ = 3.0)
      : lambda(lambda_), mu(mu_),  u(u_), rho_filter(rho_filter_),
        exponent(exponent_), rho_min(rho_min_)
   {
      MFEM_ASSERT(rho_min_ >= 0.0, "rho_min must be >= 0");
      MFEM_ASSERT(rho_min_ < 1.0,  "rho_min must be > 1");
      MFEM_ASSERT(u, "displacement field is not set");
      MFEM_ASSERT(rho_filter, "density field is not set");
   }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      double L = lambda->Eval(T, ip);
      double M = mu->Eval(T, ip);
      u->GetVectorGradient(T, grad);
      double div_u = grad.Trace();
      double density = L*div_u*div_u;
      int dim = T.GetSpaceDim();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            density += M*grad(i,j)*(grad(i,j)+grad(j,i));
         }
      }
      double val = rho_filter->GetValue(T,ip);

      return -exponent * pow(val, exponent-1.0) * (1-rho_min) * density;
   }
};


/**
 * @brief Returns f(u(x)) - f(v(x)) where u, v are scalar GridFunctions and f:R → R
 *
 */
class DiffMappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   const GridFunction *OtherGridF;
   GridFunctionCoefficient OtherGridF_cf;
   std::function<double(const double)> fun; // f:R → R
public:
   DiffMappedGridFunctionCoefficient()
      :GridFunctionCoefficient(),
       OtherGridF(nullptr),
       OtherGridF_cf(),
       fun([](double x) {return x;}) {}
   DiffMappedGridFunctionCoefficient(const GridFunction *gf,
                                     const GridFunction *other_gf,
                                     std::function<double(const double)> fun_,
                                     int comp=1)
      :GridFunctionCoefficient(gf, comp),
       OtherGridF(other_gf),
       OtherGridF_cf(OtherGridF),
       fun(fun_) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      const double value1 = fun(GridFunctionCoefficient::Eval(T, ip));
      const double value2 = fun(OtherGridF_cf.Eval(T, ip));
      return value1 - value2;
   }
   void SetFunction(std::function<double(const double)> fun_) { fun = fun_; }
};
class EllipticSolver
{
public:
   EllipticSolver(FiniteElementSpace &fes):fes(fes), parallel(false),
      ess_bdr_list(0)
   {
      AllocateBilinearForm();
      AllocateLinearForm();
   }

   void AllocateBilinearForm()
   {
#ifdef MFEM_USE_MPI
      pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
      if (pfes)
      {
         a = new ParBilinearForm(pfes);
         parallel = true;
      }
      else
      {
         a = new BilinearForm(&fes);
      }
#else
      a = new BilinearForm(&fes);
#endif
   }

   void AllocateLinearForm()
   {
#ifdef MFEM_USE_MPI
      pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
      if (pfes)
      {
         b = new ParLinearForm(pfes);
         parallel = true;
      }
      else
      {
         b = new LinearForm(&fes);
      }
#else
      b = new LinearForm(&fes);
#endif
   }

   BilinearForm* GetBilinearForm()
   {
      return a;
   }
   LinearForm* GetLinearForm()
   {
      return b;
   }
   void SetBilinearForm(BilinearForm* new_a)
   {
      MFEM_VERIFY(&fes == new_a->FESpace(),
                  "The given bilinear form is not associated with the object's finite element space.");
      a = new_a;
   }
   void SetLinearForm(LinearForm* new_b)
   {
      MFEM_VERIFY(&fes == new_b->FESpace(),
                  "The given linear form is not associated with the object's finite element space.");
      b = new_b;
   }
   void SetEssBdr(Array<int> &essential_bdr_list)
   {
      ess_bdr_list.SetSize(essential_bdr_list.Size());
      ess_bdr_list = essential_bdr_list;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         pfes->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list);
      }
      else
      {
         fes.GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list);
      }
#else
      fes.GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list);
#endif
   }
   void Update()
   {
      fes.Update();
      a->Update();
      b->Update();
      fes.GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list);
   }

   void AssembleLHS()
   {
      a->AllocateMatrix();
      a->Assemble();
   }

   void AssembleRHS()
   {
      b->Assemble();
   }

   void Assemble()
   {
      AssembleLHS();
      AssembleRHS();
   }

   void Solve(GridFunction *x)
   {
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         MFEM_VERIFY(static_cast<ParGridFunction*>(x)->ParFESpace() == pfes,
                     "The gridfunction is not associated with the given finite element space.");

      }
      else
      {
         MFEM_VERIFY(x->FESpace() == &fes,
                     "The gridfunction is not associated with the given finite element space.");
      }
#else
      MFEM_VERIFY(x.FESpace() == &fes,
                  "The gridfunction is not associated with the given finite element space.");
#endif

      OperatorPtr A;
      Vector X, B;
      a->FormLinearSystem(ess_tdof_list, *x, *b, A, X, B, true);

      CGSolver * cg;
      Solver * M;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         M = new HypreBoomerAMG;
         dynamic_cast<HypreBoomerAMG*>(M)->SetPrintLevel(0);
         cg = new CGSolver(pfes->GetComm());
      }
      else
      {
         M = new GSSmoother((SparseMatrix&)(*A));
         cg = new CGSolver;
      }
#else
      M = new GSSmoother((SparseMatrix&)(*A));
      cg = new CGSolver;
#endif
      cg->SetRelTol(1e-12);
      cg->SetMaxIter(10000);
      cg->SetPrintLevel(0);
      cg->SetPreconditioner(*M);
      cg->SetOperator(*A);
      cg->Mult(B, X);
      delete M;
      delete cg;
      a->RecoverFEMSolution(X, *b, *x);
   }
protected:
   FiniteElementSpace &fes;
   BilinearForm *a;
   LinearForm *b;
   Array<int> ess_tdof_list, ess_bdr_list;
private:
   bool parallel;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes;
#endif
};





class ElasticitySolver : public EllipticSolver
{
public:
   ElasticitySolver(FiniteElementSpace &fes): EllipticSolver(fes) {}
   ElasticitySolver(FiniteElementSpace &fes, Coefficient &lambda, Coefficient &mu,
                    VectorCoefficient &f)
      : EllipticSolver(fes)
   {
      AllocateBilinearForm();
      AllocateLinearForm();
      SetLameCoefficient(lambda, mu);
      SetForce(f);
   }
   void SetLameCoefficient(Coefficient &lambda, Coefficient &mu)
   {
      if (a->GetDBFI()->Size())
      {
         delete a;
         AllocateBilinearForm();
      }
      a->AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
   }
   void SetForce(VectorCoefficient &f)
   {
      if (b->GetDLFI()->Size())
      {
         delete b;
         AllocateLinearForm();
      }
      b->AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   }
protected:
private:
};

class FilterSolver : public EllipticSolver
{
public:
   FilterSolver(FiniteElementSpace &fes, Coefficient &eps, Coefficient &c,
                Coefficient &rho)
      : EllipticSolver(fes)
   {
      SetDiffusionCoefficient(eps, c);
      SetRho(rho);
   }
   void SetDiffusionCoefficient(Coefficient &eps, Coefficient &c)
   {
      if (a->GetDBFI()->Size())
      {
         delete a;

      }
      a->AddDomainIntegrator(new MassIntegrator(c));
      a->AddDomainIntegrator(new DiffusionIntegrator(eps));
   }
   void SetRho(Coefficient &rho)
   {
      Array<LinearFormIntegrator*> *forms = b->GetDLFI();
      forms->DeleteAll();
      b->AddDomainIntegrator(new DomainLFIntegrator(rho));
   }
protected:
private:
};

class ObjectiveFunction
{
public:
   ObjectiveFunction(FiniteElementSpace *fes) : ownData(true)
   {
      fes_arr = new Array<FiniteElementSpace*>(1);
      fes_arr[0] = fes;
   };
   ObjectiveFunction(Array<FiniteElementSpace*> &fes_arr): fes_arr(&fes_arr),
      ownData(false) {}
   ~ObjectiveFunction()
   {
      if (ownData)
      {
         for (auto &data : *fes_arr) {data = nullptr;}
         delete *fes_arr;
      }
   }

   virtual double Eval(const Vector &x)
   {
      mfem_error("Eval is not implemented");
   }
   virtual void Gradient(Vector &gradient)
   {
      mfem_error("Gradient is not implemented");
   }
   virtual void Gradient(const Vector &x, Vector &gradient)
   {
      Eval(x);
      Gradient(gradient);
   }
   virtual void DescentDirection(const Vector &x, Vector &direction)
   {
      Eval(x);
      Gradient(direction);
   }
   virtual void DescentDirection(Vector &direction)
   {
      Gradient(direction);
   }
protected:
   Array<FiniteElementSpace*>*fes_arr;
   bool ownData;
private:
};

class LVPGTopOpt : public ObjectiveFunction
{
public:
   LVPGTopOpt(FiniteElementSpace *state_fes, FiniteElementSpace *control_fes,
              FiniteElementSpace *filter_fes) : ObjectiveFunction(control_fes)
   {
      u = GridFunction(state_fes);
      filtered_rho = GridFunction(filter_fes);
      lambda = GridFunction(control_fes);
      filtered_lambda = GridFunction(filter_fes);
      rho = MappedGridFunctionCoefficient(&psi, sigmoid);
   }
   GridFunction &GetU() {return u;}
   GridFunction &GetPsi() {return psi;}
   GridFunction &GetFilteredRho() {return filtered_rho;}
   GridFunction &GetLambda() {return lambda;}
   GridFunction &GetFilteredLambda() {return filtered_lambda;}

   double Eval(const Vector &x)
   {
   }
   void Gradient(Vector &direction)
   {
   }

protected:
private:
   GridFunction u;
   GridFunction psi;
   MappedGridFunctionCoefficient rho;
   GridFunction filtered_rho;
   GridFunction lambda;
   GridFunction filtered_lambda;
};

class EBLineSearch
{
public:
   EBLineSearch(ObjectiveFunction &f, double c1, double c2): f(f), c1(c1), c2(c2)
   {
      MFEM_VERIFY(c1 > 0, "c1 should be positive");
      MFEM_VERIFY(c2 > c1, "c2 should be larger than c1");
      MFEM_VERIFY(c2 < 1, "c2 should be smaller than 1");
   }

   double search(const Vector &x, Vector &next)
   {
      Vector dfdx(x.Size()), direction(x.Size());
      double fx = f.Eval(x);
      f.Gradient(dfdx);
      f.DescentDirection(direction);
      double df = InnerProduct(direction, dfdx);
      double c1df = c1*df;
      double c2df = c2*df;
      bool searching = true;
      double a = 1.0;
      double olda;
      double da = a;
      double U = infinity();
      double L = 0;
      next = x;
      while (searching)
      {
         next.Add(da, direction);
         olda = a;
         if (f.Eval(next) > fx + a*c1df) // if not decreasing enough
         {
            // Modify search interval to (L, U=a)
            U = a;
            a = (U + L) / 2;
         }
         else
         {
            f.Gradient(dfdx);
            if (InnerProduct(dfdx, direction) < c2df) // if 
            {
               L = a;
               if (U == infinity())
               {
                  a = 2*L;
               }
               else
               {
                  a = (L + U) / 2;
               }
            }
            else
            {
               searching = false;
            }
         }
         da = a - olda;
      }
      return a;

   }
protected:
private:
   ObjectiveFunction &f;
   double c1, c2;
};



} // end of namespace mfem
