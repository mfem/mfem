#ifndef MFEM_EFEM_HPP
#define MFEM_EFEM_HPP

#include "mfem.hpp"


namespace mfem
{
/**
 * @brief Inverse sigmoid, log(x/(1-x))
 *
 * @param x -
 * @param tol tolerance to force x ∈ (tol, 1 - tol)
 * @return double log(x/(1-x))
 */
double invsigmoid(const double x, const double tol=1e-12)
{
   // forcing x to be in (0, 1)
   const double clipped_x = min(max(tol,x),1.0-tol);
   return log(clipped_x/(1.0-clipped_x));
}

// Sigmoid function
double sigmoid(const double x)
{
   return x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}

// Derivative of sigmoid function d(sigmoid)/dx
double dsigdx(const double x)
{
   double tmp = sigmoid(-x);
   return tmp - pow(tmp,2);
}


/**
 * @brief A coefficient that maps u to f(u).
 *
 */
class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
   // lambda function maps double to double
   typedef std::__1::function<double(const double)> __LambdaFunction;
private:
   __LambdaFunction fun; // a lambda function f(u(x))
public:
   /**
    * @brief Construct a mapped grid function coefficient with given gridfunction and lambda function
    *
    * @param[in] gf u
    * @param[in] double_to_double lambda function, f(x)
    * @param[in] comp (Optional) index of a vector if u is a vector
    */
   MappedGridFunctionCoefficient(const GridFunction *gf,
                                 __LambdaFunction double_to_double, int comp = 1): GridFunctionCoefficient(gf,
                                          comp), fun(double_to_double) {}

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      const double value = GridFunctionCoefficient::Eval(T, ip);
      return fun(value);
   }
};

/**
 * @brief GridFunctionCoefficient that returns exp(u)
 *
 */
class ExponentialGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   ExponentialGridFunctionCoefficient(const GridFunction *gf,
                                      int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return exp(x);},
   comp) {}
};


/**
 * @brief GridFunctionCoefficient that returns log(u)
 *
 */
class LogarithmicGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   LogarithmicGridFunctionCoefficient(const GridFunction *gf,
                                      int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return log(x);},
   comp) {}
};


/**
 * @brief GridFunctionCoefficient that returns sigmoid(u)
 *
 */
class SigmoidGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   SigmoidGridFunctionCoefficient(const GridFunction *gf,
                                  int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return sigmoid(x);},
   comp) {}
};


/**
 * @brief GridFunctionCoefficient that returns dsigdx(u) = sigmoid'(u)
 *
 */
class DerSigmoidGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   DerSigmoidGridFunctionCoefficient(const GridFunction *gf,
                                     int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return dsigdx(x);},
   comp) {}
};


/**
 * @brief GridFunctionCoefficient that returns invsigmoid(u)
 *
 */
class InvSigmoidGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   InvSigmoidGridFunctionCoefficient(const GridFunction *gf,
                                     int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return invsigmoid(x);},
   comp) {}
};


/**
 * @brief GridFunctionCoefficient that returns pow(u, exponent)
 *
 */
class PowerGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   PowerGridFunctionCoefficient(const GridFunction *gf, int exponent, int comp=1)
      : MappedGridFunctionCoefficient(gf, [exponent](double x) {return pow(x, exponent);},
   comp) {}
};


/**
 * @brief GridFunctionCoefficient that returns u^2
 *
 */
class PowerGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   PowerGridFunctionCoefficient(const GridFunction *gf, int exponent, int comp=1)
      : MappedGridFunctionCoefficient(gf, [](const double x) {return x*x;},
   comp) {}
};

/**
 * @brief SIMP Rule, r(ρ) = ρ_0 + (1-ρ_0)ρ^p
 *
 */
class SIMPCoefficient : public MappedGridFunctionCoefficient
{
public:
   /**
    * @brief Make a GridFunctionCoefficient that computes r(ρ) = ρ_0 + (1-ρ_0)ρ^p
    *
    * @param gf Density, ρ
    * @param exponent Exponent, p
    * @param rho_min minimum density, ρ_0
    */
   SIMPCoefficient(const const GridFunction *gf, const double exponent,
                   const double rho_min=1e-12)
      : MappedGridFunctionCoefficient(gf, [rho_min, exponent](const double x) {return rho_min + (1-rho_min)*pow(x, exponent);}) {}
};

/**
 * @brief Derivative of SIMP Rule, r'(ρ) = p(1-ρ_0)ρ^(p-1). Used when computing RHS
 *
 */
class SIMPDerCoefficient : public MappedGridFunctionCoefficient
{
public:
   /**
    * @brief Make a GridFunctionCoefficient that computes r'(ρ) = p(1-ρ_0)ρ^(p-1)
    *
    * @param gf Density, ρ
    * @param exponent Exponent, p
    * @param rho_min minimum density, ρ_0
    */
   SIMPDerCoefficient(const const GridFunction *gf, const double exponent,
                      const double rho_min=1e-12)
      : MappedGridFunctionCoefficient(gf, [rho_min, exponent](const double x) {return exponent*(1-rho_min)*pow(x, exponent - 1.0);}) {}
};


class SigmoidDensityProjector
{
private:
   FiniteElementSpace *fes;
   Mesh *mesh;
   const double target_volue;
   SigmoidGridFunctionCoefficient *rho = nullptr;
   DerSigmoidGridFunctionCoefficient *dsigPsi = nullptr;
   LinearForm *intRho = nullptr;
   LinearForm *intDerSigPsi = nullptr;


public:
   /**
    * @brief Projector Π : ψ → ψ + c so that ∫ ρ = θ|Ω| where ρ = sigmoid(ψ + c)
    *
    * @param fespace Finite element space for ψ
    * @param volume_fraction Volume fraction, θ
    * @param volume Total volume of the domain, |Ω|
    */
   SigmoidDensityProjector(FiniteElementSpace *fespace,
                           const double volume_fraction,
                           const double volume)
      :fes(fespace),
       mesh(fespace->GetMesh()),
       target_volue(volume_fraction*volume) {}

   /**
    * @brief Update ψ ↦ ψ + c so that ∫ ρ = θ |Ω|.
    *
    * Using Newton's method, find c such that
    *
    * ∫ sigmoid(ψ + c) = θ |Ω|
    *
    * @param psi ρ = sigmoid(ψ)
    * @param max_iteration Maximum iteration for Newton iteration
    * @param tolerance Newton update tolerance
    */
   void Apply(GridFunction &psi, const int max_iteration,
              const double tolerance=1e-12)
   {
      // 0. Make or Update Helper objects

      if (rho) // if helper objects are already created,
      {
         // update with the current GridFunction
         rho->SetGridFunction(&psi);
         dsigPsi->SetGridFunction(&psi);
      }
      else // if Apply is not called at all
      {
         // Create MappedGridFunctionCoefficients
         rho = new SigmoidGridFunctionCoefficient(&psi);
         dsigPsi = new DerSigmoidGridFunctionCoefficient(&psi);

         // Create ∫ sigmoid(ψ) and ∫ sigmoid'(ψ)
#ifdef MFEM_USE_MPI // if Using MPI,
         // try convert it to parallel version
         ParFiniteElementSpace * pfes = dynamic_cast<ParFiniteElementSpace *>(fes);
         if (pfes)
         {
            // make parallel linear forms
            intRho = new ParLinearForm(pfes);
            intDerSigPsi = new ParLinearForm(pfes);
         }
         else
         {
            // make serial linear forms
            intRho = new LinearForm(fes);
            intDerSigPsi = new LinearForm(fes);
         }
#else
         intRho = new LinearForm(fes);
         intDerSigPsi = new LinearForm(fes);
#endif
         intRho->AddDomainIntegrator(new DomainLFIntegrator(*rho));
         intDerSigPsi->AddDomainIntegrator(new DomainLFIntegrator(*dsigPsi));
      }

      // Newton Method
      for (int i=0; i<max_iteration; i++)
      {
         // Compute ∫ sigmoid(ψ + c)
         intRho->Assemble(); // necessary whenever ψ is updated
         const double f = intRho->Sum();
         // Compute ∫ sigmoid'(ψ)
         intDerSigPsi->Assemble();
         const double df = intDerSigPsi->Sum();

         // Newton increment
         const double dc = - f / df;
         // Update ψ
         psi += dc;

         if (abs(dc) < tolerance)
         {
            break;
         }
      }
   }
};

class EllipticSolver
{
private:
   FiniteElementSpace *fes; // finite element space
   BilinearForm *bilinForm; // main bilinear form
   Array<int> ess_tdof_list; // essential boundary dof list
   bool isParallel = false; // whether input fespace is parallel or not
   bool pa; // partial assembly flag
public:
   EllipticSolver(FiniteElementSpace *fespace,
                  BilinearForm *bilinearForm, Array<int> ess_bdr)
      :fes(fespace),
       bilinForm(bilinearForm)
   {
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
#ifdef MFEM_USE_MPI
      {
         ParFiniteElementSpace * pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
         if (pfes) { isParallel = true; }
      }
#endif
      pa = bilinForm->GetAssemblyLevel() == AssemblyLevel::PARTIAL;
   }


   void Assemble()
   {
      bilinForm->Assemble();
   }

   void Solve(LinearForm *b, GridFunction *sol)
   {
      OperatorPtr A;
      Vector B, X;
      b->Assemble();
      bilinForm->FormLinearSystem(ess_tdof_list, *sol, *b, A, X, B);

      // 11. Solve the linear system A X = B.
      if (!isParallel)
      {
         if (!pa)
         {
#ifndef MFEM_USE_SUITESPARSE
            // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
            GSSmoother M((SparseMatrix&)(*A));
            PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
#else
            // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
            UMFPackSolver umf_solver;
            umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
            umf_solver.SetOperator(*A);
            umf_solver.Mult(B, X);
#endif
         }
         else
         {
            if (UsesTensorBasis(*fes))
            {
               if (DeviceCanUseCeed())
               {
                  ceed::AlgebraicSolver M(*bilinForm, ess_tdof_list);
                  PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
               }
               else
               {
                  OperatorJacobiSmoother M(*bilinForm, ess_tdof_list);
                  PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
               }
            }
            else
            {
               CG(*A, B, X, 1, 400, 1e-12, 0.0);
            }
         }
      }
#ifdef MFEM_USE_MPI
      else
      {
         Solver *prec = NULL;
         if (pa)
         {
            if (UsesTensorBasis(*fes))
            {
               if (DeviceCanUseCeed)
               {
                  prec = new ceed::AlgebraicSolver(*bilinForm, ess_tdof_list);
               }
               else
               {
                  prec = new OperatorJacobiSmoother(*bilinForm, ess_tdof_list);
               }
            }
         }
         else
         {
            prec = new HypreBoomerAMG;
         }
         CGSolver cg(MPI_COMM_WORLD);
         cg.SetRelTol(1e-12);
         cg.SetMaxIter(2000);
         cg.SetPrintLevel(1);
         if (prec) { cg.SetPreconditioner(*prec); }
         cg.SetOperator(*A);
         cg.Mult(B, X);
         delete prec;
      }
#endif
      bilinForm->RecoverFEMSolution(X, *b, *sol);
   }
};
}

#endif