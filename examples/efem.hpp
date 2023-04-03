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
   const double clipped_x = std::min(std::max(tol,x),1.0-tol);
   return std::log(clipped_x/(1.0-clipped_x));
}

// Sigmoid function
double sigmoid(const double x)
{
   return x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

// Derivative of sigmoid function d(sigmoid)/dx
double dsigdx(const double x)
{
   double tmp = sigmoid(-x);
   return tmp - std::pow(tmp,2);
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
protected:
   std::string name = "NONE";
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
                                      int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return std::exp(x);},
   comp) {name = "EXP";}
};


/**
 * @brief GridFunctionCoefficient that returns log(u)
 *
 */
class LogarithmicGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   LogarithmicGridFunctionCoefficient(const GridFunction *gf,
                                      int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return std::log(x);},
   comp) {name = "LOG";}
};
/**
 * @brief GridFunctionCoefficient that returns log(max(u, tolerance))
 *
 */
class SafeLogarithmicGridFunctionCoefficient : public
   MappedGridFunctionCoefficient
{
public:
   SafeLogarithmicGridFunctionCoefficient(const GridFunction *gf,
                                          const double tolerance,
                                          int comp=1):MappedGridFunctionCoefficient(gf, [tolerance](
                                                      const double x) {return std::log(std::max(x, tolerance));},
   comp) {name = "SAFE LOG";}
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
   comp) {name = "SIGMOID";}
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
   comp) {name = "D(SIGMOID)/DX";}
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
   comp) {name = "INVSIGMOID";}
};


/**
 * @brief GridFunctionCoefficient that returns pow(u, exponent)
 *
 */
class PowerGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   PowerGridFunctionCoefficient(const GridFunction *gf, int exponent, int comp=1)
      : MappedGridFunctionCoefficient(gf, [exponent](double x) {return std::pow(x, exponent);},
   comp) {name = "POWER";}
};


/**
 * @brief GridFunctionCoefficient that returns u^2
 *
 */
class SquaredGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
public:
   SquaredGridFunctionCoefficient(const GridFunction *gf, int exponent, int comp=1)
      : MappedGridFunctionCoefficient(gf, [](const double x) {return x*x;},
   comp) {name = "SQUARE";}
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
   SIMPCoefficient(const GridFunction *gf, const double exponent,
                   const double rho_min=1e-12)
      : MappedGridFunctionCoefficient(gf, [rho_min, exponent](const double x) {return rho_min + (1-rho_min)*std::pow(x, exponent);}) {name = "SIMP";}
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
   SIMPDerCoefficient(const GridFunction *gf, const double exponent,
                      const double rho_min=1e-12)
      : MappedGridFunctionCoefficient(gf, [rho_min, exponent](const double x) {return exponent*(1-rho_min)*std::pow(x, exponent - 1.0);}) {name = "SIMPDER";}
};

/**
 * @brief Projector Π : ψ → ψ + c so that ∫ ρ = θ|Ω| where ρ = sigmoid(ψ + c)
 *
 */
class SigmoidDensityProjector
{
private:
   FiniteElementSpace *fes;
   Mesh *mesh;
   const double target_volume;
   SigmoidGridFunctionCoefficient *rho = nullptr; // ρ = sigmoid(ψ)
   DerSigmoidGridFunctionCoefficient *dsigPsi = nullptr; // d(sigmoid(ψ))/dψ
   LinearForm *intRho = nullptr; // ∫ ρ = ∫ sigmoid(ψ)
   LinearForm *intDerSigPsi = nullptr; // ∫ d(sigmoid(ψ))/dψ
   bool isParallel = false;

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
       target_volume(volume_fraction*volume) {}

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
   double Apply(GridFunction &psi, const int max_iteration,
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
            isParallel = true;
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
         intRho->AddDomainIntegrator(new DomainLFIntegrator(*rho, 2, 0));
         intDerSigPsi->AddDomainIntegrator(new DomainLFIntegrator(*dsigPsi, 2, 0));
      }

      // Newton Method
      for (int i=0; i<max_iteration; i++)
      {
         // Compute ∫ sigmoid(ψ + c)
         intRho->Assemble(); // necessary whenever ψ is updated
         double f = intRho->Sum();
         // Compute ∫ sigmoid'(ψ + c)
         intDerSigPsi->Assemble();
         double df = intDerSigPsi->Sum();

#ifdef MFEM_USE_MPI
         if (isParallel)
         {
            MPI_Allreduce(MPI_IN_PLACE, &f, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &df, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         }
#endif
         f -= target_volume;


         // Newton increment
         const double dc = - f / df;
         // Update ψ
         psi += dc;
         out << "Iteration: " << i << " (θ|Ω|, ∫ρ - θ|Ω|, Δc) = (" <<
             target_volume << ", " <<
             f << ", " << dc << ")" << std::endl;

         if (abs(dc) < tolerance)
         {
            break;
         }
         MFEM_VERIFY(std::isfinite(dc), "Projection failed");
      }
      intRho->Assemble();
      return intRho->Sum();
   }
};

class EllipticSolver
{
private:
   FiniteElementSpace *fes; // finite element space
   BilinearForm *bilinForm; // main bilinear form
   Array<int> ess_tdof_list; // essential boundary dof list
   bool isParallel = false; // whether input fespace is parallel or not
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes; // parallel
   ParMesh *pmesh;
#endif
   bool pa; // partial assembly flag
public:
   EllipticSolver(FiniteElementSpace *fespace,
                  BilinearForm *bilinearForm, Array<int> ess_bdr)
      :fes(fespace),
       bilinForm(bilinearForm)
   {
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      out << "Essential dof size: " << ess_tdof_list.Size() << std::endl;
#ifdef MFEM_USE_MPI
      {
         pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
         if (pfes) { isParallel = true; pmesh = pfes->GetParMesh();}
      }
#endif
      pa = bilinForm->GetAssemblyLevel() == AssemblyLevel::PARTIAL;
   }

   void Solve(LinearForm *b, GridFunction *sol)
   {
      OperatorPtr A;
      Vector B, X;
      bilinForm->FormLinearSystem(ess_tdof_list, *sol, *b, A, X, B);

      // 11. Solve the linear system A X = B.
      CGSolver * cg = nullptr;
      Solver * M = nullptr;
#ifdef MFEM_USE_MPI
      if (isParallel)
      {
         M = new HypreBoomerAMG;
         dynamic_cast<HypreBoomerAMG*>(M)->SetPrintLevel(0);
         cg = new CGSolver(pmesh->GetComm());
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
      bilinForm->RecoverFEMSolution(X, *b, *sol);
   }
};

//  Class for solving Poisson's equation:
//
//       - ∇ ⋅(κ ∇ u) = f  in Ω
//
class DiffusionSolver
{
private:
   Mesh * mesh = nullptr;
   // diffusion coefficient
   Coefficient * diffcf = nullptr;
   // mass coefficient
   Coefficient * masscf = nullptr;
   Coefficient * rhscf = nullptr;
   Coefficient * essbdr_cf = nullptr;
   Coefficient * neumann_cf = nullptr;
   VectorCoefficient * gradient_cf = nullptr;

   // FEM solver
   int dim;
   FiniteElementCollection * fec = nullptr;
   FiniteElementSpace * fes = nullptr;
   Array<int> ess_bdr;
   Array<int> neumann_bdr;
   LinearForm * b = nullptr;
   bool parallel = false;
#ifdef MFEM_USE_MPI
   ParMesh * pmesh = nullptr;
   ParFiniteElementSpace * pfes = nullptr;
#endif

public:
   DiffusionSolver() { }

   void SetMesh(Mesh * mesh_)
   {
      mesh = mesh_;
#ifdef MFEM_USE_MPI
      pmesh = dynamic_cast<ParMesh *>(mesh);
      if (pmesh) { parallel = true; }
#endif
   }
   void SetDiffusionCoefficient(Coefficient * diffcf_) { diffcf = diffcf_; }
   void SetMassCoefficient(Coefficient * masscf_) { masscf = masscf_; }
   void SetRHSCoefficient(Coefficient * rhscf_) { rhscf = rhscf_; }
   void SetEssentialBoundary(const Array<int> & ess_bdr_) { ess_bdr = ess_bdr_;};
   void SetNeumannBoundary(const Array<int> & neumann_bdr_) { neumann_bdr = neumann_bdr_;};
   void SetNeumannData(Coefficient * neumann_cf_) {neumann_cf = neumann_cf_;}
   void SetEssBdrData(Coefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}
   void SetGradientData(VectorCoefficient * gradient_cf_) {gradient_cf = gradient_cf_;}
   void SetFESpace(FiniteElementSpace * fespace)
   {
      fes = fespace;
#ifdef MFEM_USE_MPI
      pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      if (pmesh) {parallel = true;};
#endif
   }

   void ResetFEM();
   void SetupFEM();

   void Solve(GridFunction *u);
   LinearForm * GetLinearForm() {return b;}
#ifdef MFEM_USE_MPI
   ParLinearForm * GetParLinearForm()
   {
      if (parallel)
      {
         return dynamic_cast<ParLinearForm *>(b);
      }
      else
      {
         MFEM_ABORT("Wrong code path. Call GetLinearForm");
         return nullptr;
      }
   }
#endif

   ~DiffusionSolver();

};


void DiffusionSolver::SetupFEM()
{
   dim = mesh->Dimension();

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      b = new ParLinearForm(pfes);
   }
   else
   {
      b = new LinearForm(fes);
   }
#else
   fes = new FiniteElementSpace(mesh, fec);
   b = new LinearForm(fes);
#endif

   if (!ess_bdr.Size())
   {
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
   }
}

void DiffusionSolver::Solve(GridFunction *u)
{
   OperatorPtr A;
   Vector B, X;
   Array<int> ess_tdof_list;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      pfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
   else
   {
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
#else
   fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
#endif
   if (b)
   {
      delete b;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         b = new ParLinearForm(pfes);
      }
      else
      {
         b = new LinearForm(fes);
      }
#else
      b = new LinearForm(fes);
#endif
   }
   if (rhscf)
   {
      b->AddDomainIntegrator(new DomainLFIntegrator(*rhscf));
   }
   if (neumann_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryLFIntegrator(*neumann_cf),neumann_bdr);
   }
   else if (gradient_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*gradient_cf),
                               neumann_bdr);
   }

   b->Assemble();

   BilinearForm * a = nullptr;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      a = new ParBilinearForm(pfes);
   }
   else
   {
      a = new BilinearForm(fes);
   }
#else
   a = new BilinearForm(fes);
#endif
   a->AddDomainIntegrator(new DiffusionIntegrator(*diffcf));
   if (masscf)
   {
      a->AddDomainIntegrator(new MassIntegrator(*masscf));
   }
   a->Assemble();
   if (essbdr_cf)
   {
      u->ProjectBdrCoefficient(*essbdr_cf,ess_bdr);
   }
   a->FormLinearSystem(ess_tdof_list, *u, *b, A, X, B, 1);

   CGSolver * cg = nullptr;
   Solver * M = nullptr;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      M = new HypreBoomerAMG;
      dynamic_cast<HypreBoomerAMG*>(M)->SetPrintLevel(0);
      cg = new CGSolver(pmesh->GetComm());
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
   a->RecoverFEMSolution(X, *b, *u);
   delete a;
}

void DiffusionSolver::ResetFEM()
{
   delete fes; fes = nullptr;
   delete fec; fec = nullptr;
   delete b;
}


DiffusionSolver::~DiffusionSolver()
{
   ResetFEM();
}

}


#endif