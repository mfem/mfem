//                  MFEM Example 37 - Serial/Parallel Shared Code
#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <functional>

namespace mfem
{
void disp(Array<int> &x)
{
   for (auto v:x) { out << v << ", ";}
   out << std::endl;
}
void disp(Vector &x)
{
   for (auto v:x) { out << v << ", ";}
   out << std::endl;
}


/// @brief Inverse sigmoid function
double inv_sigmoid(const double x)
{
   const double tol = 1e-12;
   const double tmp = std::min(std::max(tol,x),1.0-tol);
   return std::log(tmp/(1.0-tmp));
}

/// @brief Sigmoid function
double sigmoid(const double x)
{
   return x>=0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x));
   // return (std::tanh(x*0.5) + 1.0) * 0.5;
}

/**
 * @brief Derivative of sigmoid function
 *
 */
double der_sigmoid(const double x)
{
   const double tmp = sigmoid(x);
   return tmp*(1.0 - tmp);
}

double simp(const double x, const double rho_min=1e-06, const double exp=3.0,
            const double rho_max=1.0)
{
   return rho_min + std::pow(x, exp) * (rho_max - rho_min);
}

double der_simp(const double x, const double rho_min=1e-06,
                const double exp=3.0, const double rho_max=1.0)
{
   return (exp - 1.0) * std::pow(x, exp - 1) * (rho_max - rho_min);
}



/// @brief Returns f(u(x)) - f(v(x)) where u, v are scalar GridFunctions and f:R → R
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
       fun([](const double x) {return x;}) {}
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


/// @brief Strain energy density coefficient
class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient * lambda=nullptr;
   Coefficient * mu=nullptr;
   GridFunction *u1 = nullptr; // displacement
   GridFunction *u2 = nullptr; // displacement
   GridFunction *rho_filter = nullptr; // filter density
   DenseMatrix grad1, grad2; // auxiliary matrix, used in Eval
   double exponent;
   double rho_min;

public:
   StrainEnergyDensityCoefficient(Coefficient *lambda_, Coefficient *mu_,
                                  GridFunction * u_, GridFunction * rho_filter_, double rho_min_=1e-6,
                                  double exponent_ = 3.0)
      : lambda(lambda_), mu(mu_),  u1(u_),  u2(nullptr), rho_filter(rho_filter_),
        exponent(exponent_), rho_min(rho_min_)
   {
      MFEM_ASSERT(rho_min_ >= 0.0, "rho_min must be >= 0");
      MFEM_ASSERT(rho_min_ < 1.0,  "rho_min must be > 1");
   }
   StrainEnergyDensityCoefficient(Coefficient *lambda_, Coefficient *mu_,
                                  GridFunction * u1_, GridFunction * u2_, GridFunction * rho_filter_,
                                  double rho_min_=1e-6,
                                  double exponent_ = 3.0)
      : lambda(lambda_), mu(mu_),  u1(u1_), u2(u2_), rho_filter(rho_filter_),
        exponent(exponent_), rho_min(rho_min_)
   {
      MFEM_ASSERT(rho_min_ >= 0.0, "rho_min must be >= 0");
      MFEM_ASSERT(rho_min_ < 1.0,  "rho_min must be > 1");
   }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      double L = lambda->Eval(T, ip);
      double M = mu->Eval(T, ip);
      double density;
      if (u2)
      {
         u1->GetVectorGradient(T, grad1);
         u2->GetVectorGradient(T, grad2);
         double div_u1 = grad1.Trace();
         double div_u2 = grad2.Trace();
         grad1.Symmetrize();

         Vector gradv1(grad1.GetData(), grad1.Width()*grad1.Height()),
                gradv2(grad2.GetData(), grad1.Width()*grad1.Height());

         density = L*div_u1*div_u2 + 2*M*InnerProduct(gradv1, gradv2);
      }
      else
      {
         u1->GetVectorGradient(T, grad1);
         double div_u = grad1.Trace();
         grad1.Symmetrize();
         density = L*div_u*div_u + 2*M*grad1.FNorm2();
      }
      double val = rho_filter->GetValue(T,ip);

      return -exponent * pow(val, exponent-1.0) * (1-rho_min) * density;
   }
   void SetDisplacement(GridFunction *u1_) { u1 = u1_; u2 = u1_;}
   void SetDisplacement(GridFunction *u1_, GridFunction *u2_) { u1 = u1_; u2 = u2_; }
   void SetFilteredDensity(GridFunction *frho) { rho_filter = frho; }
};

/// @brief Volumetric force for linear elasticity
class VolumeForceCoefficient : public VectorCoefficient
{
private:
   double r2;
   Vector &center;
   Vector &force;
public:
   VolumeForceCoefficient(double r_,Vector &  center_, Vector & force_) :
      VectorCoefficient(center_.Size()), r2(r_*r_), center(center_), force(force_) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector xx; xx.SetSize(T.GetDimension());
      T.Transform(ip,xx);
      double cr = xx.DistanceSquaredTo(center);
      V.SetSize(T.GetDimension());
      if (cr <= r2)
      {
         V = force;
      }
      else
      {
         V = 0.0;
      }
   }

   void Set(double r_,Vector & center_, Vector & force_)
   {
      r2=r_*r_;
      center = center_;
      force = force_;
   }
   void UpdateSize()
   {
      VectorCoefficient::vdim = center.Size();
   }
};

/// @brief Volumetric force for linear elasticity
class LineVolumeForceCoefficient : public VectorCoefficient
{
private:
   double r2;
   Vector &center;
   Vector &force;
   int direction_dim;
public:
   LineVolumeForceCoefficient(double r_, Vector &center_, Vector & force_,
                              int direction_dim) :
      VectorCoefficient(center_.Size()), r2(r_*r_), center(center_), force(force_),
      direction_dim(direction_dim) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector xx; xx.SetSize(T.GetDimension());
      T.Transform(ip,xx);
      xx(direction_dim) = 0.0;
      center(direction_dim) = 0.0;
      double cr = xx.DistanceSquaredTo(center);
      V.SetSize(T.GetDimension());
      if (cr <= r2)
      {
         V = force;
      }
      else
      {
         V = 0.0;
      }
   }

   void Set(double r_,Vector & center_, Vector & force_)
   {
      r2=r_*r_;
      center = center_;
      force = force_;
   }
   void UpdateSize()
   {
      VectorCoefficient::vdim = center.Size();
   }
};

/**
 * @brief Class for solving Poisson's equation:
 *
 *       - ∇ ⋅(κ ∇ u) = f  in Ω
 *
 */
class DiffusionSolver
{
private:
   Mesh * mesh = nullptr;
   int order = 1;
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
   GridFunction * u = nullptr;
   LinearForm * b = nullptr;
   bool parallel;
#ifdef MFEM_USE_MPI
   ParMesh * pmesh = nullptr;
   ParFiniteElementSpace * pfes = nullptr;
#endif

public:
   DiffusionSolver() { }
   DiffusionSolver(Mesh * mesh_, int order_, Coefficient * diffcf_,
                   Coefficient * cf_);

   void SetMesh(Mesh * mesh_)
   {
      mesh = mesh_;
      parallel = false;
#ifdef MFEM_USE_MPI
      pmesh = dynamic_cast<ParMesh *>(mesh);
      if (pmesh) { parallel = true; }
#endif
   }
   void SetOrder(int order_) { order = order_ ; }
   void SetDiffusionCoefficient(Coefficient * diffcf_) { diffcf = diffcf_; }
   void SetMassCoefficient(Coefficient * masscf_) { masscf = masscf_; }
   void SetRHSCoefficient(Coefficient * rhscf_) { rhscf = rhscf_; }
   void SetEssentialBoundary(const Array<int> & ess_bdr_) { ess_bdr = ess_bdr_;};
   void SetNeumannBoundary(const Array<int> & neumann_bdr_) { neumann_bdr = neumann_bdr_;};
   void SetNeumannData(Coefficient * neumann_cf_) {neumann_cf = neumann_cf_;}
   void SetEssBdrData(Coefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}
   void SetGradientData(VectorCoefficient * gradient_cf_) {gradient_cf = gradient_cf_;}

   void ResetFEM();
   void SetupFEM();

   void Solve();
   GridFunction * GetFEMSolution();
   LinearForm * GetLinearForm() {return b;}
#ifdef MFEM_USE_MPI
   ParGridFunction * GetParFEMSolution();
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

/**
 * @brief Class for solving linear elasticity:
 *
 *        -∇ ⋅ σ(u) = f  in Ω  + BCs
 *
 *  where
 *
 *        σ(u) = λ ∇⋅u I + μ (∇ u + ∇uᵀ)
 *
 */
class LinearElasticitySolver
{
private:
   Mesh * mesh = nullptr;
   int order = 1;
   Coefficient * lambda_cf = nullptr;
   Coefficient * mu_cf = nullptr;
   VectorCoefficient * essbdr_cf = nullptr;
   VectorCoefficient * rhs_cf = nullptr;

   // FEM solver
   int dim;
   FiniteElementCollection * fec = nullptr;
   FiniteElementSpace * fes = nullptr;
   Array<int> ess_bdr;
   Array<int> neumann_bdr;
   GridFunction * u = nullptr;
   LinearForm * b = nullptr;
   bool parallel;
#ifdef MFEM_USE_MPI
   ParMesh * pmesh = nullptr;
   ParFiniteElementSpace * pfes = nullptr;
#endif

public:
   LinearElasticitySolver() { }
   LinearElasticitySolver(Mesh * mesh_, int order_,
                          Coefficient * lambda_cf_, Coefficient * mu_cf_);

   void SetMesh(Mesh * mesh_)
   {
      mesh = mesh_;
      parallel = false;
#ifdef MFEM_USE_MPI
      pmesh = dynamic_cast<ParMesh *>(mesh);
      if (pmesh) { parallel = true; }
#endif
   }
   void SetOrder(int order_) { order = order_ ; }
   void SetLameCoefficients(Coefficient * lambda_cf_, Coefficient * mu_cf_) { lambda_cf = lambda_cf_; mu_cf = mu_cf_;  }
   void SetRHSCoefficient(VectorCoefficient * rhs_cf_) { rhs_cf = rhs_cf_; }
   void SetEssentialBoundary(const Array<int> & ess_bdr_) { ess_bdr = ess_bdr_;};
   void SetNeumannBoundary(const Array<int> & neumann_bdr_) { neumann_bdr = neumann_bdr_;};
   void SetEssBdrData(VectorCoefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}

   void ResetFEM();
   void SetupFEM();

   void Solve();
   GridFunction * GetFEMSolution();
   LinearForm * GetLinearForm() {return b;}
#ifdef MFEM_USE_MPI
   ParGridFunction * GetParFEMSolution();
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

   ~LinearElasticitySolver();

};


// Poisson solver

DiffusionSolver::DiffusionSolver(Mesh * mesh_, int order_,
                                 Coefficient * diffcf_, Coefficient * rhscf_)
   : mesh(mesh_), order(order_), diffcf(diffcf_), rhscf(rhscf_)
{

#ifdef MFEM_USE_MPI
   pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh) { parallel = true; }
#endif

   SetupFEM();
}

void DiffusionSolver::SetupFEM()
{
   dim = mesh->Dimension();
   fec = new H1_FECollection(order, dim);

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      pfes = new ParFiniteElementSpace(pmesh, fec);
      u = new ParGridFunction(pfes);
      b = new ParLinearForm(pfes);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, fec);
      u = new GridFunction(fes);
      b = new LinearForm(fes);
   }
#else
   fes = new FiniteElementSpace(mesh, fec);
   u = new GridFunction(fes);
   b = new LinearForm(fes);
#endif
   *u=0.0;

   if (!ess_bdr.Size())
   {
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
   }
}

void DiffusionSolver::Solve()
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
   *u=0.0;
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
   a->FormLinearSystem(ess_tdof_list, *u, *b, A, X, B, true);

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

GridFunction * DiffusionSolver::GetFEMSolution()
{
   return u;
}

#ifdef MFEM_USE_MPI
ParGridFunction * DiffusionSolver::GetParFEMSolution()
{
   if (parallel)
   {
      return dynamic_cast<ParGridFunction*>(u);
   }
   else
   {
      MFEM_ABORT("Wrong code path. Call GetFEMSolution");
      return nullptr;
   }
}
#endif

DiffusionSolver::~DiffusionSolver()
{
   delete u; u = nullptr;
   delete fes; fes = nullptr;
#ifdef MFEM_USE_MPI
   delete pfes; pfes=nullptr;
#endif
   delete fec; fec = nullptr;
   delete b;
}


// Elasticity solver

LinearElasticitySolver::LinearElasticitySolver(Mesh * mesh_, int order_,
                                               Coefficient * lambda_cf_, Coefficient * mu_cf_)
   : mesh(mesh_), order(order_), lambda_cf(lambda_cf_), mu_cf(mu_cf_)
{
#ifdef MFEM_USE_MPI
   pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh) { parallel = true; }
#endif
   SetupFEM();
}

void LinearElasticitySolver::SetupFEM()
{
   dim = mesh->Dimension();
   fec = new H1_FECollection(order, dim,BasisType::Positive);

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      pfes = new ParFiniteElementSpace(pmesh, fec, dim);
      u = new ParGridFunction(pfes);
      b = new ParLinearForm(pfes);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, fec,dim);
      u = new GridFunction(fes);
      b = new LinearForm(fes);
   }
#else
   fes = new FiniteElementSpace(mesh, fec, dim);
   u = new GridFunction(fes);
   b = new LinearForm(fes);
#endif
   *u=0.0;

   if (!ess_bdr.Size())
   {
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
   }
}

void LinearElasticitySolver::Solve()
{
   GridFunction * x = nullptr;
   OperatorPtr A;
   Vector B, X;
   Array<int> ess_tdof_list;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      x = new ParGridFunction(pfes);
      pfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
   else
   {
      x = new GridFunction(fes);
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
#else
   x = new GridFunction(fes);
   fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
#endif
   *u=0.0;
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
   if (rhs_cf)
   {
      b->AddDomainIntegrator(new VectorDomainLFIntegrator(*rhs_cf));
   }

   b->Assemble();

   *x = 0.0;

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
   a->AddDomainIntegrator(new ElasticityIntegrator(*lambda_cf, *mu_cf));
   a->Assemble();
   if (essbdr_cf)
   {
      u->ProjectBdrCoefficient(*essbdr_cf,ess_bdr);
   }
   a->FormLinearSystem(ess_tdof_list, *x, *b, A, X, B, true);

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
   cg->SetRelTol(1e-10);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->Mult(B, X);
   delete M;
   delete cg;
   a->RecoverFEMSolution(X, *b, *x);
   *u+=*x;
   delete a;
   delete x;
}

GridFunction * LinearElasticitySolver::GetFEMSolution()
{
   return u;
}

#ifdef MFEM_USE_MPI
ParGridFunction * LinearElasticitySolver::GetParFEMSolution()
{
   if (parallel)
   {
      return dynamic_cast<ParGridFunction*>(u);
   }
   else
   {
      MFEM_ABORT("Wrong code path. Call GetFEMSolution");
      return nullptr;
   }
}
#endif

LinearElasticitySolver::~LinearElasticitySolver()
{
   delete u; u = nullptr;
   delete fes; fes = nullptr;
#ifdef MFEM_USE_MPI
   delete pfes; pfes=nullptr;
#endif
   delete fec; fec = nullptr;
   delete b;
}

class EllipticSolver
{
public:
   EllipticSolver(BilinearForm *a, LinearForm *b, Array<int> &ess_bdr_list): a(a),
      b(b),
      ess_bdr(1, ess_bdr_list.Size()), parallel(false)
   {
      for (int i=0; i<ess_bdr_list.Size(); i++)
      {
         ess_bdr(0, i) = ess_bdr_list[i];
      }
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(a->FESpace());
      if (pfes) {parallel = true;}
#endif
   };
   EllipticSolver(BilinearForm *a, LinearForm *b, Array2D<int> &ess_bdr): a(a),
      b(b), ess_bdr(ess_bdr), parallel(false)
   {
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(a->FESpace());
      if (pfes) {parallel = true;}
#endif
   }
   void ChangeLHS(BilinearForm *a_) { a = a_; }
   void ChangeRHS(LinearForm *b_) { b = b_; }
   bool Solve(GridFunction *x)
   {
      OperatorPtr A;
      Vector B, X;
      Array<int> ess_tdof_list(0);

#ifdef MFEM_USE_MPI
      if (parallel)
      {
         if (ess_bdr.NumRows() == 1)
         {
            Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
            dynamic_cast<ParFiniteElementSpace*>(a->FESpace())->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list);
         }
         else
         {
            Array<int> ess_tdof_list_comp;
            for (int i=0; i<ess_bdr.NumRows() - 1; i++)
            {
               Array<int> ess_bdr_list(ess_bdr.GetRow(i), ess_bdr.NumCols());
               dynamic_cast<ParFiniteElementSpace*>(a->FESpace())->GetEssentialTrueDofs(
                  ess_bdr_list, ess_tdof_list_comp, i);
               ess_tdof_list.Append(ess_tdof_list_comp);
            }
            Array<int> ess_bdr_list(ess_bdr.GetRow(ess_bdr.NumRows() - 1),
                                    ess_bdr.NumCols());
            dynamic_cast<ParFiniteElementSpace*>(a->FESpace())->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list_comp, -1);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
      }
      else
      {
         if (ess_bdr.NumRows() == 1)
         {
            Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
            a->FESpace()->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list);
         }
         else
         {
            Array<int> ess_tdof_list_comp;
            for (int i=0; i<ess_bdr.NumRows() - 1; i++)
            {
               Array<int> ess_bdr_list(ess_bdr.GetRow(i), ess_bdr.NumCols());
               a->FESpace()->GetEssentialTrueDofs(
                  ess_bdr_list, ess_tdof_list_comp, i);
               ess_tdof_list.Append(ess_tdof_list_comp);
            }
            Array<int> ess_bdr_list(ess_bdr.GetRow(ess_bdr.NumRows() - 1),
                                    ess_bdr.NumCols());
            a->FESpace()->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list_comp, -1);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
      }
#else
      if (ess_bdr.NumRows() == 1)
      {
         Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
         a->FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list);
      }
      else
      {
         Array<int> ess_tdof_list_comp;
         for (int i=0; i<ess_bdr.NumRows() - 1; i++)
         {
            Array<int> ess_bdr_list(ess_bdr.GetRow(i), ess_bdr.NumCols());
            a->FESpace()->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list_comp, i);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
         Array<int> ess_bdr_list(ess_bdr.GetRow(ess_bdr.NumRows() - 1),
                                 ess_bdr.NumCols());
         a->FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, -1);
         ess_tdof_list.Append(ess_tdof_list_comp);
      }
#endif
      a->Assemble();
      b->Assemble();

      a->FormLinearSystem(ess_tdof_list, *x, *b, A, X, B, true);

#ifdef MFEM_USE_SUITESPARSE
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
      a->RecoverFEMSolution(X, *b, *x);
      bool converged = true;
#else
      CGSolver * cg = nullptr;
      Solver * M = nullptr;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         M = new HypreBoomerAMG;
         dynamic_cast<HypreBoomerAMG*>(M)->SetPrintLevel(0);
         cg = new CGSolver(dynamic_cast<ParFiniteElementSpace*>
                           (a->FESpace())->GetComm());
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
      cg->SetRelTol(1e-14);
      cg->SetMaxIter(10000);
      cg->SetPrintLevel(0);
      cg->SetPreconditioner(*M);
      cg->SetOperator(*A);
      cg->Mult(B, X);
      a->RecoverFEMSolution(X, *b, *x);
      bool converged = cg->GetConverged();
      delete M;
      delete cg;
#endif

      return converged;
   };
protected:
   BilinearForm *a;
   LinearForm *b;
   Array2D<int> ess_bdr;
   bool parallel;
private:
};

class ObjectiveFunction
{
public:
   ObjectiveFunction(
   ) {}

   void SetGridFunction(GridFunction *x) {x_gf = x;}
   void SetCoefficient(Coefficient *x) {x_cf = x;}
   GridFunction *GetGridFunction() {return x_gf;}
   Coefficient *GetCoefficient() {return x_cf;}

   virtual double Eval() {mfem_error("Eval is not implemented"); return infinity();}
   virtual double Eval(GridFunction *x) { x_gf = x; return Eval(); }
   virtual double Eval(Coefficient *x) { x_cf = x; return Eval(); }
   virtual GridFunction *Gradient() {mfem_error("Gradient is not implemented"); return nullptr;};
   virtual GridFunction *Gradient(GridFunction *x) { x_gf = x; return Gradient(); };
   virtual GridFunction *Gradient(Coefficient *x) { x_cf = x; return Gradient(); };
   double GetValue() {return current_val;}
   GridFunction * GetGradient() { return gradF_gf; }
   virtual Coefficient * dcf_dgf() { return nullptr; }
protected:
   GridFunction *x_gf = nullptr;
   GridFunction *gradF_gf = nullptr;
   Coefficient *x_cf = nullptr;
   double current_val = infinity();
};

class SIMPElasticCompliance : public ObjectiveFunction
{
public:
   SIMPElasticCompliance(Coefficient *lambda, Coefficient *mu, double epsilon,
                         Coefficient *rho, VectorCoefficient *force, const double target_volume,
                         Array<int> &ess_bdr_list,
                         FiniteElementSpace *displacement_space,
                         FiniteElementSpace *filter_space, double exponent, double rho_min):
      SIMPlambda(*lambda, design_density), SIMPmu(*mu, design_density),
      eps2(epsilon*epsilon), ess_bdr(1, ess_bdr_list.Size()),
      target_volume(target_volume), isGradientUpdated(false), isPrimalBased(false),
      u(nullptr), frho(nullptr),
      rho(rho), force(force), strainEnergy(lambda, mu, u, frho, rho_min, exponent)
   {
      for (int i=0; i<ess_bdr_list.Size(); i++) {ess_bdr(0, i) = ess_bdr_list[i]; };
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes;

      pfes = dynamic_cast<ParFiniteElementSpace*>(displacement_space);
      if (pfes)
      {
         u = new ParGridFunction(pfes);
      }
      else
      {
         u = new GridFunction(displacement_space);
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
      strainEnergy.SetDisplacement(u);
      strainEnergy.SetFilteredDensity(frho);
   }
   SIMPElasticCompliance(Coefficient *lambda, Coefficient *mu, double epsilon,
                         Coefficient *rho, VectorCoefficient *force, const double target_volume,
                         Array2D<int> &ess_bdr,
                         FiniteElementSpace *displacement_space,
                         FiniteElementSpace *filter_space, double exponent, double rho_min):
      SIMPlambda(*lambda, design_density), SIMPmu(*mu, design_density),
      eps2(epsilon*epsilon), ess_bdr(ess_bdr),
      target_volume(target_volume),
      u(nullptr), frho(nullptr),
      rho(rho), force(force), strainEnergy(lambda, mu, u, frho, rho_min, exponent)
   {
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes;

      pfes = dynamic_cast<ParFiniteElementSpace*>(displacement_space);
      if (pfes)
      {
         u = new ParGridFunction(pfes);
      }
      else
      {
         u = new GridFunction(displacement_space);
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
      strainEnergy.SetDisplacement(u);
      strainEnergy.SetFilteredDensity(frho);
   }

   virtual double Eval()
   {
      isGradientUpdated = false;
      BilinearForm *elasticity, *filter;
      LinearForm *load, *filterRHS;
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
      }
      else
      {
         elasticity = new BilinearForm(fes);
         load = new LinearForm(fes);
      }
#else
      fes = frho->FESpace();
      filter = new BilinearForm(fes);
      filterRHS = new LinearForm(fes);

      fes = u->FESpace();
      elasticity = new BilinearForm(fes);
      load = new LinearForm(fes);
#endif
      // Step 1. Projection
      double volume = proj(1e-12, 500);

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
      load->AddDomainIntegrator(new VectorDomainLFIntegrator(*force));
      EllipticSolver elasticitySolver(elasticity, load, ess_bdr);
      elasticitySolver.Solve(u);
      current_val = (*load)(*u);

      delete elasticity;
      delete load;
      delete filter;
      delete filterRHS;

      if (volume == infinity())
      {
         return -infinity();
      }
      return current_val;
   }

   virtual GridFunction *Gradient()
   {
      if (isGradientUpdated)
      {
         return gradF_gf;
      }
      BilinearForm *filter, *invmass;
      LinearForm *filterRHS, *gradH1form;
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
#else
      fes = frho->FESpace();
      filter = new BilinearForm(fes);
      filterRHS = new LinearForm(fes);
      gradH1 = new GridFunction(fes);
      fes = x_gf->FESpace();
      invmass = new BilinearForm(fes);
      gradH1form = new LinearForm(fes);

#endif

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
      isGradientUpdated = true;
      return gradF_gf;
   }
   double GetVolume() {return current_volume;}

   GridFunction *GetDisplacement() { return u; }
   GridFunction *GetFilteredDensity() { return frho; }
   MappedGridFunctionCoefficient &GetDesignDensity() { return design_density; }

   ~SIMPElasticCompliance()
   {
      delete u;
      delete frho;
   }

   void SetGridFunction(GridFunction* x, bool isPrimal=false)
   {
      isPrimalBased = isPrimal;
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
   bool isGradientUpdated;
   bool isPrimalBased;

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
      if (isPrimalBased)
      {
         return proj_primal();
      }
      else
      {
         return proj_latent();
      }
      return current_volume;
   }

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
   double proj_latent(double tol=1e-12, int max_its=10)
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
      ConstantCoefficient one_cf(1.0);
      double inv_sig_vol_fraction = inv_sigmoid(target_volume /
                                                zero_gf->ComputeL1Error(one_cf));
      double c_l = inv_sig_vol_fraction - x_gf->Max();
      double c_r = inv_sig_vol_fraction - x_gf->Min();
      while (fabs(Vc - target_volume) > tol & fabs(c_r - c_l) > 1e-09)
      {
         c = 0.5*(c_r + c_l);
         Vc = zero_gf->ComputeL1Error(rho);
         if (Vc > target_volume)
         {
            c_r = c;
         }
         else
         {
            c_l = c;
         }
      }
      c = 0.5*(c_r + c_l);
      *x_gf += c;
      c = 0.0;
      current_volume = zero_gf->ComputeL1Error(rho);

      delete zero_gf;
      return current_volume;
   }

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
   double proj_primal(double tol=1e-12, int max_its=10)
   {
      double c = 0;
      MappedGridFunctionCoefficient rho(x_gf, [&c](const double x)
      {
         return std::max(0.0, std::min(1.0, x + c));
      });

      double c_l = -(x_gf->Normlinf());
      double c_r = -c_l;
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
      while (fabs(Vc - target_volume) > tol & (c_r - c_l) > 1e-09)
      {
         c = 0.5*(c_r + c_l);
         Vc = zero_gf->ComputeL1Error(rho);
         if (Vc > target_volume)
         {
            c_r = c;
         }
         else
         {
            c_l = c;
         }
      }
      c = 0.5*(c_r + c_l);
      current_volume = zero_gf->ComputeL1Error(rho);
      x_gf->ProjectCoefficient(rho);
      return current_volume;
   }
   double current_volume;

protected:
   ProductCoefficient SIMPlambda, SIMPmu;
   ConstantCoefficient eps2;
   Array2D<int> ess_bdr;
   double target_volume;
   GridFunction *u, *frho;
   Coefficient *rho;
   VectorCoefficient *force;
   StrainEnergyDensityCoefficient strainEnergy;
   MappedGridFunctionCoefficient design_density;
};
class LineSearchAlgorithm
{
public:
   LineSearchAlgorithm(ObjectiveFunction &F, double max_step_size=1e06):F(F),
      max_step_size(max_step_size) {}
   virtual double Step(GridFunction &x, const GridFunction &d) = 0;
   virtual ~LineSearchAlgorithm() = default;

   double GetStepSize() { return step_size; }
   void SetStepSize(double s) { step_size = s; }

protected:
   ObjectiveFunction &F;
   double step_size, max_step_size;
private:
};

class LinearGrowth : public LineSearchAlgorithm
{
public:
   LinearGrowth(ObjectiveFunction &F,
                const double alpha=1.0): LineSearchAlgorithm(F), alpha(alpha) {}
   double Step(GridFunction &x, const GridFunction &d)
   {
      k++;
      step_size = alpha * k;
      x.Add(step_size, d);
      return F.Eval();
   }
protected:
private:
   int k = 0;
   double alpha;
};

class ExponentialGrowth : public LineSearchAlgorithm
{
public:
   ExponentialGrowth(ObjectiveFunction &F, const double growthRate=2.0,
                     const double alpha=1.0): LineSearchAlgorithm(F), alpha(alpha),
      growthRate(growthRate) {}
   double Step(GridFunction &x, const GridFunction &d)
   {
      step_size = alpha * pow(growthRate, k++);
      x.Add(step_size, d);
      return F.Eval();
   }
protected:
private:
   int k = 0;
   double alpha, growthRate;
};

class BackTracking : public LineSearchAlgorithm
{
public:
   BackTracking(ObjectiveFunction &F,
                LinearForm &distanceForm,
                const GridFunction &x_old,
                const double alpha=1.0, const double growthRate=2.0, const double c1 = 1e-04,
                const int maxit = 10, const double max_step_size=infinity()):
      LineSearchAlgorithm(F, max_step_size), growthRate(growthRate), c1(c1),
      distanceForm(distanceForm), x0(&x_old),
      maxit(maxit) {step_size = alpha;}
   void SetOldGF(const GridFunction &x_old)
   {
      x0 = &x_old;
   }
   double Step(GridFunction &x, const GridFunction &d)
   {
      FiniteElementSpace *fes = x.FESpace();

      double val = F.GetValue();
      GridFunction *grad = F.GetGradient();
      double d2 = 0;

      double new_val;
      new_val = val + c1*d2 + 1;
      step_size *= 2;
      int i;
      for (i=0; i<maxit; i++)
      {
         step_size *= 0.5;
         out << step_size << "," << std::flush;

         x = *x0;
         x.Add(step_size, d);
         new_val = F.Eval();
         distanceForm.Assemble();
         d2 = distanceForm(*grad);
         if (new_val < val + c1*d2 & d2 < 0)
         {
            break;
         }

      }
      out << std::endl;
      out << i << ", " << step_size;
      out.flush();

      step_size *= growthRate;
      step_size = std::min(step_size, max_step_size);
      return new_val;
   }
   void SetDistanceFunction(double(*new_distfun)(double, double))
   {
      distfun = new_distfun;
   }
protected:
   double growthRate, c1;
   double(*distfun)(double, double);
   const GridFunction *x0;
   LinearForm &distanceForm;
   int maxit;
private:
};

class LipschitzBregmanMirror : public LineSearchAlgorithm
{
public:
   LipschitzBregmanMirror(ObjectiveFunction &F, LinearForm &diff_primal,
                          GridFunction &grad, GridFunction &latent, double weight=1.0, double eps=1e-10,
                          double max_step_size=1e06):
      LineSearchAlgorithm(F, max_step_size), diff_primal(diff_primal),
      grad(grad), old_grad(grad), latent(latent), old_latent(latent), Linv(0), k(0),
      weight(weight), eps(eps) {}
   virtual double Step(GridFunction &x, const GridFunction &d)
   {
      EstimateStepSize();
      x.Add(step_size, d);
      diff_primal.Assemble();
      return F.Eval();
   }
   void EstimateStepSize()
   {
      if (k++ == 0)
      {
         step_size = 1.0;
         old_grad = grad;
         old_latent = latent;
      }
      else
      {
         double diff_grad = diff_primal(grad) - diff_primal(old_grad);
         double diff_latent = diff_primal(latent) - diff_primal(old_latent);
         Linv = std::abs(diff_latent / (diff_grad + eps));
         step_size = std::min(weight * Linv, max_step_size);
         old_grad = grad;
         old_latent = latent;
      }
   }
protected:
   LinearForm &diff_primal;
   GridFunction &grad;
   GridFunction old_grad;
   GridFunction &latent;
   GridFunction old_latent;
   double Linv; int k;
   double weight,  eps;
};


class BackTrackingLipschitzBregmanMirror : public LipschitzBregmanMirror
{
public:
   BackTrackingLipschitzBregmanMirror(ObjectiveFunction &F,
                                      LinearForm &diff_primal,
                                      GridFunction &grad, GridFunction &latent, GridFunction &latent_old, double c1,
                                      double weight=1.0,
                                      double eps=1e-10, double max_step_size=1e06, int max_its=30):
      LipschitzBregmanMirror(F, diff_primal, grad, latent, weight, eps,
                             max_step_size), backTrackingLineSearch(F, diff_primal, latent_old, 1.0, 1.0, c1)
   {
      backTrackingLineSearch.SetOldGF(latent_old);
   }
   virtual double Step(GridFunction &x, const GridFunction &d)
   {
      LipschitzBregmanMirror::EstimateStepSize();
      backTrackingLineSearch.SetStepSize(step_size);
      double value = backTrackingLineSearch.Step(x, d);
      diff_primal.Assemble();
      return value;
   }
protected:
   BackTracking backTrackingLineSearch;
};

} // end of namespace mfem
