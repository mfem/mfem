#include "topopt.hpp"
#include "helper.hpp"

namespace mfem
{

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
   return x>=0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}

/// @brief Derivative of sigmoid function
double der_sigmoid(const double x)
{
   const double tmp = sigmoid(x);
   return tmp*(1.0 - tmp);
}

/// @brief SIMP function, ρ₀ + (ρ̄ - ρ₀)*x^k
double simp(const double x, const double rho_0, const double k,
            const double rho_max)
{
   return rho_0 + std::pow(x, k) * (rho_max - rho_0);
}

/// @brief Derivative of SIMP function, k*(ρ̄ - ρ₀)*x^(k-1)
double der_simp(const double x, const double rho_0,
                const double k, const double rho_max)
{
   return k * std::pow(x, k - 1.0) * (rho_max - rho_0);
}

EllipticSolver::EllipticSolver(BilinearForm &a, LinearForm &b,
                               Array<int> &ess_bdr_list):
   a(a), b(b), ess_bdr(1, ess_bdr_list.Size()), ess_tdof_list(0), symmetric(false)
{
   for (int i=0; i<ess_bdr_list.Size(); i++)
   {
      ess_bdr(0, i) = ess_bdr_list[i];
   }
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
   if (pfes) {parallel = true; comm = pfes->GetComm(); }
   else {parallel = false;}
#endif
   GetEssentialTrueDofs();
}

EllipticSolver::EllipticSolver(BilinearForm &a, LinearForm &b,
                               Array2D<int> &ess_bdr):
   a(a), b(b), ess_bdr(ess_bdr), ess_tdof_list(0), symmetric(false)
{
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
   if (pfes) {parallel = true; comm = pfes->GetComm(); }
   else {parallel = false;}
#endif
   GetEssentialTrueDofs();
}

void EllipticSolver::GetEssentialTrueDofs()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
      if (ess_bdr.NumRows() == 1)
      {
         Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
         pfes->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list);
      }
      else
      {
         Array<int> ess_tdof_list_comp;
         for (int i=0; i<ess_bdr.NumRows() - 1; i++)
         {
            Array<int> ess_bdr_list(ess_bdr.GetRow(i), ess_bdr.NumCols());
            pfes->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list_comp, i);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
         Array<int> ess_bdr_list(ess_bdr.GetRow(ess_bdr.NumRows() - 1),
                                 ess_bdr.NumCols());
         pfes->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, -1);
         ess_tdof_list.Append(ess_tdof_list_comp);
      }
   }
   else
   {
      if (ess_bdr.NumRows() == 1)
      {
         Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list);
      }
      else
      {
         Array<int> ess_tdof_list_comp;
         for (int i=0; i<ess_bdr.NumRows() - 1; i++)
         {
            Array<int> ess_bdr_list(ess_bdr.GetRow(i), ess_bdr.NumCols());
            a.FESpace()->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list_comp, i);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
         Array<int> ess_bdr_list(ess_bdr.GetRow(ess_bdr.NumRows() - 1),
                                 ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, -1);
         ess_tdof_list.Append(ess_tdof_list_comp);
      }
   }
#else
   if (ess_bdr.NumRows() == 1)
   {
      Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
      a.FESpace()->GetEssentialTrueDofs(
         ess_bdr_list, ess_tdof_list);
   }
   else
   {
      Array<int> ess_tdof_list_comp;
      for (int i=0; i<ess_bdr.NumRows() - 1; i++)
      {
         Array<int> ess_bdr_list(ess_bdr.GetRow(i), ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, i);
         ess_tdof_list.Append(ess_tdof_list_comp);
      }
      Array<int> ess_bdr_list(ess_bdr.GetRow(ess_bdr.NumRows() - 1),
                              ess_bdr.NumCols());
      a.FESpace()->GetEssentialTrueDofs(
         ess_bdr_list, ess_tdof_list_comp, -1);
      ess_tdof_list.Append(ess_tdof_list_comp);
   }
#endif
}

bool EllipticSolver::Solve(GridFunction &x, bool A_assembled,
                           bool b_Assembled)
{
   OperatorPtr A;
   Vector B, X;
   if (!A_assembled) { a.Update(); a.Assemble(); }
   if (!b_Assembled) { b.Assemble(); }

   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, true);

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
   a.RecoverFEMSolution(X, b, x);
   bool converged = true;
#else
   std::unique_ptr<CGSolver> cg;
   std::unique_ptr<Solver> M;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      M.reset(new HypreBoomerAMG);
      auto M_ptr = new HypreBoomerAMG;
      M_ptr->SetPrintLevel(0);

      M.reset(M_ptr);
      cg.reset(new CGSolver(comm));
   }
   else
   {
      M.reset(new GSSmoother((SparseMatrix&)(*A)));
      cg.reset(new CGSolver);
   }
#else
   M.reset(new GSSmoother((SparseMatrix&)(*A)));
   cg.reset(new CGSolver);
#endif
   cg->SetRelTol(1e-14);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->Mult(B, X);
   a.RecoverFEMSolution(X, b, x);
   bool converged = cg->GetConverged();
#endif

   return converged;
}

bool EllipticSolver::SolveTranspose(GridFunction &x, LinearForm *f,
                                    bool A_assembled, bool f_Assembled)
{
   OperatorPtr A;
   Vector B, X;

   if (!A_assembled) { a.Assemble(); }
   if (!f_Assembled) { f->Assemble(); }

   a.FormLinearSystem(ess_tdof_list, x, *f, A, X, B, true);

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
   a.RecoverFEMSolution(X, *f, x);
   bool converged = true;
#else
   std::unique_ptr<CGSolver> cg;
   std::unique_ptr<Solver> M;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      M.reset(new HypreBoomerAMG);
      auto M_ptr = new HypreBoomerAMG;
      M_ptr->SetPrintLevel(0);

      M.reset(M_ptr);
      cg.reset(new CGSolver((dynamic_cast<ParFiniteElementSpace*>
                             (a.FESpace()))->GetComm()));
   }
   else
   {
      M.reset(new GSSmoother((SparseMatrix&)(*A)));
      cg.reset(new CGSolver);
   }
#else
   M.reset(new GSSmoother((SparseMatrix&)(*A)));
   cg.reset(new CGSolver);
#endif
   cg->SetRelTol(1e-14);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->Mult(B, X);
   a.RecoverFEMSolution(X, *f, x);
   bool converged = cg->GetConverged();
#endif

   return converged;
}

DesignDensity::DesignDensity(FiniteElementSpace &fes, DensityFilter &filter,
                             FiniteElementSpace &fes_filter,
                             double target_volume_fraction,
                             double volume_tolerance)
   : filter(filter), target_volume_fraction(target_volume_fraction),
     vol_tol(volume_tolerance)
{
   x_gf.reset(MakeGridFunction(&fes));
   *x_gf = target_volume_fraction;
   frho.reset(MakeGridFunction(&fes_filter));
   *frho = target_volume_fraction;
   Mesh *mesh = fes.GetMesh();
   double domain_volume = 0.0;
   for (int i=0; i<mesh->GetNE(); i++)
   {
      domain_volume += mesh->GetElementVolume(i);
   }
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
   if (pfes)
   {
      MPI_Allreduce(MPI_IN_PLACE, &domain_volume, 1, MPI_DOUBLE, MPI_SUM,
                    pfes->GetComm());
   }
#endif
   target_volume = domain_volume * target_volume_fraction;
}

SIMPProjector::SIMPProjector(const double k, const double rho0):k(k), rho0(rho0)
{
   phys_density.reset(new MappedGridFunctionCoefficient(
   nullptr, [rho0, k](double x) {return simp(x, rho0, k);}));
   dphys_dfrho.reset(new MappedGridFunctionCoefficient(
   nullptr, [rho0, k](double x) {return der_simp(x, rho0, k);}));
}
Coefficient &SIMPProjector::GetPhysicalDensity(GridFunction &frho)
{
   phys_density->SetGridFunction(&frho);
   return *phys_density;
}
Coefficient &SIMPProjector::GetDerivative(GridFunction &frho)
{
   dphys_dfrho->SetGridFunction(&frho);
   return *dphys_dfrho;
}

ThresholdProjector::ThresholdProjector(const double beta,
                                       const double eta):beta(beta), eta(eta)
{
   const double c1 = std::tanh(beta*eta);
   const double c2 = std::tanh(beta*(1-eta));
   const double inv_denominator = 1.0 / (c1 + c2);
   phys_density.reset(new MappedGridFunctionCoefficient(
                         nullptr, [c1, c2, beta, eta](double x)
   {
      return (c1 + std::tanh(beta*(x - eta))) / (c1 + c2);
   }));
   dphys_dfrho.reset(new MappedGridFunctionCoefficient(
                        nullptr, [c1, c2, beta, eta](double x)
   {
      return beta*std::pow(1.0/std::cosh(beta*(x - eta)), 2.0) / (c1 + c2);
   }));
}
Coefficient &ThresholdProjector::GetPhysicalDensity(GridFunction &frho)
{
   phys_density->SetGridFunction(&frho);
   return *phys_density;
}
Coefficient &ThresholdProjector::GetDerivative(GridFunction &frho)
{
   dphys_dfrho->SetGridFunction(&frho);
   return *dphys_dfrho;
}
LatentDesignDensity::LatentDesignDensity(FiniteElementSpace &fes,
                                         DensityFilter &filter,
                                         FiniteElementSpace &fes_filter,
                                         double vol_frac):
   DesignDensity(fes, filter, fes_filter, vol_frac),
   zero_gf(MakeGridFunction(&fes))
{
   *x_gf = inv_sigmoid(vol_frac);
   rho_cf.reset(new MappedGridFunctionCoefficient(x_gf.get(), sigmoid));
   *zero_gf = 0.0;
}

void LatentDesignDensity::Project()
{
   ComputeVolume();
   if (std::fabs(current_volume - target_volume) > vol_tol)
   {
      double inv_sig_vol_fraction = inv_sigmoid(target_volume_fraction);
      double c_l = inv_sig_vol_fraction - x_gf->Max();
      double c_r = inv_sig_vol_fraction - x_gf->Min();
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
      if (pfes)
      {
         MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
      }
#endif
      double c = 0.5 * (c_l + c_r);
      double dc = 0.5 * (c_r - c_l);
      *x_gf += c;
      while (dc > 1e-09)
      {
         dc *= 0.5;
         ComputeVolume();
         if (fabs(current_volume - target_volume) < vol_tol) { break; }
         *x_gf += current_volume < target_volume ? dc : -dc;
      }
   }
}

double LatentDesignDensity::StationarityError(GridFunction &grad,
                                              bool useL2norm)
{
   std::unique_ptr<GridFunction> x_gf_backup(MakeGridFunction(x_gf->FESpace()));
   *x_gf_backup = *x_gf;
   double volume_backup = current_volume;
   *x_gf -= grad;
   Project();
   double d;
   if (useL2norm)
   {
      MappedPairGridFunctionCoeffitient rho_diff(x_gf.get(),
                                                 x_gf_backup.get(), [](double x,
      double y) {return sigmoid(x) - sigmoid(y);});
      d = zero_gf->ComputeL2Error(rho_diff);
   }
   else
   {
      d = ComputeBregmanDivergence(x_gf.get(), x_gf_backup.get());
   }
   // Restore solution and recompute volume
   *x_gf = *x_gf_backup;
   current_volume = volume_backup;
   return d;
}
double LatentDesignDensity::ComputeBregmanDivergence(GridFunction *p,
                                                     GridFunction *q, double epsilon)
{
   // Define safe x*log(x) to avoid log(0)
   const double log_eps = std::log(epsilon);
   auto safe_xlogy = [epsilon, log_eps](double x, double y) {return x < epsilon ? 0.0 : y < epsilon ? x*log_eps : x*std::log(y); };
   MappedPairGridFunctionCoeffitient Dh(p, q, [safe_xlogy](double x, double y)
   {
      const double p = sigmoid(x);
      const double q = sigmoid(y);
      return safe_xlogy(p, p) - safe_xlogy(p, q)
             + safe_xlogy(1 - p, 1 - p) - safe_xlogy(1 - p, 1 - q);
   });
   // Since Bregman divergence is always positive, ||Dh||_L¹=∫_Ω Dh.
   return std::sqrt(zero_gf->ComputeL1Error(Dh));
}

double LatentDesignDensity::StationarityErrorL2(GridFunction &grad)
{
   double c;
   MappedPairGridFunctionCoeffitient projected_rho(x_gf.get(),
                                                   &grad, [&c](double x, double y)
   {
      return std::min(1.0, std::max(0.0, sigmoid(x) - y + c));
   });

   double c_l = target_volume_fraction - (1.0 - grad.Min());
   double c_r = target_volume_fraction - (0.0 - grad.Max());
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
   if (pfes)
   {
      MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
   }
#endif
   while (c_r - c_l > 1e-09)
   {
      c = 0.5 * (c_l + c_r);
      double vol = zero_gf->ComputeL1Error(projected_rho);
      if (fabs(vol - target_volume) < vol_tol) { break; }

      if (vol > target_volume) { c_r = c; }
      else { c_l = c; }
   }

   SumCoefficient diff_rho(projected_rho, *rho_cf, 1.0, -1.0);
   return zero_gf->ComputeL2Error(diff_rho);
}


void PrimalDesignDensity::Project()
{
   ComputeVolume();
   if (std::fabs(current_volume - target_volume) > vol_tol)
   {
      double c_l = target_volume_fraction - x_gf->Max();
      double c_r = target_volume_fraction - x_gf->Min();
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
      if (pfes)
      {
         MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
      }
#endif
      MappedGridFunctionCoefficient projected_rho(x_gf.get(), [](double x) {return std::min(1.0, std::max(0.0, x));});
      std::unique_ptr<GridFunction> zero_gf(MakeGridFunction(x_gf->FESpace()));
      *zero_gf = 0.0;
      double c = 0.5 * (c_l + c_r);
      double dc = 0.5 * (c_r - c_l);
      *x_gf += c;
      while (dc > 1e-09)
      {
         dc *= 0.5;
         current_volume = zero_gf->ComputeL1Error(projected_rho);
         if (fabs(current_volume - target_volume) < vol_tol) { break; }
         *x_gf += current_volume < target_volume ? dc : -dc;
      }
      x_gf->ProjectCoefficient(projected_rho);
   }
}

double PrimalDesignDensity::StationarityError(GridFunction &grad)
{
   // Back up current status
   std::unique_ptr<GridFunction> x_gf_backup(MakeGridFunction(x_gf->FESpace()));
   *x_gf_backup = *x_gf;
   double volume_backup = current_volume;

   // Project ρ + grad
   *x_gf -= grad;
   Project();

   // Compare the updated density and the original density
   double d = x_gf_backup->ComputeL2Error(*rho_cf);

   // Restore solution and recompute volume
   *x_gf = *x_gf_backup;
   current_volume = volume_backup;
   return d;
}
ParametrizedLinearEquation::ParametrizedLinearEquation(
   FiniteElementSpace &fes, GridFunction &filtered_density,
   DensityProjector &projector, Array2D<int> &ess_bdr):
   frho(filtered_density), projector(projector), AisStationary(false),
   BisStationary(false),
   ess_bdr(ess_bdr)
{
   a.reset(MakeBilinearForm(&fes));
   b.reset(MakeLinearForm(&fes));
}

void ParametrizedLinearEquation::SetBilinearFormStationary(bool isStationary)
{
   AisStationary = isStationary;
   if (isStationary) { a->Assemble(); }
}
void ParametrizedLinearEquation::SetLinearFormStationary(bool isStationary)
{
   BisStationary = isStationary;
   if (isStationary) { b->Assemble(); }
}
void ParametrizedLinearEquation::Solve(GridFunction &x)
{
   if (!AisStationary) { a->Update(); }
   if (!BisStationary) { b->Update(); }
   EllipticSolver solver(*a, *b, ess_bdr);
   solver.Solve(x, AisStationary, BisStationary);
}
void ParametrizedLinearEquation::DualSolve(GridFunction &x, LinearForm &new_b)
{
   if (!AisStationary) { a->Update(); }
   EllipticSolver solver(*a, new_b, ess_bdr);
   solver.Solve(x, AisStationary, false);
}

TopOptProblem::TopOptProblem(LinearForm &objective,
                             ParametrizedLinearEquation &state_equation,
                             DesignDensity &density, bool solve_dual, bool apply_projection)
   :obj(objective), state_equation(state_equation), density(density),
    solve_dual(solve_dual), apply_projection(apply_projection)
{
   state.reset(MakeGridFunction(state_equation.FESpace()));
   *state = 0.0;
   if (!solve_dual)
   {
      dual_solution = state;
   }
   else
   {
      dual_solution.reset(MakeGridFunction(state_equation.FESpace()));
      *dual_solution = 0.0;
   }
   dEdfrho = state_equation.GetdEdfrho(*state, *dual_solution,
                                       density.GetFilteredDensity());
   gradF.reset(MakeGridFunction(density.FESpace()));
   *gradF = 0.0;
   if (density.FESpace() == density.FESpace_filter())
   {
      gradF_filter = gradF;
   }
   else
   {
      gradF_filter.reset(MakeGridFunction(density.FESpace_filter()));
      *gradF_filter = 0.0;
      filter_to_density.reset(MakeBilinearForm(density.FESpace()));
      filter_to_density->AddDomainIntegrator(new InverseIntegrator(
                                                new MassIntegrator));
      filter_to_density->Assemble();
   }

#ifdef MFEM_USE_MPI
   auto pstate = dynamic_cast<ParGridFunction*>(state.get());
   if (pstate) {parallel = true; comm = pstate->ParFESpace()->GetComm(); }
#endif
}

double TopOptProblem::Eval()
{
   if (apply_projection) { density.Project(); }
   density.UpdateFilteredDensity();
   state_equation.Solve(*state);
   val = obj(*state);
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, comm);
   }
#endif
   return val;
}

void TopOptProblem::UpdateGradient()
{
   if (solve_dual)
   {
      // state equation is assumed to be a symmetric operator
      state_equation.DualSolve(*dual_solution, obj);
   }
   density.GetFilter().Apply(*dEdfrho, *gradF_filter);
   if (gradF_filter != gradF)
   {
      std::unique_ptr<LinearForm> tmp(MakeLinearForm(gradF->FESpace()));
      GridFunctionCoefficient gradF_filter_cf(gradF_filter.get());
      tmp->AddDomainIntegrator(new DomainLFIntegrator(gradF_filter_cf));
      tmp->Assemble();
      filter_to_density->Mult(*tmp, *gradF);
   }
}

double StrainEnergyDensityCoefficient::Eval(ElementTransformation &T,
                                            const IntegrationPoint &ip)
{
   double L = lambda.Eval(T, ip);
   double M = mu.Eval(T, ip);
   double density;
   if (&u2 == &u1)
   {
      u1.GetVectorGradient(T, grad1);
      double div_u = grad1.Trace();
      grad1.Symmetrize();
      density = L*div_u*div_u + 2*M*(grad1*grad1);
   }
   else
   {
      u1.GetVectorGradient(T, grad1);
      u2.GetVectorGradient(T, grad2);
      double div_u1 = grad1.Trace();
      double div_u2 = grad2.Trace();
      grad1.Symmetrize();

      density = L*div_u1*div_u2 + 2*M*(grad1*grad2);
   }
   return -dphys_dfrho.Eval(T, ip) * density;
}

double ThermalEnergyDensityCoefficient::Eval(ElementTransformation &T,
                                             const IntegrationPoint &ip)
{
   double K = kappa.Eval(T, ip);
   double density;
   if (&u2 == &u1)
   {
      u1.GetGradient(T, grad1);
      density = K*(grad1*grad1);
   }
   else
   {
      u1.GetGradient(T, grad1);
      u2.GetGradient(T, grad2);

      density = K*(grad1*grad2);
   }
   return -dphys_dfrho.Eval(T, ip) * density;
}

ParametrizedElasticityEquation::ParametrizedElasticityEquation(
   FiniteElementSpace &fes, GridFunction &filtered_density,
   DensityProjector &projector,
   Coefficient &lambda, Coefficient &mu, VectorCoefficient &f,
   Array2D<int> &ess_bdr):
   ParametrizedLinearEquation(fes, filtered_density, projector, ess_bdr),
   lambda(lambda), mu(mu), filtered_density(filtered_density),
   phys_lambda(lambda, projector.GetPhysicalDensity(filtered_density)),
   phys_mu(mu, projector.GetPhysicalDensity(filtered_density)),
   f(f)
{
   a->AddDomainIntegrator(new ElasticityIntegrator(phys_lambda, phys_mu));
   b->AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   SetLinearFormStationary();
}

ParametrizedDiffusionEquation::ParametrizedDiffusionEquation(
   FiniteElementSpace &fes,
   GridFunction &filtered_density,
   DensityProjector &projector,
   Coefficient &kappa,
   Coefficient &f, Array2D<int> &ess_bdr):
   ParametrizedLinearEquation(fes, filtered_density, projector, ess_bdr),
   kappa(kappa), filtered_density(filtered_density),
   phys_kappa(kappa, projector.GetPhysicalDensity(filtered_density)),
   f(f)
{
   a->AddDomainIntegrator(new DiffusionIntegrator(phys_kappa));
   b->AddDomainIntegrator(new DomainLFIntegrator(f));
   SetLinearFormStationary();
}

/// @brief Volumetric force for linear elasticity

VolumeForceCoefficient::VolumeForceCoefficient(double r_,Vector &  center_,
                                               Vector & force_) :
   VectorCoefficient(center_.Size()), r2(r_*r_), center(center_), force(force_) { }

void VolumeForceCoefficient::Eval(Vector &V, ElementTransformation &T,
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

void VolumeForceCoefficient::Set(double r_,Vector & center_, Vector & force_)
{
   r2=r_*r_;
   center = center_;
   force = force_;
}
void VolumeForceCoefficient::UpdateSize()
{
   VectorCoefficient::vdim = center.Size();
}

/// @brief Volumetric force for linear elasticity
LineVolumeForceCoefficient::LineVolumeForceCoefficient(double r_,
                                                       Vector &center_, Vector & force_,
                                                       int direction_dim) :
   VectorCoefficient(center_.Size()), r2(r_*r_), center(center_), force(force_),
   direction_dim(direction_dim) { }

void LineVolumeForceCoefficient::Eval(Vector &V, ElementTransformation &T,
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

void LineVolumeForceCoefficient::Set(double r_,Vector & center_,
                                     Vector & force_)
{
   r2=r_*r_;
   center = center_;
   force = force_;
}
void LineVolumeForceCoefficient::UpdateSize()
{
   VectorCoefficient::vdim = center.Size();
}
int Step_Armijo(TopOptProblem &problem, const double val, const double c1,
                double &step_size, const double shrink_factor, const int max_it)
{
   // obtain current point and gradient
   GridFunction &x_gf = problem.GetGridFunction();
   GridFunction &grad = problem.GetGradient();
#ifdef MFEM_USE_MPI
   auto pgrad = dynamic_cast<ParGridFunction*>(&grad);
   MPI_Comm comm;
   if (pgrad) { comm = pgrad->ParFESpace()->GetComm(); }
#endif
   // store current point
   std::unique_ptr<GridFunction> x0(MakeGridFunction(x_gf.FESpace()));
   *x0 = x_gf;

   // ρ - ρ_0
   std::unique_ptr<Coefficient> diff_density(problem.GetDensityDiffForm(*x0));
   std::unique_ptr<LinearForm> diff_densityForm(MakeLinearForm(x_gf.FESpace()));
   diff_densityForm->AddDomainIntegrator(new DomainLFIntegrator(*diff_density));

   double new_val, d;
   int i;
   step_size /= shrink_factor;
   for (i=0; i<max_it; i++)
   {
      step_size *= shrink_factor; // reduce step size
      x_gf = *x0; // restore original position
      x_gf.Add(-step_size, grad); // advance by updated step size
      new_val = problem.Eval(); // re-evaluate at the updated point
      diff_densityForm->Assemble(); // re-evaluate density difference inner-product
      d = (*diff_densityForm)(grad);
#ifdef MFEM_USE_MPI
      if (pgrad) { MPI_Allreduce(MPI_IN_PLACE, &d, 1, MPI_DOUBLE, MPI_SUM, comm); }
#else
      out << step_size << ", " << d << ", " << new_val << ",  " << std::flush;
#endif
      if (new_val < val + c1*d && d < 0) { break; }
   }

   return i;
}

HelmholtzFilter::HelmholtzFilter(FiniteElementSpace &fes,
                                 const double eps, Array<int> &ess_bdr):fes(fes),
   filter(MakeBilinearForm(&fes)), eps2(eps*eps), ess_bdr(ess_bdr)
{
   filter->AddDomainIntegrator(new DiffusionIntegrator(eps2));
   filter->AddDomainIntegrator(new MassIntegrator());
   filter->Assemble();
}
void HelmholtzFilter::Apply(Coefficient &rho, GridFunction &frho) const
{
   MFEM_ASSERT(frho.FESpace() != filter->FESpace(),
               "Filter is initialized with finite element space different from the given filtered density.");
   std::unique_ptr<LinearForm> rhoForm(MakeLinearForm(frho.FESpace()));
   rhoForm->AddDomainIntegrator(new DomainLFIntegrator(rho));

   EllipticSolver solver(*filter, *rhoForm, ess_bdr);
   solver.Solve(frho, true, false);
}
void MarkBoundary(Mesh &mesh, std::__1::function<bool(const Vector &)> mark,
                  const int idx)
{
   const int dim = mesh.Dimension();
   Vector c1(dim), c2(dim);
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      Element * be = mesh.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);
      c1.SetData(mesh.GetVertex(vertices[0]));
      c2.SetData(mesh.GetVertex(vertices[1]));
      c1 += c2; c1 *= 0.5;

      if (mark(c1))
      {
         mesh.SetBdrAttribute(i, idx);
      }
   }
}
}