#include <mfem.hpp>

using namespace mfem;

// #define DEBUG

double kinematic_viscosity = 1.0;
bool enable_nonlinear_term = true;

void PrintSubVector(const Array<int> &idx, const Vector &v)
{
   for (int i = 0; i < idx.Size(); i++)
   {
      printf("%.5E ", v[idx[i]]);
   }
   printf("\n");
}

class BDFIMEX : public ODESolver
{
protected:
   double alpha;
   Vector xn, fxn;

public:
   BDFIMEX() {};

   void Init(TimeDependentOperator &f_) override
   {
      ODESolver::Init(f_);
      xn.SetSize(f->Width(), mem_type);
      fxn.SetSize(f->Width(), mem_type);
   }

   void Step(Vector &x, double &t, double &dt) override
   {
      f->SetEvalMode(TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);

      f->Mult(x, fxn);

      f->SetTime(t + dt);
      for (int i = 0; i < fxn.Size(); i++)
      {
         fxn(i) = dt * fxn(i) + x(i);
      }

      f->ImplicitSolve(dt, fxn, x);

      xn = x;
      t += dt;
   }
};

/// @brief ARK112 in z-form
class ARK112 : public ODESolver
{
protected:
   double alpha;
   Vector w;

public:
   ARK112() {};

   void Init(TimeDependentOperator &f_) override
   {
      ODESolver::Init(f_);
      w.SetSize(f->Width(), mem_type);
   }

   void Step(Vector &x, double &t, double &dt) override
   {
      f->SetEvalMode(TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);

      f->Mult(x, w);
      for (int i = 0; i < w.Size(); i++)
      {
         w(i) = x(i) + dt * w(i);
      }
      f->SetTime(t + dt);
      f->ImplicitSolve(dt, w, x);

      t += dt;
   }
};

Array<int> StokesTrueOffsets(const ParFiniteElementSpace &vel_fes,
                             const ParFiniteElementSpace &pres_fes);

struct HelmholtzOperator : Operator
{
   ParFiniteElementSpace &fes;
   Array<int> ess_tdof_list;
   std::unique_ptr<ParBilinearForm> A;
   OperatorHandle Ahandle;
   ConstantCoefficient am_coef;
   ConstantCoefficient ak_coef;
   bool pa;
   HelmholtzOperator(ParFiniteElementSpace &fes,
                     const Array<int> &ess_bdr,
                     bool unsteady_,
                     bool pa_);
   void SetParameters(const double am, const double ak);
   virtual void Mult(const Vector &x, Vector &y) const;
};

struct StokesOperator : public Operator
{
   ParFiniteElementSpace &vel_fes, &pres_fes;
   HelmholtzOperator A;
   ParMixedBilinearForm Dform, Gform;
   ParNonlinearForm N_form;

   Array<int> ess_bdr, ess_bdr_pres, ess_bdr_empty;
   Array<int> ess_tdof_list, ess_tdof_list_pres, empty;
   ConstantCoefficient alpha;

   double dt;
   bool pa;

   int u_size, p_size;

   Vector one_v;
   mutable Vector u, p, rhs_u, rhs_p, grad_p;

   OperatorHandle Dhandle, Ghandle;
   StokesOperator(ParFiniteElementSpace &_vel_fes,
                  ParFiniteElementSpace &_pres_fes,
                  const Array<int> &ess_bdr,
                  const Array<int> &ess_bdr_pres_,
                  bool pa=true);
   void OrthoRHS(Vector &v) const;
   void SetParameters(const double am, const double ak);
   void ExtractComponents(const Vector &up, Vector &u_vec, Vector &p_vec) const;
   virtual void Mult(const Vector &x, Vector &y) const;
};

struct HelmholtzSolver : Solver
{
   HelmholtzOperator oper;
   ConstantCoefficient one;
   std::shared_ptr<Solver> solv;
   std::shared_ptr<CGSolver> cg;
   std::shared_ptr<LORSolver<HypreBoomerAMG>> prec;
   double dt;
   bool use_cg;

   HelmholtzSolver(ParFiniteElementSpace &fes,
                   const Array<int> &ess_bdr,
                   bool unsteady,
                   bool pa,
                   bool use_cg_=false,
                   int max_it = 10,
                   double tol_ = 1e-4);
   void SetParameters(const double am, const double ak);
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void SetOperator(const Operator &op) { MFEM_ABORT(""); }
};

struct StokesPreconditioner : Solver
{
   StokesPreconditioner(int w) : Solver(w) { }
   virtual void SetParameters(const double am, const double ak) = 0;
};

struct SchurComplementSolver : Operator
{
   Operator &L_solv, &M_solv;
   double am;
   double ak;
   mutable Vector w;
   SchurComplementSolver(Operator &L_solv_, Operator &M_solv_);
   void SetParameters(const double am, const double ak);
   virtual void Mult(const Vector &x, Vector &y) const;
};

struct BlockStokesPreconditioner : StokesPreconditioner
{
   StokesOperator &stokes;
   HelmholtzSolver A_solv, Lc_solv;
   std::unique_ptr<OperatorJacobiSmoother> invMp;
   std::unique_ptr<SchurComplementSolver> invSchur;
   bool unsteady, triangular, pa;
   double dt;
   mutable Vector rhs_u, rhs_p, u, p, rhs_p_minus_div_u;
   BlockStokesPreconditioner(StokesOperator &stokes_,
                             bool unsteady_,
                             bool triangular_,
                             bool pa_=true);
   void SetParameters(const double am, const double ak);
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void SetOperator(const Operator &op) { MFEM_ABORT(""); }
};

Array<int> StokesTrueOffsets(const ParFiniteElementSpace &vel_fes,
                             const ParFiniteElementSpace &pres_fes)
{
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = vel_fes.GetTrueVSize();
   offsets[2] = pres_fes.GetTrueVSize();
   offsets.PartialSum();
   return offsets;
}

HelmholtzOperator::HelmholtzOperator(ParFiniteElementSpace &fes,
                                     const Array<int> &ess_bdr,
                                     bool unsteady,
                                     bool pa_)
   : Operator(fes.GetTrueVSize()),
     fes(fes),
     pa(pa_)
{
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   if (!unsteady) { SetParameters(0.0, 1.0); }
}

void HelmholtzOperator::SetParameters(const double am,
                                      const double ak)
{
   if ((am_coef.constant == am) && (ak_coef.constant == ak))
   {
      return;
   }

   A.reset(new ParBilinearForm(&fes));
   am_coef.constant = am;
   ak_coef.constant = ak;

   if (fes.GetVDim() == 1)
   {
      if (am != 0.0)
      {
         A->AddDomainIntegrator(new MassIntegrator(am_coef));
      }
      if (ak != 0.0)
      {
         A->AddDomainIntegrator(new DiffusionIntegrator(ak_coef));
      }
   }
   else
   {
      if (am != 0.0)
      {
         A->AddDomainIntegrator(new VectorMassIntegrator(am_coef));
      }
      if (ak != 0.0)
      {
         A->AddDomainIntegrator(new VectorDiffusionIntegrator(ak_coef));
      }
   }
   if (pa) { A->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   A->Assemble();
   A->FormSystemMatrix(ess_tdof_list, Ahandle);
}

void HelmholtzOperator::Mult(const Vector &x, Vector &y) const
{
   Ahandle->Mult(x, y);
}

StokesOperator::StokesOperator(ParFiniteElementSpace &vel_fes_,
                               ParFiniteElementSpace &pres_fes_,
                               const Array<int> &ess_bdr_,
                               const Array<int> &ess_bdr_pres_,
                               bool pa_)
   : vel_fes(vel_fes_),
     pres_fes(pres_fes_),
     A(vel_fes, ess_bdr_, false, pa_),
     Dform(&vel_fes, &pres_fes),
     Gform(&pres_fes, &vel_fes),
     N_form(&vel_fes),
     ess_bdr(ess_bdr_),
     ess_bdr_pres(ess_bdr_pres_),
     dt(0.0),
     pa(pa_)
{
   u_size = vel_fes.GetTrueVSize();
   p_size = pres_fes.GetTrueVSize();
   height = u_size + p_size;
   width = u_size + p_size;

   ess_bdr_empty.SetSize(ess_bdr.Size());
   ess_bdr_empty = 0;

   vel_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   pres_fes.GetEssentialTrueDofs(ess_bdr_pres, ess_tdof_list_pres);

   // Divergence
   // If the divergence term has negative sign, then the system is symmetric
   // (and e.g. can solve with MINRES). However, conditioning is better if we
   // use positive sign, and our preconditioners need FGMRES anyway.
   // Dform.AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
   Dform.AddDomainIntegrator(new VectorDivergenceIntegrator);
   if (pa) { Dform.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   Dform.Assemble();
   Dform.FormRectangularSystemMatrix(ess_tdof_list, ess_tdof_list_pres, Dhandle);

   // Gradient
   Gform.AddDomainIntegrator(new GradientIntegrator);
   if (pa) { Gform.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   Gform.Assemble();
   Gform.FormRectangularSystemMatrix(ess_tdof_list_pres, ess_tdof_list, Ghandle);

   N_form.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   N_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   // N_form.SetEssentialTrueDofs(ess_tdof_list);
   // N_form.Assemble();
   // N_form.Update();
   N_form.Setup();

   grad_p.SetSize(u_size);
   grad_p.UseDevice(true);
   one_v.SetSize(p_size);
   one_v.UseDevice(true);
   one_v = 1.0;
}

// void StokesOperator::ReAssemble()
// {
//    A.A->Assemble();
//    A.A->FormSystemMatrix(ess_tdof_list, A.Ahandle);
//    Dform.Assemble();
//    Dform.FormRectangularSystemMatrix(ess_tdof_list, ess_tdof_list_pres, Dhandle);
//    Gform.Assemble();
//    Gform.FormRectangularSystemMatrix(ess_tdof_list_pres, ess_tdof_list, Ghandle);
// }

void StokesOperator::SetParameters(const double am, const double ak)
{
   A.SetParameters(am, ak);
}

void StokesOperator::ExtractComponents(const Vector &up, Vector &u_vec,
                                       Vector &p_vec) const
{
   // u_vec.Destroy();
   // p_vec.Destroy();
   u_vec.NewMemoryAndSize(
      Memory<double>(up.GetMemory(), 0, u_size), u_size, true);
   p_vec.NewMemoryAndSize(
      Memory<double>(up.GetMemory(), u_size, p_size), p_size, true);
}

void StokesOperator::Mult(const Vector &x, Vector &y) const
{
   x.UseDevice(true);
   y.UseDevice(true);
   x.Read();
   y.ReadWrite();

   ExtractComponents(x, u, p);
   ExtractComponents(y, rhs_u, rhs_p);

   A.Mult(u, rhs_u);
   Ghandle->Mult(p, grad_p);
   rhs_u += grad_p;
   Dhandle->Mult(u, rhs_p);

   // OrthoRHS(rhs_p);
}

void StokesOperator::OrthoRHS(Vector &y) const
{
   double local_sum = y*one_v; // Perform v.Sum(), but ensure it's done on GPU
   double global_sum = 0.0;
   MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                 pres_fes.GetComm());

   int local_size = one_v.Size();
   int global_size = 0;
   MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM,
                 pres_fes.GetComm());
   y -= global_sum/static_cast<double>(global_size);
}

HelmholtzSolver::HelmholtzSolver(ParFiniteElementSpace &fes,
                                 const Array<int> &ess_bdr,
                                 bool unsteady,
                                 bool pa,
                                 bool use_cg_,
                                 int max_it,
                                 double tol)
   : Solver(fes.GetTrueVSize()),
     oper(fes, ess_bdr, unsteady, pa),
     one(1.0),
     dt(0.0),
     use_cg(use_cg_)
{
   printf("setup LOR(AMG) for vdim %d\n", oper.fes.GetVDim());
   oper.SetParameters(0.0, 1.0);
   prec.reset(new LORSolver<HypreBoomerAMG>(*oper.A, oper.ess_tdof_list));
   static_cast<HypreBoomerAMG *>(&prec->GetSolver())->SetPrintLevel(0);
   if (use_cg)
   {
      cg.reset(new CGSolver(fes.GetComm()));
      cg->SetRelTol(tol);
      cg->SetMaxIter(max_it);
      cg->SetOperator(oper);
      cg->SetPreconditioner(*prec);
      cg->SetPrintLevel(-1);
      cg->iterative_mode = false;
      solv = cg;
   }
   else
   {
      solv = prec;
   }
}

void HelmholtzSolver::SetParameters(const double am, const double ak)
{
   // if ((am == oper.am_coef.constant) && ak == oper.ak_coef.constant)
   // {
   //    return;
   // }

   printf("setparameters setup LOR(AMG) for vdim %d\n", oper.fes.GetVDim());
   oper.SetParameters(am, ak);
   prec.reset(new LORSolver<HypreBoomerAMG>(*oper.A, oper.ess_tdof_list));
   static_cast<HypreBoomerAMG *>(&prec->GetSolver())->SetPrintLevel(0);
   if (use_cg)
   {
      cg->SetPreconditioner(*prec);
   }
}

void HelmholtzSolver::Mult(const Vector &x, Vector &y) const
{
   solv->Mult(x, y);
}

SchurComplementSolver::SchurComplementSolver(
   Operator &L_solv_, Operator &M_solv_)
   : Operator(L_solv_.Width()), L_solv(L_solv_), M_solv(M_solv_),
     w(L_solv_.Width())
{ }

void SchurComplementSolver::SetParameters(const double am, const double ak)
{
   this->am = am;
   this->ak = ak;
}

void SchurComplementSolver::Mult(const Vector &x, Vector &y) const
{
   L_solv.Mult(x, y);
   y *= ak;
   M_solv.Mult(x, w);
   w *= am;
   y += w;
}

BlockStokesPreconditioner::BlockStokesPreconditioner(
   StokesOperator &stokes_,
   bool unsteady_,
   bool triangular_,
   bool pa_)
   : StokesPreconditioner(stokes_.Width()),
     stokes(stokes_),
     A_solv(stokes.vel_fes, stokes.ess_bdr, unsteady_, true, true, 100, 1e-8),
     Lc_solv(stokes.pres_fes, stokes.ess_bdr_pres, false, true, true, 100, 1e-8),
     unsteady(unsteady_),
     triangular(triangular_),
     pa(pa_),
     dt(0.0)
{
   ParBilinearForm Mp(&stokes.pres_fes);
   Mp.AddDomainIntegrator(new MassIntegrator);
   Mp.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Mp.Assemble();
   Vector Mp_diag(stokes.pres_fes.GetTrueVSize());
   Mp.AssembleDiagonal(Mp_diag);
   invMp.reset(new OperatorJacobiSmoother(Mp_diag, stokes.empty));
   invSchur.reset(new SchurComplementSolver(Lc_solv, *invMp));
   rhs_p_minus_div_u.SetSize(stokes.p_size);
}

void BlockStokesPreconditioner::SetParameters(const double am, const double ak)
{
   A_solv.SetParameters(am, ak);
   invSchur->SetParameters(am, ak);
}

void BlockStokesPreconditioner::Mult(const Vector &b, Vector &x) const
{
   b.UseDevice(true);
   x.UseDevice(true);
   b.Read();
   x.ReadWrite();

   stokes.ExtractComponents(b, rhs_u, rhs_p);
   stokes.ExtractComponents(x, u, p);

   A_solv.Mult(rhs_u, u);
   if (triangular)
   {
      stokes.Dhandle->Mult(u, rhs_p_minus_div_u);
      rhs_p_minus_div_u *= -1.0;
      rhs_p_minus_div_u += rhs_p;
      invSchur->Mult(rhs_p_minus_div_u, p);
   }
   else
   {
      invSchur->Mult(rhs_p, p);
   }
}

/// Time-dependent operator defining the unsteady Stokes problem
struct StokesTimeDependentOperator : TimeDependentOperator
{
   ParFiniteElementSpace &vel_fes, &pres_fes;
   VectorCoefficient &fcoeff;
   VectorCoefficient &gcoeff;
   ParLinearForm fform;
   Vector f_tvec;
   ParBilinearForm Mu_form;
   OperatorHandle Mu_handle;

   ParBilinearForm Mp_form;
   OperatorHandle Mp_handle;

   CGSolver invMu;
   std::unique_ptr<OperatorJacobiSmoother> invMuPrec;

   CGSolver invMp;
   std::unique_ptr<OperatorJacobiSmoother> invMpPrec;

   FGMRESSolver outer_solver;

   mutable ParGridFunction vel_gf, tmp_gf;

   std::unique_ptr<ConstantCoefficient> lap_coeff;
   std::unique_ptr<BlockStokesPreconditioner> stokesprec;
   std::unique_ptr<StokesOperator> stokesop, stokesop_nobc;

   /// Work arrays
   mutable Vector w1, w1_u, w1_p, u, p, b_u, b_p, y_u, y_p;

   StokesTimeDependentOperator(ParFiniteElementSpace &vel_fes_,
                               ParFiniteElementSpace &pres_fes_,
                               const Array<int> &ess_bdr,
                               const Array<int> &ess_bdr_pres,
                               VectorCoefficient &fcoeff_,
                               VectorCoefficient &gcoeff_,
                               const double outer_solver_rtol,
                               const double outer_solver_atol);

   void Mult(const Vector &x, Vector &y) const override;
   /// Solve the system k = f(x+dt*k, t) for the unknown k
   void ImplicitSolve(const double dt, const Vector &x, Vector &k) override;
   int SUNImplicitSetup(const Vector &x, const Vector &fx,
                        int jok, int *jcur, double gamma) override;
   int SUNImplicitSolve(const Vector &b, Vector &x, double tol) override;
   void SetTime(const double t_) override;
   static int ARKStagePredictPostProcess(double t, N_Vector zpred,
                                         void *user_data);
};

StokesTimeDependentOperator::StokesTimeDependentOperator(
   ParFiniteElementSpace &vel_fes_,
   ParFiniteElementSpace &pres_fes_,
   const Array<int> &ess_bdr,
   const Array<int> &ess_bdr_pres,
   VectorCoefficient &fcoeff_,
   VectorCoefficient &gcoeff_,
   const double outer_solver_rtol,
   const double outer_solver_atol)
   : TimeDependentOperator(vel_fes_.GetTrueVSize() + pres_fes_.GetTrueVSize()),
     vel_fes(vel_fes_),
     pres_fes(pres_fes_),
     fcoeff(fcoeff_),
     gcoeff(gcoeff_),
     fform(&vel_fes),
     f_tvec(vel_fes.GetTrueVSize()),
     Mu_form(&vel_fes),
     Mp_form(&pres_fes),
     invMu(MPI_COMM_WORLD),
     invMp(MPI_COMM_WORLD),
     outer_solver(MPI_COMM_WORLD),
     vel_gf(&vel_fes),
     tmp_gf(&vel_fes)
{
   Array<int> empty;

   // Stokes operator
   stokesop.reset(new StokesOperator(vel_fes, pres_fes, ess_bdr, ess_bdr_pres,
                                     true));

   Array<int> ess_bdr_empty(ess_bdr.Size());
   ess_bdr_empty = 0;
   stokesop_nobc.reset(new StokesOperator(vel_fes, pres_fes, ess_bdr_empty,
                                          ess_bdr_empty, true));

   // Forcing term
   fform.AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));

   // Velocity mass matrix
   Mu_form.AddDomainIntegrator(new VectorMassIntegrator);
   Mu_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Mu_form.Assemble();
   Mu_form.FormSystemMatrix(empty, Mu_handle);

   // Pressure mass matrix
   Mp_form.AddDomainIntegrator(new MassIntegrator);
   Mp_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Mp_form.Assemble();
   Mp_form.FormSystemMatrix(empty, Mp_handle);

   // Stokes preconditioner
   bool triangular = true;
   int npatches = 1;
   stokesprec.reset(new BlockStokesPreconditioner(*stokesop, true, triangular,
                                                  npatches));

   invMuPrec.reset(new OperatorJacobiSmoother(Mu_form, stokesop->empty));
   invMu.SetOperator(*Mu_handle);
   invMu.SetPreconditioner(*invMuPrec);
   invMu.SetRelTol(1e-8);
   invMu.SetMaxIter(50);
   invMu.SetPrintLevel(0);

   invMpPrec.reset(new OperatorJacobiSmoother(Mp_form, stokesop->empty));
   invMp.SetOperator(*Mp_handle);
   invMp.SetPreconditioner(*invMpPrec);
   invMp.SetRelTol(1e-8);
   invMp.SetMaxIter(50);
   invMp.SetPrintLevel(0);

   // Setup GMRES solver
   outer_solver.SetRelTol(outer_solver_rtol);
   outer_solver.SetAbsTol(outer_solver_atol);
   outer_solver.SetMaxIter(200);
   outer_solver.SetKDim(30);
   outer_solver.SetOperator(*stokesop);
   outer_solver.SetPreconditioner(*stokesprec);
   outer_solver.SetPrintLevel(2);

   // Work vectors
   w1.SetSize(vel_fes.GetTrueVSize() + pres_fes.GetTrueVSize());
   w1.UseDevice(true);
   w1.ReadWrite();

   // SetTime(0.0);
}

void StokesTimeDependentOperator::ImplicitSolve(const double gamma,
                                                const Vector &fxn,
                                                Vector &x)
{
   stokesop->SetParameters(1.0, gamma * kinematic_viscosity);
   stokesprec->SetParameters(1.0, gamma * kinematic_viscosity);
   stokesprec->invSchur->SetParameters(gamma * kinematic_viscosity,
                                       kinematic_viscosity);

   stokesop->ExtractComponents(fxn, b_u, b_p);
   stokesop->ExtractComponents(x, u, p);
   stokesop->ExtractComponents(w1, w1_u, w1_p);

   Mu_handle->Mult(b_u, w1_u);

   w1_p = 0.0;

   vel_gf.Distribute(u);
   vel_gf.ProjectBdrCoefficient(gcoeff, stokesop->ess_bdr);
   vel_gf.ParallelProject(u);

#ifdef DEBUG
   printf("ImplicitSolve ess dofs: ");
   PrintSubVector(stokesop->ess_tdof_list, u);
#endif

   auto Ac = stokesop->A.Ahandle.As<ConstrainedOperator>();
   Ac->EliminateRHS(u, w1_u);

   auto Dc = stokesop->Dhandle.As<RectangularConstrainedOperator>();
   Dc->EliminateRHS(u, w1_p);

   // stokesop->OrthoRHS(w1_p);

   outer_solver.Mult(w1, x);
}

void StokesTimeDependentOperator::Mult(const Vector &x, Vector &y) const
{
   if (eval_mode == EvalMode::ADDITIVE_TERM_1 && enable_nonlinear_term)
   {
      stokesop->ExtractComponents(x, u, p);
      stokesop->ExtractComponents(w1, w1_u, w1_p);
      stokesop->ExtractComponents(y, y_u, y_p);

      stokesop->N_form.Mult(u, w1_u);
      w1_u.Neg();

      invMu.Mult(w1_u, y_u);
      // y_u = w1_u;
      // y_u.SetSubVector(stokesop->ess_tdof_list, 0.0);
      y_p = 0.0;
   }
   else if (eval_mode == EvalMode::ADDITIVE_TERM_2)
   {
      stokesop->ExtractComponents(x, u, p);
      stokesop->ExtractComponents(w1, w1_u, w1_p);
      stokesop->ExtractComponents(y, y_u, y_p);

      stokesop_nobc->SetParameters(0.0, kinematic_viscosity);
      stokesop_nobc->Mult(x, w1);

      w1.Neg();

      invMu.Mult(w1_u, y_u);
      // y_u = w1_u;
      // y_u.SetSubVector(stokesop->ess_tdof_list, 0.0);
      y_p = 0.0;
   }
}

int StokesTimeDependentOperator::SUNImplicitSetup(const Vector &x,
                                                  const Vector &fx,
                                                  int jok, int *jcur, double gamma)
{
   stokesop->SetParameters(1.0, gamma * kinematic_viscosity);
   stokesprec->SetParameters(1.0, gamma * kinematic_viscosity);
   stokesprec->invSchur->SetParameters(gamma * kinematic_viscosity,
                                       kinematic_viscosity);

   *jcur = 1;
   return 0;
}

int StokesTimeDependentOperator::SUNImplicitSolve(const Vector &b, Vector &x,
                                                  double tol)
{
   // Sundials is solving for a correction here. x is always zero and dirichlet
   // boundary conditions logically don't have to be corrected, hence
   // x[ess_tdof] = 0.
   stokesop->ExtractComponents(b, b_u, b_p);
   stokesop->ExtractComponents(x, u, p);
   stokesop->ExtractComponents(w1, w1_u, w1_p);

#ifdef DEBUG
   printf("SUNImplicitSolve ess dofs u: ");
   PrintSubVector(stokesop->ess_tdof_list, u);
#endif

   Mu_handle->Mult(b_u, w1_u);
   w1_p = 0.0;

#ifdef DEBUG
   printf("SUNImplicitSolve ess dofs w1_u: ");
   PrintSubVector(stokesop->ess_tdof_list, w1_u);
#endif

   // stokesop->OrthoRHS(w1_p);

   outer_solver.SetRelTol(tol);
   outer_solver.Mult(w1, x);

#ifdef DEBUG
   printf("SUNImplicitSolve ess dofs u: ");
   PrintSubVector(stokesop->ess_tdof_list, u);
#endif

   if (!outer_solver.GetConverged())
   {
      return -1;
   }

   return 0;
}

void StokesTimeDependentOperator::SetTime(const double t_)
{
   if (t != t_)
   {
      t = t_;
      fcoeff.SetTime(t);
      fform.Update();
      fform.Assemble();
      fform.ParallelAssemble(f_tvec);
      gcoeff.SetTime(t);
   }
}

int StokesTimeDependentOperator::ARKStagePredictPostProcess(double t,
                                                            N_Vector zpred_nv,
                                                            void *user_data)
{
   SundialsNVector zpred(zpred_nv);

   auto *ark = static_cast<ARKStepSolver*>(user_data);
   auto *self = static_cast<StokesTimeDependentOperator*>(ark->GetOperator());

   self->stokesop->ExtractComponents(zpred, self->w1_u, self->w1_p);

   self->SetTime(t);
   self->vel_gf.Distribute(self->w1_u);
   self->vel_gf.ProjectBdrCoefficient(self->gcoeff, self->stokesop->ess_bdr);
   self->vel_gf.ParallelProject(self->w1_u);

#ifdef DEBUG
   printf("ARKStagePredictPostProcess ess dofs: ");
   PrintSubVector(self->stokesop->ess_tdof_list, zpred);
#endif

   return 0;
}

// void vel_ex(const Vector &xvec, double t, Vector &u)
// {
//    double x = xvec(0);
//    double y = xvec(1);
//    u(0) = M_PI*sin(t)*pow(sin(M_PI*x),2.0)*sin(2.0*M_PI*y);
//    u(1) = -(M_PI*sin(t)*sin(2.0*M_PI*x)*pow(sin(M_PI*y),2.0));
// }

// double p_ex(const Vector &xvec, double t)
// {
//    double x = xvec(0);
//    double y = xvec(1);
//    return cos(M_PI*x)*sin(t)*sin(M_PI*y);
// }

// void f(const Vector &xvec, double t, Vector &fvec)
// {
//    double x = xvec(0);
//    double y = xvec(1);
//    fvec(0) = (M_PI * pow(sin(M_PI * x),
//                          2) * cos(t) * sin((2*M_PI) * y) - kinematic_viscosity * ((2 * pow(M_PI,
//                                                                                    3)) * pow(cos(M_PI * x),
//                                                                                          2) * sin(t) * sin((2*M_PI) * y) - (6 * pow(M_PI, 3)) * pow(sin(M_PI * x),
//                                                                                                2) * sin(t) * sin((2*M_PI) * y))) - M_PI * sin(t) * sin(M_PI * x) * sin(
//                 M_PI * y);

//    fvec(1) = (M_PI * sin(t) * cos(M_PI * x) * cos(M_PI * y) - kinematic_viscosity
//               * ((6 * pow(M_PI,
//                           3)) * pow(sin(M_PI * y), 2) * sin(t) * sin((2*M_PI) * x) - (2 * pow(M_PI,
//                                 3)) * pow(cos(M_PI * y),
//                                           2) * sin(t) * sin((2*M_PI) * x))) - M_PI * pow(sin(M_PI * y),
//                                                 2) * cos(t) * sin((2*M_PI) * x);
// }

// // forcing term with nonlinear contribution
// void f(const Vector &xvec, double t, Vector &fvec)
// {
//    double x = xvec(0);
//    double y = xvec(1);
//    fvec(0) = M_PI * sin(t) * sin(M_PI * x) * sin(M_PI * y)
//              * (-1.0
//                 + 2.0 * pow(M_PI, 2.0) * sin(t) * sin(M_PI * x)
//                 * sin(2.0 * M_PI * x) * sin(M_PI * y))
//              + M_PI
//              * (2.0 * kinematic_viscosity * pow(M_PI, 2.0)
//                 * (1.0 - 2.0 * cos(2.0 * M_PI * x)) * sin(t)
//                 + cos(t) * pow(sin(M_PI * x), 2.0))
//              * sin(2.0 * M_PI * y);

//    fvec(1) = M_PI * cos(M_PI * y) * sin(t)
//              * (cos(M_PI * x)
//                 + 2.0 * kinematic_viscosity * pow(M_PI, 2.0) * cos(M_PI * y)
//                 * sin(2.0 * M_PI * x))
//              - M_PI * (cos(t) + 6.0 * kinematic_viscosity * pow(M_PI, 2.0) * sin(t))
//              * sin(2.0 * M_PI * x) * pow(sin(M_PI * y), 2.0)
//              + 4.0 * pow(M_PI, 3.0) * cos(M_PI * y) * pow(sin(t), 2.0)
//              * pow(sin(M_PI * x), 2.0) * pow(sin(M_PI * y), 3.0);
// }

void vel_ex_problem1(const Vector &xvec, double t, Vector &u)
{
   double x = xvec(0);
   double y = xvec(1);
   double a = exp(-4.0*kinematic_viscosity*M_PI*M_PI*t);
   u(0) = -sin(2.0*M_PI*y) * a;
   u(1) = sin(2.0*M_PI*x) * a;
}

double p_ex_problem1(const Vector &xvec, double t)
{
   double x = xvec(0);
   double y = xvec(1);
   double a = exp(-8.0*kinematic_viscosity*M_PI*M_PI*t);
   return -cos(2.0*M_PI*x) * cos(2.0*M_PI*y) * a;
}

void f(const Vector &, double, Vector &fvec)
{
   fvec = 0.0;
}

void vel_ex_problem2(const Vector &xvec, double t, Vector &u)
{
   double x = xvec(0);
   double y = xvec(1);
   const double H = 0.41;
   const double U = 1.5;
   const double ramp_peak_at = 0.0;
   const double ramp = 0.5 * (1.0 - cos(M_PI / ramp_peak_at * t));
   if (x == 0.0)
   {
      if (t < ramp_peak_at)
      {
         u(0) = 4.0 * U * y * (H - y) / (H*H) * ramp;
      }
      else
      {
         u(0) = 4.0 * U * y * (H - y) / (H*H);
      }
      // if (t < ramp_peak_at)
      // {
      //    u(0) = U * 1.5 * y * (H - y) / pow(H / 2.0, 2.0) * ramp;
      // }
      // else
      // {
      //    u(0) = U * 1.5 * y * (H - y) / pow(H / 2.0, 2.0);
      // }
   }
   else
   {
      u(0) = 0.0;
   }
   u(1) = 0.0;
}

double p_ex_problem2(const Vector &xvec, double t)
{
   return 0.0;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);

   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Read in arguments
   const char *device_config = "cpu";
   int order = 2;
   int ref_levels = 0;
   int npatches = 1;
   double t_final = 1.0;
   double dt = 1e-3;
   int ode_solver_type = 3;
   double outer_solver_rtol = 1e-4;
   double outer_solver_atol = 1e-12;
   int problem_type = 1;

   OptionsParser args{argc, argv};
   args.AddOption(&problem_type, "-prob", "--problem", "problem type");
   args.AddOption(&order, "-o", "--order", "Polynomial degree");
   args.AddOption(&ref_levels, "-r", "--refine", "Number of refinement levels");
   args.AddOption(&npatches, "-n", "--npatches",
                  "Number of patches to use in additive Schwarz method");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "            12 - Implicit Midpoint, 13 - SDIRK3a, 14 - SDIRK4");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time");
   args.AddOption(&dt, "-dt", "--delta-t", "Time step");
   args.AddOption(&kinematic_viscosity, "-kv", "--kv", "kinematic viscosity");
   args.AddOption(&outer_solver_rtol, "-outer_rtol", "--outer-rtol",
                  "Outer solver relative tolerance");
   args.AddOption(&outer_solver_atol, "-outer_atol", "--outer-atol",
                  "Outer solver absolute tolerance");
   args.AddOption(&enable_nonlinear_term, "-nl", "--nonlinear", "-no-nl",
                  "--no-nonlinear", "Enable nonlinear term.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();

   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(std::cout);
   }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   int vel_order = order;
   int pres_order = order - 1;


   Mesh *mesh = nullptr;
   if (problem_type == 1)
   {
      mesh = new Mesh(Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL));
      mesh->SetCurvature(1);
      *mesh->GetNodes() -= 0.5;
   }
   else if (problem_type == 2)
   {
      mesh = new Mesh("fsi.msh");
   }

   int dim = mesh->Dimension();
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }
   ParMesh pmesh{MPI_COMM_WORLD, *mesh};

   if (Mpi::Root())
   {
      printf("Mesh #ne: %lld\n", pmesh.GetGlobalNE());
   }

   // Define ODE Solver
   std::unique_ptr<ODESolver> ode_solver;
   switch (ode_solver_type)
   {
      case 1:
         ode_solver.reset(new BDFIMEX); break;
      case 2:
         ode_solver.reset(new ARK112); break;
      case 3:
         ode_solver.reset(new ARKStepSolver(MPI_COMM_WORLD,
                                            ARKStepSolver::IMEX));
         break;
      default:
         MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type << "\n");
   }

   // Define vector FE space for velocity and scalar FE space for pressure
   H1_FECollection vel_fec{vel_order, dim};
   H1_FECollection pres_fec{pres_order};

   ParFiniteElementSpace vel_fes{&pmesh, &vel_fec, dim};
   ParFiniteElementSpace pres_fes{&pmesh, &pres_fec};

   int vel_global_vsize = vel_fes.GlobalVSize();
   int pres_global_vsize = pres_fes.GlobalVSize();
   if (myid == 0)
   {
      std::cout << "Velocity dofs: " << vel_global_vsize << std::endl;
      std::cout << "Pressure dofs: " << pres_global_vsize << std::endl;
   }

   // Homogeneous Dirichlet boundary conditions
   Array<int> ess_bdr, ess_bdr_pres;
   std::function<void(const Vector &, double, Vector &)> vel_exf;
   std::function<double(const Vector &, double)> p_exf;
   if (problem_type == 1)
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      vel_exf = vel_ex_problem1;

      ess_bdr_pres.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr_pres = 0;
      p_exf = p_ex_problem1;
   }
   else if (problem_type == 2)
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      // Outlet
      ess_bdr[1] = 0;
      // Beam
      ess_bdr[4] = 0;
      vel_exf = vel_ex_problem2;

      ess_bdr_pres.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr_pres = 0;
      ess_bdr_pres[1] = 1;
      p_exf = p_ex_problem2;
   }

   // Forcing term
   VectorFunctionCoefficient fcoeff(dim, f);

   // Dirichlet
   VectorFunctionCoefficient uexcoeff(dim, vel_exf);

   // Setup time dependent Stokes operator
   StokesTimeDependentOperator stokes(vel_fes, pres_fes, ess_bdr, ess_bdr_pres,
                                      fcoeff, uexcoeff,
                                      outer_solver_rtol, outer_solver_atol);

   int u_size = vel_fes.GetTrueVSize();
   int p_size = pres_fes.GetTrueVSize();

   // Combined solution X = [u, p] and RHS B=[rhs_u, rhs_p]
   Vector X(u_size+p_size), B(u_size+p_size);
   X.UseDevice(true);
   B.UseDevice(true);
   X.ReadWrite();
   B.ReadWrite();

   Vector u, p;
   stokes.stokesop->ExtractComponents(X, u, p);

   ParGridFunction u_gf(&vel_fes);
   u_gf.ProjectCoefficient(uexcoeff);
   u_gf.ParallelProject(u);

   ParGridFunction uerr_gf(u_gf.ParFESpace());
   uerr_gf.ProjectCoefficient(uexcoeff);
   for (int i = 0; i < uerr_gf.Size(); i++)
   {
      uerr_gf(i) = abs(uerr_gf(i) - u_gf(i));
   }

   ParGridFunction uex_gf(&vel_fes);
   uex_gf.ProjectCoefficient(uexcoeff);

   FunctionCoefficient pexcoeff{p_exf};
   ParGridFunction p_gf(&pres_fes);
   p_gf.ProjectCoefficient(pexcoeff);
   p_gf.ParallelProject(p);

   ParGridFunction perr_gf(p_gf.ParFESpace());
   perr_gf.ProjectCoefficient(pexcoeff);
   for (int i = 0; i < perr_gf.Size(); i++)
   {
      perr_gf(i) = abs(perr_gf(i) - p_gf(i));
   }

   ParGridFunction pex_gf(p_gf.ParFESpace());
   pex_gf.ProjectCoefficient(pexcoeff);

   double t = 0.0;
   stokes.SetTime(t);
   ode_solver->Init(stokes);
   if (ode_solver_type == 3)
   {
      ARKStepSolver *ark = static_cast<ARKStepSolver *>(ode_solver.get());
      ark->SetSStolerances(1e-4, 1e-4);
      ark->SetIMEXTableNum(ARKODE_ARK324L2SA_ERK_4_2_3, ARKODE_ARK324L2SA_DIRK_4_2_3);
      // double ae[2][2] = {{0, 0}, {1, 0}};
      // double be[2] = {0, 1};
      // double ce[2] = {0, 1};
      // double de[2] = {0, 0};
      // auto forward_euler = ARKodeButcherTable_Create(2, 1, 0, ce, *ae, be, de);
      // double ai[2][2] = {{0, 0}, {0, 1}};
      // double bi[2] = {1, 0};
      // double ci[2] = {0, 1};
      // double di[2] = {0, 0};
      // auto backward_euler = ARKodeButcherTable_Create(2, 1, 0, ci, *ai, bi, di);
      // ARKStepSetTables(ark->GetMem(), 1, 0, backward_euler, forward_euler);

      ark->SetFixedStep(dt);
      // ark->SetMaxStep(dt);
      ARKStepSetLinear(ark->GetMem(), 1);
      int flag = ARKStepSetStagePredictFn(ark->GetMem(),
                                          StokesTimeDependentOperator::ARKStagePredictPostProcess);
      MFEM_VERIFY(flag >= 0, "error in ARKStepSetStagePredictFn()");
      flag = ARKStepSetPostprocessStepFn(ark->GetMem(),
                                         StokesTimeDependentOperator::ARKStagePredictPostProcess);
      MFEM_VERIFY(flag >= 0, "error in ARKStepSetPostprocessStepFn()");
      flag = ARKStepSetPostprocessStageFn(ark->GetMem(),
                                          StokesTimeDependentOperator::ARKStagePredictPostProcess);
      MFEM_VERIFY(flag >= 0, "error in ARKStepSetPostprocessStageFn()");
   }
   int istep = 0;

   ParaViewDataCollection pvdc("fluid_output", &pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(vel_order);
   pvdc.RegisterField("velocity", &u_gf);
   pvdc.RegisterField("velocity_error", &uerr_gf);
   pvdc.RegisterField("velocity_exact", &uex_gf);
   pvdc.RegisterField("pressure", &p_gf);
   pvdc.RegisterField("pressure_error", &perr_gf);
   pvdc.RegisterField("pressure_exact", &pex_gf);
   pvdc.RegisterField("velocity_temp", &stokes.tmp_gf);
   pvdc.SetCycle(istep);
   pvdc.SetTime(t);
   pvdc.Save();

   while (true)
   {
      // if (istep == 10)
      // {
      //    dt = 1e-7;
      //    printf("bump dt to %.3E\n", dt);
      // }
      ode_solver->Step(X, t, dt);

      // Extract the results from the vector of true DOFs
      u_gf.Distribute(u);
      p_gf.Distribute(p);

      // Setup all integration rules for any element type
      int order_quad = std::max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }
      // Compute error by comparing against exact solution
      uexcoeff.SetTime(t);
      pexcoeff.SetTime(t);

      uex_gf.ProjectCoefficient(uexcoeff);
      for (int i = 0; i < uerr_gf.Size(); i++)
      {
         uerr_gf(i) = abs(uex_gf(i) - u_gf(i));
      }

      pex_gf.ProjectCoefficient(pexcoeff);
      for (int i = 0; i < perr_gf.Size(); i++)
      {
         perr_gf(i) = abs(pex_gf(i) - p_gf(i));
      }

      double err_u = u_gf.ComputeL2Error(uexcoeff, irs);
      double norm_u = ComputeGlobalLpNorm(2, uexcoeff, pmesh, irs);

      double err_p = p_gf.ComputeL2Error(pexcoeff, irs);
      double norm_p = ComputeGlobalLpNorm(2, pexcoeff, pmesh, irs);

      if (Mpi::Root())
      {
         printf("%d %.3E %.3E %.3E %.3E %.3E %.3E errlog\n", istep, t, dt, err_u, err_p,
                err_u / norm_u, err_p / norm_p);
         // if (ode_solver_type == 3)
         // {
         //    static_cast<ARKStepSolver *>(ode_solver.get())->PrintInfo();
         // }
      }

      istep++;

      if (istep % 50 == 0)
      {
         pvdc.SetCycle(istep);
         pvdc.SetTime(t);
         pvdc.Save();
      }

      if (t >= t_final - dt/2)
      {
         break;
      }
   }

   return 0;
}
