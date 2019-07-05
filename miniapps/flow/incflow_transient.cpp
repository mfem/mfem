#include "mfem.hpp"
#include "vec_conv_integrator.hpp"
#include "schurlsc.hpp"

using namespace std;
using namespace mfem;

enum PROB_TYPE
{
   TGV,
   CYL,
};

struct OptionSet
{
   PROB_TYPE prob_type;
   double rey;
} opt_;

void vel_cyl(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double U = 1.5;

   if (xi == 0.0)
   {
      u(0) = 4.0 * U * yi * (0.41 - yi) / (pow(0.41, 2.0));
   }
   else
   {
      u(0) = 0.0;
   }
   u(1) = 0.0;
}

void vel_ex(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double tgF = exp(-2.0 / opt_.rey * t);

   u(0) = cos(xi) * sin(yi) * tgF;
   u(1) = -sin(xi) * cos(yi) * tgF;
}

double p_ex(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);

   double tgF = exp(-2.0 / opt_.rey * t);

   return -1.0 / 4.0 * (cos(2.0 * xi) + cos(2.0 * yi)) * pow(tgF, 2.0);
}

class NavierStokesOperator : public Operator
{
public:
   ParMesh *pmesh_;
   Array<ParFiniteElementSpace *> fes_;
   Array<int> ess_bdr_attr_;
   Array<int> ess_tdof_list_;

   Array<int> block_offsets_;
   Array<int> block_trueOffsets_;

   ConstantCoefficient *dtcoeff;

   ParNonlinearForm *N;
   ParBilinearForm *sform;
   ParLinearForm *fform;
   ParBilinearForm *mpform;
   ParBilinearForm *mvform;
   ParMixedBilinearForm *dform;

   HypreParMatrix *S;
   HypreParMatrix *Mv;
   HypreParMatrix *Mp;
   HypreParMatrix *D;
   HypreParMatrix *G;
   mutable HypreParMatrix *NjacS;

   BlockOperator *jac;
   BlockOperator *lin;

   HypreSolver *invS;
   HypreSolver *invMp;
   SchurLSC *schurLSC;
   BlockDiagonalPreconditioner *stokesprec;
   IterativeSolver *jac_solver;
   NewtonSolver newton_solver;

   NavierStokesOperator(Array<ParFiniteElementSpace *> &fes) : Operator(
                                                                   fes[0]->TrueVSize() + fes[1]->TrueVSize()),
                                                               pmesh_(fes[0]->GetParMesh()), fes_(fes),
                                                               ess_bdr_attr_(pmesh_->bdr_attributes.Max()),
                                                               NjacS(nullptr),
                                                               newton_solver(MPI_COMM_WORLD)
   {
      if (opt_.prob_type == PROB_TYPE::CYL)
      {
         ess_bdr_attr_[0] = 1;
         ess_bdr_attr_[1] = 1;
         ess_bdr_attr_[2] = 1;
         ess_bdr_attr_[3] = 0;
      }
      else if (opt_.prob_type == PROB_TYPE::TGV)
      {
         ess_bdr_attr_ = 1;
      }

      fes_[0]->GetEssentialTrueDofs(ess_bdr_attr_, ess_tdof_list_);

      block_offsets_.SetSize(3);
      block_offsets_[0] = 0;
      block_offsets_[1] = fes[0]->GetVSize();
      block_offsets_[2] = fes[1]->GetVSize();
      block_offsets_.PartialSum();

      block_trueOffsets_.SetSize(3);
      block_trueOffsets_[0] = 0;
      block_trueOffsets_[1] = fes[0]->TrueVSize();
      block_trueOffsets_[2] = fes[1]->TrueVSize();
      block_trueOffsets_.PartialSum();
   }

   void BuildOperator(const double dt)
   {
      dtcoeff = new ConstantCoefficient(dt);

      N = new ParNonlinearForm(fes_[0]);
      N->AddDomainIntegrator(new VectorConvectionNLFIntegrator(*dtcoeff));
      N->SetEssentialTrueDofs(ess_tdof_list_);

      sform = new ParBilinearForm(fes_[0]);
      ConstantCoefficient kin_visc(dt / opt_.rey);
      sform->AddDomainIntegrator(new VectorMassIntegrator);
      sform->AddDomainIntegrator(new VectorDiffusionIntegrator(kin_visc));
      sform->Assemble();
      S = new HypreParMatrix;
      sform->FormSystemMatrix(ess_tdof_list_, *S);

      dform = new ParMixedBilinearForm(fes_[0], fes_[1]);
      dform->AddDomainIntegrator(new VectorDivergenceIntegrator(*dtcoeff));
      dform->Assemble();
      D = new HypreParMatrix;
      dform->FormColSystemMatrix(ess_tdof_list_, *D);

      G = D->Transpose();
      // (*G) *= -1.0;

      mvform = new ParBilinearForm(fes_[0]);
      mvform->AddDomainIntegrator(new VectorMassIntegrator);
      mvform->Assemble();
      mvform->Finalize();
      Mv = mvform->ParallelAssemble();

      mpform = new ParBilinearForm(fes_[1]);
      mpform->AddDomainIntegrator(new MassIntegrator);
      mpform->Assemble();
      mpform->Finalize();
      Mp = mpform->ParallelAssemble();

      jac = new BlockOperator(block_trueOffsets_);
      jac->SetBlock(0, 0, S);
      jac->SetBlock(0, 1, G);
      jac->SetBlock(1, 0, D);

      lin = new BlockOperator(block_trueOffsets_);
      lin->SetBlock(0, 0, S);
      lin->SetBlock(0, 1, G);
      lin->SetBlock(1, 0, D);

      invS = new HypreBoomerAMG(*S);
      static_cast<HypreBoomerAMG *>(invS)->SetPrintLevel(0);
      invS->iterative_mode = false;

      invMp = new HypreBoomerAMG(*Mp);
      static_cast<HypreBoomerAMG *>(invMp)->SetPrintLevel(0);
      invMp->iterative_mode = false;

      schurLSC = new SchurLSC(D, G);

      stokesprec = new BlockDiagonalPreconditioner(block_trueOffsets_);
      stokesprec->SetDiagonalBlock(0, invS);
      stokesprec->SetDiagonalBlock(1, schurLSC);

      jac_solver = new MINRESSolver(MPI_COMM_WORLD);
      jac_solver->iterative_mode = false;
      jac_solver->SetAbsTol(1e-12);
      jac_solver->SetRelTol(1e-5);
      jac_solver->SetMaxIter(500);
      jac_solver->SetOperator(*jac);
      jac_solver->SetPreconditioner(*stokesprec);
      jac_solver->SetPrintLevel(2);

      newton_solver.iterative_mode = true;
      newton_solver.SetSolver(*jac_solver);
      newton_solver.SetOperator(*this);
      newton_solver.SetPrintLevel(1);
      newton_solver.SetAbsTol(1e-12);
      newton_solver.SetRelTol(1e-8);
      newton_solver.SetMaxIter(15);
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      Vector tmp(block_trueOffsets_[1]);
      Vector vel_in(x.GetData(), block_trueOffsets_[1]);
      Vector vel_out(y.GetData(), block_trueOffsets_[1]);

      // Apply linear BlockOperator
      lin->Mult(x, y);

      // Apply nonlinear action to velocity
      N->Mult(vel_in, tmp);
      vel_out += tmp;
   }

   virtual Operator &GetGradient(const Vector &x) const
   {
      Vector u(x.GetData(), block_trueOffsets_[1]);

      delete NjacS;

      hypre_ParCSRMatrix *NjacS_wrap;
      hypre_ParcsrAdd(1.0, *static_cast<HypreParMatrix *>(&(N->GetGradient(u))), 1.0,
                      *S, &NjacS_wrap);
      NjacS = new HypreParMatrix(NjacS_wrap);

      HypreParMatrix *NjacS_e = NjacS->EliminateRowsCols(ess_tdof_list_);
      delete NjacS_e;

      jac->SetBlock(0, 0, NjacS);

      invS->SetOperator(*NjacS);

      schurLSC->SetA(NjacS);

      return *jac;
   }

   virtual ~NavierStokesOperator()
   {
      delete dtcoeff;
      delete sform;
      delete mvform;
      delete mpform;
      delete dform;
      delete N;
      delete S;
      delete Mv;
      delete Mp;
      delete D;
      delete G;
      if (NjacS != nullptr)
      {
         delete NjacS;
      }
      delete jac;
      delete lin;
      delete invS;
      delete invMp;
      delete stokesprec;
      delete jac_solver;
      delete schurLSC;
   }
};

class TDNavierStokesOperator : public TimeDependentOperator
{
private:
   NavierStokesOperator *nso_;

public:
   TDNavierStokesOperator(NavierStokesOperator *nso) :
      TimeDependentOperator(nso->fes_[0]->TrueVSize()), nso_(nso) {}

   void ImplicitSolve(const double dt,
                      const Vector &X, Vector &dX_dt)
   {
      BlockVector xh(nso_->block_offsets_);
      BlockVector b(nso_->block_offsets_);

      xh = 0.0;
      b = 0.0;

      BlockVector Xh(nso_->block_trueOffsets_);
      BlockVector B(nso_->block_trueOffsets_);

      Xh = 0.0;
      B = 0.0;

      VectorFunctionCoefficient *velbdrcoeff = nullptr;
      if (opt_.prob_type == PROB_TYPE::CYL)
      {
         velbdrcoeff = new VectorFunctionCoefficient(nso_->pmesh_->Dimension(), vel_cyl);
      }
      else if (opt_.prob_type == PROB_TYPE::TGV)
      {
         velbdrcoeff = new VectorFunctionCoefficient(nso_->pmesh_->Dimension(), vel_ex);
      }
      velbdrcoeff->SetTime(GetTime());

      ParGridFunction vel_gf;
      vel_gf.MakeRef(nso_->fes_[0], xh.GetBlock(0));
      vel_gf.ProjectBdrCoefficient(*velbdrcoeff, nso_->ess_bdr_attr_);

      nso_->sform->FormLinearSystem(nso_->ess_tdof_list_, xh.GetBlock(0), b.GetBlock(0),
                                    *(nso_->S), Xh.GetBlock(0), B.GetBlock(0), 1);

      nso_->dform->FormColLinearSystem(nso_->ess_tdof_list_, xh.GetBlock(0), b.GetBlock(1),
                                       *(nso_->D), Xh.GetBlock(0), B.GetBlock(1), 1);

      Vector V(nso_->block_trueOffsets_[1]);
      nso_->Mv->Mult(X, V);
      V.SetSubVector(nso_->ess_tdof_list_, 0.0);

      B.GetBlock(0) += V;

      // Retrieve initial guess from previous solution
      Xh.GetBlock(0) = X;

      nso_->newton_solver.Mult(B, Xh);

      subtract(1.0 / dt, Xh.GetBlock(0), X, dX_dt);

      delete velbdrcoeff;
   }
};

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   int prob_type = 0;
   int serial_ref_levels = 0;
   int order = 2;
   double t_final = 1e-1;
   int num_steps = 40;
   double dt = t_final / num_steps;

   opt_.rey = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Polynomial order for the velocity.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&opt_.rey, "-rey", "--reynolds", "Choose Reynolds number.");
   args.AddOption(&dt, "-dt", "--timestep", "Timestep.");
   args.AddOption(&t_final, "-tf", "--tfinal", "Final time.");
   args.AddOption(&prob_type, "-prob", "--problem_type",
                  "Choose problem type\n\t"
                  "0 - MMS\n\t"
                  "1 - FLow past a cylinder\n\t");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   opt_.prob_type = static_cast<PROB_TYPE>(prob_type);

   int vel_order = order;
   int pres_order = order - 1;

   Mesh *mesh = nullptr;

   if (opt_.prob_type == PROB_TYPE::CYL)
   {
      mesh = new Mesh("cyl.msh");
   }
   else if (opt_.prob_type == PROB_TYPE::TGV)
   {
      mesh = new Mesh(1, 1, Element::Type::QUADRILATERAL, false, 2.0 * M_PI, 2.0 * M_PI);
   }

   int dim = mesh->Dimension();

   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   FiniteElementCollection *vel_fec = new H1_FECollection(vel_order, dim);
   FiniteElementCollection *pres_fec = new H1_FECollection(pres_order);

   ParFiniteElementSpace *vel_fes = new ParFiniteElementSpace(pmesh, vel_fec, dim);
   ParFiniteElementSpace *pres_fes = new ParFiniteElementSpace(pmesh, pres_fec);

   Array<ParFiniteElementSpace *> fes(2);
   fes[0] = vel_fes;
   fes[1] = pres_fes;

   int fes_size0 = fes[0]->GlobalVSize();
   int fes_size1 = fes[1]->GlobalVSize();

   if (myid == 0)
   {
      cout << "Velocity #DOFs: " << fes_size0 << endl;
      cout << "Pressure #DOFs: " << fes_size1 << endl;
   }

   ODESolver *ode_solver = new BackwardEulerSolver;

   VectorFunctionCoefficient *velcoeff = nullptr;

   if (opt_.prob_type == PROB_TYPE::CYL)
   {
      velcoeff = new VectorFunctionCoefficient(dim, vel_cyl);
   }
   else if (opt_.prob_type == PROB_TYPE::TGV)
   {
      velcoeff = new VectorFunctionCoefficient(dim, vel_ex);
   }

   ParGridFunction vel_gf(fes[0]);
   vel_gf.ProjectCoefficient(*velcoeff);

   Vector X(fes[0]->GlobalVSize());
   vel_gf.GetTrueDofs(X);

   NavierStokesOperator nso(fes);
   nso.BuildOperator(dt);
   TDNavierStokesOperator td_nso(&nso);
   ode_solver->Init(td_nso);

   char vishost[] = "localhost";
   int visport = 19916;
   socketstream u_sock(vishost, visport);
   u_sock.precision(8);
   u_sock << "parallel " << num_procs << " " << myid << "\n";
   u_sock << "solution\n"
          << *pmesh << vel_gf << "window_title 'velocity'"
          << "keys Rjlc\n"
          << "pause\n"
          << flush;

   double t = 0.0;
   bool last_step = false;

   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      if (myid == 0)
      {
         cout << "\nTimestep: " << t << endl;
      }

      ode_solver->Step(X, t, dt);

      vel_gf.Distribute(X);

      if (prob_type == PROB_TYPE::TGV)
      {
         velcoeff->SetTime(t);
         double err_u = vel_gf.ComputeL2Error(*velcoeff, irs);
         if (myid == 0)
         {
            cout << "|| u_h - u_ex || = " << err_u << "\n";
         }
      }

      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock << "solution\n"
             << *pmesh << vel_gf << flush;
   }

   delete ode_solver;
   delete pmesh;
   delete vel_fec;
   delete pres_fec;
   delete vel_fes;
   delete pres_fes;

   return 0;
}
