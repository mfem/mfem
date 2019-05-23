#include "mfem.hpp"
#include <fstream>

using namespace std;
using namespace mfem;

enum EX
{
   MMS,
   LDC
};

EX ex = EX::LDC;

class VectorConvectionIntegrator : public BilinearFormIntegrator
{
private:
   DenseMatrix dshape, adjJ, Q_nodal, elmat_comp;
   Vector shape, vec2, DQadjJ;
   VectorCoefficient &Q;
   double alpha;

public:
   VectorConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      DenseMatrix &elmat)
   {
      int nd = el.GetDof();
      int dim = el.GetDim();

      elmat.SetSize(nd * dim);
      elmat_comp.SetSize(nd);
      dshape.SetSize(nd, dim);
      adjJ.SetSize(dim);
      shape.SetSize(nd);
      vec2.SetSize(dim);
      DQadjJ.SetSize(nd);

      Vector vec1;

      const IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         int order = trans.OrderGrad(&el) + trans.Order() + 2 * el.GetOrder();
         ir = &IntRules.Get(el.GetGeomType(), order);
      }

      Q.Eval(Q_nodal, trans, *ir);

      elmat_comp = 0.0;
      elmat = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         el.CalcDShape(ip, dshape);
         el.CalcShape(ip, shape);

         trans.SetIntPoint(&ip);
         CalcAdjugate(trans.Jacobian(), adjJ);
         Q_nodal.GetColumnReference(i, vec1);

         vec1 *= alpha * ip.weight;

         adjJ.Mult(vec1, vec2);
         dshape.Mult(vec2, DQadjJ);

         AddMultVWt(shape, DQadjJ, elmat_comp);
      }

      for (int i = 0; i < dim; i++)
      {
         elmat.AddMatrix(elmat_comp, i * nd, i * nd);
      }
   }
};

void vel_ex(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = -cos(M_PI * xi) * sin(M_PI * yi);
   u(1) = sin(M_PI * xi) * cos(M_PI * yi);
}

void vel_ldc(const Vector &x, Vector &u)
{
   double yi = x(1);

   if (yi > 1.0 - 1e-6)
   {
      u(0) = 1.0;
   }
   else
   {
      u(0) = 0.0;
   }
   u(1) = 0.0;
}

double p_ex(const Vector &x)
{
   double xi = x(0);
   double yi = x(1);

   return xi + yi - 1.0;
}

void ffun(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = 1.0 - 2.0 * M_PI * M_PI * cos(M_PI * xi) * sin(M_PI * yi);
   u(1) = 1.0 + 2.0 * M_PI * M_PI * cos(M_PI * yi) * sin(M_PI * xi);
}

class NavierStokesOperator : public Operator
{
private:
   ParMesh *pmesh_;
   Array<ParFiniteElementSpace *> fes_;
   Array<int> ess_bdr_attr_;
   Array<int> ess_tdof_list_;

   Array<int> block_offsets_;
   Array<int> block_trueOffsets_;
   BlockVector x, rhs;
   BlockVector trueX, trueRhs;

   VectorGridFunctionCoefficient vel_fc;
   ParBilinearForm *N;

   HypreParMatrix *S;
   HypreParMatrix *Mp;
   HypreParMatrix *D;
   HypreParMatrix *G;

   BlockOperator *stokesop;

   HypreSolver *invS;
   HypreDiagScale *invMp;
   BlockDiagonalPreconditioner *stokesprec;
   MINRESSolver *jac_solver;

   NewtonSolver newton_solver;

   ParGridFunction *vel_gf;
   ParGridFunction *p_gf;

public:
   NavierStokesOperator(Array<ParFiniteElementSpace *> &fes) :
      Operator(fes[0]->TrueVSize() + fes[1]->TrueVSize()),
      pmesh_(fes[0]->GetParMesh()), fes_(fes),
      ess_bdr_attr_(pmesh_->bdr_attributes.Max()),
      N(nullptr),
      S(nullptr), Mp(nullptr), D(nullptr), G(nullptr),
      stokesop(nullptr), invS(nullptr), invMp(nullptr),
      stokesprec(nullptr), jac_solver(nullptr),
      newton_solver(pmesh_->GetComm()), vel_gf(nullptr)
   {
      // Mark all attributes as essential for now
      ess_bdr_attr_ = 1;
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

      x.Update(block_offsets_);
      rhs.Update(block_offsets_);

      trueX.Update(block_trueOffsets_);
      trueRhs.Update(block_trueOffsets_);

      x = 0.0;
      rhs = 0.0;
      trueX = 0.0;
      trueRhs = 0.0;

      const int dim = pmesh_->Dimension();

      VectorFunctionCoefficient fcoeff(dim, ffun);

      vel_gf = new ParGridFunction;
      vel_gf->MakeRef(fes[0], x.GetBlock(0));

      if (ex == EX::MMS)
      {
         VectorFunctionCoefficient uexcoeff(dim, vel_ex);
         vel_gf->ProjectBdrCoefficient(uexcoeff, ess_bdr_attr_);
      }
      else if (ex == EX::LDC)
      {
         VectorFunctionCoefficient ldccoeff(dim, vel_ldc);
         vel_gf->ProjectBdrCoefficient(ldccoeff, ess_bdr_attr_);
      }

      p_gf = new ParGridFunction(fes[1]);

      // Convective nonlinear term
      // N(u,u,v) = (u \cdot \nabla u, v)
      vel_fc.SetGridFunction(vel_gf);
      N = new ParBilinearForm(fes[0]);
      N->AddDomainIntegrator(new VectorConvectionIntegrator(vel_fc));

      ParLinearForm *fform = new ParLinearForm;
      fform->Update(fes[0], rhs.GetBlock(0), 0);
      fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
      fform->Assemble();

      ParBilinearForm *sform = new ParBilinearForm(fes[0]);
      sform->AddDomainIntegrator(new VectorDiffusionIntegrator);
      sform->Assemble();
      sform->EliminateEssentialBC(ess_bdr_attr_, x.GetBlock(0), rhs.GetBlock(0));
      sform->Finalize();
      S = sform->ParallelAssemble();

      ParMixedBilinearForm *dform = new ParMixedBilinearForm(fes[0], fes[1]);
      dform->AddDomainIntegrator(new VectorDivergenceIntegrator);
      dform->Assemble();
      dform->EliminateTrialDofs(ess_bdr_attr_, x.GetBlock(0), rhs.GetBlock(1));
      dform->Finalize();
      D = dform->ParallelAssemble();

      G = D->Transpose();
      (*G) *= -1.0;

      // Flip signs to make system symmetric
      (*D) *= -1.0;
      rhs.GetBlock(1) *= -1.0;

      ParBilinearForm *mpform = new ParBilinearForm(fes[1]);
      mpform->AddDomainIntegrator(new MassIntegrator);
      mpform->Assemble();
      mpform->Finalize();
      Mp = mpform->ParallelAssemble();

      stokesop = new BlockOperator(block_trueOffsets_);
      stokesop->SetBlock(0, 0, S);
      stokesop->SetBlock(0, 1, G);
      stokesop->SetBlock(1, 0, D);

      invS = new HypreBoomerAMG(*S);
      static_cast<HypreBoomerAMG *>(invS)->SetPrintLevel(0);
      invS->iterative_mode = false;

      invMp = new HypreDiagScale(*Mp);

      stokesprec = new BlockDiagonalPreconditioner(block_trueOffsets_);
      stokesprec->SetDiagonalBlock(0, invS);
      stokesprec->SetDiagonalBlock(1, invMp);

      // Idea:
      // Implement "GetTrueDofs" in ParFiniteElementSpace s.t.
      // one can call vel_fes->GetTrueDofs(x.GetBlock(0), trueX.GetBlock(0));

      fes[0]->GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
      fes[0]->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),
                                                     trueRhs.GetBlock(0));

      fes[1]->GetRestrictionMatrix()->Mult(x.GetBlock(1), trueX.GetBlock(1));
      fes[1]->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),
                                                     trueRhs.GetBlock(1));

      jac_solver = new MINRESSolver(MPI_COMM_WORLD);
      jac_solver->iterative_mode = false;
      jac_solver->SetAbsTol(0.0);
      jac_solver->SetRelTol(1e-8);
      jac_solver->SetMaxIter(200);
      jac_solver->SetOperator(*stokesop);
      jac_solver->SetPreconditioner(*stokesprec);
      jac_solver->SetPrintLevel(2);

      newton_solver.iterative_mode = false;
      newton_solver.SetSolver(*jac_solver);
      newton_solver.SetOperator(*this);
      newton_solver.SetPrintLevel(1);
      newton_solver.SetAbsTol(0.0);
      newton_solver.SetRelTol(1e-7);
      newton_solver.SetMaxIter(10);

      newton_solver.CheckJacobian(trueX, ess_tdof_list_);
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      Vector tmp(block_trueOffsets_[1]);
      Vector vel_in(x.GetData(), block_trueOffsets_[1]);
      Vector vel_out(y.GetData(), block_trueOffsets_[1]);

      // Update velocity ParGridFunction to reflect changes
      // in the dependent coefficient for the form N
      this->UpdateVelocityGF();
      N->Assemble();

      stokesop->Mult(x, y);

      N->TrueAddMult(vel_in, tmp);

      // vel_out += tmp;
   }

   virtual Operator &GetGradient(const Vector &) const
   {
      return *stokesop;
   }

   void Solve()
   {
      newton_solver.Mult(trueRhs, trueX);
   }

   Solver* GetJacobianSolver() const
   {
      return jac_solver;
   }

   ParGridFunction* UpdateVelocityGF() const
   {
      vel_gf->Distribute(trueX.GetBlock(0));
      return vel_gf;
   }

   ParGridFunction* UpdatePressureGF() const
   {
      p_gf->Distribute(trueX.GetBlock(1));
      return p_gf;
   }

   virtual ~NavierStokesOperator()
   {

   }
};

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   int print_level = 2;
   int serial_ref_levels = 0;
   int order = 2;
   double tol = 1e-8;
   const char *mesh_file = "../data/inline-single-quad.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Solver relative tolerance");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Solver print level");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
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

   int vel_order = order;
   int pres_order = order - 1;

   Mesh *mesh = new Mesh(mesh_file);
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

   NavierStokesOperator nso(fes);

   nso.Solve();

   ParGridFunction *vel_gf = nso.UpdateVelocityGF();
   ParGridFunction *p_gf = nso.UpdatePressureGF();

   if (ex == EX::MMS)
   {
      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      VectorFunctionCoefficient uexcoeff(dim, vel_ex);
      FunctionCoefficient pexcoeff(p_ex);

      double err_u = vel_gf->ComputeL2Error(uexcoeff, irs);
      double norm_u = ComputeGlobalLpNorm(2, uexcoeff, *pmesh, irs);

      double err_p = p_gf->ComputeL2Error(pexcoeff, irs);
      double norm_p = ComputeGlobalLpNorm(2, pexcoeff, *pmesh, irs);

      if (myid == 0)
      {
         cout << "|| u_h - u_ex || = " << err_u << "\n";
         cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
         cout << "|| p_h - p_ex || = " << err_p << "\n";
         cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
      }
   }

   char vishost[] = "localhost";
   int  visport = 19916;
   socketstream u_sock(vishost, visport);
   u_sock << "parallel " << num_procs << " " << myid << "\n";
   u_sock.precision(8);
   u_sock << "solution\n" << *pmesh << *vel_gf << "window_title 'velocity'" <<
          "keys Rjlc\n"<< endl;

   socketstream p_sock(vishost, visport);
   p_sock << "parallel " << num_procs << " " << myid << "\n";
   p_sock.precision(8);
   p_sock << "solution\n" << *pmesh << *p_gf << "window_title 'pressure'" <<
          "keys Rjlc\n"<< endl;

   return 0;
}
