#include "mfem.hpp"
#include <fstream>

using namespace std;
using namespace mfem;

#define REY 1.0

enum EX
{
   MMS,
   LDC
};

EX ex = EX::MMS;

class VectorConvectionNLFIntegrator : public NonlinearFormIntegrator
{
private:
   DenseMatrix dshape, dshapex, EF, gradEF, ELV, elmat_comp;
   Vector shape;

public:
   VectorConvectionNLFIntegrator(){};

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun, Vector &elvect)
   {
      int nd = el.GetDof();
      int dim = el.GetDim();

      shape.SetSize(nd);
      dshape.SetSize(nd, dim);
      elvect.SetSize(nd * dim);
      gradEF.SetSize(dim);

      EF.UseExternalData(elfun.GetData(), nd, dim);
      ELV.UseExternalData(elvect.GetData(), nd, dim);

      double w;
      Vector vec1(dim), vec2(dim);

      const IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
         ir = &IntRules.Get(el.GetGeomType(), order);
      }

      // Same as elvect = 0.0;
      ELV = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         el.CalcShape(ip, shape);
         el.CalcPhysDShape(trans, dshape);

         w = ip.weight * trans.Weight();

         MultAtB(EF, dshape, gradEF);
         EF.MultTranspose(shape, vec1);
         gradEF.Mult(vec1, vec2);
         vec2 *= w;

         AddMultVWt(shape, vec2, ELV);
      }
   }

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun, DenseMatrix &elmat)
   {
      int nd = el.GetDof();
      int dim = el.GetDim();

      shape.SetSize(nd);
      dshape.SetSize(nd, dim);
      dshapex.SetSize(nd, dim);
      elmat.SetSize(nd * dim);
      elmat_comp.SetSize(nd);
      gradEF.SetSize(dim);

      EF.UseExternalData(elfun.GetData(), nd, dim);

      double w;
      Vector vec1(dim), vec2(dim), vec3(nd);

      const IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
         ir = &IntRules.Get(el.GetGeomType(), order);
      }

      elmat = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         el.CalcShape(ip, shape);
         el.CalcDShape(ip, dshape);

         Mult(dshape, trans.InverseJacobian(), dshapex);

         w = ip.weight;

         MultAtB(EF, dshapex, gradEF);
         EF.MultTranspose(shape, vec1);

         trans.AdjugateJacobian().Mult(vec1, vec2);

         vec2 *= w;
         dshape.Mult(vec2, vec3);
         MultVWt(shape, vec3, elmat_comp);

         for (int i = 0; i < dim; i++)
         {
            elmat.AddMatrix(elmat_comp, i * nd, i * nd);
         }

         MultVVt(shape, elmat_comp);
         w = ip.weight * trans.Weight();
         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               elmat.AddMatrix(w * gradEF(i, j), elmat_comp, i * nd, j * nd);
            }
         }
      }
   }

   virtual ~VectorConvectionNLFIntegrator(){};
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

   u(0) = 1.0 - 0.5 * M_PI * sin(2.0 * M_PI * xi)
   - 2.0 / REY * pow(M_PI, 2.0) * cos(M_PI * xi) * sin(M_PI * yi);

   u(1) = 1.0 - 0.5 * M_PI * sin(2.0 * M_PI * yi)
   + 2.0 / REY * pow(M_PI, 2.0) * cos(M_PI * yi) * sin(M_PI * xi);
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
   ParNonlinearForm *N;
   ParBilinearForm *sform;

   HypreParMatrix *S;
   HypreParMatrix *Mp;
   HypreParMatrix *D;
   HypreParMatrix *G;
   mutable HypreParMatrix *NjacS;

   BlockOperator *jac;
   BlockOperator *lin;

   HypreSolver *invS;
   HypreDiagScale *invMp;
   BlockDiagonalPreconditioner *stokesprec;
   GMRESSolver *jac_solver;

   NewtonSolver newton_solver;

   ParGridFunction *vel_gf;
   ParGridFunction *p_gf;

public:
   NavierStokesOperator(Array<ParFiniteElementSpace *> &fes) : Operator(fes[0]->TrueVSize() + fes[1]->TrueVSize()),
                                                               pmesh_(fes[0]->GetParMesh()), fes_(fes),
                                                               ess_bdr_attr_(pmesh_->bdr_attributes.Max()),
                                                               N(nullptr),
                                                               S(nullptr), Mp(nullptr), D(nullptr), G(nullptr), NjacS(nullptr),
                                                               jac(nullptr), lin(nullptr), invS(nullptr), invMp(nullptr),
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
      N = new ParNonlinearForm(fes[0]);
      N->AddDomainIntegrator(new VectorConvectionNLFIntegrator);
      N->SetEssentialTrueDofs(ess_tdof_list_);

      ParLinearForm *fform = new ParLinearForm;
      VectorDomainLFIntegrator *fvint = new VectorDomainLFIntegrator(fcoeff);
      fvint->SetIntRule(&IntRules.Get(pmesh_->GetElementBaseGeometry(0), 4 + 3)); // order + 3
      fform->Update(fes[0], rhs.GetBlock(0), 0);
      fform->AddDomainIntegrator(fvint);
      fform->Assemble();

      sform = new ParBilinearForm(fes[0]);
      sform->AddDomainIntegrator(new VectorDiffusionIntegrator);
      // sform->SetDiagonalPolicy(Matrix::DiagonalPolicy::DIAG_ZERO);
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

      ParBilinearForm *mpform = new ParBilinearForm(fes[1]);
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

      invMp = new HypreDiagScale(*Mp);

      stokesprec = new BlockDiagonalPreconditioner(block_trueOffsets_);
      stokesprec->SetDiagonalBlock(0, invS);
      stokesprec->SetDiagonalBlock(1, invMp);

      fes[0]->GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
      fes[0]->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),
                                                     trueRhs.GetBlock(0));

      fes[1]->GetRestrictionMatrix()->Mult(x.GetBlock(1), trueX.GetBlock(1));
      fes[1]->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),
                                                     trueRhs.GetBlock(1));

      jac_solver = new GMRESSolver(MPI_COMM_WORLD);
      jac_solver->iterative_mode = false;
      jac_solver->SetAbsTol(0.0);
      jac_solver->SetRelTol(1e-4);
      jac_solver->SetMaxIter(200);
      jac_solver->SetOperator(*jac);
      jac_solver->SetPreconditioner(*stokesprec);
      jac_solver->SetPrintLevel(2);

      newton_solver.iterative_mode = false;
      newton_solver.SetSolver(*jac_solver);
      newton_solver.SetOperator(*this);
      newton_solver.SetPrintLevel(1);
      newton_solver.SetAbsTol(0.0);
      newton_solver.SetRelTol(1e-7);
      newton_solver.SetMaxIter(10);

      // trueX.Randomize();
      // this->CheckJacobian(trueX, ess_tdof_list_);
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

      // NjacS = Add(1.0, *static_cast<HypreParMatrix *>(&(N->GetGradient(u))), 1.0, *S);

      // hypre_ParCSRMatrix *NjacS_wrap;
      // hypre_ParcsrAdd(1.0, *static_cast<HypreParMatrix *>(&(N->GetGradient(u))), 1.0, *S, &NjacS_wrap);
      // NjacS = new HypreParMatrix(NjacS_wrap);

      SparseMatrix *localJac = Add(1.0, N->GetLocalGradient(u), 1.0, sform->SpMat());
      NjacS = sform->ParallelAssemble(localJac);
      NjacS->EliminateRowsCols(ess_tdof_list_);

      jac->SetBlock(0, 0, NjacS);

      return *jac;
   }

   void Solve()
   {
      newton_solver.Mult(trueRhs, trueX);
   }

   Solver *GetJacobianSolver() const
   {
      return jac_solver;
   }

   ParGridFunction *UpdateVelocityGF() const
   {
      vel_gf->Distribute(trueX.GetBlock(0));
      return vel_gf;
   }

   ParGridFunction *UpdatePressureGF() const
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
      int order_quad = max(2, 2 * order + 1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; ++i)
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
   int visport = 19916;
   socketstream u_sock(vishost, visport);
   u_sock << "parallel " << num_procs << " " << myid << "\n";
   u_sock.precision(8);
   u_sock << "solution\n"
          << *pmesh << *vel_gf << "window_title 'velocity'"
          << "keys Rjlc\n"
          << endl;

   socketstream p_sock(vishost, visport);
   p_sock << "parallel " << num_procs << " " << myid << "\n";
   p_sock.precision(8);
   p_sock << "solution\n"
          << *pmesh << *p_gf << "window_title 'pressure'"
          << "keys Rjlc\n"
          << endl;

   return 0;
}
