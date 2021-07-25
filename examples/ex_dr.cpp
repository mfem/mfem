// A script which uses this code which to reproduce the results in "Smoothers for Matrix-Free
// Algebraic Multigrid Preconditioning of High-Order Finite Elements" can be found in
// ../drsmoother-scripts/run_smoothers.py

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/star.mesh";
   int order = 4;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   const char *smoother_opt = "DR";
   int mesh_refinement_steps = 3;
   const char *solver_opt = "two-level";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&smoother_opt, "-s", "--smoother",
                  "Smoother to use (one of J-Jacobi, "
                  "DR-distributive relaxation, or S-Schwarz)");
   args.AddOption(&mesh_refinement_steps, "-r", "--refine-steps",
                  "Enter the number of refinement steps");
   args.AddOption(&solver_opt, "-S", "--solver",
                  "Which solver to use (either two-level"
                  ", amg_hypre, amg_amgx, smoother, direct, or boomer-amg");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   MPI_Init(&argc, &argv);

   Device device(device_config);
   device.Print();

   Mesh *ho_mesh = new Mesh(mesh_file, 1, 1);
   int dim = ho_mesh->Dimension();

   for (int l = 0; l < mesh_refinement_steps; l++)
   {
      ho_mesh->UniformRefinement();
   }

   Mesh *mesh = new Mesh(Mesh::MakeRefined(*ho_mesh, order,
                                           BasisType::GaussLobatto));

   FiniteElementCollection *fec = new H1_FECollection(1, dim);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh, fec);

   int size = fes->GetVSize();
   cout << "Number of finite element unknowns: " << size << endl;

   ConstantCoefficient one(1.0);
   BilinearForm *a = new BilinearForm(fes);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();
   a->SetDiagonalPolicy(Matrix::DIAG_ONE);

   LinearForm *b = new LinearForm(fes);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   GridFunction x(fes);
   x = 0.0;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr = 1;
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   OperatorPtr A;
   Vector B, X;

   a->FormSystemMatrix(ess_tdof_list, A);
   X.SetSize(A->Width());
   B.SetSize(A->Height());

   StopWatch chrono;
   chrono.Start();
   Solver *prec = nullptr;
   Solver *prec2 = NULL;
   string solver(solver_opt);
   HypreParMatrix *A_par = NULL;
   int bounds[2];
   if (solver == "boomer-amg")
   {
     //A_par = SimpleAMG::ToHypreParMatrix(A.As<SparseMatrix>(), MPI_COMM_WORLD,
     //                                     bounds);
   //auto hamg = new HypreBoomerAMG(*A_par);
   //hamg->SetPrintLevel(0);
   //prec = hamg;
   }
   else if (solver == "direct")
   {
#ifdef MFEM_USE_SUITESPARSE
      UMFPackSolver *umf = new UMFPackSolver();
      umf->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf->SetOperator((SparseMatrix&) *A);
      prec = umf;
#else
      CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
      cg->SetRelTol(1e-20);
      cg->SetMaxIter(2000);
      cg->SetPrintLevel(0);
      A_par = SimpleAMG::ToHypreParMatrix((SparseMatrix *) &(*A), MPI_COMM_WORLD,
                                          bounds);
      prec2 = new HypreBoomerAMG(*A_par);
      cg->SetPreconditioner(*prec2);
      cg->SetOperator(*A_par);
      prec = cg;
#endif
   }
   else
   {
      Solver *smoother;
      string smoother_str(smoother_opt);
      SparseMatrix *A_mat = A.As<SparseMatrix>();

      if (smoother_str.find("DR") != string::npos)
      {
         LORInfo *lor_info = new LORInfo(*mesh, *ho_mesh, order);
         bool composite = lor_info->Dim() == 3;
         if (smoother_str.find("L1") != string::npos)
         {
            smoother = new DRSmoother(lor_info->Cluster(), A_mat, composite, 1.0, true);
         }
         else
         {
            smoother = new DRSmoother(lor_info->Cluster(), A_mat, composite);
         }
         delete lor_info;
      }
      else if (smoother_str.find("J") != string::npos)
      {
         if (smoother_str.find("L1") != string::npos)
         {
            smoother = new DSmoother(*A_mat, 3);
         }
         else
         {
            smoother = new DSmoother(*A_mat, 0, 2.0/3);
         }
      }
      else if (smoother_str.find("S") != string::npos)
      {
         MFEM_ABORT("Schwarz smoother not yet ported");
#if 0
         LORInfo *lor_info = new LORInfo(*mesh, *ho_mesh, order);
         DisjointSets *clustering = lor_info->Cluster();
         if (smoother_str.find("L1") != string::npos)
         {
            smoother = new SchwarzSmoother(clustering, A_mat, 2);
         }
         else
         {
            smoother = new SchwarzSmoother(clustering, A_mat, 0, 2.0/3);
         }
         delete lor_info;
#endif
      }
      else
      {
         MFEM_ABORT("Smoother '" << smoother_str << "' not recognized");
      }
      if (solver == "two-level")
      {
	const auto Alor = A.As<SparseMatrix>();
	prec = new SimpleAMG(*Alor, *smoother, SimpleAMG::solverBackend::DIRECT, MPI_COMM_WORLD);
      }
      else if (solver == "amg_hypre")
      {
	const auto Alor = A.As<SparseMatrix>();
	prec = new SimpleAMG(*Alor, *smoother, SimpleAMG::solverBackend::AMG_HYPRE, MPI_COMM_WORLD);
      }
      else if (solver == "amg_amgx")
      {
	const auto Alor = A.As<SparseMatrix>();
	prec = new SimpleAMG(*Alor, *smoother, SimpleAMG::solverBackend::AMG_AMGX, MPI_COMM_WORLD,std::string("amgx.json"));
      }
      else if (solver == "smoother")
      {
         prec = smoother;
      }
      else
      {
         MFEM_ABORT("Solver type '" << solver << "' not recognized");
      }
   }
   chrono.Stop();
   cout << "Setup time = " << chrono.RealTime() << endl;
   chrono.Clear();

   OperatorPtr op;

   FiniteElementCollection *ho_fec = NULL;
   FiniteElementSpace *ho_fes = NULL;
   BilinearForm *ho_a = NULL;
   if (pa)
   {
#if 1

#else
      // Solve using the high-order operator
      ho_fec = new H1_FECollection(order, dim);
      ho_fes = new FiniteElementSpace(ho_mesh, ho_fec);

      ho_a   = new BilinearForm(ho_fes);
      ho_a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      ho_a->AddDomainIntegrator(new DiffusionIntegrator(one));
      ho_a->Assemble();

      ho_a->FormSystemMatrix(ess_tdof_list, op);
#endif
   }
   else
   {
      op = A;
   }

   B.Randomize(1234);
   X = 0.0;
   chrono.Start();
   PCG(*op, *prec, B, X, 1, 2000, 1e-16, 0.0);
   chrono.Stop();
   cout << "Solve time = " << chrono.RealTime() << endl;

   a->RecoverFEMSolution(X, *b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   delete prec;
   if (prec2 != NULL) { delete prec2; }

   delete b;
   delete ho_mesh;
   delete mesh;
   delete a;
   delete fes;
   delete fec;
   delete A_par;

   if (pa)
   {
      delete ho_fec;
      delete ho_fes;
      delete ho_a;
   }

   MPI_Finalize();

   return 0;
}
