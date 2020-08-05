//                       MFEM Example 71 - Parallel Version
//
// Compile with: make ex71p
//
// Sample runs:
//   mpirun -np 2 ex71p -m ../data/beam-quad.mesh -pp 3.8
//   mpirun -np 2 ex71p -m ../data/beam-tri.mesh  -pp 7.2
//   mpirun -np 2 ex71p -m ../data/beam-hex.mesh
//   mpirun -np 2 ex71p -m ../data/beam-tet.mesh
//   mpirun -np 2 ex71p -m ../data/beam-wedge.mesh
//
// Description:  This examples solves a quasi-static nonlinear
//               pLaplacian problem with zero Dirichlet boundary
//               conditions applied on all defined boundaries
//
//           The example demonstrates the use of nonlinear operators
//           combined with automatic differentiation (AD). The definitions
//           of the integrators are written in the ex71.hpp.
//           Selecting integrator=0 will use the handcoded integrator.
//           Selecting integrator=1 will utilize the AD integrator.
//           The AD integrator can be modifief to use ADQFunctionTJ.
//
//           qint (the integrand) is a function which is evaluated
//           at every integration point. For implementations utilizing
//           ADQFunctionTJ, the user has to implement the function and the
//           residual evaluation. The Jacobian of the residual is evaluated
//           using AD
//
//           For implementations utilizing ADQFunctionTH, the user has
//           to implement only the function evaluation (as
//           a template) and the first derivative (the residual) and the
//           second derivatives (the Hessian) are evaluated using AD.
//
//           We recommend viewing examples 1 and 19, before viewing this
//           example.

#include "ex71.hpp"

int main(int argc, char *argv[])
{
   // 1. Initialize MPI
   int num_procs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // 2. Parse command-line options
   const char *mesh_file = "../data/beam-tet.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 2;
   bool visualization = true;
   double newton_rel_tol = 1e-4;
   double newton_abs_tol = 1e-6;
   int newton_iter = 500;
   int print_level = 0;
   double pp = 2.0;
   int integrator=1; //use AD
   mfem::StopWatch* timer=new mfem::StopWatch();

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&newton_rel_tol, "-rel", "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol, "-abs", "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter, "-it", "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption(&pp, "-pp", "--power-parameter",
                  "Power parameter (>=2.0) for the p-Laplacian.");
   args.AddOption((&print_level),"-prt","--print-level",
                  "Print level.");
   args.AddOption(&integrator, "-int","--integrator",
                  "Integrator 0: standard; 1: AD");

   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh *pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define the power parameter for the p-Laplacian and all other
   //    coefficients
   mfem::ConstantCoefficient c_pp(pp);
   mfem::ConstantCoefficient load(1.000000000);
   mfem::ConstantCoefficient c_ee(0.000000001);

   // 7. Define the finite element spaces for the solution
   mfem::H1_FECollection fec(order,dim);
   mfem::ParFiniteElementSpace fespace(pmesh,&fec,1,mfem::Ordering::byVDIM);
   HYPRE_Int glob_size=fespace.GlobalTrueVSize();
   if (myrank == 0)
   {
      std::cout << "Number of finite element unknowns: " << glob_size << std::endl;
   }

   // 8. Define the Dirichlet conditions
   mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   // 9. Define the nonlinear form
   mfem::ParNonlinearForm* nf=new mfem::ParNonlinearForm(&fespace);

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   mfem::ParGridFunction x(&fespace);
   x = 0.0;
   mfem::HypreParVector* tv=x.GetTrueDofs();
   mfem::HypreParVector* sv=x.GetTrueDofs();

   // 11. Define ParaView DataCollection
   mfem::ParaViewDataCollection *dacol=new
   mfem::ParaViewDataCollection("Example71",
                                pmesh);
   dacol->SetLevelsOfDetail(order);
   dacol->RegisterField("sol",&x);


   // 11. Set domain integrators - start with linear diffusion
   {
      // the default power coefficient is 2.0
      mfem::ConstantCoefficient lpp(2.0);
      if (integrator==0)
      {
         nf->AddDomainIntegrator(new mfem::pLaplace(lpp,c_ee,load));
      }
      else if (integrator==1)
      {
         nf->AddDomainIntegrator(new mfem::pLaplaceAD(lpp,c_ee,load));
      }
      nf->SetEssentialBC(ess_bdr);
      // compute the energy
      double energy=nf->GetEnergy(*tv);
      if (myrank==0)
      {
         std::cout<<"[2] The total energy of the system is E="<<energy<<std::endl;
      }
      // time the assembly
      timer->Clear();
      timer->Start();
      nf->GetGradient(*sv);
      timer->Stop();
      if (myrank==0)
      {
         std::cout<<"[2] The assembly time is: "<<timer->RealTime()<<std::endl;
      }
      mfem::Solver *prec=new mfem::HypreBoomerAMG();
      mfem::GMRESSolver *j_gmres = new mfem::GMRESSolver(MPI_COMM_WORLD);
      j_gmres->SetRelTol(1e-7);
      j_gmres->SetAbsTol(1e-15);
      j_gmres->SetMaxIter(300);
      j_gmres->SetPrintLevel(print_level);
      j_gmres->SetPreconditioner(*prec);

      mfem::NewtonSolver* ns;
      ns=new mfem::NewtonSolver(MPI_COMM_WORLD);
      ns->iterative_mode = true;
      ns->SetSolver(*j_gmres);
      ns->SetOperator(*nf);
      ns->SetPrintLevel(print_level);
      ns->SetRelTol(1e-6);
      ns->SetAbsTol(1e-12);
      ns->SetMaxIter(3);
      //solve the problem
      timer->Clear();
      timer->Start();
      ns->Mult(*tv, *sv);
      timer->Stop();
      if (myrank==0)
      {
         std::cout<<"Time for the NewtonSolver: "<<timer->RealTime()<<std::endl;
      }

      energy=nf->GetEnergy(*sv);
      if (myrank==0)
      {
         std::cout<<"[pp=2] The total energy of the system is E="<<energy<<std::endl;
      }

      delete ns;
      delete j_gmres;
      delete prec;

      x.SetFromTrueDofs(*sv);
      dacol->SetTime(2.0);
      dacol->SetCycle(2);
      dacol->Save();
   }

   // 12. Continue with powers higher than 2
   for (int i=3; i<pp; i++)
   {
      delete nf;
      nf=new mfem::ParNonlinearForm(&fespace);
      mfem::ConstantCoefficient lpp((double)i);
      if (integrator==0)
      {
         nf->AddDomainIntegrator(new mfem::pLaplace(lpp,c_ee,load));
      }
      else if (integrator==1)
      {
         nf->AddDomainIntegrator(new mfem::pLaplaceAD(lpp,c_ee,load));
      }
      nf->SetEssentialBC(ess_bdr);
      // compute the energy
      double energy=nf->GetEnergy(*sv);
      if (myrank==0)
      {
         std::cout<<"[pp="<<i<<"] The total energy of the system is E="<<energy<<std::endl;
      }
      // time the assembly
      timer->Clear();
      timer->Start();
      nf->GetGradient(*sv);
      timer->Stop();
      if (myrank==0)
      {
         std::cout<<"[pp="<<i<<"] The assembly time is: "<<timer->RealTime()<<std::endl;
      }
      mfem::Solver *prec=new mfem::HypreBoomerAMG();
      mfem::GMRESSolver *j_gmres = new mfem::GMRESSolver(MPI_COMM_WORLD);
      j_gmres->SetRelTol(1e-7);
      j_gmres->SetAbsTol(1e-15);
      j_gmres->SetMaxIter(300);
      j_gmres->SetPrintLevel(print_level);
      j_gmres->SetPreconditioner(*prec);

      mfem::NewtonSolver* ns;
      ns=new mfem::NewtonSolver(MPI_COMM_WORLD);
      ns->iterative_mode = true;
      ns->SetSolver(*j_gmres);
      ns->SetOperator(*nf);
      ns->SetPrintLevel(print_level);
      ns->SetRelTol(1e-6);
      ns->SetAbsTol(1e-12);
      ns->SetMaxIter(3);
      //solve the problem
      timer->Clear();
      timer->Start();
      ns->Mult(*tv, *sv);
      timer->Stop();
      if (myrank==0)
      {
         std::cout<<"Time for the NewtonSolver: "<<timer->RealTime()<<std::endl;
      }

      energy=nf->GetEnergy(*sv);
      if (myrank==0)
      {
         std::cout<<"[pp="<<i<<"] The total energy of the system is E="<<energy<<std::endl;
      }

      delete ns;
      delete j_gmres;
      delete prec;

      x.SetFromTrueDofs(*sv);
      dacol->SetTime(i);
      dacol->SetCycle(i);
      dacol->Save();
   }

   // 13. Continue with the final power
   if ( std::abs(pp-2.0) > std::numeric_limits<double>::epsilon())
   {
      delete nf;
      nf=new mfem::ParNonlinearForm(&fespace);
      if (integrator==0)
      {
         nf->AddDomainIntegrator(new mfem::pLaplace(c_pp,c_ee,load));
      }
      else if (integrator==1)
      {
         nf->AddDomainIntegrator(new mfem::pLaplaceAD(c_pp,c_ee,load));
      }
      nf->SetEssentialBC(ess_bdr);
      // compute the energy
      double energy=nf->GetEnergy(*sv);
      if (myrank==0)
      {
         std::cout<<"[pp="<<pp<<"] The total energy of the system is E="<<energy<<std::endl;
      }
      // time the assembly
      timer->Clear();
      timer->Start();
      nf->GetGradient(*sv);
      timer->Stop();
      if (myrank==0)
      {
         std::cout<<"[pp="<<pp<<"] The assembly time is: "<<timer->RealTime()<<std::endl;
      }
      mfem::Solver *prec=new mfem::HypreBoomerAMG();
      mfem::GMRESSolver *j_gmres = new mfem::GMRESSolver(MPI_COMM_WORLD);
      j_gmres->SetRelTol(1e-8);
      j_gmres->SetAbsTol(1e-15);
      j_gmres->SetMaxIter(300);
      j_gmres->SetPrintLevel(print_level);
      j_gmres->SetPreconditioner(*prec);

      mfem::NewtonSolver* ns;
      ns=new mfem::NewtonSolver(MPI_COMM_WORLD);
      ns->iterative_mode = true;
      ns->SetSolver(*j_gmres);
      ns->SetOperator(*nf);
      ns->SetPrintLevel(print_level);
      ns->SetRelTol(1e-6);
      ns->SetAbsTol(1e-12);
      ns->SetMaxIter(3);
      //solve the problem
      timer->Clear();
      timer->Start();
      ns->Mult(*tv, *sv);
      timer->Stop();
      if (myrank==0)
      {
         std::cout<<"Time for the NewtonSolver: "<<timer->RealTime()<<std::endl;
      }

      energy=nf->GetEnergy(*sv);
      if (myrank==0)
      {
         std::cout<<"[pp="<<pp<<"] The total energy of the system is E="<<energy<<std::endl;
      }

      delete ns;
      delete j_gmres;
      delete prec;

      x.SetFromTrueDofs(*sv);
      dacol->SetTime(pp);
      if (pp<2.0)
      {
         dacol->SetCycle(std::floor(pp));
      }
      else
      {
         dacol->SetCycle(std::ceil(pp));
      }
      dacol->Save();
   }



   // 19. Free the used memory
   delete dacol;
   delete sv;
   delete tv;
   delete nf;
   delete pmesh;
   delete timer;

   MPI_Finalize();

   return 0;
}


