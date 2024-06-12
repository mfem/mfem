// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Stabilized  Navier-Stokes

#include "stab_navsto.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <list>

using namespace std;
using namespace mfem;

real_t kappa_param = 1.0;
real_t pi  = (real_t)(M_PI);

using VectorFun = std::function<void(const Vector & x, Vector & a)>;
using ScalarFun = std::function<real_t(const Vector & x)>;

void sol_fun(const Vector & x, Vector &sol)
{
   sol = 0.0;
   if ((x[1] - 0.99 > 0.0) &&
       (fabs(x[0] - 0.5) < 0.49) )
   {
      sol[0] = 1.0;
   }
}

real_t kappa_fun(const Vector & x)
{
   return kappa_param;
}

void force_fun(const Vector & x, Vector &f)
{
   f = 0.0;
  // f[0] = x[1]*(1.0-x[1])*x[0]*(1.0-x[0]);
}

StabType GetStabilisationType(int stype)
{
   switch (stype)
   {
      case GALERKIN:
         mfem::out<<"Galerkin formulation"<<std::endl;
         break;
      case SUPG:
         mfem::out<<"SUPG formulation"<<std::endl;
         break;
      case GLS:
         mfem::out<<"GLS formulation"<<std::endl;
         break;
      case VMS:
         mfem::out<<"VMS formulation"<<std::endl;
         break;
      default:
         mfem::out<<"GAL"<<"\t"<<"SUPG"<<"\t"<<"GLS"<<"\t"<<"VMS"<<std::endl;
         mfem::out<<GALERKIN<<"\t"<<SUPG<<"\t"<<GLS<<"\t"<<VMS<<std::endl;
         mfem_error("Wrong formulation");
   }
   return (StabType) stype;
}

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   const char *ref_file  = "";
   int problem = 0;
   int sstype = -2;
   bool static_cond = false;
   bool visualization = false;

   real_t penalty = -1;
   int order = 1;
   int ref_levels = 0;

   bool mono = true;

   OptionsParser args(argc, argv);

   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_file, "-rf", "--ref-file",
                  "File with refinement data");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh.");
   args.AddOption(&kappa_param , "-k", "--kappa",
                  "Sets the diffusion parameters, should be positive.");
   args.AddOption(&problem, "-p", "--problem",
                  "Select the problem to solve:\n\t"
                  "  0 = convection skew-to-the mesh\n\t"
                  "  1 = manufactured solution\n");
   args.AddOption(&sstype, "-s", "--stab", " Stabilization type:\n\t"
                  " -2 = Galerkin\n\t"
                  " -1 = GLS\n\t"
                  "  0 = SUPG\n\t"
                  "  1 = VMS\n");
   args.AddOption(&mono, "-mo", "--mono", "-co",
                  "--comp",
                  "Use a monolithic integrator or a composed one.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) args.PrintOptions(cout);

   // Read the mesh from the given mesh file. We can handle triangular,
   // quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   // the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement and knot insertion of knots defined
   // in a refinement file. We choose 'ref_levels' to be the largest number
   // that gives a final mesh with no more than 50,000 elements.
   {
      // Mesh refinement as defined in refinement file
      if (mesh.NURBSext && (strlen(ref_file) != 0))
      {
         mesh.RefineNURBSFromFile(ref_file);
      }

      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
      if (myid == 0) mesh.PrintInfo();
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define a finite element space on the mesh. Here we use continuous
   // Lagrange finite elements of the specified order. If order < 1, we
   // instead use an isoparametric/isogeometric space.
   Array<FiniteElementCollection *> fecs(2);
   fecs[0] = new H1_FECollection(order, dim);
   fecs[1] = new H1_FECollection(order, dim);

   Array<ParFiniteElementSpace *> spaces(2);
   spaces[0] = new ParFiniteElementSpace(&pmesh, fecs[0], dim);//, Ordering::byVDIM);
   spaces[1] = new ParFiniteElementSpace(&pmesh, fecs[1]);

   Array<int> tdof(num_procs),udof(num_procs),pdof(num_procs);
   tdof = 0;
   tdof[myid] = spaces[0]->TrueVSize();
   MPI_Reduce(tdof.GetData(), udof.GetData(), num_procs, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

   tdof = 0;
   tdof[myid] = spaces[1]->TrueVSize();
   MPI_Reduce(tdof.GetData(), pdof.GetData(), num_procs, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

   if (myid == 0)
   {
      mfem::out << "Number of finite element unknowns:\n";
      mfem::out << "\tVelocity = "<<spaces[0]->GlobalTrueVSize() << endl;
      mfem::out << "\tPressure = "<<spaces[1]->GlobalTrueVSize() << endl;
      mfem::out << "Number of finite element unknowns per partition:\n";
      mfem::out <<  "\tVelocity = ";udof.Print(mfem::out, num_procs);
      mfem::out <<  "\tPressure = ";pdof.Print(mfem::out, num_procs);
   }

   // Mark all velocity boundary dofs as essential
   Array<Array<int> *> ess_bdr(2);
  // Array<int> ess_tdof_list;

   Array<int> ess_bdr_u(spaces[0]->GetMesh()->bdr_attributes.Max());
   Array<int> ess_bdr_p(spaces[1]->GetMesh()->bdr_attributes.Max());

   ess_bdr_p = 0;
   ess_bdr_u = 1;

   ess_bdr[0] = &ess_bdr_u;
   ess_bdr[1] = &ess_bdr_p;

   // Define the solution vector xp as a finite element grid function
   Array<int> bOffsets(3);
   bOffsets[0] = 0;
   bOffsets[1] = spaces[0]->TrueVSize();
   bOffsets[2] = spaces[1]->TrueVSize();
   bOffsets.PartialSum();

   BlockVector xp(bOffsets);

   ParGridFunction x_u(spaces[0]);
   ParGridFunction x_p(spaces[1]);

   VectorFunctionCoefficient sol(dim, sol_fun);
   x_u.ProjectCoefficient(sol);
   x_p = 0.0;

   x_u.GetTrueDofs(xp.GetBlock(0));
   x_p.GetTrueDofs(xp.GetBlock(1));

   VisItDataCollection visit_dc("navsto", &pmesh);
   visit_dc.RegisterField("u", &x_u);
   visit_dc.RegisterField("p", &x_p);
   visit_dc.SetCycle(0);
   visit_dc.Save();

   // Define the problem parameters
   FunctionCoefficient kappa(kappa_fun);
   VectorFunctionCoefficient force(dim, force_fun);

   // Define the stabilisation parameters
   VectorGridFunctionCoefficient adv(&x_u);
   ElasticInverseEstimateCoefficient invEst(spaces[0]);
   FFH92Tau tau(&adv, &kappa, &invEst, 4.0);
   FF91Delta delta(&adv, &kappa, &invEst);

   tau.print = delta.print = (myid == 0);

   // Define the block nonlinear form
   ParBlockNonlinearForm Hform(spaces);
   Hform.AddDomainIntegrator(new StabInNavStoIntegrator(kappa, force, tau, delta));
   Array<Vector *> rhs(2);
   rhs = nullptr; // Set all entries in the array
   Hform.SetEssentialBC(ess_bdr, rhs);

   // Set up the preconditioner
   JacobianPreconditioner jac_prec(bOffsets,
                                   Array<Solver *>({new HypreSmoother(),
                                                    new HypreSmoother()}));

   // Set up the Jacobian solver
   GeneralResidualMonitor j_monitor(MPI_COMM_WORLD,"\t\t\t\tFGMRES", 25);
   FGMRESSolver j_gmres(MPI_COMM_WORLD);
   j_gmres.iterative_mode = false;
   j_gmres.SetRelTol(1e-2);
   j_gmres.SetAbsTol(1e-12);
   j_gmres.SetMaxIter(300);
   j_gmres.SetPrintLevel(-1);
   j_gmres.SetMonitor(j_monitor);
   j_gmres.SetPreconditioner(jac_prec);

   // Set up the newton solver
   SystemResidualMonitor newton_monitor(MPI_COMM_WORLD,"Newton", 1, bOffsets, &visit_dc, &xp,
                                        Array<ParGridFunction *>({&x_u, &x_p}));
   NewtonSolver newton_solver(MPI_COMM_WORLD);
   newton_solver.iterative_mode = true;
   newton_solver.SetPrintLevel(-1);
   newton_solver.SetMonitor(newton_monitor);
   newton_solver.SetRelTol(1e-4);
   newton_solver.SetAbsTol(1e-8);
   newton_solver.SetMaxIter(25);
   newton_solver.SetSolver(j_gmres);
   newton_solver.SetOperator(Hform);

   // Solve the Newton system
   Vector zero;
   newton_solver.Mult(zero, xp);

   // Save data in the VisIt format
   // Define the output
   // Save data in the VisIt format
   visit_dc.SetCycle(999999);
   visit_dc.Save();

   // Free the used memory.
   for (int i = 0; i < fecs.Size(); ++i)
   {
      delete fecs[i];
   }
   for (int i = 0; i < spaces.Size(); ++i)
   {
      delete spaces[i];
   }

   return 0;
}

