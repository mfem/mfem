//                       MFEM Example normal-bc - Parallel Version
//

/*
  Solve a problem with the constraint that the normal component
  of the (vector) solution on the boundary is zero.

  The way MFEM dofs work, this is not quite trivial for non-grid-aligned
  boundaries.

  todo items:

  - change master/slave to primary/secondary in constrained.cpp
  - make some unit tests with tiny matrices, including in parallel
  - more meaningful parallel tests
  - improve Schur complement block solver (specifically the Schur block)
  - make sure penalty and elimination can do the right thing with lagrange multipliers
  - think about preconditioning interface; user may have good preconditioner for primal system that we could use in all three existing solvers?
  - make sure curved mesh works (is this a real problem or just VisIt visualization?)
  - make everything work in parallel
  - hook up to Smith or Tribol or some other contact setting
  - timing / scaling!

  square-disc attributes (not indices):

  1: south external
  2: east external
  3: north external
  4: west external
  5: southeast internal
  6: northeast internal
  7: northwest internal
  8: southwest internal

  icf attributes (not indices):

  1: west side
  2: south side
  3: ???
  4: outer edge (circle constraint)
  5: some internal boundaries??

  sphere_hex27.mesh

  1: external boundary
*/

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/**
   Given a vector space fespace, and the array constrained_att that
   includes the boundary *attributes* that are constrained to have normal
   component zero, this returns a SparseMatrix representing the
   constraints that need to be imposed.

   It also returns in x_dofs, y_dofs, z_dofs the partition of dofs
   invovled in constraints, which can be used in an elimination solver.

   @todo do this in parallel

   Probably the correct parallel algorithm is to build this on
   each processor, and then do a kind of RAP procedure, but without
   adding, as in ParDiscreteLinearOperator::ParallelAssemble()

   OLD: I used to think the parallel algorithm was as follows:
   do the same thing below, but only loop if you as the processor
   own the boundary element; then note the *true*dof for later
   constraining
*/
SparseMatrix * BuildNormalConstraints(FiniteElementSpace& fespace,
                                      Array<int> constrained_att,
                                      Array<int>& x_dofs,
                                      Array<int>& y_dofs,
                                      Array<int>& z_dofs)
{
   int dim = fespace.GetVDim();

   std::set<int> constrained_dofs;
   for (int i = 0; i < fespace.GetNBE(); ++i)
   {
      int att = fespace.GetBdrAttribute(i);
      if (constrained_att.FindSorted(att) != -1)
      {
         Array<int> dofs;
         fespace.GetBdrElementDofs(i, dofs);
         for (auto k : dofs)
         {
            constrained_dofs.insert(k);
         }
      }
   }

   std::map<int, int> dof_constraint;
   int n_constraints = 0;
   for (auto k : constrained_dofs)
   {
      dof_constraint[k] = n_constraints++;
   }
   SparseMatrix * out = new SparseMatrix(n_constraints, fespace.GetVSize());

   Vector nor(dim);
   Array<int>* d_dofs[3];
   d_dofs[0] = &x_dofs;
   d_dofs[1] = &y_dofs;
   d_dofs[2] = &z_dofs;
   for (int i = 0; i < fespace.GetNBE(); ++i)
   {
      int att = fespace.GetBdrAttribute(i);
      if (constrained_att.FindSorted(att) != -1)
      {
         ElementTransformation * Tr = fespace.GetBdrElementTransformation(i);
         const FiniteElement * fe = fespace.GetBE(i);
         const IntegrationRule& nodes = fe->GetNodes();

         Array<int> dofs;
         fespace.GetBdrElementDofs(i, dofs);
         MFEM_VERIFY(dofs.Size() == nodes.Size(),
                     "Something wrong in finite element space!");

         for (int j = 0; j < dofs.Size(); ++j)
         {
            Tr->SetIntPoint(&nodes[j]);
            CalcOrtho(Tr->Jacobian(), nor); // this normal is scaled by h or something

            int k = dofs[j]; // are we sure nodes and dofs are ordered the same?
            int constraint = dof_constraint[k];
            for (int d = 0; d < dim; ++d)
            {
               int vdof = fespace.DofToVDof(k, d);
               out->Set(constraint, vdof, nor[d]);
               d_dofs[d]->Append(vdof);
            }
         }
      }
   }
   for (int d = 0; d < dim; ++d)
   {
      d_dofs[d]->Sort();
      d_dofs[d]->Unique();
   }

   out->Finalize();
   return out;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Session session(argc, argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/square-disc-p3.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   int boundary_attribute = 0;
   int refine = -1;
   bool elimination = false;
   double reltol = 1.e-6;
   double penalty = 0.0;
   bool mass = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&boundary_attribute, "--boundary-attribute", "--boundary-attribute",
                  "Which attribute to apply essential conditions on.");
   args.AddOption(&refine, "--refine", "--refine",
                  "Levels of serial refinement (-1 for automatic)");
   args.AddOption(&elimination, "--elimination", "--elimination",
                  "--no-elimination", "--no-elimination",
                  "Use elimination solver for saddle point system.");
   args.AddOption(&reltol, "--reltol", "--reltol", 
                  "Relative tolerance for constrained solver.");
   args.AddOption(&penalty, "--penalty", "--penalty",
                  "Penalty parameter for penalty solver, used if > 0");
   args.AddOption(&mass, "--mass", "--mass", "--diffusion", "--diffusion",
                  "Which bilinear form, --mass or --diffusion");


   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   if (myid == 0) { device.Print(); }

   Mesh mesh(mesh_file, 1, 1);
   // mesh.EnsureNodes(); // ???
   int dim = mesh.Dimension();

   {
      int ref_levels;
      if (refine == -1)
      {
         ref_levels =
            (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      }
      else
      {
         ref_levels = refine;
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }
   mesh.SetCurvature(order); // ??? try to get curved mesh

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      // int par_ref_levels = 2;
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   ParFiniteElementSpace fespace(&pmesh, fec, dim); // vector space
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      // ess_bdr = 1;
      ess_bdr = 0;
      if (boundary_attribute > 0)
      {
         ess_bdr[boundary_attribute - 1] = 1;
      }
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   Array<int> constraint_atts;
   if (!strcmp(mesh_file, "../data/square-disc-p3.mesh"))
   {
      constraint_atts.SetSize(4);
      constraint_atts[0] = 5;
      constraint_atts[1] = 6;
      constraint_atts[2] = 7;
      constraint_atts[3] = 8;
   }
   else if (!strcmp(mesh_file, "icf.mesh"))
   {
      constraint_atts.SetSize(1);
      constraint_atts[0] = 4;
   }
   else if (!strcmp(mesh_file, "sphere_hex27.mesh"))
   {
      constraint_atts.SetSize(1);
      constraint_atts[0] = 1;
   }
   else
   {
      mfem_error("Unrecognized mesh!");
   }
   Array<int> x_dofs, y_dofs, z_dofs;
   SparseMatrix * constraint_mat = BuildNormalConstraints(fespace, constraint_atts,
                                                          x_dofs, y_dofs, z_dofs);
   std::cout << "constraint_mat is " << constraint_mat->Height() << " by "
             << constraint_mat->Width() << std::endl;
   std::cout << "x_dofs.Size() = " << x_dofs.Size()
             << ", y_dofs.Size() = " << y_dofs.Size()
             << ", z_dofs.Size() = " << z_dofs.Size() << std::endl;
   {
      std::ofstream out("constraint.sparsematrix");
      constraint_mat->Print(out, 1);
   }

   ParLinearForm b(&fespace);
   // todo: for diffusion we probably want a more interesting rhs
   Vector rhs_direction(dim);
   rhs_direction = 0.0;
   rhs_direction[0] = 1.0;
   VectorConstantCoefficient rhs_coeff(rhs_direction);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(rhs_coeff));
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   // a.AddDomainIntegrator(new DiffusionIntegrator(one));
   Vector ones(dim);
   ones = 1.0;
   VectorConstantCoefficient coeff(ones);
   if (mass)
   {
      a.AddDomainIntegrator(new VectorMassIntegrator(coeff));
   }
   else
   {
      a.AddDomainIntegrator(new VectorDiffusionIntegrator);
   }

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   ConstrainedSolver constrained(*A, *constraint_mat);
   HypreBoomerAMG prec;
   prec.SetPrintLevel(0);
   if (penalty > 0.0)
   {
      constrained.SetPenalty(penalty);
   }
   else if (elimination)
   {
      y_dofs.Append(z_dofs);
      y_dofs.Sort();
      constrained.SetElimination(y_dofs, x_dofs);
   }
   else
   {
      constrained.SetSchur(prec);
   }
   constrained.SetRelTol(reltol);
   constrained.SetAbsTol(1.e-12);
   constrained.SetMaxIter(500);
   constrained.SetPrintLevel(1);
   constrained.Mult(B, X);

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   if (penalty > 0.0)
   {
      std::ofstream out("penalty.vector");
      out << std::setprecision(14);
      X.Print(out, 1);
   }
   else if (elimination)
   {
      std::ofstream out("elimination.vector");
      out << std::setprecision(14);
      X.Print(out, 1);
   }
   else
   {
      std::ofstream out("schur.vector");
      out << std::setprecision(14);
      X.Print(out, 1);
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      // todo: might make more sense to .SetCycle() than to append boundary_attribute to name
      std::stringstream visitname;
      visitname << "normal" << boundary_attribute;
      // visitname << "icf";
      VisItDataCollection visit_dc(MPI_COMM_WORLD, visitname.str(), &pmesh);
      visit_dc.SetLevelsOfDetail(4);
      visit_dc.RegisterField("sol", &x);
      // visit_dc.SetCycle(boundary_attribute);
      visit_dc.Save();
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }
   delete constraint_mat;

   return 0;
}
