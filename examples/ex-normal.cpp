//                       MFEM Example normal-bc - Parallel Version
//
// Compile with: make ex-normal
//
// Sample runs:  ex-normal
//               ex-normal --diffusion --boundary-attribute 1
//               ex-normal --mesh ../miniapps/meshing/icf.mesh
//               ex-normal --mesh sphere_hex27.mesh
//               ex-normal --elimination
//               ex-normal --penalty 1e+4
//
// Description:  Demonstrates solving a linear system subject to a linear
//               constraint, using the ConstrainedSolver object.
//
//               This particular example finds the global L2 projection of the
//               vector field (1, 0, 0) or (1, 0) onto the given mesh subject
//               to the constraint that the normal component of the vector
//               field vanishes on a particular (curved) boundary.
//
//               We recommend viewing example 2 before viewing this example.

/*
  todo:

  - timing / scaling of different solvers
  - parallel BuildNormalConstraints in this example
  - make sure curved mesh works (is this a real problem or just VisIt visualization?)
  - make elimination solver parallel
  - think about preconditioning interface; user may have good preconditioner
    for primal system that we could use in all three existing solvers?
  - improve Schur complement block in Schur solver (user-defined preconditioner, but
    different interface)
  - finer grained control of hypre rigid body modes, for example in elimination
    solver the numbering gets messed up, can we deal with that?
  - hook up to Smith or Tribol or some other contact setting
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
            // the normal returned in the next line is scaled by h, which
            // is probably what we want in this application
            CalcOrtho(Tr->Jacobian(), nor);

            // next line assumes nodes and dofs are ordered the same, which
            // seems to be true
            int k = dofs[j];
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
   mesh.SetCurvature(order);

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
   ParFiniteElementSpace fespace(&pmesh, fec, dim);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
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
      // constrain the circular boundary inside
      constraint_atts.SetSize(4);
      constraint_atts[0] = 5;
      constraint_atts[1] = 6;
      constraint_atts[2] = 7;
      constraint_atts[3] = 8;
   }
   else if (!strcmp(mesh_file, "../miniapps/meshing/icf.mesh"))
   {
      // constrain the outer curved boundary
      constraint_atts.SetSize(1);
      constraint_atts[0] = 4;
   }
   else if (!strcmp(mesh_file, "sphere_hex27.mesh"))
   {
      // constrain the (entire) boundary of the sphere
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
   // for diffusion we may want a more interesting rhs
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

   // 15. Save the refined mesh and the solution in VisIt format.
   {
      // todo: might make more sense to .SetCycle() than to append boundary_attribute to name
      std::stringstream visitname;
      visitname << "normal" << boundary_attribute;
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
