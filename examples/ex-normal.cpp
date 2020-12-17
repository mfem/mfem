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
//               mpirun -np 4 ex-normal --mesh sphere_hex27.mesh --elimination
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
  ---

  - test parallel elimination with different orderings of dofs, non-contiguous,
    etc. (partly done)
  - add a scalar Eliminator? (instead of calling LAPACK on 1 by 1 matrices)
  - timing / scaling of different solvers
  - make sure curved mesh works (is this a real problem or just VisIt visualization?)
  - think about preconditioning interface; user may have good preconditioner
    for primal system that we could use in all three existing solvers?
  - improve Schur complement block in Schur solver (user-defined preconditioner, but
    different interface)
  - finer grained control of hypre rigid body modes, in elimination the numbering may
    get messed around (with square projector less of a problem)
  - hook up with user code (contact?)
*/

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/*
   Given a vector space fespace, and the array constrained_att that
   includes the boundary *attributes* that are constrained to have normal
   component zero, this returns a SparseMatrix representing the
   constraints that need to be imposed.
*/
HypreParMatrix * BuildNormalConstraints(ParFiniteElementSpace& fespace,
                                        Array<int> constrained_att)
{
   int rank, size;
   MPI_Comm_rank(fespace.GetComm(), &rank);
   MPI_Comm_size(fespace.GetComm(), &size);
   int dim = fespace.GetVDim();

   std::set<int> constrained_tdofs; // local tdofs
   for (int i = 0; i < fespace.GetNBE(); ++i)
   {
      int att = fespace.GetBdrAttribute(i);
      if (constrained_att.FindSorted(att) != -1)
      {
         Array<int> dofs;
         fespace.GetBdrElementDofs(i, dofs);
         for (auto k : dofs)
         {
            int vdof = fespace.DofToVDof(k, 0);
            int tdof = fespace.GetLocalTDofNumber(vdof);
            if (tdof >= 0) { constrained_tdofs.insert(tdof); }
         }
      }
   }

   std::map<int, int> dof_constraint;
   int n_constraints = 0;
   for (auto k : constrained_tdofs)
   {
      dof_constraint[k] = n_constraints++;
   }
   SparseMatrix * out = new SparseMatrix(n_constraints, fespace.GetTrueVSize());

   int constraint_running_total = 0;
   MPI_Scan(&n_constraints, &constraint_running_total, 1, MPI_INT, MPI_SUM, fespace.GetComm());
   int global_constraints = 0;
   if (rank == size - 1) global_constraints = constraint_running_total;
   MPI_Bcast(&global_constraints, 1, MPI_INT, size - 1, fespace.GetComm());

   Vector nor(dim);
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
            int vdof = fespace.DofToVDof(k, 0);
            int truek = fespace.GetLocalTDofNumber(vdof);
            if (truek >= 0)
            {
               int constraint = dof_constraint[truek];
               for (int d = 0; d < dim; ++d)
               {
                  int vdof = fespace.DofToVDof(k, d);
                  int truek = fespace.GetLocalTDofNumber(vdof);
                  out->Set(constraint, truek, nor[d]);
                  // d_dofs[d]->Append(vdof);
               }
            }
         }
      }
   }

   out->Finalize();

   // cols are same as for fespace; rows are... built here
   HYPRE_Int glob_num_rows = global_constraints;
   HYPRE_Int glob_num_cols = fespace.GlobalTrueVSize();
   HYPRE_Int row_starts[2] = {constraint_running_total - n_constraints, constraint_running_total};
   HYPRE_Int * col_starts = fespace.GetTrueDofOffsets();
   HypreParMatrix * h_out = new HypreParMatrix(fespace.GetComm(), glob_num_rows, glob_num_cols, row_starts,
                                               col_starts, out);
   h_out->CopyRowStarts();
   h_out->CopyColStarts();

   return h_out;
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

   Mesh * mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   {
      int ref_levels;
      if (refine == -1)
      {
         ref_levels =
            (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      }
      else
      {
         ref_levels = refine;
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   mesh->SetCurvature(order);

   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
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
   HypreParMatrix * hconstraints;
   hconstraints = BuildNormalConstraints(fespace, constraint_atts);
   hconstraints->Print("hconstraints");

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

   IterativeSolver * constrained = nullptr;
   if (penalty > 0.0)
   {
      constrained = new PenaltyConstrainedSolver(MPI_COMM_WORLD, *A.As<HypreParMatrix>(),
                                                 *hconstraints, penalty, dim);
   }
   else if (elimination)
   {
      SparseMatrix local_constraints;
      hconstraints->GetDiag(local_constraints);
      Array<int> lagrange_rowstarts(local_constraints.Height() + 1);
      for (int k = 0; k < local_constraints.Height() + 1; ++k)
      {
         lagrange_rowstarts[k] = k;
      }
      constrained = new EliminationCGSolver(*A.As<HypreParMatrix>(), local_constraints,
                                            lagrange_rowstarts, dim);
   }
   else
   {
      // reordering apparently needs to be true for this solver, needs investigation
      // (I don't think the space is reordered in this example?)
      constrained = new SchurConstrainedHypreSolver(MPI_COMM_WORLD, *A.As<HypreParMatrix>(),
                                                    *hconstraints, dim, true);
   }
   constrained->SetRelTol(reltol);
   constrained->SetAbsTol(1.e-12);
   constrained->SetMaxIter(500);
   constrained->SetPrintLevel(1);
   constrained->Mult(B, X);

   int iterations = constrained->GetNumIterations();
   if (myid == 0)
   {
      cout << "Total iterations: " << iterations << endl;
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   std::stringstream filename;
   if (penalty > 0.0)
   {
      filename << "penalty" << myid << ".vector";
      std::ofstream out(filename.str().c_str());
      out << std::setprecision(14);
      X.Print(out, 1);
   }
   else if (elimination)
   {
      filename << "elimination" << myid << ".vector";
      std::ofstream out(filename.str().c_str());
      out << std::setprecision(14);
      X.Print(out, 1);
   }
   else
   {
      filename << "schur" << myid << ".vector";
      std::ofstream out(filename.str().c_str());
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
   delete hconstraints;
   delete constrained;

   return 0;
}
