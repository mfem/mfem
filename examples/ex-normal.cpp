//                       MFEM Example normal-bc - Parallel Version
//

/*
  higher order seems to work fine?
  3D also looks fine
  curved mesh does *not* seem to be working
  (not even for the mesh itself, but this may be VisIt and not MFEM)
  solver obviously still needs some serious work

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

SparseMatrix * BuildConstraints(FiniteElementSpace& fespace, Array<int> constrained_att)
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
            }
         }
      }
   }

   out->Finalize();
   return out;
}

class ConstrainedSolver : public Solver
{
public:
   ConstrainedSolver(HypreParMatrix& A, SparseMatrix& B);
   ~ConstrainedSolver();

   void SetOperator(const Operator& op) { }

   void Mult(const Vector& b, Vector& x) const;

private:
   Array<int> offsets;
   BlockOperator * block_op;
   GMRESSolver gmres;
   TransposeOperator * tr_B;

   mutable Vector workb;
   mutable Vector workx;
};

ConstrainedSolver::ConstrainedSolver(HypreParMatrix& A, SparseMatrix& B)
   :
   // Solver(A.Height() + B.Height()),
   Solver(A.Height()),  // not sure conceptually what the size should be!
   offsets(3),
   gmres(A.GetComm())
{
   offsets[0] = 0;
   offsets[1] = A.Height();
   offsets[2] = A.Height() + B.Height();

   block_op = new BlockOperator(offsets);
   block_op->SetBlock(0, 0, &A);
   block_op->SetBlock(1, 0, &B);
   tr_B = new TransposeOperator(&B);
   block_op->SetBlock(0, 1, tr_B);

   gmres.SetOperator(*block_op);
   gmres.SetRelTol(1.e-6);
   gmres.SetAbsTol(1.e-12);
   gmres.SetMaxIter(500);
   gmres.SetPrintLevel(1);

   workb.SetSize(A.Height() + B.Height());
   workx.SetSize(A.Height() + B.Height());
}

ConstrainedSolver::~ConstrainedSolver()
{
   delete block_op;
   delete tr_B;
}

void ConstrainedSolver::Mult(const Vector& b, Vector& x) const
{
   workb = 0.0;
   workx = 0.0;
   for (int i = 0; i < b.Size(); ++i)
   {
      workb(i) = b(i);
      workx(i) = x(i);
   }

   gmres.Mult(workb, workx);

   for (int i = 0; i < b.Size(); ++i)
   {
      x(i) = workx(i);
   }
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
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

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
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
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
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

/*
   Array<int> circle_atts(4);
   circle_atts[0] = 5;
   circle_atts[1] = 6;
   circle_atts[2] = 7;
   circle_atts[3] = 8;
   SparseMatrix * constraint_mat = BuildConstraints(fespace, circle_atts);
*/

/*
   Array<int> icf_atts(1);
   icf_atts[0] = 4;
   SparseMatrix * constraint_mat = BuildConstraints(fespace, icf_atts);
*/

   Array<int> sphere_atts(1);
   sphere_atts[0] = 1;
   SparseMatrix * constraint_mat = BuildConstraints(fespace, sphere_atts);

   {
      std::ofstream out("constraint.sparsematrix");
      constraint_mat->Print(out, 1);
   }

   ParLinearForm b(&fespace);
   // ConstantCoefficient one(1.0);
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
   a.AddDomainIntegrator(new VectorMassIntegrator(coeff));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   ConstrainedSolver constrained(*A.As<HypreParMatrix>(), *constraint_mat);
   constrained.Mult(B, X);

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

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
   MPI_Finalize();

   return 0;
}
