
//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "occa.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
  // 1. Initialize MPI.
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 2. Parse command-line options.
  const char *mesh_file = "../../data/fichera.mesh";
  int order = 3;
  const char *basis_type = "G"; // Gauss-Lobatto
  const char *pc = "none";
  const char *device_info = "mode: 'Serial'";
  bool occa_verbose = false;
  bool visualization = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&basis_type, "-b", "--basis-type",
                 "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
  args.AddOption(&device_info, "-d", "--device-info",
                 "Device information to run example on (default: \"mode: 'Serial'\").");
  args.AddOption(&pc, "-pc", "--preconditioner",
                 "Preconditioner: amg (HYPREBoomerAMG), or none.");
  args.AddOption(&occa_verbose,
                 "-ov", "--occa-verbose",
                 "--no-ov", "--no-occa-verbose",
                 "Print verbose information about OCCA kernel compilation.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.Parse();
  if (!args.Good())
    {
      if (!myid) {
        args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
    }
  if (!myid) {
    args.PrintOptions(cout);
  }

  // Set the OCCA device to run example in
  occa::setDevice(device_info);

  // Load cached kernels
  occa::loadKernels();
  occa::loadKernels("mfem");

  // Set as the background device
  occa::settings()["verboseCompilation"] = occa_verbose;

  enum PCType { NONE, AMG };
  PCType pc_choice;
  if (!strcmp(pc, "amg")) { pc_choice = AMG; }
  else if (!strcmp(pc, "none")) { pc_choice = NONE; }
  else
    {
      mfem_error("Invalid Preconditioner specified");
      return 3;
    }

  // See class BasisType in fem/fe_coll.hpp for available basis types
  int basis = BasisType::GetType(basis_type[0]);
  if (!myid) {
    cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;
  }

  // 3. Read the (serial) mesh from the given mesh file on all processors.  We
  //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
  //    and volume meshes with the same code.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  // 4. Refine the serial mesh on all processors to increase the resolution. In
  //    this example we do 'ref_levels' of uniform refinement. We choose
  //    'ref_levels' to be the largest number that gives a final mesh with no
  //    more than 10,000 elements.
  int par_ref_levels = 1;
  {
    int ref_levels =
      (int)floor(log(1000./mesh->GetNE())/log(2.)/dim/par_ref_levels);
    for (int l = 0; l < ref_levels; l++)
      {
        mesh->UniformRefinement();
      }
  }

  // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
  //    this mesh further in parallel to increase the resolution. Once the
  //    parallel mesh is defined, the serial mesh can be deleted.
  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;
  {
    for (int l = 0; l < par_ref_levels; l++)
      {
        pmesh->UniformRefinement();
      }
  }

  // 6. Define a parallel finite element space on the parallel mesh. Here we
  //    use continuous Lagrange finite elements of the specified order. If
  //    order < 1, we instead use an isoparametric/isogeometric space.
  FiniteElementCollection *fec;
  if (order > 0)
    {
      fec = new H1_FECollection(order, dim, basis);
    }
  else if (pmesh->GetNodes())
    {
      fec = pmesh->GetNodes()->OwnFEC();
      if (!myid) {
        cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
    }
  else
    {
      fec = new H1_FECollection(order = 1, dim, basis);
    }

  OccaFiniteElementSpace *ofespace = new OccaFiniteElementSpace(pmesh, fec);
  ParFiniteElementSpace *fespace = (ParFiniteElementSpace*) ofespace->GetFESpace();


  HYPRE_Int size = fespace->GlobalTrueVSize();
  if (!myid) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  // 7. Determine the list of true (i.e. parallel conforming) essential
  //    boundary dofs. In this example, the boundary conditions are defined
  //    by marking all the boundary attributes from the mesh as essential
  //    (Dirichlet) and converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  if (pmesh->bdr_attributes.Size())
    {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

  // 8. Set up the parallel linear form b(.) which corresponds to the
  //    right-hand side of the FEM linear system, which in this case is
  //    (1,phi_i) where phi_i are the basis functions in fespace.
  ParLinearForm lf_b(fespace);
  ConstantCoefficient one(1.0);
  lf_b.AddDomainIntegrator(new DomainLFIntegrator(one));
  lf_b.Assemble();
  OccaVector b(lf_b);

  // 9. Define the solution vector x as a parallel finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  OccaVector x(fespace->GetVSize());
  x = 0.0;

  // 10. Set up the parallel bilinear form a(.,.) on the finite element space
  //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //     domain integrator.
  if (!myid) {
    cout << "Assembling the bilinear form ..." << endl << flush;
  }
  tic_toc.Clear();
  tic_toc.Start();
  OccaBilinearForm *a = new OccaBilinearForm(ofespace);
  a->AddDomainIntegrator(new OccaDiffusionIntegrator(1.0));

  // 11. Assemble the parallel bilinear form and the corresponding linear
  //     system, applying any necessary transformations such as: parallel
  //     assembly, eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, static condensation, etc.
  a->Assemble();
  tic_toc.Stop();
  if (!myid) {
     cout << " done, " << tic_toc.RealTime() << "s." << endl;
  }

  Operator *A;
  OccaVector B, X;
  a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  if (!myid) {
    cout << "Size of linear system: " << fespace->GlobalTrueVSize() << endl;
  }

  // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
  //     preconditioner from hypre.
  OccaCGSolver *pcg = new OccaCGSolver(MPI_COMM_WORLD);
  pcg->SetRelTol(1e-6);
  pcg->SetAbsTol(0);
  pcg->SetMaxIter(500);
  pcg->SetPrintLevel(1);
  pcg->SetOperator(*A);

  if (pc_choice != NONE) {
    if (!myid) {
      cout << "Assembling the preconditioner bilinear form ..." << endl << flush;
    }
    tic_toc.Clear();
    tic_toc.Start();
    ParBilinearForm *a_pc = new ParBilinearForm(fespace);
    a_pc->AddDomainIntegrator(new DiffusionIntegrator(one));
    a_pc->Assemble();
    tic_toc.Stop();
    if (!myid) {
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
    }
    HypreParMatrix A_pc;
    a_pc->FormSystemMatrix(ess_tdof_list, A_pc);

    if (pc_choice == AMG) {
      HypreSolver *amg = new HypreBoomerAMG(A_pc);
      pcg->SetOccaPreconditioner(*amg);
    }
    else {
      mfem_error("Invalid preconditioner");
    }
  }

  tic_toc.Clear();
  tic_toc.Start();
  if (!myid) {
    cout << "Running PCG ..." << endl;
  }

  pcg->Mult(B, X);

  tic_toc.Stop();
  if (!myid) {
    cout << " done, " << tic_toc.RealTime() << "s." << endl;
  }

  // 13. Recover the parallel grid function corresponding to X. This is the
  //     local finite element solution on each processor.
  a->RecoverFEMSolution(X, b, x);

  // 14. Save the refined mesh and the solution in parallel. This output can
  //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
  ParGridFunction gf_x(fespace);
  gf_x = x;
  {
    ostringstream mesh_name, sol_name;
    mesh_name << "mesh." << setfill('0') << setw(6) << myid;
    sol_name << "sol." << setfill('0') << setw(6) << myid;

    ofstream mesh_ofs(mesh_name.str().c_str());
    mesh_ofs.precision(8);
    pmesh->Print(mesh_ofs);

    ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(8);
    // Reuse GridFunction's Save
    gf_x.Save(sol_ofs);
  }

  // 15. Send the solution by socket to a GLVis server.
  if (visualization)
    {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << gf_x << flush;
    }

  // 16. Free the used memory.
  delete pcg;
  delete a;
  delete fespace;
  if (order > 0) { delete fec; }
  delete pmesh;

  MPI_Finalize();

  return 0;
}
