//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/square-disc-p2.vtk
//               ex1 -m ../data/square-disc-p3.mesh
//               ex1 -m ../data/square-disc-nurbs.mesh
//               ex1 -m ../data/disc-nurbs.mesh
//               ex1 -m ../data/pipe-nurbs.mesh
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
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

/*
  Missing 3D Surface
  |-----+-----+------------------------+--------------+------------|
  | CPU | GPU | Mesh                   | Element Type | Notes      |
  |-----+-----+------------------------+--------------+------------|
  |     |     | star-surf.mesh         | Square       | 3D Surface |
  |     |     | mobius-strip.mesh      | Square       | 3D Surface |
  |     |     | square-disc-surf.mesh  | Triangle     | 3D Surface |
  |-----+-----+------------------------+--------------+------------|
*/

#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "occa.hpp"

using namespace std;
using namespace mfem;

#ifndef MFEM_USE_ACROTENSOR
typedef OccaDiffusionIntegrator AcroDiffusionIntegrator;
#endif

int main(int argc, char *argv[])
{
  // 1. Parse command-line options.
  const char *mesh_file = "../../data/fichera.mesh";
  int order = 3;
  const char *basis_type = "G"; // Gauss-Lobatto
  const char *pc = "none";
  const char *device_info = "mode: 'Serial'";
  bool occa_verbose = false;
  bool use_acrotensor = false;
  bool visualization = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&basis_type, "-b", "--basis-type",
                 "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
  args.AddOption(&pc, "-pc", "--preconditioner",
                 "Preconditioner: lor - low-order-refined (matrix-free) GS, "
                 "ho - high-order (assembled) GS, none.");
  args.AddOption(&device_info, "-d", "--device-info",
                 "Device information to run example on (default: \"mode: 'Serial'\").");
  args.AddOption(&occa_verbose,
                 "-ov", "--occa-verbose",
                 "--no-ov", "--no-occa-verbose",
                 "Print verbose information about OCCA kernel compilation.");
  args.AddOption(&use_acrotensor,
                 "-ac", "--use-acro",
                 "--no-ac", "--no-acro",
                 "Use Acrotensor.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

#ifndef MFEM_USE_ACROTENSOR
  if (use_acrotensor) {
    cout << "MFEM not compiled with Acrotensor, reverting to OCCA\n";
    use_acrotensor = false;
  }
#endif

  // Set the OCCA device to run example in
  occa::setDevice(device_info);

  // Load cached kernels
  occa::loadKernels();
  occa::loadKernels("mfem");

  // Set as the background device
  occa::settings()["verboseCompilation"] = occa_verbose;

  enum PCType { NONE, LOR, HO };
  PCType pc_choice;
  if (!strcmp(pc, "ho")) { pc_choice = HO; }
  else if (!strcmp(pc, "lor")) { pc_choice = LOR; }
  else if (!strcmp(pc, "none")) { pc_choice = NONE; }
  else
    {
      mfem_error("Invalid Preconditioner specified");
      return 3;
    }

  // See class BasisType in fem/fe_coll.hpp for available basis types
  int basis = BasisType::GetType(basis_type[0]);
  cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;

  // 2. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  // 3. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final mesh with no more than 50,000
  //    elements.
  {
    int ref_levels =
      (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
    for (int l = 0; l < ref_levels; l++)
      {
        mesh->UniformRefinement();
      }
  }

  // 4. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order. If order < 1, we
  //    instead use an isoparametric/isogeometric space.
  FiniteElementCollection *fec;
  if (order > 0)
    {
      fec = new H1_FECollection(order, dim, basis);
    }
  else if (mesh->GetNodes())
    {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
    }
  else
    {
      fec = new H1_FECollection(order = 1, dim, basis);
    }

  OccaFiniteElementSpace *ofespace = new OccaFiniteElementSpace(mesh, fec);
  FiniteElementSpace *fespace = ofespace->GetFESpace();

  cout << "Number of finite element unknowns: "
       << fespace->GetTrueVSize() << endl;

  // Create the LOR mesh and finite element space. In the settings of this
  // example, we can transfer between HO and LOR with the identity operator.
  Mesh *mesh_lor = NULL;
  FiniteElementCollection *fec_lor = NULL;
  FiniteElementSpace *fespace_lor = NULL;
  if (pc_choice == LOR)
    {
      int basis_lor = basis;
      if (basis == BasisType::Positive) { basis_lor=BasisType::ClosedUniform; }
      mesh_lor = new Mesh(mesh, order, basis_lor);
      fec_lor = new H1_FECollection(1, dim);
      fespace_lor = new FiniteElementSpace(mesh_lor, fec_lor);
    }

  // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet) and
  //    converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  if (mesh->bdr_attributes.Size())
    {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

  // 6. Set up the linear form b(.) which corresponds to the right-hand side of
  //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
  //    the basis functions in the finite element fespace.
  // [MISSING] Setting up the RHS
  LinearForm lf_b(fespace);
  ConstantCoefficient one(1.0);
  lf_b.AddDomainIntegrator(new DomainLFIntegrator(one));
  lf_b.Assemble();
  OccaVector b(lf_b);

  // 7. Define the solution vector x as a finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  OccaGridFunction x(ofespace);
  x = 0.0;

  // 8. Set up the bilinear form a(.,.) on the finite element space that will
  //    hold the matrix corresponding to the Laplacian operator -Delta.
  //    Optionally setup a form to be assembled for preconditioning (a_pc).
  cout << "Assembling the bilinear form ..." << flush;
  tic_toc.Clear();
  tic_toc.Start();
  OccaBilinearForm *a = new OccaBilinearForm(ofespace);

  if (use_acrotensor) {
    a->AddDomainIntegrator(new AcroDiffusionIntegrator(1.0));
  } else {
    a->AddDomainIntegrator(new OccaDiffusionIntegrator(1.0));
  }

  BilinearForm *a_pc = NULL;
  if (pc_choice == LOR) { a_pc = new BilinearForm(fespace_lor); }
  if (pc_choice == HO)  { a_pc = new BilinearForm(fespace); }

  // 9. Assemble the bilinear form and the corresponding linear system,
  //    applying any necessary transformations such as: eliminating boundary
  //    conditions, applying conforming constraints for non-conforming AMR,
  //    static condensation, etc.
  a->Assemble();
  tic_toc.Stop();
  cout << " done, " << tic_toc.RealTime() << "s." << endl;

  Operator *A;
  OccaVector B, X;

  a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  cout << "Size of linear system: " << A->Height() << endl;

  // Setup the matrix used for preconditioning
  cout << "Assembling the preconditioning matrix ..." << flush;
  tic_toc.Clear();
  tic_toc.Start();

  SparseMatrix A_pc;
  if (pc_choice != NONE) {
    a_pc->AddDomainIntegrator(new DiffusionIntegrator(one));
    a_pc->UsePrecomputedSparsity();
    a_pc->Assemble();
    a_pc->FormSystemMatrix(ess_tdof_list, A_pc);
  }

  tic_toc.Stop();
  cout << " done, " << tic_toc.RealTime() << "s." << endl;

  cout << "Running " << (pc_choice == NONE ? "CG" : "PCG")
       << " ...\n" << flush;
  tic_toc.Clear();
  tic_toc.Start();
  // Solve with CG or PCG, depending if the matrix A_pc is available
  if (pc_choice != NONE) {
    GSSmoother M(A_pc);
    PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);
  } else {
    CG(*A, B, X, 1, 500, 1e-12, 0.0);
  }
  occa::finish();
  tic_toc.Stop();
  cout << " done, " << tic_toc.RealTime() << "s." << endl;

  // 11. Recover the solution as a finite element grid function.
  a->RecoverFEMSolution(X, b, x);

  // 12. Save the refined mesh and the solution. This output can be viewed later
  //     using GLVis: "glvis -m refined.mesh -g sol.gf".
  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh->Print(mesh_ofs);
  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  // Reuse GridFunction's Save
  GridFunction gf_x(fespace);
  gf_x = x;
  gf_x.Save(sol_ofs);

  // 13. Send the solution by socket to a GLVis server.
  if (visualization)
    {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << gf_x << flush;
    }

  // 14. Free the used memory.
  delete a;
  delete fespace;
  if (order > 0) { delete fec; }
  delete mesh;

  return 0;
}
