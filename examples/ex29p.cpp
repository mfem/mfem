//                       MFEM Example 29 - Parallel Version
//
// Compile with: make ex29p
//
// Sample runs:  mpirun -np 4 ex29p
//               mpirun -np 4 ex29p -sc
//               mpirun -np 4 ex29p -mt 3 -o 3 -sc
//               mpirun -np 4 ex29p -mt 3 -rs 1 -o 4 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               finite element discretization of a PDE on a 2 dimensional
//               surface embedded in a 3 dimensional domain. In this case we
//               solve the Laplace problem -Div(sigma Grad u) = 1, with
//               homogeneous Dirichlet boundary conditions, where sigma is an
//               anisotropic diffusion constant defined as a 3x3 matrix
//               coefficient.
//
//               This example demonstrates the use of finite element integrators
//               on 2D domains with 3D coefficients.
//
//               We recommend viewing examples 1 and 7 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

Mesh * GetMesh(int type);

void trans(const Vector &x, Vector &r);

void sigmaFunc(const Vector &x, DenseMatrix &s);

real_t uExact(const Vector &x)
{
   return (0.25 * (2.0 + x[0]) - x[2]) * (x[2] + 0.25 * (2.0 + x[0]));
}

void duExact(const Vector &x, Vector &du)
{
   du.SetSize(3);
   du[0] = 0.125 * (2.0 + x[0]) * x[1] * x[1];
   du[1] = -0.125 * (2.0 + x[0]) * x[0] * x[1];
   du[2] = -2.0 * x[2];
}

void fluxExact(const Vector &x, Vector &f)
{
   f.SetSize(3);

   DenseMatrix s(3);
   sigmaFunc(x, s);

   Vector du(3);
   duExact(x, du);

   s.Mult(du, f);
   f *= -1.0;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   int order = 3;
   int mesh_type = 4; // Default to Quadrilateral mesh
   int mesh_order = 3;
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   bool static_cond = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_type, "-mt", "--mesh-type",
                  "Mesh type: 3 - Triangular, 4 - Quadrilateral.");
   args.AddOption(&mesh_order, "-mo", "--mesh-order",
                  "Geometric order of the curved mesh.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   // 3. Construct a quadrilateral or triangular mesh with the topology of a
   //    cylindrical surface.
   Mesh *mesh = GetMesh(mesh_type);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ser_ref_levels' of uniform refinement.
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // 6. Transform the mesh so that it has a more interesting geometry.
   pmesh.SetCurvature(mesh_order);
   pmesh.Transform(trans);

   // 7. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   HYPRE_Int total_num_dofs = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << total_num_dofs << endl;
   }

   // 8. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 10. Define the solution vector x as a finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the
   //     Diffusion domain integrator.
   ParBilinearForm a(&fespace);
   MatrixFunctionCoefficient sigma(3, sigmaFunc);
   BilinearFormIntegrator *integ = new DiffusionIntegrator(sigma);
   a.AddDomainIntegrator(integ);

   // 12. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   if (myid == 0)
   {
      cout << "Size of linear system: "
           << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;
   }

   // 13. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
   //     preconditioner from hypre.
   HypreBoomerAMG *amg = new HypreBoomerAMG;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(*amg);
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete amg;

   // 14. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 15. Compute error in the solution and its flux
   FunctionCoefficient uCoef(uExact);
   real_t error = x.ComputeL2Error(uCoef);

   if (myid == 0) { cout << "|u - u_h|_2 = " << error << endl; }

   ParFiniteElementSpace flux_fespace(&pmesh, &fec, 3);
   ParGridFunction flux(&flux_fespace);
   x.ComputeFlux(*integ, flux); flux *= -1.0;

   VectorFunctionCoefficient fluxCoef(3, fluxExact);
   real_t flux_err = flux.ComputeL2Error(fluxCoef);

   if (myid == 0) { cout << "|f - f_h|_2 = " << flux_err << endl; }

   // 16. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name, flux_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;
      flux_name << "flux." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);

      ofstream flux_ofs(flux_name.str().c_str());
      flux_ofs.precision(8);
      flux.Save(flux_ofs);
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x
               << "window_title 'Solution'\n" << flush;

      socketstream flux_sock(vishost, visport);
      flux_sock << "parallel " << num_procs << " " << myid << "\n";
      flux_sock.precision(8);
      flux_sock << "solution\n" << pmesh << flux
                << "keys vvv\n"
                << "window_geometry 402 0 400 350\n"
                << "window_title 'Flux'\n"  << flush;
   }

   return 0;
}

// Defines a mesh consisting of four flat rectangular surfaces connected to form
// a loop.
Mesh * GetMesh(int type)
{
   Mesh * mesh = NULL;

   if (type == 3)
   {
      mesh = new Mesh(2, 12, 16, 8, 3);

      mesh->AddVertex(-1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0,  1.0, 1.0);
      mesh->AddVertex(-1.0,  1.0, 1.0);
      mesh->AddVertex( 0.0, -1.0, 0.5);
      mesh->AddVertex( 1.0,  0.0, 0.5);
      mesh->AddVertex( 0.0,  1.0, 0.5);
      mesh->AddVertex(-1.0,  0.0, 0.5);

      mesh->AddTriangle(0, 1, 8);
      mesh->AddTriangle(1, 5, 8);
      mesh->AddTriangle(5, 4, 8);
      mesh->AddTriangle(4, 0, 8);
      mesh->AddTriangle(1, 2, 9);
      mesh->AddTriangle(2, 6, 9);
      mesh->AddTriangle(6, 5, 9);
      mesh->AddTriangle(5, 1, 9);
      mesh->AddTriangle(2, 3, 10);
      mesh->AddTriangle(3, 7, 10);
      mesh->AddTriangle(7, 6, 10);
      mesh->AddTriangle(6, 2, 10);
      mesh->AddTriangle(3, 0, 11);
      mesh->AddTriangle(0, 4, 11);
      mesh->AddTriangle(4, 7, 11);
      mesh->AddTriangle(7, 3, 11);

      mesh->AddBdrSegment(0, 1, 1);
      mesh->AddBdrSegment(1, 2, 1);
      mesh->AddBdrSegment(2, 3, 1);
      mesh->AddBdrSegment(3, 0, 1);
      mesh->AddBdrSegment(5, 4, 2);
      mesh->AddBdrSegment(6, 5, 2);
      mesh->AddBdrSegment(7, 6, 2);
      mesh->AddBdrSegment(4, 7, 2);
   }
   else if (type == 4)
   {
      mesh = new Mesh(2, 8, 4, 8, 3);

      mesh->AddVertex(-1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0,  1.0, 1.0);
      mesh->AddVertex(-1.0,  1.0, 1.0);

      mesh->AddQuad(0, 1, 5, 4);
      mesh->AddQuad(1, 2, 6, 5);
      mesh->AddQuad(2, 3, 7, 6);
      mesh->AddQuad(3, 0, 4, 7);

      mesh->AddBdrSegment(0, 1, 1);
      mesh->AddBdrSegment(1, 2, 1);
      mesh->AddBdrSegment(2, 3, 1);
      mesh->AddBdrSegment(3, 0, 1);
      mesh->AddBdrSegment(5, 4, 2);
      mesh->AddBdrSegment(6, 5, 2);
      mesh->AddBdrSegment(7, 6, 2);
      mesh->AddBdrSegment(4, 7, 2);
   }
   else
   {
      MFEM_ABORT("Unrecognized mesh type " << type << "!");
   }
   mesh->FinalizeTopology();

   return mesh;
}

// Transforms the four-sided loop into a curved cylinder with skewed top and
// base.
void trans(const Vector &x, Vector &r)
{
   r.SetSize(3);

   real_t tol = 1e-6;
   real_t theta = 0.0;
   if (fabs(x[1] + 1.0) < tol)
   {
      theta = 0.25 * M_PI * (x[0] - 2.0);
   }
   else if (fabs(x[0] - 1.0) < tol)
   {
      theta = 0.25 * M_PI * x[1];
   }
   else if (fabs(x[1] - 1.0) < tol)
   {
      theta = 0.25 * M_PI * (2.0 - x[0]);
   }
   else if (fabs(x[0] + 1.0) < tol)
   {
      theta = 0.25 * M_PI * (4.0 - x[1]);
   }
   else
   {
      cerr << "side not recognized "
           << x[0] << " " << x[1] << " " << x[2] << endl;
   }

   r[0] = cos(theta);
   r[1] = sin(theta);
   r[2] = 0.25 * (2.0 * x[2] - 1.0) * (r[0] + 2.0);
}

// Anisotropic diffusion coefficient
void sigmaFunc(const Vector &x, DenseMatrix &s)
{
   s.SetSize(3);
   real_t a = 17.0 - 2.0 * x[0] * (1.0 + x[0]);
   s(0,0) = 0.5 + x[0] * x[0] * (8.0 / a - 0.5);
   s(0,1) = x[0] * x[1] * (8.0 / a - 0.5);
   s(0,2) = 0.0;
   s(1,0) = s(0,1);
   s(1,1) = 0.5 * x[0] * x[0] + 8.0 * x[1] * x[1] / a;
   s(1,2) = 0.0;
   s(2,0) = 0.0;
   s(2,1) = 0.0;
   s(2,2) = a / 32.0;
}
