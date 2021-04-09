#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

Mesh * GetMesh(int type);

void trans(const Vector &x, Vector &r);

void sigmaFunc(const Vector &x, DenseMatrix &s);

double uExact(const Vector &x)
{
   return (0.25 * (2.0 + x[0]) - x[2]) * (x[2] + 0.25 * (2.0 + x[0]));
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 3;
   int mesh_type = 4; // Default to Quadrilateral mesh
   int ref_levels = 0;
   bool static_cond = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_type, "-mt", "--mesh-type",
                  "Mesh type: 3 - Triangular, 4 - Quadrilateral.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = GetMesh(mesh_type);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   mesh->SetCurvature(3);
   mesh->Transform(trans);

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm a(&fespace);
   MatrixFunctionCoefficient sigma(3, sigmaFunc);
   a.AddDomainIntegrator(new DiffusionIntegrator(sigma));

   // 9. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 10. Solve the linear system A X = B.
   if (!pa)
   {
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
   }
   else // Jacobi preconditioning in partial assembly mode
   {
      if (UsesTensorBasis(fespace))
      {
         OperatorJacobiSmoother M(a, ess_tdof_list);
         PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }

   // 11. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   FunctionCoefficient uCoef(uExact);
   double err = x.ComputeL2Error(uCoef);

   mfem::out << "|u - u_h|_2 = " << err << endl;

   // 12. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 14. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }
   delete mesh;

   return 0;
}

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

void trans(const Vector &x, Vector &r)
{
   r.SetSize(3);

   double tol = 1e-6;
   double theta = 0.0;
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
      cout << "side not recognized "
           << x[0] << " " << x[1] << " " << x[2] << endl;
   }

   r[0] = cos(theta);
   r[1] = sin(theta);
   r[2] = 0.25 * (2.0 * x[2] - 1.0) * (r[0] + 2.0);
}

void sigmaFunc(const Vector &x, DenseMatrix &s)
{
   s.SetSize(3);
   double a = 17.0 - 2.0 * x[0] * (1.0 + x[0]);
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
