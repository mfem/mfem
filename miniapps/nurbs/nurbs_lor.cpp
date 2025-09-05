//               Testing NURBS LOR code snippets
//
// Compile with: make nurbs_lor
//

#include "nurbs_lor.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

// Analytic function:
real_t f_exact(real_t x0, real_t R, real_t n, const Vector & x)
{
   real_t r = 0.0;
   for (int i = 0; i < x.Size(); ++i)
   {
      r += pow(x(i)-x0, 2);
   }
   r = sqrt(r);
   return exp(-r / R)
        * pow(sin(n*x(0) * M_PI), 2)
        * pow(sin(n*x(1) * M_PI), 2);
   // return exp(-r / R)
   //      * pow(sin((3*x(0)*x(1) - 0.25) * M_PI), 2);
}

int main(int argc, char *argv[])
{
   const int vdim = 1;

   const char *mesh_file = "ho_mesh.mesh";
   int ref_levels = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   bool printX = false;
   // for exact function
   real_t x0 = 0.5;
   real_t R = 0.3;
   real_t n = 3.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&x0, "-x0", "--x0", "Center for the exact function. Default is 0.5.");
   args.AddOption(&R, "-R", "--radius", "Radius for the exact function. Default is 0.5.");
   args.AddOption(&n, "-n", "--periods", "Periods for the exact function. Default is 3.0.");
   args.AddOption(&printX, "-X", "--printX", "-noX", "--no-printX",
                  "Print the interpolation matrix.");
   args.Parse();
   // Print & verify options
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   // Mesh mesh("ho_mesh.mesh");
   // Mesh lo_mesh("lo_mesh.mesh");

   // Increase the NURBS degree.
   if (nurbs_degree_increase>0)
   {
      mesh.DegreeElevate(nurbs_degree_increase);
   }
   // Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.NURBSUniformRefinement();
   }


   // Create a GridFunction on the HO mesh
   FiniteElementCollection* fec = mesh.GetNodes()->OwnFEC();
   FiniteElementSpace fespace = FiniteElementSpace(&mesh, fec, vdim,
                                                   Ordering::byVDIM);
   const long Ndof = fespace.GetTrueVSize();
   cout << "Number of finite element unknowns: " << Ndof << endl;
   cout << "Number of elements: " << fespace.GetNE() << endl;
   cout << "Number of patches: " << mesh.NURBSext->GetNP() << endl;

   SparseMatrix* X = new SparseMatrix(Ndof, Ndof);
   Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(NURBSInterpolationRule::Botella, vdim, X);
   // if (printX) { Save("X.txt", X); }
   cout << "Finished creating low-order mesh." << endl;

   // Save meshes
   Save("ho_mesh.mesh", mesh);
   Save("lo_mesh.mesh", lo_mesh);

   // Get dimension interpolation matrices: X1, X2, X3
   const int tdim = mesh.NURBSext->Dimension();
   Array<const KnotVector*> kvs(tdim);
   Array<const KnotVector*> lo_kvs(tdim);
   Array<Vector*> lo_uknots(tdim);
   mesh.NURBSext->GetPatchKnotVectors(0, kvs);
   lo_mesh.NURBSext->GetPatchKnotVectors(0, lo_kvs);

   mesh.NURBSext->AssembleCollocationMatrix(NURBSInterpolationRule::Botella);

   SparseMatrix X0 = kvs[0]->GetInterpolationMatrix(NURBSInterpolationRule::Botella);
   SparseMatrix X1 = kvs[1]->GetInterpolationMatrix(NURBSInterpolationRule::Botella);
   if (printX)
   {
      Save("X0.txt", &X0);
      Save("X1.txt", &X1);
   }

   // Create a NURBSInterpolator object
   NURBSInterpolator interpolator(&mesh, &lo_mesh);

   // Create a GridFunction on the LO mesh
   FiniteElementCollection* lo_fec = lo_mesh.GetNodes()->OwnFEC();
   FiniteElementSpace lo_fespace = FiniteElementSpace(&lo_mesh, lo_fec, vdim,
                                                      Ordering::byVDIM);
   GridFunction lo_x(&lo_fespace);

   // // Evaluate function at low-order knots
   std::function<real_t(const Vector&)> f = std::bind(f_exact, x0, R, n, std::placeholders::_1);
   interpolator.EvaluateFunction(f, lo_x);
   cout << "Finished creating low-order grid function." << endl;
   Save("lo_x.gf", lo_x);

   // Interpolate function onto HO space by solving X * ho_x = lo_x
   GridFunction ho_x(&fespace);
   interpolator.InterpolateFunction(f, ho_x);
   Save("ho_x.gf", ho_x);

   return 0;
}