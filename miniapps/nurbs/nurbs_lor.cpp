//               Testing NURBS LOR code snippets
//
// Compile with: make nurbs_lor
//

#include "nurbs_lor.hpp"
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   const int vdim = 1;

   const char *mesh_file = "ho_mesh.mesh";
   bool printX = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&printX, "-X", "--printX", "-noX", "--no-printX",
                  "Print the interpolation matrix.");
   args.Parse();
   // Print & verify options
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   // Mesh mesh("ho_mesh.mesh");
   // Mesh lo_mesh("lo_mesh.mesh");

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

   GridFunction ho_x(&fespace);
   ho_x = 0.0;
   for (int i = 0; i < fespace.GetTrueVSize(); i++)
   {
      // ho_x(i) = 100.0 - (i-10.0)*(i-10.0); // example function
      ho_x(i) = 1.0 + i;
   }

   // Create a GridFunction on the LO mesh
   FiniteElementCollection* lo_fec = lo_mesh.GetNodes()->OwnFEC();
   FiniteElementSpace lo_fespace = FiniteElementSpace(&lo_mesh, lo_fec, vdim,
                                                      Ordering::byVDIM);
   GridFunction lo_x(&lo_fespace);
   X->Mult(ho_x, lo_x);
   cout << "Finished creating low-order grid function." << endl;

   // Now compare with the results of NURBSInterpolator
   GridFunction x_recon(&fespace);

   interpolator.Mult(lo_x, x_recon);

   // ----- Write to file -----
   Save("ho_x.gf", ho_x);
   Save("lo_x.gf", lo_x);
   Save("x_recon.gf", x_recon);

   return 0;
}
