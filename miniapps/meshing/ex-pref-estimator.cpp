//                                MFEM Example 0
//
// Compile with: make ex-pref-estimator
//
// Sample runs:  ex-pref-estimator -o 2 -rs 1 -nrand 0 -prob 0.0 -type 0 -nor
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int type = 0;

double sfun(const Vector & x)
{
    if (type == 0) { // Gaussian bump
        double xc = x(0) - 0.5;
        double yc = x(1) - 0.5;
        return std::exp(-100*(xc*xc+yc*yc));

    }
    else { // sin(2 pi x)*sin(2 pi y) + cos(2 pi x)*cos(2 pi y)
        return std::sin(x(0)*2.0*M_PI)*std::sin(x(1)*2.0*M_PI) +
               std::cos(x(0)*2.0*M_PI)*std::cos(x(1)*2.0*M_PI);
    }
}

void LogNormalizeErrors(const Vector &error, GridFunction &xl2)
{
    MFEM_VERIFY(error.Size() == xl2.Size(), "Vector and gridfunction size"
                                            "incompatible.");
    for (int i = 0; i < error.Size(); i++) {
        xl2(i) = std::log(error(i));
    }

    double minv = xl2.Min();
    double maxv = xl2.Max();

    for (int i = 0; i < xl2.Size(); i++) {
        xl2(i) = (xl2(i)-minv)/(maxv-minv);
    }
}

void CompareErrors(const Vector &exact_error, GridFunction &estimate)
{
    MFEM_VERIFY(exact_error.Size() == estimate.Size(), "Vector and gridfunction size"
                                            "incompatible.");
    for (int i = 0; i < estimate.Size(); i++) {
        estimate(i) = std::fabs(estimate(i)-exact_error(i));
        if (exact_error(i) > 0.0) {
            estimate(i) *= 1.0/exact_error(i);
        }
    }
}

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int rs = 0;
   int nrand = 0;
   double probmin = 0.0;
   bool normalize = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&rs, "-rs", "--rs", "Number of refinements");
   args.AddOption(&type, "-type", "--type", "Type of function");
   args.AddOption(&nrand, "-nrand", "--nrand", "Number of random refinement");
   args.AddOption(&probmin, "-prob", "--prob", "Min probability of refinement when nrand > 0");
   args.AddOption(&normalize, "-nor", "--nor", "-no-nor",
                  "--no-nor",
                  "Log error normalization.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   for (int i = 0; i < rs; i++) {
       mesh.UniformRefinement();
   }
   mesh.EnsureNCMesh();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   int dim = mesh.Dimension();
   L2_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);

   //Do random p-refinement
   for (int i = 0; i < nrand; i++) {
       for (int e = 0; e < mesh.GetNE(); e++) {
           double probref = (double) rand() / RAND_MAX;
           double inc = probref > probmin ? 1.0 : 0.0;
           fespace.SetElementOrder(e,fespace.GetElementOrder(e)+inc);
       }
   }
   fespace.Update(false);

   GridFunction x(&fespace);
   x = 0.0;

   // Space and Function for element-wise quantities
   L2_FECollection fecl2(0, mesh.Dimension());
   FiniteElementSpace fespacel2(&mesh, &fecl2);
   GridFunction xl2(&fespacel2);

   // Element order after p-refinement
   GridFunction ElOrder(&fespacel2);
   for (int e = 0; e < mesh.GetNE(); e++) {
       ElOrder(e) = fespace.GetElementOrder(e);
   }
   int max_order = fespace.GetMaxElementOrder();

   // Function Coefficient
   FunctionCoefficient scoeff(sfun);
   x.ProjectCoefficient(scoeff);

   // Compute exact error
   Vector elem_errors_exact(mesh.GetNE());
   x.ComputeElementL2Errors(scoeff, elem_errors_exact, NULL);

   // Kelly error estimator
   ConstantCoefficient one(1.0);
   DiffusionIntegrator integ(one);
   L2_FECollection flux_fec(max_order, mesh.Dimension());
   FiniteElementSpace flux_fespace(&mesh, &flux_fec);
   KellyErrorEstimator kelly = KellyErrorEstimator(integ, x, &flux_fespace);
   Vector elem_errors_kelly = kelly.GetLocalErrors();

   // PMinusOne Error estimator
   PRefDiffEstimator prefes = PRefDiffEstimator(x, -1);
   Vector elem_errors_pminusone = prefes.GetLocalErrors();

   // Face jump estimator
   PRefJumpEstimator prefjumpes = PRefJumpEstimator(x);
   Vector elem_errors_facejump = prefjumpes.GetLocalErrors();

   GridFunction *xprolong = ProlongToMaxOrder(&x);
   int px = 0;
   int py = 0;
   int wx = 400;
   int wy = 400;

   if (true) {
       socketstream vis1;
       common::VisualizeField(vis1, "localhost", 19916, *xprolong, "Solution",
                              px, py, wx, wy, "jRmc");
   }
   px += wx;
   if (true) {
       socketstream vis1;
       common::VisualizeField(vis1, "localhost", 19916, ElOrder, "ElementOrder",
                              px, py, wx, wy, "jRmc");
   }

   px += wx;
   if (true) {
       if (normalize) {
           LogNormalizeErrors(elem_errors_exact, xl2);
       }
       else {
           xl2 = elem_errors_exact;
       }
       socketstream vis1;
       common::VisualizeField(vis1, "localhost", 19916, xl2, "Exact error",
                              px, py, wx, wy, "jRmc");
   }
   px = 0;
   py += wy;
   if (true) {
       if (normalize) {
           LogNormalizeErrors(elem_errors_kelly, xl2);
       }
       else {
           xl2 = elem_errors_kelly;
       }
       socketstream vis1;
       common::VisualizeField(vis1, "localhost", 19916, xl2, "Kelly error",
                              px, py, wx, wy, "jRmc");
   }

   px += wx;
   if (true) {
       if (normalize) {
           LogNormalizeErrors(elem_errors_pminusone, xl2);
       }
       else {
           xl2 = elem_errors_pminusone;
       }
       socketstream vis1;
       common::VisualizeField(vis1, "localhost", 19916, xl2, "PMinusOne",
                              px, py, wx, wy, "jRmc");
   }
   px += wx;
   if (true) {
       if (normalize) {
           LogNormalizeErrors(elem_errors_facejump, xl2);
       }
       else {
           xl2 = elem_errors_facejump;
       }
       socketstream vis1;
       common::VisualizeField(vis1, "localhost", 19916, xl2, "Face Jump",
                              px, py, wx, wy, "jRmc");
   }
   px += wx;

   delete xprolong;

   return 0;
}
