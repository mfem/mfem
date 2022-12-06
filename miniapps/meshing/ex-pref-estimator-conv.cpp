//                                MFEM Example 0
//
// Compile with: make ex-pref-estimator-conv
//
// Sample runs:  ex-pref-estimator-conv -o 2 -rs 0 -nrand 3 -prob 0.0 -type 1 -es 3
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
    if (type == 0) { //Gaussian bump
        double xc = x(0) - 0.5;
        double yc = x(1) - 0.5;
        return std::exp(-100*(xc*xc+yc*yc));

    }
    else if (type == 1) { // sin(2 pi x)*sin(2 pi y) + cos(2 pi x)*cos(2 pi y)
        return std::sin(x(0)*2.0*M_PI)*std::sin(x(1)*2.0*M_PI) +
               std::cos(x(0)*2.0*M_PI)*std::cos(x(1)*2.0*M_PI);
    }
    else if (type == 2) { // sin(2 pi x)*sin(2 pi y)
        return std::sin(x(0)*2.0*M_PI)*std::sin(x(1)*2.0*M_PI);
    }
    else if (type == 3) { // sin(2 pi r) + sin(3 pi r)
        double xc = x(0) - 0.5;
        double yc = x(1) - 0.5;
        double rc =std::pow(xc*xc+yc*yc, 0.5);
        return std::sin(rc*2.0*M_PI) + std::sin(rc*3.0*M_PI);
    }
    else if (type == 4) { //Eq 3.3 from https://www.math.tamu.edu/~guermond/PUBLICATIONS/MS/non_stationnary_jlg_rp_bp.pdf
        double xc = x(0) - 1.0;
        double yc = x(1) - 1.0;
        double rc =std::pow(xc*xc+yc*yc, 0.5);
        if (std::fabs(2*rc-0.3) <= 0.25) {
            return std::exp(-300*(2*rc-0.3)*(2*rc-0.3));
        }
        else if (std::fabs(2*rc-0.9) <= 0.25) {
            return std::exp(-300*(2*rc-0.3)*(2*rc-0.3));
        }
        else if (std::fabs(2*rc-1.6) <= 0.2) {
            return std::pow(1.0 - std::pow((2*rc-1.6)/0.2, 2.0), 0.5);
        }
        else if (std::fabs(rc-0.3) < 0.2) {
            return 0.0;
        }
        return 0.0;
    }
    else {
        MFEM_ABORT(" unknown function type. ");
    }
    return 0.0;
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
   int estimator = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&rs, "-rs", "--rs", "Number of refinements");
   args.AddOption(&type, "-type", "--type", "Type of function");
   args.AddOption(&nrand, "-nrand", "--nrand", "Number of random refinement");
   args.AddOption(&probmin, "-prob", "--prob", "Min probability of refinement when nrand > 0");
   args.AddOption(&estimator, "-es", "--estimator", "ZZ(0), Kelly(1), P-1(2), FaceJump(3)");
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

   ConstantCoefficient one(1.0);
   DiffusionIntegrator integ(one);
   L2_FECollection flux_fec(max_order, mesh.Dimension());
   FiniteElementSpace flux_fespace(&mesh, &flux_fec);

   ErrorEstimator *es = NULL;
   if (estimator == 0) {
       es = new ExactError(x, scoeff);
   }
   else if (estimator == 1) {
       es = new LSZienkiewiczZhuEstimator(integ, x);
   }
   else if (estimator == 2) {
       es = new KellyErrorEstimator(integ, x, &flux_fespace);
   }
   else if (estimator == 3) {
       es = new PRefDiffEstimator(x, -1);
   }
   else if (estimator == 4) {
       es = new PRefJumpEstimator(x);
   }
   else {
       MFEM_ABORT("invalid estimator type");
   }

   ThresholdRefiner refiner(*es);
   refiner.SetTotalErrorFraction(0.8); // use purely local threshold

   int ndofs = x.FESpace()->GetNDofs();
   double global_err = x.ComputeL2Error(scoeff);

   std::cout << type << " " << estimator << " " <<
           order << " " <<
           mesh.GetNE() << " " <<
           ndofs << " " <<
           global_err << " " <<
           "Totalerror\n";

   int tarndofs = 2*ndofs;
   while (ndofs < tarndofs) {
//   for (int i = 0; i < 10; i++) {
       Vector error_estimate = es->GetLocalErrors();
       double threshold = error_estimate.Max() * 0.8;
       for (int e = 0; e < mesh.GetNE(); e++) {
           if (error_estimate(e) > threshold) {
               int setOrder = fespace.GetElementOrder(e);
               fespace.SetElementOrder(e, setOrder+1);
           }
       }
       fespace.Update(false);
       x.Update();
       x.ProjectCoefficient(scoeff);

       ndofs = x.FESpace()->GetNDofs();
       global_err = x.ComputeL2Error(scoeff);

       std::cout << type << " " <<
                    estimator << " " <<
                    order << " " <<
                    mesh.GetNE() << " " <<
                    ndofs << " " <<
                    global_err << " " <<
                    "Totalerror\n";
   }

   for (int e = 0; e < mesh.GetNE(); e++) {
       ElOrder(e) = fespace.GetElementOrder(e);
   }
   max_order = fespace.GetMaxElementOrder();

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
   return 0;
}
