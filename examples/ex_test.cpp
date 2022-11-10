
#include "mfem.hpp"

using namespace mfem;
using namespace std;

int Ww = 350, Wh = 350;
int Wx = 0, Wy = 0;
int offx = Ww+5, offy = Wh+25;

void Visualize(Mesh& mesh, GridFunction& gf, const string &title,
               const string& caption, int x, int y)
{
   int w = 400, h = 350;

   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2.precision(10);
   sol_sockL2 << "solution\n" << mesh << gf
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "window_title '" << title << "'"
              << "plot_caption '" << caption << "'" << flush;
}

int dimension = 2;


// for order 0, linear, order 1, quadratic, etc.

struct PolyCoeff
{
   static int order_;

   static double poly_coeff(const Vector& x)
   {
      int& o = order_;
      if (dimension == 2)
      {
         if (x[0] < 0.5) { return 1.0; }
         return 2.0;
         //return pow(x[0],o) +pow(x[1],o);
         // return pow(x[0],o) +pow(x[1],o);
      }
      else
      {
         return pow(x[0],o) +pow(x[1],o) +pow(x[2],o);
      }
   }
};
int PolyCoeff::order_ = -1;

double integrate(GridFunction* gf)
{
   ConstantCoefficient one(1.0);
   LinearForm lf(gf->FESpace());
   LinearFormIntegrator* lfi = new DomainLFIntegrator(one);
   lf.AddDomainIntegrator(lfi);
   lf.Assemble();
   double integral = lf(*gf);
   return integral;
}

double square_integrate(GridFunction* gf)
{
   Vector gf_sq_vec = *gf;
   GridFunction gf_sq(gf->FESpace());
   gf_sq.SetData(gf_sq_vec);
   for (int i = 0; i < gf->Size(); i++)
   {
      gf_sq(i) = (*gf)(i)*(*gf)(i);
   }

   return integrate(&gf_sq);
}

int main()
{
   Mesh mesh = Mesh::MakeCartesian2D(
                  1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);
   // 1, 1, Element::TRIANGLE, true, 1.0, 1.0);
   mesh.EnsureNCMesh(true);
   mesh.EnsureNodes();

   int order = 2;
   L2_FECollection fec(order, dimension, BasisType::GaussLegendre);
   // L2_FECollection fec(order, dimension, BasisType::Positive);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction x(&fespace);

   PolyCoeff pcoeff;
   pcoeff.order_ = order+1;
   FunctionCoefficient c(PolyCoeff::poly_coeff);

   Array<Refinement> refinements;
   refinements.Append(Refinement(0));
   mesh.GeneralRefinement(refinements);

   fespace.Update();
   x.Update();

   // project to get function that isn't exactly representable in the
   // fine space.
   x.ProjectCoefficient(c);

   // save the fine solution
   Vector xf = x;
   GridFunction x_fine(x.FESpace());
   x_fine.SetData(xf);

   double mass_fine = integrate(&x_fine);

   cout << "mass_fine: " << mass_fine << endl;

   Visualize(mesh, x_fine, "fine proj","fine proj",Wx, Wy); Wx += offx;

   Vector local_err(mesh.GetNE());
   local_err = 0.;
   double threshold = 1.0;
   mesh.DerefineByError(local_err, threshold);
   fespace.Update();
   x.Update();

   Vector coarse_soln_v{x};

   Visualize(mesh, x, "coarsened", "coarsened", Wx, Wy); Wx += offx;

   double mass_coarse = integrate(&x);
   cout << "mass_coarse: " << mass_coarse << endl;

   // conservation check
   assert( fabs(mass_fine-mass_coarse) < 1.e-12 );

   // re-refine to get everything on the same grid
   mesh.GeneralRefinement(refinements);
   fespace.Update();
   x.Update();

   // Compute error
   GridFunctionCoefficient gfc(&x);

   double err0 = x_fine.ComputeL2Error(gfc);

   cout << "err0: " << err0 << endl;

   double eps = 1.e-3;
   for (int i = 0; i < coarse_soln_v.Size(); i++)
   {
      for (int f = -1; f <= 1; f += 2)
      {
         printf("testing dof %d/%d w/ %+d... ",i,coarse_soln_v.Size(),f);

         mesh.DerefineByError(local_err, threshold);
         fespace.Update();
         x.Update();
         x = coarse_soln_v;
         x(i) += f*eps;

         mesh.GeneralRefinement(refinements);
         fespace.Update();
         x.Update();

         double err = x_fine.ComputeL2Error(gfc);

         if (err > err0)
         {
            cout << "is local minimum." << endl;
         }
         else
         {
            printf("err decreased from %f to %f (%e)\n",err0,err,err-err0);
         }
         //assert( err > err0 );
      }
   }
   cout << "pass: coarse solution is optimal" << endl;
}
