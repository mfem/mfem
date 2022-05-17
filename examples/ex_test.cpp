
#include "mfem.hpp"

using namespace mfem;
using namespace std;

int Ww = 350, Wh = 350;
int Wx = 0, Wy = 0;
int offx = Ww+5, offy = Wh+25;

void Visualize(Mesh& mesh, GridFunction& gf, const string &title, const string& caption, int x, int y)
{
    int w = 350, h = 350;

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

// linear function
double linear_coeff(const Vector& x)
{
   if (dimension == 2)
   {
      return pow(x[0],2.0)+pow(x[1],2.0);
   }
   else
   {
      return x[0]+x[1]+x[2];
   }
}

double integrate(GridFunction* gf)
{
   ConstantCoefficient one(1.0);
   LinearForm lf(gf->FESpace());
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();
   double integral = lf(*gf);
   return integral;
}

double square_integrate(GridFunction* gf)
{
   Vector gf_sq_vec = *gf;
   GridFunction gf_sq(gf->FESpace());
   gf_sq.SetData(gf_sq_vec);
   for (int i = 0; i < gf->Size(); i++) {
      gf_sq(i) = (*gf)(i)*(*gf)(i);
   }

   return integrate(&gf_sq);
}

int main()
{
   Mesh mesh = Mesh::MakeCartesian2D(
      1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);
   mesh.EnsureNCMesh();
   mesh.EnsureNodes();

   int order = 1;
   L2_FECollection fec(order, dimension, BasisType::Positive);
   //L2_FECollection fec(order, dimension, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction x(&fespace);

   FunctionCoefficient c(linear_coeff);
   x.ProjectCoefficient(c);

   Vector coarse_projection = x;

   Visualize(mesh, x, "coarse proj","coarse proj",Wx, Wy); Wx += offx;

   Array<Refinement> refinements;
   refinements.Append(Refinement(0));
   mesh.GeneralRefinement(refinements);

   fespace.Update();
   x.Update();

   // re-project to get function that isn't exactly representable in
   // the coarse space.
   x.ProjectCoefficient(c);

   double mass_fine = integrate(&x);
   cout << "mass_fine: " << mass_fine << endl;

   Vector fine_projection = x;

   Visualize(mesh, x, "fine proj","fine proj",Wx, Wy); Wx += offx;

   Vector local_err(mesh.GetNE());
   local_err = 0.;
   double threshold = 1.0;
   mesh.DerefineByError(local_err, threshold);
   fespace.Update();
   x.Update();

   Visualize(mesh, x, "coarsened", "coarsened", Wx, Wy); Wx += offx;

   double mass_coarse = integrate(&x);
   cout << "mass_coarse: " << mass_coarse << endl;

   // conservation check
   assert( fabs(mass_fine-mass_coarse) < 1.e-12 );

   Vector coarse_soln_vec = x;
   GridFunction coarse_solution(x.FESpace());
   coarse_solution.SetData(coarse_soln_vec);

   // test optimality of projection

   mesh.GeneralRefinement(refinements);
   fespace.Update();
   x.Update();

   Visualize(mesh, x, "re-refined","re-refined",Wx, Wy); Wx += offx;

   Vector coarse_soln_in_fine_space_vec = x;
   GridFunction coarse_soln_in_fine_space(x.FESpace());
   coarse_soln_in_fine_space.SetData(coarse_soln_in_fine_space_vec);

   Vector err_gf_vec = x;
   GridFunction err_gf(x.FESpace());
   err_gf.SetData(err_gf_vec);

   err_gf -= fine_projection;

   double err0 = square_integrate(&err_gf);
   cout << "err0: " << err0 << endl;

   double eps = 1.e-6;
   for (int f = -1; f <= 1; f += 2) {
      for (int i = 0; i < coarse_solution.Size(); i++) {
         printf("testing dof %d/%d... ",i,coarse_solution.Size());

         mesh.DerefineByError(local_err, threshold);
         fespace.Update();
         x.Update();
         x = coarse_solution;
         x(i) += f*eps;

         mesh.GeneralRefinement(refinements);
         fespace.Update();
         x.Update();

         err_gf = x;
         err_gf -= fine_projection;

         double err = square_integrate(&err_gf);
         if (err > err0) {
            cout << "... is local minimum." << endl;
         }

         assert( err > err0 );
      }
   }
   cout << "pass: coarse solution is optimal" << endl;
}
