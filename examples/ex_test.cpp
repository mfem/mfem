
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
      return x[0]+x[1];
   }
   else
   {
      return x[0]+x[1]+x[2];
   }
}

int main()
{   
   Mesh mesh = Mesh::MakeCartesian2D(
      1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);
   mesh.EnsureNCMesh();

   int order = 1;
   H1_FECollection fec(order, dimension, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction x(&fespace);

   FunctionCoefficient c(linear_coeff);
   x.ProjectCoefficient(c);

   Vector coarse_projection = x;

   coarse_projection.Print();
   Visualize(mesh, x, "coarse proj","coarse proj",Wx, Wy); Wx += offx;
   
   Array<Refinement> refinements;
   refinements.Append(Refinement(0));
   mesh.GeneralRefinement(refinements);

   fespace.Update();
   x.Update();

   // re-project to get function that isn't exactly representable in
   // the coarse space.
   x.ProjectCoefficient(c);

   x.Print();
   Visualize(mesh, x, "fine proj","fine proj",Wx, Wy); Wx += offx;

   Vector local_err(mesh.GetNE());
   local_err = 0.;
   double threshold = 1.0;
   mesh.DerefineByError(local_err, threshold);

   fespace.Update();
   x.Update();

   Visualize(mesh, x, "coarsened","coarsened",Wx, Wy); Wx += offx;
   
   Vector coarse_derefinement = x;

   coarse_derefinement.Print();
   
   //assert(coarse_projection.Norml2() -coarse_derefinement.Norml2() < 1e-11);
}
