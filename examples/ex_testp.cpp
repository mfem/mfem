
#include "mfem.hpp"

using namespace mfem;
using namespace std;

int Ww = 350, Wh = 350;
int Wx = 0, Wy = 0;
int offx = Ww+5, offy = Wh+25;

void Visualize(int num_procs, int myid,
               Mesh& mesh, GridFunction& gf, const string &title,
               const string& caption, int x, int y)
{
   int w = 400, h = 350;

   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2 << "parallel " << num_procs << " " << myid << "\n";
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
         return pow(x[0],o) +pow(x[1],o);
      }
      else
      {
         return pow(x[0],o) +pow(x[1],o) +pow(x[2],o);
      }
   }
};
int PolyCoeff::order_ = -1;

void RefineRandomly(ParMesh& pmesh,
                    ParFiniteElementSpace& fespace, ParGridFunction& u)
{
   double freq = 0.3;

   Array<Refinement> refinements;
   for (int k = 0; k < pmesh.GetNE(); k++)
   {
      double a = rand()/double(RAND_MAX);
      if (a < freq)
      {
         refinements.Append(Refinement(k));
      }
   }

   pmesh.GeneralRefinement(refinements);
   fespace.Update();
   u.Update();
}


void CoarsenRandomly(ParMesh& pmesh,
                     ParFiniteElementSpace& fespace, ParGridFunction& u)
{
   double freq = 0.2;

   Vector local_err(pmesh.GetNE());
   local_err = 1.1;
   double threshold = 1.0;

   for (int k = 0; k < pmesh.GetNE(); k++)
   {
      double a = rand()/double(RAND_MAX);
      if (a < freq)
      {
         local_err(k) = 0.0;
      }
   }

   int op = 0; // take min
   int nc_limit = 0;
   pmesh.DerefineByError(local_err, threshold, nc_limit, op);
   fespace.Update();
   u.Update();
}

int main()
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   Mesh mesh = Mesh::MakeCartesian2D(
                  4, 4, Element::QUADRILATERAL, true, 1.0, 1.0);
   mesh.EnsureNCMesh();
   mesh.EnsureNodes();

   ParMesh *pmeshp = new ParMesh(MPI_COMM_WORLD, mesh);
   ParMesh& pmesh{*pmeshp};

   int order = 2;
   L2_FECollection fec(order, dimension, BasisType::Positive);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   ParGridFunction x(&fespace);

   PolyCoeff pcoeff;
   pcoeff.order_ = order;
   FunctionCoefficient c(PolyCoeff::poly_coeff);

   x.ProjectCoefficient(c);

   // test correctness by:
   // initializing with an exact polynomial
   // loop:
   //   refine randomly
   //   rebalance
   //   coarsen randomly
   //   rebalance

   double refine_p = 0.3;
   double coarsen_p = 0.3;

   //srand( (unsigned)time( NULL )+myid );
   srand( 2+myid );

   int total_it = 20;
   for (int it = 0; it < total_it; it++)
   {

      printf("starting iteration %d\n",it);

      RefineRandomly(pmesh, fespace, x);

      //Visualize(num_procs, myid, pmesh, x, "after refine","after refine",Wx, Wy); Wx += offx;

      double err = x.ComputeL2Error(c);
      cout << "err after refine: " << err << endl;
      if (err > 1.e-12) { break; }

      pmesh.Rebalance();
      fespace.Update();
      x.Update();

      err = x.ComputeL2Error(c);
      cout << "err after rebalance fine: " << err << endl;
      if (err > 1.e-12) { break; }
      //assert(err < 1.e-12);

      CoarsenRandomly(pmesh, fespace, x);

      //Visualize(num_procs, myid, pmesh, x, "after coarsen","after coarsen",Wx, Wy); Wx += offx;

      err = x.ComputeL2Error(c);
      cout << "err after coarsen: " << err << endl;
      if (err > 1.e-12) { break; }
      //assert(err < 1.e-12);

      pmesh.Rebalance();
      fespace.Update();
      x.Update();

      err = x.ComputeL2Error(c);
      cout << "err after rebalance coarse: " << err << endl;
      if (err > 1.e-12) { break; }
      //assert(err < 1.e-12);

   }

   Visualize(num_procs, myid, pmesh, x, "after refine","after refine",Wx, Wy);
   Wx += offx;
}
