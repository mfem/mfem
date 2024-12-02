#include "mfem.hpp"
#include "manihyp.hpp"

using namespace mfem;
void sphere(const Vector &x, Vector &y)
{
   y = x;
   y /= y.Norml2();
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   int order = 2;
   int refinement_level = 3;

   Hypre::Init();
   Mesh mesh("./data/icosahedron.mesh");
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();
   const int num_equations = dim + 1;

   ShallowWaterFlux swe_flux(sdim);
   ManifoldCoord coord(dim, sdim);
   ManifoldFlux maniflux(swe_flux, coord, 1);

   pmesh.SetCurvature(order);
   ParGridFunction &x = static_cast<ParGridFunction&>(*pmesh.GetNodes());
   VectorFunctionCoefficient sphere_cf(3, sphere);
   x.ProjectCoefficient(sphere_cf);
   for(int i=0; i<refinement_level; i++)
   {
      pmesh.UniformRefinement();
      x.ProjectCoefficient(sphere_cf);
   }

   bool visualization = true;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << std::flush;
   }
}
