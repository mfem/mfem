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
   std::unique_ptr<ParMesh> pmesh;
   {
      Mesh mesh("./data/icosahedron.mesh");
      pmesh.reset(new ParMesh(MPI_COMM_WORLD, mesh));
      mesh.Clear();
   }

   const int dim = pmesh->Dimension();
   const int sdim = pmesh->SpaceDimension();

   pmesh->SetCurvature(order);
   ParGridFunction &x = static_cast<ParGridFunction&>(*pmesh->GetNodes());
   VectorFunctionCoefficient sphere_cf(3, sphere);
   x.ProjectCoefficient(sphere_cf);
   for (int i=0; i<refinement_level; i++)
   {
      pmesh->UniformRefinement();
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
      sol_sock << "solution\n" << *pmesh << x
               << "keys 'mj'"
               << std::flush;
   }
}
