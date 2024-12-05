#include "mfem.hpp"
#include "manihyp.hpp"
#include <cmath>

using namespace mfem;

void gaussian_initial(const Vector &x, Vector &u)
{
   const real_t theta = std::acos(x[2]/std::sqrt(x*x)); const real_t hmin = 10;
   const real_t hmax = 20;
   const real_t sigma = 0.2;
   u = 0.0;
   // u[0] = hmin + (hmax - hmin)*std::exp(-theta*theta/(2*sigma*sigma));
   u[0] = hmax;
}

void UniformSpherRefinement(ParMesh &pmesh, int ref_level)
{
   ParGridFunction &x = static_cast<ParGridFunction&>(*pmesh.GetNodes());
   VectorFunctionCoefficient sphere_cf(3, [](const Vector& x, Vector &y) {sphere(x,y,1.0);});
   x.ProjectCoefficient(sphere_cf);
   for (int i=0; i<ref_level; i++)
   {
      pmesh.UniformRefinement();
      x.ProjectCoefficient(sphere_cf);
   }
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
      Mesh mesh("./data/periodic-square-3d.mesh");
      mesh.UniformRefinement();
      pmesh.reset(new ParMesh(MPI_COMM_WORLD, mesh));
      mesh.Clear();
   }

   const int dim = pmesh->Dimension();
   const int sdim = pmesh->SpaceDimension();
   const int num_equations = dim + 1;
   const int phys_num_equations = sdim + 1;
   ShallowWaterFlux swe_phys(sdim);
   ManifoldCoord coord(dim, sdim);
   ManifoldFlux swe_mani(swe_phys, coord, 1);
   ManifoldRusanovFlux rusanovFlux(swe_mani);
   ManifoldHyperbolicFormIntegrator swe_integ(rusanovFlux);

   pmesh->SetCurvature(order);
   for (int i=0; i<refinement_level; i++)
   {
      pmesh->UniformRefinement();
   }
   // UniformSpherRefinement(*pmesh, refinement_level);

   DG_FECollection dg_fec(order, dim);
   // FE Space for state
   ParFiniteElementSpace vfes(pmesh.get(), &dg_fec, num_equations,
                              Ordering::byNODES);
   // FE space for manifold vector
   ParFiniteElementSpace dfes(pmesh.get(), &dg_fec, dim, Ordering::byNODES);
   // FE space for physical vector
   ParFiniteElementSpace sfes(pmesh.get(), &dg_fec, sdim, Ordering::byNODES);
   // FE space for scalar
   ParFiniteElementSpace fes(pmesh.get(), &dg_fec);

   // State
   ParGridFunction u(&vfes);
   // Height for visualization
   ParGridFunction height(&fes, &u);
   ParGridFunction mom(&sfes);
   ManifoldPhysVectorCoefficient mom_cf(u, 1, dim, sdim);

   VectorFunctionCoefficient u0_phys(phys_num_equations, gaussian_initial);
   ManifoldStateCoefficient u0_mani(u0_phys, 1, 1, dim);
   u.ProjectCoefficient(u0_mani);

   ManifoldDGHyperbolicConservationLaws swe(vfes, swe_integ, 1);
   Vector z(vfes.GetTrueVSize());
   swe.Mult(u, z);

   bool visualization = true;
   if (visualization)
   {
      mom.ProjectCoefficient(mom_cf);
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << mom
               << "keys 'mj'"
               << std::flush;
   }
}
