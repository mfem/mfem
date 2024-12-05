#include "mfem.hpp"
#include "manihyp.hpp"
#include <cmath>

using namespace mfem;

void gaussian_initial(const Vector &x, Vector &u)
{
   // const real_t theta = std::acos(x[2]/std::sqrt(x*x));
   // const real_t hmin = 1;
   // const real_t hmax = 2;
   // const real_t sigma = 0.2;
   // u = 0.0;
   // u[0] = hmin + (hmax - hmin)*std::exp(-theta*theta/(2*sigma*sigma));

   const real_t hmin = 10;
   const real_t hmax = 20;
   const real_t sigma = 2;
   const real_t r2 = x[0]*x[0] + x[1]*x[1];
   u = 0.0;
   u[0] = hmin + (hmax - hmin)*std::exp(-r2/(2*sigma*sigma));
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
   Mpi::Init(argc, argv);
   const int numProcs = Mpi::WorldSize();
   const int myRank = Mpi::WorldRank();
   Hypre::Init();

   int order = 2;
   int refinement_level = 4;
   bool visualization = true;
   real_t cfl = 0.3;
   real_t tF = 10.0;

   OptionsParser args(argc, argv);
   args.AddOption(&refinement_level, "-r", "--refine",
                  "Mesh refinement level");
   args.ParseCheck();

   Hypre::Init();
   std::unique_ptr<ParMesh> pmesh;
   {
      Mesh mesh("./data/periodic-square-3d.mesh");
      mesh.Transform([](const Vector &x, Vector &y){ y = x; y[0] *= 10; y[1] *= 10;});
      mesh.UniformRefinement();
      mesh.SetCurvature(order);
      pmesh.reset(new ParMesh(MPI_COMM_WORLD, mesh));
      mesh.Clear();
      for (int i=0; i<refinement_level; i++)
      {
         pmesh->UniformRefinement();
      }
   }
   {
      // Mesh mesh("./data/icosahedron.mesh");
      // mesh.SetCurvature(order);
      // pmesh.reset(new ParMesh(MPI_COMM_WORLD, mesh));
      // mesh.Clear();
      // UniformSpherRefinement(*pmesh, refinement_level);
   }

   const int dim = pmesh->Dimension();
   const int sdim = pmesh->SpaceDimension();
   const int num_equations = dim + 1;
   const int phys_num_equations = sdim + 1;

   ManifoldCoord coord(dim, sdim);
   ShallowWaterFlux swe_phys(sdim);
   ManifoldFlux swe_mani(swe_phys, coord, 1);
   ManifoldRusanovFlux rusanovFlux(swe_mani);
   ManifoldHyperbolicFormIntegrator swe_integ(rusanovFlux);

   std::unique_ptr<ODESolver> ode_solver;
   ode_solver.reset(new RK4Solver());

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
   ParGridFunction height(&fes, u.GetData());
   ParGridFunction mom(&sfes);
   ManifoldPhysVectorCoefficient mom_cf(u, 1, dim, sdim);

   VectorFunctionCoefficient u0_phys(phys_num_equations, gaussian_initial);
   ManifoldStateCoefficient u0_mani(u0_phys, 1, 1, dim);
   u.ProjectCoefficient(u0_mani);

   ManifoldDGHyperbolicConservationLaws swe(vfes, swe_integ, 1);
   swe.SetTime(0.0);
   real_t hmin=infinity();
   {
      for (int i=0; i<pmesh->GetNE(); i++)
      {
         hmin = std::min(pmesh->GetElementSize(i, 1), hmin);
      }
      MPI_Allreduce(MPI_IN_PLACE, &hmin, 1, MFEM_MPI_REAL_T, MPI_MIN,
                    pmesh->GetComm());
      Vector z(vfes.GetTrueVSize());
      swe.Mult(u,z);
   }
   real_t dt = cfl * hmin / swe.GetMaxCharSpeed() / (2 * order + 1);

   socketstream height_sock, mom_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      MPI_Barrier(MPI_COMM_WORLD);
      height_sock.open(vishost, visport);
      height_sock.precision(8);
      // Plot height
      height_sock << "parallel " << numProcs << " " << myRank << "\n";
      height_sock << "solution\n" << *pmesh << height;
      height_sock << "window_title 'momentum, t = 0'\n";
      height_sock << "view 0 0\n";  // view from top
      height_sock << "keys jm\n";  // turn off perspective and light, show mesh
      height_sock << "pause\n";
      height_sock << std::flush;
      MPI_Barrier(MPI_COMM_WORLD);

      MPI_Barrier(MPI_COMM_WORLD);
      mom.ProjectCoefficient(mom_cf);
      mom_sock.open(vishost, visport);
      mom_sock.precision(8);
      // Plot magnitude of vector-valued momentum
      mom_sock << "parallel " << numProcs << " " << myRank << "\n";
      mom_sock << "solution\n" << *pmesh << mom;
      mom_sock << "window_title 'momentum, t = 0'\n";
      mom_sock << "view 0 0\n";  // view from top
      mom_sock << "keys jm\n";  // turn off perspective and light, show mesh
      mom_sock << "pause\n";
      mom_sock << std::flush;
      MPI_Barrier(MPI_COMM_WORLD);
   }

   real_t t = 0.0;
   swe.SetTime(t);
   ode_solver->Init(swe);
   bool done = false;
   ParaViewDataCollection dacol("ParaViewSWE", pmesh.get());
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("Height", &height);
   dacol.RegisterField("Momentum", &mom);
   dacol.SetTime(t);
   dacol.SetCycle(0);
   dacol.Save();
   for (int ti = 0; !done; ti++)
   {
      real_t dt_real = std::min(dt, tF - t);
      if (Mpi::Root())
      {
         out << "time step: " << ti << ", time: " << t << std::endl;
         out << "\tMaxChar: " << swe.GetMaxCharSpeed() << std::endl;
      }
      dt = cfl * hmin / swe.GetMaxCharSpeed() / real_t(2*order+1);
      ode_solver->Step(u, t, dt);
      done = (t >= tF - 1e-8 * dt);

      mom.ProjectCoefficient(mom_cf);
      height_sock << "parallel " << numProcs << " " << myRank << "\n";
      height_sock << "solution\n" << *pmesh << height;
      height_sock << "window_title 'height, t = " << t << "'\n";
      mom_sock << "parallel " << numProcs << " " << myRank << "\n";
      mom_sock << "solution\n" << *pmesh << mom;
      mom_sock << "window_title 'momentum, t = " << t << "'\n";
      dacol.SetTime(t);
      dacol.SetCycle(ti);
      dacol.Save();
   }
}
