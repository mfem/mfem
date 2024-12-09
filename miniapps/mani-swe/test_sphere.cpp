#include "mfem.hpp"
#include "manihyp.hpp"
#include <cmath>

using namespace mfem;

void gaussian_initial(const Vector &x, Vector &u)
{
   const real_t theta = std::acos(x[2]/std::sqrt(x*x));
   const real_t hmin = 1;
   const real_t hmax = 2;
   const real_t sigma = 0.2;
   u = 0.0;
   u[0] = hmin + (hmax - hmin)*std::exp(-theta*theta/(2*sigma*sigma));
}

class SphericalHeight : public VectorCoefficient
{
private:
   GridFunctionCoefficient h;
   const real_t scale;
   Vector normal;
public:
   SphericalHeight(GridFunction &h, const real_t scale=1.0):VectorCoefficient(3),
      h(&h), scale(scale), normal(3) { }
   virtual void Eval(Vector &node, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      T.Transform(T.GetIntPoint(), node);
      node /= std::sqrt(node*node);
      normal = node;
      normal *= scale * h.Eval(T, T.GetIntPoint()) / std::sqrt(normal*normal);
      node += normal;
   }
};

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

   int order = 3;
   int refinement_level = 4;
   int vis_step = 50;
   bool visualization = true;
   bool paraview = true;
   real_t cfl = 0.3;
   real_t tF = 1.5;

   OptionsParser args(argc, argv);
   args.AddOption(&refinement_level, "-r", "--refine",
                  "Mesh refinement level");
   args.ParseCheck();

   std::unique_ptr<ParMesh> pmesh;
   {
      Mesh mesh("./data/icosahedron.mesh");
      mesh.SetCurvature(order, true);
      pmesh.reset(new ParMesh(MPI_COMM_WORLD, mesh));
      mesh.Clear();
      UniformSpherRefinement(*pmesh, refinement_level);
   }
   std::unique_ptr<ParMesh> pmesh_visualize;
   {
      Mesh mesh("./data/icosahedron.mesh");
      mesh.SetCurvature(order+4, true);
      pmesh_visualize.reset(new ParMesh(MPI_COMM_WORLD, mesh));
      mesh.Clear();
      UniformSpherRefinement(*pmesh_visualize, refinement_level);
   }
   GridFunction *nodes = pmesh_visualize->GetNodes();

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
   ParGridFunction mom_x(&fes, mom.GetData() + 0*fes.GetTrueVSize());
   ParGridFunction mom_y(&fes, mom.GetData() + 1*fes.GetTrueVSize());
   ParGridFunction mom_z(&fes, mom.GetData() + 2*fes.GetTrueVSize());
   ManifoldPhysVectorCoefficient mom_cf(u, 1, dim, sdim);
   SphericalHeight deform_cf(height);

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

   socketstream height_sock, mom_x_sock, mom_y_sock, mom_z_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      MPI_Barrier(MPI_COMM_WORLD);
      height_sock.open(vishost, visport);
      height_sock.precision(8);
      // Plot height
      nodes->ProjectCoefficient(deform_cf);
      height_sock << "parallel " << numProcs << " " << myRank << "\n";
      height_sock << "solution\n" << *pmesh_visualize << height;
      height_sock << "window_title 'momentum, t = 0'\n";
      height_sock << "view 0 0\n";  // view from top
      height_sock << "autoscale off\n valuerange 0.5 2.5\n";
      height_sock << "keys jm\n";  // turn off perspective and light, show mesh
      height_sock << std::flush;
      MPI_Barrier(MPI_COMM_WORLD);

      // MPI_Barrier(MPI_COMM_WORLD);
      // mom.ProjectCoefficient(mom_cf);
      // mom_x_sock.open(vishost, visport);
      // mom_x_sock.precision(8);
      // // Plot magnitude of vector-valued momentum
      // mom_x_sock << "parallel " << numProcs << " " << myRank << "\n";
      // mom_x_sock << "solution\n" << *pmesh << mom_x;
      // mom_x_sock << "window_title 'momentum_x, t = 0'\n";
      // mom_x_sock << "view 0 0\n";  // view from top
      // mom_x_sock << "keys jm\n";  // turn off perspective and light, show mesh
      // mom_x_sock << std::flush;
      // MPI_Barrier(MPI_COMM_WORLD);
      //
      // MPI_Barrier(MPI_COMM_WORLD);
      // mom_y_sock.open(vishost, visport);
      // mom_y_sock.precision(8);
      // // Plot magnitude of vector-valued momentum
      // mom_y_sock << "parallel " << numProcs << " " << myRank << "\n";
      // mom_y_sock << "solution\n" << *pmesh << mom_y;
      // mom_y_sock << "window_title 'momentum_y, t = 0'\n";
      // mom_y_sock << "view 0 0\n";  // view from top
      // mom_y_sock << "keys jm\n";  // turn off perspective and light, show mesh
      // mom_y_sock << std::flush;
      // MPI_Barrier(MPI_COMM_WORLD);
      //
      // MPI_Barrier(MPI_COMM_WORLD);
      // mom_z_sock.open(vishost, visport);
      // mom_z_sock.precision(8);
      // // Plot magnitude of vector-valued momentum
      // mom_z_sock << "parallel " << numProcs << " " << myRank << "\n";
      // mom_z_sock << "solution\n" << *pmesh << mom_z;
      // mom_z_sock << "window_title 'momentum_z, t = 0'\n";
      // mom_z_sock << "view 0 0\n";  // view from top
      // mom_z_sock << "keys jm\n";  // turn off perspective and light, show mesh
      // mom_z_sock << std::flush;
      // MPI_Barrier(MPI_COMM_WORLD);
   }
   std::unique_ptr<ParaViewDataCollection> dacol;
   if (paraview)
   {
      dacol.reset(new ParaViewDataCollection("ParaViewSWE", pmesh.get()));
      dacol->SetLevelsOfDetail(order);
      dacol->RegisterField("Height", &height);
      dacol->RegisterField("Momentum", &mom);
      dacol->SetTime(0.0);
      dacol->SetCycle(0);
      dacol->Save();
   }

   real_t t = 0.0;
   real_t dt; // dt will be computed using CFL condition
   swe.SetTime(t);
   ode_solver->Init(swe);
   bool done = false;

   for (int ti = 1; !done; ti++)
   {
      // CFL condition
      dt = cfl * hmin / swe.GetMaxCharSpeed() / real_t(2*order+1);
      // Adjust dt so that t + dt <= tF
      real_t dt_real = std::min(dt, tF - t);
      if (Mpi::Root())
      {
         out << "time step: " << ti << ", time: " << t << std::endl;
         out << "\tMaxChar: " << swe.GetMaxCharSpeed() << std::endl;
      }

      // ODE step
      ode_solver->Step(u, t, dt_real);
      done = (t >= tF - 1e-8 * dt);


      // Visualize
      if (ti % vis_step == 0)
      {
         mom.ProjectCoefficient(mom_cf);
         if (height_sock.is_open() && height_sock.good())
         {
            nodes->ProjectCoefficient(deform_cf);
            height_sock << "parallel " << numProcs << " " << myRank << "\n";
            height_sock << "solution\n" << *pmesh_visualize << height;
            height_sock << "window_title 'height, t = " << t << "'\n";
         }
         // if (mom_x_sock.is_open() && mom_x_sock.good())
         // {
         //    mom_x_sock << "parallel " << numProcs << " " << myRank << "\n";
         //    mom_x_sock << "solution\n" << *pmesh << mom_x;
         //    mom_x_sock << "window_title 'momentum_x, t = " << t << "'\n";
         // }
         // if (mom_y_sock.is_open() && mom_y_sock.good())
         // {
         //    mom_y_sock << "parallel " << numProcs << " " << myRank << "\n";
         //    mom_y_sock << "solution\n" << *pmesh << mom_y;
         //    mom_y_sock << "window_title 'momentum_y, t = " << t << "'\n";
         // }
         // if (mom_z_sock.is_open() && mom_z_sock.good())
         // {
         //    mom_z_sock << "parallel " << numProcs << " " << myRank << "\n";
         //    mom_z_sock << "solution\n" << *pmesh << mom_z;
         //    mom_z_sock << "window_title 'momentum_z, t = " << t << "'\n";
         // }
         if (dacol)
         {
            dacol->SetTime(t);
            dacol->SetCycle(ti);
            dacol->Save();
         }

      }
   }
}
