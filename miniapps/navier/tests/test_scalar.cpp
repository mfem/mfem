#include "navier_solver.hpp"
#include "scalar.hpp"
#include <vector>

using namespace mfem;
using namespace navier;

static inline double eddy_viscosity(const double k, const double mu,
                                    const double l)
{
   return mu * l * sqrt(k);
}

class ReactionCoefficient : public Coefficient
{
public:
   ReactionCoefficient(const GridFunction &kgf,
                       const GridFunction &vgf,
                       const GridFunction &lgf) :
      kgf(kgf), vgf(vgf), lgf(lgf) {};

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      const double k = kgf.GetValue(T, ip);
      const double l = lgf.GetValue(T, ip);

      vgf.GetVectorValue(T, ip, v);
      vgf.GetVectorGradient(T, S);

      // S = 0.5 * (\nabla v + \nabla^T v)
      S.Symmetrize();

      double abs_S_squared = 0.0;
      for (int i = 0; i < S.NumRows(); i++)
      {
         for (int j = 0; j < S.NumCols(); j++)
         {
            abs_S_squared += S(i,j) * S(i,j);
         }
      }

      return -k * sqrt(k) + sqrt(k) * abs_S_squared;
   }

private:
   const GridFunction &kgf, &vgf, &lgf;
   Vector v;
   DenseMatrix S;
};

class ViscosityCoefficient : public Coefficient
{
public:
   ViscosityCoefficient(const GridFunction &nugf, const GridFunction &kgf,
                        const GridFunction &lgf, const double mu) :
      nugf(nugf), kgf(kgf), lgf(lgf), mu(mu) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      const double k = kgf.GetValue(T, ip);
      const double nu = nugf.GetValue(T, ip);
      const double l = lgf.GetValue(T, ip);

      return nu + eddy_viscosity(k, mu, l);
   }

private:
   const GridFunction &nugf, &kgf, &lgf;
   const double mu;
};

static inline
double wall_distance(const double x, const double y)
{
   return x*x + y*y;
}

static inline
Vector dwall_distancedx(const double x, const double y)
{
   Vector dwall_distancedx(2);
   dwall_distancedx(0) = 2.0 * x;
   dwall_distancedx(1) = 2.0 * y;
   return dwall_distancedx;
}

static inline
double nu(const double x, const double y)
{
   return 1e-5 * x * y;
}

static inline
Vector dnudx(const double x, const double y)
{
   Vector dnudx(2);
   dnudx(0) = 1e-5 * y;
   dnudx(1) = 1e-5 * x;
   return dnudx;
}

double run(const unsigned int polynomial_order, const unsigned int nel)
{
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   const double mu = 1e-5;
   bool diffusion_enabled = false;

   Mesh mesh = Mesh::MakeCartesian2D(nel, nel, Element::QUADRILATERAL, false,
                                     2.0*M_PI,
                                     2.0*M_PI);

   mesh.EnsureNodes();

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   H1_FECollection fec(polynomial_order);
   ParFiniteElementSpace h1_scalar_fes(pmesh, &fec);
   ParFiniteElementSpace h1_vector_fes(pmesh, &fec, pmesh->Dimension());

   VectorFunctionCoefficient vel_coeff(pmesh->Dimension(), [&](const Vector &c,
                                                               double t,
                                                               Vector &u)
   {
      const double x = c(0);
      const double y = c(1);
      u(0) = -x;
      u(1) = y;
   });

   FunctionCoefficient k_ex_coeff([&](const Vector &c, double t)
   {
      const double x = c(0);
      const double y = c(1);
      return 2+cos(x)*sin(y);
   });

   FunctionCoefficient l_coeff([&](const Vector &c, double t)
   {
      const double x = c(0);
      const double y = c(1);
      return wall_distance(x,y);
   });

   FunctionCoefficient nu_coeff([&](const Vector &c, double t)
   {
      const double x = c(0);
      const double y = c(1);
      return nu(x,y);
   });

   FunctionCoefficient f_coeff([&](const Vector &c, double t)
   {
      const double x = c(0);
      const double y = c(1);
      if (diffusion_enabled)
      {

      }
      else
      {
         return x*sin(x)*sin(y) + cos(x)*(y*cos(y) + sin(y)*sqrt(2 + cos(x)*sin(y)));
      }
   });

   ParGridFunction vgf(&h1_vector_fes);
   vgf.ProjectCoefficient(vel_coeff);

   ScalarEquation se(*pmesh, polynomial_order, vgf);

   ParGridFunction &k_gf = se.GetScalar();
   ParGridFunction kex_gf(k_gf);

   Vector k_tdof(k_gf.ParFESpace()->GetTrueVSize());
   k_gf.ProjectCoefficient(k_ex_coeff);
   k_gf.GetTrueDofs(k_tdof);

   ParGridFunction nu_gf(&h1_scalar_fes);
   nu_gf.ProjectCoefficient(nu_coeff);

   ParGridFunction &l_gf = nu_gf;

   nu_gf.ExchangeFaceNbrData();
   l_gf.ExchangeFaceNbrData();

   ReactionCoefficient recoeff(k_gf, vgf, l_gf);
   se.AddReaction(recoeff);

   ConstantCoefficient zero(0.0);
   se.SetViscosityCoefficient(zero);

   se.SetFixedValue(k_ex_coeff, {0,1,2,3});
   se.AddForcing(f_coeff);

   se.Setup();

   double t = 0.0;
   double dt = 1e-8;
   double t_final = 100*dt;

   // ForwardEulerSolver se_ode;
   ARKStepSolver se_ode(MPI_COMM_WORLD, ARKStepSolver::EXPLICIT);
   se_ode.Init(se);
   se_ode.SetSStolerances(1e-5, 1e-5);
   se_ode.SetMaxStep(dt);
   // se_ode.SetERKTableNum(0);

   bool done = false;
   int ti = 0;
   int vis_steps = 10;

   char vishost[] = "128.15.198.77";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   if (Mpi::Root())
   {
      std::cout << "time step: " << ti << ", time: " << t << std::endl;
   }
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh << k_gf << "\n" << "pause\n" << std::flush;

   double l2error = std::numeric_limits<double>::infinity();
   while (!done)
   {
      se_ode.Step(k_tdof, t, dt);

      ti++;
      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         k_ex_coeff.SetTime(t);
         k_gf.SetFromTrueDofs(k_tdof);
         l2error = k_gf.ComputeL2Error(k_ex_coeff);
         // kex_gf.ProjectCoefficient(k_ex_coeff);
         // for (int i = 0; i < kex_gf.Size(); i++)
         // {
         //    kex_gf(i) = abs(kex_gf(i) - k_gf(i));
         // }
         if (Mpi::Root())
         {
            printf("time step: %d, time: %.3E, l2error = %.2E\n", ti, t, l2error);
            // se_ode.PrintInfo();
         }

         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << k_gf << std::flush;
      }
   }
   delete pmesh;
   return l2error;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   std::vector<int> orders({2, 3, 4});
   std::vector<int> nels({4, 8, 16});
   // std::vector<int> orders({2});
   // std::vector<int> nels({8});

   std::vector<double> l2errors(nels.size());
   for (int i = 0; i < orders.size(); i++)
   {
      l2errors.clear();
      for (int j = 0; j < nels.size(); j++)
      {
         double l2error = run(orders[i], nels[j]);
         l2errors.push_back(l2error);
         if (j > 0)
         {
            if (Mpi::Root())
            {
               printf("p=%d L2rate=%.2E\n", orders[i],
                      log(l2errors[j-1]/l2errors[j])/log(2.0));
            }
         }
      }
   }

   return 0;
}