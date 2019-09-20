#include "flow_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace flow;

struct s_FlowContext
{
   int order = 7;
   double kin_vis = 1.0 / 1600.0;
   double t_final = 10e-3;
   double dt = 1e-3;
   bool pa = false;
   bool ni = false;
} ctx;

void vel_tgv(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);

   u(0) = sin(xi) * cos(yi) * cos(zi);
   u(1) = -cos(xi) * sin(yi) * cos(zi);
   u(2) = 0.0;
}

class QOI
{
public:
   QOI(ParMesh *pmesh)
   {
      H1_FECollection h1fec(1);
      ParFiniteElementSpace h1fes(pmesh, &h1fec);

      onecoeff.constant = 1.0;
      mass_lf = new ParLinearForm(&h1fes);
      mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
      mass_lf->Assemble();

      ParGridFunction one_gf(&h1fes);
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   };

   double ComputeKineticEnergy(ParGridFunction &v)
   {
      Vector velx, vely, velz;
      double integ = 0.0;
      const FiniteElement *fe;
      ElementTransformation *T;
      FiniteElementSpace *fes = v.FESpace();

      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         double intorder = 2 * fe->GetOrder();
         const IntegrationRule *ir = &(
            IntRules.Get(fe->GetGeomType(), intorder));

         v.GetValues(i, *ir, velx, 1);
         v.GetValues(i, *ir, vely, 2);
         v.GetValues(i, *ir, velz, 3);

         T = fes->GetElementTransformation(i);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);

            double vel2 = velx(j) * velx(j) + vely(j) * vely(j)
                          + velz(j) * velz(j);

            integ += ip.weight * T->Weight() * vel2;
         }
      }

      double global_integral = 0.0;
      MPI_Allreduce(&integ,
                    &global_integral,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      return 0.5 * global_integral / volume;
   };

   ~QOI() { delete mass_lf; };

private:
   ConstantCoefficient onecoeff;
   ParLinearForm *mass_lf;
   double volume;
};

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int ser_ref_levels = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.pa,
                  "-pa",
                  "--enable-pa",
                  "-no-pi",
                  "--disable-pi",
                  "Enable partial assembly.");
   args.AddOption(&ctx.ni,
                  "-ni",
                  "--enable-ni",
                  "-no-ni",
                  "--disable-ni",
                  "Enable numerical integration rules.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(std::cout);
   }

   Mesh *orig_mesh = new Mesh("../../data/periodic-cube.mesh");
   Mesh *mesh = new Mesh(orig_mesh, ser_ref_levels, BasisType::ClosedUniform);
   delete orig_mesh;

   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   *nodes *= M_PI;

   int nel = mesh->GetNE();
   if (mpi.Root())
   {
      std::cout << "Number of elements: " << nel << std::endl;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   FlowSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);
   flowsolver.EnablePA(ctx.pa);
   flowsolver.EnableNI(ctx.ni);

   // Set the initial condition.
   // This is completely user customizeable.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv);
   u_ic->ProjectCoefficient(u_excoeff);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   QOI kin_energy(pmesh);

   VisItDataCollection visit_dc("ins", pmesh);
   visit_dc.SetPrefixPath("output");
   visit_dc.SetCycle(0);
   visit_dc.SetTime(t);
   visit_dc.RegisterField("velocity", u_gf);
   visit_dc.RegisterField("pressure", p_gf);
   visit_dc.Save();

   double u_inf_loc = u_gf->Normlinf();
   double p_inf_loc = p_gf->Normlinf();
   double u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
   double p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);
   double ke = kin_energy.ComputeKineticEnergy(*u_gf);

   FILE *f = fopen("tgv_out.txt", "w");
   if (mpi.Root())
   {
      int nel1d = std::round(pow(nel, 1.0/3.0));
      int ngridpts = p_gf->ParFESpace()->GlobalVSize();
      printf("%.5E %.5E %.5E %.5E %.5E\n", t, dt, u_inf, p_inf, ke);
      fprintf(f, "3D Taylor Green Vortex\n");
      fprintf(f, "order = %d\n", ctx.order);
      fprintf(f, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
      fprintf(f, "dofs per component = %d\n", ngridpts);
      fprintf(f, "=================================================\n");
      fprintf(f, "        time                   kinetic energy\n");
      fprintf(f, "%20.16e     %20.16e\n", t, ke);
      fflush(f);
      fflush(stdout);
   }

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      if ((step + 1) % 100 == 0 || last_step)
      {
         visit_dc.SetCycle(step);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }

      double u_inf_loc = u_gf->Normlinf();
      double p_inf_loc = p_gf->Normlinf();
      double u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
      double p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);
      double ke = kin_energy.ComputeKineticEnergy(*u_gf);
      if (mpi.Root())
      {
         printf("%.5E %.5E %.5E %.5E %.5E\n", t, dt, u_inf, p_inf, ke);
         fprintf(f, "%20.16e     %20.16e\n", t, ke);
         fflush(f);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}
