#include <mfem.hpp>
#include <navierstokes_operator.hpp>
#include <block_schur_pc.hpp>
#include <schur_pcd.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class BDFIMEX : public ODESolver
{
protected:
   Vector xnm1, xnm2, fxn;
   double h;
   int bootstrap = 0;

public:
   BDFIMEX() {};

   void Init(TimeDependentOperator &f_) override
   {
      ODESolver::Init(f_);
      xnm1.SetSize(f->Width(), mem_type);
      xnm2.SetSize(f->Width(), mem_type);
      fxn.SetSize(f->Width(), mem_type);
   }

   void Step(Vector &x, double &t, double &dt) override
   {
      if (bootstrap == 0)
      {
         xnm1 = x;
         bootstrap++;
      }

      h = dt;

      f->SetTime(t + dt);

      f->ImplicitSolve(h, xnm1, x);

      xnm2 = xnm1;
      xnm1 = x;
      t += dt;
   }
};

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

double nu = 1.0;

void forcing(const Vector &c, const double t, Vector &f)
{
   double x = c(0);
   double y = c(1);

   // f(0) = -pow(cos(t),2.0)*sin(x)*sin(y)+(-sin(t)+nu*cos(t))*cos(y)+cos(t);
   // f(1) = pow(cos(t),2.0)*cos(x)*cos(y)+(-sin(t)+nu*cos(t))*sin(x)+cos(t);
   f(0) = (-sin(t)+nu*cos(t))*cos(y)+cos(t);
   f(1) = (-sin(t)+nu*cos(t))*sin(x)+cos(t);
}

void velocity_mms(const Vector &c, const double t, Vector &u)
{
   double x = c(0);
   double y = c(1);

   u(0) = cos(t)*cos(y);
   u(1) = cos(t)*sin(x);
}

double pressure_mms(const Vector &c, const double t)
{
   double x = c(0);
   double y = c(1);

   return cos(t)*(y+x) - 1.0;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../data/inline-quad.mesh";

   int order = 2;
   int ser_ref = 0;
   int par_ref = 0;
   double dt = 0.0;
   double tf = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&dt, "-dt", "--dt", "Time step");
   args.AddOption(&tf, "-tf", "--tf", "final time");
   args.AddOption(&nu, "-nu", "--nu", "kinematic viscosity");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.ParseCheck();

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   const int dim = mesh.Dimension();
   H1_FECollection vel_fec(order, dim);
   H1_FECollection pres_fec(order - 1, dim);

   ParFiniteElementSpace vel_fes(&mesh, &vel_fec, dim);
   ParFiniteElementSpace pres_fes(&mesh, &pres_fec);

   printf("#velocity_dofs: %d\n", vel_fes.GetTrueVSize());

   Array<int> vel_ess_bdr, pres_ess_bdr;

   bool cyl = mesh.bdr_attributes.Size() >= 7;

   vel_ess_bdr.SetSize(mesh.bdr_attributes.Max());
   vel_ess_bdr = 1;
   if (cyl) { vel_ess_bdr[1] = 0; }

   pres_ess_bdr = vel_ess_bdr;
   for (int &marker : pres_ess_bdr) { marker = !marker; }

   if (cyl)
   {
      vel_ess_bdr[4] = 0;
      pres_ess_bdr[4] = 0;
   }

   VectorFunctionCoefficient velocity_mms_coeff(dim, velocity_mms);
   FunctionCoefficient pressure_mms_coeff(pressure_mms);
   VectorFunctionCoefficient forcing_coeff(dim, forcing);

   ParGridFunction u_gf(&vel_fes), uex_gf(&vel_fes), p_gf(&pres_fes),
                   pex_gf(&pres_fes);
   u_gf = 0.0;
   u_gf.ProjectCoefficient(velocity_mms_coeff);
   p_gf.ProjectCoefficient(pressure_mms_coeff);
   // u_gf.ProjectBdrCoefficient(velocity_mms_coeff, vel_ess_bdr);
   uex_gf.ProjectCoefficient(velocity_mms_coeff);
   pex_gf.ProjectCoefficient(pressure_mms_coeff);

   ParGridFunction nu_gf(&pres_fes);
   nu_gf = nu;

   NavierStokesOperator navier(vel_fes, pres_fes, vel_ess_bdr, pres_ess_bdr, nu_gf,
                               true, false, true);
   navier.SetForcing(&forcing_coeff);
   navier.Setup(dt);

   BlockVector X(navier.GetOffsets());
   X = 0.0;

   BDFIMEX ode;

   ode.Init(navier);

   double t = 0.0;
   int num_steps = 0;

   while (t < tf)
   {
      if (t + dt > tf) { dt = tf - t; }

      if (Mpi::Root())
      {
         std::cout << "Step " << std::left << std::setw(5) << ++num_steps
                   << std::setprecision(2) << std::scientific
                   << " t = " << t
                   << " dt = " << dt
                   << "\n" << std::endl;
      }

      velocity_mms_coeff.SetTime(t + dt);
      pressure_mms_coeff.SetTime(t + dt);

      uex_gf.ProjectCoefficient(velocity_mms_coeff);
      pex_gf.ProjectCoefficient(pressure_mms_coeff);

      u_gf.ProjectBdrCoefficient(velocity_mms_coeff, vel_ess_bdr);
      u_gf.ParallelProject(X.GetBlock(0));

      ode.Step(X, t, dt);

      u_gf.Distribute(X.GetBlock(0));
      p_gf.Distribute(X.GetBlock(1));

      double vel_l2_err = u_gf.ComputeL2Error(velocity_mms_coeff);
      double pres_l2_err = p_gf.ComputeL2Error(pressure_mms_coeff);

      printf("u_l2err = %.5E\np_l2err = %.5E\n", vel_l2_err, pres_l2_err);

      std::cout << std::endl;
   }

   u_gf.Distribute(X.GetBlock(0));
   p_gf.Distribute(X.GetBlock(1));

   ParGridFunction uerr_gf(&vel_fes), perr_gf(&pres_fes);

   for (int i = 0; i < uerr_gf.Size(); i++)
   {
      uerr_gf(i) = abs(u_gf(i) - uex_gf(i));
   }

   for (int i = 0; i < perr_gf.Size(); i++)
   {
      perr_gf(i) = abs(p_gf(i) - pex_gf(i));
   }

   ParaViewDataCollection paraview_dc("fluid_output", &mesh);
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&u_gf);
   paraview_dc.RegisterField("pressure",&p_gf);
   paraview_dc.RegisterField("velocity_exact",&uex_gf);
   paraview_dc.RegisterField("pressure_exact",&pex_gf);
   paraview_dc.RegisterField("velocity_error",&uerr_gf);
   paraview_dc.RegisterField("pressure_error",&perr_gf);
   paraview_dc.Save();

   return 0;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}
