#include <mfem.hpp>
#include <navierstokes_operator.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

double nu = 1.0;

void vel_shear_ic(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double rho = 30.0;
   double delta = 0.05;

   if (yi <= 0.5)
   {
      u(0) = tanh(rho * (yi - 0.25));
   }
   else
   {
      u(0) = tanh(rho * (0.75 - yi));
   }

   u(1) = delta * sin(2.0 * M_PI * xi);
}

void forcing_mms_1(const Vector &c, const double t, Vector &f)
{
   double x = c(0);
   double y = c(1);

   f(0) = (-sin(t)+nu*cos(t))*cos(y)+cos(t);
   f(1) = (-sin(t)+nu*cos(t))*sin(x)+cos(t);
}

void velocity_mms_1(const Vector &c, const double t, Vector &u)
{
   double x = c(0);
   double y = c(1);

   u(0) = cos(t)*cos(y);
   u(1) = cos(t)*sin(x);
}

double pressure_mms_1(const Vector &c, const double t)
{
   double x = c(0);
   double y = c(1);

   return cos(t)*(y+x);
}

void forcing_mms_2(const Vector &c, const double t, Vector &f)
{
   double x = c(0);
   double y = c(1);

   f(0) = 3*pow(t,2.0)*pow(y,2.0)+2*pow(t,5.0)*x*y-2*nu*pow(t,3.0)+t;
   f(1) = pow(t,5.0)*pow(y,2.0)+2*t*x+1;
}

void velocity_mms_2(const Vector &c, const double t, Vector &u)
{
   double x = c(0);
   double y = c(1);

   u(0) = pow(t,3.0)*pow(y,2.0);
   u(1) = pow(t,2.0)*x;
}

double pressure_mms_2(const Vector &c, const double t)
{
   double x = c(0);
   double y = c(1);

   return y+t*x+(-t-1)/2.0;
}

void forcing_mms_3(const Vector &c, const double t, Vector &f)
{
   double x = c(0);
   double y = c(1);

   f(0) = 0.0;
   f(1) = 0.0;
}

void velocity_mms_3(const Vector &c, const double t, Vector &u)
{
   double x = c(0);
   double y = c(1);

   u(0) = exp(-2.0*nu*t)*cos(x)*sin(y);
   u(1) = -exp(-2.0*nu*t)*sin(x)*cos(y);
}

double pressure_mms_3(const Vector &c, const double t)
{
   double x = c(0);
   double y = c(1);

   return -1.0/4.0*exp(-4.0*nu*t)*(cos(2*y)+cos(2*x));
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
   int problem_type = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&problem_type, "-prob", "--problem", "Problem type");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&dt, "-dt", "--dt", "Time step");
   args.AddOption(&tf, "-tf", "--tf", "final time");
   args.AddOption(&nu, "-nu", "--nu", "kinematic viscosity");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.ParseCheck();

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   if (problem_type == 4) {
      mesh.EnsureNodes();
      GridFunction *nodes = mesh.GetNodes();
      *nodes -= -1.0;
      *nodes /= 2.0;
   }

   const int dim = mesh.Dimension();
   H1_FECollection vel_fec(order, dim);
   H1_FECollection pres_fec(order - 1, dim);

   ParFiniteElementSpace vel_fes(&mesh, &vel_fec, dim);
   ParFiniteElementSpace pres_fes(&mesh, &pres_fec);

   HYPRE_BigInt global_velocity_tdofs = vel_fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      printf("#velocity_dofs: %d\n", global_velocity_tdofs);
   }

   Array<int> vel_ess_bdr, pres_ess_bdr;

   if (mesh.bdr_attributes.Size() > 0) {
      vel_ess_bdr.SetSize(mesh.bdr_attributes.Max());
   }

   if (problem_type == 4) {
      vel_ess_bdr = 0;
   } else {
      vel_ess_bdr = 1;
   }

   pres_ess_bdr = vel_ess_bdr;
   if (problem_type == 4) {
      pres_ess_bdr = 0;
   } else {
      for (int &marker : pres_ess_bdr) { marker = !marker; }
   }

   VectorCoefficient *velocity_mms_coeff = nullptr;
   Coefficient *pressure_mms_coeff = nullptr;
   VectorCoefficient *forcing_coeff = nullptr;

   if (problem_type == 1) {
      out << "problem type 1" << std::endl;
      velocity_mms_coeff = new VectorFunctionCoefficient(dim, velocity_mms_1);
      pressure_mms_coeff = new FunctionCoefficient(pressure_mms_1);
      forcing_coeff = new VectorFunctionCoefficient(dim, forcing_mms_1);
   } else if (problem_type == 2) {
      out << "problem type 2" << std::endl;
      velocity_mms_coeff = new VectorFunctionCoefficient(dim, velocity_mms_2);
      pressure_mms_coeff = new FunctionCoefficient(pressure_mms_2);
      forcing_coeff = new VectorFunctionCoefficient(dim, forcing_mms_2);
   } else if (problem_type == 3) {
      out << "problem type 3" << std::endl;
      velocity_mms_coeff = new VectorFunctionCoefficient(dim, velocity_mms_3);
      pressure_mms_coeff = new FunctionCoefficient(pressure_mms_3);
      forcing_coeff = new VectorFunctionCoefficient(dim, forcing_mms_3);
   } else if (problem_type == 4) {
      out << "problem type 4" << std::endl;
      velocity_mms_coeff = new VectorFunctionCoefficient(dim, vel_shear_ic);
      pressure_mms_coeff = new ConstantCoefficient(0.0);
      forcing_coeff = new VectorFunctionCoefficient(dim, forcing_mms_3);
   }

   ParGridFunction u_gf(&vel_fes), uex_gf(&vel_fes), p_gf(&pres_fes),
                   pex_gf(&pres_fes);
   u_gf = 0.0;
   u_gf.ProjectCoefficient(*velocity_mms_coeff);
   p_gf.ProjectCoefficient(*pressure_mms_coeff);

   ParGridFunction nu_gf(&pres_fes);
   nu_gf = nu;

   std::vector<VelDirichletBC> vel_dbcs;
   vel_dbcs.emplace_back(std::make_pair(velocity_mms_coeff, &vel_ess_bdr));

   NavierStokesOperator navier(vel_fes, pres_fes, vel_dbcs, pres_ess_bdr, nu_gf,
                               true, false, true);

   navier.SetForcing(forcing_coeff);

   BlockVector X(navier.GetOffsets());
   X = 0.0;

   // Set initial condition
   double t = 0.0;
   int num_steps = 0;

   u_gf.ParallelProject(X.GetBlock(0));
   p_gf.ParallelProject(X.GetBlock(1));

   velocity_mms_coeff->SetTime(t);
   uex_gf.ProjectCoefficient(*velocity_mms_coeff);

   pressure_mms_coeff->SetTime(t);
   pex_gf.ProjectCoefficient(*pressure_mms_coeff);

   ParGridFunction uerr_gf(&vel_fes), perr_gf(&pres_fes);
   ParaViewDataCollection paraview_dc("fluid_output", &mesh);
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.RegisterField("velocity",&u_gf);
   paraview_dc.RegisterField("pressure",&p_gf);
   paraview_dc.RegisterField("velocity_exact",&uex_gf);
   paraview_dc.RegisterField("pressure_exact",&pex_gf);
   paraview_dc.RegisterField("velocity_error",&uerr_gf);
   paraview_dc.RegisterField("pressure_error",&perr_gf);

   auto save_callback = [&](int cycle, double t)
   {
      paraview_dc.SetCycle(cycle);
      paraview_dc.SetTime(t);

      for (int i = 0; i < uerr_gf.Size(); i++)
      {
         uerr_gf(i) = abs(u_gf(i) - uex_gf(i));
      }

      for (int i = 0; i < perr_gf.Size(); i++)
      {
         perr_gf(i) = abs(p_gf(i) - pex_gf(i));
      }

      paraview_dc.Save();
   };

   save_callback(num_steps, t);

   while (t < tf)
   {
      // if (t + dt > tf) { dt = tf - t; }

      if (Mpi::Root())
      {
         std::cout << "Step " << std::left << std::setw(5) << ++num_steps
                   << std::setprecision(2) << std::scientific
                   << " t = " << t
                   << " dt = " << dt
                   << "\n" << std::endl;
      }

      // velocity_mms_coeff->SetTime(t+dt);
      // u_gf.ProjectBdrCoefficient(*velocity_mms_coeff, vel_ess_bdr);
      // u_gf.ParallelProject(X.GetBlock(0));

      navier.Step(X, t, dt);

      u_gf.Distribute(X.GetBlock(0));
      p_gf.Distribute(X.GetBlock(1));

      velocity_mms_coeff->SetTime(t);
      uex_gf.ProjectCoefficient(*velocity_mms_coeff);
      double vel_l2_err = u_gf.ComputeL2Error(*velocity_mms_coeff);

      pressure_mms_coeff->SetTime(t);
      pex_gf.ProjectCoefficient(*pressure_mms_coeff);
      double pres_l2_err = p_gf.ComputeL2Error(*pressure_mms_coeff);

      if (Mpi::Root()) {
         printf("u_l2err = %.5E\np_l2err = %.5E\n", vel_l2_err, pres_l2_err);
      }

      if (num_steps % 10 == 0) {
         save_callback(num_steps, t);
      }

      std::cout << std::endl;
   }

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
