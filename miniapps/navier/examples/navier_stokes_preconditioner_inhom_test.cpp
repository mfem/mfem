#include <mfem.hpp>
#include <fstream>
#include <iostream>
#include <block_schur_pc.hpp>
#include <schur_pcd.hpp>
#include <util.hpp>

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

double nu = 1.0;

void forcing(const Vector &c, Vector &f)
{
   double x = c(0);
   double y = c(1);

   f(0) = nu*cos(y)+1;
   f(1) = nu*sin(x)+1;
}

void velocity_mms(const Vector &c, Vector &u)
{
   double x = c(0);
   double y = c(1);

   u(0) = cos(y);
   u(1) = sin(x);
}

double pressure_mms(const Vector &c)
{
   double x = c(0);
   double y = c(1);

   return y+x;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../data/star.mesh";

   int order = 2;
   int ser_ref = 0;
   int par_ref = 0;
   double dt = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&dt, "-dt", "--dt", "Time step");
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

   Array<int> vel_ess_dofs;
   Array<int> pres_ess_dofs;
   vel_fes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_dofs);
   pres_fes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_dofs);

   ConstantCoefficient dt_inv_coeff(dt > 0 ? 1.0/dt : 0.0);
   ConstantCoefficient nu_coeff(nu);
   ConstantCoefficient zero_coeff(0.0);

   ParBilinearForm F_form(&vel_fes);
   F_form.AddDomainIntegrator(new VectorMassIntegrator(dt_inv_coeff));
   F_form.AddDomainIntegrator(new VectorDiffusionIntegrator(nu_coeff));
   // F_form.AddDomainIntegrator(new ElasticityIntegrator(zero_coeff, nu_coeff));
   F_form.SetDiagonalPolicy(Operator::DIAG_ONE);
   F_form.Assemble();
   F_form.Finalize();

   OperatorHandle F;
   F_form.FormSystemMatrix(vel_ess_dofs, F);

   ConstantCoefficient minus_one(-1.0);

   ParMixedBilinearForm d_form(&vel_fes, &pres_fes);
   d_form.AddDomainIntegrator(new VectorDivergenceIntegrator(&minus_one));
   d_form.Assemble();
   d_form.Finalize();

   OperatorHandle D;
   d_form.FormRectangularSystemMatrix(vel_ess_dofs, pres_ess_dofs, D);

   TransposeOperator Dt(*D);

   Array<int> offsets({0, vel_fes.GetTrueVSize(), pres_fes.GetTrueVSize()});
   offsets.PartialSum();

   BlockOperator A(offsets);
   A.SetBlock(0, 0, F.Ptr());
   A.SetBlock(0, 1, &Dt);
   A.SetBlock(1, 0, D.Ptr());

   // [  A   -D^t ] [ u ] = [ f ]
   // [ -D     0  ] [ p ]   [ 0 ]

   // HypreBoomerAMG F_pc(*F.As<HypreParMatrix>());
   // F_pc.SetSystemsOptions(dim);
   // CGSolver F_inv(MPI_COMM_WORLD);
   // F_inv.SetRelTol(1e-6);
   // F_inv.SetMaxIter(50);
   // F_inv.SetOperator(*F.As<HypreParMatrix>());
   // F_inv.SetPreconditioner(F_pc);
   SparseMatrix F_local;
   F.As<HypreParMatrix>()->GetDiag(F_local);
   UMFPackSolver F_inv(F_local);

   PCDBuilder pcd_builder(pres_fes, pres_ess_dofs, &dt_inv_coeff, &nu_coeff);
   BlockSchurPC P(offsets, F_inv, *D.Ptr(), pcd_builder.GetSolver(), vel_ess_dofs);

   FGMRESSolver krylov(MPI_COMM_WORLD);
   krylov.iterative_mode = true;
   krylov.SetRelTol(1e-8);
   krylov.SetMaxIter(2000);
   krylov.SetKDim(2000);
   krylov.SetPrintLevel(1);
   krylov.SetOperator(A);
   krylov.SetPreconditioner(P);

   VectorFunctionCoefficient velocity_mms_coeff(dim, velocity_mms);
   FunctionCoefficient pressure_mms_coeff(pressure_mms);
   VectorFunctionCoefficient forcing_coeff(dim, forcing);

   ParGridFunction u_gf(&vel_fes), uex_gf(&vel_fes), p_gf(&pres_fes),
                   pex_gf(&pres_fes), uerr_gf(&vel_fes), perr_gf(&pres_fes);
   u_gf = 0.0;
   u_gf.ProjectBdrCoefficient(velocity_mms_coeff, vel_ess_bdr);
   uex_gf.ProjectCoefficient(velocity_mms_coeff);
   pex_gf.ProjectCoefficient(pressure_mms_coeff);

   auto forcing_form = new ParLinearForm(&vel_fes);
   forcing_form->AddDomainIntegrator(new VectorDomainLFIntegrator(forcing_coeff));
   forcing_form->Assemble();
   Vector rhs_u;
   forcing_form->ParallelAssemble(rhs_u);

   BlockVector X(offsets), B(offsets);

   // prepare initial condition
   u_gf.ParallelProject(X.GetBlock(0));
   X.GetBlock(1) = 0.0;

   // prepare RHS
   B.GetBlock(0) = rhs_u;
   B.GetBlock(1) = 0.0;

   // b -= Ae*x
   F_form.GetAe()->AddMult(X.GetBlock(0), B.GetBlock(0), -1.0);
   d_form.GetAe()->AddMult(X.GetBlock(0), B.GetBlock(1), -1.0);

   PrintJulia(*F_form.GetAe(), "Fe_A.dat");

   // set ess dofs
   for (int i : vel_ess_dofs)
   {
      B.GetBlock(0)[i] = u_gf[i];
   }

   // exit(0);

   krylov.Mult(B, X);

   u_gf.Distribute(X.GetBlock(0));
   p_gf.Distribute(X.GetBlock(1));

   double u_l2err = u_gf.ComputeL2Error(velocity_mms_coeff);
   double p_l2err = p_gf.ComputeL2Error(pressure_mms_coeff);

   printf("ul2err = %.5E\n", u_l2err);
   printf("pl2err = %.5E\n", p_l2err);

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
