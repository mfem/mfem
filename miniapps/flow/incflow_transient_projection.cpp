#include "mfem.hpp"
#include "vec_conv_integrator.hpp"

using namespace mfem;
using namespace std;

void vel_ldc(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   if (yi == 1.0)
   {
      u(0) = 1.0;
   }
   else
   {
      u(0) = 0.0;
   }
   u(1) = 0.0;
}

void test_bdr(const Vector &x, Vector &u)
{
   u(0) = 100.0;
   u(1) = 200.0;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = mesh_file = "../../data/inline-quad.mesh";

   int serial_ref_levels = 0;

   int order = 2;
   int vel_order = order;
   int pres_order = order - 1;

   double Re = 100.0;

   double t = 0.0;
   double dt = 1e-5;
   double t_final = dt;

   OptionsParser args(argc, argv);
   args.AddOption(&serial_ref_levels,
                  "-rs",
                  "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&dt,
                  "-dt",
                  "--dt",
                  ".");
   args.AddOption(&t_final,
                  "-tf",
                  "--final-time",
                  ".");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   Mesh *mesh = new Mesh(mesh_file);
   int dim = mesh->Dimension();

   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   FiniteElementCollection *vel_fec = new H1_FECollection(vel_order, dim);
   FiniteElementCollection *pres_fec = new H1_FECollection(pres_order);

   ParFiniteElementSpace *vel_fes = new ParFiniteElementSpace(pmesh,
                                                              vel_fec,
                                                              dim);
   ParFiniteElementSpace *pres_fes = new ParFiniteElementSpace(pmesh, pres_fec);

   int fes_size0 = vel_fes->GlobalVSize();
   int fes_size1 = pres_fes->GlobalVSize();

   if (myid == 0)
   {
      cout << "Velocity #DOFs: " << fes_size0 << endl;
      cout << "Pressure #DOFs: " << fes_size1 << endl;
   }

   Array<int> ess_tdof_list;
   Array<int> ess_bdr_attr(pmesh->bdr_attributes.Max());
   ess_bdr_attr = 1;

   vel_fes->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);

   Vector un(vel_fes->GetTrueVSize());
   Vector uh(vel_fes->GetTrueVSize());
   Vector uhh(vel_fes->GetTrueVSize());

   un = 0.0;
   uh = 0.0;
   uhh = 0.0;

   ParGridFunction uh_gf(vel_fes);
   uh_gf = 0.0;

   Vector pn(pres_fes->GetTrueVSize());
   pn = 0.0;

   ParGridFunction p_gf(pres_fes);
   p_gf = 0.0;

   ParGridFunction bdr_projector(vel_fes);
   VectorFunctionCoefficient test_bdr_coeff(dim, test_bdr);

   // Apply boundary conditions to IC
   VectorFunctionCoefficient u_bdr_coeff(dim, vel_ldc);
   ParGridFunction u_gf(vel_fes);
   u_gf = 0.0;
   u_gf.ProjectBdrCoefficient(u_bdr_coeff, ess_bdr_attr);
   u_gf.GetTrueDofs(un);

   ConstantCoefficient nl_coeff(-1.0 * Re);
   ParNonlinearForm *N = new ParNonlinearForm(vel_fes);
   N->AddDomainIntegrator(new VectorConvectionNLFIntegrator(nl_coeff));
   N->SetEssentialTrueDofs(ess_tdof_list);

   ParBilinearForm *Mv_form = new ParBilinearForm(vel_fes);
   Mv_form->AddDomainIntegrator(new VectorMassIntegrator);
   Mv_form->Assemble();
   Mv_form->Finalize();
   HypreParMatrix Mv = *Mv_form->ParallelAssemble();
   Mv_form->FormSystemMatrix(ess_tdof_list, Mv);

   ParBilinearForm *Sp_form = new ParBilinearForm(pres_fes);
   Sp_form->AddDomainIntegrator(new DiffusionIntegrator);
   Sp_form->Assemble();
   Sp_form->Finalize();
   HypreParMatrix Sp = *Sp_form->ParallelAssemble();

   ParMixedBilinearForm *D_form = new ParMixedBilinearForm(vel_fes, pres_fes);
   D_form->AddDomainIntegrator(new VectorDivergenceIntegrator);
   D_form->Assemble();
   D_form->Finalize();
   HypreParMatrix D = *D_form->ParallelAssemble();

   // G is -D^T !!!
   HypreParMatrix G = *D.Transpose();

   ParLinearForm *g_bdr_form = new ParLinearForm(pres_fes);
   g_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(u_bdr_coeff));
   g_bdr_form->Assemble();

   VectorGridFunctionCoefficient uh_gf_coeff(&uh_gf);
   ParLinearForm *uh_bdr_form = new ParLinearForm(pres_fes);
   uh_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(uh_gf_coeff));

   ConstantCoefficient lincoeff(dt);
   ParBilinearForm *H_form = new ParBilinearForm(vel_fes);
   H_form->AddDomainIntegrator(new VectorMassIntegrator);
   H_form->AddDomainIntegrator(new VectorDiffusionIntegrator(lincoeff));
   H_form->Assemble();
   HypreParMatrix H;
   H_form->FormSystemMatrix(ess_tdof_list, H);

   HypreDiagScale MvInvPC(Mv);
   CGSolver MvInv(MPI_COMM_WORLD);
   MvInv.SetOperator(Mv);
   MvInv.SetPreconditioner(MvInvPC);
   MvInv.SetPrintLevel(2);
   MvInv.SetRelTol(1e-8);
   MvInv.SetMaxIter(50);

   HypreBoomerAMG SpInvPC = HypreBoomerAMG(Sp);
   SpInvPC.SetPrintLevel(0);
   CGSolver SpInv(MPI_COMM_WORLD);
   SpInv.SetPreconditioner(SpInvPC);
   SpInv.SetOperator(Sp);
   SpInv.SetPrintLevel(2);
   SpInv.SetRelTol(1e-8);
   SpInv.SetMaxIter(50);

   HypreBoomerAMG HInvPC = HypreBoomerAMG(H);
   HInvPC.SetPrintLevel(0);
   CGSolver HInv(MPI_COMM_WORLD);
   HInv.SetPreconditioner(HInvPC);
   HInv.SetOperator(H);
   HInv.SetPrintLevel(2);
   HInv.SetRelTol(1e-8);
   HInv.SetMaxIter(50);

   char vishost[] = "localhost";
   int visport = 19916;

   socketstream u_sock(vishost, visport);
   u_sock.precision(8);

   socketstream p_sock(vishost, visport);
   p_sock.precision(8);

   u_gf.Distribute(un);
   u_sock << "parallel " << num_procs << " " << myid << "\n"
          << "solution\n"
          << *pmesh << u_gf << "window_title 'velocity'"
          << "keys rRljc\n"
          << endl;

   Vector tmp(vel_fes->GetTrueVSize());
   Vector tmp2(pres_fes->GetTrueVSize());
   Vector tmp3(vel_fes->GetTrueVSize());
   Vector X, B;

   bool last_step = false;

   for (int step = 0; !last_step; ++step)
   {
      if (mpi.Root())
      {
         cout << "\nTime: " << t << endl;
      }
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      if (mpi.Root())
      {
         cout << "\nExtrapolation" << endl;
      }

      N->Mult(un, tmp);
      tmp *= dt;
      MvInv.Mult(tmp, uh);
      uh += un;

      if (mpi.Root())
      {
         cout << "\nPressure poisson" << endl;
      }

      uh_gf.SetFromTrueDofs(uh);
      uh_bdr_form->Assemble();
      g_bdr_form->Assemble();

      D.Mult(uh, tmp2);

      tmp2 *= -1.0;
      tmp2 -= *g_bdr_form;
      tmp2 += *uh_bdr_form;
      tmp2 *= 1.0 / dt;

      SpInv.Mult(tmp2, pn);

      if (mpi.Root())
      {
         cout << "\nProjection" << endl;
      }

      G.Mult(pn, tmp);
      tmp *= dt;
      MvInv.Mult(tmp, uhh);
      // uhh *= -1.0;
      uhh += uh;

      if (mpi.Root())
      {
         cout << "\nHelmholtz" << endl;
      }

      Mv.Mult(uhh, tmp);
      H_form->FormLinearSystem(ess_tdof_list, un, tmp, H, X, B);
      HInv.Mult(B, X);
      H_form->RecoverFEMSolution(X, tmp, un);

      t += dt;

      u_gf.Distribute(un);
      u_sock << "parallel " << num_procs << " " << myid << "\n"
             << "solution\n"
             << *pmesh << u_gf << "window_title 'velocity'"
             << endl;

      p_gf.Distribute(pn);
      p_sock << "parallel " << num_procs << " " << myid << "\n"
             << "solution\n"
             << *pmesh << p_gf << "window_title 'pressure'"
             << endl;
   }

   return 0;
}
