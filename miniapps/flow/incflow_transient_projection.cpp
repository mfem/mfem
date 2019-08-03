#include "fstream"
#include "mfem.hpp"
#include "vec_conv_integrator.hpp"
#include "vec_grad_integrator.hpp"

using namespace mfem;
using namespace std;

#define KINVIS 1.0

void vel_ex(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double F = exp(-2.0 * KINVIS * t);

   u(0) = cos(xi) * sin(yi) * F;
   u(1) = -sin(xi) * cos(yi) * F;
}

double p_ex(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);

   double F = exp(-2.0 * KINVIS * t);

   return -1.0 / 4.0 * (cos(2.0 * xi) + cos(2.0 * yi)) * pow(F, 2.0);
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = "../../data/inline-quad.mesh";

   int serial_ref_levels = 0;

   int order = 6;
   int vel_order = order;
   int pres_order = order;

   double kinvis = KINVIS;

   double t = 0.0;
   double dt = 1e-5;
   double t_final = dt;

   OptionsParser args(argc, argv);
   args.AddOption(&serial_ref_levels,
                  "-rs",
                  "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&kinvis, "-kv", "--kinvis", ".");
   args.AddOption(&dt, "-dt", "--dt", ".");
   args.AddOption(&t_final, "-tf", "--final-time", ".");
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
   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   *nodes *= M_PI;

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

   Array<int> ess_tdof_list_u;
   Array<int> ess_bdr_attr_u(pmesh->bdr_attributes.Max());
   ess_bdr_attr_u[0] = 1;
   ess_bdr_attr_u[1] = 1;
   ess_bdr_attr_u[2] = 1;
   ess_bdr_attr_u[3] = 1;
   vel_fes->GetEssentialTrueDofs(ess_bdr_attr_u, ess_tdof_list_u);

   Array<int> ess_tdof_list_p;
   Array<int> ess_bdr_attr_p(pmesh->bdr_attributes.Max());
   ess_bdr_attr_p[0] = 0;
   ess_bdr_attr_p[1] = 0;
   ess_bdr_attr_p[2] = 0;
   ess_bdr_attr_p[3] = 0;
   pres_fes->GetEssentialTrueDofs(ess_bdr_attr_p, ess_tdof_list_p);

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

   VectorFunctionCoefficient u_ex_coeff(dim, vel_ex);
   ParGridFunction u_gf(vel_fes);
   u_gf = 0.0;
   u_gf.ProjectCoefficient(u_ex_coeff);
   u_gf.GetTrueDofs(un);

   FunctionCoefficient p_ex_coeff(p_ex);
   ParGridFunction p_ex_gf(pres_fes);
   p_ex_gf.ProjectCoefficient(p_ex_coeff);

   ConstantCoefficient nlcoeff(-1.0);
   ParNonlinearForm *N = new ParNonlinearForm(vel_fes);
   N->AddDomainIntegrator(new VectorConvectionNLFIntegrator(nlcoeff));

   ParBilinearForm *Mv_form = new ParBilinearForm(vel_fes);
   VectorMassIntegrator *blfi = new VectorMassIntegrator;
   //   const IntegrationRule *ir
   //      = &IntRules.Get(vel_fes->GetFE(0)->GetGeomType(),
   //                      vel_fes->GetElementTransformation(0)->OrderW()
   //                         + 2 * (order + 1));
   //   blfi->SetIntRule(ir);
   Mv_form->AddDomainIntegrator(blfi);
   Mv_form->Assemble();
   Mv_form->Finalize();
   HypreParMatrix Mv = *Mv_form->ParallelAssemble();
   Array<int> empty;
   Mv_form->FormSystemMatrix(empty, Mv);

   ParBilinearForm *Sp_form = new ParBilinearForm(pres_fes);
   Sp_form->AddDomainIntegrator(new DiffusionIntegrator);
   Sp_form->Assemble();
   Sp_form->Finalize();
   HypreParMatrix Sp;
   Sp_form->FormSystemMatrix(ess_tdof_list_p, Sp);

   ParMixedBilinearForm *D_form = new ParMixedBilinearForm(vel_fes, pres_fes);
   D_form->AddDomainIntegrator(new VectorDivergenceIntegrator);
   D_form->Assemble();
   D_form->Finalize();
   HypreParMatrix *D = D_form->ParallelAssemble();

   ParMixedBilinearForm *G_form = new ParMixedBilinearForm(pres_fes, vel_fes);
   G_form->AddDomainIntegrator(new VectorGradientIntegrator);
   G_form->Assemble();
   G_form->Finalize();
   HypreParMatrix *G = G_form->ParallelAssemble();

   ParLinearForm *g_bdr_form = new ParLinearForm(pres_fes);
   g_bdr_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(u_ex_coeff));

   VectorGridFunctionCoefficient uh_gf_coeff(&uh_gf);
   ParLinearForm *uh_bdr_form = new ParLinearForm(pres_fes);
   uh_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(uh_gf_coeff));

   ConstantCoefficient lincoeff(dt * kinvis);
   ParBilinearForm *H_form = new ParBilinearForm(vel_fes);
   H_form->AddDomainIntegrator(new VectorMassIntegrator);
   H_form->AddDomainIntegrator(new VectorDiffusionIntegrator(lincoeff));
   H_form->Assemble();
   HypreParMatrix H;
   H_form->FormSystemMatrix(ess_tdof_list_u, H);

   HypreSmoother MvInvPC;
   MvInvPC.SetType(HypreSmoother::Jacobi, 1);
   CGSolver MvInv(MPI_COMM_WORLD);
   MvInv.SetPreconditioner(MvInvPC);
   MvInv.SetOperator(Mv);
   MvInv.SetPrintLevel(0);
   MvInv.SetRelTol(1e-8);
   MvInv.SetMaxIter(50);

   HypreBoomerAMG SpInvPC = HypreBoomerAMG(Sp);
   SpInvPC.SetPrintLevel(0);
   GMRESSolver SpInv(MPI_COMM_WORLD);
   SpInv.SetPreconditioner(SpInvPC);
   SpInv.SetOperator(Sp);
   SpInv.SetPrintLevel(0);
   SpInv.SetRelTol(1e-8);
   SpInv.SetMaxIter(50);

   HypreBoomerAMG HInvPC = HypreBoomerAMG(H);
   HInvPC.SetPrintLevel(0);
   CGSolver HInv(MPI_COMM_WORLD);
   HInv.SetPreconditioner(HInvPC);
   HInv.SetOperator(H);
   HInv.SetPrintLevel(0);
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
          << *pmesh << u_gf << "window_title 'velocity " << t << "'"
          << "keys rRljc\n"
          << "pause\n"
          << endl;

   p_gf.Distribute(pn);
   p_sock << "parallel " << num_procs << " " << myid << "\n"
          << "solution\n"
          << *pmesh << p_gf << "window_title 'pressure " << t << "'"
          << "keys rRljc\n"
          << "pause\n"
          << endl;

   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u = u_gf.ComputeL2Error(u_ex_coeff, irs);
   double err_p = 0.0;
   double u_inf = u_gf.Normlinf();
   double p_inf = p_gf.Normlinf();
   if (myid == 0)
   {
      printf("%.2E %.2E %.5E %.5E %.5E %.5E\n",
             t,
             dt,
             err_u,
             err_p,
             u_inf,
             p_inf);
   }
   Vector tmp1(vel_fes->GetTrueVSize());
   Vector tmp2(pres_fes->GetTrueVSize());

   bool last_step = false;

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      t += dt;

      u_ex_coeff.SetTime(t);
      p_ex_coeff.SetTime(t);

      if (mpi.Root())
      {
         // cout << "\nExtrapolation" << endl;
      }

      N->Mult(un, tmp1);
      tmp1 *= dt;
      MvInv.Mult(tmp1, uh);
      uh += un;

      if (mpi.Root())
      {
         // cout << "\nPressure poisson" << endl;
      }

      D->Mult(uh, tmp2);
      tmp2 *= -1.0 / dt;

      ParGridFunction pn_gf(pres_fes), tmp2_gf(pres_fes);
      pn_gf = 0.0;
      tmp2_gf = 0.0;
      // pn_gf.Distribute(pn);
      pres_fes->GetRestrictionMatrix()->MultTranspose(tmp2, tmp2_gf);
      pn_gf = 0.0;

      Vector X1, B1;
      Sp_form->FormLinearSystem(ess_tdof_list_p, pn_gf, tmp2_gf, Sp, X1, B1);
      SpInv.Mult(B1, X1);
      Sp_form->RecoverFEMSolution(X1, tmp2_gf, pn_gf);

      pn_gf.GetTrueDofs(pn);

      double sum = 0.0;
      for (int i = 0; i < pn.Size(); i++)
      {
         sum += pn(i);
      }
      pn -= sum / pn.Size();

      p_gf.Distribute(pn);

      if (mpi.Root())
      {
         // cout << "\nProjection" << endl;
      }

      G->Mult(pn, uhh);
      uhh *= -dt;
      Mv.Mult(uh, tmp1);
      uhh += tmp1;

      if (mpi.Root())
      {
         // cout << "\nHelmholtz" << endl;
      }

      ParGridFunction un_gf(vel_fes), uhh_gf(vel_fes);
      un_gf = 0.0;
      uhh_gf = 0.0;
      un_gf.ProjectBdrCoefficient(u_ex_coeff, ess_bdr_attr_u);
      vel_fes->GetRestrictionMatrix()->MultTranspose(uhh, uhh_gf);

      Vector X2, B2;
      H_form->FormLinearSystem(ess_tdof_list_u, un_gf, uhh_gf, H, X2, B2);
      HInv.Mult(B2, X2);
      H_form->RecoverFEMSolution(X2, uhh_gf, un_gf);

      un_gf.GetTrueDofs(un);

      u_gf.Distribute(un);

      if (step % 1 == 0)
      {
         u_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n"
                << *pmesh << u_gf << "window_title 'velocity " << t << "'"
                << endl;

         p_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n"
                << *pmesh << p_gf << "window_title 'pressure " << t << "'"
                << endl;
      }

      err_u = u_gf.ComputeL2Error(u_ex_coeff, irs);
      err_p = p_gf.ComputeL2Error(p_ex_coeff, irs);
      u_inf = u_gf.Normlinf();
      p_inf = p_gf.Normlinf();
      if (myid == 0)
      {
         printf("%.2E %.2E %.5E %.5E %.5E %.5E\n",
                t,
                dt,
                err_u,
                err_p,
                u_inf,
                p_inf);
      }
   }

   return 0;
}
