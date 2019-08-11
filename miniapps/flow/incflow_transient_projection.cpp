#include "fstream"
#include "mfem.hpp"
#include "vec_conv_integrator.hpp"
#include "vec_grad_integrator.hpp"

using namespace mfem;
using namespace std;

struct Context
{
   int prob_type;
   double kinvis;
} ctx;

void vel_mms(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double F = exp(-2.0 * ctx.kinvis * t);

   u(0) = cos(xi) * sin(yi) * F;
   u(1) = -sin(xi) * cos(yi) * F;
}

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

double p_ex(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);

   double F = exp(-2.0 * ctx.kinvis * t);

   return -1.0 / 4.0 * (cos(2.0 * xi) + cos(2.0 * yi)) * pow(F, 2.0);
}

void ortho(Vector &v)
{
   v -= v.Sum() / v.Size();
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   int serial_ref_levels = 0;
   int order = 5;
   int vel_order = order;
   int pres_order = order;

   double t = 0.0;
   double dt = 1e-5;
   double t_final = dt;
   ctx.prob_type = 1;
   ctx.kinvis = 1.0 / 100000.0;

   OptionsParser args(argc, argv);
   args.AddOption(&serial_ref_levels,
                  "-rs",
                  "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&ctx.kinvis, "-kv", "--kinvis", ".");
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

   std::string mesh_file;
   if (ctx.prob_type == 0)
   {
      mesh_file = "../../data/inline-quad.mesh";
   }
   else if (ctx.prob_type == 1)
   {
      mesh_file = "../../data/periodic-square.mesh";
   }

   Mesh *mesh = new Mesh(mesh_file.c_str());
   int dim = mesh->Dimension();
   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   if (ctx.prob_type == 0)
   {
      *nodes *= M_PI;
   }
   else if (ctx.prob_type == 1)
   {
      nodes->Neg();
      *nodes -= 1.0;
      nodes->Neg();
      *nodes /= 2.0;
   }

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
   Array<int> ess_bdr_attr_u;
   if (ctx.prob_type == 0)
   {
      ess_bdr_attr_u.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_attr_u[0] = 1;
      ess_bdr_attr_u[1] = 1;
      ess_bdr_attr_u[2] = 1;
      ess_bdr_attr_u[3] = 1;
   }
   vel_fes->GetEssentialTrueDofs(ess_bdr_attr_u, ess_tdof_list_u);

   Array<int> ess_tdof_list_p;
   Array<int> ess_bdr_attr_p;
   if (ctx.prob_type == 0)
   {
      ess_bdr_attr_p.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_attr_p[0] = 0;
      ess_bdr_attr_p[1] = 0;
      ess_bdr_attr_p[2] = 0;
      ess_bdr_attr_p[3] = 0;
   }
   pres_fes->GetEssentialTrueDofs(ess_bdr_attr_p, ess_tdof_list_p);

   double g0 = 1.0;
   double a0 = -1.0;
   double a1 = 0.0;
   double a2 = 0.0;
   double b0 = 1.0;
   double b1 = 0.0;
   double b2 = 0.0;

   Vector un(vel_fes->GetTrueVSize()), unm1(vel_fes->GetTrueVSize()),
      unm2(vel_fes->GetTrueVSize());
   Vector uh(vel_fes->GetTrueVSize());
   Vector uhh(vel_fes->GetTrueVSize());

   un = 0.0;
   unm1 = 0.0;
   unm2 = 0.0;
   uh = 0.0;
   uhh = 0.0;

   ParGridFunction Nu_gf(vel_fes);
   Nu_gf = 0.0;

   Vector pn(pres_fes->GetTrueVSize());
   pn = 0.0;

   ParGridFunction p_gf(pres_fes);
   p_gf = 0.0;

   VectorFunctionCoefficient *u_ex_coeff = nullptr;
   if (ctx.prob_type == 0)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_mms);
   }
   else if (ctx.prob_type == 1)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_shear_ic);
   }
   ParGridFunction u_gf(vel_fes);
   u_gf = 0.0;
   u_gf.ProjectCoefficient(*u_ex_coeff);
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
   g_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(*u_ex_coeff));

   VectorGridFunctionCoefficient Nu_gf_coeff(&Nu_gf);
   ParLinearForm *Nu_bdr_form = new ParLinearForm(pres_fes);
   Nu_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(Nu_gf_coeff));

   ConstantCoefficient lincoeff(dt * ctx.kinvis);
   ConstantCoefficient bdfcoeff(1.0);
   ParBilinearForm *H_form = new ParBilinearForm(vel_fes);
   H_form->AddDomainIntegrator(new VectorMassIntegrator(bdfcoeff));
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
   MvInv.SetRelTol(1e-12);
   MvInv.SetMaxIter(50);

   HypreBoomerAMG SpInvPC = HypreBoomerAMG(Sp);
   SpInvPC.SetPrintLevel(0);
   GMRESSolver SpInv(MPI_COMM_WORLD);
   SpInv.SetPreconditioner(SpInvPC);
   SpInv.SetOperator(Sp);
   SpInv.SetPrintLevel(0);
   SpInv.SetRelTol(1e-12);
   SpInv.SetMaxIter(50);

   HypreBoomerAMG HInvPC = HypreBoomerAMG(H);
   HInvPC.SetPrintLevel(0);
   CGSolver HInv(MPI_COMM_WORLD);
   HInv.SetPreconditioner(HInvPC);
   HInv.SetOperator(H);
   HInv.SetPrintLevel(0);
   HInv.SetRelTol(1e-12);
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
          << *pmesh << u_gf << "window_title 'velocity t=" << t << "'"
          << "keys rRljc\n"
          << "pause\n"
          << endl;

   p_gf.Distribute(pn);
   p_sock << "parallel " << num_procs << " " << myid << "\n"
          << "solution\n"
          << *pmesh << p_gf << "window_title 'pressure t=" << t << "'"
          << "keys rRljc\n"
          << "pause\n"
          << endl;

   VisItDataCollection visit_dc("mms", pmesh);
   visit_dc.SetPrefixPath("output");
   visit_dc.SetCycle(0);
   visit_dc.SetTime(t);
   visit_dc.RegisterField("velocity", &u_gf);
   visit_dc.RegisterField("pressure", &p_gf);
   visit_dc.Save();

   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u = 0.0;
   double err_p = 0.0;
   double err_inf_u = 0.0;
   double err_inf_p = 0.0;
   double u_inf = u_gf.Normlinf();
   double p_inf = p_gf.Normlinf();
   if (myid == 0)
   {
      printf("%.3E %.2E %.5E %.5E %.5E %.5E %.5E %.5E\n",
             t,
             dt,
             err_u,
             err_p,
             err_inf_u,
             err_inf_p,
             u_inf,
             p_inf);
   }
   Vector tmp1(vel_fes->GetTrueVSize());
   Vector tmp2(vel_fes->GetTrueVSize());
   Vector tmp3(vel_fes->GetTrueVSize());
   Vector tmpp(pres_fes->GetTrueVSize());

   bool last_step = false;

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      t += dt;

      u_ex_coeff->SetTime(t);
      p_ex_coeff.SetTime(t);

      // Extrapolation
#if 0
      // AB3 coefficients
      if (step == 0)
      {
         b0 = 1.0;
         b1 = 0.0;
         b2 = 0.0;
      }
      else if (step == 1)
      {
         b0 = 3.0 / 2.0;
         b1 = -1.0 / 2.0;
         b2 = 0.0;
      }
      else if (step > 1)
      {
         b0 = 23.0 / 12.0;
         b1 = -16.0 / 12.0;
         b2 = 5.0 / 12.0;
      }
#endif

      // BDFk/EXTk coefficients
      if (step == 0)
      {
         g0 = 1.0;
         a0 = -1.0;
         a1 = 0.0;
         a2 = 0.0;
         b0 = 1.0;
         b1 = 0.0;
         b2 = 0.0;
      }
      else if (step == 1)
      {
         g0 = 3.0 / 2.0;
         a0 = -2.0;
         a1 = 1.0 / 2.0;
         a2 = 0.0;
         b0 = 2.0;
         b1 = -1.0;
         b2 = 0.0;
      }
      else if (step == 2)
      {
         g0 = 11.0 / 6.0;
         a0 = -3.0;
         a1 = 3.0 / 2.0;
         a2 = -1.0 / 3.0;
         b0 = 3.0;
         b1 = -3.0;
         b2 = 1.0;
      }

      // EXT terms
      N->Mult(un, tmp1);
      tmp1 *= b0;
      N->Mult(unm1, tmp2);
      tmp2 *= b1;
      N->Mult(unm2, tmp3);
      tmp3 *= b2;

      tmp1 += tmp2;
      tmp1 += tmp3;
      tmp1 *= dt;
      MvInv.Mult(tmp1, uh);

      // BDF terms
      uh.Add(-a0, un);
      uh.Add(-a1, unm1);
      uh.Add(-a2, unm2);

      if (step <= 2)
      {
         bdfcoeff.constant = g0;
         H_form->Update();
         H_form->Assemble();
         H_form->FormSystemMatrix(ess_tdof_list_u, H);
         HInv.SetOperator(H);
      }

      // Pressure poisson

      D->Mult(uh, tmpp);
      tmpp *= -1.0 / dt;

      // Add boundary terms
      // MISSING

      ParGridFunction pn_gf(pres_fes), tmpp_gf(pres_fes);
      pn_gf = 0.0;
      tmpp_gf = 0.0;
      pn_gf.Distribute(pn);

      // ortho(tmpp);

      pres_fes->GetRestrictionMatrix()->MultTranspose(tmpp, tmpp_gf);
      pn_gf = 0.0;

      Vector X1, B1;
      Sp_form->FormLinearSystem(ess_tdof_list_p, pn_gf, tmpp_gf, Sp, X1, B1);
      SpInv.Mult(B1, X1);
      Sp_form->RecoverFEMSolution(X1, tmpp_gf, pn_gf);

      pn_gf.GetTrueDofs(pn);

      // ortho(pn);

      p_gf.Distribute(pn);

      // Project velocity

      G->Mult(pn, uhh);
      uhh *= -dt;
      Mv.Mult(uh, tmp1);
      uhh += tmp1;

      // Helmholtz

      ParGridFunction un_gf(vel_fes), uhh_gf(vel_fes);
      un_gf = 0.0;
      uhh_gf = 0.0;
      un_gf.ProjectBdrCoefficient(*u_ex_coeff, ess_bdr_attr_u);
      vel_fes->GetRestrictionMatrix()->MultTranspose(uhh, uhh_gf);

      unm2 = unm1;
      unm1 = un;

      Vector X2, B2;
      H_form->FormLinearSystem(ess_tdof_list_u, un_gf, uhh_gf, H, X2, B2);
      HInv.Mult(B2, X2);
      H_form->RecoverFEMSolution(X2, uhh_gf, un_gf);

      un_gf.GetTrueDofs(un);

      u_gf.Distribute(un);

      if (step % 10 == 0 || last_step)
      {
         u_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n"
                << *pmesh << u_gf << "window_title 'velocity t=" << t << "'"
                << endl;

         p_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n"
                << *pmesh << p_gf << "window_title 'pressure t=" << t << "'"
                << endl;

         visit_dc.SetCycle(step);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }

      err_u = u_gf.ComputeL2Error(*u_ex_coeff, irs);
      err_p = p_gf.ComputeL2Error(p_ex_coeff, irs);
      err_inf_u = u_gf.ComputeMaxError(*u_ex_coeff, irs);
      err_inf_p = p_gf.ComputeMaxError(p_ex_coeff, irs);
      u_inf = u_gf.Normlinf();
      p_inf = p_gf.Normlinf();
      if (myid == 0)
      {
         printf("%.3E %.2E %.5E %.5E %.5E %.5E %.5E %.5E\n",
                t,
                dt,
                err_u,
                err_p,
                err_inf_u,
                err_inf_p,
                u_inf,
                p_inf);
      }

      MFEM_ASSERT(u_gf.Normlinf() <= 2.0, "UNSTABLE");
   }

   return 0;
}
