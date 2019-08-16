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

void f_mms_guermond(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = M_PI * sin(t) * sin(M_PI * xi) * sin(M_PI * yi)
             * (-1.0
                + 2.0 * pow(M_PI, 2.0) * sin(t) * sin(M_PI * xi)
                     * sin(2.0 * M_PI * xi) * sin(M_PI * yi))
          + M_PI
               * (2.0 * ctx.kinvis * pow(M_PI, 2.0)
                     * (1.0 - 2.0 * cos(2.0 * M_PI * xi)) * sin(t)
                  + cos(t) * pow(sin(M_PI * xi), 2.0))
               * sin(2.0 * M_PI * yi);

   u(1) = M_PI * cos(M_PI * yi) * sin(t)
             * (cos(M_PI * xi)
                + 2.0 * ctx.kinvis * pow(M_PI, 2.0) * cos(M_PI * yi)
                     * sin(2.0 * M_PI * xi))
          - M_PI * (cos(t) + 6.0 * ctx.kinvis * pow(M_PI, 2.0) * sin(t))
               * sin(2.0 * M_PI * xi) * pow(sin(M_PI * yi), 2.0)
          + 4.0 * pow(M_PI, 3.0) * cos(M_PI * yi) * pow(sin(t), 2.0)
               * pow(sin(M_PI * xi), 2.0) * pow(sin(M_PI * yi), 3.0);
}

void vel_mms_guermond(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = M_PI * sin(t) * pow(sin(M_PI * xi), 2.0) * sin(2.0 * M_PI * yi);
   u(1) = -(M_PI * sin(t) * sin(2.0 * M_PI * xi) * pow(sin(M_PI * yi), 2.0));
}

double pres_mms_guermond(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);

   return cos(M_PI * xi) * sin(t) * sin(M_PI * yi);
}

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
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   v -= global_sum / static_cast<double>(global_size);
}

void MeanZero(ParGridFunction &v)
{
   ConstantCoefficient one(1.0);
   ParLinearForm mass_lf(v.ParFESpace());
   mass_lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   mass_lf.Assemble();

   ParGridFunction one_gf(v.ParFESpace());
   one_gf.ProjectCoefficient(one);

   double volume = mass_lf(one_gf);
   double integ = mass_lf(v);

   v -= integ / volume;
}

void ComputeCurlCurl(ParGridFunction &u, ParGridFunction &ccu)
{
   // ParFiniteElementSpace u_comp_fes(u.ParFESpace()->GetParMesh(),
   //                                  u.ParFESpace()->FEColl());
   // ParGridFunction u_comp(&u_comp_fes);
   ParGridFunction cu(u.ParFESpace());
   CurlGridFunctionCoefficient cu_gfcoeff(&u);

   cu.ProjectDiscCoefficient(cu_gfcoeff);

   if (u.ParFESpace()->GetVDim() == 2)
   {
      for (int i = 0; i < u.ParFESpace()->GetNDofs(); i++)
      {
         cu[cu.ParFESpace()->DofToVDof(i, 1)] = 0.0;
      }
   }

   cu_gfcoeff.SetGridFunction(&cu);
   cu_gfcoeff.assume_scalar = true;
   ccu.ProjectDiscCoefficient(cu_gfcoeff);
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
   ctx.prob_type = 0;
   ctx.kinvis = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&serial_ref_levels,
                  "-rs",
                  "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&ctx.prob_type, "-prob", "--prob", ".");
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
   if (ctx.prob_type == 0 || ctx.prob_type == 2)
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
   if (ctx.prob_type == 0 || ctx.prob_type == 2)
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
   if (ctx.prob_type == 0 || ctx.prob_type == 2)
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
   Vector f(vel_fes->GetTrueVSize());
   Vector f_bdr(pres_fes->GetTrueVSize());
   Vector Nuext_bdr(pres_fes->GetTrueVSize());
   Vector curlcurlu_bdr(pres_fes->GetTrueVSize());
   Vector uh_bdr(pres_fes->GetTrueVSize());
   Vector g_bdr(pres_fes->GetTrueVSize());

   un = 0.0;
   unm1 = 0.0;
   unm2 = 0.0;
   uh = 0.0;
   uhh = 0.0;
   f = 0.0;
   f_bdr = 0.0;
   uh_bdr = 0.0;

   ParGridFunction Nu_gf(vel_fes);
   Nu_gf = 0.0;

   Vector pn(pres_fes->GetTrueVSize());
   pn = 0.0;

   ParGridFunction p_gf(pres_fes);
   p_gf = 0.0;

   ParGridFunction curlcurlu_gf(vel_fes), curlcurlun_gf(vel_fes),
      curlcurlunm1_gf(vel_fes), curlcurlunm2_gf(vel_fes), un_gf(vel_fes),
      unm1_gf(vel_fes), unm2_gf(vel_fes);
   curlcurlu_gf = 0.0;

   ParGridFunction uh_gf(vel_fes);
   uh_gf = 0.0;

   VectorFunctionCoefficient *u_ex_coeff = nullptr;
   if (ctx.prob_type == 0)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_mms);
   }
   else if (ctx.prob_type == 1)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_shear_ic);
   }
   else if (ctx.prob_type == 2)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_mms_guermond);
   }
   ParGridFunction u_gf(vel_fes);
   u_gf = 0.0;
   u_gf.ProjectCoefficient(*u_ex_coeff);
   u_gf.GetTrueDofs(un);

   FunctionCoefficient *p_ex_coeff = nullptr;
   if (ctx.prob_type == 0 || ctx.prob_type == 1)
   {
      p_ex_coeff = new FunctionCoefficient(p_ex);
   }
   else if (ctx.prob_type == 2)
   {
      p_ex_coeff = new FunctionCoefficient(pres_mms_guermond);
   }

   ParGridFunction p_ex_gf(pres_fes);
   p_ex_gf.ProjectCoefficient(*p_ex_coeff);
   p_ex_gf.GetTrueDofs(pn);

   VectorCoefficient *forcing_coeff = nullptr;
   if (ctx.prob_type == 2)
   {
      forcing_coeff = new VectorFunctionCoefficient(dim, f_mms_guermond);
   }
   else
   {
      Vector v(dim);
      v = 0.0;
      forcing_coeff = new VectorConstantCoefficient(v);
   }
   ParLinearForm *f_form = new ParLinearForm(vel_fes);
   VectorDomainLFIntegrator *vdlfi = new VectorDomainLFIntegrator(
      *forcing_coeff);
   const IntegrationRule &ir = IntRules.Get(vel_fes->GetFE(0)->GetGeomType(),
                                            4 * order);
   vdlfi->SetIntRule(&ir);
   f_form->AddDomainIntegrator(vdlfi);

   ConstantCoefficient nlcoeff(-1.0);
   ParNonlinearForm *N = new ParNonlinearForm(vel_fes);
   N->AddDomainIntegrator(new VectorConvectionNLFIntegrator(nlcoeff));

   ParBilinearForm *Mv_form = new ParBilinearForm(vel_fes);
   VectorMassIntegrator *blfi = new VectorMassIntegrator;
   // IntegrationRules rules_ni(0, Quadrature1D::GaussLobatto);
   // const IntegrationRule &ir_ni = rules_ni.Get(vel_fes->GetFE(0)->GetGeomType(),
   //                                             2 * order - 1);
   // blfi->SetIntRule(&ir_ni);
   Mv_form->AddDomainIntegrator(blfi);
   Mv_form->Assemble();
   Mv_form->Finalize();
   HypreParMatrix Mv = *Mv_form->ParallelAssemble();
   Array<int> empty;
   Mv_form->FormSystemMatrix(empty, Mv);
   // Mv.Threshold(1e-12);

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

   VectorGridFunctionCoefficient uh_gf_coeff(&uh_gf);
   ParLinearForm *uh_bdr_form = new ParLinearForm(pres_fes);
   uh_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(uh_gf_coeff));

   ParLinearForm *f_bdr_form = new ParLinearForm(pres_fes);
   f_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(*forcing_coeff));

   ParGridFunction Nuext_gf(vel_fes);
   VectorGridFunctionCoefficient Nuext_gfcoeff(&Nuext_gf);
   ParLinearForm *Nuext_bdr_form = new ParLinearForm(pres_fes);
   Nuext_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(Nuext_gfcoeff));

   VectorGridFunctionCoefficient curlcurlu_gfcoeff(&curlcurlu_gf);
   ParLinearForm *curlcurlu_bdr_form = new ParLinearForm(pres_fes);
   curlcurlu_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(curlcurlu_gfcoeff, 2, 1));

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
   // HYPRE_Solver amg_precond = static_cast<HYPRE_Solver>(SpInvPC);
   // HYPRE_BoomerAMGSetCoarsenType(amg_precond, 6);
   // HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
   // HYPRE_BoomerAMGSetRelaxType(amg_precond, 6);
   // HYPRE_BoomerAMGSetInterpType(amg_precond, 0);
   // HYPRE_BoomerAMGSetPMaxElmts(amg_precond, 0);
   SpInvPC.SetPrintLevel(0);
   GMRESSolver SpInv(MPI_COMM_WORLD);
   SpInv.SetPreconditioner(SpInvPC);
   SpInv.SetOperator(Sp);
   SpInv.SetPrintLevel(0);
   SpInv.SetRelTol(1e-12);
   SpInv.SetMaxIter(50);

   HypreBoomerAMG HInvPC = HypreBoomerAMG(H);
   // amg_precond = static_cast<HYPRE_Solver>(HInvPC);
   // HYPRE_BoomerAMGSetCoarsenType(amg_precond, 6);
   // HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
   // HYPRE_BoomerAMGSetRelaxType(amg_precond, 6);
   // HYPRE_BoomerAMGSetInterpType(amg_precond, 0);
   // HYPRE_BoomerAMGSetPMaxElmts(amg_precond, 0);
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

   ParGridFunction *u_err_gf = nullptr;
   ParGridFunction *p_err_gf = nullptr;

   VisItDataCollection visit_dc("mms", pmesh);
   visit_dc.SetPrefixPath("output");
   visit_dc.SetCycle(0);
   visit_dc.SetTime(t);
   visit_dc.RegisterField("velocity", &u_gf);
   visit_dc.RegisterField("pressure", &p_gf);
   visit_dc.RegisterField("curlcurlun", &curlcurlun_gf);
   if (ctx.prob_type == 0 || ctx.prob_type == 2)
   {
      u_err_gf = new ParGridFunction(vel_fes);
      p_err_gf = new ParGridFunction(pres_fes);
      visit_dc.RegisterField("velocity_error", u_err_gf);
      visit_dc.RegisterField("pressure_error", p_err_gf);
   }
   visit_dc.Save();

   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   IntegrationRules irs_gll(0, Quadrature1D::GaussLobatto);
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(irs_gll.Get(i, order_quad));
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
      p_ex_coeff->SetTime(t);
      forcing_coeff->SetTime(t);

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

      Vector Nuext(vel_fes->GetTrueVSize());
      Nuext = tmp1;
      Nuext += tmp2;
      Nuext += tmp3;
      Nuext_gf.SetFromTrueDofs(Nuext);

      // Forcing term
      // f^{n+1} is assumed to be known
      f_form->Assemble();
      f_form->ParallelAssemble(f);
      tmp1 += f;

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
      tmp1 = 0.0;
      tmp1.Add(-a0, un);
      tmp1.Add(-a1, unm1);
      tmp1.Add(-a2, unm2);
      uh_gf.SetFromTrueDofs(tmp1);
      uh_bdr_form->Assemble();
      uh_bdr_form->ParallelAssemble(uh_bdr);
      g_bdr_form->Assemble();
      g_bdr_form->ParallelAssemble(g_bdr);
      tmpp.Add(1.0 / dt, uh_bdr);
      tmpp.Add(-g0 / dt, g_bdr);

      // Forcing
      f_bdr_form->Assemble();
      f_bdr_form->ParallelAssemble(f_bdr);
      tmpp += f_bdr;

      // Extrapolated nonlinear terms
      // Has actually an impact on the error,
      // although only in the range ~1e-6
      Nuext_bdr_form->Assemble();
      Nuext_bdr_form->ParallelAssemble(Nuext_bdr);
      tmpp += Nuext_bdr;

      // CurlCurl
      tmp1 = 0.0;
      tmp1.Add(b0, un);
      tmp1.Add(b1, unm1);
      tmp1.Add(b2, unm2);
      uh_gf.SetFromTrueDofs(tmp1);
      ComputeCurlCurl(uh_gf, curlcurlu_gf);
      curlcurlu_bdr_form->Assemble();
      curlcurlu_bdr_form->ParallelAssemble(curlcurlu_bdr);
      tmpp.Add(-ctx.kinvis, curlcurlu_bdr);

      ParGridFunction pn_gf(pres_fes), tmpp_gf(pres_fes);
      pn_gf = 0.0;
      tmpp_gf = 0.0;
      pn_gf.Distribute(pn);

      ortho(tmpp);

      pres_fes->GetRestrictionMatrix()->MultTranspose(tmpp, tmpp_gf);
      pn_gf = 0.0;

      Vector X1, B1;
      Sp_form->FormLinearSystem(ess_tdof_list_p, pn_gf, tmpp_gf, Sp, X1, B1);
      SpInv.Mult(B1, X1);
      Sp_form->RecoverFEMSolution(X1, tmpp_gf, pn_gf);

      MeanZero(pn_gf);

      pn_gf.GetTrueDofs(pn);

      p_gf.Distribute(pn);

      // Project velocity

      G->Mult(pn, uhh);
      uhh *= -dt;
      Mv.Mult(uh, tmp1);
      uhh += tmp1;

      // Helmholtz

      ParGridFunction uhh_gf(vel_fes);
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

      if ((step + 1) % 10 == 0 || last_step)
      {
         u_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n"
                << *pmesh << u_gf << "window_title 'velocity t=" << t << "'"
                << endl;

         p_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n"
                << *pmesh << p_gf << "window_title 'pressure t=" << t << "'"
                << endl;

         if (ctx.prob_type == 0 || ctx.prob_type == 2)
         {
            u_err_gf->ProjectCoefficient(*u_ex_coeff);
            p_err_gf->ProjectCoefficient(*p_ex_coeff);
            u_err_gf->Add(-1.0, u_gf);
            p_err_gf->Add(-1.0, p_gf);
         }
         visit_dc.SetCycle(step + 1);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }

      err_u = u_gf.ComputeL2Error(*u_ex_coeff, irs);
      err_p = p_gf.ComputeL2Error(*p_ex_coeff, irs);
      err_inf_u = u_gf.ComputeMaxError(*u_ex_coeff, irs);
      err_inf_p = p_gf.ComputeMaxError(*p_ex_coeff, irs);
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

      MFEM_ASSERT(u_gf.Normlinf() <= 3.0, "UNSTABLE");
   }

   return 0;
}
