#include "fstream"
#include "mfem.hpp"
#include "vec_conv_integrator.hpp"
#include "vec_grad_integrator.hpp"

using namespace mfem;
using namespace std;

struct Context
{
   int prob;
   double kinvis;
} ctx;

void vel_kovasznay(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double reynolds = 1.0 / ctx.kinvis;
   double lam = 0.5 * reynolds
                - sqrt(0.25 * reynolds * reynolds + 4.0 * M_PI * M_PI);

   u(0) = 1.0 - exp(lam * xi) * cos(2.0 * M_PI * yi);
   u(1) = lam / (2.0 * M_PI) * exp(lam * xi) * sin(2.0 * M_PI * yi);
}

double pres_kovasznay(const Vector &x)
{
   double xi = x(0);
   double yi = x(1);

   double reynolds = 1.0 / ctx.kinvis;
   double lam = 0.5 * reynolds
                - sqrt(0.25 * reynolds * reynolds + 4.0 * M_PI * M_PI);

   return (0.0 - 0.5 * exp(2.0 * lam * xi));
}

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

void vel_mms_tgv(const Vector &x, double t, Vector &u)
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

   return -0.25 * (cos(2.0 * xi) + cos(2.0 * yi)) * pow(F, 2.0);
}

void vel_mms_tgv_dt(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double F = exp(-2.0 * ctx.kinvis * t);

   u(0) = -2.0 * ctx.kinvis * cos(xi) * sin(yi) * F;
   u(1) = 2.0 * ctx.kinvis * cos(yi) * sin(xi) * F;
}

void vel_vortex(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);
   double az = 0.0;

   if (zi > 0.0)
   {
      az = pow((zi - 0.0) / (2.0 - 0.0), 5.0);
   }

   u(0) = -yi * az;
   u(1) = xi * az;
   u(2) = 0.0;
}

void vel_threedcyl(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);

   // Re = 100.0 with nu = 0.001
   double U = 2.25;

   if (xi <= 1e-8)
   {
      u(0) = 16.0 * U * yi * zi * (0.41 - yi) * (0.41 - zi) / pow(0.41, 4.0);
   }
   else
   {
      u(0) = 0.0;
   }
   u(1) = 0.0;
   u(2) = 0.0;
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
   ParGridFunction cu(u.ParFESpace());
   CurlGridFunctionCoefficient cu_gfcoeff(&u);

   cu.ProjectDiscCoefficient(cu_gfcoeff, GridFunction::AvgType::ARITHMETIC);
   // cu.ProjectDiscCoefficient(cu_gfcoeff);

   if (u.ParFESpace()->GetVDim() == 2)
   {
      for (int i = 0; i < u.ParFESpace()->GetNDofs(); i++)
      {
         cu[cu.ParFESpace()->DofToVDof(i, 1)] = 0.0;
      }
   }

   cu_gfcoeff.SetGridFunction(&cu);
   cu_gfcoeff.assume_scalar = true;
   ccu.ProjectDiscCoefficient(cu_gfcoeff, GridFunction::AvgType::ARITHMETIC);
   // ccu.ProjectDiscCoefficient(cu_gfcoeff);
}

double neknormvc(const Vector &utdof, ParFiniteElementSpace *pfes)
{
   ParGridFunction ugf(pfes);
   ugf.SetFromTrueDofs(utdof);

   ParFiniteElementSpace h1compfes(pfes->GetParMesh(), pfes->FEColl());

   ParGridFunction u(&h1compfes), v(&h1compfes), velmagl2(&h1compfes);

   for (int comp = 0; comp < pfes->GetVDim(); comp++)
   {
      for (int i = 0; i < pfes->GetNDofs(); i++)
      {
         if (comp == 0)
         {
            u(i) = ugf[ugf.ParFESpace()->DofToVDof(i, comp)];
         }
         else if (comp == 1)
         {
            v(i) = ugf[ugf.ParFESpace()->DofToVDof(i, comp)];
         }
      }
   }

   for (int i = 0; i < pfes->GetNDofs(); i++)
   {
      velmagl2(i) = pow(u(i), 2.0) + pow(v(i), 2.0);
   }

   ConstantCoefficient onecoeff(1.0);
   ParLinearForm mass_lf(&h1compfes);
   mass_lf.AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
   mass_lf.Assemble();

   ParGridFunction onegf(&h1compfes);
   onegf.ProjectCoefficient(onecoeff);

   double nekl2norm = sqrt(mass_lf(velmagl2) / mass_lf(onegf));
   return nekl2norm;
}

double neknormsc(const Vector &ptdof, ParFiniteElementSpace *pfes)
{
   ParGridFunction pgf(pfes);
   pgf.SetFromTrueDofs(ptdof);

   ParGridFunction p2(pfes);

   for (int i = 0; i < pfes->GetNDofs(); i++)
   {
      p2(i) = pgf(i) * pgf(i);
   }

   ConstantCoefficient onecoeff(1.0);
   ParLinearForm mass_lf(pfes);
   mass_lf.AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
   mass_lf.Assemble();

   ParGridFunction onegf(pfes);
   onegf.ProjectCoefficient(onecoeff);

   double nekl2norm = sqrt(mass_lf(p2) / mass_lf(onegf));
   return nekl2norm;
}

double ComputeCFL(ParGridFunction &u, double &dt_est)
{
   ParMesh *pmesh = u.ParFESpace()->GetParMesh();

   double hmin = 0.0;
   double hmin_loc = pmesh->GetElementSize(0, 1);

   for (int i = 1; i < pmesh->GetNE(); i++)
   {
      hmin_loc = min(pmesh->GetElementSize(i, 1), hmin_loc);
   }

   MPI_Allreduce(&hmin_loc, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());

   int ndofs = u.ParFESpace()->GetNDofs();
   Vector uc(ndofs), vc(ndofs);

   for (int comp = 0; comp < u.ParFESpace()->GetVDim(); comp++)
   {
      for (int i = 0; i < ndofs; i++)
      {
         if (comp == 0)
         {
            uc(i) = u[u.ParFESpace()->DofToVDof(i, comp)];
         }
         else if (comp == 1)
         {
            vc(i) = u[u.ParFESpace()->DofToVDof(i, comp)];
         }
      }
   }

   double velmag_max_loc = 0.0;
   double velmag_max = 0.0;
   for (int i = 0; i < ndofs; i++)
   {
      velmag_max_loc = max(sqrt(pow(uc(i), 2.0) + pow(vc(i), 2.0)),
                           velmag_max_loc);
   }

   MPI_Allreduce(&velmag_max_loc,
                 &velmag_max,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 pmesh->GetComm());

   double cfl = velmag_max * dt_est / hmin;

   return cfl;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   int serial_ref_levels = 0;
   int order = 5;
   double t = 0.0;
   double dt = 1e-5;
   double t_final = dt;
   ctx.prob = 0;
   ctx.kinvis = 1.0;
   bool enable_curl = true;
   double pres_rtol = 1e-12;
   double helm_rtol = 1e-12;

   OptionsParser args(argc, argv);
   args.AddOption(&serial_ref_levels,
                  "-rs",
                  "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&ctx.prob, "-prob", "--prob", ".");
   args.AddOption(&ctx.kinvis, "-kv", "--kinvis", ".");
   args.AddOption(&dt, "-dt", "--dt", ".");
   args.AddOption(&t_final, "-tf", "--final-time", ".");
   args.AddOption(&order, "-o", "--order", ".");
   args.AddOption(&enable_curl,
                  "-curl",
                  "--enable-curl",
                  "-nocurl",
                  "--disable-curl",
                  ".");
   args.AddOption(&pres_rtol, "-pres_rtol", "--pressure-rel-tolerance", ".");
   args.AddOption(&helm_rtol, "-helm_rtol", "--helmholtz-rel-tolerance", ".");
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

   if (ctx.kinvis < 0.0)
   {
      ctx.kinvis = 1.0 / abs(ctx.kinvis);
   }
   int vel_order = order;
   int pres_order = order;

   std::string mesh_file;
   if (ctx.prob == 0 || ctx.prob == 2)
   {
      mesh_file = "../../data/inline-quad.mesh";
   }
   else if (ctx.prob == 3)
   {
      mesh_file = "inline-quad-kov.mesh";
   }
   else if (ctx.prob == 1)
   {
      mesh_file = "../../data/periodic-square.mesh";
   }
   else if (ctx.prob == 4)
   {
      mesh_file = "cyl27.e";
   }
   else if (ctx.prob == 5)
   {
      mesh_file = "3dfoc.e";
   }

   Mesh *mesh = new Mesh(mesh_file.c_str());

   int dim = mesh->Dimension();
   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   if (ctx.prob == 0)
   {
      *nodes *= 2.0;
      *nodes -= 1.0;
      *nodes *= 0.5 * M_PI;
   }
   else if (ctx.prob == 1)
   {
      nodes->Neg();
      *nodes -= 1.0;
      nodes->Neg();
      *nodes /= 2.0;
   }
   else if (ctx.prob == 3)
   {
      *nodes -= 0.5;
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
   if (ctx.prob == 0 || ctx.prob == 2 || ctx.prob == 3)
   {
      ess_bdr_attr_u.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_attr_u[0] = 1;
      ess_bdr_attr_u[1] = 1;
      ess_bdr_attr_u[2] = 1;
      ess_bdr_attr_u[3] = 1;
   }
   else if (ctx.prob == 4)
   {
      ess_bdr_attr_u.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_attr_u[0] = 1;
      ess_bdr_attr_u[1] = 1;
   }
   else if (ctx.prob == 5)
   {
      ess_bdr_attr_u.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_attr_u[0] = 1;
      ess_bdr_attr_u[1] = 0;
      ess_bdr_attr_u[2] = 1;
   }
   vel_fes->GetEssentialTrueDofs(ess_bdr_attr_u, ess_tdof_list_u);

   Array<int> ess_tdof_list_p;
   Array<int> ess_bdr_attr_p;
   if (ctx.prob == 0 || ctx.prob == 2 || ctx.prob == 3)
   {
      ess_bdr_attr_p.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_attr_p[0] = 0;
      ess_bdr_attr_p[1] = 0;
      ess_bdr_attr_p[2] = 0;
      ess_bdr_attr_p[3] = 0;
   }
   else if (ctx.prob == 4)
   {
      ess_bdr_attr_p.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_attr_p[0] = 0;
      ess_bdr_attr_p[1] = 0;
   }
   else if (ctx.prob == 5)
   {
      ess_bdr_attr_p.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_attr_p[0] = 0;
      ess_bdr_attr_p[1] = 1;
      ess_bdr_attr_p[2] = 0;
   }

   pres_fes->GetEssentialTrueDofs(ess_bdr_attr_p, ess_tdof_list_p);

   double bd0 = 1.0;
   double bd1 = -1.0;
   double bd2 = 0.0;
   double bd3 = 0.0;
   double ab1 = 1.0;
   double ab2 = 0.0;
   double ab3 = 0.0;

   Vector un(vel_fes->GetTrueVSize()), unm1(vel_fes->GetTrueVSize()),
      unm2(vel_fes->GetTrueVSize());
   Vector uh(vel_fes->GetTrueVSize());
   Vector resu(vel_fes->GetTrueVSize());
   Vector Nun(vel_fes->GetTrueVSize());
   Vector Nunm1(vel_fes->GetTrueVSize());
   Vector Nunm2(vel_fes->GetTrueVSize());
   Vector f(vel_fes->GetTrueVSize());
   Vector fn(vel_fes->GetTrueVSize());
   Vector fnm1(vel_fes->GetTrueVSize());
   Vector fnm2(vel_fes->GetTrueVSize());
   Vector Lext(vel_fes->GetTrueVSize());
   Vector Fext(vel_fes->GetTrueVSize());
   Vector FText(vel_fes->GetTrueVSize());
   Vector f_bdr(pres_fes->GetTrueVSize());
   Vector Fext_bdr(pres_fes->GetTrueVSize());
   Vector curlcurlu_bdr(pres_fes->GetTrueVSize());
   Vector FText_bdr(pres_fes->GetTrueVSize());
   Vector g_bdr(pres_fes->GetTrueVSize());

   Nun = 0.0;
   Nunm1 = 0.0;
   Nunm2 = 0.0;
   un = 0.0;
   unm1 = 0.0;
   unm2 = 0.0;
   uh = 0.0;
   resu = 0.0;
   f = 0.0;
   fn = 0.0;
   fnm1 = 0.0;
   fnm2 = 0.0;
   f_bdr = 0.0;
   FText_bdr = 0.0;

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
   if (ctx.prob == 0)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_mms_tgv);
   }
   else if (ctx.prob == 1)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_shear_ic);
   }
   else if (ctx.prob == 2)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_mms_guermond);
   }
   else if (ctx.prob == 3)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_kovasznay);
   }
   else if (ctx.prob == 4)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_vortex);
   }
   else if (ctx.prob == 5)
   {
      u_ex_coeff = new VectorFunctionCoefficient(dim, vel_threedcyl);
   }

   ParGridFunction u_gf(vel_fes);
   if (!(ctx.prob == 5))
   {
      u_gf = 0.0;
      u_gf.ProjectCoefficient(*u_ex_coeff);
      u_gf.GetTrueDofs(un);
   }
   else
   {
      u_gf = 0.0;
      u_gf.ProjectBdrCoefficient(*u_ex_coeff, ess_bdr_attr_u);
      u_gf.GetTrueDofs(un);
   }

   FunctionCoefficient *p_ex_coeff = nullptr;
   if (ctx.prob == 0 || ctx.prob == 1 || ctx.prob == 4
       || ctx.prob == 5)
   {
      p_ex_coeff = new FunctionCoefficient(p_ex);
   }
   else if (ctx.prob == 2)
   {
      p_ex_coeff = new FunctionCoefficient(pres_mms_guermond);
   }
   else if (ctx.prob == 3)
   {
      p_ex_coeff = new FunctionCoefficient(pres_kovasznay);
   }

   // ParGridFunction p_ex_gf(pres_fes);
   // p_ex_gf.ProjectCoefficient(*p_ex_coeff);
   // p_ex_gf.GetTrueDofs(pn);

   VectorCoefficient *forcing_coeff = nullptr;
   if (ctx.prob == 2)
   {
      forcing_coeff = new VectorFunctionCoefficient(dim, f_mms_guermond);
   }
   else
   {
      Vector v(dim);
      v = 0.0;
      forcing_coeff = new VectorConstantCoefficient(v);
   }

   IntegrationRules rules_ni(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir_ni = rules_ni.Get(vel_fes->GetFE(0)->GetGeomType(),
                                               2 * order - 1);

   ParLinearForm *f_form = new ParLinearForm(vel_fes);
   VectorDomainLFIntegrator *vdlfi = new VectorDomainLFIntegrator(
      *forcing_coeff);
   const IntegrationRule &ir = IntRules.Get(vel_fes->GetFE(0)->GetGeomType(),
                                            4 * order + 2);
   vdlfi->SetIntRule(&ir_ni);
   f_form->AddDomainIntegrator(vdlfi);

   ConstantCoefficient nlcoeff(-1.0);
   ParNonlinearForm *N = new ParNonlinearForm(vel_fes);
   N->AddDomainIntegrator(new VectorConvectionNLFIntegrator(nlcoeff));

   ParBilinearForm *Mv_form = new ParBilinearForm(vel_fes);
   VectorMassIntegrator *vmi = new VectorMassIntegrator;
   // vmi->SetIntRule(&ir_ni);
   Mv_form->AddDomainIntegrator(vmi);
   Mv_form->Assemble();
   Mv_form->Finalize();
   HypreParMatrix Mv; //  = *Mv_form->ParallelAssemble();
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
   VectorDivergenceIntegrator *vdi = new VectorDivergenceIntegrator;
   // vdi->SetIntRule(&ir_ni);
   D_form->AddDomainIntegrator(vdi);
   D_form->Assemble();
   D_form->Finalize();
   HypreParMatrix *D = D_form->ParallelAssemble();

   ParMixedBilinearForm *G_form = new ParMixedBilinearForm(pres_fes, vel_fes);
   VectorGradientIntegrator *vgi = new VectorGradientIntegrator;
   // vgi->SetIntRule(&ir_ni);
   G_form->AddDomainIntegrator(vgi);
   G_form->Assemble();
   G_form->Finalize();
   HypreParMatrix *G = G_form->ParallelAssemble();

   ParLinearForm *g_bdr_form = new ParLinearForm(pres_fes);
   g_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(*u_ex_coeff));

   VectorGridFunctionCoefficient uh_gf_coeff(&uh_gf);
   ParLinearForm *FText_bdr_form = new ParLinearForm(pres_fes);
   FText_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(uh_gf_coeff));

   ParLinearForm *f_bdr_form = new ParLinearForm(pres_fes);
   f_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(*forcing_coeff));

   ParGridFunction Fext_gf(vel_fes);
   VectorGridFunctionCoefficient Fext_gfcoeff(&Fext_gf);
   ParLinearForm *Fext_bdr_form = new ParLinearForm(pres_fes);
   Fext_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(Fext_gfcoeff));

   VectorGridFunctionCoefficient curlcurlu_gfcoeff(&curlcurlu_gf);
   ParLinearForm *curlcurlu_bdr_form = new ParLinearForm(pres_fes);
   curlcurlu_bdr_form->AddBoundaryIntegrator(
      new BoundaryNormalLFIntegrator(curlcurlu_gfcoeff, 2, 1));

   ConstantCoefficient lincoeff(ctx.kinvis);
   ConstantCoefficient bdfcoeff(1.0 / dt);
   ParBilinearForm *H_form = new ParBilinearForm(vel_fes);
   H_form->AddDomainIntegrator(new VectorMassIntegrator(bdfcoeff));
   H_form->AddDomainIntegrator(new VectorDiffusionIntegrator(lincoeff));
   H_form->Assemble();
   HypreParMatrix H;
   H_form->FormSystemMatrix(ess_tdof_list_u, H);

   HypreSmoother MvInvPC(Mv);
   MvInvPC.SetType(HypreSmoother::Jacobi, 1);
   CGSolver MvInv(MPI_COMM_WORLD);
   MvInv.iterative_mode = false;
   MvInv.SetPreconditioner(MvInvPC);
   MvInv.SetOperator(Mv);
   MvInv.SetPrintLevel(0);
   MvInv.SetRelTol(1e-12);
   MvInv.SetMaxIter(500);

   HypreBoomerAMG SpInvPC = HypreBoomerAMG(Sp);
   HYPRE_Solver amg_precond = static_cast<HYPRE_Solver>(SpInvPC);
   HYPRE_BoomerAMGSetCoarsenType(amg_precond, 6);
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
   HYPRE_BoomerAMGSetRelaxType(amg_precond, 6);
   HYPRE_BoomerAMGSetInterpType(amg_precond, 0);
   HYPRE_BoomerAMGSetPMaxElmts(amg_precond, 0);
   SpInvPC.SetPrintLevel(0);
   CGSolver SpInv(MPI_COMM_WORLD);
   SpInv.iterative_mode = false;
   SpInv.SetPreconditioner(SpInvPC);
   SpInv.SetOperator(Sp);
   SpInv.SetPrintLevel(0);
   SpInv.SetRelTol(pres_rtol);
   SpInv.SetMaxIter(500);

   HypreSmoother HInvPC(H);
   HInvPC.SetType(HypreSmoother::Jacobi, 1);
   CGSolver HInv(MPI_COMM_WORLD);
   HInv.iterative_mode = false;
   HInv.SetPreconditioner(HInvPC);
   HInv.SetOperator(H);
   HInv.SetPrintLevel(0);
   HInv.SetRelTol(helm_rtol);
   HInv.SetMaxIter(500);

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
   visit_dc.RegisterField("curlcurlu", &curlcurlu_gf);
   if (ctx.prob == 0 || ctx.prob == 2 || ctx.prob == 3)
   {
      u_err_gf = new ParGridFunction(vel_fes);
      p_err_gf = new ParGridFunction(pres_fes);
      visit_dc.RegisterField("velocity_error", u_err_gf);
      visit_dc.RegisterField("pressure_error", p_err_gf);
   }
   visit_dc.Save();

   // int order_quad = max(2, 2 * order + 1);
   int order_quad = 2 * (order + 1) - 3;
   const IntegrationRule *irs[Geometry::NumGeom];
   IntegrationRules irs_gll(0, Quadrature1D::GaussLobatto);
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(irs_gll.Get(i, order_quad));
   }

   ConstantCoefficient onecoeff(1.0);
   ParLinearForm mass_lf(pres_fes);
   mass_lf.AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
   mass_lf.Assemble();

   ParGridFunction one(pres_fes);
   one = 1.0;
   double sqrtvol = sqrt(mass_lf(one));

   double err_u;
   double err_p;
   double err_inf_u;
   double err_inf_p;
   double u_inf;
   double p_inf;

   if (ctx.prob == 0 || ctx.prob == 2 || ctx.prob == 3)
   {
      double err_u = u_gf.ComputeL2Error(*u_ex_coeff, irs) / sqrtvol;
      double err_p = p_gf.ComputeL2Error(*p_ex_coeff, irs) / sqrtvol;
      double err_inf_u = u_gf.ComputeMaxError(*u_ex_coeff, irs);
      double err_inf_p = p_gf.ComputeMaxError(*p_ex_coeff, irs);
      double u_inf = u_gf.Normlinf();
      double p_inf = p_gf.Normlinf();
      if (myid == 0)
      {
         printf("%.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E err\n",
                t,
                dt,
                err_u,
                err_p,
                err_inf_u,
                err_inf_p,
                u_inf,
                p_inf);
      }
   }
   Vector tmp1(vel_fes->GetTrueVSize());
   Vector tmp2(vel_fes->GetTrueVSize());
   Vector tmp3(vel_fes->GetTrueVSize());
   Vector resp(pres_fes->GetTrueVSize());
   Vector scrv(vel_fes->GetTrueVSize());
   Vector scrp(pres_fes->GetTrueVSize());

   double cfl = 0.0;

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

      //
      // Extrapolation
      //

      // BDFk/EXTk coefficients
      if (step == 0)
      {
         bd0 = 1.0;
         bd1 = -1.0;
         bd2 = 0.0;
         bd3 = 0.0;
         ab1 = 1.0;
         ab2 = 0.0;
         ab3 = 0.0;
      }
      else if (step == 1)
      {
         bd0 = 3.0 / 2.0;
         bd1 = -4.0 / 2.0;
         bd2 = 1.0 / 2.0;
         bd3 = 0.0;
         ab1 = 2.0;
         ab2 = -1.0;
         ab3 = 0.0;
      }
      else if (step == 2)
      {
         bd0 = 11.0 / 6.0;
         bd1 = -18.0 / 6.0;
         bd2 = 9.0 / 6.0;
         bd3 = -2.0 / 6.0;
         ab1 = 3.0;
         ab2 = -3.0;
         ab3 = 1.0;
      }
      if (step <= 2)
      {
         bdfcoeff.constant = bd0 / dt;
         H_form->Update();
         H_form->Assemble();
         H_form->FormSystemMatrix(ess_tdof_list_u, H);
         // HInv.SetOperator(H);
      }

      //
      // Forcing term
      //
      // This is an acceleration, not a force!
      //

      // Extrapolated f^{n+1}
      forcing_coeff->SetTime(t - dt);
      f_form->Assemble();
      f_form->ParallelAssemble(fn);
      forcing_coeff->SetTime(t);

      //
      // Nonlinear EXT terms
      //

      N->Mult(un, Nun);
      Nun.Add(1.0, fn);
      Fext.Set(ab1, Nun);
      Fext.Add(ab2, Nunm1);
      Fext.Add(ab3, Nunm2);

      Nunm2 = Nunm1;
      Nunm1 = Nun;

      // Fext = M^{-1} (F(u^{n}) + f^{n+1})
      MvInv.Mult(Fext, tmp1);
      Fext.Set(1.0, tmp1);

      // BDF terms
      Fext.Add(-bd1 / dt, un);
      Fext.Add(-bd2 / dt, unm1);
      Fext.Add(-bd3 / dt, unm2);

      //
      // Pressure poisson
      //

      // Lext = \nu CurlCurl(u^{n})
      if (enable_curl)
      {
         Lext.Set(ab1, un);
         Lext.Add(ab2, unm1);
         Lext.Add(ab3, unm2);
         uh_gf.SetFromTrueDofs(Lext);
         ComputeCurlCurl(uh_gf, curlcurlu_gf);
         curlcurlu_gf.GetTrueDofs(Lext);
         Lext *= ctx.kinvis;
      }

      // \tilde{F} = F - \nu CurlCurl(u)
      FText.Set(-1.0, Lext);
      FText.Add(1.0, Fext);

      // p_r = \nabla \cdot FText
      D->Mult(FText, resp);
      resp.Neg();

      // Add boundary terms
      uh_gf.SetFromTrueDofs(FText);
      FText_bdr_form->Assemble();
      FText_bdr_form->ParallelAssemble(FText_bdr);
      g_bdr_form->Assemble();
      g_bdr_form->ParallelAssemble(g_bdr);
      resp.Add(1.0, FText_bdr);
      resp.Add(-bd0 / dt, g_bdr);

      ParGridFunction pn_gf(pres_fes), resp_gf(pres_fes);
      pn_gf = 0.0;
      resp_gf = 0.0;

      if (!(ctx.prob == 5))
      {
         ortho(resp);
      }

      pres_fes->GetRestrictionMatrix()->MultTranspose(resp, resp_gf);

      Vector X1, B1;
      Sp_form->FormLinearSystem(ess_tdof_list_p, pn_gf, resp_gf, Sp, X1, B1);
      SpInv.Mult(B1, X1);
      Sp_form->RecoverFEMSolution(X1, resp_gf, pn_gf);

      if (!(ctx.prob == 5))
      {
         MeanZero(pn_gf);
      }

      pn_gf.GetTrueDofs(pn);
      p_gf.Distribute(pn);

      //
      // Project velocity
      //

      G->Mult(pn, resu);
      resu.Neg();
      Mv.Mult(Fext, tmp1);
      resu.Add(1.0, tmp1);

      //
      // Helmholtz
      //

      ParGridFunction resu_gf(vel_fes);
      un_gf = 0.0;
      resu_gf = 0.0;
      if (ctx.prob == 5)
      {
         un_gf.ProjectBdrCoefficient(*u_ex_coeff, ess_bdr_attr_u);
      }
      else
      {
         un_gf.ProjectBdrCoefficient(*u_ex_coeff, ess_bdr_attr_u);
      }

      vel_fes->GetRestrictionMatrix()->MultTranspose(resu, resu_gf);

      unm2 = unm1;
      unm1 = un;

      Vector X2, B2;
      H_form->FormLinearSystem(ess_tdof_list_u, un_gf, resu_gf, H, X2, B2);
      HInv.Mult(B2, X2);
      H_form->RecoverFEMSolution(X2, resu_gf, un_gf);

      un_gf.GetTrueDofs(un);
      u_gf.Distribute(un);

      cfl = ComputeCFL(u_gf, dt);

      if (mpi.Root() && cfl > 0.5)
      {
         printf("*** WARNING CFL = %.5E\n", cfl);
      }

      if ((step + 1) % 100 == 0 || last_step)
      {
         u_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n"
                << *pmesh << u_gf << "window_title 'velocity t=" << t << "'"
                << endl;

         p_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n"
                << *pmesh << p_gf << "window_title 'pressure t=" << t << "'"
                << endl;

         if (ctx.prob == 0 || ctx.prob == 2 || ctx.prob == 3)
         {
            u_err_gf->ProjectCoefficient(*u_ex_coeff);
            p_err_gf->ProjectCoefficient(*p_ex_coeff);
            subtract(u_gf, *u_err_gf, *u_err_gf);
            subtract(p_gf, *p_err_gf, *p_err_gf);
         }
         visit_dc.SetCycle(step + 1);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }

      if (ctx.prob == 0 || ctx.prob == 2 || ctx.prob == 3)
      {
         err_u = u_gf.ComputeL2Error(*u_ex_coeff, irs) / sqrtvol;
         err_p = p_gf.ComputeL2Error(*p_ex_coeff, irs) / sqrtvol;
         err_inf_u = u_gf.ComputeMaxError(*u_ex_coeff, irs);
         err_inf_p = p_gf.ComputeMaxError(*p_ex_coeff, irs);
         double u_inf_loc = u_gf.Normlinf();
         u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
         double p_inf_loc = p_gf.Normlinf();
         p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);
         if (myid == 0)
         {
            printf("%.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E err\n",
                   t,
                   dt,
                   err_u,
                   err_p,
                   err_inf_u,
                   err_inf_p,
                   u_inf,
                   p_inf);
            fflush(stdout);
         }
      }
      else
      {
         double u_inf_loc = u_gf.Normlinf();
         u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);

         if (myid == 0)
         {
            printf("%.5E %.5E %.5E %.5E\n", t, dt, u_inf, cfl);
            fflush(stdout);
         }
      }
   }

   return 0;
}
