#include "lib/navier_solver.hpp"
#include "kernels/boundary_normal_stress_integrator.hpp"
#include "kernels/boundary_normal_pressure_integrator.hpp"
#include "kernels/boundary_normal_stress_evaluator.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

class MeshLaplacian
{
public:
   MeshLaplacian(ParFiniteElementSpace &fes, Array<int> &ess_bdr) :
      laplacian_form(&fes),
      b(fes.GetVSize()),
      ess_bdr(ess_bdr),
      cg(MPI_COMM_WORLD)
   {
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      laplacian_form.AddDomainIntegrator(new VectorDiffusionIntegrator);
      laplacian_form.Assemble();

      cg.SetRelTol(1e-8);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(2);
      cg.SetPreconditioner(pc);
   }

   void Solve(ParGridFunction &x)
   {
      b = 0.0;

      laplacian_form.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetOperator(*A);
      cg.Mult(B, X);

      laplacian_form.RecoverFEMSolution(X, b, x);
   }

   ParBilinearForm laplacian_form;
   Vector b;
   Array<int> &ess_bdr;
   CGSolver cg;
   Array<int> ess_tdof_list;
   Vector X, B;
   OperatorPtr A;
   OperatorJacobiSmoother pc;
};

// Assuming mesh with polynomial order equal to destination space in H1 and
// ordering byNODES
class BoundaryFieldTransfer
{
public:
   enum class Backend {Native, GSLib};
   BoundaryFieldTransfer(ParGridFunction &src_gf, ParGridFunction &dst_gf,
                         ParFiniteElementSpace &dst_fes,
                         Array<int> &dst_bdr_attr,
                         BoundaryFieldTransfer::Backend backend_ = BoundaryFieldTransfer::Backend::GSLib)
      :
      src_mesh(*(src_gf.ParFESpace()->GetParMesh())),
      dst_mesh(*(dst_gf.ParFESpace()->GetParMesh())),
      dst_fes(dst_fes),
      dst_bdr_attr(dst_bdr_attr),
      dim(dst_mesh.Dimension()),
      finder(MPI_COMM_WORLD),
      backend(backend_)
   {
      if (backend == BoundaryFieldTransfer::Backend::GSLib)
      {
         Vector dst_mesh_nodes = dst_mesh.GetNodes()->GetTrueVector();

         dst_fes.GetEssentialTrueDofs(dst_bdr_attr,
                                      bdr_tdof_list,
                                      0);

         Vector bnd_coords(bdr_tdof_list.Size() * dim);
         for (int i = 0; i < bdr_tdof_list.Size(); i++)
         {
            int idx = bdr_tdof_list[i] / (dst_fes.GetOrdering() ? dim : 1.0);
            for (int d = 0; d < dim; d++)
            {
               bnd_coords(i + d*bnd_coords.Size()/dim) =
                  dst_mesh_nodes(idx + d*dst_mesh_nodes.Size()/dim);
            }
         }

         finder.Setup(src_mesh);
         finder.FindPoints(bnd_coords);
         const Array<unsigned int> &code_out_solid = finder.GetCode();
         for (int i = 0; i < code_out_solid.Size(); i++)
         {
            if (code_out_solid[i] != 2)
            {
               dst_tdofs_found_on_coords.Append(bdr_tdof_list[i]);
            }
         }

         dst_interp_coords.SetSize(dst_tdofs_found_on_coords.Size()*dim);
         for (int i = 0; i < dst_tdofs_found_on_coords.Size(); i++)
         {
            int idx = dst_tdofs_found_on_coords[i] / (dst_fes.GetOrdering() ? dim : 1.0);;
            for (int d = 0; d < dim; d++)
            {
               dst_interp_coords(i + d*dst_interp_coords.Size()/dim) =
                  dst_mesh_nodes(idx + d*dst_mesh_nodes.Size()/dim);
            }
         }
      }
      else
      {
         native_transfer.reset(new ParTransferMap(src_gf, dst_gf));
      }
   }

   ~BoundaryFieldTransfer()
   {
      if (backend == BoundaryFieldTransfer::Backend::GSLib)
      {
         finder.FreeData();
      }
   }

   void Interpolate(ParGridFunction &src_gf, ParGridFunction &dst_gf)
   {
      if (backend == BoundaryFieldTransfer::Backend::GSLib)
      {
         finder.Interpolate(src_gf, interp_vals);

         if (src_gf.FESpace()->GetOrdering() == Ordering::byVDIM &&
             dst_gf.FESpace()->GetOrdering() == Ordering::byNODES)
         {
            for (int i = 0; i < dst_tdofs_found_on_coords.Size(); i++)
            {
               /// XXXYYYZZZ <- XYZXYZ
               const int idx = dst_tdofs_found_on_coords[i];
               for (int d = 0; d < dim; d++)
               {
                  dst_gf(idx + d*dst_gf.Size()/dim) = interp_vals[i * dim + d];
               }
            }
         }
         else if (src_gf.FESpace()->GetOrdering() == Ordering::byNODES &&
                  dst_gf.FESpace()->GetOrdering() == Ordering::byVDIM)
         {
            for (int i = 0; i < dst_tdofs_found_on_coords.Size(); i++)
            {
               /// XYZXYZ <- XXXYYYZZZ
               const int idx = dst_tdofs_found_on_coords[i] / (dst_fes.GetOrdering() ? dim :
                                                               1.0);
               for (int d = 0; d < dim; d++)
               {
                  dst_gf(idx * dim + d) = interp_vals[i + d * interp_vals.Size()/dim];
               }
            }
         }
         else
         {
            MFEM_ABORT("not implemented");
         }
      }
      else
      {
         native_transfer->Transfer(src_gf,dst_gf);
      }
   }

   ParMesh &src_mesh;
   ParMesh &dst_mesh;
   ParFiniteElementSpace &dst_fes;
   Array<int> &dst_bdr_attr, bdr_tdof_list;
   const int dim;
   Array<int> dst_tdofs_found_on_coords;
   Vector dst_interp_coords;
   FindPointsGSLIB finder;
   Vector interp_vals;
   BoundaryFieldTransfer::Backend backend;
   std::unique_ptr<ParTransferMap> native_transfer;
};

class Elasticity : public SecondOrderTimeDependentOperator
{
public:
   Elasticity(ParMesh &mesh, const int polynomial_order, const double density,
              Array<int> &ess_bdr, Array<int> &neumann_bdr, const double lambda, const double mu) :
      mesh(mesh),
      dim(mesh.Dimension()),
      ess_bdr(ess_bdr),
      neumann_bdr(neumann_bdr),
      fec(polynomial_order),
      fes(&mesh, &fec, dim, Ordering::byVDIM),
      Mform(&fes),
      Kform(&fes),
      Minv(MPI_COMM_WORLD),
      Ainv(MPI_COMM_WORLD),
      density(density)
   {
      this->height = fes.GetTrueVSize();
      this->width = this->height;
      acc.SetSize(fes.GetTrueVSize());
      disp_pred.SetSize(fes.GetTrueVSize());
      vel_pred.SetSize(fes.GetTrueVSize());
      F.SetSize(fes.GetTrueVSize());
      z.SetSize(fes.GetTrueVSize());

      printf("Elasticity #tdofs: %d\n", fes.GetTrueVSize());

      acc = 0.0;

      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      Array<int> empty;

      density_coef.constant = density;

      // fcoeff = new VectorFunctionCoefficient(dim, [density](const Vector &, Vector &u)
      // {
      //    u(0) = 0.0;
      //    u(1) = -2.0*density;
      // });

      // Fform = new ParLinearForm(&fes);
      // Fform->AddDomainIntegrator(new VectorDomainLFIntegrator(*fcoeff));
      // Fform->Assemble();
      // Fform->ParallelAssemble(F);

      Mform.AddDomainIntegrator(new VectorMassIntegrator(density_coef));
      Mform.Assemble(0);
      Mform.FormSystemMatrix(ess_tdof_list, M);

      mu_coef.constant = mu;
      lambda_coef.constant = lambda;


      Kform.AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
      Kform.Assemble(0);
      Kform.FormSystemMatrix(empty, K);
      Kform.FormSystemMatrix(ess_tdof_list, K0);

      MinvPC.reset(new HypreSmoother);
      Minv.iterative_mode = false;
      Minv.SetRelTol(1e-12);
      Minv.SetMaxIter(500);
      Minv.SetPreconditioner(*MinvPC);
      Minv.SetOperator(M);

      AinvPC.SetPrintLevel(0);
      AinvPC.SetElasticityOptions(&fes);

      Ainv.SetRelTol(1e-12);
      Ainv.SetPrintLevel(0);
      Ainv.SetMaxIter(500);
      Ainv.SetPreconditioner(AinvPC);
   }

   void Mult(const Vector &u, const Vector &du_dt,
             Vector &d2udt2) const
   {
      K.Mult(u, z);
      z.Neg();
      z.Add(1.0, F);
      Minv.Mult(z, d2udt2);
   }

   void ImplicitSolve(const double fac0, const double fac1,
                      const Vector &u, const Vector &dudt, Vector &d2udt2)
   {
      if (A == nullptr)
      {
         A = Add(1.0, M, fac0, K);
         Ainv.SetOperator(*A);
         A->EliminateBC(ess_tdof_list, DiagonalPolicy::DIAG_ONE);
      }
      K0.Mult(u, z);
      z.Neg();
      z.Add(1.0, F);

      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         z[ess_tdof_list[i]] = 0.0;
      }
      Ainv.Mult(z, d2udt2);
   }

   void SetBodyTraction(const ParGridFunction &traction_gf)
   {
      fcoeff.reset(new VectorGridFunctionCoefficient(&traction_gf));
      scaled_fcoeff.reset(new ScalarVectorProductCoefficient(-1.0, *fcoeff));
      Fform.reset(new ParLinearForm(&fes));
      Fform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*scaled_fcoeff), neumann_bdr);
      Fform->Assemble();
      Fform->ParallelAssemble(F);
   }

   void Update()
   {
      delete A;
      A = nullptr;

      Fform->Update();
      Fform->Assemble();
      Fform->ParallelAssemble(F);
   }

   ParMesh &mesh;
   const int dim;
   Array<int> &ess_bdr;
   Array<int> &neumann_bdr;
   Array<int> ess_tdof_list;
   H1_FECollection fec;
   ParFiniteElementSpace fes;

   ParBilinearForm Mform;
   ParBilinearForm Kform;
   std::unique_ptr<ParLinearForm> Fform;

   ConstantCoefficient lambda_coef, mu_coef, density_coef;
   std::unique_ptr<VectorCoefficient> fcoeff;
   std::unique_ptr<VectorCoefficient> scaled_fcoeff;

   HypreParMatrix M, K, K0, *A = nullptr;

   CGSolver Minv;
   std::unique_ptr<Solver> MinvPC;
   GMRESSolver Ainv;
   HypreBoomerAMG AinvPC;

   mutable Vector acc, disp_pred, vel_pred, z, F;

   const double beta = 0.25;
   const double gamma = 0.5;
   double current_dt = -1.0;
   const double density;
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../data/star.mesh";
   int polynomial_order = 2;
   double fluid_density = 1e3;
   double kinematic_viscosity = 1e-3;
   double U = 0.2;
   double time = 0.0;
   double t_final = 0.5;
   double dt = 1e-5;
   double dt_max = 1e-1;
   double cfl_target = 0.5;
   double cfl_tol = 1e-4;
   int max_bdf_order = 3;
   int current_step = 0;
   bool last_step = false;
   double solid_density = 1.0e3;
   int enable_retry = 1;
   double max_elem_error = 5.0e-3;
   double solid_mu = 2.0e6;
   double solid_nu = 0.4;
   double solid_lambda;
   double hysteresis = 0.15; // derefinement safety coefficient
   int nc_limit = 3;
   int refinements = 0;
   int csm_only = 0;
   int cfd_only = 0;

   const char *device_config = "cpu";
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&polynomial_order, "-o", "--order", "polynomial_order");
   args.AddOption(&t_final, "-tf", "--t_final", "t_final");
   args.AddOption(&refinements, "-r", "--r", "refinements");
   args.AddOption(&max_bdf_order, "-mbdf", "--max_bdf_order", "max_bdf_order");
   args.AddOption(&enable_retry, "-enable_retry", "--enable_retry",
                  "enable_retry");
   args.AddOption(&U, "-inflow_velocity", "--inflow_velocity",
                  "inflow_velocity");
   args.AddOption(&solid_mu, "-sh", "--shear_modulus",
                  "Shear modulus/rigidity");
   args.AddOption(&max_elem_error, "-e", "--max-err",
                  "Maximum element error");
   args.AddOption(&hysteresis, "-y", "--hysteresis",
                  "Derefinement safety coefficient.");
   args.AddOption(&csm_only, "-csm_only", "--csm_only",
                  "Run only CSM.");
   args.AddOption(&cfd_only, "-cfd_only", "--cfd_only",
                  "Run only CFD.");
   args.AddOption(&dt, "-dt", "--dt",
                  "dt.");
   args.ParseCheck();

   if(cfd_only){
      solid_density = 1.0e6;
      solid_mu = 1.0e12;
   }
   solid_lambda = 2*solid_mu*solid_nu/(1-2*solid_nu);

   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();

   for (int i = 0; i < refinements; i++)
   {
      mesh.UniformRefinement();
   }

   mesh.SetCurvature(polynomial_order, false, dim, Ordering::byNODES);

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.SetAttributes();
   pmesh.EnsureNodes();

   Array<int> fluid_domain_attributes(1);
   fluid_domain_attributes[0] = 1;
   auto fluid_mesh = ParSubMesh::CreateFromDomain(pmesh, fluid_domain_attributes);
   fluid_mesh.SetAttributes();
   fluid_mesh.EnsureNodes();

   // Create the flow solver.
   NavierSolver navier(&fluid_mesh, polynomial_order, kinematic_viscosity, fluid_density);
   navier.SetMaxBDFOrder(max_bdf_order);

   ParGridFunction *u_gf = navier.GetCurrentVelocity();
   ParGridFunction *p_gf = navier.GetCurrentPressure();
   ParGridFunction *nu_gf = navier.GetVariableViscosity();
   ParGridFunction *w_gf = navier.GetCurrentMeshVelocity();
   ParGridFunction fluid_mesh_nodes_old(*static_cast<ParGridFunction *>(fluid_mesh.GetNodes()));

   ParGridFunction sigmaN_gf(u_gf->ParFESpace());
   sigmaN_gf = 0.;
   ParGridFunction fluid_mesh_dx_gf(static_cast<ParGridFunction *>(fluid_mesh.GetNodes())->ParFESpace());

   VectorGridFunctionCoefficient fluid_mesh_dx_coef(&fluid_mesh_dx_gf);

   Array<int> no_slip(fluid_mesh.bdr_attributes.Max());
   no_slip = 0;
   no_slip[2] = 1;
   no_slip[3] = 1;
   no_slip[4] = 1;
   no_slip[5] = 1;
   navier.AddVelDirichletBC([](const Vector &, const double, Vector &u)
   {
      u(0) = 0.0;
      u(1) = 0.0;
   }, no_slip);

   // Becomes owned by navier
   auto *inlet_coeff = new VectorFunctionCoefficient(dim, [&U](
                                                        const Vector &coords,
                                                        const double time,
                                                        Vector &u)
   {
      // const double x = coords(0);
      const double y = coords(1);
      const double H = 0.41;
      const double ramp_peak_at = 2.0;
      const double ramp = 0.5 * (1.0 - cos(M_PI / ramp_peak_at * time));
      if (time < ramp_peak_at)
      {
         u(0) = U * 1.5 * y * (H - y) / pow(H / 2.0, 2.0) * ramp;
      }
      else
      {
         u(0) = U * 1.5 * y * (H - y) / pow(H / 2.0, 2.0);
      }
      u(1) = 0.0;
   });

   Array<int> inlet(fluid_mesh.bdr_attributes.Max());
   inlet = 0;
   inlet[0] = 1;
   u_gf->ProjectBdrCoefficient(*inlet_coeff, inlet);
   navier.AddVelDirichletBC(inlet_coeff, inlet);

   Array<int> outlet_attr(fluid_mesh.bdr_attributes.Max());
   outlet_attr = 0;
   outlet_attr[1] = 1;
   navier.AddPresDirichletBC([](const Vector &, double t)
   {
      return 0.0;
   }, outlet_attr);

   navier.Setup(dt);

   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "parallel " << Mpi::WorldSize() << " "
            << Mpi::WorldRank() << "\n";
   sol_sock << "solution\n" << fluid_mesh << *u_gf << std::flush;

   Array<int> solid_interface_attr(fluid_mesh.bdr_attributes.Max());
   solid_interface_attr = 0;
   solid_interface_attr[4] = 1;
   solid_interface_attr[5] = 1;

   ParLinearForm lift_drag_form(u_gf->ParFESpace());
   auto stress_integ = new BoundaryNormalStressIntegrator(*u_gf, *p_gf, *nu_gf, fluid_density);
   stress_integ->SetIntRule(&navier.gll_ir_face);
   lift_drag_form.AddBoundaryIntegrator(stress_integ,
                                        solid_interface_attr);

   auto sigmaN_coeff = VectorGridFunctionCoefficient(&sigmaN_gf);
   ParLinearForm lift_drag_form2(u_gf->ParFESpace());
   auto stress_integ2 = new VectorBoundaryLFIntegrator(sigmaN_coeff);
   stress_integ2->SetIntRule(&navier.gll_ir_face);
   lift_drag_form2.AddBoundaryIntegrator(stress_integ2,
                                         solid_interface_attr);

   Array<int> solid_domain_attributes(1);
   solid_domain_attributes[0] = 2;
   auto solid_mesh = ParSubMesh::CreateFromDomain(pmesh, solid_domain_attributes);
   solid_mesh.SetAttributes();
   solid_mesh.EnsureNodes();

   Array<int> solid_fluid_interface_attr(solid_mesh.bdr_attributes.Max());
   solid_fluid_interface_attr = 0;
   solid_fluid_interface_attr[4] = 1;

   Array<int> solid_ess_bdr(solid_mesh.bdr_attributes.Max());
   solid_ess_bdr = 0;
   solid_ess_bdr[6] = 1;

   Elasticity elasticity(solid_mesh, polynomial_order, solid_density,
                         solid_ess_bdr, solid_fluid_interface_attr, solid_lambda, solid_mu);

   Vector solid_displacement;
   ParGridFunction solid_displacement_gf(&elasticity.fes);
   solid_displacement_gf = 0.0;
   solid_displacement_gf.GetTrueDofs(solid_displacement);

   Vector solid_vel;
   ParGridFunction solid_vel_gf(&elasticity.fes);
   solid_vel_gf = 0.0;
   solid_vel_gf.GetTrueDofs(solid_vel);

   ParGridFunction solid_sigmaN_gf(&elasticity.fes);
   solid_sigmaN_gf = 0.0;

   elasticity.SetBodyTraction(solid_sigmaN_gf);

   HHTAlphaSolver hht;
   hht.Init(elasticity);

   ParGridFunction fluid_mesh_velocity_gf(static_cast<ParGridFunction *>
                                          (fluid_mesh.GetNodes())->ParFESpace());
   fluid_mesh_velocity_gf = 0.0;

   Array<int> fluid_mesh_laplacian_ess_bdr(fluid_mesh.bdr_attributes.Max());
   fluid_mesh_laplacian_ess_bdr = 1;

   MeshLaplacian fluid_mesh_laplacian(*static_cast<ParGridFunction *>
                                          (fluid_mesh.GetNodes())->ParFESpace(),
                                      fluid_mesh_laplacian_ess_bdr);

   ParaViewDataCollection pvdc_solid("solid_output", &solid_mesh);
   pvdc_solid.SetDataFormat(VTKFormat::BINARY32);
   pvdc_solid.SetHighOrderOutput(true);
   pvdc_solid.SetLevelsOfDetail(polynomial_order);
   pvdc_solid.SetCycle(current_step);
   pvdc_solid.SetTime(time);
   pvdc_solid.RegisterField("displacement", &solid_displacement_gf);
   pvdc_solid.RegisterField("velocity", &solid_vel_gf);
   pvdc_solid.Save();

   if (csm_only)
   {
      auto gravitational_force = [](const Vector &, Vector &u)
      {
         u(0) = 0.0;
         u(1) = -2.0;
      };
      VectorFunctionCoefficient gravitational_force_coeff(dim, gravitational_force);
      solid_sigmaN_gf.ProjectCoefficient(gravitational_force_coeff);
      double max_disp_y = 0.0;
      for (int step = 0; !last_step; ++step)
      {
         if (time + dt >= t_final - dt / 2)
         {
            last_step = true;
         }

         elasticity.Update();
         hht.Step(solid_displacement, solid_vel, time, dt);
         solid_displacement_gf.SetFromTrueDofs(solid_displacement);
         solid_vel_gf.SetFromTrueDofs(solid_vel);

         Vector point(2);
         point(0) = 0.6;
         point(1) = 0.2;
         DenseMatrix points(dim, 1);
         points.SetCol(0, point);

         Array<int> elem_ids;
         Array<IntegrationPoint> ips;

         solid_mesh.FindPoints(points, elem_ids, ips);

         Vector val;
         solid_displacement_gf.GetVectorValue(elem_ids[0], ips[0], val);

         double disp_x = val(0);
         double disp_y = val(1);

         max_disp_y = std::max(abs(max_disp_y), abs(val(1)));

         if ((step + 1) % 10 == 0 || last_step)
         {
            printf("%d %1.3E %1.3E %1.3E %1.3E\n", step, time, dt, disp_x, disp_y);
            time += dt;
            pvdc_solid.SetCycle(step);
            pvdc_solid.SetTime(time);
            pvdc_solid.Save();
         }
      }
      printf("max y-displacement = %1.3E\n", max_disp_y);
      exit(0);
   }

   BoundaryFieldTransfer::Backend transfer_backend =
      BoundaryFieldTransfer::Backend::Native;

   BoundaryFieldTransfer solid_fluid_bdr_transfer(solid_vel_gf, *u_gf,
                                                  *u_gf->ParFESpace(),
                                                  solid_fluid_interface_attr,
                                                  transfer_backend);

   printf("%d %d\n", fluid_mesh.bdr_attributes.Max(),
          solid_mesh.bdr_attributes.Max());

   BoundaryFieldTransfer fluid_solid_bdr_transfer(*u_gf, solid_vel_gf,
                                                  *solid_vel_gf.ParFESpace(),
                                                  solid_fluid_interface_attr,
                                                  transfer_backend);

   ParaViewDataCollection pvdc("fluid_output", &fluid_mesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(polynomial_order);
   pvdc.SetCycle(current_step);
   pvdc.SetTime(time);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.RegisterField("sigmaN", &sigmaN_gf);
   pvdc.RegisterField("mesh_velocity", &fluid_mesh_velocity_gf);
   pvdc.Save();

   int step = 0;
   int retry_step = 0;

   while (true)
   {
      if (time + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      double previous_dt = dt;

      double lift = 0.0, lift2 = 0.0;
      double drag = 0.0, drag2 = 0.0;

      navier.Step(time, dt, step, true);

      // Retrieve the computed provisional velocity
      auto u_next_gf = navier.GetProvisionalVelocity();

      // Compute the CFL based on the provisional velocity
      double cfl = navier.ComputeCFL(*u_next_gf, dt);

      double error_est = cfl / (cfl_target + cfl_tol);
      if (error_est >= 1.0 && enable_retry)
      {
         // Reject the time step
         retry_step = 1;
         dt *= 0.125;
         if (Mpi::Root())
         {
            printf(">>> Step %d reached maximum CFL or predicted CFL, retrying with smaller step size."
                   " current dt = %1.3E, retry dt = %1.3E\n", step, previous_dt, dt);
         }
      }
      else
      {
         retry_step = 0;

         // Queue new time step in the history array
         time += dt;
         navier.UpdateTimestepHistory(dt);

         {
            lift_drag_form.Assemble();

            for (int i = 0; i < lift_drag_form.FESpace()->GetNDofs(); i++)
            {
               int idx_x = Ordering::Map<Ordering::byNODES>(
                              lift_drag_form.FESpace()->GetNDofs(),
                              lift_drag_form.FESpace()->GetVDim(),
                              i,
                              0);
               drag += lift_drag_form[idx_x];

               int idx_y = Ordering::Map<Ordering::byNODES>(
                              lift_drag_form.FESpace()->GetNDofs(),
                              lift_drag_form.FESpace()->GetVDim(),
                              i,
                              1);
               lift += lift_drag_form[idx_y];
            }
         }
         {
            BoundaryNormalStressEvaluator(*u_gf, *p_gf, *nu_gf,
                                          solid_interface_attr,
                                          navier.gll_ir_face,
                                          sigmaN_gf, fluid_density);

            lift_drag_form2.Assemble();

            for (int i = 0; i < lift_drag_form2.FESpace()->GetNDofs(); i++)
            {
               int idx_x = Ordering::Map<Ordering::byNODES>(
                              lift_drag_form2.FESpace()->GetNDofs(),
                              lift_drag_form2.FESpace()->GetVDim(),
                              i,
                              0);
               drag2 += lift_drag_form2[idx_x];

               int idx_y = Ordering::Map<Ordering::byNODES>(
                              lift_drag_form2.FESpace()->GetNDofs(),
                              lift_drag_form2.FESpace()->GetVDim(),
                              i,
                              1);
               lift2 += lift_drag_form2[idx_y];
            }
         }

         {
            if (Mpi::Root())
            {
               printf("interpolating traction from fluid to solid\n");
            }
            fluid_solid_bdr_transfer.Interpolate(sigmaN_gf, solid_sigmaN_gf);

            if (Mpi::Root())
            {
               printf("solving solid\n");
            }
            elasticity.Update();
            hht.Step(solid_displacement, solid_vel, time, dt);
            solid_displacement_gf.SetFromTrueDofs(solid_displacement);
            solid_vel_gf.SetFromTrueDofs(solid_vel);

            if (Mpi::Root())
            {
               printf("interpolating velocity and displacement to fluid\n");
            }
            solid_fluid_bdr_transfer.Interpolate(solid_vel_gf, *u_gf);

            fluid_mesh_velocity_gf = 0.0;
            solid_fluid_bdr_transfer.Interpolate(solid_vel_gf, fluid_mesh_velocity_gf);
            fluid_mesh_laplacian.Solve(fluid_mesh_velocity_gf);

            fluid_mesh_dx_gf = *fluid_mesh.GetNodes();

            fluid_mesh_dx_gf.Add(dt, fluid_mesh_velocity_gf);

            const double fluid_mesh_velocity_norm = fluid_mesh_velocity_gf.Norml2();
            if (Mpi::Root())
            {
               printf("moving fluid mesh |dx|_l2 = %.3E\n", fluid_mesh_velocity_norm*dt);
            }

            *w_gf = fluid_mesh_velocity_gf;
            navier.TransformMesh(fluid_mesh_dx_coef);
         }
         // @TODO update findpts?

         if ((step + 1) % 10 == 0 || last_step)
         {
            if (Mpi::Root())
            {
               out << "writing output...\n";
            }
            pvdc.SetCycle(step);
            pvdc.SetTime(time);
            pvdc.Save();
            pvdc_solid.SetCycle(step);
            pvdc_solid.SetTime(time);
            pvdc_solid.Save();
         }

         // Predict new step size
         double fac_safety = 2.0;
         double eta = pow(1.0 / (fac_safety * error_est), 1.0 / (1.0 + 3.0));
         double fac_min = 0.1;
         double fac_max = 1.4;
         dt = dt * std::min(fac_max, std::max(fac_min, eta));
         dt = std::min(dt, dt_max);

         step++;
      }


      Vector point(2);
      point(0) = 0.2;
      point(1) = 0.2875;
      DenseMatrix points(dim, 1);
      points.SetCol(0, point);

      Array<int> elem_ids;
      Array<IntegrationPoint> ips;

      fluid_mesh.FindPoints(points, elem_ids, ips);
      Vector val;
      u_gf->GetVectorValue(elem_ids[0], ips[0], val);

      point(0) = 0.1;
      point(1) = 0.2;
      fluid_mesh.FindPoints(points, elem_ids, ips);
      double p_probe = p_gf->GetValue(elem_ids[0], ips[0]);

      point(0) = 0.6;
      point(1) = 0.2;
      points.SetCol(0, point);
      solid_mesh.FindPoints(points, elem_ids, ips);

      solid_displacement_gf.GetVectorValue(elem_ids[0], ips[0], val);

      double disp_x = val(0);
      double disp_y = val(1);

      if (Mpi::Root())
      {
         printf("%d %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %d log\n",
                step, time, dt, -lift, -drag, -lift2, -drag2, cfl, val.Norml2(), p_probe, disp_x, disp_y,
                retry_step);
      }

      if (last_step)
      {
         break;
      }
   }

   navier.PrintTimingData();

   return 0;
}
