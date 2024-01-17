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
   MeshLaplacian(FiniteElementSpace &fes, Array<int> &ess_bdr) :
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

   void Solve(GridFunction &x)
   {
      b = 0.0;

      laplacian_form.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetOperator(A);
      cg.Mult(B, X);

      laplacian_form.RecoverFEMSolution(X, b, x);
   }

   BilinearForm laplacian_form;
   Vector b;
   Array<int> &ess_bdr;
   CGSolver cg;
   Array<int> ess_tdof_list;
   Vector X, B;
   SparseMatrix A;
   GSSmoother pc;
};

// Assuming mesh with polynomial order equal to destination space in H1 and
// ordering byNODES
class BoundaryFieldTransfer
{
public:
   BoundaryFieldTransfer(ParMesh &src_mesh, ParMesh &dst_mesh,
                         ParFiniteElementSpace &dst_fes,
                         Array<int> &dst_bdr_attr) :
      src_mesh(src_mesh),
      dst_mesh(dst_mesh),
      dst_fes(dst_fes),
      dst_bdr_attr(dst_bdr_attr),
      dim(dst_mesh.Dimension()),
      finder(MPI_COMM_WORLD)
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

   ~BoundaryFieldTransfer()
   {
      finder.FreeData();
   }

   void Interpolate(GridFunction &src_gf, GridFunction &dst_gf)
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

   ParMesh &src_mesh;
   ParMesh &dst_mesh;
   ParFiniteElementSpace &dst_fes;
   Array<int> &dst_bdr_attr, bdr_tdof_list;
   const int dim;
   Array<int> dst_tdofs_found_on_coords;
   Vector dst_interp_coords;
   FindPointsGSLIB finder;
   Vector interp_vals;
};

class Elasticity : public SecondOrderTimeDependentOperator
{
public:
   Elasticity(ParMesh &mesh, const int polynomial_order, const double density,
              Array<int> &ess_bdr) :
      mesh(mesh),
      dim(mesh.Dimension()),
      ess_bdr(ess_bdr),
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

      {
         const double E = 5.6 * 1e6; // CSM2
         // const double E = 1.4 * 1e6; // CSM3
         const double nu = 0.4;
         const double mu = E / (2*(1+nu)); // CSM2
         const double lambda = nu * E / ((1+nu) * (1-2*nu));
         mu_coef.constant = mu;
         lambda_coef.constant = lambda;
         printf("lambda_coeff = %.3E mu = %.3E\n", lambda_coef.constant,
                mu_coef.constant);
      }

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

   void SetBoundaryTraction(const ParGridFunction &traction_gf)
   {
      fcoeff.reset(new VectorGridFunctionCoefficient(&traction_gf));
      scaled_fcoeff.reset(new ScalarVectorProductCoefficient(density, *fcoeff));
      Fform.reset(new ParLinearForm(&fes));
      Fform->AddDomainIntegrator(new VectorDomainLFIntegrator(*scaled_fcoeff));
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
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int polynomial_order = 2;
   double density_fluid = 1e3;
   double kinematic_viscosity = 1e-3;
   double U = 0.2;
   double time = 0.0;
   double t_final = 0.5;
   double dt = 1e-5;
   double dt_max = 1e-1;
   double cfl_target = 1.0;
   double cfl_tol = 1e-4;
   int max_bdf_order = 3;
   int current_step = 0;
   bool last_step = false;
   double solid_density = 1e3;
   int enable_retry = 1;
   double max_elem_error = 5.0e-3;
   double hysteresis = 0.15; // derefinement safety coefficient
   int nc_limit = 3;

   const char *device_config = "cpu";
   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&polynomial_order, "-o", "--order", "polynomial_order");
   args.AddOption(&t_final, "-tf", "--t_final", "t_final");
   args.AddOption(&max_bdf_order, "-mbdf", "--max_bdf_order", "max_bdf_order");
   args.AddOption(&enable_retry, "-enable_retry", "--enable_retry",
                  "enable_retry");
   args.AddOption(&U, "-inflow_velocity", "--inflow_velocity",
                  "inflow_velocity");
   args.AddOption(&max_elem_error, "-e", "--max-err",
                  "Maximum element error");
   args.AddOption(&hysteresis, "-y", "--hysteresis",
                  "Derefinement safety coefficient.");
   args.ParseCheck();

   Mesh mesh("fsi.msh");
   const int dim = mesh.Dimension();

   // mesh.UniformRefinement();
   // mesh.UniformRefinement();

   mesh.SetCurvature(polynomial_order, false, dim, Ordering::byNODES);

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   Array<int> fluid_domain_attributes(1);
   fluid_domain_attributes[0] = 1;
   auto fluid_mesh = ParSubMesh::CreateFromDomain(pmesh, fluid_domain_attributes);

   // Create the flow solver.
   NavierSolver navier(&fluid_mesh, polynomial_order, kinematic_viscosity);
   navier.SetMaxBDFOrder(max_bdf_order);

   ParGridFunction *u_gf = navier.GetCurrentVelocity();
   ParGridFunction *p_gf = navier.GetCurrentPressure();
   ParGridFunction *nu_gf = navier.GetVariableViscosity();
   GridFunction *w_gf = navier.GetCurrentMeshVelocity();
   GridFunction fluid_mesh_nodes_old(*fluid_mesh.GetNodes());

   ParGridFunction sigmaN_gf(u_gf->ParFESpace());
   ParGridFunction fluid_mesh_dx_gf(u_gf->ParFESpace());

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
      const double x = coords(0);
      const double y = coords(1);
      const double H = 0.41;
      const double ramp_peak_at = 1.0;
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
   auto stress_integ = new BoundaryNormalStressIntegrator(*u_gf, *p_gf, *nu_gf);
   stress_integ->SetIntRule(&navier.gll_ir_face);
   lift_drag_form.AddBoundaryIntegrator(stress_integ,
                                        solid_interface_attr);

   auto sigmaN_coeff = VectorGridFunctionCoefficient(&sigmaN_gf);
   ParLinearForm lift_drag_form2(u_gf->ParFESpace());
   auto stress_integ2 = new VectorBoundaryLFIntegrator(sigmaN_coeff);
   stress_integ2->SetIntRule(&navier.gll_ir_face);
   auto stress_integ3 = new BoundaryNormalPressureIntegrator(*p_gf);
   stress_integ3->SetIntRule(&navier.gll_ir_face);
   lift_drag_form2.AddBoundaryIntegrator(stress_integ2,
                                         solid_interface_attr);
   lift_drag_form2.AddBoundaryIntegrator(stress_integ3,
                                         solid_interface_attr);

   Array<int> solid_domain_attributes(1);
   solid_domain_attributes[0] = 2;
   auto solid_mesh = ParSubMesh::CreateFromDomain(pmesh, solid_domain_attributes);

   Array<int> solid_fluid_interface_attr(solid_mesh.bdr_attributes.Max());
   solid_fluid_interface_attr = 0;
   solid_fluid_interface_attr[4] = 1;

   Array<int> solid_ess_bdr(solid_mesh.bdr_attributes.Max());
   solid_ess_bdr = 0;
   solid_ess_bdr[6] = 1;

   Elasticity elasticity(solid_mesh, polynomial_order, solid_density,
                         solid_ess_bdr);

   Vector solid_displacement;
   ParGridFunction solid_displacement_gf(&elasticity.fes);
   solid_displacement_gf = 0.0;
   solid_displacement_gf.GetTrueDofs(solid_displacement);

   Vector solid_vel;
   ParGridFunction solid_vel_gf(&elasticity.fes);
   solid_vel_gf = 0.0;
   solid_vel_gf.GetTrueDofs(solid_vel);

   ParGridFunction solid_sigmaN_gf(&elasticity.fes);

   elasticity.SetBoundaryTraction(solid_sigmaN_gf);

   HHTAlphaSolver hht;
   hht.Init(elasticity);

   GridFunction fluid_mesh_velocity_gf(fluid_mesh.GetNodes()->FESpace());
   fluid_mesh_velocity_gf = 0.0;

   Array<int> fluid_mesh_laplacian_ess_bdr(fluid_mesh.bdr_attributes.Max());
   fluid_mesh_laplacian_ess_bdr = 1;

   MeshLaplacian fluid_mesh_laplacian(*fluid_mesh.GetNodes()->FESpace(),
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

#ifdef false
   {
      dt = 1e-3;
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

         if ((step + 1) % 100 == 0 || last_step)
         {
            printf("%d %1.3E %1.3E\n", step, time, dt);
            time += dt;
            pvdc_solid.SetCycle(step);
            pvdc_solid.SetTime(time);
            pvdc_solid.Save();
         }
      }
      exit(0);
   }
#endif

   // auto solid_fluid_bdr_transfer = BoundaryFieldTransfer(solid_mesh, fluid_mesh,
   //                                                       *u_gf->ParFESpace(),
   //                                                       solid_fluid_interface_attr);

   // printf("%d %d\n", fluid_mesh.bdr_attributes.Max(),
   //        solid_mesh.bdr_attributes.Max());

   // auto fluid_solid_bdr_transfer = BoundaryFieldTransfer(fluid_mesh, solid_mesh,
   //                                                       *solid_displacement_gf.ParFESpace(),
   //                                                       solid_fluid_interface_attr);

   L2_FECollection flux_fec(polynomial_order, dim);
   auto flux_fes = new ParFiniteElementSpace(&fluid_mesh, &flux_fec, dim);
   auto estimator = new KellyErrorEstimator(
      *navier.GetPressureEquationBLFI(), *p_gf,
      flux_fes);

   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(0.0); // use purely local threshold
   refiner.SetLocalErrorGoal(max_elem_error);
   refiner.PreferConformingRefinement();
   refiner.SetNCLimit(nc_limit);

   ThresholdDerefiner derefiner(*estimator);
   derefiner.SetThreshold(hysteresis * max_elem_error);
   derefiner.SetNCLimit(nc_limit);

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

      refiner.Reset();
      derefiner.Reset();

      int ok = -1;
      double previous_dt = dt;

      double lift = 0.0, lift2 = 0.0;
      double drag = 0.0, drag2 = 0.0;

      refiner.Apply(fluid_mesh);
      printf("refined to #el = %d\n", fluid_mesh.GetNE());

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

            lift *= density_fluid;
            drag *= density_fluid;
         }
         {
            BoundaryNormalStressEvaluator(*u_gf, *p_gf, *nu_gf,
                                          solid_interface_attr,
                                          navier.gll_ir_face,
                                          sigmaN_gf);

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

            lift2 *= density_fluid;
            drag2 *= density_fluid;
         }

#ifdef false
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

            fluid_mesh_velocity_gf *= dt;
            fluid_mesh_dx_gf += fluid_mesh_velocity_gf;

            const double fluid_mesh_velocity_norm = fluid_mesh_velocity_gf.Norml2();
            if (Mpi::Root())
            {
               printf("moving fluid mesh |dx|_l2 = %.3E\n", fluid_mesh_velocity_norm);
            }

            *w_gf = fluid_mesh_velocity_gf;
            navier.TransformMesh(fluid_mesh_dx_coef);
         }
#endif
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

         derefiner.Apply(fluid_mesh);
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

      if (Mpi::Root())
      {
         printf("%d %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %d log\n",
                step, time, dt, -lift, -drag, -lift2, -drag2, cfl, val.Norml2(), p_probe,
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
