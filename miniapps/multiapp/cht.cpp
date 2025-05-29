#include "mfem.hpp"
#include "multiapp.hpp"

#include <fstream>
#include <iostream>

using namespace mfem;

// Prescribed velocity profile for the convection-diffusion equation inside the
// cylinder. The profile is constructed s.t. it approximates a no-slip (v=0)
// directly at the cylinder wall boundary.
void velocity_profile(const Vector &c, Vector &q)
{
   real_t A = 1.0;
   real_t x = c(0);
   real_t y = c(1);
   real_t r = sqrt(pow(x, 2.0) + pow(y, 2.0));

   q(0) = 0.0;
   q(1) = 0.0;

   if (std::abs(r) >= 0.25 - 1e-8)
   {
      q(2) = 0.0;
   }
   else
   {
      q(2) = A * exp(-(pow(x, 2.0) / 2.0 + pow(y, 2.0) / 2.0));
   }
}

/**
 * @brief Convection-diffusion time dependent operator
 *
 *              dT/dt = κΔT - α∇•(b T)
 *
 * Can also be used to create a diffusion or convection only operator by setting
 * α or κ to zero.
 */
class ConvectionDiffusion : public Application
{
public:
   /**
    * @brief Construct a new convection-diffusion time dependent operator.
    *
    * @param fes The ParFiniteElementSpace the solution is defined on
    * @param ess_tdofs All essential true dofs (relevant if fes is using H1
    * finite elements)
    * @param alpha The convection coefficient
    * @param kappa The diffusion coefficient
    */
   ConvectionDiffusion(ParFiniteElementSpace &fes,
                        Array<int> ess_tdofs,
                        real_t alpha = 1.0,
                        real_t kappa = 1.0e-1)
                        : Application(fes.GetTrueVSize()),
                        temperature_gf(ParGridFunction(&fes)),
                        Mform(&fes),
                        Kform(&fes),
                        bform(&fes),
                        ess_tdofs_(ess_tdofs),
                        M_solver(fes.GetComm())
   {

      d = new ConstantCoefficient(-kappa);
      q = new VectorFunctionCoefficient(fes.GetParMesh()->Dimension(),
                                        velocity_profile);

      Mform.AddDomainIntegrator(new MassIntegrator);
      Mform.Assemble(0);
      Mform.Finalize();

      if (fes.IsDGSpace())
      {
         M.Reset(Mform.ParallelAssemble(), true);

         inflow = new ConstantCoefficient(0.0);
         bform.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(*inflow, *q, alpha));
      }
      else
      {
         Kform.AddDomainIntegrator(new ConvectionIntegrator(*q, -alpha));
         Kform.AddDomainIntegrator(new DiffusionIntegrator(*d));
         Kform.Assemble(0);

         Array<int> empty;
         Kform.FormSystemMatrix(empty, K);
         Mform.FormSystemMatrix(ess_tdofs_, M);

         bform.Assemble();
         b = bform.ParallelAssemble();
      }

      M_solver.iterative_mode = false;
      M_solver.SetRelTol(1e-8);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(100);
      M_solver.SetPrintLevel(0);
      M_prec.SetType(HypreSmoother::Jacobi);
      M_solver.SetPreconditioner(M_prec);
      M_solver.SetOperator(*M);

      t1.SetSize(height);
      t2.SetSize(height);
   }

   void Mult(const Vector &u, Vector &du_dt) const override
   {
      
      // Vector u_bc(u.GetData(), u.Size());
      u_bc = u;
      temperature_gf.GetTrueDofs(u_bc);

      K->Mult(u_bc, t1);
      t1.Add(1.0, *b);
      M_solver.Mult(t1, du_dt);
      du_dt.SetSubVector(ess_tdofs_, 0.0);
   }

   ~ConvectionDiffusion() override
   {
      delete q;
      delete d;
      delete b;
   }

   /// Mass form
   ParBilinearForm Mform;

   /// Stiffness form. Might include diffusion, convection or both.
   ParBilinearForm Kform;

   /// Mass opeperator
   OperatorHandle M;

   /// Stiffness opeperator. Might include diffusion, convection or both.
   OperatorHandle K;

   /// RHS form
   ParLinearForm bform;

   /// RHS vector
   Vector *b = nullptr;

   /// Velocity coefficient
   VectorCoefficient *q = nullptr;

   /// Diffusion coefficient
   Coefficient *d = nullptr;

   /// Inflow coefficient
   Coefficient *inflow = nullptr;

   /// Essential true dof array. Relevant for eliminating boundary conditions
   /// when using an H1 space.
   Array<int> ess_tdofs_;
   ParGridFunction temperature_gf;

   real_t current_dt = -1.0;

   /// Mass matrix solver
   CGSolver M_solver;

   /// Mass matrix preconditioner
   HypreSmoother M_prec;

   /// Auxiliary vectors
   mutable Vector t1, t2;
   mutable Vector u_bc;
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   int order = 2;
   real_t t_final = .005;
   real_t dt = 1.0e-5;
   bool visualization = true;
   int visport = 19916;
   int vis_steps = 10;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.ParseCheck();

   Mesh *serial_mesh = new Mesh("multidomain-hex.mesh");
   ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;

   parent_mesh.UniformRefinement();

   H1_FECollection fec(order, parent_mesh.Dimension());

   // Create the sub-domains and accompanying Finite Element spaces from
   // corresponding attributes. This specific mesh has two domain attributes and
   // 9 boundary attributes.
   Array<int> cylinder_domain_attributes(1);
   cylinder_domain_attributes[0] = 1;

   auto cylinder_submesh = ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attributes);

   ParFiniteElementSpace fes_cylinder(&cylinder_submesh, &fec);

   Array<int> inflow_attributes(cylinder_submesh.bdr_attributes.Max());
   inflow_attributes = 0;
   inflow_attributes[7] = 1;

   Array<int> inner_cylinder_wall_attributes( cylinder_submesh.bdr_attributes.Max());
   inner_cylinder_wall_attributes = 0;
   inner_cylinder_wall_attributes[8] = 1;

   // For the convection-diffusion equation inside the cylinder domain, the
   // inflow surface and outer wall are treated as Dirichlet boundary
   // conditions.
   Array<int> inflow_tdofs, interface_tdofs, ess_tdofs;
   fes_cylinder.GetEssentialTrueDofs(inflow_attributes, inflow_tdofs);
   fes_cylinder.GetEssentialTrueDofs(inner_cylinder_wall_attributes,interface_tdofs);

   ess_tdofs.Append(inflow_tdofs);
   ess_tdofs.Append(interface_tdofs);
   ess_tdofs.Sort();
   ess_tdofs.Unique();

   
   ConvectionDiffusion cd_cylinder(fes_cylinder, ess_tdofs);

   // ParGridFunction temperature_cylinder_gf(&fes_cylinder);
   ParGridFunction &temperature_cylinder_gf = cd_cylinder.temperature_gf;   
   temperature_cylinder_gf = 0.0;

   Vector temperature_cylinder;
   temperature_cylinder_gf.GetTrueDofs(temperature_cylinder);


   // Set up the diffusion equation inside the solid block
   Array<int> outer_domain_attributes(1);
   outer_domain_attributes[0] = 2;

   auto block_submesh = ParSubMesh::CreateFromDomain(parent_mesh,outer_domain_attributes);

   ParFiniteElementSpace fes_block(&block_submesh, &fec);

   Array<int> block_wall_attributes(block_submesh.bdr_attributes.Max());
   block_wall_attributes = 0;
   block_wall_attributes[0] = 1;
   block_wall_attributes[1] = 1;
   block_wall_attributes[2] = 1;
   block_wall_attributes[3] = 1;

   Array<int> outer_cylinder_wall_attributes(block_submesh.bdr_attributes.Max());
   outer_cylinder_wall_attributes = 0;
   outer_cylinder_wall_attributes[8] = 1;

   fes_block.GetEssentialTrueDofs(block_wall_attributes, ess_tdofs);

   ConvectionDiffusion diff_block(fes_block, ess_tdofs, 0.0, 1.0);

   ParGridFunction &temperature_block_gf = diff_block.temperature_gf;
   // ParGridFunction temperature_block_gf(&fes_block);
   temperature_block_gf = 0.0;

   ConstantCoefficient one(1.0);
   temperature_block_gf.ProjectBdrCoefficient(one, block_wall_attributes);

   Vector temperature_block;
   temperature_block_gf.GetTrueDofs(temperature_block);



   RK3SSPSolver cylinder_ode_solver;
   cylinder_ode_solver.Init(cd_cylinder);


   RK3SSPSolver block_ode_solver;
   block_ode_solver.Init(diff_block);


   

   // Create the transfer map needed in the time integration loop
   auto temperature_block_to_cylinder_map = ParSubMesh::CreateTransferMap(
                                               temperature_block_gf,
                                               temperature_cylinder_gf);

   Array<int> offsets({0,temperature_block.Size(), temperature_cylinder.Size()});
   offsets.PartialSum();   
   BlockVector temperature(offsets);

   temperature_block_gf.GetTrueDofs(temperature.GetBlock(0));
   temperature_cylinder_gf.GetTrueDofs(temperature.GetBlock(1));

   CoupledApplication physics(2); // A total of two couple physics
 
   // physics.AddApplication(&diff_block);  // The fluids physics
   // physics.AddApplication(&cd_cylinder); // The solid physics

   LinkedFields lf_solid_to_fluid(temperature_block_gf, temperature_cylinder_gf);
   LinkedFields lf_fluid_to_solid(temperature_cylinder_gf, temperature_block_gf);

   diff_block.AddLinkedFields(&lf_solid_to_fluid);
   // cd_cylinder.AddLinkedFields(&lf_fluid_to_solid);


   Application* app1 = physics.AddApplication(&block_ode_solver);
   Application* app2 = physics.AddApplication(&cylinder_ode_solver);

   app1->AddLinkedFields(&lf_solid_to_fluid);
   // app1->linked_fields = diff_block.linked_fields;

   physics.SetOffsets(offsets);
   physics.Finalize();


 

   RK3SSPSolver coupled_ode_solver;
   coupled_ode_solver.Init(physics);


   ParaViewDataCollection solid_pv("solid2", &block_submesh);
   ParaViewDataCollection fluid_pv("fluid2", &cylinder_submesh);

   solid_pv.SetLevelsOfDetail(order);
   solid_pv.SetDataFormat(VTKFormat::BINARY);
   solid_pv.SetHighOrderOutput(true);
   solid_pv.RegisterField("temperature",&temperature_block_gf);

   fluid_pv.SetLevelsOfDetail(order);
   fluid_pv.SetDataFormat(VTKFormat::BINARY);
   fluid_pv.SetHighOrderOutput(true);
   fluid_pv.RegisterField("temperature",&temperature_cylinder_gf);

   auto save_callback = [&](int cycle, double t)
   {
      solid_pv.SetCycle(cycle);
      solid_pv.SetTime(t);

      fluid_pv.SetCycle(cycle);
      fluid_pv.SetTime(t);

      solid_pv.Save();
      fluid_pv.Save();
   };


   real_t t = 0.0;
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      // coupled_ode_solver.Step(temperature,t,dt);
      physics.Step(temperature, t, dt);
      t += dt;

      // temperature_block_gf.SetFromTrueDofs(temperature.GetBlock(0));      
      // temperature_cylinder_gf.SetFromTrueDofs(temperature.GetBlock(1));

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            out << "step " << ti << ", t = " << t << std::endl;
         }

         save_callback(ti, t);
      }
   }

   return 0;
}
