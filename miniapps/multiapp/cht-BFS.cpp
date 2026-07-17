/**
 * Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
 * at the Lawrence Livermore National Laboratory. All Rights reserved. See files
 * LICENSE and NOTICE for details. LLNL-CODE-806117.
 * 
 * This file is part of the MFEM library. For more information and source code
 * availability visit https://mfem.org.
 * 
 * MFEM is free software; you can redistribute it and/or modify it under the
 * terms of the BSD-3 license. We welcome feedback and contributions, see file
 * CONTRIBUTING.md for details.
 * 
 *            --------------------------------------------------
 *                      Conjugate heat transfer miniapp
 *            --------------------------------------------------
 * 
 * This is a miniapp demonstrates conjugate heat transfer by coupling different
 * physics in different domains:
 *    1) Incompressible Navier-Stokes equations in a fluid domain
 *    2) Heat equation in fluid 
 *    3) Heat equation solid domains
 * 
 * The test case is a benchmark Backward Facing Step (BFS) with a heated base.
 * The following boundary conditions are applied:
 *   1) Fluid inlet (attribute 1): parabolic velocity profile, T = 0.0
 *   2) Fluid outlet (attribute 2): zero-pressure, -kappa grad(T)•n = 0.0
 *   3) Fluid walls (attribute 3 & 4): no-slip, -kappa grad(T)•n = 0.0
 *   4) Solid base (attribute 6): T = 1.0
 *   5) Solid walls (attribute 5): -kappa grad(T)•n = 0.0
 * 
 *                                  4
 *     -----------------------------------------------------------------
 *   1 |                          fluid                                | 2
 *   4 |                                                               | 2
 *     ---------------------------- 3 ----------------------------------
 *   5 |                                                               | 5
 *   5 |                          solid                                | 5
 *     -----------------------------------------------------------------
 *                                  6
 * 
 * This example demonstrates nested coupling with the fluid flow and heat
 * transfer solvers coupling where the heat transfer solver is itself a 
 * coupled solver with the fluid and solid heat transfer solvers.
 * 
 * The fluid flow and heat transfer solvers are flow maps (i.e ODE Solvers) and 
 * coupled with one-way, serial or parallel, coupling. The fluid and solid heat 
 * transfer solvers can be partitioned-coupled or monolithically-coupled. 
 * 
 * For partitioned-coupling, the following boundary conditions are applied 
 * at the fluid-solid interface:
 *  1) Dirichlet: T = T_f  on attr 3 in solid domain
 *  2) Neumann:   Q = -kappa grad(T_s)•n on attr 3 in fluid domain
 * 
 * For monolithic-coupling, the following conditions are imposed at the
 * fluid-solid interface:
 * 1) Continuity of temperature: T_f = T_s
 * 2) Continuity of heat flux: -k_f grad(T_f)•n = k_s grad(T_s)•n
 * 
 * Sample run:
 *    mpirun -np 6 cht-BFS -vs 500 -dt 1e-3 -tf 1000 -rs 2 -o 3 -ode 21 -scheme -1 -cht
 */

#include "mfem.hpp"
#include "multiapp.hpp"
#include "../navier/navier_solver.hpp"


using namespace mfem;
using namespace navier;
using namespace std;

struct BFSContext
{
   int ser_ref = 1;         // Serial mesh refinement
   int order = 3;           // Finite element order
   int ode_solver = 21;     // ODE solver
   real_t dt = 1e-2;        // Time step size
   real_t t_final = 3.0;    // Final time
   int vis_steps = 100;     // Visualization steps
   int couple_scheme = -1;  // Coupling scheme
                            // -1: Monolithic, 0: Alternating Schwarz,
                            // >0: Additive Schwarz with number of iterations
   bool ht_only = false;     // Conjugate heat transfer on/off
   bool visualization = true;// Visualization on/off

   real_t Re = 800.0;         // Reynolds number
   real_t Pr = 0.71;          // Prandtl number
   real_t density = 1.0;      // Density
   real_t kappa_ratio = 1e2;  // Conductivity ratio solid/fluid
   bool checkres = false;     // Check results

#if defined(MFEM_USE_DOUBLE)
   real_t tol_T = 1e-4;
   real_t tol_Q = 1e-4;
#elif defined(MFEM_USE_SINGLE)
   real_t tol_T = 1e-3;
   real_t tol_Q = 1e-3;
#else
#error "Only single and double precision are supported!"
   real_t tol_T = 0;
   real_t tol_Q = 0;
#endif   
} ctx;

void SetSolverParameters(IterativeSolver *solver, real_t rtol, real_t atol , int max_it, 
                         int print_level, bool iterative_mode);

/// Temperature profile 
double temp_profile(const Vector& x)
{
    return x(1) == 0.0 ? 1.0 : 0.0;
}

/// Parabolic velocity profile for channel inlet
void velocity_profile(const Vector &x, Vector &u)
{
   double xi = x(0), yi = x(1) - 2.5;
   u = 0.0;
   u(0) = 24.0*yi*(0.5-yi);
}

/**
 * @brief Coefficient to compute normal heat flux Q = -kappa grad(T) • n
 */
class HeatFluxCoefficient : public Coefficient
{
protected:
   ParGridFunction *T_gf = nullptr;
   Coefficient *conductivity = nullptr;
   Vector grad_T, normal;

public:
   HeatFluxCoefficient(ParGridFunction *T_gf_,
                       Coefficient *conductivity_) : 
                       Coefficient(), T_gf(T_gf_), 
                       conductivity(conductivity_)
                       {
                        int dim = T_gf->FESpace()->GetMesh()->Dimension();
                        grad_T.SetSize(dim);
                        normal.SetSize(dim);
                       }

    real_t Eval(ElementTransformation &T,
              const IntegrationPoint &ip) override
    {
        const DenseMatrix &jacobian = T.Jacobian();
        
        MFEM_ASSERT( (jacobian.Height() - 1 == jacobian.Width()), 
                     "Incorrect Jacobian dimension. Coefficient only "
                     "supported for boundary elements.");

        CalcOrtho(jacobian, normal);
        const double scale = normal.Norml2();
        normal /= scale;

        real_t kappa = conductivity ? conductivity->Eval(T, ip) : 1.0;
        T_gf->GetGradient(T,grad_T);
        real_t flux = -kappa*(grad_T * normal);
        
        return flux;
    }                       
};

/**
 * @brief Convection-diffusion time dependent operator
 *
 *              dT/dt = κΔT - α∇T•u
 *
 * Can also be used to create a diffusion or convection 
 * only operator by setting α or κ to zero.
 */
class ConvectionDiffusion : public Application
{
public:

   // Mesh and finite element space
   ParMesh &mesh;
   ParFiniteElementSpace &fes;

   /// Essential and natural dof array.
   Array<int> ess_attr, nat_attr;
   Array<int> ess_tdofs, nat_tdofs;

   /// Material properties
   ConstantCoefficient diffusivity, kappa, alpha;

   /// Grid functions for the temperature and heat flux
   mutable ParGridFunction T_gf, Q_gf;

   /// Used to store boundary condition data
   mutable ParGridFunction T_gf_bc, Q_gf_bc; 
   mutable GridFunctionCoefficient Q_fgc;
   mutable HeatFluxCoefficient Q_coeff;

   /// Mass form and Stiffness form. Might include 
   /// diffusion, convection or both.
   mutable ParBilinearForm Mform, Kform, Mform_e, Kform_e;

   /// RHS form
   mutable ParLinearForm bform;
   mutable Vector b;

   /// Mass and Stiffness operators
   mutable HypreParMatrix Mmat, Kmat, Kmat_e, Mmat_e;

   /// Mass matrix solver
   CGSolver M_solver;
   GMRESSolver implicit_solver;
   HypreParMatrix *T = nullptr; // T = M + dt K

   /// Preconditioners
   HypreSmoother M_prec, T_prec;

   /// Velocity coefficient
   VectorCoefficient *velocity_coeff = nullptr;

   /// Auxiliary variables
   real_t current_dt = -1.0;
   bool updated = false;
   mutable Vector z, q;

public:

   ConvectionDiffusion(ParFiniteElementSpace &fes_,
                        Array<int> ess_attr_,
                        Array<int> nat_attr_,
                        real_t diffusivity_ = 1.0,
                        real_t kappa_ = 1.0,
                        VectorCoefficient *velocity_coeff_ = nullptr,
                        real_t alpha_ = 1.0)
                        : Application(fes_.GetTrueVSize()),
                        mesh(*fes_.GetParMesh()),
                        fes(fes_),
                        ess_attr(ess_attr_),
                        nat_attr(nat_attr_),
                        diffusivity(diffusivity_), 
                        kappa(kappa_), alpha(-alpha_),
                        T_gf(&fes), Q_gf(&fes),
                        T_gf_bc(&fes), Q_gf_bc(&fes),
                        Q_fgc(&Q_gf_bc), Q_coeff(&T_gf,&kappa),
                        Mform(&fes), Kform(&fes),
                        Mform_e(&fes), Kform_e(&fes),
                        bform(&fes),
                        M_solver(mesh.GetComm()),
                        implicit_solver(mesh.GetComm()),
                        velocity_coeff(velocity_coeff_)
   {

      fes.GetEssentialTrueDofs(ess_attr, ess_tdofs);
      fes.GetEssentialTrueDofs(nat_attr, nat_tdofs);
      
      T_gf = 0.0;
      Q_gf = 0.0;
      T_gf_bc = 0.0;
      Q_gf_bc = 0.0;

      field_collection.AddSourceField("Temperature",&T_gf);
      field_collection.AddSourceField("Flux",&Q_gf);
      field_collection.AddField("Temperature_BC",&T_gf_bc);
      field_collection.AddField("Flux_BC",&Q_gf_bc);

      Mform.AddDomainIntegrator(new MassIntegrator);
      Mform_e.AddDomainIntegrator(new MassIntegrator);

      Kform.AddDomainIntegrator(new DiffusionIntegrator(diffusivity));
      Kform_e.AddDomainIntegrator(new DiffusionIntegrator(diffusivity));

      if(velocity_coeff)
      {
         Kform.AddDomainIntegrator(new ConvectionIntegrator(*velocity_coeff, alpha.constant));
         Kform_e.AddDomainIntegrator(new ConvectionIntegrator(*velocity_coeff, alpha.constant));
      }

      if(nat_attr.Max() > 0)
      {
         bform.AddBoundaryIntegrator(new BoundaryLFIntegrator(Q_fgc),nat_attr);
      }
      
      z.SetSize(fes.GetTrueVSize());
      q.SetSize(fes.GetTrueVSize());
      Assemble();   
      BuildSolvers();
   }

   /// Assemble linear and bilinear forms; called if mesh is updated
   void Assemble()
   {
      AssembleLinearForms();
      AssembleBilinearForms();
   }

   void AssembleBilinearForms()
   {
      Mform.Assemble();
      Kform.Assemble();
      Mform_e.Assemble();
      Kform_e.Assemble();

      Array<int> empty;      
      Mform.FormSystemMatrix(ess_tdofs, Mmat);
      Kform.FormSystemMatrix(ess_tdofs, Kmat);
      Mform_e.FormSystemMatrix(empty, Mmat_e);
      Kform_e.FormSystemMatrix(empty, Kmat_e);
   }

   void AssembleLinearForms()
   {
      b.SetSize(fes.GetTrueVSize());
      b = 0.0;
      bform.Assemble();
      bform.ParallelAssemble(b);
   }

   /// Update finite element space and re-assemble forms
   /// if the mesh has changed
   void Update() override 
   {
        fes.Update();

        T_gf.Update();
        T_gf_bc.Update();
        Q_gf.Update();
        Q_gf_bc.Update();
    
        Mform.Update();
        Kform.Update();
        Mform_e.Update();
        Kform_e.Update();
        bform.Update();

        Assemble();
        updated = true;
   }

   void BuildSolvers()
   {
      M_solver.iterative_mode = false;
      M_solver.SetRelTol(1e-8);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(1000);
      M_solver.SetPrintLevel(0);
      M_prec.SetType(HypreSmoother::Jacobi);
      M_solver.SetPreconditioner(M_prec);
      M_solver.SetOperator(Mmat);

      implicit_solver.iterative_mode = false;
      implicit_solver.SetRelTol(1e-8);
      implicit_solver.SetAbsTol(0.0);
      implicit_solver.SetMaxIter(500);
      implicit_solver.SetPrintLevel(0);
      T_prec.SetType(HypreSmoother::Jacobi);
      implicit_solver.SetPreconditioner(T_prec);
   }

   /// For explict time integration: k = M^{-1} ( -K u + b )
   /// Used for partitioned coupling of explicit methods
   void Mult(const Vector &u, Vector &k) const override
   {      
      Kmat.Mult(u, z);
      z.Neg();
      z.Add(1.0, b);
      M_solver.Mult(z, k);
      k.SetSubVector(ess_tdofs, 0.0);
   }

   /// For implicit time integration: k solves (M + dt K) k = -K u + b
   /// Used for partitioned coupling of implicit methods
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k)
   {   
      AssembleLinearForms();
      if((current_dt != dt) || updated)
      {
         if (T) delete T;
         T = Add(1.0, Mmat, dt, Kmat);
         implicit_solver.SetOperator(*T);
         updated = false;
         current_dt = dt;
      }

      Kmat_e.Mult(u, z);
      z.Neg();
      z.Add(1.0, b);
      implicit_solver.Mult(z, k);

      if(IsCoupled() && nat_attr.Max() == 0)
      {  // Apply interface conditions on temperature 
         // Condition imposed by prescribing k = dT/dt
         // T_gf_bc contains dT/dt from other domain
         T_gf_bc.GetTrueDofs(z);
         for (int i = 0; i < ess_tdofs.Size(); i++)
         {
            int idx = ess_tdofs[i];
            k(idx) = z(idx);
         }
      }
      else
      {  // if uncoupled, apply standard essential BCs
         k.SetSubVector(ess_tdofs, 0.0);
      }
   }

   /// Computes the residual v = M k + K u - b + (Tf - Ts)_int + (Qf + Qs)_int
   /// where ()_int represents the temperature and flux interface conditions
   /// Used for monolithic coupling
   void ImplicitMult(const Vector &u, const Vector &k, Vector &v ) const override
   {
      // v = M*k + K*u
      Mmat_e.Mult(k, v);
      Kmat_e.AddMult(u, v);

      bform.Assemble();
      bform.ParallelAssemble(b);
      v.Add(-1.0,b); // v -= b

      v.SetSubVector(ess_tdofs, 0.0); // Residual at uncoupled dofs is zero
      
      if(IsCoupled()) // Residual from interface conditions
      {  // nat_dofs represent the coupled dofs
         T_gf_bc.GetTrueDofs(z); // T from other domain
         for (int i = 0; i < nat_tdofs.Size(); i++)
         {
               int idx = nat_tdofs[i];
               v(idx) += (u(idx) - z(idx)); // T_f = T_s
         }

         Q_gf.GetTrueDofs(q); // Flux from T in this domain
         Q_gf_bc.GetTrueDofs(z); // Flux from the other domain
         for (int i = 0; i < nat_tdofs.Size(); i++)
         {
               int idx = nat_tdofs[i];
               v(idx) += (q(idx) + z(idx)); // Q_f = -Q_s
         }
      }
   }

   /// Transfer temperature and flux boundary data
   /// Called if application is monolithically coupled
   void Transfer(const Vector &x) override
   {
      T_gf.SetFromTrueDofs(x);
      Q_gf.ProjectBdrCoefficient(Q_coeff,ess_attr);
      Q_gf.ProjectBdrCoefficient(Q_coeff,nat_attr);

      Application::Transfer(); // Transfer all source fields
   }   

   /// Transfer temperature and flux boundary data
   /// Called if application is partitioned coupled
   void Transfer(const Vector &x, const Vector &k, real_t dt = 0.0) override
   {
      if(nat_attr.Max() == 0)
      {  // This domain sends flux
         add(1.0,x,dt,k,z); 
         T_gf.SetFromTrueDofs(z);
         Q_gf.ProjectBdrCoefficient(Q_coeff,ess_attr);
         field_collection.Transfer("Flux"); // Only transfer flux
      }
      else
      { // This domain is sending temperature
         T_gf.SetFromTrueDofs(k);
         field_collection.Transfer("Temperature"); // Only transfer temperature
      }      
   }   

   ~ConvectionDiffusion() override
   {
      if(T) delete T;
   }
};


int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ctx.t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&ctx.dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&ctx.ode_solver, "-ode", "--ode-solver-type",
                  "ODESolver id.");
   args.AddOption(&ctx.ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");                
   args.AddOption(&ctx.ht_only, "-ht", "--heat-transfer-only",
                  "-cht", "--conjugate-heat-transfer",
                  "Conjugate heat transfer or heat transfer only.");
   args.AddOption(&ctx.couple_scheme, "-scheme", "--coupling-scheme", 
                 "Coupling scheme: -1 = Monolithic; 0 = Alt. Schw.; >0 = Add. Schw.");
   args.AddOption(&ctx.checkres, "-cr", "--checkresult", "-no-cr", "--no-checkresult",
                 "Enable or disable checking of the result. Returns -1 on failure.");
   args.ParseCheck();


   int order = ctx.order;
   int ode_solver = ctx.ode_solver;

   Mesh *serial_mesh = new Mesh("backward-facing-step.msh");
   int dim = serial_mesh->Dimension();

   for (int i = 0; i < ctx.ser_ref; ++i) { serial_mesh->UniformRefinement(); }
   serial_mesh->SetCurvature(order, false, dim, Ordering::byNODES);

   ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   parent_mesh.UniformRefinement();
   
   
   // Create the sub-domains and accompanying Finite Element spaces 
   Array<int> domain_attributes(1);
   domain_attributes[0] = 1;
   auto fluid_mesh = ParSubMesh::CreateFromDomain(parent_mesh, domain_attributes);
   fluid_mesh.SetAttributes();
   fluid_mesh.EnsureNodes();
   fluid_mesh.Finalize();

   domain_attributes[0] = 2;
   auto solid_mesh = ParSubMesh::CreateFromDomain(parent_mesh,domain_attributes);
   solid_mesh.SetAttributes();
   solid_mesh.EnsureNodes();
   solid_mesh.Finalize();

   // Set essential and natural boundary conditions attributes
   Array<int> u_ess_attr, p_ess_attr, noslip_ess_attr;
   Array<int> Ts_ess_attr, Ts_nat_attr;
   Array<int> Tf_ess_attr, Tf_nat_attr;

   if (solid_mesh.bdr_attributes.Size() > 0)
   {
      Ts_ess_attr.SetSize(solid_mesh.bdr_attributes.Max());
      Ts_ess_attr = 0; 
      Ts_ess_attr[2] = 1; // fluid-solid interface (T_fluid -> T_solid)
      Ts_ess_attr[5] = 1; // bottom wall

      Ts_nat_attr.SetSize(solid_mesh.bdr_attributes.Max());      
      Ts_nat_attr = 0;
      if(ctx.couple_scheme < 0)
      { // If fully coupled, set nat. bc on interface for equality condition
         Ts_nat_attr[2] = 1;
      }
   }
   
   if (fluid_mesh.bdr_attributes.Size() > 0)
   {
      u_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());
      u_ess_attr = 0;
      u_ess_attr[0] = 1; // inlet

      p_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());
      p_ess_attr = 0;
      p_ess_attr[1] = 1; // outlet

      noslip_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());
      noslip_ess_attr = 1;
      noslip_ess_attr[0] = 0; // inlet
      noslip_ess_attr[1] = 0; // outlet

      Tf_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());
      Tf_ess_attr = 0;
      Tf_ess_attr[0] = 1; // inlet

      Tf_nat_attr.SetSize(fluid_mesh.bdr_attributes.Max());
      Tf_nat_attr = 0;
      Tf_nat_attr[2] = 1; // fluid-solid interface (Qs -> Qf)
   }


   // Finite element spaces for solid and fluid domains   
   H1_FECollection ufec(order, dim);   // Velocity field (fluid domain)
   H1_FECollection pfec(order-1, dim); // Pressure field (fluid domain)
   H1_FECollection Tfec(order, dim);

   ParFiniteElementSpace u_fes(&fluid_mesh, &ufec, dim, Ordering::byNODES);
   ParFiniteElementSpace p_fes(&fluid_mesh, &pfec);   
   ParFiniteElementSpace Tf_fes(&fluid_mesh, &Tfec);
   ParFiniteElementSpace Tuf_fes(&fluid_mesh, &Tfec, dim, Ordering::byNODES);
   ParFiniteElementSpace Ts_fes(&solid_mesh, &Tfec);   

   // Set material properties
   // From Backward Facing Step (BFS) Benchmark
   real_t Re = ctx.Re;
   real_t fluid_density = ctx.density;
   real_t viscosity = fluid_density/Re;
   real_t Pr = ctx.Pr ;
   real_t fluid_alpha = 1.0;
   real_t fluid_diffusivity = 1/(Re*Pr);
   real_t fluid_kappa = fluid_diffusivity;
   real_t solid_alpha = 0.0e0;
   real_t kappa_ratio = ctx.kappa_ratio;
   real_t solid_diffusivity = kappa_ratio*fluid_diffusivity;
   real_t solid_kappa = fluid_kappa*kappa_ratio;

   if(Mpi::Root())
   {
      std::cout << "Fluid Density: " << fluid_density << std::endl;
      std::cout << "Viscosity: " << viscosity << std::endl;
      std::cout << "Prandtl Number: " << Pr << std::endl;
      std::cout << "Reynolds Number: " << Re << std::endl;
      std::cout << "Solid Conductivity: " << solid_kappa << std::endl;
      std::cout << "Fluid Diffusivity: " << fluid_diffusivity << std::endl;
   }   

   
   // Navier miniapp
   NavierSolver nse_miniapp(&fluid_mesh, order, viscosity);  

   int max_bdf_order = 3;
   nse_miniapp.EnablePA(true); 
   nse_miniapp.SetMaxBDFOrder(max_bdf_order);

   ParGridFunction &uf_gf = *nse_miniapp.GetCurrentVelocity();
   ParGridFunction &p_gf = *nse_miniapp.GetCurrentPressure(); 
   
   Vector vzero(dim); vzero = 0.0;
   ConstantCoefficient one_coeff(1.0);
   Coefficient *zero_coeff = new ConstantCoefficient(0.0);   
   VectorCoefficient *zerovec = new VectorConstantCoefficient(vzero);
   VectorCoefficient *u_coeff = new VectorFunctionCoefficient(dim, velocity_profile);

   // Set initial conditions in fluid    
   p_gf.ProjectCoefficient(*zero_coeff);
   uf_gf.ProjectCoefficient(*zerovec);
   uf_gf.ProjectBdrCoefficient(*u_coeff,u_ess_attr);

   nse_miniapp.AddVelDirichletBC(u_coeff, u_ess_attr);
   nse_miniapp.AddVelDirichletBC(zerovec, noslip_ess_attr);
   nse_miniapp.AddPresDirichletBC(zero_coeff, p_ess_attr);
   nse_miniapp.Setup(ctx.dt);

   /// Create navier block vector
   Array<int> nse_offsets({0,uf_gf.ParFESpace()->GetTrueVSize(), p_gf.ParFESpace()->GetTrueVSize()});
   nse_offsets.PartialSum();   
   BlockVector up(nse_offsets);

   uf_gf.GetTrueDofs(up.GetBlock(0));
   p_gf.GetTrueDofs(up.GetBlock(1));


   // Fluid Heat Transfer
   ParGridFunction Tuf_gf(&Tuf_fes);  Tuf_gf = 0.0;
   VectorGridFunctionCoefficient fluid_velocity(&Tuf_gf);
   ConvectionDiffusion fluid_ht(Tf_fes, Tf_ess_attr, Tf_nat_attr, fluid_diffusivity, 
                                fluid_kappa, &fluid_velocity, fluid_alpha);
   
   std::unique_ptr<ODESolver> Tf_odesolver = ODESolver::Select(ode_solver);
   Tf_odesolver->Init(fluid_ht);

   ParGridFunction &Tf_gf = *fluid_ht.Fields().GetField("Temperature");
   ParGridFunction &Qf_gf = *fluid_ht.Fields().GetField("Flux");
   ParGridFunction &Tf_gf_bc = *fluid_ht.Fields().GetField("Temperature_BC");
   ParGridFunction &Qf_gf_bc = *fluid_ht.Fields().GetField("Flux_BC");

   // Set initial conditions in fluid
   Tf_gf.ProjectCoefficient(*zero_coeff);
   Tf_gf.ProjectBdrCoefficient(*zero_coeff,Tf_ess_attr);

   // Solid Heat Transfer
   ConvectionDiffusion solid_ht(Ts_fes, Ts_ess_attr, Ts_nat_attr, solid_diffusivity, 
                                solid_kappa, nullptr, solid_alpha);

   std::unique_ptr<ODESolver> Ts_odesolver = ODESolver::Select(ode_solver);
   Ts_odesolver->Init(solid_ht);

   ParGridFunction &Ts_gf = *solid_ht.Fields().GetField("Temperature");
   ParGridFunction &Qs_gf = *solid_ht.Fields().GetField("Flux");
   ParGridFunction &Ts_gf_bc = *solid_ht.Fields().GetField("Temperature_BC");
   ParGridFunction &Qs_gf_bc = *solid_ht.Fields().GetField("Flux_BC");

   // Set initial conditions in solid
   FunctionCoefficient temp_coeff(temp_profile);
   Ts_gf.ProjectCoefficient(temp_coeff);


   // Set up the coupled heat transfer multiapp
   CoupledOperator ht_operator(2); // two coupled applications: fluid and solid heat transfer

   Application* fl_ht_app = ht_operator.AddOperator(&fluid_ht);
   Application* sl_ht_app = ht_operator.AddOperator(&solid_ht);

   fl_ht_app->SetCoupled(true);
   sl_ht_app->SetCoupled(true);

   // ODE solver for the coupled operator   
   std::unique_ptr<ODESolver> ht_odesolver = ODESolver::Select(ode_solver);
   ht_odesolver->Init(ht_operator);

   Array<int> ht_offsets({0,Tf_fes.GetTrueVSize(), Ts_fes.GetTrueVSize()});
   ht_offsets.PartialSum();   
   BlockVector Tfsv(ht_offsets);

   Tf_gf.GetTrueDofs(Tfsv.GetBlock( fl_ht_app->GetOperatorIndex() ));
   Ts_gf.GetTrueDofs(Tfsv.GetBlock( sl_ht_app->GetOperatorIndex() ));
   
   // Set up conjugate heat transfer app
   CoupledOperator cht_app(2); // two coupled applications: navier, fluid-solid heat transfer

   Application* nse_app = cht_app.AddOperator(&nse_miniapp,up.Size());  // (type-erased) Navier miniapp
   Application* ht_app  = cht_app.AddOperator(ht_odesolver.get()); // Coupled fluid-solid heat transfer ODESolver;


   // Set up field transfers
   NativeTransfer Tf_Ts_map(Tf_fes, Ts_fes); // default map if none provided
   // GSLibTransfer Tf_Ts_map(Tf_fes, Ts_fes);

   /// Create link between fields in different apps
   LinkedFields uf_to_Tuf_lf(&uf_gf, &Tuf_gf); // Navier velocity to fluid-heat convection velocity
   LinkedFields Tf_to_Ts_lf(&Tf_gf, &Ts_gf_bc, &Tf_Ts_map);  // Fluid temperature to solid temperature
   LinkedFields Qs_to_Qf_lf(&Qs_gf, &Qf_gf_bc);  // Solid heat flux to fluid heat flux

   /// Different methods for adding the linked field to their source apps
   nse_app->AddLinkedFields("Velocity",&uf_to_Tuf_lf);
   fl_ht_app->AddLinkedFields("Temperature", &Tf_to_Ts_lf);
   fl_ht_app->Fields().AddTargetField("Flux", &Qs_gf_bc, &Tf_Ts_map);
   sl_ht_app->Fields().AddLinkedFields("Flux", &Qs_to_Qf_lf);
   sl_ht_app->Fields().AddTargetField("Temperature", &Tf_gf_bc);

   // Solvers
   // Select solver for partitioned coupling
   FPISolver fp_solver(MPI_COMM_WORLD); // For partitioned solves
   AitkenRelaxation fp_relax;
   SetSolverParameters(&fp_solver, 0.0, 5e-4, 500, 1, false);
   fp_relax.SetBounds(0.0,1.0e-1);
   fp_solver.SetRelaxation(5e-1, nullptr); // Use default relaxation method
   // fp_solver.SetRelaxation(1e-1, &fp_relax);

   // Select solver for monolithic/full coupling
   GMRESSolver gmres_solver(MPI_COMM_WORLD); 
   SetSolverParameters(&gmres_solver, 1e-7, 1e-7, 500, 0, false);
   gmres_solver.SetKDim(300);
   
   NewtonSolver newton_solver(MPI_COMM_WORLD);  
   SetSolverParameters(&newton_solver, 0.0, 1e-7, 100, 0, false);
   newton_solver.SetSolver(gmres_solver);

   /// Set coupling scheme and corresponding solvers   
   /// The Navier miniapp and coupled heat transfer ODESolver (both flow maps) are coupled 
   /// can be coupled in parallel (Additive Schwarz) or serial (Alternating Schwarz) but not
   /// monolithically. The fluid and solid heat transfer Applications in the coupled heat 
   /// transfer ODESolver can also be coupled monolithically.
   if(ctx.couple_scheme == -1)
   {
      cht_app.SetCouplingScheme(CoupledOperator::Scheme::ALTERNATING_SCHWARZ);
      ht_operator.SetCouplingScheme(CoupledOperator::Scheme::MONOLITHIC);
      ht_operator.SetSolver(&newton_solver);
   }
   else if(ctx.couple_scheme == 0)
   {
      cht_app.SetCouplingScheme(CoupledOperator::Scheme::ALTERNATING_SCHWARZ);
      ht_operator.SetCouplingScheme(CoupledOperator::Scheme::ALTERNATING_SCHWARZ);
      ht_operator.SetSolver(&fp_solver);
   }
   else
   {
      cht_app.SetCouplingScheme(CoupledOperator::Scheme::ADDITIVE_SCHWARZ);
      ht_operator.SetCouplingScheme(CoupledOperator::Scheme::ADDITIVE_SCHWARZ);
      ht_operator.SetSolver(&fp_solver);
   }

   ht_operator.Assemble(false);
   cht_app.Assemble(false);


   auto nse_preprocess  = [&nse_offsets, &uf_gf, &p_gf](Vector &x) mutable { 
         BlockVector up(x.GetData(), nse_offsets);
         uf_gf.GetTrueDofs(up.GetBlock(0));
         p_gf.GetTrueDofs(up.GetBlock(1));
   };
   auto nse_postprocess = [&nse_offsets, &uf_gf, &p_gf](Vector &x) mutable { 
         BlockVector up(x.GetData(), nse_offsets);
         uf_gf.SetFromTrueDofs(up.GetBlock(0));
         p_gf.SetFromTrueDofs(up.GetBlock(1));
   };

   auto ht_preprocess = [&ht_offsets, &Tf_gf, &Ts_gf](Vector &x) mutable { 
         BlockVector Tb(x.GetData(), ht_offsets);
         Tf_gf.GetTrueDofs(Tb.GetBlock(0));
         Ts_gf.GetTrueDofs(Tb.GetBlock(1));
   };
   auto ht_postprocess = [&ht_offsets, &Tf_gf, &Ts_gf](Vector &x) mutable { 
         BlockVector Tb(x.GetData(), ht_offsets);
         Tf_gf.SetFromTrueDofs(Tb.GetBlock(0));
         Ts_gf.SetFromTrueDofs(Tb.GetBlock(1));
   };

   /// Set pre/post processing lambdas to corresponding apps
   ht_app->SetPreProcessFunction(ht_preprocess);
   ht_app->SetPostProcessFunction(ht_postprocess);

   /// Not strictly necessary since Navier owns and updates GridFunctions 
   /// internally but included here for completeness
   nse_app->SetPreProcessFunction(nse_preprocess);
   nse_app->SetPostProcessFunction(nse_postprocess);


   // Set up the initial conditions in block vector for the
   // coupled application in the correct order
   int fl_id = nse_app->GetOperatorIndex();
   int ht_id = ht_app->GetOperatorIndex();

   BlockVector xb(cht_app.GetBlockOffsets());
   xb.GetBlock(fl_id) = up;  // Fluid velocity and pressure    
   xb.GetBlock(ht_id) = Tfsv;

   
   // Set up visualization 
   ParaViewDataCollection *fluid_pv = nullptr;
   ParaViewDataCollection *solid_pv = nullptr;

   if(ctx.visualization)
   {
      fluid_pv = new ParaViewDataCollection("cht-BFS-fluid", &fluid_mesh);
      solid_pv = new ParaViewDataCollection("cht-BFS-solid", &solid_mesh);

      fluid_pv->SetLevelsOfDetail(order);
      fluid_pv->SetDataFormat(VTKFormat::BINARY);
      fluid_pv->SetHighOrderOutput(true);
      fluid_pv->RegisterField("pressure",&p_gf);
      fluid_pv->RegisterField("velocity",&uf_gf);
      fluid_pv->RegisterField("convection",&Tuf_gf);
      fluid_pv->RegisterField("Temperature",&Tf_gf);
      fluid_pv->RegisterField("Flux",&Qf_gf);
      
      solid_pv->SetLevelsOfDetail(order);
      solid_pv->SetDataFormat(VTKFormat::BINARY);
      solid_pv->SetHighOrderOutput(true);
      solid_pv->RegisterField("Temperature",&Ts_gf);
      solid_pv->RegisterField("Flux",&Qs_gf);      
   }
      
   auto save_callback = [&](int cycle, double t)
   {
      if(fluid_pv)
      {
         fluid_pv->SetCycle(cycle);
         fluid_pv->SetTime(t);
         fluid_pv->Save();
      }

      if(solid_pv)
      {
         solid_pv->SetCycle(cycle);
         solid_pv->SetTime(t);
         solid_pv->Save();
      }
   };
   

   if (Mpi::Root()) { 
      out << "Starting time integration..." << std::endl; 
   }

   StopWatch timer;
   timer.Start();

   real_t t = 0.0;
   bool last_step = false;
   int tindex = 1;
   save_callback(0, t);

   last_step = false;
   for (; !last_step; tindex++)
   {
      if (t + ctx.dt >= ctx.t_final - ctx.dt/2){ last_step = true; }

      if(ctx.ht_only)
      {
         ht_odesolver->Step(Tfsv,t,ctx.dt);
         Tf_gf.SetFromTrueDofs(Tfsv.GetBlock(0));
         Ts_gf.SetFromTrueDofs(Tfsv.GetBlock(1));
      }
      else
      {
         cht_app.Step(xb, t, ctx.dt);
      }

      if (last_step || (tindex % ctx.vis_steps) == 0){
         if (Mpi::Root()) { out << "step " << tindex << ", t = " << t << std::endl;}
         save_callback(tindex, t);
      }
   }

   timer.Stop();
   if (Mpi::Root()){
      out << "Total time: " << timer.RealTime() << " seconds." << std::endl;
   }

   /// Compute interface error
   if(ctx.checkres)
   {
      Array<int> fl_int_attr(fluid_mesh.bdr_attributes.Max());
      fl_int_attr[2] = 1; // fluid-solid interface
      
      /// Create submesh and FE space for the interface
      ParSubMesh int_mesh = ParSubMesh::CreateFromBoundary(fluid_mesh, fl_int_attr);
      ParFiniteElementSpace int_fes(&int_mesh, &Tfec);

      /// Receiving grid functions on the interface
      ParGridFunction fl_int_gf(&int_fes);
      ParGridFunction sl_int_gf(&int_fes);

      /// Maps and linked fields to transfer domain grid functions to the interface
      NativeTransfer fl_to_int_submesh(Tf_fes, int_fes);
      LinkedFields TQf_to_Tint_lf(&Tf_gf, &fl_int_gf, &fl_to_int_submesh);
      LinkedFields TQs_to_Tint_lf(&Tf_gf_bc, &sl_int_gf, &fl_to_int_submesh);

      cht_app.Transfer(xb); // Transfer all fields *_gf to their target fields *_gf_bc
      
      fl_int_gf = 0.0; sl_int_gf = 0.0;
      TQf_to_Tint_lf.Transfer(); // Transfer Tf to interface
      TQs_to_Tint_lf.Transfer(); // Transfer Ts (in Tf_gf_bc) to interface
      real_t err_T = sqrt(DistanceSquared(int_mesh.GetComm(), fl_int_gf, sl_int_gf));

      // Update sources in existing linked fields (can also create new ones)
      TQf_to_Tint_lf.SetSource(&Qf_gf);
      TQs_to_Tint_lf.SetSource(&Qf_gf_bc);

      fl_int_gf = 0.0; sl_int_gf = 0.0;
      TQf_to_Tint_lf.Transfer(); // Transfer Qf to interface
      TQs_to_Tint_lf.Transfer(); // Transfer Qs (in Qf_gf_bc) to interface
      real_t err_Q = sqrt(DistanceSquared(int_mesh.GetComm(), fl_int_gf, sl_int_gf));

      if (sqrt(err_T) > ctx.tol_T || sqrt(err_Q) > ctx.tol_Q)
      {
         if (Mpi::Root())
         {
            mfem::out << "Result has a larger error than expected."
                      << "T Error = " << sqrt(err_T)
                      << ", Q Error = " << sqrt(err_Q)
                      << std::endl;
         }
         return -1;
      }
   }

   if(fluid_pv) delete fluid_pv;
   if(solid_pv) delete solid_pv;
   delete zero_coeff;
   delete zerovec;
   delete u_coeff;

   return 0;
}

void SetSolverParameters(IterativeSolver *solver, real_t rtol, real_t atol , int max_it, 
                         int print_level, bool iterative_mode)
{
    solver->SetRelTol(rtol);
    solver->SetAbsTol(atol);
    solver->SetMaxIter(max_it);
    solver->SetPrintLevel(print_level);
    solver->iterative_mode = iterative_mode;
}
