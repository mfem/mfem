// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// This miniapp aims to demonstrate how to solve two PDEs, that represent
// different physics, on the same domain. MFEM's SubMesh interface is used to
// compute on and transfer between the spaces of predefined parts of the domain.
// For the sake of simplicity, the spaces on each domain are using the same
// order H1 finite elements. This does not mean that the approach is limited to
// this configuration.
//
// A 3D domain comprised of an outer box with a cylinder shaped inside is used.
//
// A heat equation is described on the outer box domain
//
//                  dT/dt = κΔT         in outer box
//                      T = T_wall      on outside wall
//                   ∇T•n = 0           on inside (cylinder) wall
//
// with temperature T and coefficient κ (non-physical in this example).
//
// A convection-diffusion equation is described inside the cylinder domain
//
//             dT/dt = κΔT - α∇•(b T)   in inner cylinder
//                 T = T_wall           on cylinder wall (obtained from heat equation)
//              ∇T•n = 0                else
//
// with temperature T, coefficients κ, α and prescribed velocity profile b.
//
// To couple the solutions of both equations, a segregated solve with one way
// coupling approach is used. The heat equation of the outer box is solved from
// the timestep T_box(t) to T_box(t+dt). Then for the convection-diffusion
// equation T_wall is set to T_box(t+dt) and the equation is solved for T(t+dt)
// which results in a first-order one way coupling.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

// Prescribed velocity profile for the convection-diffusion equation inside the
// cylinder. The profile is constructed s.t. it approximates a no-slip (v=0)
// directly at the cylinder wall boundary.
void velocity_profile(const Vector &c, Vector &q)
{
   double x = c(0);
   double y = c(1);
   double z = c(2);
   double A = -16.0 * pow(z - 0.5, 2) * M_E;
   double r = sqrt(pow(x, 2.0) + pow(y, 2.0));

   q(0) = 0.0;
   q(1) = 0.0;
   q(2) = 0.0;

   if (std::abs(r) >= 0.25 - 1e-8)
   {
      return;
   }
   else
   {
     const double qr = -A * r * exp(-16.0 * (pow(x, 2.0) + pow(y, 2.0)));
     q(0) = qr * x;
     q(1) = qr * y;
   }
}

void square_xy(const Vector &p, Vector &v)
{
  v.SetSize(3);

  v[0] = 2.0 * p[0];
  v[1] = 2.0 * p[1];
  v[2] = 0.0;
}

/**
 * @brief Convection-diffusion time dependent operator
 *
 *              dT/dt = κΔT - α∇•(b T)
 *
 * Can also be used to create a diffusion or convection only operator by setting
 * α or κ to zero.
 */
class ConvectionDiffusionTDO : public TimeDependentOperator
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
   ConvectionDiffusionTDO(ParFiniteElementSpace &fes,
                          Array<int> ess_tdofs,
                          double alpha = 1.0,
                          double kappa = 1.0e-1)
      : TimeDependentOperator(fes.GetTrueVSize()),
        Mform(&fes),
        Kform(&fes),
        bform(&fes),
        ess_tdofs_(ess_tdofs),
        M_solver(fes.GetComm())
   {
      d = new ConstantCoefficient(-kappa);
      q = new VectorFunctionCoefficient(fes.GetParMesh()->Dimension(),
                                        velocity_profile);

      aq = new ScalarVectorProductCoefficient(alpha, *q);
      
      Mform.AddDomainIntegrator(new VectorFEMassIntegrator);
      Mform.Assemble(0);
      Mform.Finalize();

      if (fes.IsDGSpace())
      {
         M.Reset(Mform.ParallelAssemble(), true);

         inflow = new ConstantCoefficient(0.0);
         bform.AddBdrFaceIntegrator(
            new BoundaryFlowIntegrator(*inflow, *q, alpha));
      }
      else
      {
         Kform.AddDomainIntegrator(new MixedWeakGradDotIntegrator(*aq));
         Kform.AddDomainIntegrator(new DivDivIntegrator(*d));
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
      K->Mult(u, t1);
      t1.Add(1.0, *b);
      M_solver.Mult(t1, du_dt);
      du_dt.SetSubVector(ess_tdofs_, 0.0);
   }

   ~ConvectionDiffusionTDO()
   {
      delete aq;
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

   /// alpha * Velocity coefficient
   VectorCoefficient *aq = nullptr;

   /// Diffusion coefficient
   Coefficient *d = nullptr;

   /// Inflow coefficient
   Coefficient *inflow = nullptr;

   /// Essential true dof array. Relevant for eliminating boundary conditions
   /// when using an H1 space.
   Array<int> ess_tdofs_;

   double current_dt = -1.0;

   /// Mass matrix solver
   CGSolver M_solver;

   /// Mass matrix preconditioner
   HypreSmoother M_prec;

   /// Auxiliary vectors
   mutable Vector t1, t2;
};

void dump_normals(Mesh &mesh, const std::string &name, int myid);
void count_be(ParMesh &mesh, const std::string &name);

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   int order = 2;
   double t_final = 5.0;
   double dt = 1.0e-5;
   bool visualization = true;
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

   dump_normals(parent_mesh, "parent", myid);
   count_be(parent_mesh, "parent");
   
   RT_FECollection fec(order, parent_mesh.Dimension());

   // Create the sub-domains and accompanying Finite Element spaces from
   // corresponding attributes. This specific mesh has two domain attributes and
   // 9 boundary attributes.
   Array<int> cylinder_domain_attributes(1);
   cylinder_domain_attributes[0] = 1;

   auto cylinder_submesh =
      ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attributes);

   dump_normals(cylinder_submesh, "cylinder", myid);
   count_be(cylinder_submesh, "cylinder");
   
   ParFiniteElementSpace fes_cylinder(&cylinder_submesh, &fec);

   Array<int> inflow_attributes(cylinder_submesh.bdr_attributes.Max());
   inflow_attributes = 0;
   inflow_attributes[5] = 1;
   inflow_attributes[7] = 1;

   Array<int> inner_cylinder_wall_attributes(
      cylinder_submesh.bdr_attributes.Max());
   inner_cylinder_wall_attributes = 0;
   inner_cylinder_wall_attributes[8] = 1;

   // For the convection-diffusion equation inside the cylinder domain, the
   // inflow surface and outer wall are treated as Dirichlet boundary
   // conditions.
   Array<int> inflow_tdofs, interface_tdofs, ess_tdofs;
   fes_cylinder.GetEssentialTrueDofs(inflow_attributes, inflow_tdofs);
   fes_cylinder.GetEssentialTrueDofs(inner_cylinder_wall_attributes,
                                     interface_tdofs);
   ess_tdofs.Append(inflow_tdofs);
   ess_tdofs.Append(interface_tdofs);
   ess_tdofs.Sort();
   ess_tdofs.Unique();
   ConvectionDiffusionTDO cd_tdo(fes_cylinder, ess_tdofs);

   ParGridFunction temperature_cylinder_gf(&fes_cylinder);
   temperature_cylinder_gf = 0.0;

   Vector temperature_cylinder;
   temperature_cylinder_gf.GetTrueDofs(temperature_cylinder);

   RK3SSPSolver cd_ode_solver;
   cd_ode_solver.Init(cd_tdo);

   Array<int> outer_domain_attributes(1);
   outer_domain_attributes[0] = 2;

   auto block_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                     outer_domain_attributes);

   dump_normals(block_submesh, "block", myid);
   count_be(block_submesh, "block");
   
   {
     std::ostringstream mesh_name;
     mesh_name << "block_mesh." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     block_submesh.Print(mesh_ofs);
   }
   {
     std::ostringstream mesh_name;
     mesh_name << "cylinder_mesh." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     cylinder_submesh.Print(mesh_ofs);
   }
   
   ParFiniteElementSpace fes_block(&block_submesh, &fec);

   Array<int> block_wall_attributes(block_submesh.bdr_attributes.Max());
   block_wall_attributes = 1;
   block_wall_attributes[8] = 0;

   Array<int> blk_cyl_tdofs;
   Array<int> blk_cyl_vdofs;
   Array<int> block_cylinder_attributes(block_submesh.bdr_attributes.Max());
   block_cylinder_attributes = 0;
   block_cylinder_attributes[8] = 1;

   Array<int> outer_cylinder_wall_attributes(
      block_submesh.bdr_attributes.Max());
   outer_cylinder_wall_attributes = 0;
   outer_cylinder_wall_attributes[8] = 1;

   fes_block.GetEssentialTrueDofs(block_wall_attributes, ess_tdofs);
   fes_block.GetEssentialTrueDofs(block_cylinder_attributes, blk_cyl_tdofs);
   fes_block.GetEssentialVDofs(block_cylinder_attributes, blk_cyl_vdofs);
   std::cout << myid << " Found " << blk_cyl_tdofs.Size() << " true DoFs and "
	     << blk_cyl_vdofs.Sum() << " on cyl boundary"
	     << std::endl;
   
   ConvectionDiffusionTDO d_tdo(fes_block, ess_tdofs, 0.0, 1.0);

   ParGridFunction temperature_block_gf(&fes_block);
   temperature_block_gf = 0.0;

   VectorFunctionCoefficient one(3, square_xy);
   temperature_block_gf.ProjectBdrCoefficientNormal(one,
						    block_wall_attributes);

   Vector temperature_block;
   temperature_block_gf.GetTrueDofs(temperature_block);

   RK3SSPSolver d_ode_solver;
   d_ode_solver.Init(d_tdo);

   Array<int> cylinder_surface_attributes(1);
   cylinder_surface_attributes[0] = 9;

   auto cylinder_surface_submesh = ParSubMesh::CreateFromBoundary(parent_mesh,
                                                                  cylinder_surface_attributes);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream cyl_sol_sock;
   if (visualization)
   {
      cyl_sol_sock.open(vishost, visport);
      cyl_sol_sock << "parallel " << num_procs << " " << myid << "\n";
      cyl_sol_sock.precision(8);
      cyl_sol_sock << "solution\n" << cylinder_submesh << temperature_cylinder_gf <<
                   "pause\n" << std::flush;
   }
   socketstream block_sol_sock;
   if (visualization)
   {
      block_sol_sock.open(vishost, visport);
      block_sol_sock << "parallel " << num_procs << " " << myid << "\n";
      block_sol_sock.precision(8);
      block_sol_sock << "solution\n" << block_submesh << temperature_block_gf <<
                     "pause\n" << std::flush;
   }
   /*
   socketstream block_sol0_sock;
   if (visualization)
   {
      block_sol0_sock.open(vishost, visport);
      block_sol0_sock << "parallel " << num_procs << " " << myid << "\n";
      block_sol0_sock.precision(8);
      block_sol0_sock << "solution\n" << block_submesh << temperature_block_gf <<
                     "pause\n" << std::flush;
   }
   */
   // Create the transfer map needed in the time integration loop
   auto temperature_block_to_cylinder_map = ParSubMesh::CreateTransferMap(
                                               temperature_block_gf,
                                               temperature_cylinder_gf);

   double t = 0.0;
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      // Advance the diffusion equation on the outer block to the next time step
      d_ode_solver.Step(temperature_block, t, dt);
      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            out << "step " << ti << ", t = " << t << std::endl;
         }

         temperature_block_gf.SetFromTrueDofs(temperature_block);
	 /*
         if (visualization)
         {
            block_sol0_sock << "parallel " << num_procs << " " << myid << "\n";
            block_sol0_sock << "solution\n" << block_submesh << temperature_block_gf <<
                           std::flush;
         }
	 */
      }
      {
         // Transfer the solution from the inner surface of the outer block to
         // the cylinder outer surface to act as a boundary condition.
         temperature_block_gf.SetFromTrueDofs(temperature_block);

         temperature_block_to_cylinder_map.Transfer(temperature_block_gf,
                                                    temperature_cylinder_gf);

         temperature_cylinder_gf.GetTrueDofs(temperature_cylinder);
      }
      // Advance the convection-diffusion equation on the outer block to the
      // next time step
      cd_ode_solver.Step(temperature_cylinder, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            out << "step " << ti << ", t = " << t << std::endl;
         }

         temperature_cylinder_gf.SetFromTrueDofs(temperature_cylinder);
         temperature_block_gf.SetFromTrueDofs(temperature_block);

         if (visualization)
         {
            cyl_sol_sock << "parallel " << num_procs << " " << myid << "\n";
            cyl_sol_sock << "solution\n" << cylinder_submesh << temperature_cylinder_gf <<
                         std::flush;
            block_sol_sock << "parallel " << num_procs << " " << myid << "\n";
            block_sol_sock << "solution\n" << block_submesh << temperature_block_gf <<
                           std::flush;
         }
      }
   }

   return 0;
}

void dump_normals(Mesh &mesh, const std::string &name, int myid)
{
  Vector center(3);
  Vector pcenter(3);
  Vector normal(3);

  std::ostringstream norm_name;
  norm_name << name << "." << std::setfill('0') << std::setw(6) << myid;
  std::ofstream ofs(norm_name.str().c_str());
     
     for (int i=0; i<mesh.GetNBE(); i++)
       {
	 if (mesh.GetBdrAttribute(i) == 9)
	   {
	     int geom = mesh.GetBdrElementBaseGeometry(i);
	     ElementTransformation *eltransf = mesh.GetBdrElementTransformation(i);
	     eltransf->SetIntPoint(&Geometries.GetCenter(geom));

	     eltransf->Transform(Geometries.GetCenter(geom), center);
	     pcenter = center; pcenter[2] = 0.0;
	     
	     const DenseMatrix &Jac = eltransf->Jacobian();
	     CalcOrtho(Jac, normal);
	     ofs << i << '\t' << mesh.GetBdrAttribute(i)
		 << "\t(" << center[0] << "," << center[1] << "," << center[2] << ")" << center.Norml2()
	       // << Jac.NumRows() << "x" << Jac.NumCols()
		 << "\t(" << normal[0] << "," << normal[1] << "," << normal[2] << ")" << normal.Norml2()
		 << "\t" << (normal * pcenter) / (pcenter.Norml2() * normal.Norml2())
		 << std::endl;
	   }
       }
}

void count_be(ParMesh &mesh, const std::string &name)
{
  Array<int> counts(mesh.bdr_attributes.Max() + 1);
  counts = 0;
  
  for (int i=0; i<mesh.GetNBE(); i++)
  {
    counts[mesh.GetBdrAttribute(i)]++;
  }

  Array<int> glb_counts(mesh.bdr_attributes.Max() + 1);
  MPI_Reduce(counts, glb_counts, mesh.bdr_attributes.Max() + 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (Mpi::Root())
  {
    std::cout << name << std::endl;
    for (int i=0; i<glb_counts.Size(); i++)
    {
      if (glb_counts[i] != 0)
	{
	  std::cout << i << '\t' << glb_counts[i] << std::endl;
	}
    }
    std::cout << std::endl;
  }
}
