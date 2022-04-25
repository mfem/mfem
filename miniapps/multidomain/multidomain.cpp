#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

void velocity_profile(const Vector &c, Vector &q)
{
   double A = 1.0;
   double x = c(0);
   double y = c(1);
   double r = sqrt(pow(x, 2.0) + pow(y, 2.0));

   q(0) = 0.0;
   q(1) = 0.0;

   if (abs(r) >= 0.25 - 1e-8)
   {
      q(2) = 0.0;
   }
   else
   {
      q(2) = A * exp(-(pow(x, 2.0) / 2.0 + pow(y, 2.0) / 2.0));
   }
}

class ConvectionDiffusionTDO : public TimeDependentOperator
{
public:
   ConvectionDiffusionTDO(ParFiniteElementSpace &h1_fes,
                          Array<int> ess_tdofs,
                          double alpha = 1.0,
                          double kappa = 1.0e-1)
      : TimeDependentOperator(h1_fes.GetTrueVSize()),
        Mform(&h1_fes),
        Kform(&h1_fes),
        ess_tdofs_(ess_tdofs),
        M_solver(h1_fes.GetComm()),
        T_solver(h1_fes.GetComm())
   {
      Mform.AddDomainIntegrator(new MassIntegrator);
      Mform.Assemble(0);
      Mform.FormSystemMatrix(ess_tdofs_, M);

      q = new VectorFunctionCoefficient(h1_fes.GetParMesh()->Dimension(),
                                        velocity_profile);

      Kform.AddDomainIntegrator(new ConvectionIntegrator(*q, -alpha));

      d = new ConstantCoefficient(-kappa);
      Kform.AddDomainIntegrator(new DiffusionIntegrator(*d));
      Kform.Assemble(0);
      Array<int> empty;
      Kform.FormSystemMatrix(empty, K);

      M_solver.iterative_mode = false;
      M_solver.SetRelTol(1e-8);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(100);
      M_solver.SetPrintLevel(0);
      M_prec.SetType(HypreSmoother::Jacobi);
      M_solver.SetPreconditioner(M_prec);
      M_solver.SetOperator(M);

      t1.SetSize(height);
      t2.SetSize(height);
   }

   void Mult(const Vector &u, Vector &du_dt) const override
   {
      K.Mult(u, t1);
      M_solver.Mult(t1, du_dt);
      du_dt.SetSubVector(ess_tdofs_, 0.0);
   }

   ~ConvectionDiffusionTDO()
   {
      delete q;
      delete d;
   }

   ParBilinearForm Mform, Kform;
   HypreParMatrix M, K, C, *T = nullptr;
   VectorCoefficient *q;
   Coefficient *d;
   Array<int> ess_tdofs_;

   double current_dt = -1.0;

   CGSolver M_solver;
   HypreSmoother M_prec;

   CGSolver T_solver;
   HypreSmoother T_prec;

   mutable Vector t1, t2;
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   Mesh *serial_mesh = new Mesh("multidomain.mesh");

   ParMesh mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;

   int p = 2;
   H1_FECollection h1_fec(p, mesh.Dimension());

   Array<int> cylinder_domain_attributes(1);
   cylinder_domain_attributes[0] = 1;

   auto cylinder_submesh =
      ParSubMesh::CreateFromDomain(mesh, cylinder_domain_attributes);

   ParFiniteElementSpace h1_fes_cylinder(&cylinder_submesh, &h1_fec);

   Array<int> inflow_attributes(cylinder_submesh.bdr_attributes.Max());
   inflow_attributes = 0;
   inflow_attributes[7] = 1;

   Array<int> inner_cylinder_wall_attributes(
      cylinder_submesh.bdr_attributes.Max());
   inner_cylinder_wall_attributes = 0;
   inner_cylinder_wall_attributes[8] = 1;

   Array<int> inflow_tdofs, interface_tdofs, ess_tdofs;
   h1_fes_cylinder.GetEssentialTrueDofs(inflow_attributes, inflow_tdofs);
   h1_fes_cylinder.GetEssentialTrueDofs(inner_cylinder_wall_attributes,
                                        interface_tdofs);
   ess_tdofs.Append(inflow_tdofs);
   ess_tdofs.Append(interface_tdofs);
   ess_tdofs.Unique();
   ConvectionDiffusionTDO cd_tdo(h1_fes_cylinder, ess_tdofs);

   ParGridFunction temperature_cylinder_gf(&h1_fes_cylinder);
   temperature_cylinder_gf = 0.0;
   Vector temperature_cylinder;
   temperature_cylinder_gf.GetTrueDofs(temperature_cylinder);

   RK3SSPSolver cd_ode_solver;
   cd_ode_solver.Init(cd_tdo);

   Array<int> outer_domain_attributes(1);
   outer_domain_attributes[0] = 2;

   auto block_submesh = ParSubMesh::CreateFromDomain(mesh,
                                                     outer_domain_attributes);

   ParFiniteElementSpace h1_fes_block(&block_submesh, &h1_fec);

   Array<int> block_wall_attributes(block_submesh.bdr_attributes.Max());
   block_wall_attributes = 0;
   block_wall_attributes[0] = 1;
   block_wall_attributes[1] = 1;
   block_wall_attributes[2] = 1;
   block_wall_attributes[3] = 1;

   Array<int> outer_cylinder_wall_attributes(
      block_submesh.bdr_attributes.Max());
   outer_cylinder_wall_attributes = 0;
   outer_cylinder_wall_attributes[8] = 1;

   h1_fes_block.GetEssentialTrueDofs(block_wall_attributes, ess_tdofs);

   ConvectionDiffusionTDO d_tdo(h1_fes_block, ess_tdofs, 0.0, 1.0);

   ParGridFunction temperature_block_gf(&h1_fes_block);
   temperature_block_gf = 0.0;
   ConstantCoefficient one(1.0);
   temperature_block_gf.ProjectBdrCoefficient(one, block_wall_attributes);
   Vector temperature_block;
   temperature_block_gf.GetTrueDofs(temperature_block);

   RK3SSPSolver d_ode_solver;
   d_ode_solver.Init(d_tdo);

   Array<int> cylinder_surface_attributes(1);
   cylinder_surface_attributes[0] = 9;

   auto cylinder_surface_submesh = ParSubMesh::CreateFromBoundary(mesh,
                                                                  cylinder_surface_attributes);

   ParFiniteElementSpace cylinder_surface_fes(&cylinder_surface_submesh, &h1_fec);
   ParGridFunction cylinder_surface_gf(&cylinder_surface_fes);

   double dt = 1.0e-5;
   double t_final = 5.0;
   double t = 0.0;
   bool last_step = false;
   int vis_steps = 10;

   char vishost[] = "128.15.198.77";
   int  visport   = 19916;
   socketstream cyl_sol_sock(vishost, visport);
   cyl_sol_sock << "parallel " << num_procs << " " << myid << "\n";
   cyl_sol_sock.precision(8);
   cyl_sol_sock << "solution\n" << cylinder_submesh << temperature_cylinder_gf <<
                "pause\n" << std::flush;

   socketstream block_sol_sock(vishost, visport);
   block_sol_sock << "parallel " << num_procs << " " << myid << "\n";
   block_sol_sock.precision(8);
   block_sol_sock << "solution\n" << block_submesh << temperature_block_gf <<
                  "pause\n" << std::flush;

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      d_ode_solver.Step(temperature_block, t, dt);
      {
         temperature_block_gf.SetFromTrueDofs(temperature_block);

         ParSubMesh::Transfer(temperature_block_gf, cylinder_surface_gf);
         ParSubMesh::Transfer(cylinder_surface_gf, temperature_cylinder_gf);

         temperature_cylinder_gf.GetTrueDofs(temperature_cylinder);
      }
      cd_ode_solver.Step(temperature_cylinder, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            out << "step " << ti << ", t = " << t << std::endl;
         }

         temperature_cylinder_gf.SetFromTrueDofs(temperature_cylinder);
         temperature_block_gf.SetFromTrueDofs(temperature_block);

         cyl_sol_sock << "solution\n" << cylinder_submesh << temperature_cylinder_gf <<
                      std::flush;
         block_sol_sock << "solution\n" << block_submesh << temperature_block_gf <<
                        std::flush;

      }
   }

   return 0;
}
