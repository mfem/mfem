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

   if (std::abs(r) >= 0.25 - 1e-8)
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
   ConvectionDiffusionTDO(ParFiniteElementSpace &fes,
                          Array<int> ess_tdofs,
                          double alpha = 1.0,
                          double kappa = 1.0e-1)
      : TimeDependentOperator(fes.GetTrueVSize()),
        Mform(&fes),
        Kform(&fes),
        bform(&fes),
        ess_tdofs_(ess_tdofs),
        M_solver(fes.GetComm()),
        T_solver(fes.GetComm())
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
         bform.AddBdrFaceIntegrator(
            new BoundaryFlowIntegrator(*inflow, *q, alpha));
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
      K->Mult(u, t1);
      t1.Add(1.0, *b);
      M_solver.Mult(t1, du_dt);
      du_dt.SetSubVector(ess_tdofs_, 0.0);
   }

   ~ConvectionDiffusionTDO()
   {
      delete q;
      delete d;
      delete b;
   }

   ParBilinearForm Mform, Kform;
   OperatorHandle M, K;
   ParLinearForm bform;
   Vector *b = nullptr;
   VectorCoefficient *q = nullptr;
   Coefficient *d = nullptr, *inflow = nullptr;
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

   Mesh *serial_mesh = new Mesh("data/multidomain-hex.mesh");
   ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;

   parent_mesh.UniformRefinement();

   int p = 2;
   H1_FECollection fec(p, parent_mesh.Dimension());

   Array<int> cylinder_domain_attributes(1);
   cylinder_domain_attributes[0] = 1;

   auto cylinder_submesh =
      ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attributes);

   ParFiniteElementSpace fes_cylinder(&cylinder_submesh, &fec);

   Array<int> inflow_attributes(cylinder_submesh.bdr_attributes.Max());
   inflow_attributes = 0;
   inflow_attributes[7] = 1;

   Array<int> inner_cylinder_wall_attributes(
      cylinder_submesh.bdr_attributes.Max());
   inner_cylinder_wall_attributes = 0;
   inner_cylinder_wall_attributes[8] = 1;

   Array<int> inflow_tdofs, interface_tdofs, ess_tdofs;
   fes_cylinder.GetEssentialTrueDofs(inflow_attributes, inflow_tdofs);
   fes_cylinder.GetEssentialTrueDofs(inner_cylinder_wall_attributes,
                                     interface_tdofs);
   ess_tdofs.Append(inflow_tdofs);
   ess_tdofs.Append(interface_tdofs);
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

   ParFiniteElementSpace fes_block(&block_submesh, &fec);

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

   fes_block.GetEssentialTrueDofs(block_wall_attributes, ess_tdofs);

   ConvectionDiffusionTDO d_tdo(fes_block, ess_tdofs, 0.0, 1.0);

   ParGridFunction temperature_block_gf(&fes_block);
   temperature_block_gf = 0.0;
   ConstantCoefficient one(1.0);
   temperature_block_gf.ProjectBdrCoefficient(one, block_wall_attributes);
   Vector temperature_block;
   temperature_block_gf.GetTrueDofs(temperature_block);

   RK3SSPSolver d_ode_solver;
   d_ode_solver.Init(d_tdo);

   Array<int> cylinder_surface_attributes(1);
   cylinder_surface_attributes[0] = 9;

   auto cylinder_surface_submesh = ParSubMesh::CreateFromBoundary(parent_mesh,
                                                                  cylinder_surface_attributes);

   ParFiniteElementSpace cylinder_surface_fes(&cylinder_surface_submesh, &fec);
   ParGridFunction temperature_cylinder_surface_gf(&cylinder_surface_fes);

   double dt = 1.0e-5;
   double t_final = 5.0;
   double t = 0.0;
   bool last_step = false;
   int vis_steps = 10;

   char vishost[] = "128.15.198.77";
   int  visport   = 19916;

   // ParFiniteElementSpace parent_fes(&parent_mesh, &fec);
   // ParGridFunction parent_gf(&parent_fes);
   // parent_gf = 0.0;
   // FunctionCoefficient coef([](const Vector &x)
   // {
   //    return cos(x(0)) + sin(x(1));
   // });
   // temperature_block_gf = 0.0;
   // temperature_cylinder_gf = 0.0;
   // temperature_cylinder_surface_gf = 0.0;

   // temperature_cylinder_gf.ProjectCoefficient(coef);
   // temperature_block_gf = 0.1;

   // ParSubMesh::Transfer(temperature_cylinder_gf, temperature_cylinder_surface_gf);
   // ParSubMesh::Transfer(temperature_cylinder_gf, temperature_block_gf);
   // ParSubMesh::Transfer(temperature_block_gf, temperature_cylinder_gf);

   // int active_attribute =
   //    temperature_block_gf.ParFESpace()->GetParMesh()->GetAttribute(0);

   // ParSubMesh::Transfer(parent_gf, temperature_cylinder_surface_gf);

   // {
   //    Array<int> ess_tdof_list;

   //    Array<int> ess_bdr(parent_mesh.bdr_attributes.Max());
   //    ess_bdr = 0;
   //    ess_bdr[0] = 1;
   //    ess_bdr[1] = 1;
   //    ess_bdr[2] = 1;
   //    ess_bdr[3] = 1;
   //    ess_bdr[4] = 1;
   //    ess_bdr[5] = 1;
   //    ess_bdr[6] = 1;
   //    ess_bdr[7] = 1;
   //    parent_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //    ParLinearForm b(&parent_fes);
   //    ConstantCoefficient one(1.0);
   //    b.AddDomainIntegrator(new DomainLFIntegrator(one));
   //    b.Assemble();
   //    ParGridFunction x(&parent_fes);
   //    x = 0.0;
   //    ParBilinearForm a(&parent_fes);
   //    a.AddDomainIntegrator(new DiffusionIntegrator(one));
   //    a.Assemble();

   //    OperatorPtr A;
   //    Vector B, X;
   //    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   //    Solver *prec = new HypreBoomerAMG;

   //    CGSolver cg(MPI_COMM_WORLD);
   //    cg.SetRelTol(1e-12);
   //    cg.SetMaxIter(2000);
   //    cg.SetPrintLevel(1);
   //    if (prec) { cg.SetPreconditioner(*prec); }
   //    cg.SetOperator(*A);
   //    cg.Mult(B, X);
   //    delete prec;

   //    a.RecoverFEMSolution(X, b, x);
   //    char vishost[] = "128.15.198.77";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock << "parallel " << num_procs << " " << myid << "\n";
   //    sol_sock.precision(8);
   //    sol_sock << "solution\n" << parent_mesh << x << std::flush;
   // }

   // exit(0);

   // socketstream mesh_sock(vishost, visport);
   // mesh_sock << "parallel " << num_procs << " " << myid << "\n";
   // mesh_sock.precision(8);
   // mesh_sock << "solution\n" << parent_mesh << parent_gf << "pause\n" <<
   //           std::flush;

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

   // socketstream cylinder_surface(vishost, visport);
   // cylinder_surface << "parallel " << num_procs << " " << myid << "\n";
   // cylinder_surface.precision(8);
   // cylinder_surface << "solution\n" << cylinder_surface_submesh <<
   //                  temperature_cylinder_surface_gf <<
   //                  "pause\n" << std::flush;

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      d_ode_solver.Step(temperature_block, t, dt);
      {
         temperature_block_gf.SetFromTrueDofs(temperature_block);

         ParSubMesh::Transfer(temperature_block_gf, temperature_cylinder_gf);

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

         cyl_sol_sock << "parallel " << num_procs << " " << myid << "\n";
         cyl_sol_sock << "solution\n" << cylinder_submesh << temperature_cylinder_gf <<
                      std::flush;
         block_sol_sock << "parallel " << num_procs << " " << myid << "\n";
         block_sol_sock << "solution\n" << block_submesh << temperature_block_gf <<
                        std::flush;
      }
   }

   return 0;
}
