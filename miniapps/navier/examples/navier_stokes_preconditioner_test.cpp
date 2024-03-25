#include <mfem.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../data/star.mesh";

   int order = 2;
   int ser_ref = 0;
   int par_ref = 0;
   double dt = 1.0;
   double nu = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&dt, "-dt", "--dt", "Time step");
   args.AddOption(&nu, "-nu", "--nu", "kinematic viscosity");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.ParseCheck();

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   const int dim = mesh.Dimension();
   H1_FECollection vel_fec(order, dim);
   H1_FECollection pres_fec(order - 1, dim);

   ParFiniteElementSpace vel_fes(&mesh, &vel_fec, dim); 
   ParFiniteElementSpace pres_fes(&mesh, &pres_fec);

   IntegrationRules intrules(0, Quadrature1D::GaussLobatto);

   IntegrationRule ir = intrules.Get(vel_fes.GetFE(0)->GetGeomType(),
                             (int)(2*(vel_fes.GetOrder(0)+1) - 3));

   printf("#velocity_dofs: %d\n", vel_fes.GlobalTrueVSize());

   Array<int> vel_ess_bdr, pres_ess_bdr;

   bool inflow_outflow_bcs = false;

   vel_ess_bdr.SetSize(mesh.bdr_attributes.Max());
   vel_ess_bdr = 1;

   if (inflow_outflow_bcs) { vel_ess_bdr[1] = 0; }

   pres_ess_bdr = vel_ess_bdr;
   for (int &marker : pres_ess_bdr) { marker = !marker; }

   Array<int> vel_ess_dofs;
   Array<int> pres_ess_dofs;
   vel_fes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_dofs);
   pres_fes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_dofs);

   ConstantCoefficient dt_coeff(dt);
   ConstantCoefficient dt_inv_coeff(1.0 / dt);
   // ConstantCoefficient nu_coeff(dt * nu);
   ConstantCoefficient nu_inv_coeff(1.0/nu);
   ConstantCoefficient zero_coeff(0.0);
   ConstantCoefficient minus_one(-1.0);

   ConstantCoefficient F_coeff(nu * dt);
   ParBilinearForm F_form(&vel_fes);
   BilinearFormIntegrator *integrator;
   integrator = new VectorMassIntegrator;
   integrator->SetIntRule(&ir);
   F_form.AddDomainIntegrator(integrator);
   integrator = new VectorDiffusionIntegrator(F_coeff);
   integrator->SetIntRule(&ir);
   F_form.AddDomainIntegrator(integrator);
   // F_form.AddDomainIntegrator(new ElasticityIntegrator(zero_coeff, nu_coeff));
   // F_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   F_form.Assemble();
   F_form.Finalize();

   OperatorHandle F;
   F_form.FormSystemMatrix(vel_ess_dofs, F);

   ParMixedBilinearForm d(&vel_fes, &pres_fes);
   integrator = new VectorDivergenceIntegrator;
   integrator->SetIntRule(&ir);
   d.AddDomainIntegrator(integrator);
   // d.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   d.Assemble();
   d.Finalize();

   OperatorHandle D;
   d.FormRectangularSystemMatrix(vel_ess_dofs, pres_ess_dofs, D);

   // TransposeOperator g(*D);
   ParMixedBilinearForm g(&pres_fes, &vel_fes);
   integrator = new GradientIntegrator(dt_coeff);
   integrator->SetIntRule(&ir);
   g.AddDomainIntegrator(integrator);
   // g.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   g.Assemble();
   g.Finalize();

   OperatorHandle G;
   g.FormRectangularSystemMatrix(pres_ess_dofs, vel_ess_dofs, G);

   Array<int> offsets({0, vel_fes.GetTrueVSize(), pres_fes.GetTrueVSize()});
   offsets.PartialSum();

   // [  M+K   G ] [ u ] = [ f ]
   // [  D     0 ] [ p ]   [ 0 ]

   BlockOperator A(offsets);
   A.SetBlock(0, 0, F.Ptr());
   A.SetBlock(0, 1, G.Ptr());
   A.SetBlock(1, 0, D.Ptr());

   ParBilinearForm mp_form(&pres_fes);
   ConstantCoefficient mp_coeff(1.0 / nu);
   integrator = new MassIntegrator(mp_coeff);
   integrator->SetIntRule(&ir);
   mp_form.AddDomainIntegrator(integrator);
   // mp_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mp_form.Assemble();
   mp_form.Finalize();
   OperatorHandle Mp;
   mp_form.FormSystemMatrix(pres_ess_dofs, Mp);

   ParBilinearForm Lp_form(&pres_fes);
   ConstantCoefficient lp_coeff(dt);
   integrator = new DiffusionIntegrator();
   integrator->SetIntRule(&ir);
   Lp_form.AddDomainIntegrator(integrator);
   // Lp_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Lp_form.Assemble();
   Lp_form.Finalize();
   OperatorHandle Lp;
   Lp_form.FormSystemMatrix(pres_ess_dofs, Lp);

   ConstantCoefficient fp_coeff(dt * nu);
   ParBilinearForm Fp_form(&pres_fes);
   // Fp_form.AddDomainIntegrator(new MassIntegrator);
   // Fp_form.AddDomainIntegrator(new DiffusionIntegrator(fp_coeff));
   // Fp_form.Assemble();
   // Fp_form.Finalize();
   OperatorHandle Fp;
   // Fp_form.FormSystemMatrix(pres_ess_dofs, Fp);

   // HypreSmoother Mp_inv(*Mp.As<HypreParMatrix>());
   // OperatorJacobiSmoother Mp_pc(mp_form, pres_ess_dofs);
   HypreILU Mp_pc;
   Mp_pc.SetOperator(*Mp.As<HypreParMatrix>());
   // CGSolver Mp_inv(MPI_COMM_WORLD);
   // Mp_inv.SetRelTol(1e-2);
   // Mp_inv.SetMaxIter(10);
   // Mp_inv.SetPreconditioner(Mp_pc);
   // Mp_inv.SetOperator(*Mp);

   Solver &Mp_inv = Mp_pc;
   
   LORSolver<HypreBoomerAMG> Lp_pc(Lp_form, pres_ess_dofs);
   CGSolver Lp_krylov(MPI_COMM_WORLD);
   Lp_krylov.SetRelTol(1e-2);
   Lp_krylov.SetMaxIter(10);
   Lp_krylov.SetOperator(*Lp);
   Lp_krylov.SetPreconditioner(Lp_pc);
   // OrthoSolver Lp_inv(MPI_COMM_WORLD);
   // Lp_inv.SetOperator(*Lp);
   // Lp_inv.SetSolver(Lp_krylov);

   Solver &Lp_inv = Lp_krylov;

   struct CahouetChabardPC : Solver
   {
      Solver &Mp_inv;
      Solver &Lp_inv;
      OperatorHandle Fp;
      mutable Vector z, w;
      const double dt;
      CahouetChabardPC(Solver &Mp_inv_, Solver &Lp_inv_, OperatorHandle &Fp_, const double dt) :
         Solver(Mp_inv_.Height()),
         Mp_inv(Mp_inv_),
         Lp_inv(Lp_inv_),
         Fp(Fp_),
         dt(dt) { }
      void Mult(const Vector &x, Vector &y) const
      {
         z.SetSize(y.Size());
         w.SetSize(y.Size());

         Lp_inv.Mult(x, y);
         y /= dt;
         // Fp->Mult(z, w);
         Mp_inv.Mult(x, w);
         // y = z;
         y += w;
      }
      void SetOperator(const Operator &op) { }
   };

   CahouetChabardPC S_inv(Mp_inv, Lp_inv, Fp, dt);

   // PMM S_inv(Mp_inv);

   // MUMPSSolver F_inv;
   // F_inv.SetOperator(*F.As<HypreParMatrix>());

   CGSolver F_inv(MPI_COMM_WORLD);
   F_inv.SetRelTol(1e-4);
   F_inv.SetMaxIter(4);
   F_inv.SetOperator(*F);
   LORSolver<HypreBoomerAMG> F_pc(F_form, vel_ess_dofs);
   // F_pc.GetSolver().SetSystemsOptions(dim, true);
   // HypreBoomerAMG F_pc(*F.As<HypreParMatrix>());
   // F_pc.SetSystemsOptions(dim);
   // HypreILU F_pc;
   // F_pc.SetOperator(*F.As<HypreParMatrix>());
   // OperatorJacobiSmoother F_pc(F_form, vel_ess_dofs);
   F_inv.SetPreconditioner(F_pc);
   // Solver &F_inv = F_pc;

   BlockLowerTriangularPreconditioner P(offsets);
   P.SetBlock(0, 0, &F_inv);
   P.SetBlock(1, 0, D.Ptr());
   P.SetBlock(1, 1, &S_inv);

   FGMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-6);
   krylov.SetAbsTol(1e-12);
   krylov.SetMaxIter(2000);
   krylov.SetPrintLevel(1);
   krylov.SetOperator(A);
   krylov.SetPreconditioner(P);

   Vector X(A.Height()), B(A.Height());

   B.Randomize(1);
   B *= 1e-4;
   B.SetSubVector(vel_ess_dofs, 0.0);
   for (int i = offsets[1]; i < offsets[2]; ++i) { B[i] = 0.0; }

   X = 0.0;
   krylov.Mult(B, X);

   return 0;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}
