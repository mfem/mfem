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

   printf("#velocity_dofs: %d\n", vel_fes.GetTrueVSize());

   Array<int> vel_ess_bdr, pres_ess_bdr;

   // if (mesh.bdr_attributes.Size() > 0)
   // {

   bool cyl = mesh.bdr_attributes.Size() >= 7;

   vel_ess_bdr.SetSize(mesh.bdr_attributes.Max());
   vel_ess_bdr = 1;
   if (cyl) { vel_ess_bdr[1] = 0; }

   pres_ess_bdr = vel_ess_bdr;
   for (int &marker : pres_ess_bdr) { marker = !marker; }

   if (cyl)
   {
      vel_ess_bdr[4] = 0;
      pres_ess_bdr[4] = 0;
   }
   // }
   // ess_bdr = 0;

   // cout << "Velocity:\n";
   // vel_ess_bdr.Print(std::cout, 1);
   // cout << "Pressure:\n";
   // pres_ess_bdr.Print(std::cout, 1);
   // return 0;

   Array<int> vel_ess_dofs;
   Array<int> pres_ess_dofs;
   vel_fes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_dofs);
   pres_fes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_dofs);

   ConstantCoefficient dt_inv_coeff(1.0/dt);
   ConstantCoefficient nu_coeff(nu);

   ParBilinearForm F_form(&vel_fes);
   F_form.AddDomainIntegrator(new VectorMassIntegrator(dt_inv_coeff));
   F_form.AddDomainIntegrator(new VectorDiffusionIntegrator(nu_coeff));
   F_form.Assemble();
   F_form.Finalize();

   OperatorHandle F;
   F_form.FormSystemMatrix(vel_ess_dofs, F);

   ConstantCoefficient minus_one(-1.0);

   ParMixedBilinearForm d(&vel_fes, &pres_fes);
   d.AddDomainIntegrator(new VectorDivergenceIntegrator(&minus_one));
   d.Assemble();
   d.Finalize();

   OperatorHandle D;
   d.FormRectangularSystemMatrix(vel_ess_dofs, pres_ess_dofs, D);

   TransposeOperator Dt(*D);

   Array<int> offsets({0, vel_fes.GetTrueVSize(), pres_fes.GetTrueVSize()});
   offsets.PartialSum();

   BlockOperator A(offsets);
   A.SetBlock(0, 0, F.Ptr());
   A.SetBlock(0, 1, &Dt);
   A.SetBlock(1, 0, D.Ptr());

   // [  A   -D^t ] [ u ] = [ f ]
   // [ -D     0  ] [ p ]   [ 0 ]

   ParBilinearForm mp_form(&pres_fes);
   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.Finalize();
   OperatorHandle Mp;
   mp_form.FormSystemMatrix(pres_ess_dofs, Mp);

   ParBilinearForm Lp_form(&pres_fes);
   Lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   Lp_form.Assemble();
   Lp_form.Finalize();
   OperatorHandle Lp;
   Lp_form.FormSystemMatrix(pres_ess_dofs, Lp);

   ParBilinearForm Fp_form(&pres_fes);
   Fp_form.AddDomainIntegrator(new MassIntegrator(dt_inv_coeff));
   Fp_form.AddDomainIntegrator(new DiffusionIntegrator(nu_coeff));
   Fp_form.Assemble();
   Fp_form.Finalize();
   OperatorHandle Fp;
   Fp_form.FormSystemMatrix(pres_ess_dofs, Fp);

   // OperatorJacobiSmoother M_inv(m, pres_ess_dofs);
   SparseMatrix M_local;
   Mp.As<HypreParMatrix>()->GetDiag(M_local);
   UMFPackSolver Mp_inv(M_local);

   SparseMatrix Lp_local;
   Lp.As<HypreParMatrix>()->GetDiag(Lp_local);
   UMFPackSolver Lp_inv(Lp_local);

   struct ApproxSchurInv : Solver
   {
      Solver &Mp_inv;
      Solver &Lp_inv;
      OperatorHandle Fp;
      mutable Vector z, w;
      ApproxSchurInv(Solver &Mp_inv_, Solver &Lp_inv_, OperatorHandle &Fp_) :
         Solver(Mp_inv_.Width()),
         Mp_inv(Mp_inv_),
         Lp_inv(Lp_inv_),
         Fp(Fp_) { }
      void Mult(const Vector &x, Vector &y) const
      {
         z.SetSize(y.Size());
         w.SetSize(y.Size());

         Lp_inv.Mult(x, z);
         Fp->Mult(z, w);
         Mp_inv.Mult(w, y);
      }
      void SetOperator(const Operator &op) { }
   };

   ApproxSchurInv S_inv(Mp_inv, Lp_inv, Fp);

   // HypreBoomerAMG K_inv(*K.As<HypreParMatrix>());
   SparseMatrix F_diag;
   F.As<HypreParMatrix>()->GetDiag(F_diag);
   UMFPackSolver F_inv(F_diag);

   // {
   //    Vector B_vel(vel_fes.GetTrueVSize());
   //    Vector X_vel(vel_fes.GetTrueVSize());

   //    B_vel.Randomize(1);
   //    X_vel = 0.0;

   //    CGSolver cg(MPI_COMM_WORLD);
   //    cg.SetRelTol(1e-12);
   //    cg.SetOperator(*K);
   //    cg.SetPreconditioner(F_inv);
   //    cg.SetPrintLevel(1);
   //    cg.Mult(B_vel, X_vel);
   // }
   // return 0;

   // BlockDiagonalPreconditioner P(offsets);
   // P.SetDiagonalBlock(0, &F_inv);
   // P.SetDiagonalBlock(1, &S_inv);

   BlockLowerTriangularPreconditioner P(offsets);
   P.SetBlock(0, 0, &F_inv);
   P.SetBlock(1, 0, D.Ptr());
   P.SetBlock(1, 1, &S_inv);

   GMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-8);
   krylov.SetMaxIter(2000);
   krylov.SetKDim(2000);
   krylov.SetPrintLevel(1);
   krylov.SetOperator(A);
   krylov.SetPreconditioner(P);

   Vector X(A.Height()), B(A.Height());

   B.Randomize(1);
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
