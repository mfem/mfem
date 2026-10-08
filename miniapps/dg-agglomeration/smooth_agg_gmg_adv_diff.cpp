// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//                     ------------------------
//                     DG Smooth-Agg GMG Solver
//                     ------------------------

#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "mg_agglom.hpp"

using namespace std;
using namespace mfem;

int problem;

// RHS
real_t rhs_function(const Vector &x);

// true solution
real_t u_true(const Vector &x);

void velocity_func(const Vector &x, Vector &v)
{

   switch (problem)
   {
      case 1:
      {
         v(0) = 1.0;
         v(1) = 2.0;
         v(2) = 3.0;
         v *= 1.0/sqrt(14);
         break;
      }
      case 2: 
      {
         int dim = x.Size();
         Vector X(dim);
         for (int i = 0; i < dim; i++)
         {
            real_t center = 0.5;
            X(i) = 2 * (x(i) - center);
         }
         const real_t w = M_PI/2;
         v(0) = -w*X(1); 
         v(1) = w*X(0); 
         v(2) = 0.0;
         v /= v.Norml2();
      }
   }
}

// Algebraic multigrid preconditioner for advective problems based on
// approximate ideal restriction (AIR). Most effective when matrix is
// first scaled by DG block inverse, and AIR applied to scaled matrix.
// See https://doi.org/10.1137/17M1144350.
class AIR_prec : public Solver
{
private:
   const HypreParMatrix *A;
   // Copy of A scaled by block-diagonal inverse
   HypreParMatrix A_s;

   HypreBoomerAMG *AIR_solver;
   int blocksize;

public:
   AIR_prec(int blocksize_) : AIR_solver(NULL), blocksize(blocksize_) { }

   void SetOperator(const Operator &op) override
   {
      width = op.Width();
      height = op.Height();


      A = dynamic_cast<const HypreParMatrix *>(&op);
      MFEM_VERIFY(A != NULL, "AIR_prec requires a HypreParMatrix.")

      // Scale A by block-diagonal inverse
      BlockInverseScale(A, &A_s, NULL, NULL, blocksize,
                        BlockInverseScaleJob::MATRIX_ONLY);
      delete AIR_solver;
      AIR_solver = new HypreBoomerAMG(A_s);
      AIR_solver->SetAdvectiveOptions(1, "", "FA");
      AIR_solver->SetPrintLevel(0);
      AIR_solver->SetMaxLevels(50);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // Scale the rhs by block inverse and solve system
      HypreParVector z_s;
      BlockInverseScale(A, NULL, &x, &z_s, blocksize,
                        BlockInverseScaleJob::RHS_ONLY);
      AIR_solver->Mult(z_s, y);
   }

   ~AIR_prec() override
   {
      delete AIR_solver;
   }
};

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    const char *mesh_file = "../../data/inline-hex.mesh";
    int order = 1;
    real_t kappa_0 = 1.0;
    int num_levels = 2;
    real_t diff_c = 1.0;
    int agglom = 0;
    problem = 2;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
    args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
    args.AddOption(&agglom, "-agg", "--agglomeration-prec", "Choose Preconditioner. 0 is agglom. 1 is AIR.");
    // args.AddOption(&ref_levels, "-r", "--refine", "Refinement levels.");
    args.AddOption(&order, "-o", "--order", "Polynomial degree.");
    args.AddOption(&kappa_0, "-k", "--kappa", "DG penalty parameter.");
    args.AddOption(&diff_c, "-dc", "--diff_c", "Diffusion Coefficient.");
    // args.AddOption(&ncoarse, "-nc", "--ncoarse", "Number of Fine Elements per Coarse.");
    args.AddOption(&num_levels, "-nl", "--levels", "Number of Multigrid Levels.");
    args.ParseCheck();

    MFEM_VERIFY(agglom == 0 || agglom == 1, "agg must be set to 0 or 1.")

    Mesh mesh(mesh_file);
    const int dim = mesh.Dimension();
    int ncoarse = pow(2, dim); 
    int ref_levels = num_levels - 2;


    for (int i = 0; i < ref_levels; ++i) { mesh.UniformRefinement(); }

    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

    DG_FECollection fec(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace* fespace = new ParFiniteElementSpace(pmesh, &fec);
    cout << "Number of unknowns: " << fespace->GetTrueVSize() << endl;

    if (diff_c != 0) {cout << "Peclet Number: " << 2.0 / diff_c << endl;}
    else {cout << "Peclet Number: inf" << endl;}

    int ne = mesh.GetNE();

    const real_t sigma = -1.0;
    const real_t kappa = kappa_0 * (order + 1.0) * (order + 1.0) / 2.0;

    // LinearForm b(&fespace);
    ConstantCoefficient diff_coef(diff_c);
    ConstantCoefficient mone(-1.0);
    ConstantCoefficient one(1.0);
    ConstantCoefficient zero(0.0);
    VectorFunctionCoefficient velocity(dim, velocity_func);

    ParGridFunction x(fespace);
    x = 0.0;

    ParBilinearForm a(fespace);
    if (diff_c != 0.0)
    {
      a.AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
      a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));
      a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));
    }
    a.AddDomainIntegrator(new ConvectionIntegrator(velocity, 1.0));
    a.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, 1.0));
    a.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, 1.0));
    a.AddDomainIntegrator(new MassIntegrator);
    a.Assemble();
    a.Finalize();

    ParLinearForm b(fespace);
    b.AddDomainIntegrator(new DomainLFIntegrator(one));
    b.Assemble();
    SparseMatrix &A = a.SpMat();
    Solver *prec;

    if(agglom == 0)
    {
        std::cout << "Using Smooth Agg Agglom Preconditioner. " << std::endl;

        // std::string file_name = "A_mfem.mtx";
        // std::ofstream ofs1("../../../adaptiveMG/" + file_name);
        // A.PrintMM(ofs1); 
        // ofs1.close();

        SmoothedAggregationGMG mg(*fespace, A, ncoarse, num_levels, false);
        mg.SetCycleType(mfem::MultigridBase::CycleType::VCYCLE, 3, 3);

        GMRESSolver gmres;
        gmres.SetRelTol(1e-7);
        gmres.SetMaxIter(5000);
        gmres.SetPrintLevel(1);
        gmres.SetKDim(1000);
        gmres.SetOperator(A);
        gmres.SetPreconditioner(mg);
        Vector& b_vec = b;
        // std::string file_name_b = "b_mfem.txt";
        // std::ofstream ofsb("../../../adaptiveMG/" + file_name_b);
        // b_vec.Print(ofsb); 
        // ofsb.close();
        x = 0.0;
        gmres.Mult(b_vec, x);
    }
    else
    {
        std::cout << "Using AIR Preconditioner. " << std::endl;
        HypreParMatrix* Ap = a.ParallelAssemble();
        int block_size = fespace->GetTypicalFE()->GetDof();
        prec = new AIR_prec(block_size);
        GMRESSolver gmres;
        gmres.SetRelTol(1e-7);
        gmres.SetMaxIter(5000);
        gmres.SetPrintLevel(1);
        gmres.SetKDim(1000);
        gmres.SetPreconditioner(*prec);
        gmres.SetOperator(*Ap);
        HypreParVector* b_vec = b.ParallelAssemble();
        x = 1.0;
        gmres.Mult(*b_vec, x);
        delete prec;
        delete Ap;
        delete b_vec;

    }

    std::cout << "Solution norm = " << x.Norml2() << std::endl;


   GridFunction xgf(fespace);
   xgf = x;
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << xgf << flush;
   }


    delete fespace;
    delete pmesh;
    return 0;
}
