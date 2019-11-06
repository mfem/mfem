#include "mfem.hpp"
#include "multigrid.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


MGSolver::MGSolver(HypreParMatrix * Af_, std::vector<HypreParMatrix *> P_,std::vector<ParFiniteElementSpace * > fespaces)
   : Solver(Af_->Height(), Af_->Width()), Af(Af_), P(P_) {

   NumGrids = P.size();
   S.resize(NumGrids);
   A.resize(NumGrids + 1);

   A[NumGrids] = Af;
   for (int i = NumGrids ; i > 0; i--)
   {
      A[i - 1] = RAP(A[i], P[i - 1]);
   }
   // Set up coarse solve operator
   petsc = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   // Convert to PetscParMatrix
   petsc->SetOperator(PetscParMatrix(A[0], Operator::PETSC_MATAIJ));
   invAc = petsc;
   for (int i = NumGrids - 1; i >= 0 ; i--)
   {
      S[i] = new ParSchwarzSmoother(fespaces[i]->GetParMesh(),1,fespaces[i+1],A[i+1]);
      S[i]->SetDumpingParam(1.0/5.0);
      // S[i]->SetType(HypreSmoother::Jacobi);
      // S[i]->SetOperator(*A[i+1]);
   }
}

void MGSolver::Mult(const Vector &r, Vector &z) const
{
   // Residual vectors
   std::vector<Vector> rv(NumGrids + 1);
   // correction vectors
   std::vector<Vector> zv(NumGrids + 1);
   // allocation
   for (int i = 0; i <= NumGrids ; i++)
   {
      int n = A[i]->Width();
      rv[i].SetSize(n);
      zv[i].SetSize(n);
   }
   // Initial residual
   rv[NumGrids] = r;
   // smooth and update residuals down to the coarsest level
   for (int i = NumGrids; i > 0 ; i--)
   {
      // Pre smooth
      S[i - 1]->Mult(rv[i], zv[i]); zv[i] *= theta;
      // compute residual
      Vector w(A[i]->Height());
      A[i]->Mult(zv[i], w);
      rv[i] -= w;
      // Restrict
      P[i - 1]->MultTranspose(rv[i], rv[i - 1]);
   }

   // Coarse grid Solve
   invAc->Mult(rv[0], zv[0]);
   //
   for (int i = 1; i <= NumGrids ; i++)
   {
      // Prolong correction
      Vector u(P[i - 1]->Height());
      P[i - 1]->Mult(zv[i - 1], u);
      // Update correction
      zv[i] += u;
      // Update residual
      Vector v(A[i]->Height());
      A[i]->Mult(u, v); rv[i] -= v;
      // Post smooth
      S[i - 1]->Mult(rv[i], v); v *= theta;
      // Update correction
      zv[i] += v;
   }
   z = zv[NumGrids];
}

MGSolver::~MGSolver() 
{
   int n = S.size();
   for (int i = n - 1; i >= 0 ; i--)
   {
      delete S[i];
      delete A[i];
   }
   S.clear();
   A.clear();
   delete invAc;
}

BlockMGSolver::BlockMGSolver(Array2D<HypreParMatrix *> Af_, std::vector<HypreParMatrix *> P_,std::vector<ParFiniteElementSpace * > fespaces)
   : Solver(Af_(0,0)->Height()+Af_(1,0)->Height(), Af_(0,0)->Width()+Af_(1,0)->Width()), Af(Af_), P(P_) {

   NumGrids = P.size();
   BlkP.resize(NumGrids);
   BlkA.resize(NumGrids+1);
   S.resize(NumGrids);
   A.resize(NumGrids + 1);
   A[NumGrids] = Af;
   Aoffsets.resize(NumGrids+1);
   Poffsets_i.resize(NumGrids);
   Poffsets_j.resize(NumGrids);
   // Construct Bilinear form Matrices on each level
   for (int k = NumGrids ; k > 0; k--)
   {
      A[k - 1].SetSize(2,2);
      Aoffsets[k].SetSize(3); Aoffsets[k][0] = 0;
      Aoffsets[k][1] = A[k](0,0)->Height();
      Aoffsets[k][2] = A[k](1,1)->Height();
      Aoffsets[k].PartialSum();
      BlkA[k] = new BlockOperator(Aoffsets[k]);
      for (int i=0; i<2; i++)
      {
         for (int j=0; j<2; j++)
         {
            A[k - 1](i,j) = RAP(A[k](i,j), P[k - 1]);
            BlkA[k]->SetBlock(i,j,A[k](i,j));
         }
      }

      Poffsets_i[k-1].SetSize(3); Poffsets_i[k-1][0] = 0;
      Poffsets_j[k-1].SetSize(3); Poffsets_j[k-1][0] = 0;
      Poffsets_i[k-1][1] = P[k-1]->Height(); Poffsets_j[k-1][1] = P[k-1]->Width();
      Poffsets_i[k-1][2] = P[k-1]->Height(); Poffsets_j[k-1][2] = P[k-1]->Width();
      Poffsets_i[k-1].PartialSum();
      Poffsets_j[k-1].PartialSum();

      BlkP[k-1] = new BlockOperator(Poffsets_i[k-1],Poffsets_j[k-1]);
      BlkP[k-1]->SetBlock(0,0,P[k-1]);
      BlkP[k-1]->SetBlock(1,1,P[k-1]);
   }
   // Set up coarse solve operator
   // Convert the corse grid block matrix to a HypreParMatrix
   Array<int> offsets(3);
   offsets[0]=0;
   offsets[1]=A[0](0,0)->Height();
   offsets[2]=A[0](1,1)->Height();
   offsets.PartialSum();
   Array2D<double> coeff(2,2);
   coeff(0,0) = 1.0;  coeff(0,1) = 1.0;
   coeff(1,0) = 1.0;  coeff(1,1) = 1.0;
   // Convert to PetscParMatrix
   HypreParMatrix * Ac;
   Ac = CreateHypreParMatrixFromBlocks(MPI_COMM_WORLD, offsets, A[0], coeff);
   // Convert to PetscParMatrix
   PetscParMatrix * petsc = new PetscParMatrix(Ac, Operator::PETSC_MATAIJ);
   delete Ac;
   invAc = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   invAc->SetOperator(*petsc);
   delete petsc;
   // Smoother
   for (int i = NumGrids - 1; i >= 0 ; i--)
   {
      S[i] = new BlkParSchwarzSmoother(fespaces[i]->GetParMesh(),1,fespaces[i+1],A[i+1]);
      // S[i]->SetDumpingParam(1.0/5.0);
   }
}

 void BlockMGSolver::Mult(const Vector &r, Vector &z) const
{
   // Residual vectors
   std::vector<Vector> rv(NumGrids + 1);
   // correction vectors
   std::vector<Vector> zv(NumGrids + 1);
   // allocation
   for (int i = 0; i <= NumGrids ; i++)
   {
      int n = (i==0) ? invAc->Height(): BlkA[i]->Width();
      rv[i].SetSize(n);
      zv[i].SetSize(n);
   }
   // Initial residual
   rv[NumGrids] = r;

   // smooth and update residuals down to the coarsest level
   for (int i = NumGrids; i > 0 ; i--)
   {
      // Pre smooth
      S[i - 1]->Mult(rv[i], zv[i]); zv[i] *= theta;
      // compute residual
      int n = BlkA[i]->Width();
      Vector w(n);
      BlkA[i]->Mult(zv[i], w);
      rv[i] -= w;
      // Restrict
      BlkP[i - 1]->MultTranspose(rv[i], rv[i - 1]);
   }

   // Coarse grid Solve
   invAc->Mult(rv[0], zv[0]);
   //
   for (int i = 1; i <= NumGrids ; i++)
   {
      // Prolong correction
      Vector u(BlkP[i - 1]->Height());
      BlkP[i - 1]->Mult(zv[i - 1], u);
      // Update correction
      zv[i] += u;
      // Update residual
      Vector v(BlkA[i]->Height());
      BlkA[i]->Mult(u, v); rv[i] -= v;
      // Post smooth
      S[i - 1]->Mult(rv[i], v); v *= theta;
      // Update correction
      zv[i] += v;
   }
   z = zv[NumGrids];
}

BlockMGSolver::~BlockMGSolver() 
{
   for (int i = NumGrids - 1; i >= 0 ; i--)
   {
      delete S[i];
      delete BlkP[i];
      delete BlkA[i];
      for (int j=0; j<2; j++)
      {
         for (int k=0; k<2; k++)
         {
            delete A[i](j,k);
         }
      }
      A[i].DeleteAll();
   }
   delete BlkA[NumGrids];
   delete invAc;
   A.clear();
}