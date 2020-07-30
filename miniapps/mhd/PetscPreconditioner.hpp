#include "mfem.hpp"
#include <iostream>
#include <fstream>

//copied from petsc.cpp
#include "petsc.h"
#if defined(PETSC_HAVE_HYPRE)
#include "petscmathypre.h"
#endif
// Error handling
// Prints PETSc's stacktrace and then calls MFEM_ABORT
// We cannot use PETSc's CHKERRQ since it returns a PetscErrorCode
#define PCHKERRQ(obj,err) do {                                                   \
     if ((err))                                                                  \
     {                                                                           \
        PetscError(PetscObjectComm((PetscObject)(obj)),__LINE__,_MFEM_FUNC_NAME, \
                   __FILE__,(err),PETSC_ERROR_REPEAT,NULL);                      \
        MFEM_ABORT("Error in PETSc. See stacktrace above.");                     \
     }                                                                           \
  } while(0);


using namespace std;
using namespace mfem;

int  ijacobi=4; //number of Jacobi iteration 
bool smoothOmega=true;

//------FULL physics-based petsc pcshell preconditioenr------
class FullBlockSolver : public Solver
{
private:
   Mat **sub; 

   // Create internal KSP objects to handle the subproblems
   KSP kspblock[4];

   // Create PetscParVectors as placeholders X and Y
   mutable PetscParVector *X, *Y;

   IS index_set[3];

   //solutions holder
   mutable Vec b0, b1, b2, y0, y1, y2;
   mutable Vec b, bhat, rhs, tmp; 
   Vec diag;

public:
   FullBlockSolver(const OperatorHandle &oh);

   virtual void SetOperator (const Operator &op)
   { MFEM_ABORT("FullBlockSolver::SetOperator is not supported.");}

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual ~FullBlockSolver();
};

FullBlockSolver::FullBlockSolver(const OperatorHandle &oh) : Solver() { 
   PetscErrorCode ierr; 

   // Get the PetscParMatrix out of oh.       
   PetscParMatrix *PP;
   oh.Get(PP);
   Mat P = *PP; // type cast to Petsc Mat
   
   // update base (Solver) class
   width = PP->Width();
   height = PP->Height();
   X = new PetscParVector(PETSC_COMM_WORLD, *this, true, false);
   Y = new PetscParVector(PETSC_COMM_WORLD, *this, false, false);

   PetscInt M, N;
   ierr=MatNestGetSubMats(P,&N,&M,&sub); PCHKERRQ(sub[0][0], ierr);// sub is an N by M array of matrices
   ierr=MatNestGetISs(P, index_set, NULL);  PCHKERRQ(index_set, ierr);// get the index sets of the blocks

   //stiffness
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[0]);    PCHKERRQ(kspblock[0], ierr);
   ierr=KSPSetOperators(kspblock[0], sub[2][0], sub[2][0]);PCHKERRQ(sub[2][0], ierr);
   KSPAppendOptionsPrefix(kspblock[0],"s1_");
   KSPSetFromOptions(kspblock[0]);
   KSPSetUp(kspblock[0]);

   //schur complement
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[1]);    PCHKERRQ(kspblock[1], ierr);
   ierr=KSPSetOperators(kspblock[1], sub[1][1], sub[1][1]);PCHKERRQ(sub[1][1], ierr);
   KSPAppendOptionsPrefix(kspblock[1],"s2_");
   KSPSetFromOptions(kspblock[1]);
   KSPSetUp(kspblock[1]);
         
   //mass matrix
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[2]);    PCHKERRQ(kspblock[2], ierr);
   ierr=KSPSetOperators(kspblock[2], sub[2][2], sub[2][2]);PCHKERRQ(sub[2][2], ierr);
   KSPAppendOptionsPrefix(kspblock[2],"s3_");
   KSPSetFromOptions(kspblock[2]);
   KSPSetUp(kspblock[2]);

   Mat ARe = sub[0][0];
   Mat Mmatlp = sub[0][2];

   if (smoothOmega)
   {
      //ARe matrix (for version 2)
      ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[3]);    PCHKERRQ(kspblock[3], ierr);
      ierr=KSPSetOperators(kspblock[3], ARe, ARe);PCHKERRQ(ARe, ierr);
      KSPAppendOptionsPrefix(kspblock[3],"s4_");
      KSPSetFromOptions(kspblock[3]);
      KSPSetUp(kspblock[3]);
   }

   MatCreateVecs(sub[0][0], &b, NULL);
   MatCreateVecs(sub[0][0], &bhat, NULL);
   MatCreateVecs(sub[0][0], &diag, NULL);
   MatCreateVecs(sub[0][0], &rhs, NULL);
   MatCreateVecs(sub[0][0], &tmp, NULL);

   //get diag consistent with Schur complement
   if(Mmatlp==NULL)
      MatGetDiagonal(ARe,diag);
   else
      MatGetDiagonal(Mmatlp,diag);
}

//Mult will only be called once
void FullBlockSolver::Mult(const Vector &x, Vector &y) const
{

   Mat ARe = sub[0][0];
   Mat Mmat = sub[2][2];
   Mat Kmat = sub[2][0];

   X->PlaceArray(x.GetData()); 
   Y->PlaceArray(y.GetData());

   VecGetSubVector(*X,index_set[0],&b0);
   VecGetSubVector(*X,index_set[1],&b1);
   VecGetSubVector(*X,index_set[2],&b2);

   //note [y0, y1, y2]=[phi, psi, w]
   VecGetSubVector(*Y,index_set[0],&y0);
   VecGetSubVector(*Y,index_set[1],&y1);
   VecGetSubVector(*Y,index_set[2],&y2);

   //first solve b=M*K^-1 (-b2 + ARe*M^-1 b0)
   KSPSolve(kspblock[2],b0,tmp);
   MatMult(ARe, tmp, rhs);
   VecAXPY(rhs, -1., b2);
   KSPSolve(kspblock[0],rhs,tmp);
   MatMult(Mmat, tmp, b);

   //Jacobi iteration with Schur complement
   for (int j = 0; j<ijacobi; j++)
   {
      if (j==0)
          VecCopy(b, bhat);
      else 
      {
          VecPointwiseMult(bhat,diag,y0);
          VecAXPY(bhat, 1.0, b);
          MatMult(ARe, y0, tmp);
          VecAXPY(bhat, -1.0, tmp);
      }

      VecPointwiseDivide(tmp, bhat, diag);
      MatMultAdd(sub[0][1], tmp, b1, rhs);

      //update y1 by inverting schur complemet 
      KSPSolve(kspblock[1],rhs,y1);

      //update y0
      MatMultAdd(sub[0][1], y1, bhat, rhs);
      VecPointwiseDivide(y0, rhs, diag);
   }

   //update y2 (version 2 increase smoothness in omega)
   if(!smoothOmega)
   {
      //version 1
      MatMult(Kmat, y0, rhs);
      VecAYPX(rhs, -1., b0);
      KSPSolve(kspblock[2],rhs,y2);
   }
   else{
      //version 2
      MatMult(sub[0][1], y1, rhs);
      KSPSolve(kspblock[2],rhs,rhs);
      MatMult(Kmat, rhs, tmp);
      MatMultAdd(sub[1][0], y0, b2, rhs);
      VecAXPY(rhs, -1., tmp);
      KSPSolve(kspblock[3],rhs,y2);

      MatMult(Mmat, y2, rhs);
      VecAYPX(rhs, -1., b0);
      KSPSolve(kspblock[0],rhs,y0);
   }

   VecRestoreSubVector(*X,index_set[0],&b0);
   VecRestoreSubVector(*X,index_set[1],&b1);
   VecRestoreSubVector(*X,index_set[2],&b2);

   VecRestoreSubVector(*Y,index_set[0],&y0);
   VecRestoreSubVector(*Y,index_set[1],&y1);
   VecRestoreSubVector(*Y,index_set[2],&y2);

   X->ResetArray();
   Y->ResetArray();
}

FullBlockSolver::~FullBlockSolver()
{
    for (int i=0; i<3; i++)
    {
        KSPDestroy(&kspblock[i]);
    }
    if(smoothOmega)
        KSPDestroy(&kspblock[3]);
    
    VecDestroy(&b);
    VecDestroy(&bhat); 
    VecDestroy(&diag);
    VecDestroy(&rhs);
    VecDestroy(&tmp);

    delete X;
    delete Y;
}

//------FULL-supg physics-based petsc pcshell preconditioenr------
class SupgBlockSolver : public Solver
{
private:
   Mat **sub; 
   Mat AReFull;

   // Create internal KSP objects to handle the subproblems
   KSP kspblock[4];

   // Create PetscParVectors as placeholders X and Y
   mutable PetscParVector *X, *Y;

   IS index_set[3];

   //solutions holder
   mutable Vec b0, b1, b2, y0, y1, y2;
   mutable Vec b, bhat, rhs, tmp; 
   Vec diag;

public:
   SupgBlockSolver(const OperatorHandle &oh);

   virtual void SetOperator (const Operator &op)
   { MFEM_ABORT("SupgBlockSolver::SetOperator is not supported.");}

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual ~SupgBlockSolver();
};

SupgBlockSolver::SupgBlockSolver(const OperatorHandle &oh) : Solver() { 
   PetscErrorCode ierr; 

   // Get the PetscParMatrix out of oh.       
   PetscParMatrix *PP;
   oh.Get(PP);
   Mat P = *PP; // type cast to Petsc Mat
   
   // update base (Solver) class
   width = PP->Width();
   height = PP->Height();
   X = new PetscParVector(PETSC_COMM_WORLD, *this, true, false);
   Y = new PetscParVector(PETSC_COMM_WORLD, *this, false, false);

   PetscInt M, N;
   ierr=MatNestGetSubMats(P,&N,&M,&sub); PCHKERRQ(sub[0][0], ierr);// sub is an N by M array of matrices
   ierr=MatNestGetISs(P, index_set, NULL);  PCHKERRQ(index_set, ierr);// get the index sets of the blocks

   //stiffness
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[0]);    PCHKERRQ(kspblock[0], ierr);
   ierr=KSPSetOperators(kspblock[0], sub[2][0], sub[2][0]);PCHKERRQ(sub[2][0], ierr);
   KSPAppendOptionsPrefix(kspblock[0],"s1_");
   KSPSetFromOptions(kspblock[0]);
   KSPSetUp(kspblock[0]);

   //schur complement
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[1]);    PCHKERRQ(kspblock[1], ierr);
   ierr=KSPSetOperators(kspblock[1], sub[1][1], sub[1][1]);PCHKERRQ(sub[1][1], ierr);
   KSPAppendOptionsPrefix(kspblock[1],"s2_");
   KSPSetFromOptions(kspblock[1]);
   KSPSetUp(kspblock[1]);
         
   //mass matrix
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[2]);    PCHKERRQ(kspblock[2], ierr);
   ierr=KSPSetOperators(kspblock[2], sub[2][2], sub[2][2]);PCHKERRQ(sub[2][2], ierr);
   KSPAppendOptionsPrefix(kspblock[2],"s3_");
   KSPSetFromOptions(kspblock[2]);
   KSPSetUp(kspblock[2]);

   Mat ARe = sub[0][0];
   Mat Stab = sub[0][2];
   if (Stab==NULL)
   { MFEM_ABORT("SupgBlockSolver: Stab Mat is not found.");}

   //PetscPrintf(PETSC_COMM_WORLD,"===inside petsc full-supg preconditioner===");

   //sum ARe and Stabilzation term
   MatDuplicate(ARe, MAT_COPY_VALUES, &AReFull);
   MatAXPY(AReFull,1.,Stab, DIFFERENT_NONZERO_PATTERN);

   //AReFull matrix 
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[3]);      PCHKERRQ(kspblock[3], ierr);
   ierr=KSPSetOperators(kspblock[3], AReFull, AReFull); PCHKERRQ(AReFull, ierr);
   KSPAppendOptionsPrefix(kspblock[3],"s4_");
   KSPSetFromOptions(kspblock[3]);
   KSPSetUp(kspblock[3]);

   MatCreateVecs(sub[0][0], &b, NULL);
   MatCreateVecs(sub[0][0], &bhat, NULL);
   MatCreateVecs(sub[0][0], &diag, NULL);
   MatCreateVecs(sub[0][0], &rhs, NULL);
   MatCreateVecs(sub[0][0], &tmp, NULL);

   //get diag consistent with Schur complement
   MatGetDiagonal(ARe,diag);
}

//Mult will only be called once
void SupgBlockSolver::Mult(const Vector &x, Vector &y) const
{
   Mat ARe = sub[0][0];
   Mat Mmat = sub[2][2];
   Mat Kmat = sub[2][0];
   Mat Stab = sub[0][2];

   X->PlaceArray(x.GetData()); 
   Y->PlaceArray(y.GetData());

   VecGetSubVector(*X,index_set[0],&b0);
   VecGetSubVector(*X,index_set[1],&b1);
   VecGetSubVector(*X,index_set[2],&b2);

   //note [y0, y1, y2]=[phi, psi, w]
   VecGetSubVector(*Y,index_set[0],&y0);
   VecGetSubVector(*Y,index_set[1],&y1);
   VecGetSubVector(*Y,index_set[2],&y2);

   //first solve b=M*K^-1 (-b2 + AReFull*M^-1 b0)
   KSPSolve(kspblock[2],b0,tmp);
   MatMult(AReFull, tmp, rhs);
   VecAXPY(rhs, -1., b2);
   KSPSolve(kspblock[0],rhs,tmp);
   MatMult(Mmat, tmp, b);

   //Jacobi iteration with Schur complement
   for (int j = 0; j<ijacobi; j++)
   {
      if (j==0)
      {
          VecCopy(b, bhat);
      }
      else 
      {
          VecPointwiseMult(bhat,diag,y0);
          VecAXPY(bhat, 1.0, b);
          MatMult(ARe, y0, tmp);
          VecAXPY(bhat, -1.0, tmp);

          //bhat-=M*K^(-1)*Stab*M^(-1)*K*y0
          MatMult(Kmat, y0, tmp);
          KSPSolve(kspblock[2],tmp,rhs);
          MatMult(Stab,rhs, tmp);
          KSPSolve(kspblock[0],tmp,rhs);
          MatMult(Mmat, rhs, tmp);
          VecAXPY(bhat, -1.0, tmp);
      }

      VecPointwiseDivide(tmp, bhat, diag);
      MatMultAdd(sub[0][1], tmp, b1, rhs);

      //update y1 by inverting schur complemet 
      KSPSolve(kspblock[1],rhs,y1);

      //update y0
      MatMultAdd(sub[0][1], y1, bhat, rhs);
      VecPointwiseDivide(y0, rhs, diag);
   }

   //update y2 
   MatMult(sub[0][1], y1, rhs);
   KSPSolve(kspblock[2],rhs,rhs);
   MatMult(Kmat, rhs, tmp);
   MatMultAdd(sub[1][0], y0, b2, rhs);
   VecAXPY(rhs, -1., tmp);
   KSPSolve(kspblock[3],rhs,y2);

   MatMult(Mmat, y2, rhs);
   VecAYPX(rhs, -1., b0);
   KSPSolve(kspblock[0],rhs,y0);

   VecRestoreSubVector(*X,index_set[0],&b0);
   VecRestoreSubVector(*X,index_set[1],&b1);
   VecRestoreSubVector(*X,index_set[2],&b2);

   VecRestoreSubVector(*Y,index_set[0],&y0);
   VecRestoreSubVector(*Y,index_set[1],&y1);
   VecRestoreSubVector(*Y,index_set[2],&y2);

   X->ResetArray();
   Y->ResetArray();
}

SupgBlockSolver::~SupgBlockSolver()
{
    for (int i=0; i<4; i++)
    {
        KSPDestroy(&kspblock[i]);
    }
    
    VecDestroy(&b);
    VecDestroy(&bhat); 
    VecDestroy(&diag);
    VecDestroy(&rhs);
    VecDestroy(&tmp);
    MatDestroy(&AReFull);

    delete X;
    delete Y;
}

//------(simplified) petsc pcshell preconditioenr------
class MyBlockSolver : public Solver
{
private:
   Mat **sub; 

   // Create internal KSP objects to handle the subproblems
   KSP kspblock[3];

   // Create PetscParVectors as placeholders X and Y
   mutable PetscParVector *X, *Y;

   IS index_set[3];

public:
   MyBlockSolver(const OperatorHandle &oh);

   virtual void SetOperator (const Operator &op)
   { MFEM_ABORT("MyBlockSolver::SetOperator is not supported.");}

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual ~MyBlockSolver();
};

MyBlockSolver::MyBlockSolver(const OperatorHandle &oh) : Solver() { 
   PetscErrorCode ierr; 

   // Get the PetscParMatrix out of oh.       
   PetscParMatrix *PP;
   oh.Get(PP);
   Mat P = *PP; // type cast to Petsc Mat

   //X = new PetscParVector(P, true, false); 
   //Y = new PetscParVector(P, false, false);
   
   // update base (Solver) class
   width = PP->Width();
   height = PP->Height();
   X = new PetscParVector(PETSC_COMM_WORLD, *this, true, false);
   Y = new PetscParVector(PETSC_COMM_WORLD, *this, false, false);

   PetscInt M, N;
   ierr=MatNestGetSubMats(P,&N,&M,&sub); PCHKERRQ(sub[0][0], ierr);// sub is an N by M array of matrices
   ierr=MatNestGetISs(P, index_set, NULL);  PCHKERRQ(index_set, ierr);// get the index sets of the blocks

   /*
   ISView(index_set[0],PETSC_VIEWER_STDOUT_WORLD);
   ISView(index_set[1],PETSC_VIEWER_STDOUT_WORLD);
   ISView(index_set[2],PETSC_VIEWER_STDOUT_WORLD);
   */

   for (int i=0; i<3; i++)
   {
     ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[i]);    PCHKERRQ(kspblock[i], ierr);
     ierr=KSPSetOperators(kspblock[i], sub[i][i], sub[i][i]);PCHKERRQ(sub[i][i], ierr);

     if (i==0) 
         KSPAppendOptionsPrefix(kspblock[i],"s1_");
     else
         KSPAppendOptionsPrefix(kspblock[i],"s2_");
     KSPSetFromOptions(kspblock[i]);
     KSPSetUp(kspblock[i]);
   }
}

void MyBlockSolver::Mult(const Vector &x, Vector &y) const
{
   Vec blockx, blocky;
   Vec blockx0, blocky0;

   X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc
   Y->PlaceArray(y.GetData());

   //solve the last two equations first
   for (int i = 1; i<3; i++)
   {
     VecGetSubVector(*X,index_set[i],&blockx);
     VecGetSubVector(*Y,index_set[i],&blocky);

     KSPSolve(kspblock[i],blockx,blocky);

     if (i==2)
     {
        VecGetSubVector(*X,index_set[0],&blockx0);
        VecGetSubVector(*Y,index_set[0],&blocky0);
        VecScale(blockx0, -1);
        MatMultAdd(sub[0][2], blocky, blockx0, blockx0);
        VecScale(blockx0, -1);
     }

     VecRestoreSubVector(*X,index_set[i],&blockx);
     VecRestoreSubVector(*Y,index_set[i],&blocky);
   }
   
   //compute blockx
   KSPSolve(kspblock[0],blockx0,blocky0);
   VecRestoreSubVector(*X,index_set[0],&blockx0);
   VecRestoreSubVector(*Y,index_set[0],&blocky0);

   X->ResetArray();
   Y->ResetArray();
}

MyBlockSolver::~MyBlockSolver()
{
    for (int i=0; i<3; i++)
    {
        KSPDestroy(&kspblock[i]);
        //ISDestroy(&index_set[i]); no need to delete it
    }
    
    delete X;
    delete Y;
}


