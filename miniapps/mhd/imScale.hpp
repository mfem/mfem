#include "mfem.hpp"
#include <iostream>
#include <fstream>

//copied from petsc.cpp
#include "petsc.h"
#if defined(PETSC_HAVE_HYPRE)
#include "petscmathypre.h"
#endif

using namespace std;
using namespace mfem;

// reduced system 
class ReducedSystemOperator : public Operator
{
private:
   ParFiniteElementSpace &fespace;
   ParBilinearForm *M, *K, *KB, *DRe, *DSl; 
   HypreParMatrix &Mmat, &Kmat, *DRematpr, *DSlmatpr;
   //own by this:
   HypreParMatrix *Mdtpr, *ARe, *ASl;
   mutable HypreParMatrix *ScFull, *AReFull, *NbFull, *PwMat;
   bool initialMdt, useFull;
   HypreParVector *E0Vec;
   ParGridFunction *j0;
   Array<int> block_trueOffsets;

   CGSolver *M_solver;

   int myid;
   double dt, dtOld;
   const Vector *phi, *psi, *w;
   const Array<int> &ess_tdof_list;
   const Array<int> &ess_bdr;

   mutable ParGridFunction phiGf, psiGf, wGf;
   mutable ParBilinearForm *Nv, *Nb, *Pw;
   mutable BlockOperator *Jacobian;
   mutable Vector z, zFull;

public:
   ReducedSystemOperator(ParFiniteElementSpace &f,
                         ParBilinearForm *M_, HypreParMatrix &Mmat_,
                         ParBilinearForm *K_, HypreParMatrix &Kmat_,
                         ParBilinearForm *KB_, 
                         ParBilinearForm *DRe_, HypreParMatrix *DRemat_,
                         ParBilinearForm *DSl_, HypreParMatrix *DSlmat_,
                         CGSolver *M_solver_, const Array<int> &ess_tdof_list_,
                         const Array<int> &ess_bdr_, bool useFull_);

   /// Set current values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *phi_, const Vector *psi_, const Vector *w_)
   {   
       dtOld=dt; dt=dt_; phi=phi_; psi=psi_; w=w_;
       if (dtOld!=dt && initialMdt)
       {
           if (myid==0) 
              cout <<"------update Mdt------"<<endl;
           double rate=dtOld/dt;
           *Mdtpr*=rate;

           if (useFull)
           {
               delete ARe;
               delete ASl;
               if (DRematpr!=NULL)
                  ARe = ParAdd(Mdtpr, DRematpr);
               else
                  ARe = new HypreParMatrix(*Mdtpr);
                  
               if (DSlmatpr!=NULL)
                  ASl = ParAdd(Mdtpr, DSlmatpr);
               else
                  ASl = new HypreParMatrix(*Mdtpr);
           }
       }
       if(initialMdt==false)
       {
           if (myid==0) 
              cout <<"------initial Mdt-------"<<endl;
           *Mdtpr*=(1./dt); 
           initialMdt=true;

           if (useFull)
           {
              if (DRematpr!=NULL)
                 ARe = ParAdd(Mdtpr, DRematpr);
              else
                 ARe = new HypreParMatrix(*Mdtpr);
                 
              if (DSlmatpr!=NULL)
                 ASl = ParAdd(Mdtpr, DSlmatpr);
              else
                 ASl = new HypreParMatrix(*Mdtpr);
           }
       }
   }

   void setCurrent(ParGridFunction *gf)
   { j0=gf;}

   void setE0(HypreParVector *E0Vec_)
   { E0Vec=E0Vec_;}

   /// Define F(k) 
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Define J 
   virtual Operator &GetGradient(const Vector &k) const;

   virtual ~ReducedSystemOperator();

};

/** After spatial discretization, the resistive MHD model can be written as a
 *  system of ODEs:
 *     dPsi/dt = M^{-1}*F1,
 *     dw  /dt = M^{-1}*F2,
 *  coupled with two linear systems
 *     j   = -M^{-1}*(K-B)*Psi 
 *     Phi = -K^{-1}*M*w
 *  so far there seems no need to do a BlockNonlinearForm
 *
 *  Class ResistiveMHDOperator represents the right-hand side of the above
 *  system of ODEs. */
class ResistiveMHDOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list;

   ParBilinearForm *M, *K, *KB, DSl, DRe; //mass, stiffness, diffusion with SL and Re
   ParBilinearForm *Nv, *Nb;
   ParLinearForm *E0, *Sw; //two source terms
   HypreParMatrix Kmat, Mmat, DSlmat, DRemat;
   HypreParVector *E0Vec;
   double viscosity, resistivity;

   //for implicit stepping
   ReducedSystemOperator *reduced_oper;
   PetscNonlinearSolver *pnewton_solver;
   PetscPreconditionerFactory *J_factory;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec;  // Preconditioner for the mass matrix M

   HypreSolver *K_amg; //BoomerAMG for stiffness matrix
   HyprePCG *K_pcg;

   mutable Vector z, zFull; // auxiliary vector 
   mutable ParGridFunction j;  //auxiliary variable (to store the boundary condition)
   ParBilinearForm *DRetmp, *DSltmp;    //hold the matrices for DRe and DSl

public:
   ResistiveMHDOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi, bool use_petsc, bool use_factory); 

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const
   { MFEM_ABORT("ResistiveMHDSolver::Mult is not defined.");}

   //Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
   //here vector are block vectors
   virtual void ImplicitSolve(const double dt, const Vector &vx, Vector &k);

   //set rhs E0 
   void SetRHSEfield(FunctionCoefficient Efield);
   void SetInitialJ(FunctionCoefficient initJ);
   void DestroyHypre();
   virtual ~ResistiveMHDOperator();
};

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
   bool version2;

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
   ierr=MatNestGetSubMats(P,&N,&M,&sub); // sub is an N by M array of matrices
   ierr=MatNestGetISs(P, index_set, NULL); // get the index sets of the blocks

   //stiffness
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[0]); 
   ierr=KSPSetOperators(kspblock[0], sub[2][0], sub[2][0]);
   KSPAppendOptionsPrefix(kspblock[0],"s1_");
   KSPSetFromOptions(kspblock[0]);
   KSPSetUp(kspblock[0]);

   //schur complement
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[1]);    
   ierr=KSPSetOperators(kspblock[1], sub[1][1], sub[1][1]);
   KSPAppendOptionsPrefix(kspblock[1],"s2_");
   KSPSetFromOptions(kspblock[1]);
   KSPSetUp(kspblock[1]);
         
   //mass matrix
   ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[2]);    
   ierr=KSPSetOperators(kspblock[2], sub[2][2], sub[2][2]);
   KSPAppendOptionsPrefix(kspblock[2],"s3_");
   KSPSetFromOptions(kspblock[2]);
   KSPSetUp(kspblock[2]);

   Mat ARe = sub[0][0];
   version2 = true;
   if (version2)
   {
      //ARe matrix (for version 2)
      ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[3]);    
      ierr=KSPSetOperators(kspblock[3], ARe, ARe);
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
   MatGetDiagonal(ARe,diag);
}

//Mult will only be called once
void FullBlockSolver::Mult(const Vector &x, Vector &y) const
{
   PetscInt iter=4; //number of Jacobi iteration 

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
   for (int j = 0; j<iter; j++)
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
   if(version2)
   {
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
   else
   {
      //version 1
      MatMult(Kmat, y0, rhs);
      VecAYPX(rhs, -1., b0);
      KSPSolve(kspblock[2],rhs,y2);
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
    if(version2)
        KSPDestroy(&kspblock[3]);
    
    VecDestroy(&b);
    VecDestroy(&bhat); 
    VecDestroy(&diag);
    VecDestroy(&rhs);
    VecDestroy(&tmp);

    delete X;
    delete Y;
}

// Auxiliary class to provide preconditioners for matrix-free methods 
class FullPreconditionerFactory : public PetscPreconditionerFactory
{
private:
   const ReducedSystemOperator& op;

public:
   FullPreconditionerFactory(const ReducedSystemOperator& op_,
                         const string& name_): PetscPreconditionerFactory(name_), op(op_) {};
   virtual mfem::Solver* NewPreconditioner(const mfem::OperatorHandle &oh)
   { return new FullBlockSolver(oh);}

   virtual ~FullPreconditionerFactory() {};
};



ResistiveMHDOperator::ResistiveMHDOperator(ParFiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc, double resi, 
                                         bool use_petsc = false, bool use_factory=false)
   : TimeDependentOperator(3*f.TrueVSize(), 0.0), fespace(f),
     M(NULL), K(NULL), KB(NULL), DSl(&fespace), DRe(&fespace),
     Nv(NULL), Nb(NULL), E0(NULL), Sw(NULL), E0Vec(NULL),
     viscosity(visc),  resistivity(resi), 
     reduced_oper(NULL), pnewton_solver(NULL), J_factory(NULL),
     M_solver(f.GetComm()), 
     K_amg(NULL), K_pcg(NULL), z(height/3), zFull(f.GetVSize()), j(&fespace),
     DRetmp(NULL), DSltmp(NULL)
{
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //mass matrix
   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = true;
   M_solver.SetRelTol(1e-7);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(2000);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   //stiffness matrix
   K = new ParBilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   /*
   K_amg = new HypreBoomerAMG(Kmat);
   K_pcg = new HyprePCG(Kmat);
   K_pcg->iterative_mode = false;
   K_pcg->SetTol(1e-7);
   K_pcg->SetMaxIter(200);
   K_pcg->SetPrintLevel(0);
   K_pcg->SetPreconditioner(*K_amg);
   */

   KB = new ParBilinearForm(&fespace);
   KB->AddDomainIntegrator(new DiffusionIntegrator);      //  K matrix
   KB->AddBdrFaceIntegrator(new BoundaryGradIntegrator);  // -B matrix
   KB->Assemble();

   ConstantCoefficient visc_coeff(viscosity);
   DRe.AddDomainIntegrator(new DiffusionIntegrator(visc_coeff));    
   DRe.Assemble();

   ConstantCoefficient resi_coeff(resistivity);
   DSl.AddDomainIntegrator(new DiffusionIntegrator(resi_coeff));    
   DSl.Assemble();
  
   if (use_petsc)
   {
      ParBilinearForm *DRepr=NULL, *DSlpr=NULL;
      HypreParMatrix *DRematpr=NULL, *DSlmatpr=NULL;
      if (viscosity != 0.0)
      {   
          //assemble diffusion matrices (cannot delete DRetmp if ParAdd is used later)
          DRetmp = new ParBilinearForm(&fespace);
          DRetmp->AddDomainIntegrator(new DiffusionIntegrator(visc_coeff));    
          DRetmp->Assemble();
          DRetmp->FormSystemMatrix(ess_tdof_list, DRemat);

          DRematpr = &DRemat;
          DRepr = &DRe;
      }

      if (resistivity != 0.0)
      {
          DSltmp = new ParBilinearForm(&fespace);
          DSltmp->AddDomainIntegrator(new DiffusionIntegrator(resi_coeff));    
          DSltmp->Assemble();
          DSltmp->FormSystemMatrix(ess_tdof_list, DSlmat);

          DSlmatpr = &DSlmat;
          DSlpr = &DSl;
      }

      bool useFull = true;
      reduced_oper  = new ReducedSystemOperator(f, M, Mmat, K, Kmat,
                         KB, DRepr, DRematpr, DSlpr, DSlmatpr, &M_solver, 
                         ess_tdof_list, ess_bdr, useFull);

      const double rel_tol=1.e-8;
      pnewton_solver = new PetscNonlinearSolver(f.GetComm(),*reduced_oper);
      if (use_factory)
      {
         if (useFull)
            J_factory = new FullPreconditionerFactory(*reduced_oper, "JFNK preconditioner");
         else
            MFEM_ABORT("Error! use Full preconditioner");
         pnewton_solver->SetPreconditionerFactory(J_factory);
      }
      pnewton_solver->SetPrintLevel(0); // print Newton iterations
      pnewton_solver->SetRelTol(rel_tol);
      pnewton_solver->SetAbsTol(0.0);
      pnewton_solver->SetMaxIter(20);
      pnewton_solver->iterative_mode=true;
   }
}

void ResistiveMHDOperator::SetRHSEfield(FunctionCoefficient Efield) 
{
   delete E0;
   E0 = new ParLinearForm(&fespace);
   E0->AddDomainIntegrator(new DomainLFIntegrator(Efield));
   E0->Assemble();
   E0Vec=E0->ParallelAssemble();

   //add E0 to reduced_oper
   if (reduced_oper!=NULL)
      reduced_oper->setE0(E0Vec);
}

void ResistiveMHDOperator::SetInitialJ(FunctionCoefficient initJ) 
{
    j.ProjectCoefficient(initJ);
    j.SetTrueVector();

    //add current to reduced_oper
    if (reduced_oper!=NULL)
        reduced_oper->setCurrent(&j);
}

void ResistiveMHDOperator::ImplicitSolve(const double dt,
                                         const Vector &vx, Vector &k)
{
   int sc = height/3;
   Vector phi(vx.GetData() +   0, sc);
   Vector psi(vx.GetData() +  sc, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   reduced_oper->SetParameters(dt, &phi, &psi, &w);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   
   k = vx; //Provide the initial guess as vx and use iterative_mode
   pnewton_solver->Mult(zero, k);  //here k is solved as vx^{n+1}
   MFEM_VERIFY(pnewton_solver->GetConverged(),
                  "Newton solver did not converge.");
   //modify k so that it fits into the backward euler framework
   k-=vx;
   k/=dt;
}


void ResistiveMHDOperator::DestroyHypre()
{
    //hypre and petsc needs to be deleted earilier
    delete K_amg;
    delete reduced_oper;
    delete J_factory;
    delete pnewton_solver;
}


ResistiveMHDOperator::~ResistiveMHDOperator()
{
    //free used memory
    delete M;
    delete K;
    delete E0;
    delete E0Vec;
    delete Sw;
    delete KB;
    delete Nv;
    delete Nb;
    delete K_pcg;
    delete DRetmp;
    delete DSltmp;
}

ReducedSystemOperator::ReducedSystemOperator(ParFiniteElementSpace &f,
   ParBilinearForm *M_, HypreParMatrix &Mmat_,
   ParBilinearForm *K_, HypreParMatrix &Kmat_,
   ParBilinearForm *KB_, 
   ParBilinearForm *DRe_, HypreParMatrix *DRemat_,
   ParBilinearForm *DSl_, HypreParMatrix *DSlmat_,
   CGSolver *M_solver_,const Array<int> &ess_tdof_list_,
   const Array<int> &ess_bdr_, bool useFull_)
   : Operator(3*f.TrueVSize()), fespace(f), 
     M(M_), K(K_), KB(KB_), DRe(DRe_), DSl(DSl_), Mmat(Mmat_), Kmat(Kmat_), 
     initialMdt(false),
     E0Vec(NULL), M_solver(M_solver_), 
     dt(0.0), dtOld(0.0), phi(NULL), psi(NULL), w(NULL), 
     ess_tdof_list(ess_tdof_list_), ess_bdr(ess_bdr_),
     Nv(NULL), Nb(NULL), Pw(NULL), 
     Jacobian(NULL), z(height/3), zFull(f.GetVSize())
{ 
    useFull = useFull_;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    Mdtpr = new HypreParMatrix(Mmat_);
    ARe=NULL; ASl=NULL;

    DRematpr = DRemat_;
    DSlmatpr = DSlmat_;

    AReFull=NULL; ScFull=NULL; NbFull=NULL; PwMat=NULL;

    int sc = height/3;
    block_trueOffsets.SetSize(4);
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = sc;
    block_trueOffsets[2] = 2*sc;
    block_trueOffsets[3] = 3*sc;
}

/*
 * the full preconditioner is (note the sign of Nb)
 * [  ARe Nb  0 ]
 * [  Pw  Sc  0 ]
 * [  K   0   M ]
*/

Operator &ReducedSystemOperator::GetGradient(const Vector &k) const
{
   MFEM_ASSERT(initialMdt, "Mdt not initialized correctly!"); 

   if (useFull)
   {
       delete Jacobian;
       delete AReFull; 
       delete ScFull; 
       delete NbFull;
       delete PwMat;

       Vector &k_ = const_cast<Vector &>(k);

       int sc = height/3;
       //form Nv matrix
       delete Nv;
       phiGf.MakeTRef(&fespace, k_, 0);
       phiGf.SetFromTrueVector();
       Nv = new ParBilinearForm(&fespace);
       MyCoefficient velocity(&phiGf, 2);   //we update velocity
       Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
       Nv->Assemble(); 
       Nv->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
       Nv->Finalize();
       HypreParMatrix *NvMat = Nv->ParallelAssemble();

       //change AReFull to the true ARe operator and ScFull to the true ASl operator
       AReFull = ParAdd(ARe, NvMat);
       HypreParMatrix *ASltmp = ParAdd(ASl, NvMat);    

       //form Nb matrix
       delete Nb;
       psiGf.MakeTRef(&fespace, k_, sc);
       psiGf.SetFromTrueVector();
       Nb = new ParBilinearForm(&fespace);
       MyCoefficient Bfield(&psiGf, 2);   //we update B
       Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
       Nb->Assemble();
       Nb->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
       Nb->Finalize();
       NbFull = Nb->ParallelAssemble();

       //form Pw operator        
       delete Pw;
       wGf.MakeTRef(&fespace, k_, 2*sc);
       wGf.SetFromTrueVector();
       Pw = new ParBilinearForm(&fespace);
       MyCoefficient curlw(&wGf, 2);
       Pw->AddDomainIntegrator(new ConvectionIntegrator(curlw));
       Pw->Assemble();
       Pw->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
       Pw->Finalize();
       PwMat = Pw->ParallelAssemble();

       //here we use B^T D^-1 B = (D^-1 B)^T B
       HypreParMatrix *DinvNb = new HypreParMatrix(*NbFull);
       HypreParVector *ARed = new HypreParVector(AReFull->GetComm(), AReFull->GetGlobalNumRows(),
                                        AReFull->GetRowStarts());
       HypreParMatrix *NbtDinv=NULL, *S=NULL;

       //VERSION0: same as Luis's preconditioner
       AReFull->GetDiag(*ARed);
       DinvNb->InvScaleRows(*ARed);
       NbtDinv=DinvNb->Transpose();
       S = ParMult(NbtDinv, NbFull);
       ScFull = ParAdd(ASltmp, S);

       delete DinvNb;
       delete ARed;
       delete NbtDinv;
       delete S;
       delete NvMat;
       delete ASltmp;

       Jacobian = new BlockOperator(block_trueOffsets);
       Jacobian->SetBlock(0,0,AReFull);
       Jacobian->SetBlock(0,1,NbFull);
       Jacobian->SetBlock(1,0,PwMat);
       Jacobian->SetBlock(1,1,ScFull);
       Jacobian->SetBlock(2,0,&Kmat);
       Jacobian->SetBlock(2,2,&Mmat);
   }
   else
   {
       MFEM_ABORT("Error! use Full preconditioner");
   }

   return *Jacobian;
}

ReducedSystemOperator::~ReducedSystemOperator()
{
   delete Mdtpr;
   delete ARe;
   delete ASl;
   delete AReFull;
   delete ScFull;
   delete NbFull;
   delete PwMat;
   delete Jacobian;
   delete Nv;
   delete Nb;
   delete Pw;
}

void ReducedSystemOperator::Mult(const Vector &k, Vector &y) const
{
   int sc = height/3;

   Vector phiNew(k.GetData() +   0, sc);
   Vector psiNew(k.GetData() +  sc, sc);
   Vector   wNew(k.GetData() +2*sc, sc);

   Vector y1(y.GetData() +   0, sc);
   Vector y2(y.GetData() +  sc, sc);
   Vector y3(y.GetData() +2*sc, sc);

   Vector &k_ = const_cast<Vector &>(k);
   phiGf.MakeTRef(&fespace, k_, 0);
   phiGf.SetFromTrueVector();
   psiGf.MakeTRef(&fespace, k_, sc);
   psiGf.SetFromTrueVector();

   //------assemble Nv and Nb (operators are assembled locally)------
   delete Nv;
   Nv = new ParBilinearForm(&fespace);
   MyCoefficient velocity(&phiGf, 2);   //we update velocity
   Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
   Nv->Assemble(); 

   //------compute the current as an auxilary variable------
   Vector J, Z;
   HypreParMatrix A;
   KB->Mult(psiGf, zFull);
   zFull.Neg(); // z = -z
   M->FormLinearSystem(ess_tdof_list, *j0, zFull, A, J, Z); //apply Dirichelt boundary 
   M_solver->Mult(Z, J); 

   //+++++compute y1
   Kmat.Mult(phiNew,y1);
   Mmat.Mult(wNew,z);
   y1+=z;
   y1.SetSubVector(ess_tdof_list, 0.0);

   //+++++compute y2
   add(psiNew, -1., *psi, z);
   z/=dt;
   Mmat.Mult(z,y2);
   Nv->TrueAddMult(psiNew,y2);

   if (DSl!=NULL)
       DSl->TrueAddMult(psiNew,y2);
   if (E0Vec!=NULL)
       y2 += *E0Vec;
   y2.SetSubVector(ess_tdof_list, 0.0);

   //+++++compute y3
   add(wNew, -1., *w, z);
   z/=dt;
   Mmat.Mult(z,y3);
   Nv->TrueAddMult(wNew,y3);

   if (DRe!=NULL)
       DRe->TrueAddMult(wNew,y3);

   delete Nb;
   Nb = new ParBilinearForm(&fespace);
   MyCoefficient Bfield(&psiGf, 2);   //we update B
   Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
   Nb->Assemble();
   Nb->TrueAddMult(J, y3, -1.); 
   
   y3.SetSubVector(ess_tdof_list, 0.0);
}
