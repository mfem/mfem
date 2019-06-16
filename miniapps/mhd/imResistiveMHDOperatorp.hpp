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

// reduced system (it will only own Jacobian and Mdt)
class ReducedSystemOperator : public Operator
{
private:
   ParFiniteElementSpace &fespace;
   ParBilinearForm *M, *K, *KB, *DRe, *DSl;
   HypreParMatrix &Mmat, &Kmat;
   HypreParMatrix *Mdtpr;
   bool initialMdt;
   HypreParVector *E0Vec;
   ParGridFunction *j0;
   Array<int> block_trueOffsets;

   CGSolver *M_solver;

   double dt, dtOld;
   const Vector *phi, *psi, *w;
   const Array<int> &ess_tdof_list;

   mutable ParGridFunction phiGf, psiGf;
   mutable ParBilinearForm *Nv, *Nb;
   mutable BlockOperator *Jacobian;
   //mutable HypreParMatrix Mtmp;
   //mutable Vector z, zFull, Z, J;
   mutable Vector z, zFull;

public:
   ReducedSystemOperator(ParFiniteElementSpace &f,
                         ParBilinearForm *M_, HypreParMatrix &Mmat_,
                         ParBilinearForm *K_, HypreParMatrix &Kmat_,
                         ParBilinearForm *KB_, ParBilinearForm *DRe_, ParBilinearForm *DSl_,
                         CGSolver *M_solver_, const Array<int> &ess_tdof_list_);

   /// Set current values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *phi_, const Vector *psi_, const Vector *w_)
   {   dtOld=dt; dt=dt_; phi=phi_; psi=psi_; w=w_;
       if (dtOld!=dt && initialMdt)
       {
           cout <<"------update Mdt------"<<endl;
           double rate=dtOld/dt;
           *Mdtpr*=rate;
       }
       if(initialMdt==false)
       {
           cout <<"------initial Mdt-------"<<endl;
           *Mdtpr*=(1./dt); initialMdt=true;
       }
   }

   void setCurrent(ParGridFunction *gf)
   { j0=gf;}

   void setE0(HypreParVector *E0Vec_)
   { E0Vec=E0Vec_;}

   //link gf with psi
   void BindingGF(Vector &k){
       int sc = height/3;
       phiGf.MakeTRef(&fespace, k, 0);
       psiGf.MakeTRef(&fespace, k, sc);
   }

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
   HypreParMatrix Kmat, Mmat;
   HypreParVector *E0Vec;
   double viscosity, resistivity;
   double jBdy;
   bool useAMG;

   //for implicit stepping
   ReducedSystemOperator *reduced_oper;
   PetscNonlinearSolver *pnewton_solver;
   PetscPreconditionerFactory *J_factory;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver K_solver; // Krylov solver for inverting the stiffness matrix K
   HypreSmoother K_prec;  // Preconditioner for the stiffness matrix K

   HypreSolver *K_amg; //BoomerAMG for stiffness matrix
   HyprePCG *K_pcg;

   mutable Vector z, zFull; // auxiliary vector 
   mutable ParGridFunction j, gf;  //auxiliary variable (to store the boundary condition)

public:
   ResistiveMHDOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi, bool use_petsc); 

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   //Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
   //here vector are block vectors
   virtual void ImplicitSolve(const double dt, const Vector &vx, Vector &k);

   //link gf with psi
   void BindingGF(Vector &vx)
   {int sc = height/3; gf.MakeTRef(&fespace, vx, sc);}

   //set rhs E0 
   void SetRHSEfield(FunctionCoefficient Efield);
   void SetInitialJ(FunctionCoefficient initJ);
   void SetJBdy(double jBdy_) 
   {jBdy = jBdy_;}

   void UpdatePhi(Vector &vx);
   void assembleNv(ParGridFunction *gf);
   void assembleNb(ParGridFunction *gf);

   void DestroyHypre();
   virtual ~ResistiveMHDOperator();
};



//------petsc pcshell preconditioenr------
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
   PetscInt M, N;
   ierr=MatNestGetSubMats(P,&N,&M,&sub); PCHKERRQ(sub[0][0], ierr);// sub is an N by M array of matrices
   ierr=MatNestGetISs(P, index_set, NULL);  PCHKERRQ(index_set, ierr);// get the index sets of the blocks

   /*
   ISView(index_set[0],PETSC_VIEWER_STDOUT_WORLD);
   ISView(index_set[1],PETSC_VIEWER_STDOUT_WORLD);
   ISView(index_set[2],PETSC_VIEWER_STDOUT_WORLD);
   */

   X = new PetscParVector(P, true, false); 
   Y = new PetscParVector(P, false, false);

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

//should I also do iterative_mode here?
void MyBlockSolver::Mult(const Vector &x, Vector &y) const
{
   //Mat &mass = sub[0][2];
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

// Auxiliary class to provide preconditioners for matrix-free methods 
class PreconditionerFactory : public PetscPreconditionerFactory
{
private:
   const ReducedSystemOperator& op;

public:
   PreconditionerFactory(const ReducedSystemOperator& op_,
                         const string& name_): PetscPreconditionerFactory(name_), op(op_) {};
   virtual mfem::Solver* NewPreconditioner(const mfem::OperatorHandle &oh)
   { return new MyBlockSolver(oh);}

   virtual ~PreconditionerFactory() {};
};


ResistiveMHDOperator::ResistiveMHDOperator(ParFiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc, double resi, 
                                         bool use_petsc = false)
   : TimeDependentOperator(3*f.TrueVSize(), 0.0), fespace(f),
     M(NULL), K(NULL), KB(NULL), DSl(&fespace), DRe(&fespace),
     Nv(NULL), Nb(NULL), E0(NULL), Sw(NULL), E0Vec(NULL),
     viscosity(visc),  resistivity(resi), useAMG(false), 
     reduced_oper(NULL), pnewton_solver(NULL), J_factory(NULL),
     M_solver(f.GetComm()), K_solver(f.GetComm()), 
     K_amg(NULL), K_pcg(NULL), z(height/3), zFull(f.GetVSize()), j(&fespace) 
{
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //mass matrix
   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = true;
   M_solver.SetRelTol(1e-12);
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

   useAMG=true;
   if (useAMG && !use_petsc)
   {
      K_amg = new HypreBoomerAMG(Kmat);
      K_pcg = new HyprePCG(Kmat);
      K_pcg->iterative_mode = false;
      K_pcg->SetTol(1e-7);
      K_pcg->SetMaxIter(200);
      K_pcg->SetPrintLevel(0);
      K_pcg->SetPreconditioner(*K_amg);
   }
   else
   {
      K_solver.iterative_mode = true;
      K_solver.SetRelTol(1e-7);
      K_solver.SetAbsTol(0.0);
      K_solver.SetMaxIter(2000);
      K_solver.SetPrintLevel(3);
      //K_prec.SetType(HypreSmoother::GS);
      K_prec.SetType(HypreSmoother::Chebyshev); //this is faster
      K_solver.SetPreconditioner(K_prec);
      K_solver.SetOperator(Kmat);
   }

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
      ParBilinearForm *DRepr, *DSlpr;
      if (viscosity != 0.0)
          DRepr = &DRe;
      else
          DRepr = NULL;

      if (resistivity != 0.0)
          DSlpr = &DSl;
      else
          DSlpr = NULL;


      reduced_oper  = new ReducedSystemOperator(f, M, Mmat, K, Kmat,
                         KB, DRepr, DSlpr, &M_solver, ess_tdof_list);

      const double rel_tol=1.e-8;
      pnewton_solver = new PetscNonlinearSolver(f.GetComm(),*reduced_oper);
      if (false)
      {
         J_factory = new PreconditionerFactory(*reduced_oper, "JFNK preconditioner");
         pnewton_solver->SetPreconditionerFactory(J_factory);
      }
      pnewton_solver->SetPrintLevel(1); // print Newton iterations
      pnewton_solver->SetRelTol(rel_tol);
      pnewton_solver->SetAbsTol(0.0);
      pnewton_solver->SetMaxIter(10);
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

void ResistiveMHDOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors and time derivative
   int sc = height/3;
   dvx_dt=0.0;

   Vector phi(vx.GetData() +   0, sc);
   Vector psi(vx.GetData() +  sc, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   Vector dphi_dt(dvx_dt.GetData() +   0, sc);
   Vector dpsi_dt(dvx_dt.GetData() +  sc, sc);
   Vector   dw_dt(dvx_dt.GetData() +2*sc, sc);

   //for (int i=0; i<ess_tdof_list.Size(); i++)
   //    cout<<j(ess_tdof_list[i])<<" "<<ess_tdof_list[i]<<" "; //set homogeneous Dirichlet condition by hand
   //cout<<endl<<"j size is "<<j.Size()<<endl;

   /*
   //compute the current as an auxilary variable
   KB->Mult(psi, z);
   z.Neg(); // z = -z
   //z.SetSubVector(ess_tdof_list,jBdy);
   //Vector J(sc);
   //M_solver.Mult(z, J);

   HypreParMatrix tmp;
   Vector Y, Z;
   M->FormLinearSystem(ess_tdof_list, j, z, tmp, Y, Z); 
   M_solver.Mult(Z, Y);
   M->RecoverFEMSolution(Y, z, j);
   */


   /*
   cout << "Size of matrix in KB: " <<  KB->Size()<< endl;
   cout << "Size of matrix in Nv: " <<  Nv->Size()<< endl;
   cout << "Size of matrix in DSl: " <<  DSl.Size()<< endl;
   */

   //compute the current as an auxilary variable
   gf.SetFromTrueVector();  //recover psi

   Vector J, Z;
   HypreParMatrix A;
   KB->Mult(gf, zFull);
   zFull.Neg(); // z = -z
   M->FormLinearSystem(ess_tdof_list, j, zFull, A, J, Z); //apply Dirichelt boundary 
   M_solver.Mult(Z, J);

   //evolve the dofs
   z=0.;
   Nv->TrueAddMult(psi, z);
   if (resistivity != 0.0)
   {
      DSl.TrueAddMult(psi, z);
   }
   if (E0Vec!=NULL)
     z += *E0Vec;
   z.Neg(); // z = -z
   z.SetSubVector(ess_tdof_list,0.0);
   M_solver.Mult(z, dpsi_dt);

   z=0.;
   Nv->TrueAddMult(w, z);
   if (viscosity != 0.0)
   {
      DRe.TrueAddMult(w, z);
   }
   z.Neg(); // z = -z
   Nb->TrueAddMult(J, z); 
   z.SetSubVector(ess_tdof_list,0.0);
   M_solver.Mult(z, dw_dt);

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
   //reduced_oper->BindingGF(k); use type cast in Mult instead
   pnewton_solver->Mult(zero, k);  //here k is solved as vx^{n+1}
   MFEM_VERIFY(pnewton_solver->GetConverged(),
                  "Newton solver did not converge.");

   //modify k so that it fits into the backward euler framework
   k-=vx;
   k/=dt;
}


void ResistiveMHDOperator::assembleNv(ParGridFunction *gf) 
{
   delete Nv;
   Nv = new ParBilinearForm(&fespace);
   MyCoefficient velocity(gf, 2);   //we update velocity

   Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
   Nv->Assemble(); 
}

void ResistiveMHDOperator::assembleNb(ParGridFunction *gf) 
{
   delete Nb;
   Nb = new ParBilinearForm(&fespace);
   MyCoefficient Bfield(gf, 2);   //we update B

   Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
   Nb->Assemble();
}

void ResistiveMHDOperator::UpdatePhi(Vector &vx)
{
   //Phi=-K^{-1}*M*w
   int sc = height/3;
   Vector phi(vx.GetData() +   0, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   Mmat.Mult(w, z);
   z.Neg(); // z = -z
   z.SetSubVector(ess_tdof_list,0.0);

   if (useAMG)
      K_pcg->Mult(z,phi);
   else 
      K_solver.Mult(z, phi);
}

void ResistiveMHDOperator::DestroyHypre()
{
    //hypre and petsc needs to be deleted earilier
    delete K_amg;
    delete pnewton_solver;
    delete reduced_oper;
    delete J_factory;
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
}

ReducedSystemOperator::ReducedSystemOperator(ParFiniteElementSpace &f,
   ParBilinearForm *M_, HypreParMatrix &Mmat_,
   ParBilinearForm *K_, HypreParMatrix &Kmat_,
   ParBilinearForm *KB_, ParBilinearForm *DRe_, ParBilinearForm *DSl_,
   CGSolver *M_solver_,
   const Array<int> &ess_tdof_list_)
   : Operator(3*f.TrueVSize()), fespace(f), 
     M(M_), K(K_), KB(KB_), DRe(DRe_), DSl(DSl_), Mmat(Mmat_), Kmat(Kmat_), 
     initialMdt(false),
     E0Vec(NULL), M_solver(M_solver_),
     dt(0.0), dtOld(0.0), 
     phi(NULL), psi(NULL), w(NULL), ess_tdof_list(ess_tdof_list_),
     Nv(NULL), Nb(NULL), Jacobian(NULL), z(height/3), zFull(f.GetVSize())
{ 
    
    //this is not right because Mdtpr shares the same matrix with Mmat_
    //hypre_ParCSRMatrix *csrM = (hypre_ParCSRMatrix*)(Mmat_);
    //Mdtpr = new HypreParMatrix(csrM, true);
    
    //XXX this is the right way to deep copy
    Mdtpr = new HypreParMatrix(Mmat_);

    int sc = height/3;
    block_trueOffsets.SetSize(4);
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = sc;
    block_trueOffsets[2] = 2*sc;
    block_trueOffsets[3] = 3*sc;
}

Operator &ReducedSystemOperator::GetGradient(const Vector &k) const
{
   if (Jacobian == NULL)    //in the first pass we just set Jacobian once
   {
      MFEM_ASSERT(initialMdt, "Mdt not initialized correctly!"); 
      Jacobian = new BlockOperator(block_trueOffsets);
      Jacobian->SetBlock(0,0,&Kmat);
      Jacobian->SetBlock(0,2,&Mmat);
      Jacobian->SetBlock(1,1,Mdtpr);
      Jacobian->SetBlock(2,2,Mdtpr);
   }
   return *Jacobian;
}

ReducedSystemOperator::~ReducedSystemOperator()
{
   delete Mdtpr;
   delete Jacobian;
   delete Nv;
   delete Nb;
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

   //------assemble Nv and Nb (operators are assembled locally)------
   delete Nv;
   phiGf.MakeTRef(&fespace, k_, 0);
   phiGf.SetFromTrueVector();
   Nv = new ParBilinearForm(&fespace);
   MyCoefficient velocity(&phiGf, 2);   //we update velocity
   Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
   Nv->Assemble(); 

   delete Nb;
   psiGf.MakeTRef(&fespace, k_, sc);
   psiGf.SetFromTrueVector();
   Nb = new ParBilinearForm(&fespace);
   MyCoefficient Bfield(&psiGf, 2);   //we update B
   Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
   Nb->Assemble();

   //------compute the current as an auxilary variable------
   Vector J, Z;
   HypreParMatrix A;
   KB->Mult(psiGf, zFull);
   zFull.Neg(); // z = -z
   M->FormLinearSystem(ess_tdof_list, *j0, zFull, A, J, Z); //apply Dirichelt boundary 
   M_solver->Mult(Z, J); //XXX is this okay in mult? probably

   //compute y1
   Kmat.Mult(phiNew,y1);
   Mmat.Mult(wNew,z);
   y1+=z;
   y1.SetSubVector(ess_tdof_list, 0.0);

   //compute y2
   //note z=psiNew-*psi
   z=psiNew;
   z-=*psi;
   z/=dt;
   Mmat.Mult(z,y2);
   Nv->TrueAddMult(psiNew,y2);
   if (DSl!=NULL)
       DSl->TrueAddMult(psiNew,y2);
   if (E0Vec!=NULL)
       y2 += *E0Vec;
   y2.SetSubVector(ess_tdof_list, 0.0);

   //compute y3
   //note z=wNew-*w
   z=wNew;
   z-=*w;
   z/=dt;
   Mmat.Mult(z,y3);
   Nv->TrueAddMult(wNew,y3);
   if (DRe!=NULL)
       DRe->TrueAddMult(wNew,y3);

   //note J=-M^{-1} KB*Psi; so let J=-J
   J.Neg();
   Nb->TrueAddMult(J, y3); 
   y3.SetSubVector(ess_tdof_list, 0.0);
}

