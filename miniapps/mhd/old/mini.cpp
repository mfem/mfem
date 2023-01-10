// example to demonstrate pcshell
// Description:  it solves a time dependent resistive MHD problem 

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "BoundaryGradIntegrator.hpp"
#include <memory>
#include <iostream>
#include <fstream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

#include "petsc.h"
#if defined(PETSC_HAVE_HYPRE)
#include "petscmathypre.h"
#endif

using namespace std;
using namespace mfem;

//initial condition
double InitialZero(const Vector &x)
{return 0.0;}

double InitialJ(const Vector &x)
{ return -M_PI*M_PI*(1.0+4.0/9.0)*.001*sin(M_PI*x(1))*cos(2.0*M_PI/3.0*x(0)); }

double InitialPsi(const Vector &x)
{ return -x(1)+.001*sin(M_PI*x(1))*cos(2.0*M_PI/3.0*x(0)); }

class ReducedSystemOperator : public Operator
{
private:
   ParFiniteElementSpace &fespace;
   ParBilinearForm *M, *K, *KB;
   HypreParMatrix &Mmat, &Kmat;
   HypreParMatrix *Mdtpr;
   bool initialMdt;
   ParGridFunction *j0;
   Array<int> block_trueOffsets;
   CGSolver *M_solver;

   double dt, dtOld;
   const Vector *phi, *psi, *w;
   const Array<int> &ess_tdof_list;

   mutable ParGridFunction phiGf, psiGf;
   mutable ParBilinearForm *Nv, *Nb;
   mutable BlockOperator *Jacobian;
   mutable Vector z, zFull;

public:
   ReducedSystemOperator(ParFiniteElementSpace &f,
                         ParBilinearForm *M_, HypreParMatrix &Mmat_,
                         ParBilinearForm *K_, HypreParMatrix &Kmat_,
                         ParBilinearForm *KB_, 
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
   
   /// Define F(k) 
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Define J 
   virtual Operator &GetGradient(const Vector &k) const;

   virtual ~ReducedSystemOperator();

};

class ResistiveMHDOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list;

   ParBilinearForm *M, *K, *KB; 
   ParBilinearForm *Nv, *Nb;
   HypreParMatrix Kmat, Mmat;

   //for implicit stepping
   ReducedSystemOperator *reduced_oper;
   PetscNonlinearSolver *pnewton_solver;
   PetscPreconditionerFactory *J_factory;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec;  // Preconditioner for the mass matrix M

   mutable ParGridFunction j;  //auxiliary variable (to store the boundary condition)

public:
   ResistiveMHDOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr, 
                       bool use_petsc, bool use_factory); 

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const
   {MFEM_ABORT("Not needed.");}

   //Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
   //here vector are block vectors
   virtual void ImplicitSolve(const double dt, const Vector &vx, Vector &k);
   void SetInitialJ(FunctionCoefficient initJ);

   void DestroyPetsc();
   virtual ~ResistiveMHDOperator();
};

//------petsc pcshell preconditioenr------
class MyBlockSolver : public Solver
{
private:
   Mat **sub; 
   KSP kspblock[3];
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

   // Get the PetscParMatrix out of oh.       
   PetscParMatrix *PP;
   oh.Get(PP);
   Mat P = *PP; // type cast to Petsc Mat
   PetscInt M, N;
   MatNestGetSubMats(P,&N,&M,&sub); // sub is an N by M array of matrices
   MatNestGetISs(P, index_set, NULL);  // get the index sets of the blocks

   // update base (Solver) class
   width = PP->Width();
   height = PP->Height();

   // There's some strange bug when creating vectors out of a MATNEST
   // I need to track this down. Anyway, since the index sets are always
   // contiguous when converting from an MFEM BlockOperator, we
   // get the same optimization by constructing the needed vectors matching
   // the shape of *this.
   // VecGetSubVector will return a subvector that will actually use the
   // same memory of the parent vector
   //X = new PetscParVector(P, true, false);
   //Y = new PetscParVector(P, false, false);
   X = new PetscParVector(PETSC_COMM_WORLD, *this, true, false);
   Y = new PetscParVector(PETSC_COMM_WORLD, *this, false, false);

   for (int i=0; i<3; i++)
   {
     KSPCreate(PETSC_COMM_WORLD, &kspblock[i]);    
     KSPSetOperators(kspblock[i], sub[i][i], sub[i][i]);

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

ResistiveMHDOperator::ResistiveMHDOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr, 
                                         bool use_petsc = false, bool use_factory=false)
   : TimeDependentOperator(3*f.TrueVSize(), 0.0), fespace(f),
     M(NULL), K(NULL), KB(NULL), Nv(NULL), Nb(NULL),       
     reduced_oper(NULL), pnewton_solver(NULL), J_factory(NULL),
     M_solver(f.GetComm()), j(&fespace) 
{
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

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

   K = new ParBilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   KB = new ParBilinearForm(&fespace);
   KB->AddDomainIntegrator(new DiffusionIntegrator);      //  K matrix
   KB->AddBdrFaceIntegrator(new BoundaryGradIntegrator);  // -B matrix
   KB->Assemble();

   {
      reduced_oper  = new ReducedSystemOperator(f, M, Mmat, K, Kmat,
                         KB, &M_solver, ess_tdof_list);

      const double rel_tol=1.e-8;
      pnewton_solver = new PetscNonlinearSolver(f.GetComm(),*reduced_oper);
      if (use_factory)
      {
         //cout <<"use pcshell"<<endl;
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

void ResistiveMHDOperator::SetInitialJ(FunctionCoefficient initJ) 
{
    j.ProjectCoefficient(initJ);
    j.SetTrueVector();
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

void ResistiveMHDOperator::DestroyPetsc()
{
    delete reduced_oper;
    delete J_factory;
    delete pnewton_solver;
}

ResistiveMHDOperator::~ResistiveMHDOperator()
{
    delete M;
    delete K;
    delete KB;
    delete Nv;
    delete Nb;
}

ReducedSystemOperator::ReducedSystemOperator(ParFiniteElementSpace &f,
   ParBilinearForm *M_, HypreParMatrix &Mmat_,
   ParBilinearForm *K_, HypreParMatrix &Kmat_,
   ParBilinearForm *KB_, CGSolver *M_solver_, const Array<int> &ess_tdof_list_)
   : Operator(3*f.TrueVSize()), fespace(f), 
     M(M_), K(K_), KB(KB_), Mmat(Mmat_), Kmat(Kmat_), 
     initialMdt(false), M_solver(M_solver_),
     dt(0.0), dtOld(0.0), 
     phi(NULL), psi(NULL), w(NULL), ess_tdof_list(ess_tdof_list_),
     Nv(NULL), Nb(NULL), Jacobian(NULL), z(height/3), zFull(f.GetVSize())
{ 
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
   M_solver->Mult(Z, J); 

   //compute y1
   Kmat.Mult(phiNew,y1);
   Mmat.Mult(wNew,z);
   y1+=z;
   y1.SetSubVector(ess_tdof_list, 0.0);

   //compute y2
   z=psiNew;
   z-=*psi;
   z/=dt;
   Mmat.Mult(z,y2);
   Nv->TrueAddMult(psiNew,y2);
   y2.SetSubVector(ess_tdof_list, 0.0);

   //compute y3
   z=wNew;
   z-=*w;
   z/=dt;
   Mmat.Mult(z,y3);
   Nv->TrueAddMult(wNew,y3);

   J.Neg();
   Nb->TrueAddMult(J, y3); 
   y3.SetSubVector(ess_tdof_list, 0.0);
}

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   const char *mesh_file = "./xperiodic-square.mesh";
   int ser_ref_levels = 2;
   int order = 2;
   double t_final = 1.0;
   double dt = 0.0001;
   bool use_factory = false;
   const char *petscrc_file = "";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&use_factory, "-shell", "--shell", "-no-shell",
                  "--no-shell",
                  "Use user-defined preconditioner factory (PCSHELL).");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   
   if (myid == 0) args.PrintOptions(cout);

   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   ODESolver *ode_solver2=new BackwardEulerSolver; 

   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll); 

   HYPRE_Int global_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of total scalar unknowns: " << global_size << endl;
   }

   int fe_size = fespace.TrueVSize();
   Array<int> fe_offset(4);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;

   BlockVector vx(fe_offset);
   ParGridFunction psi, phi, w;
   phi.MakeTRef(&fespace, vx, fe_offset[0]);
   psi.MakeTRef(&fespace, vx, fe_offset[1]);
     w.MakeTRef(&fespace, vx, fe_offset[2]);

   //Set the initial conditions, and the boundary conditions
   FunctionCoefficient phiInit(InitialZero);
   phi.ProjectCoefficient(phiInit);
   phi.SetTrueVector();
   phi.SetFromTrueVector(); 

   FunctionCoefficient psiInit(InitialPsi);
   psi.ProjectCoefficient(psiInit);
   psi.SetTrueVector();
   psi.SetFromTrueVector(); 

   FunctionCoefficient wInit(InitialZero);
   w.ProjectCoefficient(wInit);
   w.SetTrueVector();
   w.SetFromTrueVector();
   
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;  

   ResistiveMHDOperator oper(fespace, ess_bdr, true, use_factory);

   //set initial J
   FunctionCoefficient jInit(InitialJ);
   oper.SetInitialJ(jInit);

   double t = 0.0;
   oper.SetTime(t);
   ode_solver2->Init(oper);

   MPI_Barrier(MPI_COMM_WORLD); 
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);
      ode_solver2->Step(vx, t, dt_real);
      last_step = (t >= t_final - 1e-8*dt);
   }

   MPI_Barrier(MPI_COMM_WORLD); 

   // 10. Free the used memory.
   delete ode_solver2;
   delete pmesh;

   oper.DestroyPetsc();
   MFEMFinalizePetsc(); 
   MPI_Finalize();

   return 0;
}
