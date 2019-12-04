#include "mfem.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

/** After spatial discretization, the resistive MHD model can be written as a
 *  system of ODEs:
 *     dPsi/dt = M^{-1}*F1,
 *     dw  /dt = M^{-1}*F2,
 *  coupled with two linear systems
 *     j   = -M^{-1}*(K-B)*Psi 
 *     Phi = -K^{-1}*M*w
 *  so far there seems no need to do a BlockNonlinearForm
 *
 *  Class AMRResistiveMHDOperator represents the right-hand side of the above
 *  system of ODEs. */
class AMRResistiveMHDOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list;

   ParBilinearForm *M, *Mrhs, *K, *KB, *DSl, *DRe; //mass, stiffness, diffusion with SL and Re
   ParBilinearForm *Nv, *Nb;
   ParLinearForm *E0; //source terms
   HypreParMatrix Mmat, Kmat, *MrhsMat;
   ConstantCoefficient visc_coeff, resi_coeff;
   double viscosity, resistivity;
   FunctionCoefficient *E0rhs;
   bool useAMG;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   HypreSmoother *M_prec;  // Preconditioner for the mass matrix M

   CGSolver M_solver2; // Krylov solver for inverting the mass matrix Mrhs
   HypreSmoother *M_prec2;  // Preconditioner for the mass matrix M

   CGSolver K_solver; // Krylov solver for inverting the stiffness matrix K
   HypreSmoother *K_prec;  // Preconditioner for the stiffness matrix K

   HypreSolver *K_amg; // Krylov solver for inverting the stiffness matrix K
   HyprePCG *K_pcg;  // Preconditioner for the stiffness matrix K

   mutable Vector z; // auxiliary vector 

public:
   AMRResistiveMHDOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi);

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   //set rhs E0
   void SetRHSEfield( double(* f)( const Vector&) );

   void UpdateJ(Vector &vx, ParGridFunction *j);
   void UpdatePhi(Vector &vx);
   void BackSolvePsi(Vector &vx);
   void assembleNv(ParGridFunction *gf);
   void assembleNb(ParGridFunction *gf);
   void assembleProblem(Array<int> &ess_bdr);
   void UpdateProblem();

   void DestroyHypre();
   virtual ~AMRResistiveMHDOperator();
};

AMRResistiveMHDOperator::AMRResistiveMHDOperator(ParFiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc, double resi)
   : TimeDependentOperator(4*f.GetVSize(), 0.0), fespace(f), 
     M(NULL), Mrhs(NULL), K(NULL), KB(NULL), DSl(NULL), DRe(NULL), Nv(NULL), Nb(NULL), E0(NULL), MrhsMat(NULL),
     visc_coeff(visc), resi_coeff(resi),
     viscosity(visc),  resistivity(resi), useAMG(false),
     M_solver(f.GetComm()), M_prec(NULL), M_solver2(f.GetComm()), M_prec2(NULL), 
     K_solver(f.GetComm()), K_prec(NULL),
     E0rhs(NULL), K_amg(NULL), K_pcg(NULL)
{
   //mass matrix
   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);

   Mrhs = new ParBilinearForm(&fespace);
   Mrhs->AddDomainIntegrator(new MassIntegrator);

   //stiffness matrix
   K = new ParBilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);

   KB = new ParBilinearForm(&fespace);
   KB->AddDomainIntegrator(new DiffusionIntegrator);      //  K matrix
   KB->AddBdrFaceIntegrator(new BoundaryGradIntegrator);  // -B matrix

   //resi_coeff and visc_coeff have to be stored for assembling for some reason
   DRe = new ParBilinearForm(&fespace);
   DRe->AddDomainIntegrator(new DiffusionIntegrator(visc_coeff));    

   DSl = new ParBilinearForm(&fespace);
   DSl->AddDomainIntegrator(new DiffusionIntegrator(resi_coeff));    
}

void AMRResistiveMHDOperator::assembleProblem(Array<int> &ess_bdr)
{
   //update ess_tdof_list
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //update mass matrix
   M->Assemble();
   Mrhs->Assemble();
   Mrhs->Finalize();

   //update stiffness matrix
   K->Assemble();

   //update solvers
   M->FormSystemMatrix(ess_tdof_list, Mmat);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(200);
   M_solver.SetPrintLevel(0);
   delete M_prec;
   M_prec = new HypreSmoother;
   M_prec->SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(Mmat);

   delete MrhsMat;
   MrhsMat=Mrhs->ParallelAssemble();
   M_solver2.iterative_mode = false;
   M_solver2.SetRelTol(1e-8); 
   M_solver2.SetAbsTol(0.0);
   M_solver2.SetMaxIter(200);
   M_solver2.SetPrintLevel(0);
   delete M_prec2;
   M_prec2 = new HypreSmoother;
   M_prec2->SetType(HypreSmoother::Jacobi);
   M_solver2.SetPreconditioner(*M_prec2);
   M_solver2.SetOperator(*MrhsMat);

   K->FormSystemMatrix(ess_tdof_list, Kmat);
   useAMG=true;
   if(useAMG)
   {
      delete K_amg;
      K_amg = new HypreBoomerAMG(Kmat);
      delete K_pcg;
      K_pcg = new HyprePCG(Kmat);
      K_pcg->iterative_mode = false;
      K_pcg->SetTol(1e-7);
      K_pcg->SetMaxIter(200);
      K_pcg->SetPrintLevel(0);
      K_pcg->SetPreconditioner(*K_amg); 
   }
   else
   {
      K_solver.iterative_mode = false;
      K_solver.SetRelTol(1e-7);
      K_solver.SetAbsTol(0.0);
      K_solver.SetMaxIter(1000);
      K_solver.SetPrintLevel(0);
      delete K_prec;
      K_prec = new HypreSmoother;
      K_prec->SetType(HypreSmoother::Chebyshev);
      K_solver.SetPreconditioner(*K_prec);
      K_solver.SetOperator(Kmat);
   }

   //assemble KB
   KB->Assemble();

   //update DRe and DSl 
   if (viscosity != 0.0)
   {   
      DRe->Assemble();
   }

   if (resistivity != 0.0)
   {
      DSl->Assemble();
   }

   if (E0!=NULL)
      E0->Assemble();

}

void AMRResistiveMHDOperator::UpdateProblem()
{
   M->Update();
   Mrhs->Update();
   K->Update();
   KB->Update();

   // tell DRe and DSl that space is change
   if (viscosity != 0.0)
   {   
      DRe->Update();
   }

   if (resistivity != 0.0)
   {
      DSl->Update();
   }   

   if (E0!=NULL)
   {
      E0->Update();
   }

   //this probably should be done in main loop
   width = height = fespace.GetVSize()*4;
}          

void AMRResistiveMHDOperator::SetRHSEfield( double(* f)( const Vector&) ) 
{
   delete E0;
   delete E0rhs;
   E0rhs = new FunctionCoefficient(f);
   E0 = new ParLinearForm(&fespace);
   E0->AddDomainIntegrator(new DomainLFIntegrator(*E0rhs));
}          


void AMRResistiveMHDOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors and time derivative
   int sc = height/4;
   Vector phi(vx.GetData() +   0, sc);
   Vector psi(vx.GetData() +  sc, sc);
   Vector   w(vx.GetData() +2*sc, sc);
   Vector   j(vx.GetData() +3*sc, sc);

   dvx_dt=0.0;
   z.SetSize(sc);

   Vector dpsi_dt(dvx_dt.GetData() +  sc, sc);
   Vector   dw_dt(dvx_dt.GetData() +2*sc, sc);

   Nv->Mult(psi, z);
   if (resistivity != 0.0)
   {
      DSl->AddMult(psi, z);
   }
   if (E0!=NULL)
     z += *E0;
   z.Neg(); // z = -z

   HypreParMatrix A;
   Vector B, X;
   M->FormLinearSystem(ess_tdof_list, dpsi_dt, z, A, X, B); // Alters matrix and rhs to enforce bc
   M_solver.Mult(B, X);
   M->RecoverFEMSolution(X, z, dpsi_dt);

   Nv->Mult(w, z);
   if (viscosity != 0.0)
   {
      DRe->AddMult(w, z);
   }
   z.Neg(); // z = -z
   Nb->AddMult(j, z);

   M->FormLinearSystem(ess_tdof_list, dw_dt, z, A, X, B); // Alters matrix and rhs to enforce bc
   M_solver.Mult(B, X);
   M->RecoverFEMSolution(X, z, dw_dt);

}

void AMRResistiveMHDOperator::assembleNv(ParGridFunction *gf) 
{
   delete Nv;
   Nv = new ParBilinearForm(&fespace);
   MyCoefficient velocity(gf, 2);   //update velocity

   Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
   Nv->Assemble(); 
}

void AMRResistiveMHDOperator::assembleNb(ParGridFunction *gf) 
{
   delete Nb;
   Nb = new ParBilinearForm(&fespace);
   MyCoefficient Bfield(gf, 2);   //update B

   Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
   Nb->Assemble();
}

void AMRResistiveMHDOperator::UpdateJ(Vector &vx, ParGridFunction *j)
{
   //the current is J=-M^{-1}*K*Psi
   int sc = height/4;
   Vector psi(vx.GetData() +  sc, sc);

   ParLinearForm zLF(&fespace);
   KB->Mult(psi, zLF);
   zLF.Neg(); // z = -z

   int trueSize=fespace.GetTrueVSize();

   //no boundary condition is applied here
   Vector zTmp(trueSize), jTmp(trueSize);

   zLF.ParallelAssemble(zTmp);
   M_solver2.Mult(zTmp, jTmp);
   j->SetFromTrueDofs(jTmp);
   
   //this is the wrong way
   /*
   Vector   j(vx.GetData() +3*sc, sc);  //it creates a reference
   z.SetSize(sc);
   KB->Mult(psi, z);
   z.Neg(); // z = -z

   HypreParMatrix A;
   Vector Y, Z;

   //apply Dirichelt boundary 
   //(j is initially from a projection with initial condition, so it satisfies the boundary conditino all the time)
   M->FormLinearSystem(ess_tdof_list, j, z, A, Y, Z);
   M_solver.Mult(Z, Y);
   M->RecoverFEMSolution(Y, z, j);
   */

}

void AMRResistiveMHDOperator::BackSolvePsi(Vector &vx)
{
   int sc = height/4;
   Vector psi(vx.GetData() +  sc, sc);
   Vector   j(vx.GetData() +3*sc, sc);

   cout <<"===Back Solve Psi: this is not tested yet!!==="<<endl;
   z.SetSize(sc);

   Mrhs->Mult(j, z);
   z.Neg(); // z = -z

   HypreParMatrix A;
   Vector B, X;
   K->FormLinearSystem(ess_tdof_list, psi, z, A, X, B); // Alters matrix and rhs to enforce bc
   M_solver.Mult(B, X);
   K->RecoverFEMSolution(X, z, psi);
}

void AMRResistiveMHDOperator::UpdatePhi(Vector &vx)
{
   //Phi=-K^{-1}*M*w
   int sc = height/4;
   Vector phi(vx.GetData() +   0, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   Mrhs->Mult(w, z);
   z.Neg(); // z = -z

   HypreParMatrix A;
   Vector B, X;
   K->FormLinearSystem(ess_tdof_list, phi, z, A, X, B); // Alters matrix and rhs to enforce bc
   if(useAMG)
      K_pcg->Mult(B, X);
   else
      K_solver.Mult(B, X);
   K->RecoverFEMSolution(X, z, phi);
}

void AMRResistiveMHDOperator::DestroyHypre()
{
    //hypre needs to be deleted earilier
    delete K_amg;
}

AMRResistiveMHDOperator::~AMRResistiveMHDOperator()
{
    //free used memory
    delete M;
    delete Mrhs;
    delete MrhsMat;
    delete K;
    delete KB;
    delete Nv;
    delete Nb;
    delete DRe;
    delete DSl;
    delete E0;
    delete E0rhs;
    delete M_prec;
    delete M_prec2;
    delete K_prec;
    delete K_pcg;
}


