#include "mfem.hpp"
#include "InitialConditions.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

extern int icase;

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
   ParBilinearForm *Mrhs;
   ParBilinearForm *Nv, *Nb;
   ParLinearForm *E0, *Sw; //two source terms
   mutable ParLinearForm zLF;
   HypreParMatrix Kmat, Mmat, *MrhsMat, *NbMat;
   double viscosity, resistivity;
   bool useAMG;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver M_solver2; // Krylov solver for inverting the mass matrix M without essential boundary
   HypreSmoother M_prec2;  // Preconditioner for the mass matrix M

   CGSolver K_solver; // Krylov solver for inverting the stiffness matrix K
   HypreSmoother K_prec;  // Preconditioner for the stiffness matrix K

   HypreSolver *K_amg; //BoomerAMG for stiffness matrix
   HyprePCG *K_pcg;

   mutable Vector z, zTrue; // auxiliary vector 

public:
   ResistiveMHDOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi); 

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   //set rhs E0
   void SetRHSEfield(FunctionCoefficient Efield);

   void UpdateJ(Vector &vx, ParGridFunction *j);
   void UpdatePhi(Vector &vx);
   void assembleNv(ParGridFunction *gf);
   void assembleNb(ParGridFunction *gf);

   void DestroyHypre();
   virtual ~ResistiveMHDOperator();
};

ResistiveMHDOperator::ResistiveMHDOperator(ParFiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc, double resi)
   : TimeDependentOperator(4*f.GetVSize(), 0.0), fespace(f),
     M(NULL), K(NULL), KB(NULL), DSl(&fespace), DRe(&fespace), Mrhs(NULL),
     Nv(NULL), Nb(NULL), E0(NULL), Sw(NULL), zLF(&fespace),
     viscosity(visc),  resistivity(resi), MrhsMat(NULL), NbMat(NULL), useAMG(false), 
     M_solver(f.GetComm()), M_solver2(f.GetComm()), K_solver(f.GetComm()), 
     K_amg(NULL), K_pcg(NULL), z(height/4), zTrue(f.GetTrueVSize())
{
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //mass matrix
   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   Mrhs = new ParBilinearForm(&fespace);
   Mrhs->AddDomainIntegrator(new MassIntegrator);
   Mrhs->Assemble();
   Mrhs->Finalize();
   MrhsMat=Mrhs->ParallelAssemble();

   M_solver.iterative_mode = true;
   M_solver.SetRelTol(1e-7);    //solving too tight can be problematic
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(2000);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   M_solver2.iterative_mode = false;
   M_solver2.SetRelTol(1e-7); 
   M_solver2.SetAbsTol(0.0);
   M_solver2.SetMaxIter(2000);
   M_solver2.SetPrintLevel(0);
   M_prec2.SetType(HypreSmoother::Jacobi);
   M_solver2.SetPreconditioner(M_prec2);
   M_solver2.SetOperator(*MrhsMat);

   //stiffness matrix
   K = new ParBilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   useAMG=true;
   if (useAMG)
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

   if (false)
   {
        cout << Kmat.Height()<<" "<<Kmat.Width()<<endl;
        cout << Mmat.Height()<<" "<<Mmat.Width()<<endl;

        ofstream myfile ("Kmat.m");
        Kmat.PrintMatlab(myfile);

        ofstream myfile2 ("Mmat.m");
   }

   Coefficient *visc_coeff, *resi_coeff;
   if (icase==5)
   {
      visc_coeff= new FunctionCoefficient(resiVari);
   }
   else
      visc_coeff= new ConstantCoefficient(viscosity);
   DRe.AddDomainIntegrator(new DiffusionIntegrator(*visc_coeff));    
   DRe.Assemble();

   if (icase==5)
      resi_coeff= new FunctionCoefficient(resiVari);
   else
      resi_coeff= new ConstantCoefficient(resistivity);
   DSl.AddDomainIntegrator(new DiffusionIntegrator(*resi_coeff));    
   DSl.Assemble();

   delete visc_coeff;
   delete resi_coeff;
}

void ResistiveMHDOperator::SetRHSEfield(FunctionCoefficient Efield) 
{
   delete E0;
   E0 = new ParLinearForm(&fespace);
   E0->AddDomainIntegrator(new DomainLFIntegrator(Efield));
   E0->Assemble();
}

void ResistiveMHDOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors and time derivative
   int sc = height/4;
   Vector phi(vx.GetData() +   0, sc);
   Vector psi(vx.GetData() +  sc, sc);
   Vector   w(vx.GetData() +2*sc, sc);
   Vector   j(vx.GetData() +3*sc, sc);

   dvx_dt=0.0;

   Vector dphi_dt(dvx_dt.GetData() +   0, sc);
   Vector dpsi_dt(dvx_dt.GetData() +  sc, sc);
   Vector   dw_dt(dvx_dt.GetData() +2*sc, sc);
   Vector   dj_dt(dvx_dt.GetData() +3*sc, sc);

   Nv->Mult(psi, z);
   if (resistivity != 0.0)
   {
      DSl.AddMult(psi, z);
   }
   if (E0!=NULL)
     z += *E0;
   z.Neg(); // z = -z

   //z.SetSubVector(ess_tdof_list, 0.0);
   //M_solver.Mult(z, dpsi_dt);

   HypreParMatrix A;
   Vector Y, Z;
   M->FormLinearSystem(ess_tdof_list, dpsi_dt, z, A, Y, Z); 
   M_solver.Mult(Z, Y);
   M->RecoverFEMSolution(Y, z, dpsi_dt);

   Nv->Mult(w, zLF);
   if (viscosity != 0.0)
   {
      DRe.AddMult(w, zLF);
   }
   zLF.Neg(); // z = -z
   Nb->AddMult(j, zLF);
   //Vector tmp;
   //NbMat->Mult(j, tmp);
   //z+=tmp;

   if(true) //we enforce Dirichlet on w
   {
      M->FormLinearSystem(ess_tdof_list, dw_dt, zLF, A, Y, Z); 
      M_solver.Mult(Z, Y);
      M->RecoverFEMSolution(Y, zLF, dw_dt);
   }
   else
   {   
      //here we try to force a Neumann boundary condition
      zLF.ParallelAssemble(zTrue);
      M_solver2.Mult(zTrue, Y);
      const Operator *P= fespace.GetProlongationMatrix();
      P->Mult(Y, dw_dt);
   }
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
   delete NbMat;
   Nb = new ParBilinearForm(&fespace);
   MyCoefficient Bfield(gf, 2);   //we update B

   Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
   Nb->Assemble();
   Nb->Finalize();
   NbMat=Nb->ParallelAssemble();
}

void ResistiveMHDOperator::UpdateJ(Vector &vx, ParGridFunction *j)
{
   //the current is J=-M^{-1}*K*Psi
   int sc = height/4;
   Vector psi(vx.GetData() +  sc, sc);

   KB->Mult(psi, zLF);
   zLF.Neg(); // z = -z

   //no boundary condition is applied here
   Vector jTmp(fespace.GetTrueVSize());

   zLF.ParallelAssemble(zTrue);
   M_solver2.Mult(zTrue, jTmp);
   j->SetFromTrueDofs(jTmp);

   //another equivalent way
   //const Operator &P= *fespace.GetProlongationMatrix();
   //P.MultTranspose(zGf, zTmp);

   // (this is the old way but it has a bug: J should not the essential boundary condition!)
   /* 
   KB->Mult(psi, z);
   z.Neg(); // z = -z
   HypreParMatrix tmp;
   Vector Y, Z;
   M->FormLinearSystem(ess_tdof_list, j, z, tmp, Y, Z); //apply Dirichelt boundary (j is initially from a projection with initial condition, so it satisfies the boundary conditino all the time)
   M_solver.Mult(Z, Y);
   M->RecoverFEMSolution(Y, z, j);
   */
   
}

void ResistiveMHDOperator::UpdatePhi(Vector &vx)
{
   //Phi=-K^{-1}*M*w
   int sc = height/4;
   Vector phi(vx.GetData() +   0, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   Mrhs->Mult(w, z);
   z.Neg(); // z = -z

   //z.SetSubVector(ess_tdof_list, 0.0);
   //K_solver.Mult(z, phi);

   HypreParMatrix A;
   Vector Y, Z;
   K->FormLinearSystem(ess_tdof_list, phi, z, A, Y, Z); 
   if (useAMG)
      K_pcg->Mult(Z,Y);
   else 
      K_solver.Mult(Z, Y);

   K->RecoverFEMSolution(Y, z, phi);
}

void ResistiveMHDOperator::DestroyHypre()
{
    //hypre needs to be deleted earilier
    delete K_amg;
}


ResistiveMHDOperator::~ResistiveMHDOperator()
{
    delete M;
    delete K;
    delete E0;
    delete Sw;
    delete KB;
    delete Nv;
    delete Nb;
    delete MrhsMat;
    delete NbMat;
    delete Mrhs;
    delete K_pcg;
}


