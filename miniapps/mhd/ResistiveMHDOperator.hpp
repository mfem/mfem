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
 *  Class ResistiveMHDOperator represents the right-hand side of the above
 *  system of ODEs. */
class ResistiveMHDOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list;

   BilinearForm *M, *K, *KB, DSl, DRe; //mass, stiffness, diffusion with SL and Re
   BilinearForm *Nv, *Nb;
   LinearForm *E0, *Sw, *PB_VPsi, *PB_VOmega, *PB_BJ; //two source terms and Poisson Bracket
   SparseMatrix Mmat, Kmat;

   double viscosity, resistivity;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   GSSmoother *M_prec;  // Preconditioner for the mass matrix M

   CGSolver K_solver; // Krylov solver for inverting the stiffness matrix K
   GSSmoother *K_prec;  // Preconditioner for the stiffness matrix K

   mutable Vector z; // auxiliary vector 

public:
   ResistiveMHDOperator(FiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi);   //this is old
   ResistiveMHDOperator(FiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi, int icase);

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   //set rhs E0
   void SetRHSEfield(FunctionCoefficient Efield);

   void UpdateJ(Vector &vx);
   void UpdatePhi(Vector &vx);
   void assembleNv(GridFunction *gf);
   void assembleNb(GridFunction *gf);

   void SetVPsi(GridFunction *phi, GridFunction *psi);
   void SetVOmega(GridFunction *phi, GridFunction *omega);
   void SetBJ(GridFunction *psi, GridFunction *j);

   virtual ~ResistiveMHDOperator();
};

ResistiveMHDOperator::ResistiveMHDOperator(FiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc, double resi)
   : TimeDependentOperator(4*f.GetVSize(), 0.0), fespace(f),
     M(NULL), K(NULL), KB(NULL), DSl(&fespace), DRe(&fespace), 
     Nv(NULL), Nb(NULL), E0(NULL), Sw(NULL), 
     PB_VPsi(NULL), PB_VOmega(NULL),PB_BJ(NULL),
     viscosity(visc),  resistivity(resi), M_prec(NULL), K_prec(NULL), z(height/4)
{
   const double rel_tol = 1e-10;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   if (false)
   {
        Array<int> ess_vdof;
        fespace.GetEssentialVDofs(ess_bdr, ess_vdof);
        ofstream myfile0 ("vdof.dat"), myfile3("tdof.dat");
        ess_tdof_list.Print(myfile3, 1000);
        ess_vdof.Print(myfile0, 1000);
   }

   //mass matrix
   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   GSSmoother *M_prec_gs = new GSSmoother(Mmat);
   M_prec=M_prec_gs;

   M_solver.iterative_mode = true;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(1000);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(Mmat);  //this is probably not owned by M_solver

   //stiffness matrix
   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   GSSmoother *K_prec_gs = new GSSmoother(Kmat);
   K_prec=K_prec_gs;

   K_solver.iterative_mode = true;
   K_solver.SetRelTol(rel_tol);
   K_solver.SetAbsTol(0.0);
   K_solver.SetMaxIter(2000);
   K_solver.SetPrintLevel(0);
   K_solver.SetPreconditioner(*K_prec);
   K_solver.SetOperator(Kmat);

   KB = new BilinearForm(&fespace);
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

   ConstantCoefficient visc_coeff(viscosity);
   DRe.AddDomainIntegrator(new DiffusionIntegrator(visc_coeff));    
   DRe.Assemble();

   ConstantCoefficient resi_coeff(resistivity);
   DSl.AddDomainIntegrator(new DiffusionIntegrator(resi_coeff));    
   DSl.Assemble();

   /*
   cout << "Number of total scalar unknowns: " << fespace.GetVSize()<< endl;
   cout << "Number of GetTrueVSize unknowns: " << fespace.GetTrueVSize() << endl;
   cout << "Number of matrix in M: " <<  Mmat.Size()<< endl;
   cout << "Number of matrix in K: " <<  Kmat.Size()<< endl;
   cout << "Number of matrix in KB: " << KB->SpMat().Size()<< endl;
   cout << "Number of matrix in DSl: " << DSl.SpMat().Size()<< endl;
   cout << "Number of matrix in DRe: " << DRe.SpMat().Size()<< endl;
   */

}

void ResistiveMHDOperator::SetRHSEfield(FunctionCoefficient Efield) 
{
   delete E0;
   E0 = new LinearForm(&fespace);
   E0->AddDomainIntegrator(new DomainLFIntegrator(Efield));
   E0->Assemble();
}

void ResistiveMHDOperator::SetVPsi(GridFunction *phi, GridFunction *psi)
{
   delete PB_VPsi;
   PB_VPsi = new LinearForm(&fespace);
   PBCoefficient pbCoeff(phi, psi);

   //intOrder = 3*k+0
   PB_VPsi->AddDomainIntegrator(new DomainLFIntegrator(pbCoeff, 3, 0));
   PB_VPsi->Assemble();
}

void ResistiveMHDOperator::SetVOmega(GridFunction *phi, GridFunction *omega)
{
   delete PB_VOmega;
   PB_VOmega = new LinearForm(&fespace);
   PBCoefficient pbCoeff(phi, omega);

   PB_VOmega->AddDomainIntegrator(new DomainLFIntegrator(pbCoeff, 3, 0));
   PB_VOmega->Assemble();
}

void ResistiveMHDOperator::SetBJ(GridFunction *psi, GridFunction *j)
{
   delete PB_BJ;
   PB_BJ = new LinearForm(&fespace);
   PBCoefficient pbCoeff(psi, j);

   PB_BJ->AddDomainIntegrator(new DomainLFIntegrator(pbCoeff, 3, 0));
   PB_BJ->Assemble();
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

   if (PB_VPsi!=NULL)
   {
       //cout <<"VPsi ";
       z=*PB_VPsi;
   }
   else
       Nv->Mult(psi, z);

   if (resistivity != 0.0)
   {
      DSl.AddMult(psi, z);
   }
   if (E0!=NULL)
     z += *E0;
   z.Neg(); // z = -z

   if (true)
   {
      //for (int i=0; i<ess_tdof_list.Size(); i++)
      //    z(ess_tdof_list[i])=0.0; //set homogeneous Dirichlet condition by hand

      //ofstream myfile("z0.dat");
      //z.Print(myfile, 10);
      //cout<<z.Size()<<endl;
      //

      z.SetSubVector(ess_tdof_list, 0.0);
      M_solver.Mult(z, dpsi_dt);
   }
   else
   {
       //another way; but it is slower
       SparseMatrix A;
       Vector B, X;
       M->FormLinearSystem(ess_tdof_list, dpsi_dt, z, A, X, B); // Alters matrix and rhs to enforce bc
       PCG(Mmat, *M_prec, B, X, 0, 1000, 1e-12, 0.0); 
       //CG(A, B, X);
       M->RecoverFEMSolution(X, z, dpsi_dt);
   }
   //ofstream myfile("zLHS1.dat");
   //z.Print(myfile, 1000);

   if (PB_VOmega!=NULL)
   {
       //cout <<"VOmgea ";
       z=*PB_VOmega;
   }
   else
       Nv->Mult(w, z);

   if (viscosity != 0.0)
   {
      DRe.AddMult(w, z);
   }
   z.Neg(); // z = -z
   if (PB_BJ!=NULL)
   {
      //cout <<"BJ ";
      z+=*PB_BJ;
   }
   else
      Nb->AddMult(j, z);

   //for (int i=0; i<ess_tdof_list.Size(); i++)
   //    z(ess_tdof_list[i])=0.0; //set Dirichlet condition by hand
   //ofstream myfile2("zLHS2.dat");
   //z.Print(myfile2, 1000);

   z.SetSubVector(ess_tdof_list, 0.0);
   M_solver.Mult(z, dw_dt);

}

void ResistiveMHDOperator::assembleNv(GridFunction *gf) 
{
   //M_solver.Mult(*gf, z);
   //Vector phi(vx.GetData() +   0, sc);
   //cout <<phi(0)<<endl;   //debug
   //GridFunction phiGF(&fespace); 
   //phiGF.SetFromTrueDofs(phi);
   

   delete Nv;
   Nv = new BilinearForm(&fespace);
   MyCoefficient velocity(gf, 2);   //we update velocity

   Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
   Nv->Assemble(); 
}

void ResistiveMHDOperator::assembleNb(GridFunction *gf) 
{
   //Vector psi(vx.GetData() +  sc, sc);
   //GridFunction psiGF(&fespace); 
   //psiGF.SetFromTrueDofs(psi);


   delete Nb;
   Nb = new BilinearForm(&fespace);
   MyCoefficient Bfield(gf, 2);   //we update B

   Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
   Nb->Assemble();
}

void ResistiveMHDOperator::UpdateJ(Vector &vx)
{
   //the current is J=-M^{-1}*K*Psi
   int sc = height/4;
   Vector psi(vx.GetData() +  sc, sc);
   Vector   j(vx.GetData() +3*sc, sc);  //it creates a reference
   SparseMatrix tmp;
   Vector Y, Z;

   KB->Mult(psi, z);
   z.Neg(); // z = -z
   M->FormLinearSystem(ess_tdof_list, j, z, tmp, Y, Z); //apply Dirichelt boundary (j is initially from a projection with initial condition, so it satisfies the boundary conditino all the time)
   M_solver.Mult(Z, Y);
   M->RecoverFEMSolution(Y, z, j);

   //cout <<"======Update J======"<<endl;

   /* debugging for the boundary terms
   if (false){
       for (int i=0; i<ess_tdof_list.Size(); i++)
       { 
         cout <<ess_tdof_list[i]<<" ";
         z(ess_tdof_list[i])=0.0;
       }
       ofstream myfile("zv.dat");
       z.Print(myfile, 1000);
   }
   M_solver.Mult(z, j);
   */

}



void ResistiveMHDOperator::UpdatePhi(Vector &vx)
{
   //Phi=-K^{-1}*M*w
   int sc = height/4;
   Vector phi(vx.GetData() +   0, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   M->Mult(w, z);
   z.Neg(); // z = -z
   K_solver.Mult(z, phi);
}


ResistiveMHDOperator::~ResistiveMHDOperator()
{
    //free used memory
    delete M;
    delete K;
    delete E0;
    delete Sw;
    delete PB_VPsi;
    delete PB_VOmega;
    delete PB_BJ;
    delete KB;
    delete Nv;
    delete Nb;
    delete M_prec;
    delete K_prec;
}


