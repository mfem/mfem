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
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list;

   BilinearForm *M, *Mrhs, *K, *KB, *DSl, *DRe; //mass, stiffness, diffusion with SL and Re
   BilinearForm *Nv, *Nb;
   LinearForm *E0; //source terms
   SparseMatrix Mmat, Kmat;
   ConstantCoefficient visc_coeff, resi_coeff;
   double viscosity, resistivity;
   FunctionCoefficient *E0rhs;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   GSSmoother *M_prec;  // Preconditioner for the mass matrix M

   CGSolver K_solver; // Krylov solver for inverting the stiffness matrix K
   GSSmoother *K_prec;  // Preconditioner for the stiffness matrix K

   mutable Vector z; // auxiliary vector 

public:
   AMRResistiveMHDOperator(FiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi);
   AMRResistiveMHDOperator(FiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi, int icase);

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   //set rhs E0
   void SetRHSEfield( double(* f)( const Vector&) );

   void UpdateJ(Vector &vx);
   void UpdatePhi(Vector &vx);
   void BackSolvePsi(Vector &vx);
   void assembleNv(GridFunction *gf);
   void assembleNb(GridFunction *gf);
   void assembleProblem(Array<int> &ess_bdr);
   void UpdateProblem();

   virtual ~AMRResistiveMHDOperator();
};

AMRResistiveMHDOperator::AMRResistiveMHDOperator(FiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc, double resi)
   : TimeDependentOperator(4*f.GetVSize(), 0.0), fespace(f), 
     M(NULL), Mrhs(NULL), K(NULL), KB(NULL), DSl(NULL), DRe(NULL), Nv(NULL), Nb(NULL), E0(NULL),
     visc_coeff(visc), resi_coeff(resi),
     viscosity(visc),  resistivity(resi), 
     E0rhs(NULL), M_prec(NULL), K_prec(NULL)
{
   //mass matrix
   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);

   Mrhs = new BilinearForm(&fespace);
   Mrhs->AddDomainIntegrator(new MassIntegrator);

   //stiffness matrix
   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);

   KB = new BilinearForm(&fespace);
   KB->AddDomainIntegrator(new DiffusionIntegrator);      //  K matrix
   KB->AddBdrFaceIntegrator(new BoundaryGradIntegrator);  // -B matrix

   //resi_coeff and visc_coeff have to be stored for assembling for some reason
   DRe = new BilinearForm(&fespace);
   DRe->AddDomainIntegrator(new DiffusionIntegrator(visc_coeff));    

   DSl = new BilinearForm(&fespace);
   DSl->AddDomainIntegrator(new DiffusionIntegrator(resi_coeff));    
}

void AMRResistiveMHDOperator::assembleProblem(Array<int> &ess_bdr)
{
   //update ess_tdof_list
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //update mass matrix
   M->Assemble();
   Mrhs->Assemble();

   //update stiffness matrix
   K->Assemble();

   //update solvers
   M->FormSystemMatrix(ess_tdof_list, Mmat);
   delete M_prec;
   GSSmoother *M_prec_gs = new GSSmoother(Mmat);
   M_prec=M_prec_gs;
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-7);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(200);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(Mmat);

   K->FormSystemMatrix(ess_tdof_list, Kmat);
   delete K_prec;
   GSSmoother *K_prec_gs = new GSSmoother(Kmat);
   K_prec=K_prec_gs;
   K_solver.iterative_mode = false;
   K_solver.SetRelTol(1e-7);
   K_solver.SetAbsTol(0.0);
   K_solver.SetMaxIter(1000);
   K_solver.SetPrintLevel(0);
   K_solver.SetPreconditioner(*K_prec);
   K_solver.SetOperator(Kmat);

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

   /*
   cout << "Number of matrix in M: " <<  M->SpMat().Size()<< endl;
   cout << "Number of matrix in K: " <<  K->SpMat().Size()<< endl;
   cout << "Number of matrix in KB: " << KB->SpMat().Size()<< endl;
   cout << "Number of matrix in DSl: " << DSl->SpMat().Size()<< endl;
   cout << "Number of matrix in DRe: " << DRe->SpMat().Size()<< endl;
   */

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

   cout<<"True V size = "<<fespace.GetTrueVSize()<<endl;
   cout<<"Problem size = "<<fespace.GetVSize()<<endl;
   cout << "Number of elements in mesh: " << fespace.GetNE() << endl;
   width = height = fespace.GetVSize()*4;
}          

void AMRResistiveMHDOperator::SetRHSEfield( double(* f)( const Vector&) ) 
{
   delete E0;
   E0rhs = new FunctionCoefficient(f);
   E0 = new LinearForm(&fespace);
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

   Vector dphi_dt(dvx_dt.GetData() +   0, sc);
   Vector dpsi_dt(dvx_dt.GetData() +  sc, sc);
   Vector   dw_dt(dvx_dt.GetData() +2*sc, sc);
   Vector   dj_dt(dvx_dt.GetData() +3*sc, sc);

   z.SetSize(sc);

   /*
   ofstream myfile0("psi.dat");
   psi.Print(myfile0, 1000);
   ofstream myfile1("phi.dat");
   phi.Print(myfile1, 1000);
   ofstream myfile2("w.dat");
   w.Print(myfile2, 1000);
   ofstream myfile3("j.dat");
   j.Print(myfile3, 1000);
   ofstream myfile4("vx.dat");
   //ofstream myfile4("vx.dat");
   //vx.Print(myfile4, 1000);

   cout << "vs size ="<<vx.Size()<<" sc ="<<sc<<" h ="<<height<<endl;
   cout << "Number of scalar unknowns in psi: " <<psi.Size()<< endl;
   cout << "Number of scalar unknowns in phi: " <<phi.Size()<< endl;
   cout << "Number of scalar unknowns in   w: " <<  w.Size()<< endl;
   cout << "Number of scalar unknowns in   j: " <<  j.Size()<< endl;
   */

   Nv->Mult(psi, z);
   if (resistivity != 0.0)
   {
      DSl->AddMult(psi, z);
   }
   if (E0!=NULL)
     z += *E0;
   z.Neg(); // z = -z

   //another way; but it is slower
   SparseMatrix A;
   Vector B, X;
   M->FormLinearSystem(ess_tdof_list, dpsi_dt, z, A, X, B); // Alters matrix and rhs to enforce bc

   //the run time of these two options are almost identical
   //note pcg tolerance will be a sqrt of when setting SetRelTol!!
   //GSSmoother Mpre(A);
   //PCG(A, Mpre, B, X, 1, 1000, 1e-14, 0.0); 

   M_solver.Mult(B, X);
   M->RecoverFEMSolution(X, z, dpsi_dt);

   //ofstream myfile("A1.dat");
   //A.PrintCSR(myfile);

   Nv->Mult(w, z);
   /*
   cout << "Number of scalar unknowns in   z!!!: " <<  z.Size()<< endl;
   ofstream myfile2("zLHS2.dat");
   z.Print(myfile2, 1000);
   ofstream myfile3("w2.dat");
   w.Print(myfile2, 1000);
   */
   if (viscosity != 0.0)
   {
      DRe->AddMult(w, z);
   }
   z.Neg(); // z = -z
   Nb->AddMult(j, z);

   M->FormLinearSystem(ess_tdof_list, dw_dt, z, A, X, B); // Alters matrix and rhs to enforce bc
   //PCG(A, Mpre, B, X, 0, 1000, 1e-14, 0.0); 
   M_solver.Mult(B, X);
   M->RecoverFEMSolution(X, z, dw_dt);
}

void AMRResistiveMHDOperator::assembleNv(GridFunction *gf) 
{
   delete Nv;
   Nv = new BilinearForm(&fespace);
   MyCoefficient velocity(gf, 2);   //update velocity

   Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
   Nv->Assemble(); 
}

void AMRResistiveMHDOperator::assembleNb(GridFunction *gf) 
{
   delete Nb;
   Nb = new BilinearForm(&fespace);
   MyCoefficient Bfield(gf, 2);   //update B

   Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
   Nb->Assemble();
}

void AMRResistiveMHDOperator::UpdateJ(Vector &vx)
{
   //the current is J=-M^{-1}*K*Psi
   int sc = height/4;
   Vector psi(vx.GetData() +  sc, sc);
   Vector   j(vx.GetData() +3*sc, sc);  //it creates a reference
   
   z.SetSize(sc);
   //cout <<"======Update J======"<<endl;

   /*
   cout << "Number of scalar unknowns in psi: (UpdateJ) " <<  psi.Size()<< endl;
   cout << "Number of scalar unknowns in   j: (UpdateJ) " <<  j.Size()<< endl;
   cout << "Number of scalar unknowns in   z: (UpdateJ) " <<  z.Size()<< endl;
   cout << "Number of scalar unknowns in  sc: (UpdateJ) " <<  sc<< endl;
   */

   KB->Mult(psi, z);
   z.Neg(); // z = -z

   SparseMatrix A;
   Vector Y, Z;

   //apply Dirichelt boundary 
   //(j is initially from a projection with initial condition, so it satisfies the boundary conditino all the time)
   M->FormLinearSystem(ess_tdof_list, j, z, A, Y, Z);
   /*
   GSSmoother Mpre(A);
   PCG(A, Mpre, Z, Y, 0, 1000, 1e-14, 0.0); 
   */
   M_solver.Mult(Z, Y);
   M->RecoverFEMSolution(Y, z, j);

}

void AMRResistiveMHDOperator::BackSolvePsi(Vector &vx)
{
   int sc = height/4;
   Vector psi(vx.GetData() +  sc, sc);
   Vector   j(vx.GetData() +3*sc, sc);

   cout <<"===Back Solve Psi==="<<endl;
   z.SetSize(sc);

   Mrhs->Mult(j, z);
   z.Neg(); // z = -z

   SparseMatrix A;
   Vector B, X;
   K->FormLinearSystem(ess_tdof_list, psi, z, A, X, B); // Alters matrix and rhs to enforce bc
   /*
   GSSmoother Mpre(A);
   PCG(A, Mpre, B, X, 0, 1000, 1e-14, 0.0); 
   */
   M_solver.Mult(B, X);
   K->RecoverFEMSolution(X, z, psi);
}

void AMRResistiveMHDOperator::UpdatePhi(Vector &vx)
{
   //Phi=-K^{-1}*M*w
   int sc = height/4;
   Vector phi(vx.GetData() +   0, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   /*
   cout << "Number of scalar unknowns in z: (UpdatePhi) " <<  z.Size()<< endl;
   cout << "Number of operator size in M: (UpdatePhi) " <<  M->SpMat().Size()<< endl;
   */

   Mrhs->Mult(w, z);
   z.Neg(); // z = -z

   SparseMatrix A;
   Vector B, X;
   K->FormLinearSystem(ess_tdof_list, phi, z, A, X, B); // Alters matrix and rhs to enforce bc
   /*
   GSSmoother Mpre(A);
   PCG(A, Mpre, B, X, 0, 1000, 1e-14, 0.0); 
   */
   K_solver.Mult(B, X);

   K->RecoverFEMSolution(X, z, phi);
 
   //K_solver.Mult(z, phi);
}


AMRResistiveMHDOperator::~AMRResistiveMHDOperator()
{
    //free used memory
    delete M;
    delete Mrhs;
    delete K;
    delete KB;
    delete Nv;
    delete Nb;
    delete DRe;
    delete DSl;
    delete E0;
    delete E0rhs;
    delete M_prec;
    delete K_prec;
}


