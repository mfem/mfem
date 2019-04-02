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

   BilinearForm *M, *K, *KB, *DSl, *DRe; //mass, stiffness, diffusion with SL and Re
   BilinearForm *Nv, *Nb;
   LinearForm *E0, *Sw; //two source terms
   SparseMatrix Mmat, Kmat;
   ConstantCoefficient visc_coeff, resi_coeff;
   double viscosity, resistivity;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   GSSmoother *M_prec;  // Preconditioner for the mass matrix M

   CGSolver K_solver; // Krylov solver for inverting the stiffness matrix K
   GSSmoother *K_prec;  // Preconditioner for the stiffness matrix K

   mutable Vector z; // auxiliary vector 

public:
   AMRResistiveMHDOperator(FiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi);   //this is old
   AMRResistiveMHDOperator(FiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi, int icase);

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   //set rhs E0
   void SetRHSEfield(FunctionCoefficient Efield);

   void UpdateJ(Vector &vx);
   void UpdatePhi(Vector &vx);
   void assembleNv(GridFunction *gf);
   void assembleNb(GridFunction *gf);
   void assembleProblem(Array<int> &ess_bdr);
   void UpdateProblem();

   virtual ~AMRResistiveMHDOperator();
};

AMRResistiveMHDOperator::AMRResistiveMHDOperator(FiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc, double resi)
   : TimeDependentOperator(4*f.GetTrueVSize(), 0.0), fespace(f), 
     M(NULL), K(NULL), KB(NULL), DSl(NULL), DRe(NULL), Nv(NULL), Nb(NULL), E0(NULL), Sw(NULL),
     visc_coeff(visc), resi_coeff(resi),
     viscosity(visc),  resistivity(resi), 
     M_prec(NULL), K_prec(NULL)
{
   //mass matrix
   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);

   //stiffness matrix
   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);

   KB = new BilinearForm(&fespace);
   KB->AddDomainIntegrator(new DiffusionIntegrator);      //  K matrix
   KB->AddBdrFaceIntegrator(new BoundaryGradIntegrator);  // -B matrix

   //resi_coeff and visc_coeff have to be stored for assembling for some strange reason
   DRe = new BilinearForm(&fespace);
   DRe->AddDomainIntegrator(new DiffusionIntegrator(visc_coeff));    

   DSl = new BilinearForm(&fespace);
   DSl->AddDomainIntegrator(new DiffusionIntegrator(resi_coeff));    
}

void AMRResistiveMHDOperator::assembleProblem(Array<int> &ess_bdr)
{
   const double rel_tol = 1e-8;

   //update ess_tdof_list
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //update mass matrix
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   delete M_prec;
   GSSmoother *M_prec_gs = new GSSmoother(Mmat);
   M_prec=M_prec_gs;

   M_solver.iterative_mode = true;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(500);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(Mmat);

   //update stiffness matrix
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   delete K_prec;
   GSSmoother *K_prec_gs = new GSSmoother(Kmat);
   K_prec=K_prec_gs;

   K_solver.iterative_mode = true;
   K_solver.SetRelTol(rel_tol);
   K_solver.SetAbsTol(0.0);
   K_solver.SetMaxIter(500);
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
}

void AMRResistiveMHDOperator::UpdateProblem()
{
   M->Update();
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

   cout<<"problem size ="<<fespace.GetTrueVSize()<<endl;
   width = height = fespace.GetTrueVSize()*4;
}          
           
void AMRResistiveMHDOperator::SetRHSEfield(FunctionCoefficient Efield) 
{
   delete E0;
   E0 = new LinearForm(&fespace);
   E0->AddDomainIntegrator(new DomainLFIntegrator(Efield));
}

void AMRResistiveMHDOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors and time derivative
   int sc = height/4;
   Vector phi(vx.GetData() +   0, sc);
   Vector psi(vx.GetData() +  sc, sc);
   Vector   w(vx.GetData() +2*sc, sc);
   Vector   j(vx.GetData() +3*sc, sc);

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
   */
   //ofstream myfile4("vx.dat");
   //vx.Print(myfile4, 1000);

   /*
   cout << "vs size ="<<vx.Size()<<" sc ="<<sc<<" h ="<<height<<endl;
   cout << "Number of scalar unknowns in psi: " <<psi.Size()<< endl;
   cout << "Number of scalar unknowns in phi: " <<phi.Size()<< endl;
   cout << "Number of scalar unknowns in   w: " <<  w.Size()<< endl;
   cout << "Number of scalar unknowns in   j: " <<  j.Size()<< endl;
   */

   dphi_dt=0.0;
   dj_dt=0.0;

   Nv->Mult(psi, z);
   if (resistivity != 0.0)
   {
      DSl->AddMult(psi, z);
   }
   if (E0!=NULL)
     z += *E0;
   z.Neg(); // z = -z

   /*
   ofstream myfile("zLHS1.dat");
   z.Print(myfile, 1000);
   */

   if (true)
   {
      for (int i=0; i<ess_tdof_list.Size(); i++)
          z(ess_tdof_list[i])=0.0; //set homogeneous Dirichlet condition by hand
      M_solver.Mult(z, dpsi_dt);
   }
   else
   {
       //another way; but it is slower
       SparseMatrix A;
       Vector B, X;
       M->FormLinearSystem(ess_tdof_list, dpsi_dt, z, A, X, B); // Alters matrix and rhs to enforce bc
       PCG(Mmat, *M_prec, B, X, 0, 200, 1e-12, 0.0); 
       //CG(A, B, X);
       M->RecoverFEMSolution(X, z, dpsi_dt);
   }


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

   for (int i=0; i<ess_tdof_list.Size(); i++)
       z(ess_tdof_list[i])=0.0; //set Dirichlet condition by hand


   M_solver.Mult(z, dw_dt);
   //cout << "Number of scalar unknowns in   z: " <<  z.Size()<< endl;

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
   SparseMatrix tmp;
   Vector Y, Z;
   
   z.SetSize(sc);

   /*
   cout << "Number of scalar unknowns in psi: " <<  psi.Size()<< endl;
   cout << "Number of scalar unknowns in   j: " <<  j.Size()<< endl;
   cout << "Number of scalar unknowns in   z: " <<  z.Size()<< endl;
   cout << "Number of scalar unknowns in  sc: " <<  sc<< endl;
   */

   KB->Mult(psi, z);
   z.Neg(); // z = -z
   M->FormLinearSystem(ess_tdof_list, j, z, tmp, Y, Z); //apply Dirichelt boundary (j is initially from a projection with initial condition, so it satisfies the boundary conditino all the time)
   M_solver.Mult(Z, Y);
   M->RecoverFEMSolution(Y, z, j);

}

void AMRResistiveMHDOperator::UpdatePhi(Vector &vx)
{
   //Phi=-K^{-1}*M*w
   int sc = height/4;
   Vector phi(vx.GetData() +   0, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   Mmat.Mult(w, z);
   z.Neg(); // z = -z
   K_solver.Mult(z, phi);
}


AMRResistiveMHDOperator::~AMRResistiveMHDOperator()
{
    //free used memory
    delete M;
    delete K;
    delete KB;
    delete Nv;
    delete Nb;
    delete DRe;
    delete DSl;
    delete M_prec;
    delete K_prec;
}


