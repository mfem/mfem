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

   /*
   //new things for implicit stepping
   ReducedSystemOperator *reduced_oper;
   PetscNonlinearSolver* pnewton_solver;
   PetscPreconditionerFactory *J_factory;
   Solver *J_solver;
   */

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
                       double visc, double resi); 

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

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

/** reduced system
class ReducedSystemOperator : public Operator
{
private:
   ParBilinearForm *M, *S;
   ParNonlinearForm *H;
   mutable HypreParMatrix *Jacobian;
   double dt;
   const Vector *v, *x;
   mutable Vector w, z;
   const Array<int> &ess_tdof_list;

public:
   ReducedSystemOperator(ParBilinearForm *M_, ParBilinearForm *S_,
                         ParNonlinearForm *H_, const Array<int> &ess_tdof_list);

   /// Set current dt, v, x values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *v_, const Vector *x_);

   /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &k) const;

   virtual ~ReducedSystemOperator();

};
 *
 */

/*** Auxiliary class to provide preconditioners for matrix-free methods 
class PreconditionerFactory : public PetscPreconditionerFactory
{
private:
   const ReducedSystemOperator& op;

public:
   PreconditionerFactory(const ReducedSystemOperator& op_,
                         const string& name_): PetscPreconditionerFactory(name_), op(op_) {};
   virtual mfem::Solver* NewPreconditioner(const mfem::OperatorHandle&);
   virtual ~PreconditionerFactory() {};
};
*/

ResistiveMHDOperator::ResistiveMHDOperator(ParFiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc, double resi)
   : TimeDependentOperator(3*f.TrueVSize(), 0.0), fespace(f),
     M(NULL), K(NULL), KB(NULL), DSl(&fespace), DRe(&fespace),
     Nv(NULL), Nb(NULL), E0(NULL), Sw(NULL), E0Vec(NULL),
     viscosity(visc),  resistivity(resi), useAMG(false), 
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
   //KB->FormSystemMatrix(ess_tdof_list, KBmat);

   ConstantCoefficient visc_coeff(viscosity);
   DRe.AddDomainIntegrator(new DiffusionIntegrator(visc_coeff));    
   DRe.Assemble();

   ConstantCoefficient resi_coeff(resistivity);
   DSl.AddDomainIntegrator(new DiffusionIntegrator(resi_coeff));    
   DSl.Assemble();
}

void ResistiveMHDOperator::SetRHSEfield(FunctionCoefficient Efield) 
{
   delete E0;
   E0 = new ParLinearForm(&fespace);
   E0->AddDomainIntegrator(new DomainLFIntegrator(Efield));
   E0->Assemble();
   E0Vec=E0->ParallelAssemble();
}

void ResistiveMHDOperator::SetInitialJ(FunctionCoefficient initJ) 
{
    j.ProjectCoefficient(initJ);
    j.SetTrueVector();
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

   /*
   ofstream myfile("z0.dat");
   z.Print(myfile, 10);
   cout<<z.Size()<<endl;

   ofstream myfile2("dwdt.dat");
   dw_dt.Print(myfile2, 10);
   cout<<dw_dt.Size()<<endl;
   */

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
    //hypre needs to be deleted earilier
    delete K_amg;
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
    //delete K_amg;
    //delete M_solver;
    //delete K_solver;
    //delete M_prec;
    //delete K_prec;
}


