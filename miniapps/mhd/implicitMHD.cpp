//                                MFEM modified from Example 10 and 16
//
// Compile with: make implicitMHD
//
// Sample runs:
//    implicitMHD -m ../../data/beam-quad.mesh -s 3 -r 2 -o 2 -dt 3
//
// Description:  it solves a time dependent resistive MHD problem 
// Author: QT

#include "mfem.hpp"
#include "myCoefficient.hpp"
//#include "PDSolver.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;


double alpha; //a global value of magnetude for the pertubation
double Lx;  //size of x domain

/** After spatial discretization, the resistive MHD model can be written as a
 *  system of ODEs:
 *     dPsi/dt = M^{-1}*F1,
 *     dw  /dt = M^{-1}*F2,
 *  coupled with two linear systems
 *     j   = -M^{-1}*K*Psi 
 *     Phi = -K^{-1}*M*w
 *  so far there seems no need to do a BlockNonlinearForm
 *
 *  Class ImplicitMHDOperator represents the right-hand side of the above
 *  system of ODEs. */
class ImplicitMHDOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list;

   BilinearForm *M, *K, DSl, DRe; //mass, stiffness, diffusion with SL and Re
   BilinearForm *Nv, *Nb;
   SparseMatrix Mmat, Kmat;
   SparseMatrix NvMat, NbMat;

   double viscosity, resistivity;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver K_solver; // Krylov solver for inverting the stiffness matrix K
   DSmoother K_prec;  // Preconditioner for the stiffness matrix K

   mutable Vector z; // auxiliary vector 

   //VectorGridFunctionCoefficient *velocity,*Bfield;
   //MyCoefficient velocity, Bfield;

public:
   ImplicitMHDOperator(FiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi);

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   void UpdateJ(Vector &vx);
   void UpdatePhi(Vector &vx);

   //void assembleNv(const Vector &vx);
   //void assembleNb(const Vector &vx);
   void assembleNv(GridFunction *gf);
   void assembleNb(GridFunction *gf);

   // Compute the right-hand side of the ODE system in the predictor step.
   //void MultPre(const Vector &vx, Vector &dvx_dt) const;

   virtual ~ImplicitMHDOperator();
};

//initial condition
double InitialPhi(const Vector &x)
{
    return 0.0;
}

double InitialW(const Vector &x)
{
    return 0.0;
}

double InitialJ(const Vector &x)
{
   return -M_PI*M_PI*(1.0+4.0/Lx/Lx)*alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double InitialPsi(const Vector &x)
{
   //cout <<"("<<x(0)<<" "<<x(1)<<") ";
   //cout<<M_PI<<" "<<cos(M_PI*x(1))<<" "<<cos(2.0*M_PI/Lx*x(0))<<" ";
   return -x(1)+alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "./xperiodic-square.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 2;
   double t_final = 5.0;
   double dt = 0.0001;
   double visc = 0.0;
   double resi = 0.0;
   alpha = 0.001; 
   Lx=3.0;

   bool visualization = true;
   int vis_steps = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 2 - Brailovskaya,\n\t"
                  " only FE supported 13 - RK3 SSP, 14 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visc, "-visc", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&resi, "-resi", "--resistivity",
                  "Resistivity coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODESolver *ode_solver, *ode_predictor;
   switch (ode_solver_type)
   {
     // Explicit methods XXX: waring: FE is not stable 
     case 1: ode_solver = new ForwardEulerSolver; break;
     case 2: 
             ode_solver = new ForwardEulerSolver; 
             ode_predictor = new ForwardEulerSolver; 
             break; //first order predictor-corrector
     case 3: ode_solver = new RK3SSPSolver; break;
     default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution.    
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the vector finite element spaces representing 
   //  [Psi, Phi, w, j]
   // in block vector bv, with offsets given by the
   //    fe_offset array.
   // All my fespace is 1D bu the problem is multi-dimensional!!
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll); 
   if (false)
   {
        cout <<"dim="<<dim<<endl;
        cout <<"fespace dim="<<fespace.GetVDim()<<endl;
   }

   int fe_size = fespace.GetTrueVSize();
   cout << "Number of scalar unknowns: " << fe_size << endl;
   Array<int> fe_offset(5);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;
   fe_offset[4] = 4*fe_size;

   BlockVector vx(fe_offset);
   GridFunction psi, phi, w, j;
   phi.MakeTRef(&fespace, vx.GetBlock(0), 0);
   psi.MakeTRef(&fespace, vx.GetBlock(1), 0);
     w.MakeTRef(&fespace, vx.GetBlock(2), 0);
     j.MakeTRef(&fespace, vx.GetBlock(3), 0);

   // 6. Set the initial conditions, and the boundary conditions
   FunctionCoefficient phiInit(InitialPhi);
   phi.ProjectCoefficient(phiInit);
   phi.SetTrueVector();

   FunctionCoefficient psiInit(InitialPsi);
   psi.ProjectCoefficient(psiInit);
   psi.SetTrueVector();

   FunctionCoefficient wInit(InitialW);
   w.ProjectCoefficient(wInit);
   w.SetTrueVector();

   FunctionCoefficient jInit(InitialJ);
   j.ProjectCoefficient(jInit);
   j.SetTrueVector();

   //this is a periodic boundary condition in x, so ess_bdr
   //but may need other things here if not periodic
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;  //set attribute 1 to Direchlet boundary 

   // 7. Initialize the MHD operator, the GLVis visualization    
   ImplicitMHDOperator oper(fespace, ess_bdr, visc, resi);

   socketstream vis_phi;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_phi.open(vishost, visport);
      if (!vis_phi)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         vis_phi.precision(8);
         vis_phi << "phi\n" << *mesh << phi;
         vis_phi << "pause\n";
         vis_phi << flush;
         vis_phi << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
    }

   double t = 0.0;
   oper.SetTime(t);
   ode_predictor->Init(oper);   //FIXME
   ode_solver->Init(oper);


   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      //---Predictor stage---
      //assemble the nonlinear terms
      oper.assembleNv(&phi);
      oper.assembleNb(&psi);

      ode_predictor->Step(vx, t, dt_real);
      
      oper.UpdateJ(vx);

      //---Corrector stage---
      //assemble the nonlinear terms (only psi is updated)
      oper.assembleNb(&psi);
      ode_solver->Step(vx, t, dt_real);
      oper.UpdateJ(vx); 
      oper.UpdatePhi(vx);

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
        /*
         double ee = oper.ElasticEnergy(x.GetTrueVector());
         double ke = oper.KineticEnergy(v.GetTrueVector());

         cout << "step " << ti << ", t = " << t << ", EE = " << ee << ", KE = "
              << ke << ", ΔTE = " << (ee+ke)-(ee0+ke0) << endl;
        */

        cout << "step " << ti << ", t = " << t <<endl;

         if (visualization)
         {
            vis_phi << "phi\n" << *mesh << phi << flush;
         }
      }

   }

   // 9. Save the solutions.
   {
      ofstream osol("phi.sol");
      osol.precision(8);
      phi.Save(osol);

      ofstream osol2("current.sol");
      osol2.precision(8);
      j.Save(osol2);

      ofstream osol3("psi.sol");
      osol3.precision(8);
      psi.Save(osol3);

      ofstream osol4("omega.sol");
      osol4.precision(8);
      w.Save(osol4);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete ode_predictor;
   delete mesh;

   return 0;
}


ImplicitMHDOperator::ImplicitMHDOperator(FiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc,double resi)
   : TimeDependentOperator(4*f.GetTrueVSize(), 0.0), fespace(f),
     M(NULL), K(NULL), DSl(&fespace), DRe(&fespace), Nv(NULL), Nb(NULL),
     viscosity(visc),  resistivity(resi), z(height/4)
{
   const double rel_tol = 1e-8;
   const int skip_zero_entries = 0;
   ConstantCoefficient one(1.0);
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //direct solver for M and K for now
   //mass matrix
   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);
   M->Assemble(skip_zero_entries);
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);


   //cout <<tmp(25947,25947);

   //stiffness matrix
   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);
   K->Assemble(skip_zero_entries);
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   K_solver.iterative_mode = false;
   K_solver.SetRelTol(rel_tol);
   K_solver.SetAbsTol(0.0);
   K_solver.SetMaxIter(30);
   K_solver.SetPrintLevel(0);
   K_solver.SetPreconditioner(K_prec);
   K_solver.SetOperator(Kmat); //this is a real matrix

   /*
   //this has to be assembled in each time step? but how could I do implicitly?
   //TODO add nonlinear form here v=[-Phi_y, Phi_x]
   //VectorGridFunctionCoefficient velocity(dim, velocity_function);
   //velocity->SetGridFunction(stuff??);
   velocity(Phi);
   Nv.AddDomainIntegrator(new ConvectionIntegrator(*velocity));
   Nv.SetEssentialTrueDofs(ess_tdof_list);
   Nv.Assemble(skip_zero_entries);

   //TODO add nonlinear form here B=[-Psi_y, Psi_x]
   //VectorGridFunctionCoefficient Bfield(dim, B_function);
   Bfield->SetGridFunction(stuff??);
   Nb.AddDomainIntegrator(new ConvectionIntegrator(*Bfield));
   Nb.SetEssentialTrueDofs(ess_tdof_list);
   Nb.Assemble(skip_zero_entries);
   */

   ConstantCoefficient visc_coeff(viscosity);
   DRe.AddDomainIntegrator(new DiffusionIntegrator(visc_coeff));    
   DRe.Assemble(skip_zero_entries);

   ConstantCoefficient resi_coeff(resistivity);
   DSl.AddDomainIntegrator(new DiffusionIntegrator(resi_coeff));    
   DSl.Assemble(skip_zero_entries);
}

void ImplicitMHDOperator::Mult(const Vector &vx, Vector &dvx_dt) const
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

   dphi_dt=0.0;
   dj_dt=0.0;

   NvMat.Mult(psi, z);
   if (resistivity != 0.0)
   {
      DSl.AddMult(psi, z);
   }
   z.Neg(); // z = -z
   M_solver.Mult(z, dpsi_dt);
   //abort();  //debug


   NvMat.Mult(w, z);
   if (viscosity != 0.0)
   {
      DRe.AddMult(w, z);
   }
   z.Neg(); // z = -z
   NbMat.AddMult(j, z);
   M_solver.Mult(z, dw_dt);
}

void ImplicitMHDOperator::assembleNv(GridFunction *gf) 
{
   int sc = height/4;
   //cout << "sc ="<<sc<<" height="<<height<<endl;   //debug

   //M_solver.Mult(*gf, z);
   //Vector phi(vx.GetData() +   0, sc);
   //cout <<phi(0)<<endl;   //debug
   //GridFunction phiGF(&fespace); 
   //phiGF.SetFromTrueDofs(phi);
   
   int skip_zero_entries=0;


   delete Nv;
   Nv = new BilinearForm(&fespace);
   MyCoefficient velocity(gf, 2);   //we update velocity


   //cout << "sc ="<<sc<<" height="<<height<<endl;   //debug
   Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
   Nv->Assemble(skip_zero_entries); 
   Nv->FormSystemMatrix(ess_tdof_list, NvMat);

}

void ImplicitMHDOperator::assembleNb(GridFunction *gf) 
{
   int sc = height/4;
   //cout << "sc ="<<sc<<" height="<<height<<endl;   //debug

   //Vector psi(vx.GetData() +  sc, sc);
   //GridFunction psiGF(&fespace); 
   //psiGF.SetFromTrueDofs(psi);

   int skip_zero_entries=0;

   delete Nb;
   Nb = new BilinearForm(&fespace);
   MyCoefficient Bfield(gf, 2);   //we update B

   Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
   Nb->Assemble(skip_zero_entries);
   Nb->FormSystemMatrix(ess_tdof_list, NbMat);
}

void ImplicitMHDOperator::UpdateJ(Vector &vx)
{
   //the current is J=-M^{-1}*K*Psi
   int sc = height/4;
   Vector psi(vx.GetData() +  sc, sc);
   Vector   j(vx.GetData() +3*sc, sc);  //it creates a reference, so it should be good

   Kmat.Mult(psi, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, j);
}

void ImplicitMHDOperator::UpdatePhi(Vector &vx)
{
   //Phi=-K^{-1}*M*w
   int sc = height/4;
   Vector phi(vx.GetData() +   0, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   Mmat.Mult(w, z);
   z.Neg(); // z = -z
   K_solver.Mult(z, phi);
}


ImplicitMHDOperator::~ImplicitMHDOperator()
{
    //free used memory
    //delete Bfield;
    //delete velocity;
}


