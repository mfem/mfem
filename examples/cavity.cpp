//                                MFEM Example Cavity
//
// Compile with: make cavity
//
// Sample runs:  cavity -m ../data/cavity_2d.mesh -re 10
//
// Description:  This example code solves a simple 2D/3D mixed Navier-Stokes 
//               problem corresponding to the saddle point system
//                        - nu*div(grad u)+ u.grad u + grad p = f
//                                                    - div u = 0
//               Equal order u/p is considered and GLS type stabilization 
//               is applied. The final stabilized form takes the following form:
//
//                       (grad u , grad v) - (p, div v) + (div u, q)
//                                     + (u.grad u, v)
//                    + tau1 (-u.grad v+ nu lap v - grad q, 
//                                          f - grad p + nu lap u - u.grad u)
//                                       + tau2 (div v, div u) = <f,v>
//
//               Two "integrators" are added, the first one to add the all 
//               linear terms called "VectorGalerkinNSIntegrator" and the second
//               one to add all nonlinear terms i.e convective term and 
//               stabilization terms called "VectorNonLinearNSIntegrator". 
//               The "FiniteElementSpace" has dimension of "dim+1" representing 
//               velocity dofs following pressure dof.  
//
//               This example model a driven cavity problem. The domain is a 
//               1 x 1 square where no-slip b.c. is assigned to all boundaries
//               except the top one in which velocity is assigned to {1,0}. 
//               Pressure is also fixed on the top wall. 
//
//               An option "-re" is added to set the Reynolds number at the 
//               command line. It basically sets the nu equal to 1/re. 


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

class NSOperator : public Operator
{
public:
    NSOperator(FiniteElementSpace* VQ_space_, Array<int>& ess_vel_bdr_, 
            Vector& Gravity_, double dyn_visc_, double density_, 
            BlockVector& x_, BlockVector& rhs_);
    
    //Compute the residual vector      
    virtual void Mult(const Vector& acc, Vector &RHS) const;
    
    //Compute the LHS matrix
    virtual Operator &GetGradient(const Vector& acc) const;

    virtual ~NSOperator();
    
private:
    mutable SparseMatrix *Jacobian;
    FiniteElementSpace *VQ_space;
    double dyn_visc, density;
    Vector Gravity;
    Vector x;
    BilinearForm GalVarf;
    NonlinearForm noLinVarf;

    LinearForm fform;
    Array<int> *ess_dofs;
};

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/cavity_2d.mesh";
   int order = 1;
   bool visualization = 1;
   double re = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&re, "-re", "--Reynolds Number", 
                  "Input Re number");
   
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   {
      int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use the
   //    equal order H1 finite elements of the specified order.
   //FiniteElementCollection *fe_coll(new H1_FECollection(order, dim));
   //FiniteElementCollection *fe_coll(new QuadraticFECollection());
   FiniteElementCollection *fe_coll(new CubicFECollection());
   //FiniteElementCollection *l2_coll(new H1_FECollection(order, dim));

   //1 is added for pressure; u v (w) p
   FiniteElementSpace *VQspace = new FiniteElementSpace(mesh, fe_coll, dim+1); 

   // 5. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   int d1_size = VQspace->GetNDofs();
   Array<int> fe_offsets(3); // number of variables + 1
   fe_offsets[0] = 0;
   fe_offsets[1] = dim * d1_size;
   fe_offsets[2] = (dim + 1) * d1_size;

   std::cout << "***********************************************************\n";
   std::cout << "dim(V) = " << fe_offsets[1] - fe_offsets[0] << "\n";
   std::cout << "dim(P) = " << fe_offsets[2] - fe_offsets[1] << "\n";
   std::cout << "***********************************************************\n";

   //Solution and rhs
   BlockVector up(fe_offsets);
   BlockVector rhs(fe_offsets);
   
   up  = 0.0;   
   rhs = 0.0;
   
   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0);
   double dyn_visc = 1./re;
   double density = 1.0;
   ConstantCoefficient nuCoef(dyn_visc);
   
   // Determine the list of essential true dofs
   int bdr_attr_size = mesh->bdr_attributes.Max();
   Array<int> aux_vel, ess_vel_tdof_list;
   Array<int> ess_vel_bdr(bdr_attr_size);
   ess_vel_bdr = 0;
   ess_vel_bdr[0] = 1;
   for (int kk = 0; kk < dim; kk++){
       VQspace->GetEssentialTrueDofs(ess_vel_bdr, aux_vel, kk);
       ess_vel_tdof_list.Append(aux_vel);
   }

   //Do sth for pressure
   Array<int> ess_pr_tdof_list;
   Array<int> ess_pr_bdr(bdr_attr_size);
   ess_pr_bdr = 0;
   ess_pr_bdr[1] = 1;
   VQspace->GetEssentialTrueDofs(ess_pr_bdr, ess_pr_tdof_list, dim);

   //Define inlet as edge/faces marked as 2
   Array<int> aux_inlet, inlet_dof_list; 
   Array<int> inlet_bdr(bdr_attr_size);
   inlet_bdr = 0;
   inlet_bdr[1] = 1;
   for (int kk = 0; kk < dim ; kk++){
       VQspace->GetEssentialTrueDofs(inlet_bdr, aux_inlet, kk);
       inlet_dof_list.Append(aux_inlet);
   }

   //Assign inlet (1,0) to the solution vector
   VQspace->GetEssentialTrueDofs(inlet_bdr, aux_inlet, 0);
   for (int kk = 0; kk < aux_inlet.Size(); kk++)
           up[aux_inlet[kk]] = 1.0;
   
   //fixed the ones common with the fixed wall
   for (int kk = 0; kk < ess_vel_tdof_list.Size(); kk++)
       up[ess_vel_tdof_list[kk]] = 0.0;
   
   //Total Essential dofs
   Array<int> tot_ess_tdof_list;
   ess_vel_tdof_list.Copy(tot_ess_tdof_list);
   tot_ess_tdof_list.Append(ess_pr_tdof_list);
   tot_ess_tdof_list.Append(inlet_dof_list);
   tot_ess_tdof_list.Sort();
   tot_ess_tdof_list.Unique();   
   
   //Add body force
   Vector Gravity(dim);
   Gravity = 0.0;
   Gravity[1] = 0.0; //g is assumed in y direction
 
   //define operator 
   NSOperator NSoper(VQspace, tot_ess_tdof_list, Gravity, dyn_visc, 
                         density, up, rhs);
   

   // 10. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.
   int maxIter(1000);
   double rtol(1.e-10);
   double atol(1.e-12);
 
   chrono.Clear();
   chrono.Start();
   Solver* J_prec;
   //J_prec = new DSmoother(1);
   BiCGSTABSolver J_solver;//BiCGSTABSolver solver;//MINRESSolver solver;FGMRESSolver
   J_solver.SetPrintLevel(-1);
   J_solver.SetRelTol(rtol);
   J_solver.SetAbsTol(0.0);
   J_solver.SetMaxIter(maxIter);
   //J_solver.SetPreconditioner(*J_prec);
           
           
   NewtonSolver newton_solver;
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(J_solver);
   newton_solver.SetOperator(NSoper); 
   newton_solver.SetPrintLevel(1); 
   newton_solver.SetRelTol(rtol);
   newton_solver.SetAbsTol(0.0);
   newton_solver.SetMaxIter(8);

   //x = 0.0;
   newton_solver.Mult(rhs, up);
   chrono.Stop();

   // 11. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u, p;
   FiniteElementSpace *Vspace = new FiniteElementSpace(mesh, fe_coll, dim);
   FiniteElementSpace *Pspace = new FiniteElementSpace(mesh, fe_coll, 1);   
   u.MakeRef(Vspace, up.GetBlock(0), 0);
   p.MakeRef(Pspace, up.GetBlock(1), 0);

   // 12. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("ex5.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream u_ofs("sol_u.gf");
      u_ofs.precision(8);
      u.Save(u_ofs);

      ofstream p_ofs("sol_p.gf");
      p_ofs.precision(8);
      p.Save(p_ofs);
   }

   // 13. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5", mesh);
   visit_dc.RegisterField("velocity", &u);
   visit_dc.RegisterField("pressure", &p);
   visit_dc.Save();

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << *mesh << u << "window_title 'Velocity'" << endl;
      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n" << *mesh << p << "window_title 'Pressure'" << endl;
   }

   // 15. Free the used memory.
   //delete J_prec;
   delete Vspace;
   delete Pspace;
   delete VQspace;
   delete fe_coll;
   delete mesh;

   return 0;
}

NSOperator::NSOperator(FiniteElementSpace* VQ_space_, 
            Array<int>& ess_tdof_list_, Vector& Gravity_, double dyn_visc_, 
            double density_, BlockVector& x_, BlockVector& rhs)
: Operator(VQ_space_->GetVSize()), VQ_space(VQ_space_), GalVarf(VQ_space_), 
  noLinVarf(VQ_space_), fform(VQ_space_), Gravity(Gravity_), dyn_visc(dyn_visc_), 
  density(density), x(x_), Jacobian(NULL), ess_dofs(&ess_tdof_list_)    
{

   //(v,f)
   Vector exGrav(Gravity.Size() + 1); //Expanding for pressure
   exGrav = 0.0;
   for (int kk = 0; kk < Gravity.Size(); ++ kk)
       exGrav[kk] = Gravity[kk];
   
   VectorConstantCoefficient exBdForce(exGrav);    
   fform.AddDomainIntegrator(new VectorDomainLFIntegrator(exBdForce));
   fform.Assemble();

   //Build M, B and Q;  
   // components of the global block matrix       
   ConstantCoefficient nuCoef(dyn_visc);
   GalVarf.AddDomainIntegrator(new VectorGalerkinNSIntegrator(nuCoef));
   GalVarf.Assemble();
   GalVarf.Finalize();

   //Add nonlinear form integrator
   noLinVarf.AddDomainIntegrator(new VectorNonLinearNSIntegrator(exGrav, dyn_visc));  
}

NSOperator::~NSOperator()
{
    delete Jacobian;
}

void NSOperator::Mult(const Vector& acc, Vector &RHS) const
{
   // init 
    int ss = x.Size();
    RHS.SetSize(ss);

    //Add body force related stabilization 
    noLinVarf.Mult(acc, RHS);

    //Add Matrix stabilization
    SparseMatrix *grad_nonLin = dynamic_cast<SparseMatrix *>(&noLinVarf.GetGradient(acc));
    grad_nonLin->AddMult(acc, RHS);
   
    //Add linear terms
    GalVarf.AddMult(acc, RHS);

    for (int kk = 0; kk < ess_dofs->Size(); kk++)
    {
        int ess_ro = (*ess_dofs)[kk];
        RHS[ess_ro] = 0.0;
    }
   
}

Operator &NSOperator::GetGradient(const Vector& acc) const
{    
    delete Jacobian; 
    
    SparseMatrix *grad_nonLin = dynamic_cast<SparseMatrix *>(&noLinVarf.GetGradient(acc));
    //grad_nonLin->Print();

    Jacobian = Add(1.0, GalVarf.SpMat(), 1.0, *grad_nonLin);
    
    for (int kk = 0; kk < ess_dofs->Size(); kk++)
    {
        int ess_ro = (*ess_dofs)[kk];
        Jacobian->EliminateRowCol(ess_ro);
    }

   return *Jacobian;

}