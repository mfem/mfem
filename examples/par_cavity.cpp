//                                MFEM Example parallel Cavity
//
// Compile with: make cavity
//
// Sample runs:  mpirun -np 4 par_cavity -m ../data/cavity_2d.mesh -re 10
//
// Description:  This is the parallel version of Cavity example.
//               This example code solves a simple 2D/3D mixed Navier-Stokes 
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
//#include <memory>

using namespace std;
using namespace mfem;


class NSOperator : public Operator
{
public:
    NSOperator(ParFiniteElementSpace* VQ_space_, Array<int>& ess_vel_bdr_, 
            Vector& Gravity_, double dyn_visc_, double density_, 
            BlockVector& x_, BlockVector& rhs_);
    
    //Compute the residual vector      
    virtual void Mult(const Vector& acc, Vector &RHS) const;
    
    //Compute the LHS matrix
    virtual Operator &GetGradient(const Vector& acc) const;

    virtual ~NSOperator();
    
private:
    mutable HypreParMatrix *Jacobian;
    ParFiniteElementSpace *VQ_space;
    double dyn_visc, density;
    Vector Gravity;
    Vector x;
    ParBilinearForm *GalVarf;
    ParNonlinearForm noLinVarf;
    ParLinearForm fform;
    Array<int> *ess_dofs;
};

int main(int argc, char *argv[])
{
    //initialize mpi
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
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
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

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
      int ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   
   //Build parallel Mesh
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
       int par_ref_levels = 1;
        for (int kk = 0; kk < par_ref_levels; kk++)
        {
            pmesh->UniformRefinement();
        }
   }
       
   // 4. Define a finite element space on the mesh. Here we use the
   //    equal order H1 finite elements of the specified order.
   FiniteElementCollection *fe_coll(new H1_FECollection(order, dim));
   //FiniteElementCollection *fe_coll(new QuadraticFECollection());
   //FiniteElementCollection *fe_coll(new CubicFECollection());
   //FiniteElementCollection *l2_coll(new H1_FECollection(order, dim));

   //1 is added for pressure; u v (w) p
   ParFiniteElementSpace *VQspace = new ParFiniteElementSpace(pmesh, fe_coll, dim+1); 

   // 5. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   HYPRE_Int glob_size = VQspace->GlobalTrueVSize();
   if (myid == 0)
   {
       cout << " Number of velocity/pressure unknowns : " << glob_size <<endl;
   }
   
   //Local and true local offsets and vectors
   int d1_size = VQspace->GetNDofs();
   Array<int> serial_offsets(3); // number of variables + 1
   serial_offsets[0] = 0;
   serial_offsets[1] = dim * d1_size;
   serial_offsets[2] = (dim + 1) * d1_size;   
   
   BlockVector serial_up(serial_offsets);
   BlockVector serial_rhs(serial_offsets);   
   
   int true_size = VQspace->TrueVSize();
   Array<int> fe_offsets(3); // number of variables + 1
   fe_offsets[0] = 0;
   fe_offsets[1] = dim * true_size / (dim + 1);
   fe_offsets[2] = true_size;//(dim + 1) * d1_size;

   //Solution and rhs
   BlockVector up(fe_offsets);
   BlockVector rhs(fe_offsets);
   
   up  = 0.0;   
   rhs = 0.0;
   
   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0);
   double dyn_visc = 1./re;;
   double density = 1.0;
   ConstantCoefficient nuCoef(dyn_visc);
   
   // Determine the list of essential true dofs
   int loc_attr_size = (VQspace->GetMesh())->bdr_attributes.Max();
   int bdr_attr_size;
   MPI_Allreduce(&loc_attr_size, &bdr_attr_size, 1, MPI_INT, MPI_MAX, 
                        MPI_COMM_WORLD);

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
 
   HypreSmoother *J_hypreSmoother = new HypreSmoother;
   J_hypreSmoother->SetType(HypreSmoother::l1GStr);
   J_hypreSmoother->SetPositiveDiagonal(true);
 
   
   BiCGSTABSolver *J_solver = new BiCGSTABSolver(VQspace->GetComm());//BiCGSTABSolver solver;//MINRESSolver solver;FGMRESSolver
   J_solver->SetPrintLevel(-1);
   J_solver->SetRelTol(rtol);
   J_solver->SetAbsTol(0.0);
   J_solver->SetMaxIter(maxIter);
   J_solver->SetPreconditioner(*J_hypreSmoother);
           
           
   NewtonSolver newton_solver(VQspace->GetComm());
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(NSoper); //StokesMatrix
   newton_solver.SetPrintLevel(1); // print Newton iterations
   newton_solver.SetRelTol(rtol);
   newton_solver.SetAbsTol(0.0);
   newton_solver.SetMaxIter(10);

   newton_solver.Mult(rhs, up);


   // 11. Create the grid functions u and p. Compute the L2 error norms.
   ParGridFunction *u(new ParGridFunction);
   ParGridFunction *p(new ParGridFunction);
   ParFiniteElementSpace *Vspace = new ParFiniteElementSpace(pmesh, fe_coll, dim);
   ParFiniteElementSpace *Pspace = new ParFiniteElementSpace(pmesh, fe_coll, 1);   
   u->MakeRef(Vspace, serial_up.GetBlock(0), 0);
   p->MakeRef(Pspace, serial_up.GetBlock(1), 0);   
   u->Distribute(&(up.GetBlock(0)));
   p->Distribute(&(up.GetBlock(1)));
   
   
   // 12. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   {
      ostringstream mesh_name, u_name, p_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      u_name << "sol_u." << setfill('0') << setw(6) << myid;
      p_name << "sol_p." << setfill('0') << setw(6) << myid;
      
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream u_ofs(u_name.str().c_str());
      u_ofs.precision(8);
      u->Save(u_ofs);

      ofstream p_ofs(p_name.str().c_str());
      p_ofs.precision(8);
      p->Save(p_ofs);      

   }

   // 13. Save data in the VisIt format
   VisItDataCollection visit_dc("Parallel_Cavity2D", pmesh);
   visit_dc.RegisterField("velocity", u);
   visit_dc.RegisterField("pressure", p);
   visit_dc.Save();

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << *u << "window_title 'Velocity'"
             << endl;
      
      MPI_Barrier(pmesh->GetComm());
      
      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << num_procs << " " << myid <<"\n";
      p_sock.precision(8);
      p_sock << "solution\n" << *pmesh << *p << "window_title 'Pressure'" << endl;
        
   }

   // 15. Free the used memory.
   delete u;
   delete p;
   delete Vspace;
   delete Pspace;
   delete VQspace;
   delete fe_coll;
   delete J_solver;
   delete J_hypreSmoother;
   delete pmesh;

   MPI_Finalize();
   
   return 0;
}

NSOperator::NSOperator(ParFiniteElementSpace* VQ_space_, 
            Array<int>& ess_tdof_list_, Vector& Gravity_, double dyn_visc_, 
            double density_, BlockVector& x_, BlockVector& rhs)
: Operator(VQ_space_->TrueVSize()), VQ_space(VQ_space_),  
  noLinVarf(VQ_space_), fform(VQ_space_), Gravity(Gravity_), dyn_visc(dyn_visc_), 
  density(density), x(x_), Jacobian(NULL), ess_dofs(&ess_tdof_list_)    
{   
    GalVarf = new ParBilinearForm(VQ_space);
   //(v,f)
   Vector exGrav(Gravity.Size() + 1); //Expanding for pressure
   exGrav = 0.0;
   for (int kk = 0; kk < Gravity.Size(); ++ kk)
       exGrav[kk] = Gravity[kk];
   
   VectorConstantCoefficient exBdForce(exGrav);    
   fform.AddDomainIntegrator(new VectorDomainLFIntegrator(exBdForce));
   fform.Assemble();
   fform.ParallelAssemble(rhs);

   //Build M, B and Q;  
   // components of the global block matrix       
   ConstantCoefficient nuCoef(dyn_visc);
   GalVarf->AddDomainIntegrator(new VectorGalerkinNSIntegrator(nuCoef));
   GalVarf->Assemble();
   GalVarf->Finalize();
   
   //Add nonlinear form integrator
   noLinVarf.AddDomainIntegrator(new VectorNonLinearNSIntegrator(exGrav, dyn_visc)); 
   
}

NSOperator::~NSOperator()
{
    delete Jacobian;
    delete GalVarf;
}

void NSOperator::Mult(const Vector& acc, Vector &RHS) const
{
   // init 
    int ss = acc.Size();
    RHS.SetSize(ss);

    //Add body force related stabilization 
    noLinVarf.Mult(acc, RHS);

    //Add Matrix stabilization
    SparseMatrix grad_nonLin = noLinVarf.GetLocalGradient(acc);
    ParGridFunction X,Y;
    X.SetSpace(VQ_space);
    Y.SetSpace(VQ_space);
    X.Distribute(&acc);
    grad_nonLin.Mult(X,Y);
    VQ_space->Dof_TrueDof_Matrix()->MultTranspose(1.0, Y, 1.0, RHS);
 
    //Add linear terms
    GalVarf->TrueAddMult(acc, RHS);
     
    for (int kk = 0; kk < ess_dofs->Size(); kk++)
    {
        int ess_ro = (*ess_dofs)[kk];
        RHS[ess_ro] = 0.0;
    }
 
}

Operator &NSOperator::GetGradient(const Vector& acc) const
{    
    delete Jacobian; 
    
    SparseMatrix grad_nonLin = noLinVarf.GetLocalGradient(acc);

    SparseMatrix *localJ =  Add(1.0, GalVarf->SpMat(), 1.0, grad_nonLin);

    Jacobian = GalVarf->ParallelAssemble(localJ);   
    
    Jacobian->EliminateRowsCols(*ess_dofs);

    return *Jacobian;

}