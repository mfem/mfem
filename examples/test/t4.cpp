#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Problem to solve
int problem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.01;
const double   Pr  = 0.72;

// Velocity coefficient
void init_function(const Vector &x, Vector &v);

void getInvFlux(int dim, const Vector &u, Vector &f);

void getVisFlux(int dim, const Vector &u, const Vector &u_grad, Vector &f);

void getUGrad(int dim, const SparseMatrix &K_x, const SparseMatrix &K_y, const SparseMatrix &m, const Vector &u, Vector &u_grad);

void getFields(const GridFunction &u_sol, GridFunction &rho, GridFunction &u1, GridFunction &u2, GridFunction &E);

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &K_inv_x, &K_inv_y, &K_vis_x, &K_vis_y;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_K_inv_x, SparseMatrix &_K_inv_y, SparseMatrix &_K_vis_x, SparseMatrix &_K_vis_y, const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};



int main(int argc, char *argv[])
{
   const char *mesh_file = "periodic-square.mesh";
   int    order      = 1;
   double t_final    = 0.0010;
   double dt         = 0.0010;
   int    vis_steps  = 25;
   int    ref_levels = 0;

          problem    = 2;

   int precision = 8;
   cout.precision(precision);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int     dim = mesh->Dimension();
   int var_dim = dim + 2;

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec, var_dim);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   VectorFunctionCoefficient u0(var_dim, init_function);
   GridFunction u_sol(&fes);
   u_sol.ProjectCoefficient(u0);

   FiniteElementSpace fes_vec(mesh, &fec, dim*var_dim);
   GridFunction f_inv(&fes_vec);
   getInvFlux(dim, u_sol, f_inv);

   ///////////////////////////////////////////////////////////
   // Setup bilinear form for x derivative and the mass matrix
   Vector dir(dim);
   dir(0) = 1.0; dir(1) = 0.0;
   VectorConstantCoefficient x_dir(dir);
   dir(0) = 0.0; dir(1) = 1.0;
   VectorConstantCoefficient y_dir(dir);

   FiniteElementSpace fes_op(mesh, &fec);
   BilinearForm m(&fes_op);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k_inv_x(&fes_op);
   k_inv_x.AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));
   BilinearForm k_inv_y(&fes_op);
   k_inv_y.AddDomainIntegrator(new ConvectionIntegrator(y_dir, -1.0));

   m.Assemble();
   m.Finalize();
   int skip_zeros = 1;
   k_inv_x.Assemble(skip_zeros);
   k_inv_x.Finalize(skip_zeros);
   k_inv_y.Assemble(skip_zeros);
   k_inv_y.Finalize(skip_zeros);
   /////////////////////////////////////////////////////////////
   
   VectorGridFunctionCoefficient u_vec(&u_sol);
   VectorGridFunctionCoefficient f_vec(&f_inv);

   /////////////////////////////////////////////////////////////
   // Linear form
   LinearForm b(&fes);
   b.AddFaceIntegrator(
      new DGEulerIntegrator(u_vec, f_vec, var_dim, -1.0));
   ///////////////////////////////////////////////////////////
   //Creat vsicous derivative matrices
   BilinearForm k_vis_x(&fes_op);
   k_vis_x.AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));
   k_vis_x.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(x_dir, 1.0,  0.0)));// Beta 0 means central flux

   BilinearForm k_vis_y(&fes_op);
   k_vis_y.AddDomainIntegrator(new ConvectionIntegrator(y_dir, -1.0));
   k_vis_y.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(y_dir, 1.0,  0.0)));// Beta 0 means central flux

   k_vis_x.Assemble(skip_zeros);
   k_vis_x.Finalize(skip_zeros);
   k_vis_y.Assemble(skip_zeros);
   k_vis_y.Finalize(skip_zeros);

   SparseMatrix &K_vis_x = k_vis_x.SpMat();
   SparseMatrix &K_vis_y = k_vis_y.SpMat();
   SparseMatrix &M   = m.SpMat();
   ////////////////////////////////////////

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   
   dc = new VisItDataCollection("Euler", mesh);
   dc->SetPrecision(precision);
   
   FiniteElementSpace fes_fields(mesh, &fec);
   GridFunction rho(&fes_fields);
   GridFunction u1(&fes_fields);
   GridFunction u2(&fes_fields);
   GridFunction E(&fes_fields);

   dc->RegisterField("rho", &rho);
   dc->RegisterField("u1", &u1);
   dc->RegisterField("u2", &u2);
   dc->RegisterField("E", &E);

   getFields(u_sol, rho, u1, u2, E);

   dc->SetCycle(0);
   dc->SetTime(0.0);
   dc->Save();


   FE_Evolution adv(m.SpMat(), k_inv_x.SpMat(), k_inv_y.SpMat(), K_vis_x, K_vis_y, b);
   ODESolver *ode_solver = new ForwardEulerSolver; 

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      b.Assemble();

      double dt_real = min(dt, t_final - t);
      ode_solver->Step(u_sol, t, dt_real);
      ti++;

      cout << "time step: " << ti << ", time: " << t << ", max_sol: " << u_sol.Max() << endl;

      getInvFlux(dim, u_sol, f_inv); // To update f_vec

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
          getFields(u_sol, rho, u1, u2, E);

          dc->SetCycle(ti);
          dc->SetTime(t);
          dc->Save();
      }
   }
  
   // Print all nodes in the finite element space 
   FiniteElementSpace fes_nodes(mesh, &fec, dim);
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
//       int offset = nodes.Size()/dim;
//       int sub1 = i, sub2 = offset + i, sub3 = 2*offset + i, sub4 = 3*offset + i;
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << b[sub4] << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << inv_flux(sub1) << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << u_sol(sub2) << '\t' << u_sol(sub3) << '\t' << u_sol(sub4) << endl;      
   }


   return 0;
}



// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &_M, SparseMatrix &_K_inv_x, SparseMatrix &_K_inv_y, SparseMatrix &_K_vis_x, SparseMatrix &_K_vis_y, const Vector &_b)
   : TimeDependentOperator(_M.Size()), M(_M), K_inv_x(_K_inv_x), K_inv_y(_K_inv_y), K_vis_x(_K_vis_x), K_vis_y(_K_vis_y), b(_b), z(_M.Size())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
    int dim = x.Size()/K_inv_x.Size() - 2;
    int var_dim = dim + 2;

    y.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    Vector y_temp;
    y_temp.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    Vector f(dim*x.Size());
    getInvFlux(dim, x, f);

    int offset  = K_inv_x.Size();
    Array<int> offsets[dim*var_dim];
    for(int i = 0; i < dim*var_dim; i++)
    {
        offsets[i].SetSize(offset);
    }

    for(int j = 0; j < dim*var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = j*offset + i ;
        }
    }

    Vector f_sol(offset), f_x(offset), f_x_m(offset);
    Vector b_sub(offset);
    y = 0.0;
    for(int i = 0; i < var_dim; i++)
    {
        f.GetSubVector(offsets[i], f_sol);
        K_inv_x.Mult(f_sol, f_x);
        b.GetSubVector(offsets[i], b_sub);
        f_x += b_sub; // Needs to be added only once
        M_solver.Mult(f_x, f_x_m);
        y_temp.SetSubVector(offsets[i], f_x_m);
    }
    y += y_temp;
    for(int i = var_dim + 0; i < 2*var_dim; i++)
    {
        f.GetSubVector(offsets[i], f_sol);
        K_inv_y.Mult(f_sol, f_x);
        M_solver.Mult(f_x, f_x_m);
        y_temp.SetSubVector(offsets[i - var_dim], f_x_m);
    }
    y += y_temp;

    //////////////////////////////////////////////
    //Get viscous contribution

    Vector u_grad(dim*x.Size()), f_vis(dim*x.Size());
    getUGrad(dim, K_vis_x, K_vis_y, M, x, u_grad);
    getVisFlux(dim, x, u_grad, f_vis);

    for(int i = 0; i < var_dim; i++)
    {
        f_vis.GetSubVector(offsets[i], f_sol);
        K_vis_x.Mult(f_sol, f_x);
        M_solver.Mult(f_x, f_x_m);
        y_temp.SetSubVector(offsets[i], f_x_m);
    }
    y += y_temp;
    for(int i = var_dim + 0; i < 2*var_dim; i++)
    {
        f_vis.GetSubVector(offsets[i], f_sol);
        K_vis_y.Mult(f_sol, f_x);
        M_solver.Mult(f_x, f_x_m);
        y_temp.SetSubVector(offsets[i - var_dim], f_x_m);
    }
    y += y_temp;


//    for (int j = 0; j < offset; j++) cout << x(offset + j) << '\t'<< f(var_dim*offset + offset + j) << endl;
//    for (int j = 0; j < offset; j++) cout << j << '\t'<< b(j) << endl;
//    for (int j = 0; j < offset; j++) cout << x(offset + j) << '\t'<< f(var_dim*offset + 3*offset + j) << endl;
//    for (int j = 0; j < offset; j++) cout << x(offset + j) << '\t'<< y(1*offset + j) << endl;

}



// Inviscid flux 
void getInvFlux(int dim, const Vector &u, Vector &f)
{
    int var_dim = dim + 2;
    int offset  = u.Size()/var_dim;

    Array<int> offsets[dim*var_dim];
    for(int i = 0; i < dim*var_dim; i++)
    {
        offsets[i].SetSize(offset);
    }

    for(int j = 0; j < dim*var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = j*offset + i ;
        }
    }
    Vector rho, rho_u1, rho_u2, E;
    u.GetSubVector(offsets[0], rho   );
    u.GetSubVector(offsets[1], rho_u1);
    u.GetSubVector(offsets[2], rho_u2);
    u.GetSubVector(offsets[3],      E);

    Vector f1(offset), f2(offset), f3(offset);
    Vector g1(offset), g2(offset), g3(offset);
    for(int i = 0; i < offset; i++)
    {
        double u1   = rho_u1(i)/rho(i);
        double u2   = rho_u2(i)/rho(i);
        
        double v_sq = pow(u1, 2) + pow(u2, 2);
        double p    = (E(i) - 0.5*rho(i)*v_sq)*(gamm - 1);

        f1(i) = rho_u1(i)*u1 + p; //rho*u*u + p    
        f2(i) = rho_u1(i)*u2    ; //rho*u*v
        f3(i) = (E(i) + p)*u1   ; //(E+p)*u
        
        g1(i) = rho_u2(i)*u1    ; //rho*u*v 
        g2(i) = rho_u2(i)*u2 + p; //rho*v*v + p
        g3(i) = (E(i) + p)*u2   ; //(E+p)*v
    }

    f.SetSubVector(offsets[0], rho_u1);
    f.SetSubVector(offsets[1],     f1);
    f.SetSubVector(offsets[2],     f2);
    f.SetSubVector(offsets[3],     f3);

    f.SetSubVector(offsets[4], rho_u2);
    f.SetSubVector(offsets[5],     g1);
    f.SetSubVector(offsets[6],     g2);
    f.SetSubVector(offsets[7],     g3);

}


// Get gradient of primitive variable 
void getUGrad(int dim, const SparseMatrix &K_x, const SparseMatrix &K_y, const SparseMatrix &M, const Vector &u, Vector &u_grad)
{
    CGSolver M_solver;

    M_solver.SetOperator(M);
    M_solver.iterative_mode = false;

    int var_dim = dim + 2; 
    int offset = u.Size()/var_dim;

    Array<int> offsets[dim*var_dim];
    for(int i = 0; i < dim*var_dim; i++)
    {
        offsets[i].SetSize(offset);
    }

    for(int j = 0; j < dim*var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = j*offset + i ;
        }
    }

    u_grad = 0.0;

    Vector u_sol(offset), u_x(offset);
    for(int i = 0; i < var_dim; i++)
    {
        u.GetSubVector(offsets[i], u_sol);

        K_x.Mult(u_sol, u_x);
        M_solver.Mult(u_x, u_x);
        u_grad.SetSubVector(offsets[          i], u_x);
        
        K_y.Mult(u_sol, u_x);
        M_solver.Mult(u_x, u_x);
        u_grad.SetSubVector(offsets[var_dim + i], u_x);
    }
}


// Viscous flux 
void getVisFlux(int dim, const Vector &u, const Vector &u_grad, Vector &f)
{
    int var_dim = dim + 2;
    int offset  = u.Size()/var_dim;

    Array<int> offsets[dim*var_dim];
    for(int i = 0; i < dim*var_dim; i++)
    {
        offsets[i].SetSize(offset);
    }

    for(int j = 0; j < dim*var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = j*offset + i ;
        }
    }

    Vector rho(offset), rho_u1(offset), rho_u2(offset), E(offset);
    u.GetSubVector(offsets[0], rho   );
    u.GetSubVector(offsets[1], rho_u1);
    u.GetSubVector(offsets[2], rho_u2);
    u.GetSubVector(offsets[3],      E);

    for(int i = 0; i < offset; i++)
    {
        double    u1  = rho_u1(i)/rho(i);
        double    u2  = rho_u2(i)/rho(i);

        double v_sq   = pow(u1, 2) + pow(u2, 2);

        double rho_x  = u_grad(i);
        double rho_y  = u_grad(var_dim*offset + i);

        double rhou_x = u_grad(offset + i);
        double rhou_y = u_grad(var_dim*offset + offset + i);

        double rhov_x = u_grad(2*offset + i);
        double rhov_y = u_grad(var_dim*offset + 2*offset + i);
        
        double E_x    = u_grad(3*offset + i);
        double E_y    = u_grad(var_dim*offset + 3*offset + i);

        double u_x    = (rhou_x - rho_x*u1)/rho(i);
        double u_y    = (rhou_y - rho_y*u1)/rho(i);

        double v_x    = (rhov_x - rho_x*u2)/rho(i);
        double v_y    = (rhov_y - rho_y*u2)/rho(i);

        double div    = u_x + v_y; 
        double tauxx  = 2.0*mu*(u_x - div/3.0); 
        double tauxy  =     mu*(u_y + v_x    ); 

        double tauyx  =     tauxy          ; 
        double tauyy  = 2.0*mu*(v_y - div/3.0); 

        double ke     = 0.5*rho(i)*v_sq; 
        double inte   = (E(i) - ke)/rho(i);

        double ke_x   = 0.5*(v_sq*rho_x) + rho(i)*(u_x*u1 + v_x*u2);
        double ke_y   = 0.5*(v_sq*rho_y) + rho(i)*(u_y*u1 + v_y*u2);

        double inte_x = (E_x - ke_x - rho_x*inte)/rho(i);
        double inte_y = (E_y - ke_y - rho_y*inte)/rho(i);

        f(i           )       = 0.0;
        f(  offset + i)       = tauxx; 
        f(2*offset + i)       = tauxy; 
        f(3*offset + i)       = u1*tauxx + u2*tauxy + (mu/Pr)*gamm*inte_x; 

        f(var_dim*offset            + i) = 0.0;
        f(var_dim*offset +   offset + i) = tauyx; 
        f(var_dim*offset + 2*offset + i) = tauyy; 
        f(var_dim*offset + 3*offset + i) = u1*tauxy + u2*tauyy + (mu/Pr)*gamm*inte_y; 
    }

//    for (int j = 0; j < offset; j++) cout << u(offset + j) << '\t'<< f(var_dim*offset + 3*offset + j) << endl;
//    for (int j = 0; j < offset; j++) cout << u(offset + j) << '\t'<< u_x << endl;

}



//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       double rho, u1, u2, p;
       if (problem == 0)
       {
           rho = 1 + 0.2*sin(M_PI*(x(0) + x(1)));
           u1  = 1.0; u2 =-0.5;
           p   = 1;
       }
       else if (problem == 1)
       {
           if (x(0) < 0.0)
           {
               rho = 1.0; 
               u1  = 0.0; u2 = 0.0;
               p   = 1;
           }
           else
           {
               rho = 0.125;
               u1  = 0.0; u2 = 0.0;
               p   = 0.1;
           }
       }
       else if (problem == 2) //Taylor Green Vortex
       {
           rho =  1.0;
           p   =  100 + rho/16.0*(cos(2.0*M_PI*x(0)) + cos(2.0*M_PI*x(1)))*(3.0);
           u1  =      sin(M_PI*x(0)/2.)*cos(M_PI*x(1)/2.)/rho;
           u2  = -1.0*cos(M_PI*x(0)/2.)*sin(M_PI*x(1)/2.)/rho;
       }

    
       double v_sq = pow(u1, 2) + pow(u2, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
   }
}


void getFields(const GridFunction &u_sol, GridFunction &rho, GridFunction &u1, GridFunction &u2, GridFunction &E)
{

    int vDim  = u_sol.VectorDim();
    int dofs  = u_sol.Size()/vDim;

    for (int i = 0; i < dofs; i++)
    {
        rho[i] = u_sol[         i];        
        u1 [i] = u_sol[  dofs + i]/rho[i];        
        u2 [i] = u_sol[2*dofs + i]/rho[i];        
        E  [i] = u_sol[3*dofs + i];        
    }
}
