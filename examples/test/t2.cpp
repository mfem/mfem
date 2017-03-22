#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;

// Velocity coefficient
void init_function(const Vector &x, Vector &v);

void getInvFlux(int dim, const Vector &u, Vector &f);

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &K;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};



int main(int argc, char *argv[])
{
   const char *mesh_file = "periodic-square.mesh";
   int order = 1;
   double t_final = 0.01;
   double dt = 0.01;

   int precision = 8;
   cout.precision(precision);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec, dim + 2);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   VectorFunctionCoefficient u0(dim + 2, init_function);
   GridFunction u_sol(&fes);
   u_sol.ProjectCoefficient(u0);

   GridFunction inv_flux(&fes);
   getInvFlux(dim, u_sol, inv_flux);

   ///////////////////////////////////////////////////////////
   // Setup bilinear form for x derivative and the mass matrix
   Vector dir(dim);
   dir(0) = 1.0; dir(1) = 0.0;
   VectorConstantCoefficient x_dir(dir);

   FiniteElementSpace fes_op(mesh, &fec);
   BilinearForm m(&fes_op);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k(&fes_op);
   k.AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));

   m.Assemble();
   m.Finalize();
   int skip_zeros = 1;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   /////////////////////////////////////////////////////////////
   // Linear form
   LinearForm b(&fes);
//   b.AddFaceIntegrator(
//      new DGRiemIntegrator(u_vec, -1, 0));
   b.Assemble();
   ///////////////////////////////////////////////////////////
   
   FE_Evolution adv(m.SpMat(), k.SpMat(), b);
   ODESolver *ode_solver = new ForwardEulerSolver; 

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   ode_solver->Step(u_sol, t, dt);

   // Print all nodes in the finite element space 
   FiniteElementSpace fes_nodes(mesh, &fec, dim);
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
       int offset = nodes.Size()/dim;
       int sub1 = i, sub2 = offset + i, sub3 = 2*offset + i, sub4 = 3*offset + i;
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub4) << '\t' << inv_flux(sub4) << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << u_sol(sub2) << '\t' << u_sol(sub3) << '\t' << u_sol(sub4) << endl;      
   }


   return 0;
}



// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Size()), M(_M), K(_K), b(_b), z(_M.Size())
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
    y.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    int dim = x.Size()/K.Size() - 2;
    Vector f(x.Size());
    getInvFlux(dim, x, f);

    int offset  = K.Size();
    Array<int> offsets[dim + 2];
    for(int i = 0; i < dim + 2; i++)
    {
        offsets[i].SetSize(offset);
    }

    for(int i = 0; i < offset; i++)
    {
        offsets[0][i] =            i ;
        offsets[1][i] =   offset + i ;
        offsets[2][i] = 2*offset + i ;
        offsets[3][i] = 3*offset + i ;
    }
    Vector f_sol(offset), f_x(offset), f_x_m(offset);
    for(int i = 0; i < dim + 2; i++)
    {
        f.GetSubVector(offsets[i], f_sol);
        K.Mult(f_sol, f_x);
        M_solver.Mult(f_x, f_x_m);
        y.SetSubVector(offsets[i], f_x_m);
    }
}



// Inviscid flux 
void getInvFlux(int dim, const Vector &u, Vector &f)
{
    int size = u.Size();

    int var_dim = dim + 2;
    int offset  = u.Size()/var_dim;

    Array<int> offsets1(offset), offsets2(offset), offsets3(offset), offsets4(offset);

    for(int i = 0; i < offset; i++)
    {
        offsets1[i] =            i ;
        offsets2[i] =   offset + i ;
        offsets3[i] = 2*offset + i ;
        offsets4[i] = 3*offset + i ;

    }
    Vector rho, rho_u1, rho_u2, E;
    u.GetSubVector(offsets1, rho   );
    u.GetSubVector(offsets2, rho_u1);
    u.GetSubVector(offsets3, rho_u2);
    u.GetSubVector(offsets4,      E);

    Vector f1(offset), f2(offset), f3(offset);
    for(int i = 0; i < offset; i++)
    {
        double u1   = rho_u1(i)/rho(i);
        double u2   = rho_u2(i)/rho(i);
        
        double v_sq = pow(u1, 2) + pow(u2, 2);
        double p    = (E(i) - 0.5*rho(i)*v_sq)*(gamm - 1);

        f1(i) = rho_u1(i)*u1 + p; //rho*u*u + p    
        f2(i) = rho_u1(i)*u2    ; //rho*u*v
        f3(i) = (E(i) + p)*u1   ; //(E+p)*u
    }

    f.SetSubVector(offsets1, rho_u1);
    f.SetSubVector(offsets2,     f1);
    f.SetSubVector(offsets3,     f2);
    f.SetSubVector(offsets4,     f3);
}

//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   //Variable dimensions
   int dim_var = v.Size();

   double rho = 1 + 0.2*sin(M_PI*x(0));
   double u1  = 1, u2 = 0.0;
   double p   = 1;

   double v_sq = pow(u1, 2) + pow(u2, 2);

   v(0) = 1 + 0.2*sin(M_PI*x(0));  //rho
   v(1) = rho * u1;                //rho * u
   v(2) = rho * u2;                //rho * v
   v(3) = p/(gamm - 1) + 0.5*rho*v_sq;

//   cout << x(0) << '\t' << x(1) << '\t' << v(0) << '\t' << v(1) << '\t' << v(2) << '\t' << v(3) << endl;      
}


