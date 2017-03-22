#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

void dir_velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;



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
   FiniteElementSpace fes(mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // Setup bilinear form for x derivative and the mass matrix
   VectorFunctionCoefficient x_dir(dim, dir_velocity_function);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));

   m.Assemble();
   m.Finalize();
   int skip_zeros = 1;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);

   SparseMatrix &K = k.SpMat();
   /////////////////////////////////////////////////////////////

   FunctionCoefficient u0(u0_function);
   GridFunction u_sol(&fes);
   u_sol.ProjectCoefficient(u0);

   GridFunction v(&fes);
   K.Mult(u_sol, v);

   FiniteElementSpace fes_v(mesh, &fec, dim);
   GridFunction u(&fes_v);

   VectorFunctionCoefficient velocity(dim, velocity_function);
   u.ProjectCoefficient(velocity);

   VectorGridFunctionCoefficient u_vec(&u);

   LinearForm b(&fes);
   b.AddFaceIntegrator(
      new DGRiemIntegrator(u_vec, -1, 0));
   b.Assemble();


   // We need a vector finite element space for nodes
   FiniteElementSpace fes_nodes(mesh, &fec, dim);
   // Print all nodes in the finite element space 
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
       int sub1 = i, sub2 = nodes.Size()/dim + i;
       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << (v(sub1) + b[sub1])/m.SpMat().GetRowEntries(sub1)[0] << endl;      
   }


   return 0;
}



// Velocity coefficient
void dir_velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   v(0) = 1.0; v(1) = 0.0; 
}


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   v(0) = 1 + 0.2*sin(M_PI*x(0)); v(1) = 0.0; 
//   v(0) = 1 ; v(1) = 0.0; 
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   const double f = M_PI;
   return 1 + 0.2*sin(f*x(0)) ;
}

