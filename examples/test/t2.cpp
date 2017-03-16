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
void dir_velocity_function(const Vector &x, Vector &v);

void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;



int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "periodic-square.mesh";
   int order = 1;
   double t_final = 0.02;
   double dt = 0.01;
   bool visit = true;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   VectorFunctionCoefficient x_dir(dim, dir_velocity_function);
   VectorFunctionCoefficient velocity(dim, velocity_function);

   FiniteElementSpace fes_v(mesh, &fec, dim);
   GridFunction u(&fes_v);
   u.ProjectCoefficient(velocity);

   VectorGridFunctionCoefficient u_vec(&u);

   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));

   k.Assemble();
   k.Finalize();

   LinearForm b(&fes);
   b.AddFaceIntegrator(
      new DGRiemIntegrator(u_vec, 1, 1));
   b.Assemble();


   SparseMatrix K = k.SpMat();

   GridFunction v(&fes);
   K.Mult(u, v);

////   K.Print();

   // We need a vector finite element space for nodes
   FiniteElementSpace fes_nodes(mesh, &fec, dim);
   // Print all nodes in the finite element space 
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
       int sub1 = i, sub2 = nodes.Size()/dim + i;
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u(sub1) << '\t'<< 9*v(sub1) << endl;   
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << b[sub1] << endl;   
   }


   return 0;
}


// Velocity coefficient
void dir_velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

//   v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); 
   v(0) = 1.0; v(1) = 0.0; 
}


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   v(0) = 1 + 0.2*sin(M_PI*x(0)); v(1) = 0.0; 
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   double fn = cos(M_PI*X(0))*sin(M_PI*X(1));

//   cout << x(0) << '\t' << x(1) << '\t' << fn << endl;

   return fn;
}


