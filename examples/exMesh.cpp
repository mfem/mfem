#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#endif
  
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2.0;
   int nx = 200;
   int ny = 200;
   unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree, nx, ny);
   std::cout <<"Number of elements " << smesh->GetNE() <<'\n';
   std::cout <<"Number of vertices " << smesh->GetNV() <<'\n';
   // Vector node_coord;
   // smesh->GetNodes(node_coord);
   // cout << node_coord.Size() << endl;
   // ofstream mesh_ofs("square_mesh_periodic.mesh");
   // mesh_ofs.precision(14);
   // smesh->Print(mesh_ofs);
   //ofstream sol_ofs("flow_over_ellipse.vtk");
   ofstream sol_ofs("circle_mesh.vtk");
   // sol_ofs.precision(14);
   // smesh->PrintVTK(sol_ofs);
   //ofstream sol_ofs("vortex_mesh.vtk");
   //ofstream sol_ofs("ellipse_mesh.vtk");
   sol_ofs.precision(14);
   smesh->PrintVTK(sol_ofs);
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}
#if 0
/// use this for flow over an ellipse
unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::QUADRILATERAL, true /* gen. edges */,
                                             40.0, 2*M_PI, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   // H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   // FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
   //                                                  Ordering::byVDIM);

   // // This lambda function transforms from (r,\theta) space to (x,y) space
   // auto xy_fun = [](const Vector& rt, Vector &xy)
   // {
   //    // double ax = rt(0);
   //    // double ay = ax/5.0;

   //    // double r = sqrt((ax * ax * ay * ay) / ((ay * ay * cos(rt(1)) * cos(rt(1))) 
   //    //                + (ax * ax * sin(rt(1)) * sin(rt(1)))));
   //    // //r = rt(0);
   //    // xy(0) = r * cos(rt(1)) + 20.0; // need + 20.0 to shift r away from origin
   //    // xy(1) = r * sin(rt(1)) + 20.0 ;
   //    double r_far = 40.0;
   //    // double a0 = 0.5;
   //    // double b0 = a0/5.0;
   //    // double r = rt(0) + 1.0;
   //    // double theta = rt(1);
   //    // double b = b0 + (a0 - b0)*r/(r_far + 1.0);
   //    // xy(0) = a0*r*cos(theta) + 20.0;
   //    // xy(1) = b*r*sin(theta) + 20.0;
   //    double a0 = 0.5;
   //    double b0 = a0 / 10.0;
   //    double delta = 2.00; // We will have to experiment with this
   //    double r = 1.0 + tanh(delta * (rt(0) / r_far - 1.0)) / tanh(delta);
   //    double theta = rt(1);
   //    double b = b0 + (a0 - b0) * r;
   //    xy(0) = a0*(r*r_far + 1.0)*cos(theta) + 20.0;
   //    xy(1) = b*(r*r_far + 1.0)*sin(theta) + 20.0;
   // };
   // VectorFunctionCoefficient xy_coeff(2, xy_fun);
   // GridFunction *xy = new GridFunction(fes);
   // xy->MakeOwner(fec);
   // xy->ProjectCoefficient(xy_coeff);

   // mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
#endif
#if 0
/// try ellipse
unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::QUADRILATERAL, true /* gen. edges */,
                                             0.5, 2*M_PI, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector& rt, Vector &xy)
   {
      double ax = rt(0);
      double ay = rt(0)/5.0;

      double r = sqrt((ax * ax * ay * ay) / ((ay * ay * cos(rt(1)) * cos(rt(1))) 
                     + (ax * ax * sin(rt(1)) * sin(rt(1)))));
      //r = rt(0);
      xy(0) = r*cos(rt(1)) + 20.0; // need + 1.0 to shift r away from origin
      xy(1) = r*sin(rt(1)) + 20.0 ;
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
#endif
#if 0
/// this is for vortex mesh
unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             2.0, M_PI * 0.5, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes

   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector &rt, Vector &xy) {
      xy(0) = (rt(0) + 1.0) * cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = (rt(0) + 1.0) * sin(rt(1));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
#endif
#if 1
/// use this for circle
unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::QUADRILATERAL, true /* gen. edges */,
                                             1.0, 2*M_PI, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector& rt, Vector &xy)
   {
      xy(0) = ((rt(0)+0)*cos(rt(1))) + 2.0; // need + 1.0 to shift r away from origin
      xy(1) = (rt(0)+0)*sin(rt(1)) + 2.0 ;
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
#endif
// 