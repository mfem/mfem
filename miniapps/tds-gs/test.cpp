#include "test.hpp"
#include <math.h>

using namespace std;
using namespace mfem;


int test_plasma_point_calculator() {

  // load mesh and project initial solution
  const char *mesh_file = "meshes/gs_mesh.msh";
  const char *data_file = "separated_file.data";
  int order = 1;
  Mesh mesh(mesh_file);
  mesh.UniformRefinement();
  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  InitialCoefficient init_coeff = read_data_file(data_file);
  GridFunction psi_init(&fespace);
  psi_init.ProjectCoefficient(init_coeff);
  psi_init.Save("psi_init.gf");

  //
  const int attr_lim = 1000;
  map<int, vector<int>> vertex_map = compute_vertex_map(mesh, attr_lim);
  int ind_min, ind_max;
  double min_val, max_val;
  int iprint = 1;
  set<int> plasma_inds;
  compute_plasma_points(psi_init, mesh, vertex_map, plasma_inds, ind_min, ind_max, min_val, max_val, iprint);
  
  init_coeff.SetPlasmaInds(plasma_inds);
  psi_init.ProjectCoefficient(init_coeff);
  psi_init.Save("psi_init_masked.gf");

  return 1;
}



int test()
{
  test_plasma_point_calculator();

  double a = 1.0;
  // double b = S_p_prime(a);
  // cout << "a" << a << "b" << b << endl;
  return 1;
}

double TestCoefficient::Eval(ElementTransformation & T,
                             const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double x1(x(0));
   double x2(x(1));
   double k = 4.0;
   
   return cos(k * x1) * cos(k * x2) * exp(- pow(x1, 2.0) - pow(x2, 2.0));
}
