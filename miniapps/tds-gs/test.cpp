#include "test.hpp"
#include "gs.hpp"
#include <math.h>

using namespace std;
using namespace mfem;


void test_plasma_point_calculator() {

  cout << "test_plasma_point_calculator" << endl;
  
  // load mesh and project initial solution
  // const char *mesh_file = "meshes/gs_mesh.msh";
  // const char *mesh_file = "meshes/test.msh";
  const char *mesh_file = "meshes/test_off_center.msh";
  const char *data_file = "separated_file.data";
  int order = 1;
  Mesh mesh(mesh_file);
  mesh.UniformRefinement();
  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  // InitialCoefficient init_coeff = read_data_file(data_file);
  InitialCoefficient init_coeff = from_manufactured_solution();
  GridFunction psi_init(&fespace);
  psi_init.ProjectCoefficient(init_coeff);
  psi_init.Save("out/01_psi_init.gf");

  // compute plasma points - magnetic axis and x point
  const int attr_lim = 1000;
  map<int, vector<int>> vertex_map = compute_vertex_map(mesh, attr_lim);
  int ind_ma, ind_x;
  double val_ma, val_x;
  int iprint = 1;
  set<int> plasma_inds;
  compute_plasma_points(psi_init, mesh, vertex_map, plasma_inds, ind_ma, ind_x, val_ma, val_x, iprint);

  init_coeff.SetPlasmaInds(plasma_inds);
  psi_init.ProjectCoefficient(init_coeff);
  psi_init.Save("out/01_psi_init_masked.gf");

  mesh.Save("out/01_mesh.mesh");

  // magnetic axis value: 0
  // saddle value: 1
  double TOL = 0.01;
  assert(abs(val_ma - 0.0) < TOL);
  assert(abs(val_x - 1.0) < TOL);

  // magnetic axis at (1, 0)
  const double* b = mesh.GetVertex(ind_ma);
  double bx = b[0];
  double by = b[1];
  assert(abs(bx - 1.0) < TOL);
  assert(abs(by - 0.0) < TOL);

  // saddles at corners of |x-1| + |y| <= 0.35
  const double* c = mesh.GetVertex(ind_x);
  double cx = c[0];
  double cy = c[1];
  assert((((abs(cx - 1.0) < TOL) && (abs(cy + 0.35) < TOL)) ||
          ((abs(cx - 1.0) < TOL) && (abs(cy - 0.35) < TOL)) ||
          ((abs(cx - 0.65) < TOL) && (abs(cy + 0.0) < TOL)) ||
          ((abs(cx - 1.35) < TOL) && (abs(cy + 0.0) < TOL))));

}

void test_read_data_file() {
  cout << "test_read_data_file" << endl;

  // load mesh and project initial solution
  // const char *mesh_file = "meshes/gs_mesh.msh";
  // const char *mesh_file = "meshes/test.msh";
  const char *mesh_file = "meshes/test_off_center.msh";
  const char *data_file = "separated_file.data";
  int order = 1;
  Mesh mesh(mesh_file);
  mesh.UniformRefinement();
  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  InitialCoefficient init_coeff = read_data_file(data_file);
  GridFunction psi_init(&fespace);
  psi_init.ProjectCoefficient(init_coeff);

  psi_init.Save("out/02_psi_init.gf");
  mesh.Save("out/02_mesh.mesh");


}


void test_solve() {
  const char *mesh_file = "meshes/test_off_center.msh";
  const char *data_file = "separated_file.data";
  int order = 1;
  int d_refine = 0;

  vector<double> errors;
  double error;

  for (d_refine = 0; d_refine <= 2; ++d_refine) {
    error = gs(mesh_file, data_file, order, d_refine);
    errors.push_back(error);
  }

  printf("Convergence Table\n");
  for (d_refine = 0; d_refine <= 2; ++d_refine) {
    if (d_refine == 0) {
      printf("%d %.3e\n", d_refine+1, errors[d_refine]);
    } else {
      printf("%d %.3e %.2f\n", d_refine+1, errors[d_refine], errors[d_refine-1] / errors[d_refine]);
    }
    
  }
  
}

int test()
{
  cout << "performing tests..." << endl;
  test_plasma_point_calculator();
  test_read_data_file();
  test_solve();
  

  cout << "tests finished..." << endl << endl << endl;
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
