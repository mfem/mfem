#include "test.hpp"
#include "double_integrals.hpp"
#include "gs.hpp"
#include "boundary.hpp"
#include <math.h>

using namespace std;
using namespace mfem;


void test_plasma_point_calculator() {

  cout << "*** test_plasma_point_calculator" << endl;
  
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
  cout << "*** test_read_data_file" << endl;

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
  cout << "*** test_solve" << endl;
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


double M_func(const Vector &x1, const Vector &x2)
{
  return x1[0] + x2[0];
}


double M2_func(const Vector &x1, const Vector &x2)
{
  // if (true) {
  //   return 1.0;
  // }
  double theta, phi;
  if ((x1[0] == 0.0) && (x1[1] == 0.0)) {
    theta = 0.0;
  } else {
    theta = atan2(x1[1], x1[0]);
  }
  if ((x2[0] == 0.0) && (x2[1] == 0.0)) {
    phi = 0.0;
  } else {
    phi = atan2(x2[1], x2[0]);
  }

  // get angle
  return theta + phi;
  // return 1.0;
}

double a_func(const Vector & x)
{
  return x[0];
}
double b_func(const Vector & x)
{
  return x[0];
}

double c_func(const Vector & x)
{
  if ((x[0] == 0.0) && (x[1] == 0.0)) {
    return 0.0;
  }

  // get angle
  double theta = atan2(x[1], x[0]);

  // if (true) {
  //   return theta;
  // }
  double k = 3.2;
  return sin(k * theta) + theta;
}

double d_func(const Vector & x)
{
  if ((x[0] == 0.0) && (x[1] == 0.0)) {
    return 0.0;
  }

  // get angle
  double theta = atan2(x[1], x[0]);

  // if (true) {
  //   return theta;
  // }

  double k = 2.3;
  return cos(k * theta) + theta * theta;
}

void Print__(const Vector &y) {
  double tol = 1e-5;
  for (int i = 0; i < y.Size(); ++i) {
    if (abs(y[i]) > tol) {
      printf("%d %.14e\n", i+1, y[i]);
    }
  }
}

void PrintMatlab_(SparseMatrix *Mat) {

  int *I = Mat->GetI();
  int *J = Mat->GetJ();
  double *A = Mat->GetData();
  int height = Mat->Height();

  double tol = 1e-5;
  
  int i, j;
  for (i = 0; i < height; ++i) {
    for (j = I[i]; j < I[i+1]; ++j) {
      if (abs(A[j]) > tol) {
        printf("i=%d, j=%d, a=%10.3e \n", i, J[j], A[j]);
      }
    }
  }
}

void test_square_boundary_integral()
{
  const char *mesh_file = "meshes/square_with_top.msh";
  int attrib = 7;
  int order = 1;
  Mesh mesh(mesh_file);
  for (int k = 0; k < 1; ++k) {
    mesh.UniformRefinement();
  }
  
  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  BilinearForm a(&fespace);

  a.Assemble();
  DoubleBoundaryBFIntegrator i(M_func);
  AssembleDoubleBoundaryIntegrator(a, i, attrib);
  a.Finalize();

  // define functions to integrate over
  FunctionCoefficient a_coeff(a_func);
  FunctionCoefficient b_coeff(b_func);
  GridFunction a_gf(&fespace);
  GridFunction b_gf(&fespace);

  a_gf.ProjectCoefficient(a_coeff);
  b_gf.ProjectCoefficient(b_coeff);

  mesh.Save("out/03_mesh.mesh");
  a_gf.Save("out/03_a.gf");
  a_gf.Save("out/03_b.gf");

  //
  GridFunction out(&fespace);
  out = 0.0;
  a.Mult(a_gf, out);
  Vector vec;

  double result = 0.0;
  for (int i = 0; i < b_gf.Size(); ++i) {
    result += b_gf[i] * out[i];
  }
  cout << "  " << mesh_file << endl;
  printf("    result: %f\n", result);
  printf("    true:   %f\n", 1.0 / 6.0);

  double TOL = 1e-5;
  assert(abs( result - 1.0 / 6.0 ) < TOL);
}

void test_circle_boundary_integral()
{
  const char *mesh_file = "meshes/test.msh";
  int attrib = 831;
  // const char *mesh_file = "meshes/semicirc.msh";
  // int attrib = 3;
  // const char *mesh_file = "meshes/semicirc_2p5.msh";
  // int attrib = 3;
  int order = 1;
  Mesh mesh(mesh_file);
  for (int k = 0; k < 3; ++k) {
    mesh.UniformRefinement();
  }
  
  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  BilinearForm a(&fespace);

  a.Assemble();
  DoubleBoundaryBFIntegrator i(M2_func);
  AssembleDoubleBoundaryIntegrator(a, i, attrib);
  a.Finalize();

  // define functions to integrate over
  FunctionCoefficient c_coeff(c_func);
  FunctionCoefficient d_coeff(d_func);
  GridFunction c_gf(&fespace);
  GridFunction d_gf(&fespace);

  c_gf.ProjectCoefficient(c_coeff);
  d_gf.ProjectCoefficient(d_coeff);

  mesh.Save("out/04_mesh.mesh");
  c_gf.Save("out/04_c.gf");
  d_gf.Save("out/04_d.gf");

  //
  GridFunction out(&fespace);
  out = 0.0;
  a.Mult(c_gf, out);
  Vector vec;

  double result = 0.0;
  for (int i = 0; i < d_gf.Size(); ++i) {
    result += d_gf[i] * out[i];
  }
  cout << "  " << mesh_file << endl;
  printf("    result: %f\n", result);
  double true_result = -7.17656339762192;
  printf("    true:   %f\n", true_result);
  double TOL = 1e-2;
  assert(abs( result -  true_result) < TOL);
}

void test_double_integral()
{
  cout << "*** test_double_integral" << endl;

  test_square_boundary_integral();
  test_circle_boundary_integral();
}

void test_boundary_coefficients()
{
  cout << "*** test_boundary_coefficients" << endl;
  Vector x(2), y(2);
  x(0) = 2.0;
  x(1) = 3.0;
  y(0) = 1.0;
  y(1) = 2.0;
  double mu = 2.0;
  double rho_Gamma = 2.5;
  double out;
  double true_out;

  printf("  x = (%.f, %.f)\n", x(0), x(1));
  printf("  y = (%.f, %.f)\n", y(0), y(1));
    
  true_out = 0.06398569540400453;
  out = N_coefficient(x, rho_Gamma, mu);
  printf("  N(x) = %.8f ?= (%.8f)\n", out, true_out);
  double TOL = 1e-12;
  assert(abs(out - true_out) < TOL);

  true_out = 0.35567882440928605;
  out = N_coefficient(y, rho_Gamma, mu);
  printf("  N(y) = %.8f ?= (%.8f)\n", out, true_out);
  assert(abs(out - true_out) < TOL);

  true_out = 0.009755140876768808;
  out = M_coefficient(x, y, mu);
  printf("  M(x, y) = %.8f ?= (%.8f)\n", out, true_out);
  assert(abs(out - true_out) < TOL);

  out = M_coefficient(y, x, mu);
  printf("  M(y, x) = %.8f ?= (%.8f)\n", out, true_out);
  assert(abs(out - true_out) < TOL);
  
  true_out = 0.0;
  out = M_coefficient(x, x, mu);
  printf("  M(x, x) = %.8f ?= (%.8f)\n", out, true_out);
  assert(abs(out - true_out) < TOL);

  out = M_coefficient(y, y, mu);
  printf("  M(y, y) = %.8f ?= (%.8f)\n", out, true_out);
  assert(abs(out - true_out) < TOL);

  x(0) = 0.0;
  printf("  x = (%.f, %.f)\n", x(0), x(1));
  assert(abs(out - true_out) < TOL);
    
  out = N_coefficient(x, rho_Gamma, mu);
  printf("  N(x) = %.8f ?= (%.8f)\n", out, true_out);
  assert(abs(out - true_out) < TOL);
  
  out = M_coefficient(x, y, mu);
  printf("  M(x, y) = %.8f ?= (%.8f)\n", out, true_out);
  assert(abs(out - true_out) < TOL);
  
  out = M_coefficient(y, x, mu);
  printf("  M(y, x) = %.8f ?= (%.8f)\n", out, true_out);
  assert(abs(out - true_out) < TOL);
  
  
}


int test()
{
  cout << "*** performing tests..." << endl;
  test_plasma_point_calculator();
  test_read_data_file();
  test_double_integral();
  test_boundary_coefficients();
  // if (true) {
  //   return 1;
  // }
  
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
