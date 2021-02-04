/// Solve the steady isentropic vortex problem on a quarter annulus

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "euler_integ.hpp"
#include "evolver.hpp"
using namespace std;
using namespace mfem;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);
static std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);
const double rho = 0.9856566615165173;
const double rhoe = 2.061597236955558;
const double rhou[3] = {0.09595562550099601, -0.030658751626551423, -0.13471469906596886};

template <int dim>
void randBaselinePert(const mfem::Vector &x, mfem::Vector &u)
{

   const double scale = 0.01;
   u(0) = rho * (1.0 + scale * uniform_rand(gen));
   u(dim + 1) = rhoe * (1.0 + scale * uniform_rand(gen));
   for (int di = 0; di < dim; ++di)
   {
      u(di + 1) = rhou[di] * (1.0 + scale * uniform_rand(gen));
   }
}

double calcDrag(mfem::FiniteElementSpace *fes, mfem::GridFunction u,
                int num_state, double alpha)
{
   /// check initial drag value
   mfem::Vector drag_dir(2);

   drag_dir = 0.0;
   int iroll = 0;
   int ipitch = 1;
   double aoa_fs = 0.0;
   double mach_fs = 1.0;

   drag_dir(iroll) = cos(aoa_fs);
   drag_dir(ipitch) = sin(aoa_fs);
   drag_dir *= 1.0 / pow(mach_fs, 2.0); // to get non-dimensional Cd

   Array<int> bndry_marker_drag;
   bndry_marker_drag.Append(0);
   bndry_marker_drag.Append(0);
   bndry_marker_drag.Append(0);
   bndry_marker_drag.Append(1);
   NonlinearForm *dragf = new NonlinearForm(fes);

   dragf->AddBdrFaceIntegrator(
       new PressureForce<2, entvar>(drag_dir, num_state, alpha),
       bndry_marker_drag);

   double drag = dragf->GetEnergy(u);
   return drag;
}

/// function to calculate conservative variables l2error
template <int dim, bool entvar>
double calcConservativeVarsL2Error(
    void (*u_exact)(const mfem::Vector &, mfem::Vector &), GridFunction *u, mfem::FiniteElementSpace *fes,
    int num_state, int entry)
{
   // This lambda function computes the error at a node
   // Beware: this is not particularly efficient, given the conditionals
   // Also **NOT thread safe!**
   Vector qdiscrete(dim + 2), qexact(dim + 2); // define here to avoid reallocation
   auto node_error = [&](const Vector &discrete, const Vector &exact) -> double {
      if (entvar)
      {
         calcConservativeVars<dim>(discrete.GetData(),
                                   qdiscrete.GetData());
         calcConservativeVars<dim>(exact.GetData(), qexact.GetData());
      }
      else
      {
         qdiscrete = discrete;
         qexact = exact;
      }
      double err = 0.0;
      if (entry < 0)
      {
         for (int i = 0; i < dim + 2; ++i)
         {
            double dq = qdiscrete(i) - qexact(i);
            err += dq * dq;
         }
      }
      else
      {
         err = qdiscrete(entry) - qexact(entry);
         err = err * err;
      }
      return err;
   };

   VectorFunctionCoefficient exsol(num_state, u_exact);
   DenseMatrix vals, exact_vals;
   Vector u_j, exsol_j;
   double loc_norm = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      const FiniteElement *fe = fes->GetFE(i);
      const IntegrationRule *ir;

      int intorder = 2 * fe->GetOrder() + 3;
      ir = &(IntRules.Get(fe->GetGeomType(), intorder));

      ElementTransformation *T = fes->GetElementTransformation(i);
      u->GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         vals.GetColumnReference(j, u_j);
         exact_vals.GetColumnReference(j, exsol_j);
         loc_norm += ip.weight * T->Weight() * node_error(u_j, exsol_j);
      }
   }

   double norm = loc_norm;
   if (norm < 0.0) // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

void randState(const mfem::Vector &x, mfem::Vector &u)
{
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = 2.0 * uniform_rand(gen) - 1.0;
   }
}

double calcResidualNorm(NonlinearForm *res, FiniteElementSpace *fes, GridFunction &u)
{
   GridFunction residual(fes);
   residual = 0.0;
   res->Mult(u, residual);
   return residual.Norml2();
}

/// get freestream state values for the far-field bcs
template <int dim, bool entvar>
void getFreeStreamState(mfem::Vector &q_ref)
{
   double mach_fs = 0.3;
   q_ref = 0.0;
   q_ref(0) = 1.0;
   q_ref(1) = q_ref(0) * mach_fs; // ignore angle of attack
   q_ref(2) = 0.0;
   q_ref(dim + 1) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
}

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector &p);

/// \brief Returns the value of the integrated math entropy over the domain
double calcEntropyTotalExact();

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &u);

/// Generate quarter annulus mesh
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

/// main
int main(int argc, char *argv[])
{
   // Parse command-line options
   const char *mesh_file = "periodic_ellipse_square.mesh";
   OptionsParser args(argc, argv);
   int nx = 2;
   int ny = 3;
   int order = 1;
   int degree = 2;
   int ref_levels = -1;
   int nc_ref = -1;
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "refine levels");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   /// number of state variables
   int num_state = 4;
   // construct the mesh
   unique_ptr<Mesh> mesh = buildQuarterAnnulusMesh(degree, nx, ny);
   cout << "Number of elements " << mesh->GetNE() << '\n';
   /// dimension
   const int dim = mesh->Dimension();
   /// refine the mesh if needed
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }
   cout << "Number of elements after refinement " << mesh->GetNE() << '\n';
   cout << "#boundary elements " << mesh->GetNBE() << endl;
   cout << "#bdr attributes " << mesh->bdr_attributes.Size() << endl;

   // save the initial mesh
   ofstream mesh_ofs("ellipse.mesh");
   mesh_ofs.precision(14);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("ellipse_mesh_non_cut.vtk");
   sol_ofs.precision(14);
   mesh->PrintVTK(sol_ofs, 1);
   sol_ofs.close();

   /// finite element collection
   FiniteElementCollection *fec = new DG_FECollection(order, dim);

   /// finite element space
   FiniteElementSpace *fes = new FiniteElementSpace(mesh.get(), fec, num_state,
                                                    Ordering::byVDIM);
   cout << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;

   /// `bndry_marker_*` lists the boundaries associated with a particular BC
   Array<int> bndry_marker_slipwall;
   Array<int> bndry_marker_farfield;

   bndry_marker_farfield.Append(0);
   bndry_marker_farfield.Append(1);
   bndry_marker_farfield.Append(0);
   bndry_marker_farfield.Append(1);

   bndry_marker_slipwall.Append(0);
   bndry_marker_slipwall.Append(0);
   bndry_marker_slipwall.Append(0);
   bndry_marker_slipwall.Append(1);

   Vector qfs(dim + 2);

   getFreeStreamState<2, 0>(qfs);
   double alpha = 1.0;

   /// nonlinearform
   NonlinearForm *res = new NonlinearForm(fes);
   res->AddDomainIntegrator(new EulerDomainIntegrator<2>(num_state, alpha));
   res->AddBdrFaceIntegrator(new EulerBoundaryIntegrator<2, 3, 0>(fec, num_state, qfs, alpha),
                             bndry_marker_farfield);
   // res->AddBdrFaceIntegrator(new EulerBoundaryIntegrator<2, 3, 0>(fec, num_state, qfs, alpha),
   //                           bndry_marker_slipwall);
   res->AddInteriorFaceIntegrator(new EulerFaceIntegrator<2>(fec, 1.0, num_state, alpha));

   /// bilinear form
   BilinearForm *mass = new BilinearForm(fes);

   // set up the mass matrix
   mass->AddDomainIntegrator(new EulerMassIntegrator(num_state));
   mass->Assemble();
   mass->Finalize();

   /// grid function
   GridFunction u(fes);
   VectorFunctionCoefficient u0(num_state, uexact);
   u.ProjectCoefficient(u0);

   GridFunction residual(fes);
   residual = 0.0;
   res->Mult(u, residual);
   cout << "residual sum " << residual.Sum() << endl;
   ofstream res_ofs("residual.vtk");
   res_ofs.precision(14);
   mesh->PrintVTK(res_ofs, 1);
   residual.SaveVTK(res_ofs, "Residual", 1);
   res_ofs.close();

   /// time-marching method
   std::unique_ptr<mfem::ODESolver> ode_solver;
   //ode_solver.reset(new RK4Solver);
   ode_solver.reset(new BackwardEulerSolver);
   cout << "ode_solver set " << endl;

   /// TimeDependentOperator
   unique_ptr<mfem::TimeDependentOperator> evolver(new mfem::EulerEvolver(mass, res,
                                                                          0.0, TimeDependentOperator::Type::IMPLICIT));
   /// set up the evolver
   auto t = 0.0;
   evolver->SetTime(t);
   ode_solver->Init(*evolver);

   /// solve the ode problem
   double res_norm0 = calcResidualNorm(res, fes, u);
   double t_final = 1000;
   double dt_init = 100;
   double dt_old;

   /// initial l2_err
   double l2_err_init = calcConservativeVarsL2Error<2, 0>(uexact, &u, fes,
                                                          num_state, 0);
   cout << "l2_err_init " << l2_err_init << endl;
   cout << "initial residual norm " << calcResidualNorm(res, fes, u) << endl;
   double dt = 0.2;
   double res_norm;
   int exponent = 2;
   
   for (auto ti = 0; ti < 30000; ++ti)
   {
      /// calculate timestep
      res_norm = calcResidualNorm(res, fes, u);
      dt_old = dt;
      dt = dt_init * pow(res_norm0 / res_norm, exponent);
      dt = max(dt, dt_old);
      // print iterations
      std::cout << "iter " << ti << ": time = " << t << ": dt = " << dt << endl;
      //std::cout << " (" << round(100 * t / t_final) << "% complete)";

      if (res_norm <= 1e-11)
         break;

      if (isnan(res_norm))
         break;

      ode_solver->Step(u, t, dt);
   }

   cout << "=========================================" << endl;

   std::cout << "final residual norm: " << res_norm << "\n";
   double drag = calcDrag(fes, u, num_state, alpha);
   double drag_err = abs(drag - (-1 / 1.4));
   cout << "drag: " << drag << endl;
   cout << "drag_error: " << drag_err << endl;
   ofstream finalsol_ofs("final_sol_euler_vortex.vtk");
   finalsol_ofs.precision(14);
   mesh->PrintVTK(finalsol_ofs, 1);
   u.SaveVTK(finalsol_ofs, "Solution", 1);
   finalsol_ofs.close();

   /// calculate final solution error
   double l2_err_rho = calcConservativeVarsL2Error<2, 0>(uexact, &u, fes,
                                                         num_state, 0);
   cout << "|| rho_h - rho ||_{L^2} = " << l2_err_rho << endl;

   cout << "=========================================" << endl;
}

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector &p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}

// Returns the exact total entropy value over the quarter annulus
// Note: the number 8.74655... that appears below is the integral of r*rho over the radii
// from 1 to 3.  It was approixmated using a degree 51 Gaussian quadrature.
double calcEntropyTotalExact()
{
   double rhoi = 2.0;
   double prsi = 1.0 / euler::gamma;
   double si = log(prsi / pow(rhoi, euler::gamma));
   return -si * 8.746553803443305 * M_PI * 0.5 / 0.4;
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const Vector &x, Vector &q)
{
   q.SetSize(4);
   Vector u(4);
   double ri = 1.0;
   double Mai = 0.5; //0.95
   double rhoi = 2.0;
   double prsi = 1.0 / euler::gamma;
   double rinv = ri / sqrt(x(0) * x(0) + x(1) * x(1));
   double rho = rhoi * pow(1.0 + 0.5 * euler::gami * Mai * Mai * (1.0 - rinv * rinv),
                           1.0 / euler::gami);
   double Ma = sqrt((2.0 / euler::gami) * ((pow(rhoi / rho, euler::gami)) *
                                               (1.0 + 0.5 * euler::gami * Mai * Mai) -
                                           1.0));
   double theta;
   if (x(0) > 1e-15)
   {
      theta = atan(x(1) / x(0));
   }
   else
   {
      theta = M_PI / 2.0;
   }
   double press = prsi * pow((1.0 + 0.5 * euler::gami * Mai * Mai) /
                                 (1.0 + 0.5 * euler::gami * Ma * Ma),
                             euler::gamma / euler::gami);
   double a = sqrt(euler::gamma * press / rho);

   u(0) = rho;
   u(1) = -rho * a * Ma * sin(theta);
   u(2) = rho * a * Ma * cos(theta);
   u(3) = press / euler::gami + 0.5 * rho * a * a * Ma * Ma;

   //q = u;
   double mach_fs = 0.3;
   q(0) = 1.0;
   q(1) = q(0) * mach_fs; // ignore angle of attack
   q(2) = 0.0;
   q(3) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
}

/// use this for flow over an ellipse
unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   int ref_levels = 3;
   // auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
   //                                          Element::QUADRILATERAL, true /* gen. edges */,
   //                                          40.0, 2*M_PI, true));
   //const char *mesh_file = "periodic_rectangle.mesh";
   const char *mesh_file = "/users/kaurs3/Spring_20/quadrature_rule/mfem/examples/periodic_square_y.mesh";
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(mesh_file, 1, 1));
   /// save the initial mesh
   ofstream sol_ofs("ellipse_square_mesh_periodic.vtk");
   sol_ofs.precision(14);
   mesh_ptr->PrintVTK(sol_ofs, 1);
   sol_ofs.close();
   ofstream mesh_ofs("ellipse_square_mesh_periodic.mesh");
   mesh_ofs.precision(14);
   mesh_ptr->Print(mesh_ofs);
   for (int l = 0; l < ref_levels; l++)
   {
      mesh_ptr->UniformRefinement();
   }
   cout << "Number of elements " << mesh_ptr->GetNE() << '\n';

   //    save the initial mesh
   //    ofstream sol_ofs("ellipse_square_mesh.vtk");
   //    sol_ofs.precision(14);
   //    mesh_ptr->PrintVTK(sol_ofs, 1);
   //    sol_ofs.close();
      // ofstream mesh_ofs("ellipse_square_mesh.mesh");
      // mesh_ofs.precision(14);
      // mesh_ptr->Print(mesh_ofs);
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
      double r_far = 40.0;
      double a0 = 0.5;
      double b0 = a0 / 10.0;
      double delta = 2.00; // We will have to experiment with this
      double r = 1.0 + tanh(delta * (rt(0) / r_far - 1.0)) / tanh(delta);
      double theta = rt(1);
      double b = b0 + (a0 - b0) * r;
      xy(0) = a0 * (r * r_far + 1.0) * cos(theta) + 20.0;
      xy(1) = b * (r * r_far + 1.0) * sin(theta) + 20.0;
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}