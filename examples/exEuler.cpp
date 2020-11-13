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

void randState(const mfem::Vector &x, mfem::Vector &u)
{
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = 2.0 * uniform_rand(gen) - 1.0;
   }
}
double calcStepSize(int iter, double t, double t_final,
                    double dt_old)
{
   double dt = 0.02;
   dt = min(dt, t_final - t);
   return dt;
}

template <int dim, bool entvar>
void getFreeStreamState(mfem::Vector &q_ref) 
{
   double mach_fs = 0.3;
   q_ref = 0.0;
   q_ref(0) = 1.0;
   q_ref(1) = q_ref(0)*mach_fs; // ignore angle of attack
   q_ref(2) = 0.0;
   q_ref(dim+1) = 1/(euler::gamma*euler::gami) + 0.5*mach_fs*mach_fs;
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

// main
int main(int argc, char *argv[])
{
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2;
   int nx = 4;
   int ny = 4;
   int order = 1;
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
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

   // save the initial mesh
   ofstream sol_ofs("steady_vortex_mesh.vtk");
   sol_ofs.precision(14);
   mesh->PrintVTK(sol_ofs, 0);

   // finite element collection
   FiniteElementCollection *fec = new DG_FECollection(order, dim);

   // finite element space
   FiniteElementSpace *fes = new FiniteElementSpace(mesh.get(), fec, num_state,
                                                    Ordering::byVDIM);
   cout << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;

   /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   Array<int> bndry_marker_isentropic;
   Array<int> bndry_marker_slipwall;

   bndry_marker_isentropic.Append(1);
   bndry_marker_isentropic.Append(1);
   bndry_marker_isentropic.Append(1);
   bndry_marker_isentropic.Append(0);

   bndry_marker_slipwall.Append(0);
   bndry_marker_slipwall.Append(0);
   bndry_marker_slipwall.Append(0);
   bndry_marker_slipwall.Append(1);

   Vector qfs(dim+2);

   getFreeStreamState<2, 0>(qfs);

   /// nonlinearform
   NonlinearForm *res = new NonlinearForm(fes);
   res->AddDomainIntegrator(new EulerDomainIntegrator<2>(num_state, 1));
   res->AddBdrFaceIntegrator(new EulerBoundaryIntegrator<2, 1, 0>(fec, num_state, qfs, 1.0),
                             bndry_marker_isentropic);
   res->AddBdrFaceIntegrator(new EulerBoundaryIntegrator<2, 2, 0>(fec, num_state, qfs, 1.0),
                              bndry_marker_slipwall);                           
   res->AddInteriorFaceIntegrator(new EulerFaceIntegrator<2>(fec, 1.0, num_state, 1.0));
   
   /// check if the integrators are correct
   double delta = 1e-5;

   // initialize state; here we randomly perturb a constant state
   GridFunction q(fes);
   VectorFunctionCoefficient pert(num_state, randBaselinePert<2>);
   q.ProjectCoefficient(pert);

   // initialize the vector that the Jacobian multiplies
   GridFunction v(fes);
   VectorFunctionCoefficient v_rand(num_state, randState);
   v.ProjectCoefficient(v_rand);

   // evaluate the Jacobian and compute its product with v
   Operator &Jac = res->GetGradient(q);
   GridFunction jac_v(fes);
   Jac.Mult(v, jac_v);

   // now compute the finite-difference approximation...
   GridFunction q_pert(q), r(fes), jac_v_fd(fes);
   q_pert.Add(-delta, v);
   res->Mult(q_pert, r);
   q_pert.Add(2 * delta, v);
   res->Mult(q_pert, jac_v_fd);
   jac_v_fd -= r;
   jac_v_fd /= (2 * delta);

   for (int i = 0; i < jac_v.Size(); ++i)
   {
     // std::cout << std::abs(jac_v(i) - (jac_v_fd(i))) << "\n";
      MFEM_ASSERT(abs(jac_v(i) - (jac_v_fd(i))) < 1e-09, "jacobian is incorrect");
   }

   /// bilinear form
   BilinearForm *mass = new BilinearForm(fes);

   // set up the mass matrix
   // mass->AddDomainIntegrator(new EulerMassIntegrator(num_state));
   // mass->Assemble();
   // mass->Finalize();

   auto mass_integ = new VectorMassIntegrator();
   mass_integ->SetVDim(dim + 2);
   mass->AddDomainIntegrator(mass_integ);
   mass->Assemble();
   mass->Finalize();
   // SparseMatrix &mass_matrix = mass->SpMat();

   /// grid function
   GridFunction u(fes);
   VectorFunctionCoefficient u0(num_state, uexact);
   u.ProjectCoefficient(u0);
   /// time-marching method (might be NULL)
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
   mfem::GridFunction residual(fes);
   residual = 0.0;
   double dt = 0.0;
   double t_final = 100;
   res->Mult(u, residual);
   ofstream adj_ofs("residual.vtk");
   adj_ofs.precision(14);
   mesh->PrintVTK(adj_ofs, 1);
   residual.SaveVTK(adj_ofs, "Residual", 1);
   adj_ofs.close();

   for (auto ti = 0; ti < 10000; ++ti)
   {
      dt = calcStepSize(ti, t, t_final, dt);
      std::cout << "iter " << ti << ": time = " << t << ": dt = " << dt << endl;
      //   std::cout << " (" << round(100 * t / t_final) << "% complete)";
      res->Mult(u, residual);
      auto res_norm = residual.Norml2();
      std::cout << "residual norm: " << res_norm << "\n";
      if (res_norm <= 1e-12)
         break;

      if (isnan(res_norm))
         break;

      ode_solver->Step(u, t, dt);
   }
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
   u(1) = rho * a * Ma * sin(theta);
   u(2) = -rho * a * Ma * cos(theta);
   u(3) = press / euler::gami + 0.5 * rho * a * a * Ma * Ma;

   q = u;
   // double mach_fs = 0.3;
   // q(0) = 1.0;
   // q(1) = q(0)*mach_fs; // ignore angle of attack
   // q(2) = 0.0;
   // q(3) = 1/(euler::gamma*euler::gami) + 0.5*mach_fs*mach_fs;
}

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