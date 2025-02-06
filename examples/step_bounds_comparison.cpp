//                                MFEM Example 41
//
// Compile with: make ex41
//
// Sample runs:
//
// Device sample runs:
//
// Description:   This example code demonstrates bounds-preserving limiters for
//                Discontinuous Galerkin (DG) approximations of hyperbolic
//                conservation laws. The code solves the solves the time-dependent
//                advection equation du(x,t)/dt + v.grad(u) = 0, where v is a given
//                fluid velocity, and u_0(x) = u(x,0) is a given initial condition.
//                The solution of this equation exhibits a minimum principle of the
//                form min[u_0(x)] <= u(x,t) <= max[u_0(x)].

#include "mfem.hpp"
#include "ex18.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "gslib.h"
#include "bounds.hpp"

using namespace std;
using namespace mfem;

int problem;

// Mesh bounding box
Vector bb_min, bb_max;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 4;
   int ref_levels = 1;
   int order = 3;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 1;
   int limiter_type = 1;
   bool use_modal_basis = true;
   real_t t_final = 1;
   real_t dt = 2e-4;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool binary = false;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup: 1 - 1D smooth advection,\n\t"
                  "               2 - 2D smooth advection (structured mesh),\n\t"
                  "               3 - 2D smooth advection (unstructured mesh),\n\t"
                  "               4 - 1D discontinuous advection,\n\t"
                  "               5 - 2D solid body rotation (structured mesh),\n\t"
                  "               6 - 2D solid body rotation (unstructured mesh)\n\t");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 0 - Forward Euler,\n\t"
                  "            1 - RK3 SSP");
   args.AddOption(&limiter_type, "-l", "--limiter",
                  "Limiter: 0 - None,\n\t"
                  "         1 - Discrete,\n\t"
                  "         2 - Continuous");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   // 2. Generate 1D/2D structured/unstructured periodic mesh for the given problem
   Mesh mesh = Mesh::MakeCartesian1D(1);
   int dim = mesh.Dimension();


   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   DG_FECollection fec_pos(order, dim, BasisType::Positive);
   FiniteElementSpace fes_pos(&mesh, &fec_pos);
   const FiniteElement * fe = fes.GetFE(0);

   
   GridFunction u(&fes);
   Vector u_elem(order+1);
   for (int i = 0; i < order+1; i++) {
      if (i < (order)/2.0) {
         u_elem(i) = -0.5;
      }
      else if (i > (order)/2.0) {
         u_elem(i) = 0.5;
      }
      else {
         u_elem(i) = 0.0;
      }
   }
   u.SetElementDofValues(0, u_elem);
   GridFunction u_pos(&fes_pos);
   u_pos.ProjectGridFunction(u);

   Vector x = Vector(dim);
   double ux, umin = 10000, umax = -10000;
   int npts = 10000;
   Vector shape(u_elem.Size());
   IntegrationPoint ip;
   ip.Init(1);
   for (int j = 0; j < npts; j++) {
      x(0) = j/(npts - 1.0);
      ip.Set(x, dim);
      fe->CalcShape(ip, shape);
      
      ux = u_elem*shape;
      umin = min(umin, ux);
      umax = max(umax, ux);
   }
   std::cout << "True min/max          : " << umin << ", " << umax << std::endl;

   // Get Bernstein element DOF values
   u_pos.GetElementDofValues(0, u_elem);
   double pmin = u_elem.Min();
   double pmax = u_elem.Max();
   std::cout << "Bernstein min/max     : " << pmin << ", " << pmax << std::endl;

   u.GetElementDofValues(0, u_elem);
   int n1D = order + 1;
   IntegrationRule irule(n1D);
   QuadratureFunctions1D::GaussLobatto(n1D, &irule);
   Vector gllW(n1D), gllX(n1D);

   // for (int mr = 4; mr < 10; mr++) {
   int mr = order+1;
   std::string filename = "../scripts/bounds/bnddata_spts_lobatto_" + std::to_string(n1D) + "_bpts_opt_" + std::to_string(mr) + ".txt";

   DenseMatrix lboundT, uboundT;
   Vector gllT, intT;
   ReadCustomBounds(gllT, intT, lboundT, uboundT, filename);
   for (int i = 0; i < n1D; i++)
   {
      gllW(i) = irule.IntPoint(i).weight;
      gllX(i) = irule.IntPoint(i).x;
   }
   
   Vector qpminCus, qpmaxCus;
   Get1DBounds(gllX, intT, gllW, lboundT, uboundT, u_elem, qpminCus, qpmaxCus, true);
   
   std::cout << "PLB M" << mr << " min/max        : " << qpminCus.Min() << ", " << qpmaxCus.Max() << std::endl;
   std::cout << "Error reduction " << 100*(1 - (qpmaxCus.Max() - umax)/(pmax - umax)) << endl;
   // }

   return 0;
}