//                                MFEM Example 9
//
// Compile with: make ex9
//
// Sample runs:
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex9 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.02 -s 23 -tf 9
//    ex9 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/star-mixed.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex9 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//    ex9 -m ../data/periodic-square.msh -p 0 -r 2 -dt 0.005 -tf 2
//    ex9 -m ../data/periodic-cube.msh -p 0 -r 1 -o 2 -tf 2
//
// Device sample runs:
//    ex9 -pa
//    ex9 -ea
//    ex9 -fa
//    ex9 -pa -m ../data/periodic-cube.mesh
//    ex9 -pa -m ../data/periodic-cube.mesh -d cuda
//    ex9 -ea -m ../data/periodic-cube.mesh -d cuda
//    ex9 -fa -m ../data/periodic-cube.mesh -d cuda
//    ex9 -pa -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9 -d cuda
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. The saving of time-dependent data files for external
//               visualization with VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) is also illustrated.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

bool HasNegative(const Vector &u);

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
real_t u0_function(const Vector &x);

// Initial bounds
Vector bounds(const int dim);

// Inflow boundary condition
real_t inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

class DG_Solver : public Solver
{
private:
   SparseMatrix &M, &K, A;
   GMRESSolver linear_solver;
   BlockILU prec;
   real_t dt;
public:
   DG_Solver(SparseMatrix &M_, SparseMatrix &K_, const FiniteElementSpace &fes)
      : M(M_),
        K(K_),
        prec(fes.GetTypicalFE()->GetDof(),
             BlockILU::Reordering::MINIMUM_DISCARDED_FILL),
        dt(-1.0)
   {
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(0);
      linear_solver.SetPreconditioner(prec);
   }

   void SetTimeStep(real_t dt_)
   {
      if (dt_ != dt)
      {
         dt = dt_;
         // Form operator A = M - dt*K
         A = K;
         A *= -dt;
         A += M;

         // this will also call SetOperator on the preconditioner
         linear_solver.SetOperator(A);
      }
   }

   void SetOperator(const Operator &op) override
   {
      linear_solver.SetOperator(op);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      linear_solver.Mult(x, y);
   }
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   BilinearForm &M, &K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;
   DG_Solver *dg_solver;

   mutable Vector z;

public:
   FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_);

   void Mult(const Vector &x, Vector &y) const override;
   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;

   ~FE_Evolution() override;
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   // const char *mesh_file = "../data/periodic-segment.mesh";
   int ref_levels = 0;
   int order = 3;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 4;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool binary = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea",
                  "--no-element-assembly", "Enable Element Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
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
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   cout << "dim is " << dim << endl;

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   // DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   DG_FECollection fec(order, dim, BasisType::Positive);

   FiniteElementSpace fes(&mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);
   Vector bds = bounds(dim);
   bds.Print();

   int quadorder = 2*order;
   const FiniteElement *fe = fes.GetFE(0);
   IntegrationRules irGL(0, Quadrature1D::GaussLobatto);
   IntegrationRule ir = irGL.Get(fe->GetGeomType(), quadorder);

   cout << "quadorder is " << quadorder << endl;
   cout << "num of quad is " << ir.GetNPoints() << endl;

   BilinearForm m(&fes);
   BilinearForm k(&fes);
   if (pa)
   {
      m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   else if (ea)
   {
      m.SetAssemblyLevel(AssemblyLevel::ELEMENT);
      k.SetAssemblyLevel(AssemblyLevel::ELEMENT);
   }
   else if (fa)
   {
      m.SetAssemblyLevel(AssemblyLevel::FULL);
      k.SetAssemblyLevel(AssemblyLevel::FULL);
   }
   m.AddDomainIntegrator(new MassIntegrator(&ir));
   // m.AddDomainIntegrator(new MassIntegrator);
   constexpr real_t alpha = -1.0;
   auto convecInt = new ConvectionIntegrator(velocity, alpha);
   // const IntegrationRule* intrul = convecInt->GetIntegrationRule();
   // cout << "int rule order is default: " << intrul->GetNPoints() << endl;
   convecInt->SetIntRule(&ir);     
   k.AddDomainIntegrator(convecInt);
   // k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   k.AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));
   k.AddBdrFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha));

   m.Assemble();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   b.Assemble();
   m.Finalize();
   k.Finalize(skip_zeros);

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex9.mesh");
      omesh.precision(precision);
      mesh.Print(omesh);
      ofstream osol("ex9-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example9", &mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9", &mesh);
         dc->SetPrecision(precision);
      }
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("Example9", &mesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", &u);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m, k, b);

   real_t t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   // check if same as when using physical values 
   // Vector nodes(u.Size());
   // Vector node_coords(dim);
   // Vector P(u.Size());
   int offset = 0;
   // for (int el = 0; el < mesh.GetNE(); el++)
   // {
   //    const FiniteElement *fe = fes.GetFE(el);
   //    ElementTransformation *trans = mesh.GetElementTransformation(el);

   //    int edof = fe->GetDof();
   //    const IntegrationRule &ir = fe->GetNodes(); // Node locations in reference element

   //    for (int i = 0; i < edof; i++)
   //    {
   //       const IntegrationPoint &ip = ir.IntPoint(i);

   //       // Map reference node to physical coordinates (optional, for info)
   //       trans->Transform(ip, node_coords);

   //       // Evaluate GridFunction at this node
   //       real_t value = u.GetValue(el, ip);
   //       P[offset+i] = value;

   //       // cout << "Element " << el << ", node " << i << " at "; node_coords.Print(); cout << ": value = " << value << endl;
   //    }
   //    offset += edof;
   // }
   // cout << "u physical values: " << endl;
   // P.Print();
   // cout << "\n";

   // GridFunction avgs(&fes);

   // Vector Pint;
   // int ndofs = fes.GetNDofs();
   // cout << "ndofs = " << fes.GetNDofs() << endl;
   // cout << "ne = " << fes.GetNE() << endl;
   // const IntegrationRule ir = IntRules.Get(mesh.GetElementGeometry(0), order);
   // cout << "nqpts = " << ir.GetNPoints() << endl; 
   // // cout << "nwgts = " << ir.GetWeights().Size() << endl;
   // Pint.SetSize(ir.GetNPoints()*dim*fes.GetNE());    // Pint is by NQPT x VDIM x NE
   // // cout << "Pint size = " << Pint.Size() << endl;
   // QuadratureInterpolator qi(fes,ir);
   // cout << "u size = " << u.Size() << endl;
   // cout << "u coefficient values:" << endl;
   // u.Print();
   // cout << "\n";
   
   
   // qi.PhysValues(u,Pint); // should be on coefficient values
   // cout << "Pint is = " << endl;
   // Pint.Print();

   // MFEM_VERIFY(qi.GetOutputLayout()==QVectorLayout::byNODES, "quadrature points layout not byNODES")
   // cout << "pass \n" << endl;


   // int nelem = fes.GetNE();
   // int nqpts = ir.GetNPoints();
   // // Vector wgts = ir.GetWeights();
   // Vector Pbar(nelem);
   // offset = 0;
   // for (int j=0; j<nelem; j++) {
   //    double sum = 0;
   //    for (int k=0; k<nqpts; k++) {
   //       sum += Pint[offset+k]*ir.IntPoint(k).weight;
   //    }
   //    Pbar[j] = sum/mesh.GetElementSize(j);
   //    offset += nqpts;
   // }
   // cout << "cell avg with QI is:" << endl;
   // Pbar.Print();
   // cout << "\n" << endl;



   // // try for cell avg with quadratureinterpolator function  on reference 

   // Vector cell_averages(mesh.GetNE());

   // // 1. Choose quadrature order (usually 2*fe order for accuracy)
   // int ordert = 2 * fes.GetOrder(0);

   // // 2. Build a quadrature rule for each element type
   // Array<const IntegrationRule*> irs(mesh.GetNE());
   // for (int el = 0; el < mesh.GetNE(); el++)
   // {
   //    const FiniteElement *fe = fes.GetFE(el);
   //    irs[el] = &IntRules.Get(fe->GetGeomType(), ordert);
   // }

   // // 3. Create a QuadratureSpace and QuadratureFunction for the mesh
   // QuadratureSpace qspace(mesh, *irs[0]);
   // QuadratureFunction qvals(&qspace);

   // // 4. Interpolate u at all quadrature points in all elements
   // QuadratureInterpolator qit(fes, qspace);
   // // qit.SetIntRule(irs); // Set per-element rules
   // qit.Values(u, qvals);
   // qvals.Print();

   // // 5. Compute cell averages
   // int qf_offset = 0;
   // for (int el = 0; el < mesh.GetNE(); el++)
   // {
   //    const IntegrationRule &ir = *irs[el];
   //    ElementTransformation *trans = mesh.GetElementTransformation(el);

   //    double int_val = 0.0;

   //    for (int i = 0; i < ir.GetNPoints(); i++)
   //    {
   //       const IntegrationPoint &ip = ir.IntPoint(i);
   //       trans->SetIntPoint(&ip);

   //       double weight = ip.weight * trans->Weight();
   //       double u_val = qvals(qf_offset + i);

   //       int_val += u_val * weight;
   //    }
   //    // cell_averages(el) = int_val / meas;
   //    cell_averages(el) = int_val / mesh.GetElementSize(el);
   //    // cout << "element size for " << el << " is " << mesh.GetElementSize(el) << endl;
   //    qf_offset += ir.GetNPoints();
   // }
   // cout << "cell avg with QI on reference is: " << endl;
   // cell_averages.Print();
   // cout << "\n" << endl;

   // // visualization of cell averages
   // int offsethere = 0;
   // for (int i=0; i < mesh.GetNE(); i++)
   // {
   //    const FiniteElement *fe = fes.GetFE(i);

   //    int edof = fe->GetDof();
   //    for (int j=0; j < edof; j++)
   //    {
   //       u[offsethere + j] = cell_averages[i];
   //    }
   //    offsethere += edof;
   // }
   // // cout << "u now is: " << endl;
   // // u.Print();




   // // try without using quadrature interpolator 
   // Vector cell_avg(mesh.GetNE());

   // for (int el = 0; el < mesh.GetNE(); el++)
   // {
   //    const FiniteElement *fe = fes.GetFE(el);
   //    ElementTransformation *trans = mesh.GetElementTransformation(el);

   //    int order = 2 * fe->GetOrder();
   //    const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), order);

   //    double int_val = 0.0;

   //    for (int i = 0; i < ir.GetNPoints(); i++)
   //    {
   //       const IntegrationPoint &ip = ir.IntPoint(i);
   //       trans->SetIntPoint(&ip);

   //       double weight = ip.weight * trans->Weight();
   //       double u_val = u.GetValue(el, ip);

   //       int_val += u_val * weight;
   //    }

   //    cell_avg(el) = int_val / mesh.GetElementSize(el);
   // }
   // cout << "cell avg is:" << endl;
   // cell_avg.Print();
   // cout << "\n" << endl;




   cout << "MFEM version: "
      << MFEM_VERSION_MAJOR << "."
      << MFEM_VERSION_MINOR << "." 
      << MFEM_VERSION_PATCH << std::endl;

   bool done = false;
   double Pmin = bds(1), Pmax = bds(0);
   for (int ti = 0; !done; )
   {
      // done = true;
      real_t dt_real = min(dt, t_final - t);

      ode_solver->Step(u, t, dt_real); 

      // --- MPP Limiter with Quadrature Cell Average: Insert after ode_solver->Step(...) in the time loop ---

      // Set up integration rule for cell averages
      const FiniteElement *fe = nullptr;
      const IntegrationRule *ir = nullptr;
      Vector u_local, u_vals;
      DenseMatrix elfun;

      // Loop over elements
      for (int i = 0; i < fes.GetNE(); i++)
      {
         Array<int> dofs;
         fes.GetElementDofs(i, dofs);
         fe = fes.GetFE(i);
         ElementTransformation *trans = mesh.GetElementTransformation(i);

         // Use a suitable integration rule (e.g., order = 2*fe->GetOrder())
         int intorder = floor((order+3)/2.0);
         IntegrationRules irGL(0, Quadrature1D::GaussLobatto);
         ir = &irGL.Get(fe->GetGeomType(), intorder);
         // ir = &IntRules.Get(fe->GetGeomType(), intorder);

         u_local.SetSize(dofs.Size());
         u.GetSubVector(dofs, u_local);

         // Evaluate solution at quadrature points
         u_vals.SetSize(ir->GetNPoints());
         // fe->GetValues(*ir, u_local, u_vals);

         // Compute cell average via quadrature
         double int_val = 0.0, int_w = 0.0;
         for (int q = 0; q < ir->GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir->IntPoint(q);
            trans->SetIntPoint(&ip);

            double w = ip.weight * trans->Weight();
            u_vals(q) = u.GetValue(i, ip);

            int_val += w * u_vals(q);
            int_w += w;
         }
         double cell_avg = int_val / int_w;
         // cout << cell_avg << endl;

         double m_i = bds(0), M_i = bds(1);
         // // Find min/max among this and neighbor cell averages
         // double m_i = cell_avg, M_i = cell_avg;
         // Array<int> neighbors;
         // mesh.GetElementNeighbors(i, neighbors);
         // for (int j = 0; j < neighbors.Size(); j++)
         // {
         //    Array<int> ndofs;
         //    fes.GetElementDofs(neighbors[j], ndofs);
         //    const FiniteElement *nfe = fes.GetFE(neighbors[j]);
         //    int norder = 2 * nfe->GetOrder();
         //    const IntegrationRule *nir = &IntRules.Get(nfe->GetGeomType(), norder);
         //    Vector n_local(ndofs.Size()), n_vals(nir->GetNPoints());
         //    u.GetSubVector(ndofs, n_local);
         //    nfe->GetValues(*nir, n_local, n_vals);

         //    double n_int_val = 0.0, n_int_w = 0.0;
         //    for (int q = 0; q < nir->GetNPoints(); q++)
         //    {
         //          double w = nir->IntPoint(q).weight;
         //          n_int_val += w * n_vals(q);
         //          n_int_w  += w;
         //    }
         //    double neighbor_avg = n_int_val / n_int_w;
         //    m_i = std::min(m_i, neighbor_avg);
         //    M_i = std::max(M_i, neighbor_avg);
         // }

         // Compute theta for limiting
         double theta = 1.0;
         for (int k = 0; k < u_local.Size(); k++)
         {
            double diff = u_local[k] - cell_avg;
            if (diff > 0.0)
            {
                  if (cell_avg < M_i)
                     theta = std::min(theta, (M_i - cell_avg) / diff);
                  else
                     theta = 0.0;
            }
            else if (diff < 0.0)
            {
                  if (cell_avg > m_i)
                     theta = std::min(theta, (m_i - cell_avg) / diff);
                  else
                     theta = 0.0;
            }
         }
         // cout << "theta is " << theta << endl;

         // Apply limiter
         for (int k = 0; k < u_local.Size(); k++)
            u_local(k) = theta * (u_local(k) - cell_avg) + cell_avg;
            // if (hold < bds(0) && abs(u_local(k)) < (bds(0)+1e-14) ) u_local(k) = bds(0);
         u.SetSubVector(dofs, u_local);
      }

      // // limiter step here 
      // // 1. Choose quadrature order (based on paper)
      // // int quadorder = floor((order+3)/2.0);
      // // int quadorder = 2 * order;
      // // cout << "quadorder is " << quadorder << endl;

      // // 2. Build a quadrature rule for each element type
      // // const FiniteElement *fe = fes.GetFE(0);
      // // IntegrationRules irGL(0, Quadrature1D::GaussLobatto);
      // // IntegrationRule ir = irGL.Get(fe->GetGeomType(), quadorder);
      
      
      // // 3. Create a QuadratureSpace and QuadratureFunction for the mesh
      // QuadratureSpace qspace(mesh, ir);
      // QuadratureFunction qvals(&qspace);
      // // cout << "size of qvals is: " << qvals.Size() << endl;
      // // Vector qvals(mesh.GetNE()*dim*ir.GetNPoints());
      
      
      // // 4. Interpolate u at all quadrature points in all elements
      // QuadratureInterpolator qi(fes, qspace);
      // // QuadratureInterpolator qi(fes, ir);
      // qi.PhysValues(u, qvals);
      
      // // cout << "qi quad rule num points is: " << qi.IntRule->GetNPoints() << endl;
      // // cout << "qvals is: " << endl;
      // // qvals.Print();
      // // cout << "\n"; 

      // // grab min/max for each element and fill in theta
      // Vector mins(mesh.GetNE());
      // Vector maxs(mesh.GetNE()); 
      // Vector theta(mesh.GetNE());
      // offset = 0;
      // for (int el=0; el < mesh.GetNE(); el++)
      // {
      //    int qpts = ir.GetNPoints();
      //    for (int j=1; j < qpts; j++)
      //    {
      //       mins[el] = qvals[offset]; // ohysical values 
      //       mins[el] = min(mins[el], qvals[offset+j]);

      //       maxs[el] = qvals[offset];
      //       maxs[el] = max(maxs[el], qvals[offset+j]);
      //    }
      //    offset += qpts;

      // }
      // // cout << "element mins are: " << endl;
      // // mins.Print();
      // // cout << "element maxs are: " << endl;
      // // maxs.Print();
      // // cout << "element thetas are: " << endl;
      // // theta.Print();

      // qi.Values(u, qvals);

      // // 5. Compute cell averages
      // int qf_offset = 0;
      // for (int el = 0; el < mesh.GetNE(); el++)
      // {
      //    ElementTransformation *trans = mesh.GetElementTransformation(el);

      //    double int_val = 0.0;

      //    for (int i = 0; i < ir.GetNPoints(); i++)
      //    {
      //       const IntegrationPoint &ip = ir.IntPoint(i);
      //       trans->SetIntPoint(&ip);

      //       double weight = ip.weight * trans->Weight();
      //       double u_val = qvals(qf_offset + i);

      //       int_val += u_val * weight;
      //    }
      //    cell_averages(el) = int_val / mesh.GetElementSize(el);
      //    // cout << "element size for " << el << " is " << mesh.GetElementSize(el) << endl;
      //    qf_offset += ir.GetNPoints();

      //    // compute theta of limiter
      //    theta[el] = min(1.0, min(abs((1-cell_averages[el])/(maxs[el]-cell_averages[el])), abs((0-cell_averages[el])/(mins[el]-cell_averages[el]))));
      
      // }
      // // cout << "cell avg with QI on reference is: " << endl;
      // // cell_averages.Print();
      // // cout << "\n" << endl;

      // // theta = 0.0;



      // offset = 0;
      // for (int el=0; el < mesh.GetNE(); el++)
      // {
      //    const FiniteElement *fe = fes.GetFE(el);

      //    int edof = fe->GetDof();
      //    for (int j=0; j < edof; j++)
      //    {
      //       u[offset + j] = theta[el]*u[offset + j] + (1-theta[el])*cell_averages[el];
      //    }
      //    offset += edof;
      // }
      



      



      ti++;
      // cout << "ti is " << ti << endl;
      
      Vector nodes(u.Size());
      Vector node_coords(dim);
      Vector P(u.Size());
      offset = 0;
      for (int el = 0; el < mesh.GetNE(); el++)
      {
         const FiniteElement *fe = fes.GetFE(el);
         ElementTransformation *trans = mesh.GetElementTransformation(el);

         int edof = fe->GetDof();
         const IntegrationRule &ir = fe->GetNodes(); // Node locations in reference element

         for (int i = 0; i < edof; i++)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);

            // Map reference node to physical coordinates (optional, for info)
            trans->Transform(ip, node_coords);

            // Evaluate GridFunction at this node
            real_t value = u.GetValue(el, ip);
            P[offset+i] = value;

            // cout << "Element " << el << ", node " << i << " at "; node_coords.Print(); cout << ": value = " << value << endl;
         }
         offset += edof;
      }
      // cout << "u coef values: " << u.Min() << " and " << u.Max() << endl;
      // cout << "u phys values: " << P.Min() << " and " << P.Max() << endl;
      Pmin = min(Pmin, P.Min());
      Pmax = max(Pmax, P.Max());
      
      if (HasNegative(P)) {
         cout << "P has negative value at step " << ti << endl;
      }
      // P.Print();
      // cout << "\n";

      
      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         // u.Print();

         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }

         if (paraview)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }
      }
   }
   cout << "Minimum and maximum value of P over all time is " << Pmin << " and " << Pmax << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex9.mesh -g ex9-final.gf".
   {
      ofstream osol("ex9-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Free the used memory.
   delete pd;
   delete dc;

   return 0;
}


bool HasNegative(const Vector &u)
{
    for (int i = 0; i < u.Size(); i++)
    {
        if (u(i) < 0.0)
        {
            return true; // Found a negative value
        }
    }
    return false; // No negative values
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_)
   : TimeDependentOperator(M_.FESpace()->GetTrueVSize()),
     M(M_), K(K_), b(b_), z(height)
{
   Array<int> ess_tdof_list;
   if (M.GetAssemblyLevel() == AssemblyLevel::LEGACY)
   {
      M_prec = new DSmoother(M.SpMat());
      M_solver.SetOperator(M.SpMat());
      dg_solver = new DG_Solver(M.SpMat(), K.SpMat(), *M.FESpace());
   }
   else
   {
      M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
      M_solver.SetOperator(M);
      dg_solver = NULL;
   }
   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void FE_Evolution::ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
{
   MFEM_VERIFY(dg_solver != NULL,
               "Implicit time integration is not supported with partial assembly");
   K.Mult(x, z);
   z += b;
   dg_solver->SetTimeStep(dt);
   dg_solver->Mult(z, k);
}

FE_Evolution::~FE_Evolution()
{
   delete M_prec;
   delete dg_solver;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const real_t w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const real_t w = M_PI/2;
         real_t d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
real_t u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               // return exp(-40.*pow(X(0)-0.5,2))+0.5;
               return 0.5*cos(M_PI*X[0])+1.0;
               // return -(X[0]-1)*(X[0]+1)+1.5;
               // return 1.0;
            case 2:
            case 3:
            {
               real_t rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const real_t s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( std::erfc(w*(X(0)-cx-rx))*std::erfc(-w*(X(0)-cx+rx)) *
                        std::erfc(w*(X(1)-cy-ry))*std::erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         real_t x_ = X(0), y_ = X(1), rho, phi;
         rho = std::hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const real_t f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Initial bounds 
Vector bounds(const int dim)
{
   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
            {
               Vector bds(2);
               bds(0) = 0.5; bds(1) = 1.5;
               return bds;
            }
               
         }
      }
   }
   Vector bds;
   return bds;
}

// Inflow boundary condition (zero for the problems considered in this example)
real_t inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
