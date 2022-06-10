// ===========================================================================
//
//        Mini-App: surrogate model for imperfect materials.
//  
//  Details: refer to README
//
//  Runs:
//    mpirun -np 4 ./miniapps/materials/synthetic_materials
//
// ===========================================================================


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>

#include "rational_approximation.hpp"
#include "material_metrics.hpp"
#include "util.hpp"
#include "solvers.hpp"

using namespace std;
using namespace mfem;

enum TopologicalSupport { kParticles, kOctetTruss};

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/ref-cube.mesh";
   int order = 1;
   int num_refs = 3;
   int number_of_particles = 3;
   int topological_support = TopologicalSupport::kParticles;
   double nu = 1.0;
   double tau = 1.0;
   double zeta = 1.0;
   double l1 = 1;
   double l2 = 1;
   double l3 = 1;
   double e1 = 0;
   double e2 = 0;
   double e3 = 0;
   double pl1 = 1.0;
   double pl2 = 1.0;
   double pl3 = 1.0;
   bool paraview_export = true;
   bool glvis_export = true;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&num_refs, "-r", "--refs",
                  "Number of uniform refinements");
   args.AddOption(&topological_support, "-top", "--topology",
                  "Topological support. 0 particles, 1 octet-truss");
   args.AddOption(&nu, "-nu", "--nu",
                  "Fractional exponent nu (smoothness)");
   args.AddOption(&tau, "-t", "--tau",
                  "Parameter for topology generation");
   args.AddOption(&zeta, "-z", "--zeta",
                  "Parameter to scale the mixing of topology and randomness");
   args.AddOption(&l1, "-l1", "--l1", 
                  "First component of diagonal core of theta");
   args.AddOption(&l2, "-l2", "--l2",
                  "Second component of diagonal core of theta");
   args.AddOption(&l3, "-l3", "--l3",
                  "Third component of diagonal core of theta");
   args.AddOption(&e1, "-e1", "--e1",
                  "First euler angle for rotation of theta");
   args.AddOption(&e2, "-e2", "--e2",
                  "Second euler angle for rotation of theta");
   args.AddOption(&e3, "-e3", "--e3",
                  "Third euler angle for rotation of theta");
   args.AddOption(&pl1, "-pl1", "--pl1", 
                  "Length scale 1 of particles");
   args.AddOption(&pl2, "-pl2", "--pl2",
                  "Length scale 2 of particles");
   args.AddOption(&pl3, "-pl3", "--pl3",
                  "Length scale 3 of particles");
   args.AddOption(&number_of_particles, "-n", "--number-of-particles",
                   "Number of particles");
   args.AddOption(&paraview_export, "-pvis", "--paraview-visualization", 
                  "-no-pvis", "--no-paraview-visualization",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&glvis_export, "-gvis", "--glvis-visualization", "-no-gvis",
                  "--no-glvis-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution.
   for (int i = 0; i < num_refs; i++)
   {
      mesh.UniformRefinement();
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 5. Define a finite element space on the mesh.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   if (Mpi::Root())
   {
      cout << "Number of finite element unknowns: "
           << fespace.GetTrueVSize() << endl;
   }

   // 6. Boundary conditions
   const Array<int> ess_tdof_list;

   // ========================================================================
   // II. Generate topological support
   // ========================================================================
   
   // II.1 Define the metric for the topological support.
   MaterialTopology* mdm = nullptr;
   if (topological_support == TopologicalSupport::kOctetTruss){
      mdm = new OctetTrussTopology();
   } else{
      // Create the same random particles on all processors.
      std::vector<double> random_positions (3 * number_of_particles);
      std::vector<double> random_rotations (9 * number_of_particles);
      if (Mpi::Root())
      {
         if (topological_support != TopologicalSupport::kParticles){
            mfem::out << "Warning: Selected topological support not valid.\n" 
                      << "         Fall back to kParticles." << std::endl;
         }
         // Generate random positions and rotations. We generate them on the root
         // process and then broadcast them to all processes because we need the 
         // same random positions and rotations on all processes.
         FillWithRandomNumbers(random_positions, 0.2, 0.8);
         FillWithRandomRotations(random_rotations);
      }

      // Broadcast the random positions and rotations to all processes.
      MPI_Bcast(random_positions.data(), 3 * number_of_particles, MPI_DOUBLE, 
               0, MPI_COMM_WORLD);
      MPI_Bcast(random_rotations.data(), 9 * number_of_particles, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

      mdm = new ParticleTopology(pl1,pl2,pl3, random_positions, 
                                              random_rotations);
   }
   
   // II.2 Define lambda to wrap the call to the distance metric.
   auto topo = [&mdm, &tau, &zeta](const Vector& x)
   {
      return (tau - mdm->ComputeMetric(x));
   };

   // II.1 Create a grid funtion for the topological support.
   FunctionCoefficient topo_coeff(topo);
   ParGridFunction v(&fespace);
   v.ProjectCoefficient(topo_coeff);


   // ========================================================================
   // III. Generate random imperfections via fractional PDE
   // ========================================================================

   // III.1 Compute the rational expansion coefficients that define the
   //       integer-order PDEs.
   Array<double> coeffs, poles;
   double alpha = (nu + dim / 2.0 ) / 2.0; // fractional exponent
   const int int_order_of_operator = floor(alpha);
   double exponent_to_approximate = alpha - int_order_of_operator;
   bool integer_order = false;
   // Check if alpha is an integer or not.
   if (abs(exponent_to_approximate) > 1e-12)
   {
      if (Mpi::Root())
      {
         mfem::out << "Approximating the fractional exponent "
                   << exponent_to_approximate
                   << endl;
      }
      ComputePartialFractionApproximation(exponent_to_approximate, coeffs,
                                          poles);

      // If the example is build without LAPACK, the exponent_to_approximate
      // might be modified by the function call above.
      alpha = exponent_to_approximate + int_order_of_operator;
   }
   else
   {
      integer_order = true;
      if (Mpi::Root())
      {
         mfem::out << "Treating integer order PDE." << endl;
      }
   }

   // III.2 Define coeffiecients and solution GridFunction(s).
   ConstantCoefficient one(1.0);
   auto diffusion_tensor = ConstructMatrixCoefficient(l1, l2, l3, 
                                                      e1, e2, e3, nu, dim);
   MatrixConstantCoefficient diffusion_coefficient(diffusion_tensor);
   ParGridFunction u(&fespace); // Solution for the fractional PDE.
   ParGridFunction g(&fespace); // Solution for the integer-order PDE.
   ParGridFunction x(&fespace); // Solution for the integer-order PDEs (approx).
   u = 0.0;
   g = 0.0;
   x = 0.0;

   // III.3 Set up the linear form b(.) for integer-order PDE solves.
   ParLinearForm b(&fespace);
   int seed = 4000;
   auto *WhiteNoise = new WhiteGaussianNoiseDomainLFIntegrator(seed);
   b.AddDomainIntegrator(WhiteNoise);
   b.Assemble();
   double normalization = ConstructNormalizationCoefficient(nu, l1, l2, 
                                                            l3, dim);
   b *= normalization;

   // ------------------------------------------------------------------------
   // III.4 Solve the PDE (-Δ)^N g = f, i.e. compute g = (-Δ)^{-1}^N f.
   // ------------------------------------------------------------------------
   

   if (int_order_of_operator > 0)
   {
      if (Mpi::Root())
         {
            mfem::out << "\nSolving PDE (A)^" << int_order_of_operator
                      << " u = f" << endl;
         }
      materials::PDESolver solver(1.0, 
                                 diffusion_coefficient,
                                 ess_tdof_list,
                                 &fespace,
                                 int_order_of_operator);
      solver.Solve(b, g);
      if (integer_order)
      {
         u += g;
      }
   }

   // ------------------------------------------------------------------------
   // III.5 Solve the fractional PDE by solving M integer order PDEs and adding
   //       up the solutions.
   // ------------------------------------------------------------------------
   if (!integer_order)
   {
      // Iterate over all expansion coefficient that contribute to the
      // solution.
      for (int i = 0; i < coeffs.Size(); i++)
      {
         if (Mpi::Root())
         {
            mfem::out << "\nSolving PDE -Δ u + " << -poles[i]
                      << " u = " << coeffs[i] << " g " << endl;
         }
         x = 0.0;
         materials::PDESolver solver(1-poles[i], 
                                     diffusion_coefficient,
                                     ess_tdof_list,
                                     &fespace, 1);
         solver.Solve(b, x);
         x *= coeffs[i];
         u += x;
      }
   }

   // ========================================================================
   // III. Combine topological support and random field
   // ========================================================================

   ParGridFunction w(&fespace);
   w = 0.0;
   w.Add(zeta, u);
   w.Add(1.0 - zeta, v);
   
   // ========================================================================
   // VI. Export visualization to ParaView and GLVis
   // ========================================================================
   
   if (paraview_export){
      ParaViewDataCollection paraview_dc("SurrogateMaterial", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("random_field",&u);
      paraview_dc.RegisterField("topological_support",&v);
      paraview_dc.RegisterField("imperfect_topology",&w);
      paraview_dc.Save();
   }

   if (glvis_export){
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream uout, vout, wout;
      ostringstream oss_u, oss_v, oss_w;
      uout.open(vishost, visport);
      uout.precision(8);
      vout.open(vishost, visport);
      vout.precision(8);
      wout.open(vishost, visport);
      wout.precision(8);
      oss_u.str(""); oss_u.clear();
      oss_v.str(""); oss_v.clear();
      oss_w.str(""); oss_w.clear();
      oss_u << "Random Field";
      oss_v << "Topological Support";
      if (topological_support == TopologicalSupport::kParticles){
         oss_w << "Imperfect Particles";
      } else {
         oss_w << "Imperfect Octet-Truss";
      }
      uout << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << pmesh << u
            << "window_title '" << oss_u.str() << "'" << flush;
      vout << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << pmesh << v
            << "window_title '" << oss_v.str() << "'" << flush;
      wout << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << pmesh << w
            << "window_title '" << oss_w.str() << "'" << flush;
   }

   delete mdm;
   return 0;
}

