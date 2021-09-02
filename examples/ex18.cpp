//                                MFEM Example 18
//
// Compile with: make ex18
//
// Sample runs:
//
//       ex18 -p 1 -r 2 -o 1 -s 3
//       ex18 -p 1 -r 1 -o 3 -s 4
//       ex18 -p 1 -r 0 -o 5 -s 6
//       ex18 -p 2 -r 1 -o 1 -s 3
//       ex18 -p 2 -r 0 -o 3 -s 3
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates user-defined bilinear and nonlinear
//               form integrators for systems of equations that are defined with
//               block vectors, and how these are used with an operator for
//               explicit time integrators. In this case the system also
//               involves an external approximate Riemann solver for the DG
//               interface flux. It also demonstrates how to use GLVis for
//               in-situ visualization of vector grid functions.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>

// Classes FE_Evolution, RiemannSolver, DomainIntegrator and FaceIntegrator
// shared between the serial and parallel version of the example.
#include "ex18.hpp"

// Choice for the problem setup. See InitialCondition in ex18.hpp.
int problem;

// The dim and hence number of equations is taken from the mesh.
int dim;
int num_equation;

// Equation constant parameters.
const double specific_heat_ratio = 1.4;
const double gas_constant = 1.0;

// Maximum characteristic speed (updated by integrators)
double max_char_speed;

void amr_update(FiniteElementSpace& fes,
                FiniteElementSpace& dfes,
                FiniteElementSpace& vfes,
                BlockVector& u_block,
                GridFunction& rho,
                GridFunction& rho_u,
                GridFunction& rho_e,
                GridFunction& sol)
{
   fes.Update();
   dfes.Update();
   vfes.Update();

   rho.Update();
   rho_u.Update();
   rho_e.Update();

   Array<int> offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++) {
      offsets[k] = k * vfes.GetNDofs();
   }
   u_block.Update(offsets);

   Vector& sub0 = u_block.GetBlock(0);
   Vector& sub1 = u_block.GetBlock(1);
   Vector& sub2 = u_block.GetBlock(2);

   sub0 = rho;
   sub1 = rho_u;
   sub2 = rho_e;

   sol.MakeRef(&vfes, u_block.GetData());

   rho.MakeRef(&fes, u_block.GetData() + offsets[0]);
   rho_u.MakeRef(&dfes, u_block.GetData() + offsets[1]);
   rho_e.MakeRef(&fes, u_block.GetData() + offsets[2]);
}

double estimate_dt(double cfl, Mesh& mesh,
                   GridFunction& sol, NonlinearForm& A, int order)
{
   // Determine the minimum element size.
   double hmin = 0.0;
   hmin = mesh.GetElementSize(0, 1);
   for (int i = 1; i < mesh.GetNE(); i++)
   {
      hmin = min(mesh.GetElementSize(i, 1), hmin);
   }

   double dt = -0.01;
   // Find a safe dt, using a temporary vector. Calling Mult() computes the
   // maximum char speed at all quadrature points on all faces.
   Vector z(A.Width());
   max_char_speed = 0.;
   A.Mult(sol, z);
   dt = cfl * hmin / max_char_speed / (2*order+1);

   return dt;
}

void show_maps(Mesh& mesh, vector<int>& coarse_map, vector<int>& fine_map)
{
   for (int i = 0; i < mesh.GetNE(); i++) {
      int depth = mesh.ncmesh->GetElementDepth(i);
      if (depth == 0) {
         printf("| %2d  ",i);
      }
      else {
         printf("|%2d|%2d",i,i+1);
         i++;
      }
   }
   printf("| mesh i\n");

   for (size_t i = 0; i < coarse_map.size(); i++) {
      printf("| %2d  ",coarse_map[i]);
   }
   printf("| coarse map\n");

   for (size_t i = 0; i < coarse_map.size(); i++) {
      printf("| %2d  ",fine_map[i]);
   }
   printf("| fine map\n");

   for (size_t i = 0; i < coarse_map.size(); i++) {
      printf("| %2lu  ",i);
   }
   printf("| ref idx\n");

   // some validation checks
   for (size_t i = 0; i < coarse_map.size(); i++) {
      //printf("checking reference index %d\n",i);
      if (coarse_map[i] > -1) assert(  fine_map[i] == -1);
      if (fine_map[i]   > -1) assert(coarse_map[i] == -1);
      if (coarse_map[i] > -1) assert(mesh.ncmesh->GetElementDepth(coarse_map[i]) == 0);
      if (fine_map[i]   > -1) assert(mesh.ncmesh->GetElementDepth(fine_map[i])   == 1);
      if (fine_map[i]   > -1) assert(mesh.ncmesh->GetElementDepth(fine_map[i]+1) == 1);
   }
}

void compute_reference_errors(int ti,
                              FiniteElementCollection& fec,
                              Mesh& mesh,
                              GridFunction& den,
                              const vector<int>& coarse_map,
                              const vector<int>& fine_map,
                              vector<double>& errors)
{
   // Create a temp copy of the mesh for refinement to reference resolution
   Mesh tmp_mesh(mesh);
   FiniteElementSpace tmp_fes(&tmp_mesh, &fec);
   GridFunction tmp_den(&tmp_fes);
   tmp_den = den; // copy data

   // refine mesh and soln to reference resolution
   Array<int> refs;
   for (int i = 0; i < tmp_mesh.GetNE(); i++) {
      if (tmp_mesh.ncmesh->GetElementDepth(i) == 0) {
         refs.Append(i);
      }
   }
   tmp_mesh.GeneralRefinement(refs);
   tmp_den.FESpace()->Update();
   tmp_den.Update();

   // read in reference mesh
   ostringstream fn;
   fn << "reference-" << ti << ".mesh";
   Mesh ref_mesh(fn.str().c_str());

   // read in reference density
   ostringstream sol_name;
   sol_name << "reference-" << ti << "-" << 0 << ".gf";
   ifstream gf_ifs(sol_name.str());
   GridFunction ref_den(&ref_mesh, gf_ifs);

   // integrate L2 errors
   GridFunctionCoefficient ref_den_coeff(&ref_den);
   Vector err(ref_mesh.GetNE());
   tmp_den.ComputeElementL2Errors(ref_den_coeff, err);

   {
      ostringstream sol_name;
      sol_name << "refined-current-rho-" << ti << ".gf";
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs << tmp_den;
   }


   // sum L2 errors to reference mesh. The reference errors are always
   // on a fully refined mesh.
   int k = 0;
   for (size_t i = 0; i < coarse_map.size(); i++) {
      double e0 = err[k];
      double e1 = err[k+1];
      errors[i] = sqrt(e0*e0+e1*e1);
      k += 2;
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 1;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 2.0;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = false;
   int vis_steps = 50;
   Array<int> rseq;
   int nseq = 0;
   int regrid_period = 1;
   int greedy_refine = 0;
   int output_cycle_soln = 0;
   int output_cycle_errors = 0;

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
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&rseq, "-rseq", "--refine-sequence",
                  "Element sequence to refine.");
   args.AddOption(&regrid_period, "-rp", "--regrid-period",
                  "Timesteps per regrid.");
   args.AddOption(&greedy_refine, "-gr", "--greedy-refine",
                  "Use greedy refinement strategy.");
   args.AddOption(&output_cycle_soln, "-cs", "--output-cycle-solutions",
                  "Output per-cycle solutions.");
   args.AddOption(&output_cycle_errors, "-ce", "--output-cycle-errors",
                  "Output per-cycle errors.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   printf("dt = %f\n",dt);

   // 2. Read the mesh from the given mesh file. This example requires a 2D
   //    periodic mesh, such as ../data/periodic-square.mesh.
   Mesh mesh(mesh_file, 1, 1);
   mesh.EnsureNCMesh();
   const int dim = mesh.Dimension();

   num_equation = 2+dim;

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   FiniteElementSpace fes(&mesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   FiniteElementSpace dfes(&mesh, &fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   FiniteElementSpace vfes(&mesh, &fec, num_equation, Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   cout << "Number of unknowns: " << vfes.GetVSize() << endl;

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // The solution u has components {density, x-momentum, y-momentum, energy}.
   // These are stored contiguously in the BlockVector u_block.
   Array<int> offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++) { offsets[k] = k * vfes.GetNDofs(); }
   BlockVector u_block(offsets);

   // Momentum grid function on dfes for visualization.
   GridFunction rho(&fes, u_block.GetData() + offsets[0]);
   GridFunction rho_u(&dfes, u_block.GetData() + offsets[1]);
   GridFunction rho_e(&fes, u_block.GetData() + offsets[2]);

   // Initialize the state.
   VectorFunctionCoefficient u0(num_equation, InitialCondition);
   GridFunction sol(&vfes, u_block.GetData());
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      ofstream mesh_ofs("vortex.mesh");
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equation; k++)
      {
         GridFunction uk(&fes, u_block.GetBlock(k));
         ostringstream sol_name;
         sol_name << "vortex-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }

   }

   // 7. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   MixedBilinearForm Aflux(&dfes, &fes);
   Aflux.AddDomainIntegrator(new DomainIntegrator(dim));
   Aflux.Assemble();

   NonlinearForm A(&vfes);
   RiemannSolver rsolver;
   A.AddInteriorFaceIntegrator(new FaceIntegrator(rsolver, dim));

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution euler(vfes, A, Aflux);

   // Visualize the density
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
         sout << "solution\n" << mesh << rho;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   double t = 0.0;
   euler.SetTime(t);
   ode_solver->Init(euler);

   // map from base coarse numbering to current coarse numbering.
   // -1 entries for elements that are currently refined.
   vector<int> coarse_map(mesh.GetNE());
   for (size_t i = 0; i < coarse_map.size(); ++i) {
      coarse_map[i] = i;
   }

   // map from base coarse numbering to current fine numbering. The
   // second element is always this index+1.
   // -1 entries for elements that are currently coarse.
   vector<int> fine_map(mesh.GetNE());
   for (size_t i = 0; i < fine_map.size(); ++i) {
      fine_map[i] = -1;
   }

   //show_maps(mesh,coarse_map,fine_map);

   // vector<int> coarse_map(mesh.GetNE());
   // for (int i = 0; i < coarse_map.size(); ++i) {
   //    coarse_map[i] = i;
   // }

   // the elements that are currently refined
   set<int> cur_ref_set; // in base numbering

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done; )
   {

      if (greedy_refine && !(ti % regrid_period )) {
         vector<double> ref_errors(coarse_map.size());
         compute_reference_errors(ti, fec, mesh, rho,
                                  coarse_map, fine_map, ref_errors);
         vector<double>::iterator it;
         it = std::max_element(ref_errors.begin(), ref_errors.end());
         int ref_max = std::distance(ref_errors.begin(), it);
         std::cout << "max at: " << std::distance(ref_errors.begin(), it) << '\n';
         rseq.SetSize(nseq+1);
         rseq[nseq] = ref_max;
      }

      // adapt mesh to new element(s)
      if (rseq.Size() && !(ti % regrid_period )) {

         //printf("*** begin refinement ***\n");
         set<int> new_ref_set; // in base numbering

         // use base numbering
         int ne = coarse_map.size();
         int el1 = rseq[nseq++];
         int el2 = el1-1;
         int el3 = el1+1;
         el2 = (el2 + ne) % ne;
         el3 = (el3 + ne) % ne;
         new_ref_set.insert(el1);
         new_ref_set.insert(el2);
         new_ref_set.insert(el3);

         // The coarsen set is elements from cur_ref_set not in new_ref_set
         set<int> coarsen_set;
         std::set_difference(cur_ref_set.begin(), cur_ref_set.end(),
                             new_ref_set.begin(), new_ref_set.end(),
                             std::inserter(coarsen_set, coarsen_set.begin()));

         // The refine set is elements from new_ref_set not in cur_ref_set
         set<int> refine_set;
         std::set_difference(new_ref_set.begin(), new_ref_set.end(),
                             cur_ref_set.begin(), cur_ref_set.end(),
                             std::inserter(refine_set, refine_set.begin()));

         set<int>::iterator it;

         // printf("cur_ref set\n");
         //
         // for (it = cur_ref_set.begin(); it != cur_ref_set.end(); ++it) {
         //    const int& i = *it;
         //    printf("%d ",i);
         // }
         // printf("\n");

         // printf("new_ref set\n");
         // for (it = new_ref_set.begin(); it != new_ref_set.end(); ++it) {
         //    const int& i = *it;
         //    printf("%d ",i);
         // }
         // printf("\n");

         // printf("refine set: ");
         // for (it = refine_set.begin(); it != refine_set.end(); ++it) {
         //    const int& i = *it;
         //    printf("%d ",i);
         // }
         // printf("\n");

         // printf("coarsen set: ");
         // for (it = coarsen_set.begin(); it != coarsen_set.end(); ++it) {
         //    const int& i = *it;
         //    printf("%d ",i);
         // }
         // printf("\n");


         // Perform any new refinements. Translate into current numbering.
         Array<int> els;
         for (it = refine_set.begin(); it != refine_set.end(); ++it) {
            const int& i = *it;
            assert(coarse_map[i] >= 0);
            //printf("adding %d -> %d to be refined\n",i,coarse_map[i]);
            els.Append(coarse_map[i]);
         }

         if (els.Size()) {
            //printf("  ** start refinements **\n");
            mesh.GeneralRefinement(els);

            const CoarseFineTransformations &cft = mesh.GetRefinementTransforms();
            //printf("updating coarse_map and fine_map after refinement\n");
            Table c2f;
            cft.GetCoarseToFineMap(mesh, c2f);
            Array<int> row;
            for (size_t i = 0; i < coarse_map.size(); ++i) {

               int old = coarse_map[i];
               if (old > -1) {
                  // was coarse
                  c2f.GetRow(old,row);
                  if (row.Size() == 1) {
                     // was coarse, still coarse
                     coarse_map[i] = row[0];
                  }
                  else {
                     // was coarse, now fine
                     coarse_map[i] = -1;
                     fine_map[i] = row[0];
                  }
               }
               else {
                  // was fine, still fine
                  old = fine_map[i];
                  assert(old != -1);
                  c2f.GetRow(old,row);
                  assert(row.Size() == 1);
                  fine_map[i] = row[0];
               }
            }

            // printf("new coarse_map is:\n");
            // for (size_t i = 0; i < coarse_map.size(); i++) {
            //    printf("%lu -> %d\n",i,coarse_map[i]);
            // }
            // printf("new fine_map is:\n");
            // for (size_t i = 0; i < fine_map.size(); i++) {
            //    printf("%lu -> %d\n",i,fine_map[i]);
            // }

            amr_update(fes, dfes, vfes, u_block, rho, rho_u, rho_e, sol);
            A.Update();
            Aflux.Update();
            Aflux.Assemble();
            euler.Update();
            ode_solver->Init(euler);
            //printf("  ** done refinements **\n");

            //printf("maps after refinement\n");
            //show_maps(mesh,coarse_map,fine_map);
         }


         cur_ref_set = new_ref_set;

         // Perform any new derefinements
         if (coarsen_set.size()) {

            //printf("  ** start derefinements **\n");

            Array<double> mock_error(mesh.GetNE());
            mock_error = 1.0;

            // We only mark one of the fine elements, but that's fine
            // because we can set the threshold low enough to always
            // derefine.
            for (it = coarsen_set.begin(); it != coarsen_set.end(); ++it) {
               const int& i = *it;
               //printf("coarsening ref idx %d\n",i);
               assert(coarse_map[i] == -1);
               int i1 = fine_map[i];
               //printf("which is fine element %d (and +1)\n",i1);
               assert(i1 >= 0);
               mock_error[i1] = 0.0;
               //printf("setting mock error to 0.0 in %d\n",i1);
            }
            mesh.DerefineByError(mock_error, 2.0);
            amr_update(fes, dfes, vfes, u_block, rho, rho_u, rho_e, sol);
            A.Update();
            Aflux.Update();
            Aflux.Assemble();
            euler.Update();
            ode_solver->Init(euler);

            //printf("updating coarse_map and fine_map after derefinement\n");

            const CoarseFineTransformations& cft = mesh.ncmesh->GetDerefinementTransforms();

            Table c2f;
            cft.GetCoarseToFineMap(mesh, c2f);
            //c2f.Print();

            Array<int> row;
            map<int,int> old2new;
            for (int i = 0; i < c2f.Size(); ++i) {
               c2f.GetRow(i,row);
               for (int j = 0; j < row.Size(); j++) {
                  old2new[row[j]] = i;
               }
            }

            // map<int,int>::iterator it;
            // printf("old2new\n");
            // for (it = old2new.begin(); it != old2new.end(); ++it) {
            //    printf("%d -> %d\n",it->first,it->second);
            // }

            for (size_t i = 0; i < coarse_map.size(); i++) {
               //printf("ref i = %lu\n",i);
               int jfine = fine_map[i];
               int jcoarse = coarse_map[i];
               //printf("mesh j old grid (fine) = %d\n",jfine);
               if (jfine > -1) {
                  int newj = old2new[jfine];
                  //printf("mesh j new grid (coarse) = %d\n",newj);
                  c2f.GetRow(newj,row);
                  if (row.Size() > 1) {
                     //printf("  was fine, now coarse\n");
                     // was fine, now coarse
                     coarse_map[i] = newj;
                     fine_map[i] = -1;
                  }
                  else {
                     //printf("  was fine, still fine\n");
                     // was fine, still fine
                     fine_map[i] = newj;
                  }
               }
               //printf("mesh j old grid coarse = %d\n",jcoarse);
               if (jcoarse > -1) {
                  int newj = old2new[jcoarse];
                  coarse_map[i] = newj;
               }
            }

            // printf("new coarse_map is:\n");
            // for (size_t i = 0; i < coarse_map.size(); i++) {
            //    printf("%lu -> %d\n",i,coarse_map[i]);
            // }
            // printf("new fine_map is:\n");
            // for (size_t i = 0; i < fine_map.size(); i++) {
            //    printf("%lu -> %d\n",i,fine_map[i]);
            // }

            //printf("  ** done derefinements **\n");

            //printf("maps after derefinement\n");
            //show_maps(mesh,coarse_map,fine_map);

         }

      }

      // Output the current solution.
      if (output_cycle_soln) {

         ostringstream fn;
         fn << "soln-" << ti << ".mesh";
         ofstream mesh_ofs(fn.str());
         mesh_ofs.precision(precision);
         mesh_ofs << mesh;

         for (int k = 0; k < num_equation; k++)
         {
            GridFunction uk(&fes, u_block.GetBlock(k));
            ostringstream sol_name;
            sol_name << "soln-" << ti << "-" << k << ".gf";
            ofstream sol_ofs(sol_name.str().c_str());
            sol_ofs.precision(precision);
            sol_ofs << uk;
         }
      }

      // Output the current error on the reference mesh
      if (output_cycle_errors) {
         vector<double> ref_errors(coarse_map.size());
         compute_reference_errors(ti, fec, mesh, rho,
                                  coarse_map, fine_map, ref_errors);
         ostringstream fn;
         fn << "err-" << ti << ".dat";
         ofstream err_ofs(fn.str());
         vector<double>::iterator it;
         for (it = ref_errors.begin(); it != ref_errors.end(); ++it) {
            int i = std::distance(ref_errors.begin(), it);
            double err = *it;
            // "mark" refinements with negation. We'll use this to vis
            // the refined regions on the error plots.
            if (coarse_map[i] < 0) err *= -1;
            err_ofs << i << " " << err << endl;
         }
      }


      if (cfl > 0) {
         dt = estimate_dt(cfl, mesh, sol, A, order);
      }

      double dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);

      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            sout << "solution\n" << mesh << rho_u << flush;
         }
      }

   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m vortex.mesh -g vortex-1-final.gf".
   {
      ofstream mesh_ofs("vortex-final.mesh");
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;


      for (int k = 0; k < num_equation; k++)
      {
         GridFunction uk(&fes, u_block.GetBlock(k));
         ostringstream sol_name;
         sol_name << "vortex-" << k << "-final.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 10. Compute the L2 solution error summed for all components.
   if (rseq.Size())
   {
      // const CoarseFineTransformations& cft = mesh.GetRefinementTransforms();
      // Table c2f;
      // cft.GetCoarseToFineMap(mesh, c2f);

      // Refine to the reference mesh by refining every coarse el that
      // wasn't refined.
      Array<int> refs;
      for (int i = 0; i < mesh.GetNE(); i++) {
         if (mesh.ncmesh->GetElementDepth(i) == 0) {
            refs.Append(i);
         }
      }
      mesh.GeneralRefinement(refs);

      amr_update(fes, dfes, vfes, u_block, rho, rho_u, rho_e, sol);

      Mesh mesh_ref("reference.mesh");
      ifstream rho_ifs("reference-rho.gf");
      ifstream rho_u_ifs("reference-rho-u.gf");
      ifstream rho_e_ifs("reference-rho-e.gf");
      GridFunction rho_ref(&mesh_ref, rho_ifs);
      GridFunction rho_u_ref(&mesh_ref, rho_u_ifs);
      GridFunction rho_e_ref(&mesh_ref, rho_e_ifs);
      rho_ref.Print();
      rho.Print();

      GridFunctionCoefficient rho_ref_coeff(&rho_ref);
      const double err = rho.ComputeL2Error(rho_ref_coeff);
      printf("final L2 error %e\n",err);
      // const double error = sol.ComputeLpError(2, u0);
      // cout << "Solution error: " << error << endl;
   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}
