
//                                MFEM Euler Example
//
// Compile with: make euler-href
//
// Sample runs:
//
//       euler-href -p 1 -r 2 -o 1 -s 3
//       euler-href -p 1 -r 1 -o 3 -s 4
//       euler-href -p 1 -r 0 -o 5 -s 6
//       euler-href -p 2 -r 1 -o 1 -s 3
//       euler-href -p 2 -r 0 -o 3 -s 3

#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
// Classes FE_Evolution, RiemannSolver, DomainIntegrator and FaceIntegrator
// shared between the serial and parallel version of the example.
#include "fem/auxiliary.hpp"


void Prefine(FiniteElementSpace & den_fes_old,
             FiniteElementSpace & mom_fes_old,
             FiniteElementSpace & sol_fes_old,
             GridFunction & den,
             GridFunction & mom,
             GridFunction & sol,
             GridFunction & pref_sol, GridFunction &orders_gf,
             double min_thresh, double max_thresh);

Table * Hrefine(GridFunction & den,
                GridFunction & mom,
                GridFunction &sol,
                GridFunction &ref_sol, Table * refT,
                double min_thresh, double max_thresh);

Table * Refine(Array<int> ref_actions,
               GridFunction & den,
               GridFunction & mom,
               GridFunction &sol,
               int depth_limit);

// Choice for the problem setup. See InitialCondition in ex18.hpp.
int problem;

enum ref_kind
{
   order,        // p refinement
   geometric     // h refinement
};

// Equation constant parameters.
const int num_equation = 4;
const double specific_heat_ratio = 1.4;
const double gas_constant = 8.3145;

// Maximum characteristic speed (updated by integrators)
double max_char_speed;

// Initial condition
void InitialCondition(const Vector &x, Vector &y)
{
   MFEM_ASSERT(x.Size() == 2, "");

   double radius = 0, Minf = 0, beta = 0;
   if (problem == 1)
   {
      // "Fast vortex"
      radius = 0.2;
      Minf = 0.5;
      beta = 1. / 5.;
   }
   else if (problem == 2)
   {
      // "Slow vortex"
      radius = 0.2;
      Minf = 0.05;
      beta = 1. / 50.;
   }
   else
   {
      mfem_error("Cannot recognize problem."
                 "Options are: 1 - fast vortex, 2 - slow vortex");
   }

   const double xc = 0.0, yc = 0.0;

   // Nice units
   const double vel_inf = 1.;
   const double den_inf = 1.;

   // Derive remainder of background state from this and Minf
   const double pres_inf = (den_inf / specific_heat_ratio) * (vel_inf / Minf) *
                           (vel_inf / Minf);
   const double temp_inf = pres_inf / (den_inf * gas_constant);

   double r2rad = 0.0;
   r2rad += (x(0) - xc) * (x(0) - xc);
   r2rad += (x(1) - yc) * (x(1) - yc);
   r2rad /= (radius * radius);

   const double shrinv1 = 1.0 / (specific_heat_ratio - 1.);

   const double velX = vel_inf * (1 - beta * (x(1) - yc) / radius * exp(
                                     -0.5 * r2rad));
   const double velY = vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
   const double vel2 = velX * velX + velY * velY;

   const double specific_heat = gas_constant * specific_heat_ratio * shrinv1;
   const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                       (vel_inf * beta) / specific_heat * exp(-r2rad);

   const double den = den_inf * pow(temp/temp_inf, shrinv1);
   const double pres = den * gas_constant * temp;
   const double energy = shrinv1 * pres / den + 0.5 * vel2;

   y(0) = den;
   y(1) = den * velX;
   y(2) = den * velY;
   y(3) = den * energy;
}
int ref_levels = 1;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 1;
   const char *mesh_file = "../data/periodic-square.mesh";
   int order = 2;
   int ode_solver_type = 4;
   double t_final = 2.0;
   double dt = 0.001;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 50;
   int refmode = 0;

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
   args.AddOption(&refmode, "-rm", "--refinement-mode",
                  "0: 'p', 1: 'h' ");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   ref_kind ref_mode = (ref_kind)refmode;


   // 2. Read the mesh from the given mesh file. This example requires a 2D
   //    periodic mesh, such as ../data/periodic-square.mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   MFEM_ASSERT(dim == 2, "Need a two-dimensional mesh for the problem definition");


   mesh.EnsureNCMesh();
   // compute reference solution
   Mesh ref_mesh(mesh);

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   ODESolver *ref_ode_solver = NULL;
   // ODESolver *pref_ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1:
         ode_solver = new ForwardEulerSolver;
         ref_ode_solver = new ForwardEulerSolver;
         // pref_ode_solver = new ForwardEulerSolver;
         break;
      case 2:
         ode_solver = new RK2Solver(1.0);
         ref_ode_solver = new RK2Solver(1.0);
         // pref_ode_solver = new RK2Solver(1.0);
         break;
      case 3:
         ode_solver = new RK3SSPSolver;
         ref_ode_solver = new RK3SSPSolver;
         // pref_ode_solver = new RK3SSPSolver;
         break;
      case 4:
         ode_solver = new RK4Solver;
         ref_ode_solver = new RK4Solver;
         // pref_ode_solver = new RK4Solver;
         break;
      case 6:
         ode_solver = new RK6Solver;
         ref_ode_solver = new RK6Solver;
         // pref_ode_solver = new RK6Solver;
         break;
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
      ref_mesh.UniformRefinement();
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   FiniteElementSpace den_fes(&mesh, &fec);
   // FiniteElementSpace den_fes_old(&mesh, &fec);
   // FiniteElementSpace pref_den_fes(&mesh, &fec);
   FiniteElementSpace ref_den_fes(&ref_mesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   FiniteElementSpace mom_fes(&mesh, &fec, dim, Ordering::byNODES);
   // FiniteElementSpace mom_fes_old(&mesh, &fec, dim, Ordering::byNODES);
   // FiniteElementSpace pref_mom_fes(&mesh, &fec, dim, Ordering::byNODES);
   FiniteElementSpace ref_mom_fes(&ref_mesh, &fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   FiniteElementSpace sol_fes(&mesh, &fec, num_equation, Ordering::byNODES);
   FiniteElementSpace ref_sol_fes(&ref_mesh, &fec, num_equation,
                                  Ordering::byNODES);
   // FiniteElementSpace sol_fes_old(&mesh, &fec, num_equation, Ordering::byNODES);
   // FiniteElementSpace pref_sol_fes(&mesh, &fec, num_equation, Ordering::byNODES);

   // for (int i = 0; i<mesh.GetNE(); i++)
   // {
      // int el_order = pref_den_fes.GetElementOrder(i);
      // pref_den_fes.SetElementOrder(i,el_order+2);
      // pref_mom_fes.SetElementOrder(i,el_order+2);
      // pref_sol_fes.SetElementOrder(i,el_order+2);
   // }
   // pref_den_fes.Update(false);
   // pref_mom_fes.Update(false);
   // pref_sol_fes.Update(false);


   // This example depends on this ordering of the space.
   MFEM_ASSERT(den_fes.GetOrdering() == Ordering::byNODES, "");

   cout << "Number of unknowns: " << sol_fes.GetVSize() << endl;

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // The solution u has components {density, x-momentum, y-momentum, energy}.
   // These are stored contiguously in the BlockVector u_block.
   Array<int> offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++) { offsets[k] = k * sol_fes.GetNDofs(); }
   BlockVector u_block(offsets);

   // Momentum grid function on dfes for visualization.
   GridFunction den(&den_fes, u_block.GetData());
   GridFunction mom(&mom_fes, u_block.GetData() + offsets[1]);

   // Initialize the state.
   VectorFunctionCoefficient u0(num_equation, InitialCondition);
   GridFunction sol(&sol_fes, u_block.GetData());
   sol.ProjectCoefficient(u0);

   // 7. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   MixedBilinearForm Aflux(&mom_fes, &den_fes);
   Aflux.AddDomainIntegrator(new TransposeIntegrator(new GradientIntegrator()));
   Aflux.Assemble();

   NonlinearForm A(&sol_fes);
   RiemannSolver rsolver(specific_heat_ratio, num_equation);
   A.AddInteriorFaceIntegrator(new FaceIntegrator(rsolver, dim, num_equation));

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   EulerSystem * euler = new EulerSystem(sol_fes, A, Aflux.SpMat(),
                                         specific_heat_ratio, num_equation);

   //-------------------------------------------
   Array<int> ref_offsets(num_equation + 1);
   // Array<int> pref_offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++)
   {
      ref_offsets[k] = k * ref_sol_fes.GetNDofs();
      // pref_offsets[k] = k * pref_sol_fes.GetNDofs();
   }
   BlockVector ref_u_block(ref_offsets);
   // BlockVector pref_u_block(pref_offsets);

   // // Momentum grid function on dfes for visualization.
   GridFunction ref_den(&ref_den_fes, ref_u_block.GetData());
   GridFunction ref_mom(&ref_mom_fes, ref_u_block.GetData() + ref_offsets[1]);
   GridFunction ref_sol(&ref_sol_fes, ref_u_block.GetData());

   // GridFunction pref_sol(&pref_sol_fes);
   // pref_sol.ProjectCoefficient(u0);

   // GridFunction pref_den(&pref_den_fes, pref_sol.GetData());
   // GridFunction pref_mom(&pref_mom_fes, pref_sol.GetData() + pref_offsets[1]);


   // MixedBilinearForm pref_Aflux(&pref_mom_fes, &pref_den_fes);
   // pref_Aflux.AddDomainIntegrator(new TransposeIntegrator(new
   //                                                        GradientIntegrator()));
   // pref_Aflux.Assemble();

   // NonlinearForm pref_A(&pref_sol_fes);
   // RiemannSolver pref_rsolver(specific_heat_ratio, num_equation);
   // pref_A.AddInteriorFaceIntegrator(new FaceIntegrator(pref_rsolver, dim,
   //                                                     num_equation));
   // EulerSystem pref_euler(pref_sol_fes, pref_A, pref_Aflux.SpMat(),
   //                        specific_heat_ratio, num_equation);


   ref_sol.ProjectCoefficient(u0);
   Array<int> refinements(ref_mesh.GetNE());
   refinements = 1;
   Table * T1 = Refine(refinements, ref_den, ref_mom,ref_sol,  2+ref_levels);
   refinements.SetSize(ref_mesh.GetNE());
   refinements = 1;
   Table * T2 = Refine(refinements, ref_den, ref_mom, ref_sol, 2+ref_levels);
   Table * refT = Mult(*T1,*T2);

   ref_sol.ProjectCoefficient(u0);

   for (int k = 0; k <= num_equation; k++)
   {
      ref_offsets[k] = k * ref_sol_fes.GetNDofs();
   }
   ref_den.MakeRef(&ref_den_fes,ref_sol.GetData());
   ref_mom.MakeRef(&ref_mom_fes,ref_sol.GetData()+ref_offsets[1]);

   MixedBilinearForm ref_Aflux(&ref_mom_fes, &ref_den_fes);
   ref_Aflux.AddDomainIntegrator(new TransposeIntegrator(new
                                                         GradientIntegrator()));
   ref_Aflux.Assemble();
   NonlinearForm ref_A(&ref_sol_fes);
   RiemannSolver ref_rsolver(specific_heat_ratio, num_equation);
   ref_A.AddInteriorFaceIntegrator(new FaceIntegrator(ref_rsolver, dim,
                                                      num_equation));
   EulerSystem ref_euler(ref_sol_fes, ref_A, ref_Aflux.SpMat(),
                         specific_heat_ratio, num_equation);

   // L2_FECollection orders_fec(0,dim);
   // FiniteElementSpace orders_fes(&mesh,&orders_fec);
   // GridFunction orders_gf(&orders_fes);
   // for (int i = 0; i<mesh.GetNE(); i++) { orders_gf(i) = order; }


   //-------------------------------------------

   // Visualize the density
   socketstream sout;
   socketstream ref_sout;
   // socketstream pref_sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      sout.open(vishost, visport);
      ref_sout.open(vishost, visport);
      // pref_sout.open(vishost, visport);
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
         sout << "solution\n" << mesh << mom 
              << "window_title 'Moment'" << flush;

         ref_sout.precision(precision);
         ref_sout << "solution\n" << ref_mesh << ref_mom
              << "window_title 'Reference Moment'" << flush;

         // pref_sout.precision(precision);
         // pref_sout << "solution\n" << mesh << pref_mom
         //      << "window_title 'PReference Moment'" << flush;

      }
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   double t = 0.0;
   euler->SetTime(t);
   double ref_t = 0.0;
   ref_euler.SetTime(ref_t);
   // double pref_t = 0.0;
   // pref_euler.SetTime(pref_t);

   ode_solver->Init(*euler);
   ref_ode_solver->Init(ref_euler);
   // pref_ode_solver->Init(pref_euler);
   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done; )
   {
      mfem::out << "ti = " << ti << endl;
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      ref_ode_solver->Step(ref_sol, ref_t, dt_real);
      // pref_ode_solver->Step(pref_sol, pref_t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         cout << "time step: " << ti << ", time: " << ref_t << endl;

         // if (ref_mode == ref_kind::geometric)
         // {
            refT = Hrefine(den,mom, sol, ref_sol, refT, 5e-5, 5e-4);
         // }
         // else
         // {
         //    Prefine(den_fes_old, mom_fes_old, sol_fes_old,
         //            den, mom, sol, pref_sol, orders_gf, 5e-5, 5e-4);
         // }

         // update offsets
         for (int k = 0; k <= num_equation; k++)
         {
            offsets[k] = k * sol_fes.GetNDofs();
         }

         den.MakeRef(&den_fes, sol.GetData());
         mom.MakeRef(&mom_fes, sol.GetData() + offsets[1]);

         Aflux.Update();
         Aflux.Assemble();
         A.Update();

         delete euler;
         euler = new EulerSystem(sol_fes, A, Aflux.SpMat(), specific_heat_ratio,
                                 num_equation);
         ode_solver->Init(*euler);

         if (visualization)
         {
            sout << "solution\n" << mesh << mom << flush;
            ref_sout << "solution\n" << ref_mesh << ref_mom << flush;
            // pref_sout << "solution\n" << mesh << pref_mom << flush;
         }
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 10. Compute the L2 solution error summed for all components.
   if (t_final == 2.0)
   {
      const double error = sol.ComputeLpError(2, u0);
      cout << "Solution error: " << error << endl;
   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}

Table * Hrefine(GridFunction & den,
                GridFunction & mom,
                GridFunction & sol,
                GridFunction &sol_ref,
                Table * refT,
                double min_thresh, double max_thresh)
{
   FiniteElementSpace * den_fes = den.FESpace();
   FiniteElementSpace * mom_fes = mom.FESpace();
   FiniteElementSpace * sol_fes = sol.FESpace();
   Mesh * mesh = den_fes->GetMesh();
   int ne = mesh->GetNE();
   Vector errors(ne);
   Mesh fine_mesh(*mesh);

   FiniteElementSpace den_fes_copy(&fine_mesh,den_fes->FEColl(), den_fes->GetVDim());
   FiniteElementSpace mom_fes_copy(&fine_mesh,mom_fes->FEColl(), mom_fes->GetVDim());
   FiniteElementSpace sol_fes_copy(&fine_mesh,sol_fes->FEColl(), sol_fes->GetVDim());
   GridFunction den_fine(&den_fes_copy);
   GridFunction mom_fine(&mom_fes_copy);
   GridFunction sol_fine(&sol_fes_copy);

   // copy data;
   den_fine = den;
   mom_fine = mom;
   sol_fine = sol;
   Array<int>refinements(fine_mesh.GetNE());
   refinements = 1;
   Table * T1 = Refine(refinements,den_fine, mom_fine, sol_fine,2+ref_levels);
   refinements.SetSize(fine_mesh.GetNE());
   refinements = 1;
   Table * T2 = Refine(refinements,den_fine, mom_fine, sol_fine,2+ref_levels);
   Table * T = Mult(*T1, *T2);

   delete T1;
   delete T2;

   // constract map
   int n = T->Size();
   int m = T->Width();
   Array<int> elem_map(m);
   for (int i = 0; i< n; i++)
   {
      int nr = T->RowSize(i);
      int * row = T->GetRow(i);
      int * ref_row = refT->GetRow(i);
      for (int j = 0; j<nr ; j++ )
      {
         elem_map[row[j]] = ref_row[j];
      }
   }

   GridFunction diff(sol_fine);
   // this needs to change for reordering
   diff-= sol_ref;

   Vector vzero(sol_fes->GetVDim()); vzero = 0.0;
   VectorConstantCoefficient zero(vzero);
   Vector fine_errors(fine_mesh.GetNE());
   diff.ComputeElementL2Errors(zero,fine_errors);

   for (int i = 0; i<T->Size(); i++)
   {
      int rs = T->RowSize(i);
      int *row = T->GetRow(i);
      double er = 0.;
      for (int j = 0; j<rs; j++)
      {
         er += fine_errors[row[j]]*fine_errors[row[j]];
      }
      errors[i] = sqrt(er);
   }

   Array<int> actions(ne);
   for (int i = 0; i<ne; i++)
   {
      double error = errors(i);
      if (error > max_thresh)
      {
         actions[i] = 1;
      }
      else if (error < min_thresh)
      {
         actions[i] = -1;
      }
      else
      {
         actions[i] = 0;
      }
   }

   Table * T3 = Refine(actions,den,mom,sol,1+ref_levels);

   if (T3)
   {
      Table *Ttt = Mult(*Transpose(*T3), *refT);
      delete T3;
      delete refT;
      refT = Ttt;
   }
   return refT;
}


void Prefine(FiniteElementSpace & den_fes_old,
             FiniteElementSpace & mom_fes_old,
             FiniteElementSpace & sol_fes_old,
             GridFunction &den,
             GridFunction &mom,
             GridFunction &sol,
             GridFunction &pref_sol, GridFunction &orders_gf,
             double min_thresh, double max_thresh)
{
   // get element errors
   FiniteElementSpace * den_fes = den.FESpace();
   FiniteElementSpace * mom_fes = mom.FESpace();
   FiniteElementSpace * sol_fes = sol.FESpace();
   int ne = den_fes->GetMesh()->GetNE();
   Vector errors(ne);

   GridFunction prsol(pref_sol.FESpace());
   PRefinementTransferOperator * P =
      new PRefinementTransferOperator(*sol_fes, *pref_sol.FESpace());
   P->Mult(sol,prsol);
   delete P;
   prsol-=pref_sol;
   ConstantCoefficient zero(0.0);
   prsol.ComputeElementL2Errors(zero,errors);
   for (int i = 0; i<ne; i++)
   {
      double error = errors(i);
      int order = den_fes->GetElementOrder(i);
      if (error < min_thresh && order > 1)
      {
         den_fes->SetElementOrder(i,order-1);
         mom_fes->SetElementOrder(i,order-1);
         sol_fes->SetElementOrder(i,order-1);
      }
      else if (error > max_thresh && order < 2)
      {
         den_fes->SetElementOrder(i, order+1);
         mom_fes->SetElementOrder(i, order+1);
         sol_fes->SetElementOrder(i, order+1);
      }
      else
      {
      }
   }

   den_fes->Update(false);
   mom_fes->Update(false);
   sol_fes->Update(false);

   PRefinementTransferOperator * T_den =
      new PRefinementTransferOperator(den_fes_old,*den_fes);
   PRefinementTransferOperator * T_mom =
      new PRefinementTransferOperator(mom_fes_old,*mom_fes);
   PRefinementTransferOperator * T_sol =
      new PRefinementTransferOperator(sol_fes_old,*sol_fes);

   GridFunction den_fine(den_fes);
   GridFunction mom_fine(mom_fes);
   GridFunction sol_fine(sol_fes);

   T_den->Mult(den,den_fine);
   T_mom->Mult(mom,mom_fine);
   T_sol->Mult(sol,sol_fine);

   // copy the orders to the old space
   for (int i = 0; i<ne; i++)
   {
      int order = den_fes->GetElementOrder(i);
      den_fes_old.SetElementOrder(i,order);
      mom_fes_old.SetElementOrder(i,order);
      sol_fes_old.SetElementOrder(i,order);
      orders_gf(i) = order;
   }
   den_fes_old.Update(false);
   mom_fes_old.Update(false);
   sol_fes_old.Update(false);

   delete T_den;
   delete T_mom;
   delete T_sol;
   // update old gridfuntion;
   den = den_fine;
   mom = mom_fine;
   sol = sol_fine;
}

Table * Refine(Array<int> ref_actions,
               GridFunction & den,
               GridFunction & mom,
               GridFunction &sol,
               int depth_limit)
{
   FiniteElementSpace * den_fes = den.FESpace();
   FiniteElementSpace * mom_fes = mom.FESpace();
   FiniteElementSpace * sol_fes = sol.FESpace();
   Mesh * mesh = den_fes->GetMesh();
   int ne = mesh->GetNE();

   //  ovewrite to no action if an element is marked for refinement but it exceeds the depth limit
   for (int i = 0; i<ne; i++)
   {
      int depth = mesh->ncmesh->GetElementDepth(i);
      if (depth >= depth_limit && ref_actions[i] == 1)
      {
         ref_actions[i] = 0;
      }
   }

   Array<int> actions(ne);
   Array<int> actions_marker(ne);
   actions_marker = 0;

   const Table & deref_table = mesh->ncmesh->GetDerefinementTable();

   for (int i = 0; i<deref_table.Size(); i++)
   {
      int n = deref_table.RowSize(i);
      const int * row = deref_table.GetRow(i);
      int sum_of_actions = 0;
      bool ref_flag = false;
      for (int j = 0; j<n; j++)
      {
         int action = ref_actions[row[j]];
         sum_of_actions+=action;
         if (action == 1)
         {
            ref_flag = true;
            break;
         }
      }
      if (ref_flag)
      {
         for (int j = 0; j<n; j++)
         {
            actions[row[j]] = max(0,ref_actions[row[j]]);
            actions_marker[row[j]] = 1;
         }
      }
      else
      {
         bool dref_flag = (2*abs(sum_of_actions) >= n) ? true : false;
         for (int j = 0; j<n; j++)
         {
            actions[row[j]] = (dref_flag) ? -1 : 0;
            actions_marker[row[j]] = 1;
         }
      }
   }

   for (int i = 0; i<ne; i++)
   {
      if (actions_marker[i] != 1)
      {
         if (ref_actions[i] == -1)
         {
            actions[i] = 0;
         }
         else
         {
            actions[i] = ref_actions[i];
         }
      }
   }

   // now the actions array holds feasible actions of -1,0,1
   Array<Refinement> refinements;
   for (int i = 0; i<ne; i++)
   {
      if (actions[i] == 1) {refinements.Append(Refinement(i,0b11));}
   }
   if (refinements.Size())
   {
      mesh->GeneralRefinement(refinements);
      den_fes->Update();
      mom_fes->Update();
      sol_fes->Update();
      den.Update();
      mom.Update();
      sol.Update();
      ne = mesh->GetNE();
   }

   Table * ref_table = nullptr;
   Table * dref_table = nullptr;
   // now the derefinements
   Array<int> new_actions(ne);
   if (refinements.Size())
   {
      new_actions = 1;
      const CoarseFineTransformations & tr = mesh->GetRefinementTransforms();
      ref_table = new Table();
      tr.MakeCoarseToFineTable(*ref_table);
      for (int i = 0; i<ref_table->Size(); i++)
      {
         int n = ref_table->RowSize(i);
         if (n == 1)
         {
            int * row = ref_table->GetRow(i);
            new_actions[row[0]] = actions[i];
         }
      }
   }
   else
   {
      new_actions = actions;
   }

   Vector dummy_errors(ne);
   dummy_errors = 1.0;
   for (int i = 0; i<ne; i++)
   {
      if (new_actions[i] < 0)
      {
         dummy_errors[i] = 0.;
      }
   }
   mesh->DerefineByError(dummy_errors,0.5);

   den_fes->Update();
   mom_fes->Update();
   sol_fes->Update();
   den.Update();
   mom.Update();
   sol.Update();

   if (mesh->GetNE() < ne)
   {
      const CoarseFineTransformations & tr =
         mesh->ncmesh->GetDerefinementTransforms();
      Table coarse_to_fine_table;
      tr.MakeCoarseToFineTable(coarse_to_fine_table);
      dref_table = Transpose(coarse_to_fine_table);
   }

   // Build combined table of mesh modifications
   Table * T = nullptr;
   if (ref_table && dref_table)
   {
      T = Mult(*ref_table, * dref_table);
      delete dref_table;
      delete ref_table;
   }
   else if (ref_table)
   {
      T = ref_table;
      delete dref_table;
   }
   else if (dref_table)
   {
      T= dref_table;
      delete ref_table;
   }
   else
   {
      // do nothing: no mesh modifications happened
   }

   return T;
}