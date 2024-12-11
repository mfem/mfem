//                               Parallel contact example
// mpirun -np 8  ./contact -ls 6 -sr 0 -testno 4 -nsteps 10 -omaxit 20 -nonlin
// mpirun -np 8  ./contact -ls 6 -sr 0 -testno 4 -nsteps 10 -omaxit 20 -lin

// mpirun -np 8  ./contact -ls 6 -sr 0 -testno 5 -nsteps 10 -omaxit 20 -nonlin
// mpirun -np 8  ./contact -ls 6 -sr 0 -testno 5 -nsteps 10 -omaxit 20 -lin

// mpirun -np 8  ./contact -ls 6 -sr 0 -testno 6 -nsteps 10 -omaxit 20 -nonlin
// mpirun -np 8  ./contact -ls 6 -sr 0 -testno 6 -nsteps 10 -omaxit 20 -lin

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ipsolver/ParIPsolver.hpp"

using namespace std;
using namespace mfem;

void OutputData(ostringstream & file_name, double E0, double Ef, int dofs, int constr, int optit, const Array<int> & iters)
{
   file_name << ".csv";
   std::ofstream outputfile(file_name.str().c_str());
   if (!outputfile.is_open()) 
   {
      MFEM_ABORT("Failed to open file for writing.\n");
   }
   outputfile << "Initial Energy objective        = " << E0 << endl;
   outputfile << "Final Energy objective          = " << Ef << endl;
   outputfile << "Global number of dofs           = " << dofs << endl;
   outputfile << "Global number of constraints    = " << constr << endl;
   outputfile << "Optimizer number of iterations  = " << optit << endl;
   outputfile << "CG iteration numbers            = "; iters.Print(outputfile, iters.Size());
   outputfile << "OptimizerIteration,CGIterations" << endl;
   for (int i = 0; i< iters.Size(); i++)
   {
      outputfile << i+1 <<","<< iters[i] << endl;
   }
   outputfile.close();   
   std::cout << " Data has been written to " << file_name.str().c_str() << endl;
}

void OutputFinalData(ostringstream & file_name, double E0, double Ef, int dofs, int constr, const std::vector<Array<int>> & iters)
{
   file_name << ".csv";
   std::ofstream outputfile(file_name.str().c_str());
   if (!outputfile.is_open()) 
   {
      MFEM_ABORT("Failed to open file for writing.\n");
   }
   outputfile << "Initial Energy objective        = " << E0 << endl;
   outputfile << "Final Energy objective          = " << Ef << endl;
   outputfile << "Global number of dofs           = " << dofs << endl;
   outputfile << "Global number of constraints    = " << constr << endl;
   outputfile << "TimeStep, OptimizerIterations" << endl;
   for (int i = 0; i < iters.size(); i++)
   {
      outputfile << i+1 <<","<< iters[i].Size() << endl;
   }
   outputfile << "CGIterations" << endl;
   for (int i = 0; i< iters.size(); i++)
   {
      for (int j = 0; j < iters[i].Size(); j++)
      {
         outputfile << iters[i][j] ;
         if (j < iters[i].Size()-1)
         {
            outputfile << ", ";
         }
      }
      outputfile << endl;
   }
   outputfile.close();   
   std::cout << " Data has been written to " << file_name.str().c_str() << endl;
}


int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();

   int sref = 1;
   int pref = 0;
   bool visualization = true;
   bool paraview = false;
   double linsolverrtol = 1e-10;
   double linsolveratol = 1e-12;
   int relax_type = 88;
   double optimizer_tol = 1e-6;
   int optimizer_maxit = 20;
   int linsolver = 2; // PCG  - AMG
   bool elast = false;
   int testNo = -1; // 0-6
   int nsteps = 1;
   int msteps = 0;
   bool outputfiles = false;
   bool doublepass = false;
   bool dynamicsolver = false;
   bool nonlinear = false;
   bool bound_constraints = true;
   bool qp = true;
   bool monitor = false;
   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&testNo, "-testno", "--test-number",
                  "Choice of test problem:"
                  "-1: default (original 2 block problem)"
                  "0: not implemented yet"
                  "1: not implemented yet"
                  "2: not implemented yet"
                  "3: not implemented yet"
                  "4: two block problem - diablo"
                  "41: two block problem - twisted"
                  "5: ironing problem"
                  "51: ironing problem extended"
                  "6: nested spheres problem");
   args.AddOption(&nonlinear, "-nonlin", "--nonlinear", "-lin",
                  "--linear", "Choice between linear and non-linear Elasticiy model.");          
   args.AddOption(&qp, "-qp", "--quadratic", "-no-qp",
                  "--non-quadratic", "Enable/disable quadratic approximation of the non-linear elasticity operator.");                              
   args.AddOption(&sref, "-sr", "--serial-refinements",
                  "Number of uniform refinements.");                  
   args.AddOption(&nsteps, "-nsteps", "--nsteps",
                  "Number of steps.");
   args.AddOption(&msteps, "-msteps", "--msteps",
                  "Number of extra steps.");                  
   args.AddOption(&pref, "-pr", "--parallel-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&linsolverrtol, "-srtol", "--solver-rel-tol",
                  "Linear Solver Relative Tolerance.");
   args.AddOption(&linsolveratol, "-satol", "--solver-abs-tol",
                  "Linear Solver Abs Tolerance.");
   args.AddOption(&elast, "-elast", "--elast", "-no-elast",
                  "--no-elast",
                  "Enable or disable AMG Elasticity options.");
   args.AddOption(&doublepass, "-doublepass", "--double-pass", "-singlepass",
                  "--single-pass",
                  "Enable or disable double pass for contact constraints.");                  
   args.AddOption(&optimizer_tol, "-otol", "--optimizer-tol",
                  "Interior Point Solver Tolerance.");
   args.AddOption(&optimizer_maxit, "-omaxit", "--optimizer-maxit",
                  "Interior Point Solver maximum number of iterations.");
   args.AddOption(&relax_type, "-rt", "--relax-type",
                  "Selection of Smoother for AMG");
   args.AddOption(&linsolver, "-ls", "--linear-solver",
                  "Selection of inner linear solver:" 
                  "0: mumps," 
                  "1: mumps-reduced,"
                  "2: PCG-AMG-reduced,"
                  "3: PCG- with block-diag(AMG,direct solver)"
                  "4: with static cond of contact dofs");
   args.AddOption(&dynamicsolver, "-dynamic-solver", "--dynamic-solver", "-no-dynamic-solver",
                  "--no-dynamic-solver",
                  "Enable or disable dynamic choice between AMG and two-level solver.");                  
   args.AddOption(&bound_constraints, "-bound-constraints", "--bound-constraints", "-no-bound-constraints",
		   "--no-bound-constraints",
		   "Enable or disable displacement bound constraints -eps <= d - dl <= eps");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&outputfiles, "-out", "--output", "-no-out",
                  "--no-ouput",
                  "Enable or disable ouput to files.");          
   args.AddOption(&monitor, "-monitor", "--monitor", "-no-monitor",
                  "--no-monitor",
                  "Enable or disable internal solution monitoring with paraview.");                                  
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   if (Mpi::Root())
   {
      mfem::out << "Solving test problem number: " << testNo << endl;
   }

   const char *mesh_file = nullptr;

   switch (testNo)
   {
      case -1:
         mesh_file = "meshes/two-block.mesh";
         break;
      case 0:
      case 1:
      case 2:
      case 3:
      {
         MFEM_ABORT("Problem not implemented yet");
         break;
      }
      case 4:
         mesh_file = "meshes/Test4.mesh";
         break;
      case 40:
         mesh_file = "meshes/Test40.mesh";
         break;   
      case 41:
         mesh_file = "meshes/Test41.mesh";
         break;
      case 42:
         mesh_file = "meshes/Test42.mesh";
         break;         
      case 43:
         mesh_file = "meshes/Test43.mesh";
         break;       
      case 44:
         mesh_file = "meshes/Test44.mesh";
         break;                         
      case 5:
         mesh_file = "meshes/Test5.mesh";
         break;
      case 51:
         mesh_file = "meshes/Test51.mesh";
         break;
      case 6:
         mesh_file = "meshes/Test6.mesh";
         break;
      case 61:
         // Something wrong with this mesh
         mesh_file = "meshes/Test61.mesh";
         break;
      case 62:
         mesh_file = "meshes/Test62.mesh";
         break;         
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }

   Mesh mesh(mesh_file,1);
   for (int i = 0; i<sref; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD,mesh);
   mesh.Clear();
   for (int i = 0; i<pref; i++)
   {
      pmesh.UniformRefinement();
   }

   Array<int> ess_bdr_attr;
   Array<int> ess_bdr_attr_comp;
   if (testNo == 6 || testNo == 61)
   {
      ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(1);
      ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(2);
      ess_bdr_attr.Append(4); ess_bdr_attr_comp.Append(0);
      ess_bdr_attr.Append(5); ess_bdr_attr_comp.Append(-1);
   }
   else if (testNo == 62)
   {
      ess_bdr_attr.Append(4); ess_bdr_attr_comp.Append(0);
      ess_bdr_attr.Append(5); ess_bdr_attr_comp.Append(-1);
   }
   else if (testNo == 40)
   {
      ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(-1);
      ess_bdr_attr.Append(10); ess_bdr_attr_comp.Append(-1);
   }
   else
   {
      ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(-1);
      ess_bdr_attr.Append(6); ess_bdr_attr_comp.Append(-1);
   }
   
  
   Vector E(pmesh.attributes.Max());
   Vector nu(pmesh.attributes.Max());

   if (testNo == -1 )
   {
      E = 1e2;
      nu = 0.3;
   }
   else if (testNo == 6 || testNo == 61 || testNo == 62)
   {
      E = 1e3;
      nu = 0.3;
   }
   else
   {
      E[0] = 1.0;  E[1] = 1e3;
      nu[0] = 0.45;  nu[1] = 0.0;
   }


   ElasticityOperator prob(&pmesh, ess_bdr_attr,ess_bdr_attr_comp, E,nu,nonlinear);

   int dim = pmesh.Dimension();
   Vector ess_values(dim);
   int essbdr_attr;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());

   ess_values = 0.0;

   std::set<int> mortar_attr;
   std::set<int> nonmortar_attr;

   if (testNo == 6 || testNo == 61)
   {
      ess_values = 0.0;
      ess_bdr = 0;
      ess_bdr[0] = 1;
      ess_bdr[1] = 1;
      ess_bdr[3] = 1;
      ess_bdr[4] = 1;
      prob.SetDisplacementDirichletData(ess_values, ess_bdr);
      ess_bdr = 0;
      ess_bdr[2] = 1;
      mortar_attr.insert(6);
      mortar_attr.insert(9);
      nonmortar_attr.insert(7);
      nonmortar_attr.insert(8);
   }
   else if(testNo == 62)
   {
      ess_values = 0.0;
      ess_bdr = 0;
      ess_bdr[3] = 1;
      ess_bdr[4] = 1;
      prob.SetDisplacementDirichletData(ess_values, ess_bdr);
      ess_bdr = 0;
      ess_bdr[2] = 1;
      // prob.SetNeumanData(0,3,-2.0);
      mortar_attr.insert(6);
      mortar_attr.insert(9);
      nonmortar_attr.insert(7);
      nonmortar_attr.insert(8);
   }
   else
   {
      if (testNo == -1 || testNo == 41)
      {
         ess_values[0] = 0.1/nsteps;
      }
      else
      {
         // ess_values[2] = 1.0/1.4/nsteps;
         ess_values[2] = 1.0/1.4/nsteps;
         // ess_values[0] = -2.0/nsteps;
      }
      essbdr_attr = (testNo == 40) ? 1 : 2;
      ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      essbdr_attr = (testNo == 40) ? 10 : 6;
      ess_values = 0.0; ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      if (testNo == 40)
      {
         mortar_attr.insert(4);
         nonmortar_attr.insert(7);
      }
      else
      {
         mortar_attr.insert(3);
         nonmortar_attr.insert(4);
      }
   }

   ParFiniteElementSpace * fes = prob.GetFESpace();
   Array<int> ess_tdof_list = prob.GetEssentialDofs();
   
   int gndofs = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      mfem::out << "--------------------------------------" << endl;
      mfem::out << "Global number of dofs = " << gndofs << endl;
      mfem::out << "--------------------------------------" << endl;
   }
   ParGridFunction x_gf(fes); x_gf = 0.0;
   ParGridFunction xnew(fes); xnew = 0.0;
   ParaViewDataCollection * paraview_dc = nullptr;
   ParMesh pmesh_copy(pmesh);
   ParFiniteElementSpace fes_copy(*fes,pmesh_copy);
   ParGridFunction xcopy_gf(&fes_copy); xcopy_gf = 0.0;


   ParGridFunction xBC(fes); xBC = 0.0;
   
   if (paraview)
   {
      std::ostringstream paraview_file_name;
      paraview_file_name << "QPContact-Test_" << testNo
                         << "_par_ref_" << pref
                         << "_ser_ref_" << sref;
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh_copy);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(1);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      // paraview_dc->RegisterField("u", &x_gf);
      paraview_dc->RegisterField("u", &xcopy_gf);
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(double(0));
      paraview_dc->Save();
   }
   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      sol_sock.open(vishost, visport);
      sol_sock.precision(8);
   }
   ParGridFunction ref_coords(prob.GetFESpace()); 
   ParGridFunction new_coords(prob.GetFESpace()); 
   pmesh.GetNodes(new_coords);
   pmesh.GetNodes(ref_coords);
   
   // deviation from the reference configuration
   Vector xref(x_gf.GetTrueVector().Size()); xref = 0.0;
   Vector xrefbc(x_gf.GetTrueVector().Size()); xrefbc = 0.0;

   // bound constraints
   // - eps <= x - xl <= eps
   // warning: eps_i = 0 will guarantee that LICQ is violated and
   // issues with the optimizer
   // eps_min > 0 ensures that this issue will not occur
   Vector xl(xref.Size()); xl = 0.0;
   Vector eps(xref.Size()); eps = 0.0;
   Vector dx(xref.Size()); dx = 0.0;
   double eps_min = 1.e-4;




   double p = 20.0;
   ConstantCoefficient f(p);
   std::vector<Array<int>> CGiter;
   int total_steps = nsteps + msteps;
   Vector DCvals;
   
   for (int i = 0; i<total_steps; i++)
   {
      if (testNo == 6)
      {
         ess_bdr = 0;
         ess_bdr[2] = 1;
         f.constant = -p*(i+1)/nsteps;
         prob.SetNeumanPressureData(f,ess_bdr);
      }
      else if (testNo == 4 || testNo == 40 || testNo == 5 || testNo == 51 || testNo == 43 || testNo == 44)
      {
         ess_bdr = 0;
         // essbdr_attr = (testNo == 40) ? 1 : 6;
         essbdr_attr = (testNo == 40) ? 1 : 2;
         ess_bdr[essbdr_attr-1] = 1;
         ess_values = 0.0;
         if (i < nsteps)
         {
            // ess_values[2] = -2.0/1.4*(i+1)/nsteps;
            ess_values[2] = 1.0/1.4*(i+1)/nsteps;
            // ess_values[2] = 0.0;
         }
         else
         {
            // ess_values[0] = 8.0/1.4*(i+1-nsteps)/msteps;
            // ess_values[2] = -2.0/1.4;
            ess_values[0] = -8.0/1.4*(i+1-nsteps)/msteps;
            ess_values[2] = 1.0/1.4;
         }
         prob.SetDisplacementDirichletData(ess_values, ess_bdr);
      }
      else if (testNo == 41)
      {
         ess_values = 0.0;
         ess_values[0] = 1.0/1.4*nsteps*(i+1);
         essbdr_attr =  2;
         ess_bdr[essbdr_attr-1] = 1;
         prob.SetDisplacementDirichletData(ess_values, ess_bdr);
         essbdr_attr = 6;
         ess_values = 0.0; 
         ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
         prob.SetDisplacementDirichletData(ess_values, ess_bdr);
      }
      else if (testNo == -1)
      {
         ess_values = 0.0;
         essbdr_attr = 2;
         ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
         ess_values[0] = 0.1/nsteps*(i+1);
         prob.SetDisplacementDirichletData(ess_values, ess_bdr);
      }

      prob.FormLinearSystem();
      x_gf.SetTrueVector();
     
      // xref will also satisfy the essential boundary conditions and the nonessential
      // dofs will be equal to the solution at the previous time step (if it exists)
      // or zero  
      // xref will be used to set the reference/expansion point used for the QPOptContactProblem
      // and also used as the initial point for the IP solver 
      if (i == 0)
      {
         // xref.Set(1.0, *x_gf.GetTrueDofs());      
         xref = 0.0;
         xrefbc = 0.0;
      }
      else
      {
         x_gf.GetTrueDofs(xref);      
         x_gf.GetTrueDofs(xrefbc);      
      }

      // set essential dofs with respect
      // to a deformation relative to the "frame" or 
      // the reference configuration given by the original mesh
      // xBC is a grid function that satisfies the essential boundary conditions
      xBC = 0.0;
      VectorConstantCoefficient xBC_cf(ess_values);
      xBC.ProjectBdrCoefficient(xBC_cf, ess_bdr);
      Vector xBCtrue;
      xBC.GetTrueDofs(xBCtrue);      
      xBCtrue.GetSubVector(ess_tdof_list, DCvals);
      xrefbc.SetSubVector(ess_tdof_list, DCvals);
   
      OptContactProblem contact(&prob, mortar_attr, nonmortar_attr, &new_coords, doublepass, xref,xrefbc,qp);
      
      if (testNo == 6 || testNo == 61)
      {
	 if( i > int(total_steps / 2) && bound_constraints)
	 {
            eps_min = max(eps_min, GlobalLpNorm(infinity(), eps.Normlinf(), MPI_COMM_WORLD));  
	    // update eps and set parameters
            for (int j = 0; j < eps.Size(); j++)
	    {
	       eps(j) = max(eps_min, eps(j));
	    }
	    xl.Set(1.0, xrefbc);
	    contact.SetBoundConstraints(xl, eps);
	 }
	 else if( i > 0)
	 {
	    for (int j = 0; j < eps.Size(); j++)
	    {
               eps(j) = max(eps(j), abs(dx(j)));
	    }
	 }
      }




      bool compute_dof_projections = (linsolver == 3 || linsolver == 6 || linsolver == 7) ? true : false;

      int gncols = (compute_dof_projections) ? contact.GetRestrictionToContactDofs()->GetGlobalNumCols() : -1;
      
      int numconstr = contact.GetGlobalNumConstraints();
      ParInteriorPointSolver optimizer(&contact);
      if (monitor)
      {
         optimizer.EnableMonitor();
      }

      if (dynamicsolver) optimizer.EnableDynamicSolverChoice();

      optimizer.SetTol(optimizer_tol);
      optimizer.SetMaxIter(optimizer_maxit);
      optimizer.SetLinearSolver(linsolver);
      optimizer.SetLinearSolveRelTol(linsolverrtol);
      optimizer.SetLinearSolveAbsTol(linsolveratol);
      optimizer.SetLinearSolveRelaxType(relax_type);

      if (elast)
      {
         optimizer.SetElasticityOptions(prob.GetFESpace());
      }

      x_gf.SetTrueVector();
      int ndofs = prob.GetFESpace()->GetTrueVSize();
      Vector x0(ndofs); x0 = 0.0;
      x0.Set(1.0, xrefbc);
      Vector xf(ndofs); xf = 0.0;

      optimizer.Mult(x0, xf);
      dx.Set(1.0, xf);
      dx.Add(-1.0, x0);



      double Einitial = contact.E(x0);
      double Efinal = contact.E(xf);
      Array<int> & CGiterations = optimizer.GetCGIterNumbers();
      Array<double> & DMaxMinRatios  = optimizer.GetDMaxMinRatios();
      CGiter.push_back(CGiterations);
      int gndofs = prob.GetGlobalNumDofs();
      if (Mpi::Root())
      {
         mfem::out << endl;
         mfem::out << " Initial Energy objective        = " << Einitial << endl;
         mfem::out << " Final Energy objective          = " << Efinal << endl;
         mfem::out << " Global number of dofs           = " << gndofs << endl;
         mfem::out << " Global number of constraints    = " << numconstr << endl;
         mfem::out << " Global number of contact dofs   = " << gncols << endl;
         mfem::out << " Optimizer number of iterations  = " <<
                optimizer.GetNumIterations() << endl;
         if (linsolver == 2 || linsolver == 3 || linsolver == 4 || linsolver == 6 || linsolver == 7)
         {
            mfem::out << " CG iteration numbers            = " ;
            for (int i = 0; i < CGiterations.Size(); ++i) 
            {
               std::cout << " " << std::setw(7) << CGiterations[i] << " |";
            }
            std::cout << std::endl;
            mfem::out << " D Max / Min Ratios              = " ;
            std::ios oldState(nullptr);
            oldState.copyfmt(std::cout);
            std::cout << std::setprecision(1) << std::scientific;
            for (int i = 0; i < DMaxMinRatios.Size(); ++i) 
            {
               std::cout << " " << DMaxMinRatios[i] << " |";
            }
            std::cout << std::endl;
            std::cout.copyfmt(oldState);

         }
         if (outputfiles)
         {
            ostringstream file_name;
            file_name << "output/test"<<testNo<<"/ref"<<sref+pref <<"/solver-"<<linsolver<<"-dynamic-"<<(int)dynamicsolver<<"-nsteps-" << nsteps << "-step-" << i; 
            OutputData(file_name, Einitial, Efinal, gndofs,numconstr, optimizer.GetNumIterations(), CGiterations);
            if (i == nsteps-1)
            {
               ostringstream final_file_name;
               final_file_name << "output/test"<<testNo<<"/ref"<<sref+pref <<"/solver-"<<linsolver<<"-dynamic-"<<(int)dynamicsolver<<"-nsteps-" << nsteps << "-final"; 
               OutputFinalData(final_file_name, Einitial, Efinal, gndofs, numconstr, CGiter);
            }
         }
      }

      x_gf.SetFromTrueDofs(xf);
      add(ref_coords,x_gf,new_coords);
      pmesh_copy.SetNodes(new_coords);
      xcopy_gf = x_gf;
      xcopy_gf.SetTrueVector();
      if (paraview)
      {
         paraview_dc->SetCycle(i+1);
         paraview_dc->SetTime(double(i+1));
         paraview_dc->Save();
      }

      if (visualization)
      {
        sol_sock << "parallel " << num_procs << " " << myid << "\n"
                 << "solution\n" << pmesh_copy << x_gf << flush;
      
        if (i == total_steps - 1)
        {
           pmesh.MoveNodes(x_gf);
           char vishost[] = "localhost";
           int  visport   = 19916;
           socketstream sol_sock1(vishost, visport);
           sol_sock1 << "parallel " << num_procs << " " << myid << "\n";
           sol_sock1.precision(8);
           sol_sock1 << "solution\n" << pmesh << x_gf << flush;
        }
      }
      if (i == total_steps-1) break;
      prob.UpdateRHS();
   }

   if (paraview_dc) delete paraview_dc;
   return 0;
}
