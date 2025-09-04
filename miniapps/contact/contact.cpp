// mpirun -np 4 ./contact -prob 0 -sr 1 -pr 0 -tr 2 -nsteps 4  -msteps 0 -amgf
// mpirun -np 4 ./contact -prob 0 -sr 1 -pr 0 -tr 2 -nsteps 4  -msteps 0 -no-amgf
// mpirun -np 4 ./contact -prob 1 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 6 -amgf
// mpirun -np 4 ./contact -prob 1 -sr 0 -pr 0 -tr 2 -nsteps 4  -msteps 6 -no-amgf
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 10 -msteps 0 -lin -amgf
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 10 -msteps 0 -lin -no-amgf
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 10 -msteps 0 -nonlin -amgf
// mpirun -np 4 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 10 -msteps 0 -nonlin -no-amgf

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "utils/ip.hpp"

using namespace std;
using namespace mfem;

enum problem_name
{
   twoblock,
   ironing,
   beamsphere
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();

   problem_name prob_name;

   int sref = 1;
   int pref = 0;
   bool visualization = true;
   bool paraview = false;
   int prob_no = 0; // 0,1,2
   int nsteps = 1;
   int msteps = 0;
   bool nonlinear = false;

   real_t tribol_ratio = 8.0;
   bool amgf = false;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&prob_no, "-prob", "--problem-number",
                  "Choice of problem:"
                  "0: two-block problem"
                  "1: ironing problem"
                  "2: beam-sphere problem");
   args.AddOption(&nonlinear, "-nonlin", "--nonlinear", "-lin",
                  "--linear", "Choice between linear and non-linear Elasticiy model.");
   args.AddOption(&sref, "-sr", "--serial-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&nsteps, "-nsteps", "--nsteps",
                  "Number of steps.");
   args.AddOption(&msteps, "-msteps", "--msteps",
                  "Number of extra steps.");
   args.AddOption(&pref, "-pr", "--parallel-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&amgf, "-amgf", "--amgf", "-no-amgf",
                  "--no-amgf",
                  "Enable or disable AMGF with Filtering solver.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&tribol_ratio, "-tr", "--tribol-proximity-parameter",
                  "Tribol-proximity-parameter.");

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


   MFEM_VERIFY(prob_no >= 0 &&
               prob_no <= 2, "Unknown test problem number: " << prob_no);

   prob_name = (problem_name)prob_no;

   if (nonlinear && prob_name!=problem_name::beamsphere)
   {
      if (myid == 0)
      {
         cout << "Non-linear elasticity not supported for the two-block and ironing problems"
              << endl;
         cout << "Switching to the linear model ..." << endl;
      }
      nonlinear = false;
   }


   bool bound_constraints = (nonlinear) ? true : false;

   const char *mesh_file = nullptr;

   switch (prob_name)
   {
      case twoblock:
         mesh_file = "meshes/two-block.mesh";
         break;
      case ironing:
         mesh_file = "meshes/ironing.mesh";
         break;
      case beamsphere:
         mesh_file = "meshes/beam-sphere.mesh";
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


   Vector E(pmesh.attributes.Max());
   Vector nu(pmesh.attributes.Max());

   Array<int> ess_bdr_attr;
   Array<int> ess_bdr_attr_comp;
   switch (prob_name)
   {
      case twoblock:
         ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(-1);
         ess_bdr_attr.Append(10); ess_bdr_attr_comp.Append(-1);
         E[0] = 1.0;  E[1] = 1e3;
         nu[0] = 0.499;  nu[1] = 0.0;
         break;
      case ironing:
         ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(-1);
         ess_bdr_attr.Append(6); ess_bdr_attr_comp.Append(-1);
         E[0] = 1.0;  E[1] = 1e3;
         nu[0] = 0.499;  nu[1] = 0.0;
         break;
      case beamsphere:
         ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(1);
         ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(2);
         ess_bdr_attr.Append(4); ess_bdr_attr_comp.Append(0);
         ess_bdr_attr.Append(5); ess_bdr_attr_comp.Append(-1);
         E = 1.e3;
         nu = 0.3;
         break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }

   ElasticityOperator prob(&pmesh, ess_bdr_attr,ess_bdr_attr_comp, E, nu,
                           nonlinear);

   int dim = pmesh.Dimension();
   Vector ess_values(dim);
   int essbdr_attr;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());

   ess_values = 0.0;

   std::set<int> mortar_attr;
   std::set<int> nonmortar_attr;

   switch (prob_name)
   {
      case twoblock:
         mortar_attr.insert(4);
         nonmortar_attr.insert(7);
         break;
      case  ironing:
         mortar_attr.insert(3);
         nonmortar_attr.insert(4);
         break;
      case beamsphere:
         mortar_attr.insert(6);
         mortar_attr.insert(9);
         nonmortar_attr.insert(7);
         nonmortar_attr.insert(8);
         break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
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
   ParMesh pmesh_copy(pmesh);
   ParFiniteElementSpace fes_copy(*fes,pmesh_copy);
   ParGridFunction xcopy_gf(&fes_copy); xcopy_gf = x_gf;

   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      std::ostringstream paraview_file_name;
      paraview_file_name << "contact-problem_" << prob_no
                         << "_par_ref_" << pref
                         << "_ser_ref_" << sref
                         << "_nonlinear_" << nonlinear;
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh_copy);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(2);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->RegisterField("u", &xcopy_gf);
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
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

   real_t p = 30.0;
   ConstantCoefficient f(p);
   std::vector<Array<int>> CGiter;
  
   OptContactProblem contact(&prob, mortar_attr, nonmortar_attr,
                              tribol_ratio, bound_constraints);

   int total_steps = nsteps + msteps;
   for (int i = 0; i<total_steps; i++)
   {
      switch (prob_name)
      {
         case twoblock:
            ess_bdr = 0;
            essbdr_attr = 10;
            ess_bdr[essbdr_attr-1] = 1;
            ess_values = 0.0;
            if (i < nsteps)
            {
               ess_values[2] = -1.0/1.4*(i+1)/nsteps;
            }
            else
            {
               ess_values[0] = 6.0/1.4*(i+1-nsteps)/msteps;
               ess_values[2] = -1.0/1.4;
            }
            prob.SetDisplacementDirichletData(ess_values, ess_bdr);
            break;
         case ironing:
            ess_bdr = 0;
            essbdr_attr = 6;
            ess_bdr[essbdr_attr-1] = 1;
            ess_values = 0.0;
            if (i < nsteps)
            {
               ess_values[2] = -1.0/1.4*(i+1)/nsteps;
            }
            else
            {
               ess_values[0] = 3.0/1.4*(i+1-nsteps)/msteps;
               ess_values[2] = -1.0/1.4;
            }
            prob.SetDisplacementDirichletData(ess_values, ess_bdr);
            break;
         case beamsphere:
            ess_bdr = 0;
            ess_bdr[2] = 1;
            f.constant = -p*(i+1)/nsteps;
            prob.SetNeumanPressureData(f,ess_bdr);
            break;
         default:
            MFEM_ABORT("Should be unreachable");
            break;
      }
      
      prob.FormLinearSystem();

      // deviation from the reference configuration
      Vector xref(x_gf.GetTrueVector().Size()); xref = 0.0;
      x_gf.SetTrueVector();
      x_gf.GetTrueDofs(xref);
     
      // TODO: update logic to remove this if/else statement 
      if (i == 0)
      {
         contact.FormContactSystem(&new_coords, xref);
      }
      else
      {
         contact.UpdateContactSystem(&new_coords, xref);
      }
      if (bound_constraints)
      {
         contact.SetBoundConstraints(i);
      }


      Solver * prec = nullptr;
      Solver * subspacesolver = nullptr;
      if (amgf)
      {
#ifdef MFEM_USE_MUMPS
         subspacesolver = new MUMPSSolver(MPI_COMM_WORLD);
         dynamic_cast<MUMPSSolver*>(subspacesolver)->SetPrintLevel(0);
#else
#ifdef MFEM_USE_MKL_CPARDISO
         subspacesolver = new CPardisoSolver(MPI_COMM_WORLD);
#else
         MFEM_ABORT("MFEM must be built with MUMPS or MKL_CPARDISO in order to use AMGF");
#endif
#endif
         prec = new AMGFSolver();
         auto * amgfprec = dynamic_cast<AMGFSolver *>(prec);
         amgfprec->AMG().SetSystemsOptions(3);
         amgfprec->AMG().SetPrintLevel(0);
         amgfprec->AMG().SetRelaxType(88);
         amgfprec->SetFilteredSubspaceSolver(*subspacesolver);
         amgfprec->SetFilteredSubspaceTransferOperator(
            *contact.GetContactSubspaceTransferOperator());
      }
      else
      {
         prec = new HypreBoomerAMG();
         auto * amgprec = dynamic_cast<HypreBoomerAMG *>(prec);
         amgprec->SetSystemsOptions(3);
         amgprec->SetPrintLevel(0);
         amgprec->SetRelaxType(88);
      }

      CGSolver cgsolver(MPI_COMM_WORLD);
      cgsolver.SetPrintLevel(3);
      cgsolver.SetRelTol(1e-10);
      cgsolver.SetMaxIter(10000);
      cgsolver.SetPreconditioner(*prec);

      IPSolver optimizer(&contact);
      optimizer.SetTol(1e-6);
      optimizer.SetMaxIter(100);
      optimizer.SetLinearSolver(&cgsolver);
      optimizer.SetUsingMassWeights(true);
      optimizer.SetPrintLevel(0);

      x_gf.SetTrueVector();
      int ndofs = prob.GetFESpace()->GetTrueVSize();
      Vector x0(ndofs); x0 = 0.0;
      x0.Set(1.0, xref);
      Vector xf(ndofs); xf = 0.0;
      optimizer.Mult(x0, xf);

      delete prec;
      if (subspacesolver) { delete subspacesolver; }

      Vector dx(ndofs); dx = 0.0;
      dx.Set(1.0, xf);
      dx.Add(-1.0, x0);
      contact.SetTimeStepDisplacement(i, dx);

      int eval_err;
      real_t Einitial = contact.E(x0, eval_err);
      real_t Efinal = contact.E(xf, eval_err);
      Array<int> & CGiterations = optimizer.GetCGNumIterations();
      CGiter.push_back(CGiterations);
      int gndofs = prob.GetGlobalNumDofs();

      if (Mpi::Root())
      {
         mfem::out << endl;
         mfem::out << " Initial Energy objective        = " << Einitial << endl;
         mfem::out << " Final Energy objective          = " << Efinal << endl;
         mfem::out << " Global number of dofs           = " << gndofs << endl;
         mfem::out << " Optimizer number of iterations  = " <<
                   optimizer.GetNumIterations() << endl;
         mfem::out << " CG iteration numbers            = " ;
         for (int i = 0; i < CGiterations.Size(); ++i)
         {
            std::cout << " " << std::setw(7) << CGiterations[i] << " |";
         }
         mfem::out << "\n";
      }

      x_gf.SetFromTrueDofs(xf);
      add(ref_coords,x_gf,new_coords);
      pmesh_copy.SetNodes(new_coords);
      xcopy_gf = x_gf;
      xcopy_gf.SetTrueVector();
      if (paraview)
      {
         paraview_dc->SetCycle(i+1);
         paraview_dc->SetTime(real_t(i+1));
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
            socketstream sol_sock_final(vishost, visport);
            sol_sock_final << "parallel " << num_procs << " " << myid << "\n";
            sol_sock_final.precision(8);
            sol_sock_final << "solution\n" << pmesh << x_gf << flush;
         }
      }


      if (i == total_steps-1) { break; }
      prob.UpdateRHS();
   }

   if (paraview_dc) { delete paraview_dc; }
   return 0;
}
