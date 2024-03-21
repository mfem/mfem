//                               Parallel contact example
// mpirun -np 8 ./contact -ls 2 -sr 2 -testno 4
// CG iteration numbers = 110 122 126 126 125 123 120 151 116 112 110 107 175 287 282 318 340

// mpirun -np 8 ./contact -ls 2 -sr 1 -testno 5
// CG iteration numbers = 119 130 130 130 129 127 124 121 120 117 115 142 287 801

// mpirun -np 8 ./contact -ls 2 -sr 0 -testno 6
// CG iteration numbers = 18 18 18 17 18 18 17 17 30 27 108 116 143 150 348 1099
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

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();

   int order = 1;
   int sref = 1;
   int pref = 0;
   Array<int> attr;
   Array<int> m_attr;
   bool visualization = true;
   bool paraview = false;
   double linsolverrtol = 1e-10;
   double linsolveratol = 1e-12;
   int relax_type = 8;
   double optimizer_tol = 1e-6;
   int optimizer_maxit = 20;
   int linsolver = 2; // PCG  - AMG
   bool elast = false;
   bool nocontact = false;
   int testNo = -1; // 0-6
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
   args.AddOption(&attr, "-at", "--attributes-surf",
                  "Attributes of boundary faces on contact surface for mesh 2.");
   args.AddOption(&sref, "-sr", "--serial-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&pref, "-pr", "--parallel-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&linsolverrtol, "-srtol", "--solver-rel-tol",
                  "Linear Solver Relative Tolerance.");
   args.AddOption(&linsolveratol, "-satol", "--solver-abs-tol",
                  "Linear Solver Abs Tolerance.");
   args.AddOption(&elast, "-elast", "--elast", "-no-elast",
                  "--no-elast",
                  "Enable or disable AMG Elasticity options.");
   args.AddOption(&nocontact, "-nocontact", "--nocontact", "-no-nocontact",
                  "--no-nocontact",
                  "Enable or disable AMG solve with no contact for testing.");
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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
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
      case 41:
         mesh_file = "meshes/Test41.mesh";
         break;
      case 42:
         mesh_file = "meshes/Test42.mesh";
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
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }

   Mesh * mesh = new Mesh(mesh_file,1);
   for (int i = 0; i<sref; i++)
   {
      mesh->UniformRefinement();
   }

   ParMesh * pmesh = new ParMesh(MPI_COMM_WORLD,*mesh);

   for (int i = 0; i<pref; i++)
   {
      pmesh->UniformRefinement();
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
   else
   {
      ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(-1);
      ess_bdr_attr.Append(6); ess_bdr_attr_comp.Append(-1);
   }
   ParElasticityProblem * prob = new ParElasticityProblem(pmesh,
                                                          ess_bdr_attr,ess_bdr_attr_comp,
                                                          order);
   Vector lambda(prob->GetMesh()->attributes.Max());


   Vector mu(prob->GetMesh()->attributes.Max());


   if (testNo == -1 )
   {
      lambda = 57.6923076923;
      mu = 38.4615384615;
   }
   else if (testNo == 6 || testNo == 61 )
   {
      lambda = (1000*0.3)/(1.3*0.4);
      mu = 500/(1.3);
   }
   else
   {
      lambda[0] = 0.499/(1.499*0.002);
      lambda[1] = 0.0;
      mu[0] = 1./(2*1.499);
      mu[1] = 500;
   }

   prob->SetLambda(lambda); prob->SetMu(mu);

   int dim = pmesh->Dimension();
   Vector ess_values(dim);
   int essbdr_attr;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());

   ess_values = 0.0;
   ConstantCoefficient one(-1.0);

   std::set<int> mortar_attr;
   std::set<int> nonmortar_attr;

   if (testNo == 6 || testNo == 61)
   {
      /* material 1: e = 1000
                     nu = 0.3
         material 2: e = 1000
                     nu = 0.3
         material 3: e = 1000
                     nu = 0.3
         Attr    value
         1      Dirichlet y = 0
         2      Dirichlet z = 0
         3      Neuman = 1.0
         4      Dirichlet x = 0
         5      Dirichlet x=y=z=0
         6      Mortar
         7      NonMortar
         8      NonMortar
         9      Mortar
      */
      ess_values = 0.0;
      ess_bdr = 0;
      ess_bdr[0] = 1;
      ess_bdr[1] = 1;
      ess_bdr[3] = 1;
      ess_bdr[4] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      ess_bdr = 0;
      ess_bdr[2] = 1;
      prob->SetNeumanPressureData(one,ess_bdr);
      mortar_attr.insert(6);
      mortar_attr.insert(9);
      nonmortar_attr.insert(7);
      nonmortar_attr.insert(8);
   }
   else
   {
      if (testNo == -1 || testNo == 41)
      {
         ess_values[0] = 0.1;
      }
      else
      {
         ess_values[2] = 0.7;
      }
      essbdr_attr = 2;
      ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      essbdr_attr = 6;
      ess_values = 0.0; ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      mortar_attr.insert(3);
      nonmortar_attr.insert(4);
   }



   ParContactProblem contact(prob, mortar_attr, nonmortar_attr);
   QPOptParContactProblem qpopt(&contact);
   int numconstr = contact.GetGlobalNumConstraints();
   ParInteriorPointSolver optimizer(&qpopt);
   optimizer.SetTol(optimizer_tol);
   optimizer.SetMaxIter(optimizer_maxit);
   optimizer.SetLinearSolver(linsolver);
   optimizer.SetLinearSolveRelTol(linsolverrtol);
   optimizer.SetLinearSolveAbsTol(linsolveratol);
   optimizer.SetLinearSolveRelaxType(relax_type);
   if (nocontact)
   {
      optimizer.EnableNoContactSolve();
   }
   if (elast)
   {
      optimizer.SetElasticityOptions(prob->GetFESpace());
   }
   ParGridFunction x = prob->GetDisplacementGridFunction();
   Vector x0 = x.GetTrueVector();
   int ndofs = x0.Size();
   Vector xf(ndofs); xf = 0.0;
   optimizer.Mult(x0, xf);
   double Einitial = contact.E(x0);
   double Efinal = contact.E(xf);
   Array<int> & CGiterations = optimizer.GetCGIterNumbers();
   int gndofs = prob->GetGlobalNumDofs();
   if (Mpi::Root())
   {
      mfem::out << endl;
      mfem::out << " Initial Energy objective        = " << Einitial << endl;
      mfem::out << " Final Energy objective          = " << Efinal << endl;
      mfem::out << " Global number of dofs           = " << gndofs << endl;
      mfem::out << " Global number of constraints    = " << numconstr << endl;
      mfem::out << " Optimizer number of iterations  = " <<
                optimizer.GetNumIterations() << endl;
      if (linsolver == 2 || linsolver == 3 || linsolver == 4)
      {
         mfem::out << " CG iteration numbers            = " ;
         CGiterations.Print(mfem::out, CGiterations.Size());
      }
      if (nocontact)
      {
         Array<int> & CGNoContactIterations = optimizer.GetCGNoContactIterNumbers();
         mfem::out << " CG no Contact iteration numbers = " ;
         CGNoContactIterations.Print(mfem::out, CGNoContactIterations.Size());
      }
      ostringstream file_name;
      file_name << "output/Testno-"<<testNo<<"-sref-"<<sref; 
      OutputData(file_name, Einitial, Efinal, gndofs,numconstr, optimizer.GetNumIterations(), CGiterations);
   }

   // MFEM_VERIFY(optimizer.GetConverged(),
   //             "Interior point solver did not converge.");


   if (visualization || paraview)
   {
      ParFiniteElementSpace * fes = prob->GetFESpace();
      ParMesh * pmesh = fes->GetParMesh();

      Vector X_new(xf.GetData(),fes->GetTrueVSize());

      ParGridFunction x_gf(fes);

      x_gf.SetFromTrueDofs(X_new);

      pmesh->MoveNodes(x_gf);

      if (paraview)
      {
         std::ostringstream paraview_file_name;
         paraview_file_name << "QPContactBodyTribol"
                            << "_par_ref_" << pref
                            << "_ser_ref_" << sref;
         ParaViewDataCollection paraview_dc(paraview_file_name.str(), pmesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(1);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetCycle(0);
         paraview_dc.SetTime(0.0);
         paraview_dc.RegisterField("Body", &x_gf);
         paraview_dc.Save();
      }


      if (visualization)
      {
         char vishost[] = "localhost";
         int visport = 19916;

         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << *pmesh << x_gf << flush;
      }
   }

   delete prob;
   delete pmesh;
   delete mesh;
   return 0;
}
