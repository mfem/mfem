//                               Parallel contact example
//
// Compile with: make pcontact_driver
// sample run
// mpirun -np 6 ./pcontact_driver -sr 2 -pr 2
// mpirun -np 8 ./pcontact_single_mesh -ls 2 -sr 5 -omaxit 20  -otol 1e-6 -tribol -rt 18
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ipsolver/ParIPsolver.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();

   int order = 1;
   int sref = 0;
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
   bool enable_tribol = true;
   int linsolver = 2; // PCG  - AMG
   bool elast = false;
   bool nocontact = false;
   int testNo = 4; // 0-6
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
   args.AddOption(&enable_tribol, "-tribol", "--tribol", "-no-tribol",
                  "--no-tribol",
                  "Enable or disable Tribol interface.");
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
                  "Selection of inner linear solver: 0: mumps, 1: mumps-reduced,",
                  "2: PCG-AMG-reduced, 3 PCG- with block-diag(AMG,direct solver), 4: with static cont of contact dofs");
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
         mesh_file = "meshes/merged_new.mesh";
         break;
      case 0:
      case 1:
      case 2:
      case 3:
      case 6:
      {
         MFEM_ABORT("Problem not implemented yet");
         break;
      }
      case 4:
         mesh_file = "meshes/test4.mesh";
         break;
      case 41:
         mesh_file = "meshes/newmesh1.mesh";
         break;
      case 5:
         mesh_file = "meshes/iron.mesh";
         break;
      case 51:
         mesh_file = "meshes/iron-extended.mesh";
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

   Array<int> part;
   Array<int> attr1, attr2;
   attr1.Append(1);
   attr2.Append(2);
   SubMesh mesh1 = SubMesh::CreateFromDomain(*mesh,attr1);
   SubMesh mesh2 = SubMesh::CreateFromDomain(*mesh,attr2);

   Array<int> part1(mesh1.GeneratePartitioning(num_procs),mesh1.GetNE());
   Array<int> part2(mesh2.GeneratePartitioning(num_procs),mesh2.GetNE());

   part.Append(part1);
   part.Append(part2);

   ParMesh * pmesh = new ParMesh(MPI_COMM_WORLD,*mesh,part.GetData());
   // ParMesh * pmesh = new ParMesh(MPI_COMM_WORLD,*mesh);

   for (int i = 0; i<pref; i++)
   {
      pmesh->UniformRefinement();
   }

   // MFEM_VERIFY(pmesh->GetNE(), "Empty partition pmesh");

   Array<int> ess_bdr_attr;
   Array<int> ess_bdr_attr_comp;

   ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(-1);
   ess_bdr_attr.Append(6); ess_bdr_attr_comp.Append(-1);
   ParElasticityProblem * prob = new ParElasticityProblem(pmesh,
                                                          ess_bdr_attr,ess_bdr_attr_comp,
                                                          order);
   Vector lambda(prob->GetMesh()->attributes.Max());
   lambda[1] = 0.0;
   lambda[0] = 0.499/(1.499*0.002);

   Vector mu(prob->GetMesh()->attributes.Max());
   mu[1] = 500;
   mu[0] = 1./(2*1.499);
   prob->SetLambda(lambda); prob->SetMu(mu);

   int dim = pmesh->Dimension();
   Vector ess_values(dim);
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   int essbdr_attr = 2;

   ess_values = 0.0;
   if (testNo == -1 || testNo == 41)
   {
      ess_values[0] = 0.1;
   }
   else
   {
      ess_values[2] = 0.7;
   }
   ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
   prob->SetDisplacementDirichletData(ess_values, ess_bdr);
   essbdr_attr = 6;
   ess_values = 0.0; ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
   prob->SetDisplacementDirichletData(ess_values, ess_bdr);


   std::set<int> mortar_attr({3});
   std::set<int> nonmortar_attr({4});

   ParContactProblemSingleMesh contact(prob, mortar_attr, nonmortar_attr,
                                       enable_tribol);
   QPOptParContactProblemSingleMesh qpopt(&contact);
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
      // x_gf*=-1.0;

      pmesh->MoveNodes(x_gf);

      if (paraview)
      {
         std::ostringstream paraview_file_name;
         paraview_file_name << "QPContactBody"
                            <<"_Tribol_" << (int)enable_tribol
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
