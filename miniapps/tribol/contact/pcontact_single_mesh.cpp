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
   // 1. Parse command-line options.
   // const char *mesh_file = "meshes/merged.mesh";
   // const char *mesh_file = "meshes/merged_new.mesh";
   // const char *mesh_file = "meshes/newmesh1.mesh";
   // const char *mesh_file = "meshes/iron.mesh";
   const char *mesh_file = "meshes/iron-extended.mesh";
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
   bool enable_tribol = false;
   int linsolver = 2; // PCG  - AMG
   bool elast = false;
   bool nocontact = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
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
                  "Selection of inner linear solver: 0: mumps, 1: mumps-reduced, 2: PCG-AMG-reduced");
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

   for (int i = 0; i<pref; i++)
   {
      pmesh->UniformRefinement();
   }

   MFEM_VERIFY(pmesh->GetNE(), "Empty partition pmesh");

   Array<int> ess_bdr_attr;
   Array<int> ess_bdr_attr_comp;

   bool ironing_problem = false;
   if (ironing_problem)
   {
      ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(2);
      ess_bdr_attr.Append(3); ess_bdr_attr_comp.Append(0);
      ess_bdr_attr.Append(4); ess_bdr_attr_comp.Append(1);
      ess_bdr_attr.Append(6); ess_bdr_attr_comp.Append(-1);
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
   lambda[1] = 0.0;
   lambda[0] = 0.499/(1.499*0.002);

   Vector mu(prob->GetMesh()->attributes.Max());
   mu[1] = 500;
   mu[0] = 1./(2*1.499);
   prob->SetLambda(lambda); prob->SetMu(mu);

   int dim = pmesh->Dimension();
   Vector ess_values(dim);
   // Dirichlet BCs attributes
   // boundary attributes:
   // 2. bottom plane of bottom body
   // 3. top plane of bottom body
   // 4. bottom surface of top body
   // 6. top surface of top body
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   int essbdr_attr;
   if (ironing_problem)
   {
      // bottom body - bottom plane
      essbdr_attr = 2;
      ess_values = 0.0; ess_values[2] = -1.0;
      ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      // bottom body - top plane
      essbdr_attr = 3;
      ess_values = 0.0; ess_values[0] = -1.0;
      ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      // top body - bottom surface
      essbdr_attr = 4;
      ess_values = 0.0;
      ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      // top body - top surface
      essbdr_attr = 6;
      ess_values = 0.0;
      ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
   }
   else
   {
      essbdr_attr = 2;
      ess_values = 0.0; ess_values[2] = 0.7;
      ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      essbdr_attr = 6;
      ess_values = 0.0; ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
   }

   ParContactProblemSingleMesh contact(prob, enable_tribol);
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
