//                                MFEM Example 26
//

#include "mfem.hpp"
#include "exact_sol.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "element-smoother.hpp"
using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   const char *mesh_file = "l-shape-benchmark.mesh";
   int init_geometric_refinements = 0;
   int pinit_geometric_refinements = 0;
   int geometric_refinements = 0;
   int order_refinements = 2;
   const char *device_config = "cpu";
   bool visualization = true;
   int order = 1;
   int solver = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&solver, "-solver", "--solver", "Solver: 0:MG-Cheb-Jac, 1: MG-Cheb-ElemSmoother");
   args.AddOption(&init_geometric_refinements, "-ref", "--initial-geometric-refinements",
                  "Number of serial geometric refinements defining the coarse mesh.");
   args.AddOption(&pinit_geometric_refinements, "-pref", "--initial-geometric-refinements",
                  "Number of parallel geometric refinements defining the coarse mesh.");                  
   args.AddOption(&geometric_refinements, "-gr", "--geometric-refinements",
                  "Number of geometric refinements done prior to order refinements.");
   args.AddOption(&order_refinements, "-or", "--order-refinements",
                  "Number of order refinements. Finest level in the hierarchy has order 2^{or}.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int l = 0; l < init_geometric_refinements; l++)
   {
      mesh->UniformRefinement();
   }
   mesh->EnsureNCMesh();
   ParMesh * pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   mesh->Clear();
   {
      for (int l = 0; l < pinit_geometric_refinements; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   Array<int> ess_bdr;
   if(pmesh->bdr_attributes.Size())
   {   
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
   }


   ParGridFunction x(fespace);
   FunctionCoefficient ex_coeff(lshape_exact);
   x.ProjectCoefficient(ex_coeff);

   // ------------------------------------------------- 
   // Bilinear and linear forms
   // ------------------------------------------------- 
   ConstantCoefficient cf(1.0);
   ParBilinearForm a(fespace);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   DiffusionIntegrator * aa = new DiffusionIntegrator(cf);
   // int order1 = fespace->GetElementOrder(0);
   IntegrationRule *irs = TensorIntegrationRule(*fespace,order); 
   aa->SetIntegrationRule(*irs);
   a.AddDomainIntegrator(aa);

   ParLinearForm b(fespace);
   FunctionCoefficient rhscf(lshape_rhs);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhscf));
   // ------------------------------------------------- 

   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sout;
   if (visualization)
   {
      sout.open(vishost, visport);
      if (!sout)
      {
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
         visualization = false;
      }

      sout.precision(8);

      sout << "parallel " << num_procs << " " << myid << "\n";
      sout << "solution\n" << *pmesh << x << flush;
   }

   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(pmesh, &flux_fec, dim);
   FiniteElementCollection *smooth_flux_fec = NULL;
   ParFiniteElementSpace *smooth_flux_fes = NULL;
   smooth_flux_fec = new RT_FECollection(order-1, dim);
   smooth_flux_fes = new ParFiniteElementSpace(pmesh, smooth_flux_fec, 1);
   L2ZienkiewiczZhuEstimator estimator(*aa, x, flux_fes, *smooth_flux_fes);
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);
   refiner.SetNCLimit(1);
   StopWatch chrono;
   Array<double> ts0, ts1, tsol;
   int ref_amr = 20;   
   Array<int> iter;
   Array<int> dofs;

   ostringstream file_name;
   file_name << "lshape-amr_" << order << ".csv";
   ofstream conv(file_name.str().c_str());
   conv << "DOFs " << ", " << "it-Cheb-Jac" << ", " << "it-ChebElemSmoother" << endl;

   for (int it = 0; it < ref_amr ; it++)
   {
      HYPRE_BigInt global_dofs = fespace->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      Array<int> ess_tdof_list;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      a.Assemble();
      b.Assemble();

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);


      chrono.Clear();
      chrono.Start();
      chrono.Stop();
      ts0.Append(chrono.RealTime());
      chrono.Clear();
      chrono.Start();
      Solver * prec1 = nullptr;
      Solver * prec2 = nullptr;
      Solver * S = nullptr;
      // if (solver)
      // {
         S = new ElementSmoother(fespace,ess_bdr, &cf); 
         prec1 = new OperatorChebyshevSmoother(*A, *S, 4, MPI_COMM_WORLD,7);
      // }
      // else
      // {
         Vector diag(fespace->GetTrueVSize());
         a.AssembleDiagonal(diag);
         prec2 = new OperatorChebyshevSmoother(*A, diag,ess_tdof_list, 4, MPI_COMM_WORLD,10);
      // }

      chrono.Stop();
      ts1.Append(chrono.RealTime());



      int print_level = 3;
      int max_iter = 2000;
      double rtol = 1e-8;

      CGSolver pcg(MPI_COMM_WORLD);
      pcg.SetPrintLevel(print_level);
      pcg.SetMaxIter(max_iter);
      pcg.SetRelTol(rtol);
      pcg.SetOperator(*A);
      pcg.SetPreconditioner(*prec1);


      // chrono.Clear();
      // chrono.Start();
      Vector Y = X;
      pcg.Mult(B,Y);
      int iter1 = pcg.GetNumIterations();


      pcg.SetPreconditioner(*prec2);
      pcg.Mult(B,X);
      int iter2 = pcg.GetNumIterations();

      // chrono.Stop();
      // tsol.Append(chrono.RealTime());
      // iter.Append(pcg.GetNumIterations());
      // dofs.Append(global_dofs);

      delete S; 
      delete prec1;
      delete prec2;

      conv << global_dofs << ", " << iter1 << ", " << iter2 << endl;


      a.RecoverFEMSolution(X,b,x);

      if (visualization)
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "solution\n" << *pmesh << x << flush;
      }
      refiner.Apply(*pmesh);
      if (refiner.Stop())
      {
         if (myid == 0)
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }
      fespace->Update();
      x.Update();
      x.ProjectCoefficient(ex_coeff);

      a.Update();
      b.Update();
   }



   if (myid==0)
   {
      cout << "ts0 total  = " << ts0.Sum() << endl;
      cout << "ts1 total  = " << ts1.Sum() << endl;
      cout << "tsol total = " << tsol.Sum() << endl;
   }

   if (myid == 0)
   {
      cout << "num iterations = "; iter.Print(cout, iter.Size());
      cout << "dofs = "; dofs.Print(cout, dofs.Size());
   }

   delete smooth_flux_fes;
   delete smooth_flux_fec;
   delete pmesh;

   // 13. Free the used memory.
   MPI_Finalize();
   return 0;
}



