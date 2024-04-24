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

double GetBdrElementVolume(int i, Mesh & mesh)
{
   ElementTransformation *et = mesh.GetBdrElementTransformation(i);
   const IntegrationRule &ir = IntRules.Get(mesh.GetBdrElementGeometry(i),
                                            et->OrderJ());
   double volume = 0.0;
   for (int j = 0; j < ir.GetNPoints(); j++)
   {
      const IntegrationPoint &ip = ir.IntPoint(j);
      et->SetIntPoint(&ip);
      volume += ip.weight * et->Weight();
   }

   return volume;
}


double GetBdrArea(int bdrattr, Mesh&mesh)
{
   double area = 0.0;
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      if (mesh.GetBdrAttribute(i) == bdrattr)
      {
         area += GetBdrElementVolume(i,mesh);
      }
   }

   MPI_Allreduce(MPI_IN_PLACE,&area,1, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   return area;
}

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
   int nsteps = 1;
   bool outputfiles = false;
   bool doublepass = false;
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
   args.AddOption(&nsteps, "-nsteps", "--nsteps",
                  "Number of steps.");
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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&outputfiles, "-out", "--output", "-no-out",
                  "--no-ouput",
                  "Enable or disable ouput to files.");                  
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
   else if (testNo == 6 || testNo == 61 || testNo == 62)
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


   double area = GetBdrArea(3,*mesh);

   // ConstantCoefficient one(-area);
   ConstantCoefficient one(-1.0);

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
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      ess_bdr = 0;
      ess_bdr[2] = 1;
      // prob->SetNeumanPressureData(one,ess_bdr);
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
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      ess_bdr = 0;
      ess_bdr[2] = 1;
      // prob->SetNeumanPressureData(one,ess_bdr);
      prob->SetNeumanData(0,3,-2.0);
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
         ess_values[2] = 1.0/1.4/nsteps;
         // ess_values[0] = -2.0/nsteps;
      }
      essbdr_attr = (testNo == 40) ? 1 : 2;
      ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      essbdr_attr = (testNo == 40) ? 10 : 6;
      ess_values = 0.0; ess_bdr = 0; ess_bdr[essbdr_attr - 1] = 1;
      prob->SetDisplacementDirichletData(ess_values, ess_bdr);
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

   ParFiniteElementSpace * fes = prob->GetFESpace();
   ParGridFunction x_gf(fes); x_gf = 0.0;
   ParGridFunction xnew(fes); xnew = 0.0;
   ParaViewDataCollection * paraview_dc = nullptr;
   ParMesh pmesh_copy(*pmesh);
   ParFiniteElementSpace fes_copy(*fes,pmesh_copy);
   ParGridFunction xcopy_gf(&fes_copy); xcopy_gf = 0.0;
   
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
   // ParGridFunction coords(prob->GetFESpace()); 
   ParGridFunction ref_coords(prob->GetFESpace()); 
   ParGridFunction new_coords(prob->GetFESpace()); 
   pmesh->GetNodes(new_coords);
   pmesh->GetNodes(ref_coords);
   

   double p = 1;
   ConstantCoefficient f(p);
   for (int i = 0; i<nsteps; i++)
   {
      if (testNo == 6)
      {
         ess_bdr = 0;
         ess_bdr[2] = 1;
         f.constant = -p*(i+1)/nsteps;
         prob->SetNeumanPressureData(f,ess_bdr);
         // prob->SetNeumanData(0,3,-p*(i+1)/nsteps);
      }
      else if (testNo == 4 || testNo == 40 || testNo == 5 || testNo == 51)
      {
         ess_bdr = 0;
         essbdr_attr = (testNo == 40) ? 1 : 2;
         ess_bdr[essbdr_attr-1] = 1;
         ess_values = 0.0;
         ess_values[2] = 1.0/1.4*(i+1)/nsteps;
         prob->SetDisplacementDirichletData(ess_values, ess_bdr);
      }

      // add(ref_coords,x_gf,coords);
      // coords += xnew;
      ParContactProblem contact(prob, mortar_attr, nonmortar_attr, &new_coords, doublepass);
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
      // ParGridFunction x = prob->GetDisplacementGridFunction();
      // x.SetTrueVector();
      // Vector x0 = x.GetTrueVector();

      x_gf.SetTrueVector();
      Vector x0 = x_gf.GetTrueVector();
      int ndofs = x0.Size();
      Vector xf(ndofs); xf = 0.0;
      optimizer.Mult(x0, xf);

      Vector xf_copy(xf);
      xf_copy+=x0;
      double Einitial = contact.E(x0);
      // double Efinal = contact.E(xf);
      double Efinal = contact.E(xf_copy);
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
         if (outputfiles)
         {
            ostringstream file_name;
            file_name << "output/Testno-"<<testNo<<"-ref-"<<sref+pref << "-step-" << i; 
            OutputData(file_name, Einitial, Efinal, gndofs,numconstr, optimizer.GetNumIterations(), CGiterations);
         }
      }

      // Vector X_new(xf.GetData(),fes->GetTrueVSize());
      // xnew.SetFromTrueDofs(X_new);
      // x_gf = xnew;
      x_gf.SetFromTrueDofs(xf);
      // mfem::out << "x_gf norm = " << x_gf.Norml2() << endl;
      // cin.get();
      // pmesh->MoveNodes(xnew);
      // pmesh_copy.MoveNodes(xnew);
      // pmesh_copy.MoveNodes(xnew);
      add(ref_coords,x_gf,new_coords);
      // mfem::out << " ref_coords norm " << ref_coords.Norml2() << endl;
      // mfem::out << " x_gf norm " << x_gf.Norml2() << endl;
      // mfem::out << " new_coords norm " << new_coords.Norml2() << endl;
      // pmesh_copy.SetNodes(new_coords);
      pmesh_copy.SetNodes(new_coords);
      xcopy_gf = x_gf;
      // pmesh_copy.MoveNodes(x_gf);
      // pmesh_copy.SetNodes(x_gf);
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
      
         if (i == nsteps - 1)
         {
            pmesh->MoveNodes(x_gf);
            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream sol_sock1(vishost, visport);
            sol_sock1 << "parallel " << num_procs << " " << myid << "\n";
            sol_sock1.precision(8);
            sol_sock1 << "solution\n" << *pmesh << x_gf << flush;
         }
      }
      if (i == nsteps-1) break;

      prob->UpdateStep();
      if (testNo == 6 )
      {
         double area_new = GetBdrArea(3,*pmesh);
         if (myid == 0)
         {
            mfem::out << "New area = " << area_new << endl;
         }
      }
   }

   delete prob;
   delete pmesh;
   delete mesh;
   return 0;
}
