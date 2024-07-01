#include "mfem.hpp"

#include "CurlCurlIntegrator.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double B_norm(const Vector &);
void B_exact(const Vector &, Vector &);
double alpha = 1.0;
double q0 = 1.2, q2 = 2.8, a_i=2.7832, R0=6.2, Z0=5.1944, B0=1.0;
int dim;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "extruder_tokamak.mesh";
   int order = 1;
   int par_ref_levels = 1;
   int icase = 2;
   int maxiter = 500;
   const char *device_config = "cpu";
   bool visualization = false;
   bool paraview = false;
   bool static_cond = false;
   bool hybridization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element order (polynomial degree).");
   args.AddOption(&icase, "-i", "--icase", "icase.");
   args.AddOption(&maxiter, "-iter", "--iter", "max iteration.");
   args.AddOption(&alpha, "-alpha", "--alpha", "alpha.");
   args.AddOption(&paraview, "-para", "--para", "-no-para", "--no-para",
                  "Enable or disable Paraview visualization.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&par_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
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

   Device device(device_config);
   if (myid == 0) { device.Print(); }

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();
   if (dim!=3 || sdim!=3){
      cout << "wrong dimensions in mesh!"<<endl;
      MPI_Finalize();
      return 1;
   }

   {
      int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
      if (myid == 0) cout << "serial refine levels = "<<ref_levels<<endl;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   fec = new RT_FECollection(order, dim);
   fespace = new ParFiniteElementSpace(pmesh, fec);

   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
      cout << "dim = " << dim << " sdim = "<< sdim << endl;
      cout << "Local number of elements: "<<pmesh->GetNE()<<endl;
   }

   // Note that ess_tdof_list is for scalar fespace
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParGridFunction x(fespace), Bvec(fespace);
   FunctionCoefficient *B2Coeff;
   B2Coeff = new FunctionCoefficient(B_norm);
   VectorFunctionCoefficient *VecCoeff;
   VecCoeff = new VectorFunctionCoefficient(sdim, B_exact);

   if(icase>1){
     //vector test
     Vector onevec(3); onevec=1.0;
     VectorConstantCoefficient one(onevec);
     x = 0.0;

     FiniteElementCollection *hfec = NULL;
     ParFiniteElementSpace *hfes = NULL;

     ConstantCoefficient sigma(alpha);
     Bvec.ProjectCoefficient(*VecCoeff);
     ParBilinearForm *a = new ParBilinearForm(fespace);
     a->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
     a->AddDomainIntegrator(new DivDivIntegrator(*B2Coeff));
     if (static_cond)
     {
        a->EnableStaticCondensation();
     }
     else if (hybridization)
     {
        hfec = new DG_Interface_FECollection(order, dim);
        hfes = new ParFiniteElementSpace(pmesh, hfec);
        a->EnableHybridization(hfes, new NormalTraceJumpIntegrator(),
                               ess_tdof_list);
     }
     a->Assemble();

     ParLinearForm b(fespace);
     b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(one));
     b.Assemble();

     HypreParMatrix A;
     Vector B, X;
     a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);

     if (myid == 0)
     {
        cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
     }

     CGSolver *pcg = new CGSolver(MPI_COMM_WORLD);
     pcg->SetOperator(A);
     Solver *prec;
     if (false){ 
         HypreBoomerAMG *prec2 = new HypreBoomerAMG(A);
         prec2->SetRelaxType(18);
         prec = prec2;
     } 
     else if (hybridization) { prec = new HypreBoomerAMG(A); }
     else{
        ParFiniteElementSpace *prec_fespace =
         (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
        prec = new HypreADS(A, prec_fespace);
     }
     //HypreSolver *prec = new HypreILU();
     pcg->SetRelTol(1e-12);
     pcg->SetMaxIter(maxiter);
     pcg->SetPrintLevel(1);
     pcg->SetPreconditioner(*prec);
     pcg->Mult(B, X);
     a->RecoverFEMSolution(X, b, x);
     delete prec;
     delete a;
     delete pcg;
     delete B2Coeff;
     delete VecCoeff;
     delete hfec;
     delete hfes;
   }

   if (true)
   {
      ostringstream mesh_name, sol_name, err_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;
      err_name << "err." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);

      if (paraview)
      {
         ParaViewDataCollection paraview_dc("rt", pmesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(order);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetCycle(0);
         paraview_dc.RegisterField("vec",&x);
         paraview_dc.RegisterField("B",&Bvec);
         paraview_dc.Save();
      }
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();
   return 0;
}


void B_exact(const Vector &x, Vector &B)
{
   const double R = sqrt(x(0)*x(0)+x(1)*x(1)), Z = x(2);
   const double q = q0 + q2*((R-R0)*(R-R0)+Z*Z)/a_i/a_i;
   double B_R, B_Z, B_phi, cosphi, sinphi;

   B_R = -Z/q/R*B0;
   B_Z = (R-R0)/q/R*B0;
   B_phi = R0/R*B0;

   cosphi = x(0)/R;
   sinphi = x(1)/R;

   B(0) = B_R*cosphi-B_phi*sinphi;
   B(1) = B_R*sinphi+B_phi*cosphi;
   B(2) = B_Z;
}

double B_norm(const Vector &x)
{
   Vector B(3);
   B_exact(x, B);
   return B(0)*B(0)+B(1)*B(1)+B(2)*B(2);
}

