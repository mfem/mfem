#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void E_exact(const Vector &, Vector &);
void B_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
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
   bool hcurl = false;
   bool Evec = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact solution.");
   args.AddOption(&hcurl, "-hcurl", "--hcurl", "-no-hcurl", "--no-hcurl", "Use Hcurl or H1.");
   args.AddOption(&hcurl, "-Evec", "--Evec", "-no-Evec", "--no-Evec", "Test Evec.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&par_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");

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
   kappa = freq * M_PI;

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();
   if (dim!=3 || sdim!=3){
      cout << "wrong dimensions in mesh!"<<endl;
      MPI_Finalize();
      return 1;
   }

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec, *fec_s;
   ParFiniteElementSpace *fespace, *fespace_s;
   if (hcurl){
       // maybe it does not support high-order mesh
       fec = new ND_FECollection(order, dim);
       fespace = new ParFiniteElementSpace(pmesh, fec);
   }
   else{
       fec = new H1_FECollection(order, dim);
       fespace = new ParFiniteElementSpace(pmesh, fec, dim);
   }
   fec_s = new H1_FECollection(order);
   fespace_s = new ParFiniteElementSpace(pmesh, fec_s);

   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
      cout << "dim = " << dim << " sdim = "<< sdim << endl;
   }

   // Note that ess_tdof_list is for scalar fespace
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace_s->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParGridFunction x(fespace);
   VectorFunctionCoefficient *VecCoeff;
   if (false){
       VecCoeff = new VectorFunctionCoefficient(sdim, E_exact);
   }
   else{
       VecCoeff = new VectorFunctionCoefficient(sdim, B_exact);
   }
   x.ProjectCoefficient(*VecCoeff);

   ParGridFunction divB(fespace_s);
   divB = 0.0;
   ParMixedBilinearForm *dform=NULL;
   ParBilinearForm *mpform=NULL;
   if(true && !hcurl){
     dform = new ParMixedBilinearForm(fespace, fespace_s);
     dform->AddDomainIntegrator(new VectorDivergenceIntegrator);
     dform->Assemble();

     mpform = new ParBilinearForm(fespace_s);
     mpform->AddDomainIntegrator(new MassIntegrator);
     mpform->Assemble();

     ParLinearForm rhs(fespace_s);
     dform->Mult(x, rhs);

     OperatorPtr A;
     Vector B, X;
     mpform->FormLinearSystem(ess_tdof_list, divB, rhs, A, X, B);

     CGSolver M_solver(MPI_COMM_WORLD);
     M_solver.iterative_mode = false; 
     M_solver.SetRelTol(1e-7);
     M_solver.SetAbsTol(0.0);
     M_solver.SetMaxIter(2000);
     M_solver.SetPrintLevel(0);
     HypreSmoother *M_prec = new HypreSmoother;
     M_prec->SetType(HypreSmoother::Jacobi);
     M_solver.SetPreconditioner(*M_prec);
     M_solver.SetPrintLevel(1);
     M_solver.SetOperator(*A);

     M_solver.Mult(B, X);
     mpform->RecoverFEMSolution(X, rhs, divB);
   }

   /*
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   {
      if (myid == 0)
      {
         cout << "Size of linear system: "
              << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;
      }

      ParFiniteElementSpace *prec_fespace =
         (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
      HypreAMS ams(*A.As<HypreParMatrix>(), prec_fespace);
      HyprePCG pcg(*A.As<HypreParMatrix>());
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(ams);
      pcg.Mult(B, X);
   }
   a->RecoverFEMSolution(X, *b, x);
   */

   if (true)
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << divB << flush;
   }

   {
      ParaViewDataCollection paraview_dc("curlcurl", pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.RegisterField("vec",&x);
      paraview_dc.RegisterField("div",&divB);
      paraview_dc.Save();
   }

   delete fespace;
   delete fec;
   delete pmesh;
   delete fespace_s;
   delete fec_s;
   if (dform!=NULL)     delete dform;
   if (mpform!=NULL)    delete mpform;

   MPI_Finalize();

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

// transform matrix:
// [v_X] = [cos -sin][v_R  ]
// [v_Y] = [sin  cos][v_phi]
//
// Safty factor:
// q = q0+q2 r^2 with r^2 = (R-R0)^2/a^2 + z^2/Z0^2
//
void B_exact(const Vector &x, Vector &B)
{
   const double R = sqrt(x(0)*x(0)+x(1)*x(1)), Z = x(2);
   const double q = q0 + q2*((R-R0)*(R-R0)+Z*Z)/a_i/a_i;
   double B_R, B_Z, B_phi, cosphi, sinphi;

   B_R = -Z/q/R*B0;
   B_Z = (R-R0)/q/R*B0;
   //B_R = 0.0;
   //B_Z = 0.0;
   B_phi = R0/R*B0;

   cosphi = x(0)/R;
   sinphi = x(1)/R;

   B(0) = B_R*cosphi-B_phi*sinphi;
   B(1) = B_R*sinphi+B_phi*cosphi;
   B(2) = B_Z;
}
