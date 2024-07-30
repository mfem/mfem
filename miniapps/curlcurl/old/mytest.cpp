#include "mfem.hpp"
#include "CurlCurlIntegrator.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void B_exact2(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
void u_exact(const Vector &, Vector &);
double B2(const Vector &x);
void BBt(const Vector &, DenseMatrix &);
double freq = 1.0, kappa, alpha = 1.0;
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
   int icase = 1;
   bool hcurl = false;
   bool Evec = false;
   const char *device_config = "cpu";
   bool visualization = false;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact solution.");
   args.AddOption(&hcurl, "-hcurl", "--hcurl", "-no-hcurl", "--no-hcurl", "Use Hcurl or H1.");
   args.AddOption(&Evec, "-Evec", "--Evec", "-no-Evec", "--no-Evec", "Test Evec.");
   args.AddOption(&icase, "-i", "--icase", "icase.");
   args.AddOption(&alpha, "-alpha", "--alpha", "alpha.");
   args.AddOption(&paraview, "-para", "--para", "-no-para", "--no-para",
                  "Enable or disable Paraview visualization.");
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
   int ref_levels;
   {
      ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
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
      cout << "ser_ref_levels =" << ref_levels << endl;
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
      if (icase==1)
      {
        fespace_s->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      else if(icase==2)
      {
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
   }

   ParGridFunction x(fespace), Bvec(fespace);
   ParGridFunction x_error(fespace);
   x_error=0.0;
   VectorFunctionCoefficient *VecCoeff;
   VecCoeff = new VectorFunctionCoefficient(sdim, B_exact2);
   Bvec.ProjectCoefficient(*VecCoeff);
   BmatCoeff bmatcoeff(&Bvec);

   MatrixFunctionCoefficient coeffbbt(3, BBt);

   ParGridFunction divB(fespace_s);
   divB = 0.0;
   ParMixedBilinearForm *dform=NULL;
   ParBilinearForm *mpform=NULL;
   Vector onevec(3); onevec=1.0;
   VectorConstantCoefficient one(onevec);
   VectorFunctionCoefficient f_rhs(sdim, f_exact), u_coeff(sdim, u_exact);
   x.ProjectCoefficient(u_coeff);

   Coefficient *sigma = new ConstantCoefficient(alpha);
   ConstantCoefficient zero(0.0);
   FunctionCoefficient *Bsquare = new FunctionCoefficient(B2);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new SpecialVectorCurlCurlIntegrator(*VecCoeff, bmatcoeff));
   //a->AddDomainIntegrator(new ElasticityIntegrator(*Bsquare, zero));
   a->AddDomainIntegrator(new VectorMassIntegrator(*sigma));
   a->Assemble();

   // debug the matrxi
   //a->Finalize();
   //HypreParMatrix *amat=a->ParallelAssemble();
   //ofstream myf ("amat.m");
   //amat->PrintMatlab(myf);
   //exit(1);

   ParLinearForm b(fespace);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(f_rhs));
   b.Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   HyprePCG pcg(A);
   //HypreBoomerAMG *prec = new HypreBoomerAMG(A);
   HypreSolver *prec = new HypreILU();
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(500);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(*prec);
   pcg.Mult(B, X);
   a->RecoverFEMSolution(X, b, x);
   delete prec;
   delete sigma;
   delete a;

   x_error.ProjectCoefficient(u_coeff);
   x_error -= x;
   double error = x.ComputeL2Error(u_coeff);
   if (myid == 0)
   {
       cout << "\n|| u_h - u ||_{L^2} = " << error << '\n' << endl;
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

      ofstream err_ofs(err_name.str().c_str());
      err_ofs.precision(8);
      x_error.Save(sol_ofs);

      if (paraview)
      {
         ParaViewDataCollection paraview_dc("curlcurl", pmesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(order);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetCycle(0);
         paraview_dc.RegisterField("vec",&x);
         paraview_dc.RegisterField("error",&x_error);
         if (icase==1) {
             paraview_dc.RegisterField("div",&divB);
         }
         else{
             paraview_dc.RegisterField("B",&Bvec);
         }
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
      sol_sock << "solution\n" << *pmesh << divB << flush;
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


void u_exact(const Vector &x, Vector &u)
{
   u(0) = x(1);
   u(1) = x(2);
   u(2) = x(0);
}

//exact forcing for 
//f = alpha*u + B x curl(curl(u x B))
void f_exact(const Vector &x, Vector &f)
{
    double R2 = x(0)*x(0)+x(1)*x(1);
    double R6 = R2*R2*R2;
    f(0) = alpha*x(1) + 4*x(0)*x(0)*x(1)/R6;
    f(1) = alpha*x(2) + 4*x(0)*x(1)*x(1)/R6;
    f(2) = alpha*x(0) + (x(0)+x(1))/R2/R2;
}


void B_exact2(const Vector &x, Vector &B)
{
   const double R = sqrt(x(0)*x(0)+x(1)*x(1));
   double B_R, B_Z, B_phi, cosphi, sinphi;
   B_R = 0.0;
   B_Z = 0.0;
   B_phi = 1.0/R;
   cosphi = x(0)/R;
   sinphi = x(1)/R;

   B(0) = B_R*cosphi-B_phi*sinphi;
   B(1) = B_R*sinphi+B_phi*cosphi;
   B(2) = B_Z;
}

double B2(const Vector &x)
{
   const double R = sqrt(x(0)*x(0)+x(1)*x(1));
   double B_R, B_Z, B_phi;
   B_R = 0.0;
   B_Z = 0.0;
   B_phi = 1.0/R;

   return -(B_R*B_R+B_Z*B_Z+B_phi*B_phi);
}

void BBt(const Vector &x, DenseMatrix &MatBBt)
{
   const double R = sqrt(x(0)*x(0)+x(1)*x(1));
   double B_R, B_Z, B_phi, cosphi, sinphi;
   Vector B(3);
   B_R = 0.0;
   B_Z = 0.0;
   B_phi = 1.0/R;
   cosphi = x(0)/R;
   sinphi = x(1)/R;

   B(0) = 0.0; //B_R*cosphi-B_phi*sinphi;
   B(1) = 0.0; //B_R*sinphi+B_phi*cosphi;
   B(2) = 1.0; //B_Z;
   
   for (int i=0; i<3; i++){
       for (int j=0; j<3; j++){
         MatBBt(i,j) = B(i)*B(j);
       }
   }
}
