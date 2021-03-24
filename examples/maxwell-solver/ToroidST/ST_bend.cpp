

// sample runs: ./ST_bend -ref 2 -o 2 -f 0.6
//              ./ST_bend -ref 3 -o 2 -f 1.2  (6 iterations)
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ToroidST.hpp"

using namespace std;
using namespace mfem;

void maxwell_solution(const Vector &x, vector<complex<double>> &E);
void maxwell_curl(const Vector &x, vector<complex<double>> &curlE);

void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);


Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;
double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_SELF, &num_procs);
   MPI_Comm_rank(MPI_COMM_SELF, &myid);
   // 1. Parse command-line options.
   const char *mesh_file = "meshes/toroid3_4_2.mesh";

   int order = 1;
   int ref_levels = 1;
   double freq = 0.6;
   bool herm_conv = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-ref", "--refinements",
                  "Number of refinements");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency (in Hz).");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // 2. Setup the mesh
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);


   Mesh * mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   mesh->RemoveInternalBoundaries();

   cout << "Initial number of elements = " << mesh->GetNE() << endl;


   for (int iter = 0; iter<ref_levels; iter++)
   {
      mesh->UniformRefinement();
   }

   // Angular frequency
   omega = 2.0 * M_PI * freq;

   ToroidPML tpml(mesh);
   Vector zlim, rlim, alim;
   tpml.GetDomainBdrs(zlim,rlim,alim);
   Vector zpml_thickness(2); zpml_thickness = 0.0;
   Vector rpml_thickness(2); rpml_thickness = 0.0;
   Vector apml_thickness(2); apml_thickness = 0.0; 
   bool zstretch = false;
   bool astretch = false;
   bool rstretch = false;
   apml_thickness[1] = 20.0;
   astretch = true;
   tpml.SetPmlAxes(zstretch,rstretch,astretch);
   tpml.SetPmlWidth(zpml_thickness,rpml_thickness,apml_thickness);
   tpml.SetOmega(omega); 



   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;


   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   int size = fespace->GetTrueVSize();
   cout << "Number of finite element unknowns: " << size << endl;
   tpml.SetAttributes(mesh); 


   ComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
   }
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ComplexLinearForm b(fespace, conv);
   b.Vector::operator=(0.0);
   b.Assemble();

   x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);

   Array<int> attr;
   Array<int> attrPML;
   if (mesh->attributes.Size())
   {
      attr.SetSize(mesh->attributes.Max());
      attrPML.SetSize(mesh->attributes.Max());
      attr = 0;   attr[0] = 1;
      attrPML = 0;
      if (mesh->attributes.Max() > 1)
      {
         attrPML[1] = 1;
      }
   }

   ConstantCoefficient muinv(1.0/mu);
   ConstantCoefficient omeg(-pow(omega, 2) * epsilon);
   RestrictedCoefficient restr_muinv(muinv,attr);
   RestrictedCoefficient restr_omeg(omeg,attr);

   // Integrators inside the computational domain (excluding the PML region)
   SesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv),NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg),NULL);

   int cdim = (dim == 2) ? 1 : dim;

   PMLMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, &tpml);
   PMLMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, &tpml);
   ScalarMatrixProductCoefficient c1_Re(muinv,pml_c1_Re);
   ScalarMatrixProductCoefficient c1_Im(muinv,pml_c1_Im);
   MatrixRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
   MatrixRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

   PMLMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,&tpml);
   PMLMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,&tpml);
   ScalarMatrixProductCoefficient c2_Re(omeg,pml_c2_Re);
   ScalarMatrixProductCoefficient c2_Im(omeg,pml_c2_Im);
   MatrixRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
   MatrixRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

   // Integrators inside the PML region
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                        new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                        new VectorFEMassIntegrator(restr_c2_Im));

   a.Assemble(0);

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   Vector Y(X);




   // SparseMatrix * SpMat = (*A.As<ComplexSparseMatrix>()).GetSystemMatrix();
   // // SpMat->Threshold(0.0);
   // // SpMat->PrintMatlab(cout);
   // // cin.get();
   // HYPRE_Int global_size = SpMat->Height();
   // HYPRE_Int row_starts[2]; row_starts[0] = 0; row_starts[1] = global_size;
   // HypreParMatrix * HypreMat = new HypreParMatrix(MPI_COMM_SELF,global_size,row_starts,SpMat);
   // {
   //    MUMPSSolver mumps;
   //    mumps.SetOperator(*HypreMat);
   //    mumps.Mult(B,X);
   // }
   // cout << "X norm = " << X.Norml2() << endl;


   // double overlap = 7; // in degrees;
   // double ovlerlap = 0.5; // in degrees;
   // int nrmeshes = 9;

   // Array<Array<int> *> ElemMaps, DofMaps0, DofMaps1, OvlpMaps0, OvlpMaps1;
   // Array<FiniteElementSpace *> fespaces;
   // PartitionFE(fespace,nrmeshes,overlap,fespaces, 
   //             ElemMaps,
   //             DofMaps0, DofMaps1,
   //             OvlpMaps0, OvlpMaps1);

     // Test local to global dof Maps
   // for (int i = 0; i<nrmeshes; i++)
   // {
   //    DofMapTests(*fespaces[i],*fespace,*DofMaps0[i], *DofMaps1[i]);
   //    // DofMapTests(*fespace,*fespaces[i], *DofMaps1[i], *DofMaps0[i]);
   //    cin.get();
   // }

   // for (int i = 0; i<nrmeshes-1; i++)
   // {
   //    // DofMapTests(*fespaces[i],*fespaces[i+1],*OvlpMaps0[i], *OvlpMaps1[i]);
   //    // DofMapTests(*fespaces[i+1],*fespaces[i],*OvlpMaps1[i], *OvlpMaps0[i]);
   //    Array<int> rdofs;
   //    RestrictDofs(*fespaces[i],0,overlap,rdofs);
   //    DofMapOvlpTest(*fespaces[i],rdofs);
   //    cin.get();
   // }
   // a.RecoverFEMSolution(X, b, x);


   int nrsubdomains = 5;
   ToroidST * STSolver = new ToroidST(&a,apml_thickness,omega,nrsubdomains);
   STSolver->Mult(B,Y);

   GMRESSolver gmres;
	// gmres.iterative_mode = true;
   gmres.SetPreconditioner(*STSolver);
	gmres.SetOperator(*A);
	gmres.SetRelTol(1e-8);
	gmres.SetMaxIter(100);
	gmres.SetPrintLevel(1);
	gmres.Mult(B, Y);
   delete STSolver;

   cout << "Y norm = " << Y.Norml2() << endl;


   // cin.get();

   a.RecoverFEMSolution(Y, b, x);
   // a.RecoverFEMSolution(X, b, x);





   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Define visualization keys for GLVis (see GLVis documentation)
      string keys;
      keys = (dim == 3) ? "keys macF\n" : keys = "keys amrRljcUUuu\n";

      char vishost[] = "localhost";
      int visport = 19916;

      socketstream sol_sock_re(vishost, visport);
      sol_sock_re.precision(8);
      sol_sock_re << "solution\n"
                  << *mesh << x.real() << keys
                  << "window_title 'Solution real part'" << flush;

      socketstream sol_sock_im(vishost, visport);
      sol_sock_im.precision(8);
      sol_sock_im << "solution\n"
                  << *mesh << x.imag() << keys
                  << "window_title 'Solution imag part'" << flush;

      GridFunction x_t(fespace);
      x_t = x.real();
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *mesh << x_t << keys << "autoscale off\n"
               << "window_title 'Harmonic Solution (t = 0.0 T)'"
               << "pause\n" << flush;
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
      int num_frames = 16;
      int i = 0;
      while (sol_sock)
      {
         double t = (double)(i % num_frames) / num_frames;
         ostringstream oss;
         oss << "Harmonic Solution (t = " << t << " T)";

         add(cos(2.0 * M_PI * t), x.real(),
             sin(2.0 * M_PI * t), x.imag(), x_t);
         sol_sock << "solution\n"
                  << *mesh << x_t
                  << "window_title '" << oss.str() << "'" << flush;
         i++;
      }
   }

   // 17. Free the used memory.
   // delete pml;
   delete fespace;
   delete fec;
   delete mesh;
   MPI_Finalize();
   return 0;
}


void E_bdr_data_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   if (abs(x(1))<1e-12 && x(0)>0) 
   {
      vector<complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].real();
      }
   }
}

// Define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   if (abs(x(1))<1e-12 && x(0)>0) 
   {
      vector<complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].imag();
      }
   }
}

void E_exact_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   vector<complex<double>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].real();
   }
}

void E_exact_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   vector<complex<double>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].imag();
   }
}

void maxwell_solution(const Vector &x, vector<complex<double>> &E)
{
   complex<double> zi = complex<double>(0., 1.);

   double k = omega * sqrt(epsilon * mu);
   // T_10 mode
   double k10 = sqrt(k * k - M_PI * M_PI);
   E[2] = -zi * k / M_PI * sin(M_PI*(x(0)))*exp(zi * k10 * x(1)); 
}




   // double ovlerlap = 7.5; // in degrees;
   // // double ovlerlap = 0.5; // in degrees;
   // int nrmeshes = 9;

   // Array<Array<int> *> ElemMaps, DofMaps0, DofMaps1, OvlpMaps0, OvlpMaps1;
   // Array<FiniteElementSpace *> fespaces;
   // PartitionFE(fespace,nrmeshes,ovlerlap,fespaces, 
   //             ElemMaps,
   //             DofMaps0, DofMaps1,
   //             OvlpMaps0, OvlpMaps1);


   // Test local to global dof Maps
   // for (int i = 0; i<nrmeshes; i++)
   // {
   //    DofMapTests(*fespaces[i],*fespace,*DofMaps0[i], *DofMaps1[i]);
   //    // DofMapTests(*fespace,*fespaces[i], *DofMaps1[i], *DofMaps0[i]);
   //    cin.get();
   // }

   // for (int i = 0; i<nrmeshes-1; i++)
   // {
   //    // DofMapTests(*fespaces[i],*fespaces[i+1],*OvlpMaps0[i], *OvlpMaps1[i]);
   //    DofMapTests(*fespaces[i+1],*fespaces[i],*OvlpMaps1[i], *OvlpMaps0[i]);
   //    cin.get();
   // }
   

   // if (visualization)
   // {
   //    // GLVis server to visualize to
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;

   //    socketstream mesh0_sock(vishost, visport);
   //    mesh0_sock.precision(8);
   //    mesh0_sock << "mesh\n" << *mesh << flush;

   //    socketstream mesh1_sock(vishost, visport);
   //    mesh1_sock.precision(8);
   //    mesh1_sock << "mesh\n" << *mesh1 << flush;

   //    socketstream mesh2_sock(vishost, visport);
   //    mesh2_sock.precision(8);
   //    mesh2_sock << "mesh\n" << *mesh2 << flush;
   // }


   // mesh = mesh1;
   // return 0;