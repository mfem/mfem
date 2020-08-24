//
// Compile with: make maxwellp
//
//               mpirun maxwell -o 2 -f 8.0 -ref 3 -prob 4 -m ../data/inline-quad.mesh
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ParDST/ParDST.hpp"
#include "common/PML.hpp"

using namespace std;
using namespace mfem;

void source_re(const Vector &x, Vector & f);
void source_im(const Vector &x, Vector & f);
double wavespeed(const Vector &x);

double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;
double length = 1.0;
Array2D<double> comp_bdr;
Array2D<double> domain_bdr;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   // number of serial refinements
   int ser_ref_levels = 1;
   // number of parallel refinements
   int par_ref_levels = 2;
   double freq = 5.0;
   bool herm_conv = true;
   bool visualization = 1;
   int nd=2;
   int nx=2;
   int ny=2;
   int nz=2;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&nd, "-nd", "--dim","Problem space dimension");
   args.AddOption(&nx, "-nx", "--nx","Number of subdomains in x direction");
   args.AddOption(&ny, "-ny", "--ny","Number of subdomains in y direction");
   args.AddOption(&nz, "-nz", "--nz","Number of subdomains in z direction");               
   args.AddOption(&ser_ref_levels, "-sr", "--ser_ref_levels",
                  "Number of Serial Refinements.");
   args.AddOption(&par_ref_levels, "-pr", "--par_ref_levels",
                  "Number of Parallel Refinements."); 
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

   // check if the inputs are correct
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

   // Angular frequency
   omega = 2.0 * M_PI * freq;

   Mesh *mesh;


   int nel = 1;

   if (nd == 2)
   {
      mesh = new Mesh(nel, nel, Element::QUADRILATERAL, true, length, length, false);
   }
   else
   {
      mesh = new Mesh(nel, nel, nel, Element::HEXAHEDRON, true, length, length, length,false);
   }

   dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define a parallel mesh by a partitioning of the serial mesh.
   // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   // int nprocs = sqrt(num_procs);
   // MFEM_VERIFY(nprocs*nprocs == num_procs, "Check MPI partitioning");
   // int nxyz[3] = {num_procs,1,1};
   // int nxyz[3] = {nprocs,nprocs,1};
   // int nxyz[3] = {1,num_procs,1};
   int nxyz[3] = {num_procs,1,1};
   int * part = mesh->CartesianPartitioning(nxyz);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD,*mesh,part);
   // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD,*mesh);
   delete [] part;

   
   delete mesh;

   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   //    char vishost[] = "localhost";
   // int visport = 19916;
   // socketstream mesh_sock1(vishost, visport);
   // mesh_sock1.precision(8);
   // mesh_sock1 << "parallel " << num_procs << " " << myid << "\n"
   //            << "mesh\n"
   //            << *pmesh << "window_title 'Global mesh'" << flush;

   double hl = GetUniformMeshElementSize(pmesh);
   int nrlayers = 2;

   Array2D<double> lengths(dim,2);
   lengths = hl*nrlayers;
   // lengths[0][1] = 0.0;
   // lengths[1][1] = 0.0;
   // lengths[1][0] = 0.0;
   // lengths[0][0] = 0.0;
   // CartesianPML pml(mesh,lengths);
   CartesianPML pml(pmesh,lengths);
   pml.SetOmega(omega);
   comp_bdr.SetSize(dim,2);
   comp_bdr = pml.GetCompDomainBdr(); 


   // 6. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // ConstantCoefficient one(0.0);
   // Vector vec(dim); vec = 0.0;
   // VectorConstantCoefficient vecc(vec);
   // ParGridFunction tone(fespace); tone = 0.0;
   // tone.ProjectCoefficient(vecc);
   // tone.Print();
   // if (myid == 0)
   // {
   //    for (int i = 0; i<fespace->GetVSize(); i++)
   //    {
   //       if (fespace->GetDofSign(i) < 0)
   //          cout << fespace->GetGlobalTDofNumber(i) << ", " << fespace->GetDofSign(i) << endl;
   //    }   
   //    cout << "myid = " <<  myid <<  " no neg dofs" << endl;
   // }
   
   // MPI_Finalize();
   // return 0;
   

   // 7. Determine the list of true essential boundary dofs. In this example,
   //    the boundary conditions are defined based on the specific mesh and the
   //    problem type.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   VectorFunctionCoefficient f_re(dim, source_re);
   VectorFunctionCoefficient f_im(dim, source_re);
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_re),
                         new VectorFEDomainLFIntegrator(f_im));
   b.Vector::operator=(0.0);
   b.Assemble();

   // 10. Define the solution vector x as a complex finite element grid function
   //     corresponding to fespace.
   ParComplexGridFunction x(fespace);
   x = 0.0;
   // 11. Set up the sesquilinear form a(.,.)
   //
   //       1/mu (1/det(J) J^T J Curl E, Curl F)
   //        - omega^2 * epsilon (det(J) * (J^T J)^-1 * E, F)
   //
   FunctionCoefficient ws(wavespeed);
   ConstantCoefficient omeg(-pow(omega, 2));
   int cdim = (dim == 2) ? 1 : dim;
   PmlMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, &pml);
   PmlMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, &pml);

   PmlMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,&pml);
   PmlMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,&pml);
   ScalarMatrixProductCoefficient c2_Re0(omeg,pml_c2_Re);
   ScalarMatrixProductCoefficient c2_Im0(omeg,pml_c2_Im);
   ScalarMatrixProductCoefficient c2_Re(ws,c2_Re0);
   ScalarMatrixProductCoefficient c2_Im(ws,c2_Im0);

   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(pml_c1_Re),
                         new CurlCurlIntegrator(pml_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(c2_Re),
                         new VectorFEMassIntegrator(c2_Im));

   a.Assemble(0);

   OperatorHandle Ah;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);

   ComplexSparseMatrix * Ac = Ah.As<ComplexSparseMatrix>();
   StopWatch chrono;


   chrono.Clear();
   chrono.Start();
   ParDST S(&a,lengths, omega, &ws, nrlayers, nx, ny, nz);
   chrono.Stop();
   double t1 = chrono.RealTime();

   chrono.Clear();
   chrono.Start();
   X = 0.0;
	GMRESSolver gmres(MPI_COMM_WORLD);
	// gmres.iterative_mode = true;
   gmres.SetPreconditioner(S);
	gmres.SetOperator(*Ac);
	gmres.SetRelTol(1e-8);
	gmres.SetMaxIter(10);
	gmres.SetPrintLevel(1);
	gmres.Mult(B, X);
   
   chrono.Stop();
   double t2 = chrono.RealTime();

   MPI_Barrier(MPI_COMM_WORLD);


   cout << " myid: " << myid 
         << ", setup time: " << t1
         << ", solution time: " << t2 << endl; 

   a.RecoverFEMSolution(X, b, x);





   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      string keys;
      if (dim ==2 )
      {
         keys = "keys mrRljc\n";
      }
      else
      {
         keys = "keys mc\n";
      }
      // socketstream mesh_sock(vishost, visport);
      // mesh_sock.precision(8);
      // mesh_sock << "parallel " << num_procs << " " << myid << "\n"
      //             << "mesh\n" << *pmesh  << flush;
      socketstream sol_sock_re(vishost, visport);
      sol_sock_re.precision(8);
      sol_sock_re << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << *pmesh << x.real() << keys 
                  << "window_title 'E: Real Part' " << flush;                     

      socketstream sol_sock_im(vishost, visport);
      sol_sock_im.precision(8);
      sol_sock_im << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << *pmesh << x.imag() << keys 
                  << "window_title 'E: Imag Part' " << flush;                     
   }

   // 18. Free the used memory.
   delete fespace;
   delete fec;
   delete pmesh;
   MPI_Finalize();
   return 0;
}

void source_re(const Vector &x, Vector &f)
{
   f = 0.0;
   double x0 = length/2.0;
   double x1 = length/2.0;
   double x2 = length/2.0;
   x0 = 0.45;
   x1 = 0.35;
   x2 = 0.25;
   double alpha,beta;
   double n = 4.0*omega/M_PI;
   beta = pow(x0-x(0),2) + pow(x1-x(1),2);
   if (dim == 3) { beta += pow(x2-x(2),2); }
   double coeff = 16.0*omega*omega/M_PI/M_PI/M_PI;
   alpha = -pow(n,2) * beta;
   f[0] = coeff*exp(alpha);
   // f[1] = coeff*exp(alpha);



   x0 = 0.8;
   x1 = 0.8;
   beta = pow(x0-x(0),2) + pow(x1-x(1),2);
   if (dim == 3) { beta += pow(x2-x(2),2); }
   alpha = -pow(n,2) * beta;
   // f[0] += coeff*exp(alpha);


   bool in_pml = false;
   for (int i = 0; i<dim; i++)
   {
      if (x(i)<=comp_bdr(i,0) || x(i)>=comp_bdr(i,1))
      {
         in_pml = true;
         break;
      }
   }
   if (in_pml) f = 0.0;
}

void source_im(const Vector &x, Vector &f)
{
   f = 0.0;
}

double wavespeed(const Vector &x)
{
   double ws;
   ws = 1.0;
   return ws;
}