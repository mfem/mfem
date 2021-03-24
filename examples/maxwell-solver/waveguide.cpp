//
// Compile with: make maxwellp
//
//               mpirun ./maxwellp -o 3 -f 8.0 -sr 2 -pr 2 -m ../../data/inline-quad.mesh -nx 4 -ny 4
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ParDST/ParDST.hpp"
#include "common/PML.hpp"

using namespace std;
using namespace mfem;
  
void maxwell_solution(const Vector &x, vector<complex<double>> &Eval);
void ess_data_func_re(const Vector & x, Vector & E);
void ess_data_func_im(const Vector & x, Vector & E);

double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;
double length = 1.0;
Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
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
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&nd, "-nd", "--dim",
                  "Problem space dimension");                  
   args.AddOption(&nx, "-nx", "--nx","Number of subdomains in x direction");
   args.AddOption(&ny, "-ny", "--ny","Number of subdomains in y direction");
   args.AddOption(&nz, "-nz", "--nz","Number of subdomains in z direction");     
   args.AddOption(&ser_ref_levels, "-sr", "--ser_ref_levels",
                  "Number of Serial Refinements.");
   args.AddOption(&par_ref_levels, "-pr", "--par_ref_levels",
                  "Number of Parallel Refinements."); 
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency (in Hz).");
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
   int nelx = 8;
   double lengthx = 8*length;
   if (nd == 3)
   {
      mesh = new Mesh(nelx, nel, nel, Element::HEXAHEDRON, true, lengthx, length, length,false);
   }
   else
   {
      mesh = new Mesh(nelx, nel, Element::QUADRILATERAL, true, lengthx, length,false);
   }
   dim = mesh->Dimension();
   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ser_ref_levels; l++) { mesh->UniformRefinement(); }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD,*mesh);
   delete mesh;
   for (int l = 0; l < par_ref_levels; l++) {pmesh->UniformRefinement(); }


   double hl = GetUniformMeshElementSize(pmesh);
   int nrlayers = 4;
   Array2D<double> lengths(dim,2);
   lengths = 0.0;
   // lengths = hl*nrlayers;
   lengths(0, 1) = hl*nrlayers;
   CartesianPML pml(pmesh,lengths);
   pml.SetOmega(omega);
   comp_domain_bdr.SetSize(dim,2);
   comp_domain_bdr = pml.GetCompDomainBdr(); 

   // 6. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;

   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   ParComplexLinearForm b(fespace);
   b.Vector::operator=(0.0);
   b.Assemble();

   // 10. Define the solution vector x as a complex finite element grid function
   //     corresponding to fespace.
   ParComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_re(dim,ess_data_func_re);
   VectorFunctionCoefficient E_im(dim,ess_data_func_im);
   x.ProjectBdrCoefficientTangent(E_re, E_im, ess_bdr);
   // 11. Set up the sesquilinear form a(.,.)
   //
   //       1/mu (1/det(J) J^T J Curl E, Curl F)
   //        - omega^2 * epsilon (det(J) * (J^T J)^-1 * E, F)
   //
   ConstantCoefficient omeg(-pow(omega, 2));
   int cdim = (dim == 2) ? 1 : dim;
   PmlMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, &pml);
   PmlMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, &pml);

   PmlMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,&pml);
   PmlMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,&pml);
   ScalarMatrixProductCoefficient c2_Re(omeg,pml_c2_Re);
   ScalarMatrixProductCoefficient c2_Im(omeg,pml_c2_Im);

   ParSesquilinearForm a(fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(pml_c1_Re),
                         new CurlCurlIntegrator(pml_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(c2_Re),
                         new VectorFEMassIntegrator(c2_Im));
   a.Assemble(0);

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   ConstantCoefficient one(1.0);
   ParDST * S = new ParDST(&a,lengths, omega, &one, nrlayers, nx, ny, nz);
   X = 0.0;
	GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetPreconditioner(*S);
	gmres.SetOperator(*A);
	gmres.SetRelTol(1e-8);
	gmres.SetMaxIter(50);
	gmres.SetPrintLevel(1);
	gmres.Mult(B, X);
   delete S;

   // {
   //    ComplexMUMPSSolver mumps;
   //    mumps.SetOperator(*A.As<ComplexHypreParMatrix>());
   //    mumps.Mult(B,X);
   // }

   a.RecoverFEMSolution(X, b, x);


   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      string keys;
      // keys = "keys mc\n";
      keys = "keys macFFiYYYYYYYYYYYYYYYYYY\n";
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
      {
         ParGridFunction x_t(fespace);
         x_t = x.real();

         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << *pmesh << x_t << keys << "autoscale off\n"
                  << "window_title 'Harmonic Solution (t = 0.0 T)'"
                  << "pause\n" << flush;

         if (myid == 0)
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }

         int num_frames = 32;
         int i = 0;
         while (sol_sock)
         {
            double t = (double)(i % num_frames) / num_frames;
            ostringstream oss;
            oss << "Harmonic Solution (t = " << t << " T)";

            add(cos(2.0*M_PI*t), x.real(), sin(2.0*M_PI*t), x.imag(), x_t);
            sol_sock << "parallel " << num_procs << " " << myid << "\n";
            sol_sock << "solution\n" << *pmesh << x_t
                     << "window_title '" << oss.str() << "'" << flush;
            i++;
         }
      }
   }

   // 18. Free the used memory.
   delete fespace;
   delete fec;
   delete pmesh;
   MPI_Finalize();
   return 0;
}

void maxwell_solution(const Vector &x, vector<complex<double>> &E)
{
   complex<double> zi = complex<double>(0., 1.);
   if (dim == 3)
   {
      double k10 = sqrt(omega * omega - M_PI * M_PI);
      E[1] = -zi * omega / M_PI * sin(M_PI*x(2))*exp(zi * k10 * x(0));
   }
   else
   {
      E[1] = -zi * omega / M_PI * exp(zi * omega * x(0));
   }
   // E[1] = -zi * omega / M_PI * sin(M_PI*x(0))*exp(zi * k10 * x(2));
   E[0] = 0.0;
   if (dim == 3) E[2] = 0.0;
}


void ess_data_func_re(const Vector & x, Vector & E)
{
   E = 0.0;
   bool in_pml = false;

   for (int i = 0; i < dim; ++i)
   {
      // check if in PML
      if (x(i) - comp_domain_bdr(i, 0) < 0.0 ||
          x(i) - comp_domain_bdr(i, 1) > 0.0)
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      vector<complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].real();
      }
   }

}

void ess_data_func_im(const Vector & x, Vector & E)
{
   E = 0.0;
   bool in_pml = false;

   for (int i = 0; i < dim; ++i)
   {
      // check if in PML
      if (x(i) - comp_domain_bdr(i, 0) < 0.0 ||
          x(i) - comp_domain_bdr(i, 1) > 0.0)
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      vector<complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].imag();
      }
   }
}
