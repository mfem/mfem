// Example run: ./FOSLS2D_maxwell -ref 4  -o 3 -sol 1 -k 3.0

//  ∇ × E - ω H = 0
// -ω E + ∇ × H = J

// --------------------------------------------------------------------------
// |   |            E             |             H              |     RHS    | 
// --------------------------------------------------------------------------
// | F | (∇ × E,∇ × F)+ ω^2 (E,F) | - ω (∇ × H,F) - ω (H,curF) | - ω (J,F)  |
// |   |                          |                            |            |
// | G |-ω (E,∇ × G)-ω (∇ × E,G)  | (∇ × H,∇ × G)+ ω^2(H,G)    | (J,∇ × G)  |

// for E in H1 (scalar) we have ∇ × E = [0 1;-1 0] ∇

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Define exact solution
double E_exact(const Vector &x);
void H_exact(const Vector &x, Vector &H);
double frhs(const Vector &x);
void fvrhs(const Vector &x, Vector &f);
void get_maxwell_solution(const Vector &x, double & E, Vector & curlE, double & curl2E);

int dim;
double omega;
int isol = 0;

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   // geometry file
   // finite element order of approximation
   int order = 1;
   // visualization flag
   bool visualization = 1;
   int ref = 1;
   // number of wavelengths
   double k = 0.6;
   // optional command line inputs
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref, "-ref", "--init-refinements",
                  "Number of initial mesh refinements");
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&isol, "-sol", "--solution",
                  "Exact Solution: 0) Polynomial, 1) Sinusoidal.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   omega = 2.0 * M_PI * k;

   Mesh mesh(1, 1, Element::QUADRILATERAL, true, 1.0, 1.0, false);

   dim = mesh.Dimension();
   if (dim == 3) {MFEM_ABORT("This is 2D Maxwell")};

   for (int i = 0; i < ref; i++)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection H1fec(order,dim);
   FiniteElementSpace H1fes(&mesh, &H1fec);

   ND_FECollection NDfec(order, dim);
   FiniteElementSpace NDfes(&mesh, &NDfec);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      // Essential BC on E. Nothing on H
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = H1fes.GetVSize();
   block_offsets[2] = NDfes.GetVSize();
   block_offsets.PartialSum();

   BlockVector x(block_offsets), b(block_offsets);
   x = 0.0;
   b = 0.0;

   FunctionCoefficient Eex(E_exact);
   VectorFunctionCoefficient Hex(dim, H_exact);
   GridFunction E_gf;
   GridFunction H_gf;
   E_gf.MakeRef(&H1fes, x.GetBlock(0));
   E_gf.ProjectBdrCoefficient(Eex,ess_bdr);
   H_gf.MakeRef(&NDfes, x.GetBlock(1));

   FunctionCoefficient f(frhs);
   ProductCoefficient f_E(-omega, f);
   VectorFunctionCoefficient f_H(1,fvrhs);
   LinearForm b_E;
   b_E.Update(&H1fes, b.GetBlock(0), 0);
   b_E.AddDomainIntegrator(new DomainLFIntegrator(f_E));
   b_E.Assemble();

   LinearForm b_H;
   b_H.Update(&NDfes, b.GetBlock(1), 0);
   b_H.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_H));
   b_H.Assemble();

   // 7. Bilinear form a(.,.) on the finite element space
   ConstantCoefficient one(1.0);
   ConstantCoefficient omeg2(pow(omega, 2));
   ConstantCoefficient negomega(-(omega));
   DenseMatrix mat(2);
   mat(0,0) = 0.; mat(0,1) = 1.;
   mat(1,0) = -1.; mat(1,1) = 0.;
   MatrixConstantCoefficient rot(mat);

   BilinearForm a_EE(&H1fes);
   a_EE.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_EE.AddDomainIntegrator(new MassIntegrator(omeg2));
   a_EE.Assemble();
   a_EE.EliminateEssentialBC(ess_bdr, x.GetBlock(0), b.GetBlock(0));
   a_EE.Finalize();
   SparseMatrix &A_EE = a_EE.SpMat();

   ScalarMatrixProductCoefficient c1(-omega, rot);
   MixedBilinearForm a_EH(&H1fes,&NDfes);
   // - omega (rot grad E, G) - (omega E, curl G)
   a_EH.AddDomainIntegrator(new MixedVectorGradientIntegrator(c1));
   a_EH.AddDomainIntegrator(new MixedScalarWeakCurlIntegrator(negomega));
   a_EH.Assemble();
   a_EH.EliminateTrialDofs(ess_bdr, x.GetBlock(0), b.GetBlock(1));
   a_EH.Finalize();
   SparseMatrix &A_EH = a_EH.SpMat();
   SparseMatrix * A_HE = Transpose(A_EH);

   BilinearForm a_HH(&NDfes);
   a_HH.AddDomainIntegrator(new CurlCurlIntegrator(one)); // one is the coeff
   a_HH.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2)); // one is the coeff
   a_HH.Assemble();
   a_HH.Finalize();
   SparseMatrix &A_HH = a_HH.SpMat();

   BlockMatrix LS_Maxwellop(block_offsets);
   LS_Maxwellop.SetBlock(0, 0, &A_EE);
   LS_Maxwellop.SetBlock(0, 1, A_HE);
   LS_Maxwellop.SetBlock(1, 0, &A_EH);
   LS_Maxwellop.SetBlock(1, 1, &A_HH);

   UMFPackSolver invE;
   invE.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   invE.SetOperator(LS_Maxwellop.GetBlock(0,0));

   UMFPackSolver invH;
   invH.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   invH.SetOperator(LS_Maxwellop.GetBlock(1,1));

   BlockDiagonalPreconditioner prec(block_offsets);
   prec.SetDiagonalBlock(0, &invE);
   prec.SetDiagonalBlock(1, &invH);

   int maxit(5000);
   double rtol(1.e-16);
   double atol(0.0);

   CGSolver pcg;
   pcg.SetAbsTol(atol);
   pcg.SetRelTol(rtol);
   pcg.SetMaxIter(maxit);
   pcg.SetOperator(LS_Maxwellop);
   pcg.SetPreconditioner(prec);
   pcg.SetPrintLevel(3);
   pcg.Mult(b, x);

   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double Error_E = E_gf.ComputeL2Error(Eex, irs);
   double Error_H = H_gf.ComputeL2Error(Hex, irs);
   
   cout << "|| E_h - E || = " << Error_E << "\n";
   cout << "|| H_h - H || = " << Error_H << "\n";
   cout << "Total error = " << sqrt(Error_H*Error_H+Error_E*Error_E) << "\n";

   GridFunction E_exgf(&H1fes);
   E_exgf.ProjectCoefficient(Eex);

   GridFunction H_exgf(&NDfes);
   H_exgf.ProjectCoefficient(Hex);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      socketstream ex_sock(vishost, visport);
      ex_sock.precision(8);
      socketstream sol_sockH(vishost, visport);
      sol_sockH.precision(8);
      socketstream ex_sockH(vishost, visport);
      ex_sockH.precision(8);
      sol_sock << "solution\n"
               << mesh << E_gf << "window_title 'Numerical E'" << "keys rRljc\n"
               << flush;
      ex_sock << "solution\n"
               << mesh << E_exgf << "window_title 'Exact E'" << "keys rRljc\n"
               << flush;
      sol_sockH << "solution\n"
               << mesh << H_gf << "window_title 'Numerical H'" << "keys rRljc\n"
               << flush;
      ex_sockH << "solution\n"
               << mesh << H_exgf << "window_title 'Exact H'" << "keys rRljc\n"
               << flush;        
   }
   delete A_HE;
   return 0;
}

double E_exact(const Vector &x)
{
   double E, curl2E;
   Vector curlE(2);
   get_maxwell_solution(x, E, curlE, curl2E);
   return E;  //Scalar
}

//define exact solution
void H_exact(const Vector &x, Vector &H)
{
   double E, curl2E;
   Vector curlE(2);
   get_maxwell_solution(x, E, curlE, curl2E);
   H[0] = curlE[0]/omega;
   H[1] = curlE[1]/omega;
}

double frhs(const Vector &x)
{
   double E, curl2E;
   Vector curlE(2);
   get_maxwell_solution(x, E, curlE, curl2E);

   // - omega E + curl H = f
   // - omega E + curl (curl E) / omega = f
   double f = - omega * E + curl2E / omega;
   return f;
}

void fvrhs(const Vector &x, Vector &f)
{
   double E, curl2E;
   Vector curlE(2);
   get_maxwell_solution(x, E, curlE, curl2E);
   f[0] = - omega * E + curl2E / omega;
}


void get_maxwell_solution(const Vector &X, double & E, Vector & curlE, double & curl2E)
{
   double x = X[0];
   double y = X[1];
   double Ex, Ey, Exx, Eyy;
   if (isol == 0) // polynomial
   {
      E = x * (1.0 - x) * y * (1.0 - y);
      Ex = (1.0 - 2.0 * x) * y * (1.0 - y);
      Ey = x * (1.0 - x) * (1.0 - 2.0 * y);

      Exx = -2.0 * y * (1.0 - y);
      Eyy = -2.0 * x * (1.0 - x);
   }
   else 
   {
      double s = omega * (y+x);
      E = cos(s);
      Ex = -omega * sin(s);  
      Ey = Ex;
      Exx = - omega * omega * E;  
      Eyy = Exx;  
   }
   curlE[0] = Ey;
   curlE[1] = -Ex;
   curl2E = -Exx - Eyy;
}