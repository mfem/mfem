//                                MFEM Example multigrid-grid Cycle
//
// Compile with: make mg_maxwellp
//
// Sample runs:  mg_maxwellp -m ../data/one-hex.mesh

#include "mfem.hpp"
#include "Schwarz.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Define exact solution
void E_exact(const Vector &x, Vector &E);
void H_exact(const Vector &x, Vector &H);
void scaledf_exact_H(const Vector &x, Vector &f_H);
void f_exact_H(const Vector &x, Vector &f_H);
void get_maxwell_solution(const Vector &x, double E[], double curlE[], double curl2E[]);

int dim;
double omega;
int sol = 1;

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   // geometry file
   // const char *mesh_file = "../data/star.mesh";
   const char *mesh_file = "../../data/one-hex.mesh";
   // finite element order of approximation
   int order = 1;
   int sdim = 3;
   // static condensation flag
   bool static_cond = false;
   // visualization flag
   bool visualization = 1;
   int ref_levels = 1;
   int initref = 1;
   // number of wavelengths
   double k = 0.5;
   double theta = 0.5;
   double smth_maxit = 1;
   // optional command line inputs
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sdim, "-d", "--dimension", "Dimension");
   args.AddOption(&ref_levels, "-ref", "--serial-refinements",
                  "Number of mesh refinements");
   args.AddOption(&initref, "-initref", "--init-refinements",
                  "Number of initial mesh refinements");
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&smth_maxit, "-sm", "--smoother-maxit",
                  "Number of smoothing steps.");
   args.AddOption(&theta, "-th", "--theta",
                  "Dumping parameter for the smoother.");
   args.AddOption(&sol, "-sol", "--solution",
                  "Exact Solution: 0) Polynomial, 1) Sinusoidal.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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

   // Angular frequency
   // omega = k;
   // omega = 2.0 * M_PI * k;
   omega = 2.0 * M_PI * k;

   // 3. Read the mesh from the given mesh file.
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);

   Mesh *mesh;
   // Define a simple square or cubic mesh
   mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0, false);
   dim = mesh->Dimension();
   
   for (int i = 0; i < initref; i++)
   {
      mesh->UniformRefinement();
   }

   Mesh *cmesh = new Mesh(*mesh);

   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define a finite element space on the mesh.
   FiniteElementCollection *NDfec = new ND_FECollection(order, dim);
   FiniteElementSpace *NDfespace = new FiniteElementSpace(mesh, NDfec);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      // Essential BC on E. Nothing on H
      NDfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = NDfespace->GetVSize();
   block_offsets[2] = NDfespace->GetVSize();
   block_offsets.PartialSum();

   //    _           _    _  _       _  _
   //   |             |  |    |     |    |
   //   |  A00   A01  |  | E  |     |F_E |
   //   |             |  |    |  =  |    |
   //   |  A10   A11  |  | H  |     |F_G |
   //   |_           _|  |_  _|     |_  _|
   //
   // A00 = (curl E, curl F) + \omega^2 (E,F)
   // A01 = - \omega *( (curl E, F) + (E,curl F)
   // A10 = - \omega *( (curl H, G) + (H,curl G)
   // A11 = (curl H, curl H) + \omega^2 (H,G)

   BlockVector x(block_offsets), b(block_offsets);
   x = 0.0;
   b = 0.0;


   VectorFunctionCoefficient * Eex = new VectorFunctionCoefficient(dim, E_exact);
   GridFunction *E_gf = new GridFunction;
   E_gf->MakeRef(NDfespace, x.GetBlock(0));
   E_gf->ProjectCoefficient(*Eex);

   VectorFunctionCoefficient * Hex;
   Hex = new VectorFunctionCoefficient(sdim, H_exact);
   
   GridFunction *H_gf = new GridFunction;
   H_gf->MakeRef(NDfespace, x.GetBlock(1));
   H_gf->ProjectCoefficient(*Hex);

   // 6. Set up the linear form
   VectorFunctionCoefficient sf_H(dim, scaledf_exact_H);
   VectorFunctionCoefficient f_H(dim, f_exact_H);


   LinearForm *b_E = new LinearForm;
   b_E->Update(NDfespace, b.GetBlock(0), 0);
   b_E->AddDomainIntegrator(new VectorFEDomainLFIntegrator(sf_H));
   b_E->Assemble();

   LinearForm *b_H = new LinearForm;
   b_H->Update(NDfespace, b.GetBlock(1), 0);
   b_H->AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_H));
   b_H->Assemble();

   // 7. Bilinear form a(.,.) on the finite element space
   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(pow(omega, 2));
   ConstantCoefficient neg(-abs(omega));
   ConstantCoefficient pos(abs(omega));
   //
   BilinearForm *a_EE = new BilinearForm(NDfespace);
   a_EE->AddDomainIntegrator(new CurlCurlIntegrator(one));
   a_EE->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a_EE->Assemble();
   a_EE->EliminateEssentialBC(ess_bdr, x.GetBlock(0), b.GetBlock(0));
   a_EE->Finalize();
   SparseMatrix &A_EE = a_EE->SpMat();

   MixedBilinearForm *a_EH = new MixedBilinearForm(NDfespace, NDfespace);
   a_EH->AddDomainIntegrator(new MixedVectorCurlIntegrator(neg));
   a_EH->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(neg));
   a_EH->Assemble();
   a_EH->EliminateTrialDofs(ess_bdr, x.GetBlock(0), b.GetBlock(1));
   a_EH->Finalize();
   SparseMatrix &A_EH = a_EH->SpMat();

   // MixedBilinearForm *a_HE = new MixedBilinearForm(NDfespace, NDfespace);
   // a_HE->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(neg));
   // a_HE->AddDomainIntegrator(new MixedVectorCurlIntegrator(neg));
   // a_HE->Assemble();
   // a_HE->EliminateTestDofs(ess_bdr);
   // a_HE->Finalize();
   // SparseMatrix &A_HE = a_HE->SpMat();
   SparseMatrix &A_HE = *Transpose(A_EH);


   BilinearForm *a_HH = new BilinearForm(NDfespace);
   a_HH->AddDomainIntegrator(new CurlCurlIntegrator(one)); // one is the coeff
   a_HH->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a_HH->Assemble();
   a_HH->Finalize();
   SparseMatrix &A_HH = a_HH->SpMat();

   BlockMatrix *LS_Maxwellop = new BlockMatrix(block_offsets);
   LS_Maxwellop->SetBlock(0, 0, &A_EE);
   LS_Maxwellop->SetBlock(0, 1, &A_HE);
   LS_Maxwellop->SetBlock(1, 0, &A_EH);
   LS_Maxwellop->SetBlock(1, 1, &A_HH);

   SparseMatrix * S = LS_Maxwellop->CreateMonolithic();

   UMFPackSolver *invE = new UMFPackSolver;
   invE->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   invE->SetOperator(LS_Maxwellop->GetBlock(0,0));

   UMFPackSolver *invH = new UMFPackSolver;
   invH->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   invH->SetOperator(LS_Maxwellop->GetBlock(1,1));

   BlockDiagonalPreconditioner *prec = new BlockDiagonalPreconditioner(block_offsets);
   prec->SetDiagonalBlock(0, invE);
   prec->SetDiagonalBlock(1, invH);

   BlkSchwarzSmoother * M = new BlkSchwarzSmoother(cmesh,ref_levels, NDfespace, S);
   M->SetNumSmoothSteps(smth_maxit);
   M->SetDumpingParam(theta);


   cout << "Size of fine grid system: "
        << 2.0 * A_EE.NumRows() << " x " << 2.0 * A_EE.NumCols() << endl;


   int maxit(100);
   double rtol(1.e-6);
   double atol(0.0);
   x = 0.0;

   CGSolver pcg;
   // GMRESSolver pcg;
   pcg.SetAbsTol(atol);
   pcg.SetRelTol(rtol);
   pcg.SetMaxIter(maxit);
   pcg.SetPreconditioner(*M);
   pcg.SetOperator(*LS_Maxwellop);
   // pcg.SetOperator(*S);
   // pcg.SetPreconditioner(*prec);
   pcg.SetPrintLevel(1);
   pcg.Mult(b, x);

   E_gf->MakeRef(NDfespace, x.GetBlock(0), 0);
   H_gf->MakeRef(NDfespace, x.GetBlock(1), 0);

   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double Error_E = E_gf->ComputeL2Error(*Eex, irs);
   double Error_H;
   Error_H = H_gf->ComputeL2Error(*Hex, irs);
   cout << "|| E_h - E || = " << Error_E << "\n";
   cout << "|| H_h - H || = " << Error_H << "\n";

   cout << "Total error = " << sqrt(Error_H*Error_H+Error_E*Error_E) << "\n";

   GridFunction *E_exgf = new GridFunction(NDfespace);
   E_exgf->ProjectCoefficient(*Eex);


   GridFunction *H_exgf = new GridFunction(NDfespace);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      // socketstream cmesh_sock(vishost, visport);
      // cmesh_sock.precision(8);
      // socketstream mesh_sock(vishost, visport);
      // mesh_sock.precision(8);
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      socketstream ex_sock(vishost, visport);
      ex_sock.precision(8);
      socketstream sol_sockH(vishost, visport);
      sol_sockH.precision(8);
      socketstream ex_sockH(vishost, visport);
      ex_sockH.precision(8);
      if (dim == 2)
      {
         sol_sock << "solution\n"
                  << *mesh << *E_gf << "window_title 'Numerical E'" << "keys rRljc\n"
                  << flush;
         ex_sock << "solution\n"
                 << *mesh << *E_exgf << "window_title 'Exact E'" << "keys rRljc\n"
                 << flush;
         sol_sockH << "solution\n"
                  << *mesh << *H_gf << "window_title 'Numerical H'" << "keys rRljc\n"
                  << flush;
         ex_sockH << "solution\n"
                 << *mesh << *H_exgf << "window_title 'Exact H'" << "keys rRljc\n"
                 << flush;        
      }
      else
      {
         sol_sock << "solution\n"
                  << *mesh << *E_gf << "keys lc\n"
                  << flush;
         ex_sock << "solution\n"
                 << *mesh << *E_exgf << "keys lc\n"
                 << flush;
      }
   }
   delete a_EE;
   delete a_EH;
   delete a_HH;
   delete b_E;
   delete b_H;
   delete NDfec;
   delete NDfespace;
   return 0;
}

//define exact solution
void E_exact(const Vector &x, Vector &E)
{
   double curlE[3], curl2E[3];
   get_maxwell_solution(x, E, curlE, curl2E);
}

void H_exact(const Vector &x, Vector &H)
{
   double E[3], curlE[3], curl2E[3];
   get_maxwell_solution(x, E, curlE, curl2E);
   for (int i = 0; i < dim; i++) H(i) = curlE[i] / omega;
}

void f_exact_H(const Vector &x, Vector &f)
{
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   // curl H - omega E = f
   // = curl (curl E / omega) - omega E
   f(0) = curl2E[0] / omega - omega * E[0];
   f(1) = curl2E[1] / omega - omega * E[1];
   f(2) = curl2E[2] / omega - omega * E[2];
}


void scaledf_exact_H(const Vector &x, Vector &f)
{
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   // curl H - omega E = f
   // = - omega *( curl (curl E / omega) - omega E)
   f(0) = -omega * (curl2E[0] / omega - omega * E[0]);
   f(1) = -omega * (curl2E[1] / omega - omega * E[1]);
   f(2) = -omega * (curl2E[2] / omega - omega * E[2]);
}

void get_maxwell_solution(const Vector &X, double E[], double curlE[], double curl2E[])
{
   double x = X[0];
   double y = X[1];
   double z = X[2];
   if (sol == 0) // polynomial
   {
      // Polynomial vanishing on the boundary
      E[0] = y * z * (1.0 - y) * (1.0 - z);
      E[1] = (1.0 - x) * x * y * (1.0 - z) * z;
      E[2] = (1.0 - x) * x * (1.0 - y) * y;
      //
      curlE[0] = -(-1.0 + x) * x * (1.0 + y * (-3.0 + 2.0 * z));
      curlE[1] = -2.0 * (-1.0 + y) * y * (x - z);
      curlE[2] = (1.0 + (-3.0 + 2.0 * x) * y) * (-1.0 + z) * z;

      curl2E[0] = -2.0 * (-1.0 + y) * y + (-3.0 + 2.0 * x) * (-1.0 + z) * z;
      curl2E[1] = -2.0 * y * (-x + x * x + (-1.0 + z) * z);
      curl2E[2] = -2.0 * (-1.0 + y) * y + (-1.0 + x) * x * (-3.0 + 2.0 * z);
   }
   else if (sol == 1) // sinusoidal
   {
      E[0] = sin(omega * y);
      E[1] = sin(omega * z);
      E[2] = sin(omega * x);

      curlE[0] = -omega * cos(omega * z);
      curlE[1] = -omega * cos(omega * x);
      curlE[2] = -omega * cos(omega * y);

      curl2E[0] = omega * omega * E[0];
      curl2E[1] = omega * omega * E[1];
      curl2E[2] = omega * omega * E[2];
   }
   else if (sol == 2) // point source
   {
      MFEM_ABORT("Case unfinished");
   }
   else if (sol == 3) // plane wave
   {
      double coeff = omega / sqrt(3.0);
      E[0] = cos(coeff * (x + y + z));
      E[1] = 0.0;
      E[2] = 0.0;

      curlE[0] = 0.0;
      curlE[1] = -coeff * sin(coeff * (x + y + z));
      curlE[2] = coeff * sin(coeff * (x + y + z));

      curl2E[0] = 2.0 * coeff * coeff * E[0];
      curl2E[1] = -coeff * coeff * E[0];
      curl2E[2] = -coeff * coeff * E[0];
   }
   else if (sol == -1) 
   {
      E[0] = cos(omega * y);
      E[1] = 0.0;

      curlE[0] = 0.0;
      curlE[1] = 0.0;
      curlE[2] = -omega * sin(omega * y);

      curl2E[0] = omega*omega * cos(omega*y);  
      curl2E[1] = 0.0;
      curl2E[2] = 0.0;
   }
}
