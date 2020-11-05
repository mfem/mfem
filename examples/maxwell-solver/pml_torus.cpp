
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "common/PML.hpp"

using namespace std;
using namespace mfem;

void maxwell_solution(const Vector &x, vector<complex<double>> &E);
void maxwell_curl(const Vector &x, vector<complex<double>> &curlE);

int prob_kind=0;
double L;
double ylim;

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   ToroidPML * pml = nullptr;
   void (*Function)(const Vector &, ToroidPML * , Vector &);
public:
   PMLDiagMatrixCoefficient(int dim, void(*F)(const Vector &, ToroidPML *,
                                              Vector &),
                            ToroidPML * pml_)
      : VectorCoefficient(dim), pml(pml_), Function(F)
   {}

   using VectorCoefficient::Eval;

   virtual void Eval(Vector &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(vdim);
      (*Function)(transip, pml, K);
   }
};

class PMLMatrixCoefficient : public MatrixCoefficient
{
private:
   ToroidPML * pml = nullptr;
   void (*Function)(const Vector &, ToroidPML * , DenseMatrix &);
public:
   PMLMatrixCoefficient(int dim, void(*F)(const Vector &, ToroidPML *,
                                              DenseMatrix &),
                            ToroidPML * pml_)
      : MatrixCoefficient(dim), pml(pml_), Function(F)
   {}

   using MatrixCoefficient::Eval;

   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      M.SetSize(height,width);
      (*Function)(transip, pml, M);
   }
};


void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);

void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, ToroidPML * pml, Vector &D);
void detJ_JT_J_inv_Im(const Vector &x, ToroidPML * pml, Vector &D);
void detJ_inv_JT_J_Re(const Vector &x, ToroidPML * pml, Vector &D);
void detJ_inv_JT_J_Im(const Vector &x, ToroidPML * pml, Vector &D);


void detJ_JT_J_inv_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M);
void detJ_JT_J_inv_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M);
void detJ_inv_JT_J_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M);
void detJ_inv_JT_J_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M);



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
   // const char *mesh_file = "torus1_4.mesh";
   // const char *mesh_file = "waveguide-bend2.mesh";
   const char *mesh_file = "meshes/waveguide-bend.mesh";

   int order = 1;
   int ref_levels = 3;
   double freq = 5.0;
   bool herm_conv = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&prob_kind, "-prob", "--problem-kind",
                  "Problem/mesh choice");                  
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


   switch (prob_kind)
   {
   case 0: 
   {
      mesh_file = "meshes/waveguide-bend.mesh"; 
      L = -2.;
      ylim = -3;
   }
   break;
   case 1: 
   {
      mesh_file = "meshes/waveguide-bend2.mesh";
      L = -5.;
      ylim = 0.0;
   }
   break;
   case 2: mesh_file = "meshes/toroid3_4_2.mesh"; break;
   // case 3: mesh_file = "toroid-hex-o3-s0_r.mesh"; break;
   // case 3: mesh_file = "../../data/square-disc.mesh"; break;
   case 3: mesh_file = "meshes/annulus-quad-o3.mesh"; break;
   // case 3: mesh_file = "cylinder.mesh"; break;
   default:
      MFEM_ABORT("Not a valid problem choice ");
      break;
   }

   Mesh * mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   mesh->RemoveInternalBoundaries();

   // Angular frequency
   omega = 2.0 * M_PI * freq;

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   ToroidPML tpml(mesh);
   Vector zlim, rlim, alim;
   tpml.GetDomainBdrs(zlim,rlim,alim);
   Vector zpml_thickness(2); zpml_thickness = 0.0;
   Vector rpml_thickness(2); rpml_thickness = 0.0;
   Vector apml_thickness(2); apml_thickness = 0.0; 
   bool zstretch = false;
   bool astretch = false;
   bool rstretch = false;
   switch (prob_kind)
   {
      case 0: break;
      case 1: break;
      case 2: 
      {
         apml_thickness[1] = 45.0; 
         astretch = true;
      }
      break;// degrees 
      case 3: 
      {
         rpml_thickness[1] = 0.3; 
         rstretch = true;
      }
      break;
      default: break;
   }
   
   tpml.SetPmlAxes(zstretch,rstretch,astretch);
   tpml.SetPmlWidth(zpml_thickness,rpml_thickness,apml_thickness);
   tpml.SetAttributes(mesh); 
   tpml.SetOmega(omega); 

   // Array<int> * marked_elems = new Array<int>(mesh->GetNE());
   // marked_elems = tpml.GetMarkedPMLElements();
   // marked_elems->Print();
   // cout << "nrelems = " << mesh->GetNE() << endl;

   cout << "axial range     = " ; zlim.Print();
   cout << "radial range    = " ; rlim.Print();
   cout << "azimuthal range = " ; alim.Print();

    if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      socketstream mesh_sock(vishost, visport);
      mesh_sock.precision(8);
      mesh_sock << "mesh\n"
                  << *mesh << "window_title 'Mesh'" << flush;
      ofstream mesh_ofs("output/pml_torus.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);                  
   }
   // 6. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   int size = fespace->GetTrueVSize();

   cout << "Number of finite element unknowns: " << size << endl;

   // 7. Essential boundary dofs on the whole boundary.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
   }
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // ess_tdof_list.Print();

   // 8. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   VectorFunctionCoefficient f(dim, source);
   ComplexLinearForm b(fespace, conv);
   // b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   b.Vector::operator=(0.0);
   b.Assemble();

   // 10. Define the solution vector x as a complex finite element grid function
   //     corresponding to fespace.
   ComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);
   x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);
   // x.ProjectCoefficient(E_Re, E_Im);

   VectorFunctionCoefficient E_Re_ex(dim, E_exact_Re);
   VectorFunctionCoefficient E_Im_ex(dim, E_exact_Im);
   ComplexGridFunction x_ex(fespace);
   x_ex.ProjectCoefficient(E_Re_ex, E_Im_ex);

   cout << "x.norm = " << x.Norml2() << endl;

   // 11. Set up the sesquilinear form a(.,.)
   //
   //     In Comp
   //     Domain:   1/mu (Curl E, Curl F) - omega^2 * epsilon (E,F)
   //
   //     In PML:   1/mu (1/det(J) J^T J Curl E, Curl F)
   //               - omega^2 * epsilon (det(J) * (J^T J)^-1 * E, F)
   //
   //     where J denotes the Jacobian Matrix of the PML Stretching function
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
   // PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, &tpml);
   // PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, &tpml);
   // ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
   // ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
   // VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
   // VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

   // PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,&tpml);
   // PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,&tpml);
   // ScalarVectorProductCoefficient c2_Re(omeg,pml_c2_Re);
   // ScalarVectorProductCoefficient c2_Im(omeg,pml_c2_Im);
   // VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
   // VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

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

   // 12. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: assembly, eliminating
   //     boundary conditions, applying conforming constraints for
   //     non-conforming AMR, etc.
   a.Assemble(0);

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   

   SparseMatrix * SpMat = (*A.As<ComplexSparseMatrix>()).GetSystemMatrix();
   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   HYPRE_Int global_size = SpMat->Height();
   HYPRE_Int row_starts[2]; row_starts[0] = 0; row_starts[1] = global_size;
   HypreParMatrix * HypreMat = new HypreParMatrix(MPI_COMM_SELF,global_size,row_starts,SpMat);
   {
      MUMPSSolver mumps;
      mumps.SetOperator(*HypreMat);
      mumps.Mult(B,X);
   }
   chrono.Stop();

   // cout << "mumps time = " << chrono.RealTime() << endl;

   // // 13. Solve using a direct or an iterative solver
   // chrono.Clear();
   // chrono.Start();
   // // ComplexUMFPackSolver csolver(*A.As<ComplexSparseMatrix>());
   // UMFPackSolver csolver(*SpMat);
   // csolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   // csolver.SetPrintLevel(1);
   // csolver.Mult(B, X);
   // chrono.Stop();
   // cout << "UMFPack time = " << chrono.RealTime() << endl;

   // 14. Recover the solution as a finite element grid function and compute the
   //     errors if the exact solution is known.
   a.RecoverFEMSolution(X, b, x);

   // If exact is known compute the error
   bool exact_known = (prob_kind == 3) ? true : false;
   if (exact_known)
   {
      VectorFunctionCoefficient E_ex_Re(dim, E_exact_Re);
      VectorFunctionCoefficient E_ex_Im(dim, E_exact_Im);
      int order_quad = max(2, 2 * order + 1);

      ConvergenceStudy rates_r;
      ConvergenceStudy rates_i;

      rates_r.AddHcurlGridFunction(&x.real(),&E_ex_Re);
      rates_i.AddHcurlGridFunction(&x.imag(),&E_ex_Im);

      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double L2Error_Re = x.real().ComputeL2Error(E_ex_Re, irs,
                                                  tpml.GetMarkedPMLElements());
      double L2Error_Im = x.imag().ComputeL2Error(E_ex_Im, irs,
                                                  tpml.GetMarkedPMLElements());

      ComplexGridFunction x_gf0(fespace);
      x_gf0 = 0.0;
      double norm_E_Re, norm_E_Im;
      norm_E_Re = x_gf0.real().ComputeL2Error(E_ex_Re, irs,
                                              tpml.GetMarkedPMLElements());
      norm_E_Im = x_gf0.imag().ComputeL2Error(E_ex_Im, irs,
                                              tpml.GetMarkedPMLElements());

      cout << "\n Relative Error (Re part): || E_h - E || / ||E|| = "
            << L2Error_Re / norm_E_Re
            << "\n Relative Error (Im part): || E_h - E || / ||E|| = "
            << L2Error_Im / norm_E_Im
            << "\n Total Error: "
            << sqrt(L2Error_Re*L2Error_Re + L2Error_Im*L2Error_Im) << "\n\n";

      rates_r.Print(true);      
      rates_i.Print(true);      
   }



   // // ComplexDenseMatrix MatZ(2,2);
   // ComplexDenseMatrix MatZ(3,3);
   // MatZ(0,0) =  complex<double>(0.286569,0.736547);  
   // MatZ(0,1) =  complex<double>(0.051340,0.018433);  
   // MatZ(0,2) =  complex<double>(0.526620,0.077061);  
   // MatZ(1,0) =  complex<double>(0.881512,0.036553);  
   // MatZ(1,1) =  complex<double>(0.727566,0.000757);  
   // MatZ(1,2) =  complex<double>(0.853221,0.513477);  
   // MatZ(2,0) =  complex<double>(0.834360,0.822687);  
   // MatZ(2,1) =  complex<double>(0.335619,0.406582);  
   // MatZ(2,2) =  complex<double>(0.029811,0.382764);  

   // cout << "A  " << endl;
   // MatZ.PrintMatlab(cout);

   // ComplexDenseMatrixInverse InvZ(MatZ);
   // cout << "B  " << endl;
   // InvZ.PrintMatlab(cout);
   // cout << "DetMatZ = " << MatZ.Det() << endl;
   // cout << "DetInvZ = " << InvZ.Det() << endl;
   
   // DenseMatrix * Ar = MatZ.real();
   // DenseMatrix * Ai = MatZ.imag();
   // ComplexDenseMatrix M(MatZ.Height());
   // cout << "A * B " << endl;
   // Mult(MatZ,InvZ,M);
   // M.PrintMatlab(cout);

   // cout << "At * B " << endl;
   // MultAtB(MatZ,InvZ,M);
   // M.PrintMatlab(cout);

   // cout << "A' * B " << endl;
   // MultAhB(MatZ,InvZ,M);
   // M.PrintMatlab(cout);
   // cout << "Ar = " ; Ar->PrintMatlab(cout);
   // cout << endl;
   // cout << "Ai = " ; Ai->PrintMatlab(cout);
   // cout << endl;

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

void source(const Vector &x, Vector &f)
{
   Vector center(dim);
   double r = 0.0;
   center = 0.5;
   center(2) = 0.15;

   for (int i = 0; i < dim; ++i)
   {
      r += pow(x[i] - center[i], 2.);
   }
   double n = 5.0 * omega * sqrt(epsilon * mu) / M_PI;
   double coeff = pow(n, 2) / M_PI;
   double alpha = -pow(n, 2) * r;
   f = 0.0;
   f[0] = coeff * exp(alpha);
}

void E_bdr_data_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   if (prob_kind == 2)
   {
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
   else if (prob_kind == 3)
   {
      double r = sqrt(x(0)*x(0) + x(1)*x(1));
      // check if in pml

      // if (abs(r-1.0)<1e-10) 
      // if (r < 0.3) // not in pml
      // if (x(0) <0.8 && x(0)>0.2 && x(1) < 0.8 && x(1) >0.2 )
      // if (x(0) <0.3 && x(0)>-0.3 && x(1) < 0.3 && x(1) >-0.3 )
      if (r < 0.3 )
      {
         vector<complex<double>> Eval(E.Size());
         maxwell_solution(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].real();
         }
      }
   }
   else
   {
      if (x(1) == ylim) 
      {
         vector<complex<double>> Eval(E.Size());
         maxwell_solution(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].real();
         }
      }
   }
}

// Define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   if (prob_kind == 2)
   {
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
   else if (prob_kind == 3)
   {
      double r = sqrt(x(0)*x(0) + x(1)*x(1));
      // if (abs(r-1.0)<1e-10) 
      // if (r < 0.3) // not in pml
      // if (x(0) < 0.5) // not in pml
      // if (x(0) <0.8 && x(0)>0.2 && x(1) < 0.8 && x(1) >0.2 )
      // if (x(0) <0.3 && x(0)>-0.3 && x(1) < 0.3 && x(1) >-0.3 )
      if (r < 0.3 )
      {
         vector<complex<double>> Eval(E.Size());
         maxwell_solution(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].imag();
         }
      }
   }
   else
   {
      if (x(1) == ylim) 
      {
         vector<complex<double>> Eval(E.Size());
         maxwell_solution(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].imag();
         }
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

   if (prob_kind == 2)
   {
      double k = omega * sqrt(epsilon * mu);
      // T_10 mode
      double k10 = sqrt(k * k - M_PI * M_PI);
      E[2] = -zi * k / M_PI * sin(M_PI*(x(0)))*exp(zi * k10 * x(1));
   }
   else
   {
      double k = omega * sqrt(epsilon * mu);
      Vector shift(dim);
      shift = 0.0;
      double x0 = x(0) + shift(0);
      double x1 = x(1) + shift(1);
      double r = sqrt(x0 * x0 + x1 * x1);
      double beta = k * r;

      // Bessel functions
      complex<double> Ho, Ho_r, Ho_rr;
      Ho = jn(0, beta) + zi * yn(0, beta);
      Ho_r = -k * (jn(1, beta) + zi * yn(1, beta));
      Ho_rr = -k * k * (1.0 / beta *
                        (jn(1, beta) + zi * yn(1, beta)) -
                        (jn(2, beta) + zi * yn(2, beta)));

      // First derivatives
      double r_x = x0 / r;
      double r_y = x1 / r;
      double r_xy = -(r_x / r) * r_y;
      double r_xx = (1.0 / r) * (1.0 - r_x * r_x);

      complex<double> val, val_xx, val_xy;
      val = 0.25 * zi * Ho;
      val_xx = 0.25 * zi * (r_xx * Ho_r + r_x * r_x * Ho_rr);
      val_xy = 0.25 * zi * (r_xy * Ho_r + r_x * r_y * Ho_rr);
      E[0] = zi / k * (k * k * val + val_xx);
      E[1] = zi / k * val_xy;
   }
}

void maxwell_curl(const Vector &x, vector<complex<double>> &curlE)
{
   complex<double> zi = complex<double>(0., 1.);

   double k = omega * sqrt(epsilon * mu);
   Vector shift(dim);
   shift = 0.0;
   double x0 = x(0) + shift(0);
   double x1 = x(1) + shift(1);
   double r = sqrt(x0 * x0 + x1 * x1);
   double beta = k * r;

   // Bessel functions
   complex<double> Ho, Ho_r, Ho_rr, Ho_rrr;
   Ho = jn(0, beta) + zi * yn(0, beta);
   Ho_r = -k * (jn(1, beta) + zi * yn(1, beta));
   Ho_rr = -k * k * (1.0 / beta *
                     (jn(1, beta) + zi * yn(1, beta)) -
                     (jn(2, beta) + zi * yn(2, beta)));

   // Ho_rrr = 

   // First derivatives
   double r_x = x0 / r;
   double r_y = x1 / r;
   double r_xy = -(r_x / r) * r_y;
   double r_xx = (1.0 / r) * (1.0 - r_x * r_x);

   complex<double> val, val_x, val_xx, val_xxx, val_xy, val_xyy;
   val = 0.25 * zi * Ho;
   val_xx = 0.25 * zi * (r_xx * Ho_r + r_x * r_x * Ho_rr);
   val_xy = 0.25 * zi * (r_xy * Ho_r + r_x * r_y * Ho_rr);
   vector<complex<double>> E(2);
   E[0] = zi / k * (k * k * val + val_xx);
   E[1] = zi / k * val_xy;

   // 2D curl
   // curlE = E[0]_x - E[1]_y
   complex<double> E0_x = zi/k * ( k * k * val_x + val_xxx);

}


void detJ_JT_J_inv_Re(const Vector &x, ToroidPML * pml, Vector &D)
{
   // vector<complex<double>> dxs(dim);
   // complex<double> det(1.0, 0.0);
   // pml->StretchFunction(x, dxs,omega);
   ComplexDenseMatrix J(dim);
   pml->StretchFunction(x,J,omega);
   complex<double> det = J.Det();

   // for (int i = 0; i < dim; ++i)
   // {
   //    det *= dxs[i];
   // }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(J(i,i), 2)).real();
   }
}

void detJ_JT_J_inv_Im(const Vector &x, ToroidPML * pml, Vector &D)
{
   // vector<complex<double>> dxs(dim);
   // complex<double> det = 1.0;
   // pml->StretchFunction(x, dxs,omega);
   ComplexDenseMatrix J(dim);
   pml->StretchFunction(x,J,omega);
   complex<double> det = J.Det();

   // for (int i = 0; i < dim; ++i)
   // {
   //    det *= dxs[i];
   // }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(J(i,i), 2)).imag();
   }
}

void detJ_inv_JT_J_Re(const Vector &x, ToroidPML * pml, Vector &D)
{
   // vector<complex<double>> dxs(dim);
   // complex<double> det(1.0, 0.0);
   // pml->StretchFunction(x, dxs,omega);
   ComplexDenseMatrix J(dim);
   pml->StretchFunction(x,J,omega);
   complex<double> det = J.Det();
   // for (int i = 0; i < dim; ++i)
   // {
   //    det *= dxs[i];
   // }
   // in the 2D case the coefficient is scalar 1/det(J)
   if (dim == 2)
   {
      D = (1.0 / det).real();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(J(i,i), 2) / det).real();
      }
   }
}

void detJ_inv_JT_J_Im(const Vector &x, ToroidPML * pml, Vector &D)
{
   // vector<complex<double>> dxs(dim);
   // complex<double> det = 1.0;
   // pml->StretchFunction(x, dxs,omega);
   ComplexDenseMatrix J(dim);
   pml->StretchFunction(x,J,omega);
   complex<double> det = J.Det();
   // for (int i = 0; i < dim; ++i)
   // {
   //    det *= dxs[i];
   // }

   if (dim == 2)
   {
      D = (1.0 / det).imag();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(J(i,i), 2) / det).imag();
      }
   }
}


//-----------------------------------------------------------------

void detJ_JT_J_inv_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M)
{
   ComplexDenseMatrix J(dim);
   pml->StretchFunction(x,J,omega);
   complex<double> det = J.Det();
   ComplexDenseMatrix JtJ(dim);
   MultAtB(J,J,JtJ);
   ComplexDenseMatrixInverse InvJtJ(JtJ);
   InvJtJ *=det;
   InvJtJ.GetReal(M);
}

void detJ_JT_J_inv_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M)
{
   ComplexDenseMatrix J(dim);
   pml->StretchFunction(x,J,omega);
   complex<double> det = J.Det();
   ComplexDenseMatrix JtJ(dim);
   MultAtB(J,J,JtJ);
   ComplexDenseMatrixInverse InvJtJ(JtJ);
   InvJtJ *=det;
   InvJtJ.GetImag(M);
}

void detJ_inv_JT_J_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M)
{
   ComplexDenseMatrix J(dim);
   pml->StretchFunction(x,J,omega);
   complex<double> det = J.Det();
   if (dim == 2)
   {
      M = (1.0 / det).real();
   }
   else
   {
      ComplexDenseMatrix JtJ(dim);
      MultAtB(J,J,JtJ);
      JtJ *= 1.0/det;
      JtJ.GetReal(M);
   }
}

void detJ_inv_JT_J_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M)
{
   ComplexDenseMatrix J(dim);
   pml->StretchFunction(x,J,omega);
   complex<double> det = J.Det();
   if (dim == 2)
   {
      M = (1.0 / det).imag();
   }
   else
   {
      ComplexDenseMatrix JtJ(dim);
      MultAtB(J,J,JtJ);
      JtJ *= 1.0/det;
      JtJ.GetImag(M);
   }
}
