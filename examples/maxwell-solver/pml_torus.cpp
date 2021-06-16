

// sample runs: ./pml_torus -prob 2 -ref 2 -o 2 -f 0.6

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

// class PMLMatrixCoefficient : public MatrixCoefficient
// {
// private:
//    ToroidPML * pml = nullptr;
//    void (*Function)(const Vector &, ToroidPML * , DenseMatrix &);
// public:
//    PMLMatrixCoefficient(int dim, void(*F)(const Vector &, ToroidPML *,
//                                               DenseMatrix &),
//                             ToroidPML * pml_)
//       : MatrixCoefficient(dim), pml(pml_), Function(F)
//    {}

//    using MatrixCoefficient::Eval;

//    virtual void Eval(DenseMatrix &M, ElementTransformation &T,
//                      const IntegrationPoint &ip)
//    {
//       double x[3];
//       Vector transip(x, 3);
//       T.Transform(ip, transip);
//       M.SetSize(height,width);
//       (*Function)(transip, pml, M);
//    }
// };


void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);

void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);

void E_exact_Curl_Re(const Vector &x, Vector &E);
void E_exact_Curl_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, ToroidPML * pml, Vector &D);
void detJ_JT_J_inv_Im(const Vector &x, ToroidPML * pml, Vector &D);
void detJ_inv_JT_J_Re(const Vector &x, ToroidPML * pml, Vector &D);
void detJ_inv_JT_J_Im(const Vector &x, ToroidPML * pml, Vector &D);


// void detJ_JT_J_inv_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M);
// void detJ_JT_J_inv_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M);
// void detJ_inv_JT_J_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M);
// void detJ_inv_JT_J_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M);



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
   int ref_levels = 1;
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
         rpml_thickness[1] = 0.5; 
         rstretch = true;
      }
      break;
      default: break;
   }
   
   tpml.SetPmlAxes(zstretch,rstretch,astretch);
   tpml.SetPmlWidth(zpml_thickness,rpml_thickness,apml_thickness);
   tpml.SetOmega(omega); 

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   ComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);

   ConvergenceStudy rates_r;
   ConvergenceStudy rates_i;

   for (int iter = 0; iter<ref_levels; iter++)
   {
      int size = fespace->GetTrueVSize();
      cout << "Number of finite element unknowns: " << size << endl;
      tpml.SetAttributes(mesh); 

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      VectorFunctionCoefficient f(dim, source);
      ComplexLinearForm b(fespace, conv);
      // b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
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

      SparseMatrix * SpMat = (*A.As<ComplexSparseMatrix>()).GetSystemMatrix();
      HYPRE_Int global_size = SpMat->Height();
      HYPRE_Int row_starts[2]; row_starts[0] = 0; row_starts[1] = global_size;
      HypreParMatrix * HypreMat = new HypreParMatrix(MPI_COMM_SELF,global_size,row_starts,SpMat);
      {
         MUMPSSolver mumps;
         mumps.SetOperator(*HypreMat);
         mumps.Mult(B,X);
      }

      a.RecoverFEMSolution(X, b, x);

      if (prob_kind == 3)
      {
         rates_r.SetElementList(tpml.GetMarkedPMLElements());
         rates_i.SetElementList(tpml.GetMarkedPMLElements());

         VectorFunctionCoefficient E_ex_Re(dim, E_exact_Re);
         VectorFunctionCoefficient E_ex_Im(dim, E_exact_Im);
         VectorFunctionCoefficient E_Curl_Re(cdim, E_exact_Curl_Re);
         VectorFunctionCoefficient E_Curl_Im(cdim, E_exact_Curl_Im);

         rates_r.AddHcurlGridFunction(&x.real(),&E_ex_Re,&E_Curl_Re);
         rates_i.AddHcurlGridFunction(&x.imag(),&E_ex_Im,&E_Curl_Im);
      }

      if (iter == ref_levels) break;
      mesh->UniformRefinement();
      fespace->Update();
      x.Update();
   }

   if (prob_kind == 3)
   {
      rates_r.Print(false);      
      rates_i.Print(false);     
   }
   


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

      ParaViewDataCollection * pd = new ParaViewDataCollection("PML_circle16", mesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", &x_t);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();


      // while (sol_sock)
      // {
      for (int i = 1; i<num_frames; i++)
      {   
         cout << i << endl;
         double t = (double)(i % num_frames) / num_frames;
         // ostringstream oss;
         // oss << "Harmonic Solution (t = " << t << " T)";

         add(cos(2.0 * M_PI * t), x.real(),
             sin(2.0 * M_PI * t), x.imag(), x_t);
         // sol_sock << "solution\n"
         //          << *mesh << x_t
         //          << "window_title '" << oss.str() << "'" << flush;
         // i++;

         pd->SetCycle(i);
         pd->SetTime((double)i);
         pd->Save();
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
   {  // for a straight waveguide
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
      complex<double> H0, H0_r, H0_rr, H0_rrr;
      complex<double> H1, H1_r, H1_rr;
      complex<double> H2, H2_r;
      complex<double> H3;
      H0 = jn(0,beta) + zi * yn(0,beta);
      H1 = jn(1,beta) + zi * yn(1,beta);
      H2 = jn(2,beta) + zi * yn(2,beta);
      // H3 = jn(3,beta) + zi * yn(3,beta);

      H0_r = - k * H1;
      H0_rr = - k * k * (1.0/beta * H1 - H2); 

      // First derivatives
      double r_x = x0 / r;
      double r_y = x1 / r;
      double r_xy = -(r_x / r) * r_y;
      double r_xx = (1.0 / r) * (1.0 - r_x * r_x);

      complex<double> val, val_x, val_xx, val_xxx, val_xy, val_xyy;
      val = 0.25 * zi * H0;
      val_xx = 0.25 * zi * (r_xx * H0_r + r_x * r_x * H0_rr);
      val_xy = 0.25 * zi * (r_xy * H0_r + r_x * r_y * H0_rr);
      E[0] = zi / k * (k * k * val + val_xx);
      E[1] = zi / k * val_xy;
   }
}

void E_exact_Curl_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   vector<complex<double>> Eval(E.Size());
   maxwell_curl(x, Eval);
   for (int i = 0; i < E.Size(); ++i)
   {
      E[i] = Eval[i].real();
   }
}

void E_exact_Curl_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   vector<complex<double>> Eval(E.Size());
   maxwell_curl(x, Eval);
   for (int i = 0; i < E.Size(); ++i)
   {
      E[i] = Eval[i].imag();
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
   complex<double> H0_r;
   complex<double> H1;
   // complex<double> H2, H2_r;
   // complex<double> H3;
   // H0 = jn(0,beta) + zi * yn(0,beta);
   H1 = jn(1,beta) + zi * yn(1,beta);
   // H2 = jn(2,beta) + zi * yn(2,beta);
   // H3 = jn(3,beta) + zi * yn(3,beta);

   H0_r   = - k * H1;
   // H1_r   =   k * (1.0/beta * H1 - H2);
   // H2_r   = - k * (2.0/beta * H2 - H3); 
   // H0_rr  = - k * H1_r;
   // H1_rr  = k * k * (- 2.0 /(beta * beta) * H1 + 1.0/beta * H1_r - H2_r); 
   // H0_rrr = - k * H1_rr;

   // First derivatives
   // double r_x = x0 / r;
   double r_y = x1 / r;
   // double r_xy = -(r_x / r) * r_y;
   // double r_yx = r_xy;
   // double r_yy = (1.0 / r) * (1.0 - r_y * r_y);
   // double r_xx = (1.0 / r) * (1.0 - r_x * r_x);
   // double r_xxx = r_x * (r_x * r_x - 2. * r_xx * r - 1.0) /(r * r);
   // double r_xyy = (r_x * r_y * r_y - r * r_xy * r_y - r * r_x * r_yy)/(r * r);

   complex<double> val_y;
   // val = 0.25 * zi * H0;
   val_y = 0.25 * zi * H0_r * r_y;
   // val_xx = 0.25 * zi * (r_xx * H0_r + r_x * r_x * H0_rr);
   // val_xy = 0.25 * zi * (r_xy * H0_r + r_x * r_y * H0_rr);
   curlE[0] = zi / k * (- k * k * val_y);
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

// void detJ_JT_J_inv_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M)
// {
//    ComplexDenseMatrix J(dim);
//    pml->StretchFunction(x,J,omega);
//    complex<double> det = J.Det();
//    ComplexDenseMatrix JtJ(dim);
//    MultAtB(J,J,JtJ);
//    ComplexDenseMatrixInverse InvJtJ(JtJ);
//    InvJtJ *=det;
//    InvJtJ.GetReal(M);
// }

// void detJ_JT_J_inv_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M)
// {
//    ComplexDenseMatrix J(dim);
//    pml->StretchFunction(x,J,omega);
//    complex<double> det = J.Det();
//    ComplexDenseMatrix JtJ(dim);
//    MultAtB(J,J,JtJ);
//    ComplexDenseMatrixInverse InvJtJ(JtJ);
//    InvJtJ *=det;
//    InvJtJ.GetImag(M);
// }

// void detJ_inv_JT_J_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M)
// {
//    ComplexDenseMatrix J(dim);
//    pml->StretchFunction(x,J,omega);
//    complex<double> det = J.Det();
//    if (dim == 2)
//    {
//       M = (1.0 / det).real();
//    }
//    else
//    {
//       ComplexDenseMatrix JtJ(dim);
//       MultAtB(J,J,JtJ);
//       JtJ *= 1.0/det;
//       JtJ.GetReal(M);
//    }
// }

// void detJ_inv_JT_J_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M)
// {
//    ComplexDenseMatrix J(dim);
//    pml->StretchFunction(x,J,omega);
//    complex<double> det = J.Det();
//    if (dim == 2)
//    {
//       M = (1.0 / det).imag();
//    }
//    else
//    {
//       ComplexDenseMatrix JtJ(dim);
//       MultAtB(J,J,JtJ);
//       JtJ *= 1.0/det;
//       JtJ.GetImag(M);
//    }
// }
