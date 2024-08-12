//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh ex1 -m ../data/star.mesh ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/octahedron.mesh -o 1
//               ex1 -m ../data/periodic-annulus-sector.msh
//               ex1 -m ../data/periodic-torus-sector.msh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -fa -d cuda
//               ex1 -pa -d raja-cuda
//             * ex1 -pa -d raja-hip
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cpu -o 4 -a
//               ex1 -pa -d ceed-cpu -m ../data/square-mixed.mesh
//               ex1 -pa -d ceed-cpu -m ../data/fichera-mixed.mesh
//             * ex1 -pa -d ceed-cuda
//             * ex1 -pa -d ceed-hip
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m
//               ../data/square-mixed.mesh ex1 -pa -d ceed-cuda:/gpu/cuda/shared
//               -m ../data/fichera-mixed.mesh ex1 -m ../data/beam-hex.mesh -pa
//               -d cuda ex1 -m ../data/beam-tet.mesh -pa -d ceed-cpu ex1 -m
//               ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
   std::function<double(const double)> f;
public:
   MappedGridFunctionCoefficient(GridFunction &gf,
                                 std::function<double(const double)> f):GridFunctionCoefficient(&gf), f(f) {}

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return f(GridFunctionCoefficient::Eval(T, ip));
   }
};

class VecGF2MatrixCF : public MatrixCoefficient
{
   GridFunction &gf;
   std::function<void(const Vector &, DenseMatrix &)> vec2mat;
   Vector val;
public:
   VecGF2MatrixCF(GridFunction &gf,
                  std::function<void(const Vector &, DenseMatrix &)> vec2mat)
      : MatrixCoefficient(gf.VectorDim()), gf(gf), vec2mat(vec2mat),
        val(gf.VectorDim()) {}
   VecGF2MatrixCF(GridFunction &gf,
                  std::function<void(const Vector &, DenseMatrix &)> vec2mat,
                  const int w, const int h)
      : MatrixCoefficient(w, h), gf(gf), vec2mat(vec2mat), val(gf.VectorDim()) {}
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      gf.GetVectorValue(T, ip, val);
      vec2mat(val, K);
   };
};

class VecGF2VecCF : public VectorCoefficient
{
   GridFunction &gf;
   std::function<void(const Vector &, Vector &)> vec2vec;
   Vector val;
public:
   VecGF2VecCF(GridFunction &gf,
               std::function<void(const Vector &, Vector &)> vec2vec)
      : VectorCoefficient(gf.VectorDim()), gf(gf), vec2vec(vec2vec),
        val(gf.VectorDim()) {}
   VecGF2VecCF(GridFunction &gf,
               std::function<void(const Vector &, Vector &)> vec2vec,
               const int h)
      : VectorCoefficient(h), gf(gf), vec2vec(vec2vec), val(gf.VectorDim()) {}
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      gf.GetVectorValue(T, ip, val);
      vec2vec(val, V);
   };
};

class VecGF2CF : public MatrixCoefficient
{
   GridFunction &gf;
   std::function<double(const Vector &)> vec2double;
   Vector val;
public:
   VecGF2CF(GridFunction &gf,
            std::function<double(const Vector &)> vec2double)
      : MatrixCoefficient(gf.VectorDim()), gf(gf), vec2double(vec2double),
        val(gf.VectorDim()) {}
   VecGF2CF(GridFunction &gf,
            std::function<double(const Vector &)> vec2double,
            const int w, const int h)
      : MatrixCoefficient(w, h), gf(gf), vec2double(vec2double),
        val(gf.VectorDim()) {}
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      gf.GetVectorValue(T, ip, val);
      return vec2double(val);
   };
};

const double r(0.2);
inline double varphi(const Vector &x) {return x*x-r*r; }
inline double sign(const double x) { return x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : 0.0; }

class VectorEntropy
{
   std::function<double(const Vector &)> varphi;
   std::function<void(const Vector &, Vector &)> primal2dual;
   std::function<void(const Vector &, Vector &)> dual2primal;
   std::function<void(const Vector &, DenseMatrix &)> der_dual2primal;
public:
   VectorEntropy(
      std::function<double(const Vector &)> varphi,
      std::function<void(const Vector &, Vector &)> primal2dual,
      std::function<void(const Vector &, Vector &)> dual2primal,
      std::function<void(const Vector &, DenseMatrix &)> der_dual2primal,
      BilinearFormIntegrator *L=nullptr, LinearFormIntegrator* l=nullptr)
      :varphi(varphi), primal2dual(primal2dual), dual2primal(dual2primal),
       der_dual2primal(der_dual2primal) {}
   VecGF2MatrixCF GetDerd2p_cf(GridFunction &psi)
   {
      return VecGF2MatrixCF(psi, der_dual2primal);
   }
   VecGF2VecCF Getd2p_cf(GridFunction &psi)
   {
      return VecGF2VecCF(psi, dual2primal);
   }

};

VectorEntropy get_hellinger()
{
   return VectorEntropy(
   [](const Vector &x) {return -std::sqrt(1.0 - (x*x)); },
   [](const Vector &x, Vector &y) {y = x; y *= -1.0 / std::sqrt(1.0 - (x*x)); },
   [](const Vector &x, Vector &y) {y = x; y *= 1.0 / std::sqrt(1.0 + (x*x)); },
   [](const Vector &x, DenseMatrix &K)
   {
      const double phi = 1.0 / std::sqrt(1.0 + (x*x));
      K.Diag(phi, x.Size());
      AddMult_a_VVt(-std::pow(phi,3.0), x, K);
   });
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/periodic-square.mesh";
   int order = 1; int ref_levels = 3;
   bool visualization = true;
   double growth_rate = 2.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Uniform refinement level");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&growth_rate, "-gr", "--growth-rate",
                  "Step size growth rate, alpha = r^k");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   // Mesh mesh(mesh_file, 1, 1);
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::Type::QUADRILATERAL, false, 2.0, 2.0);
   mesh.Transform([](const Vector &x, Vector &newx){newx=x; newx -= 1.0; });
   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection h1fec(order, sdim);
   L2_FECollection l2fec(order, sdim);
   FiniteElementSpace h1fes(&mesh, &h1fec);
   FiniteElementSpace l2fes_vec(&mesh, &l2fec, sdim);

   int dof_h1(h1fes.GetTrueVSize()), dof_l2(l2fes_vec.GetTrueVSize());

   Array<int> offsets({0, dof_h1, dof_l2, 1});
   offsets.PartialSum();
   BlockVector x(offsets), x_old(offsets), x_old_newt(offsets), rhs(offsets);
   BlockMatrix A(offsets);

   x = 0.0;

   GridFunction u(&h1fes, x.GetBlock(0)), psi(&l2fes_vec, x.GetBlock(1));
   GridFunction u_old(&h1fes, x_old.GetBlock(0)),
                psi_old(&l2fes_vec, x_old.GetBlock(1));
   GridFunction u_old_newt(&h1fes, x_old_newt.GetBlock(0)),
                psi_old_newt(&l2fes_vec, x_old_newt.GetBlock(1));
   GridFunction grad_u_from_psi(&l2fes_vec);

   // Discrete coefficients
   GridFunctionCoefficient u_cf(&u);
   GradientGridFunctionCoefficient grad_u_cf(&u);
   GradientGridFunctionCoefficient grad_u_old_cf(&u_old);

   VectorGridFunctionCoefficient psi_cf(&psi);
   VectorGridFunctionCoefficient psi_old_cf(&psi_old);
   DivergenceGridFunctionCoefficient div_psi_cf(&psi);

   // Entropy related coefficients
   VectorEntropy hellinger = get_hellinger();

   VecGF2VecCF psi2u = hellinger.Getd2p_cf(psi);
   VecGF2MatrixCF der_psi2u = hellinger.GetDerd2p_cf(psi);
   ScalarMatrixProductCoefficient neg_der_psi2u(-1.0, der_psi2u);

   MatrixVectorProductCoefficient der_psi2u_psi(der_psi2u, psi_cf);
   VectorSumCoefficient psi_newton_res(psi2u, der_psi2u_psi, 1.0, -1.0);
   // Energy related coefficient
   FunctionCoefficient neg_sign_varphi([](const Vector &x) { return -sign(varphi(x)); });

   // Other coefficients
   ConstantCoefficient one_cf(1.0), neg_one_cf(-1.0);

   // Setup bilinear forms for Newton iteration
   BilinearForm psi_newton(&l2fes_vec), laplacian(&h1fes);
   MixedBilinearForm grad(&h1fes, &l2fes_vec);

   psi_newton.AddDomainIntegrator(new VectorMassIntegrator(neg_der_psi2u));
   laplacian.AddDomainIntegrator(new DiffusionIntegrator());
   grad.AddDomainIntegrator(new GradientIntegrator());

   laplacian.Assemble();
   laplacian.Finalize(true);
   grad.Assemble();
   grad.Finalize(true);

   auto gradT = *Transpose(grad.SpMat());

   A.SetBlock(0, 0, &laplacian.SpMat());
   A.SetBlock(0, 1, &gradT);
   A.SetBlock(1, 0, &grad.SpMat());

   // Avg 0 condition
   LinearForm avg0_data(&h1fes);
   avg0_data.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
   avg0_data.Assemble();
   Array<int> avg0_i({0, dof_h1}), avg0_j(dof_h1);
   std::iota(avg0_j.begin(), avg0_j.end(), 0);
   SparseMatrix avg0(avg0_i.GetData(), avg0_j.GetData(), avg0_data.GetData(), 1,
                     dof_h1, false, false, true);
   auto avg0T = *Transpose(avg0);

   A.SetBlock(2, 0, &avg0);
   A.SetBlock(0, 2, &avg0T);

   // linear forms
   LinearForm obj(&h1fes),
              prox_res_lf(&h1fes, rhs.GetBlock(0).GetData()),
              psi_newton_res_lf(&l2fes_vec, rhs.GetBlock(1).GetData());

   obj.AddDomainIntegrator(new DomainLFIntegrator(neg_sign_varphi));
   prox_res_lf.AddDomainIntegrator(new DomainLFGradIntegrator(psi_old_cf));
   prox_res_lf.AddDomainIntegrator(new DomainLFGradIntegrator(grad_u_old_cf));
   psi_newton_res_lf.AddDomainIntegrator(new VectorDomainLFIntegrator(
                                            psi_newton_res));

   double volume_shift;
   MappedGridFunctionCoefficient shifted_sign_u(u, [&volume_shift](const double x){return sign(x - volume_shift);});
   LinearForm int_shifted_sign_u(&h1fes);
   int_shifted_sign_u.AddDomainIntegrator(new DomainLFIntegrator(shifted_sign_u));

   obj.Assemble();

   socketstream dist_sock, grad_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      dist_sock.open(vishost, visport);
      dist_sock.precision(8);
      dist_sock << "solution\n" << mesh << u << flush;
      grad_sock.open(vishost, visport);
      grad_sock.precision(8);
      grad_u_from_psi.ProjectCoefficient(psi2u);
      grad_sock << "solution\n" << mesh << grad_u_from_psi << flush;
   }
   const double prox_tol = 1e-06;
   double prox_residual = prox_tol + 1;

   const int prox_max_it = 1e3;
   int prox_it = 0;

   const double newt_tol = 1e-06;
   double newt_residual = newt_tol + 1;

   const int newt_max_it = 20;
   int newt_it = 0;

   double step_size = 1.0;
   while (prox_residual > prox_tol && prox_it < prox_max_it)
   {
      prox_it++;
      x_old = x;
      prox_res_lf.Assemble();
      prox_res_lf.Add(-step_size, obj);
      newt_residual = newt_tol + 1;
      newt_it = 0;
      std::cout << "Newton Iteration Start" << std::endl;
      while (newt_residual > newt_tol && newt_it < newt_max_it)
      {
         std::cout << "\tIteration " << newt_it++ << ": ";
         x_old_newt = x;

         psi_newton.Assemble(false);
         psi_newton.Finalize(false);
         psi_newton_res_lf.Assemble();

         A.SetBlock(1, 1, &psi_newton.SpMat());

         SparseMatrix *A_mono = A.CreateMonolithic();
         UMFPackSolver umf(*A_mono);
         umf.Mult(rhs, x);

         const double newt_residual_psi = psi_old_newt.ComputeL2Error(psi_cf);
         const double newt_residual_u = u_old_newt.ComputeGradError(&grad_u_cf);

         newt_residual = std::sqrt(std::pow(newt_residual_psi, 2.0)
                                   + std::pow(newt_residual_u, 2.0));
         std::cout << newt_residual << std::endl;
         psi_newton.Update();
      }
      const double prox_residual_psi = psi_old.ComputeL2Error(psi_cf)/step_size;
      const double prox_residual_u = u_old.ComputeGradError(&grad_u_cf);

      prox_residual = std::sqrt(std::pow(prox_residual_psi, 2.0)
                                + std::pow(prox_residual_u, 2.0));
      std::cout << "Proxi Iteration: " << prox_it << ": " << prox_residual <<
                " (" << prox_residual_psi << ", " << prox_residual_u << ")" << std::endl;
      if (visualization)
      {
         dist_sock << "solution\n" << mesh << u << flush;
         grad_u_from_psi.ProjectCoefficient(psi2u);
         grad_sock << "solution\n" << mesh << grad_u_from_psi << flush;
      }
      step_size *= growth_rate;
      step_size = std::min(step_size, 1e09);
   }
}
