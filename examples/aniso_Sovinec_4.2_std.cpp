//                       MFEM Example 1 (Modified) - Parallel Version
//
// Compile with: make ex1p_aniso
//
// Sample runs:  mpirun -np 4 ex1p_aniso -chi 1e6 -o 2 -n 16
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
#include <cassert>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static int prob_ = 1;
static int gamma_ = 10;
static double alpha_ = NAN;
static double chi_ratio_ = 1.0;

double QFunc(const Vector &x)
{
   switch (prob_)
   {
      case 1:
         return 2.0 * M_PI * M_PI * cos(M_PI * x[0]) * cos(M_PI * x[1]);
      case 2:
      {
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double c2x = cos(2.0 * M_PI * x[0]);
         double s2x = sin(2.0 * M_PI * x[0]);
         double c2y = cos(2.0 * M_PI * x[1]);
         double s2y = sin(2.0 * M_PI * x[1]);
         double c2a = cos(2.0 * alpha_);
         double s2a = sin(2.0 * alpha_);
         double ccg = 0.5 * M_PI * M_PI * gamma_ * pow(cx * cy, gamma_ - 2);
         double perp = 1.0 * gamma_ * (c2x * c2y - 1.0) + c2x + c2y + 2.0;
         double para = 0.5 * (gamma_ * (c2x * c2y - s2a * s2x * s2y - 1.0) +
                              (gamma_ - 1.0) * c2a * (c2x - c2y) +
                              c2x + c2y + 2.0);
         return ccg * (1.0 * perp + (chi_ratio_ - 1.0) * para);
      }
   }
}

double TFunc(const Vector &x)
{
   switch (prob_)
   {
      case 1:
         return cos(M_PI * x[0]) * cos(M_PI * x[1]);
      case 2:
         return pow(cos(M_PI * x[0]) * cos(M_PI * x[1]), gamma_);
   }
}

void qParaFunc(const Vector &x, Vector &q)
{
   q.SetSize(2);

   switch (prob_)
   {
      case 1:
         q[0] = 0.0;
         q[1] = 0.0;
         break;
      case 2:
      {
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);
         double ca = cos(alpha_);
         double sa = sin(alpha_);
         double ccg = M_PI * gamma_ * chi_ratio_ * pow(cx * cy, gamma_ - 1);
         double axy = ca * sx * cy + sa * cx * sy;
         q[0] = ccg * axy * ca;
         q[1] = ccg * axy * sa;
         break;
      }
   }
}

void qPerpFunc(const Vector &x, Vector &q)
{
   q.SetSize(2);

   switch (prob_)
   {
      case 1:
      {
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);
         q[0] = M_PI * sx * cy;
         q[1] = M_PI * cx * sy;
      }
      break;
      case 2:
      {
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);
         double ca = cos(alpha_);
         double sa = sin(alpha_);
         double ccg = M_PI * gamma_ * pow(cx * cy, gamma_ - 1);
         double axy = sa * sx * cy - ca * cx * sy;
         q[0] =  ccg * axy * sa;
         q[1] = -ccg * axy * ca;
         break;
      }
   }
}

void qFunc(const Vector &x, Vector &q)
{
   Vector qperp;
   qPerpFunc(x,qperp);
   qParaFunc(x,q);
   q += qperp;
}

void BFunc(const Vector &x, Vector &B)
{
   B.SetSize(2);

   switch (prob_)
   {
      case 1:
      {
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);

         if ( fabs(sx) < 1e-6 && fabs(sy) < 1e-6 )
         {
            B[0] = 0.0;
            B[1] = 0.0;
         }
         else if ( fabs(cx) < 1e-6 && fabs(cy) < 1e-6 )
         {
            B[0] =  M_SQRT2 * x[1];
            B[1] = -M_SQRT2 * x[0];
         }
         else
         {
            B[0] =  cx * sy;
            B[1] = -sx * cy;
            double nrm = B.Norml2();
            if ( nrm > 0.0 ) { B /= nrm; }
         }
      }
      break;
      case 2:
      {
         B[0] = cos(alpha_);
         B[1] = sin(alpha_);
      }
      break;
   }
}

long int factorial(unsigned int n)
{
   long int fact = 1;
   for (unsigned int i=2; i<=n; i++)
   {
      fact *= i;
   }
   return fact;
}

// Returns the Gamma(n) function for a positive integer n
long int gamma(unsigned int n)
{
   assert(n > 0);
   return factorial(n-1);
}

// Returns Gamma(n+1/2) for a positive integer n
double gamma1_2(unsigned int n)
{
   return sqrt(M_PI) * factorial(2*n) / (pow(4, n) * factorial(n));
}

double TNorm()
{
   switch (prob_)
   {
      case 1:
         return 0.5;
      case 2:
         return (gamma1_2((unsigned int)gamma_) /
                 gamma((unsigned int)gamma_+1)) / sqrt(M_PI);
   }
}

double qPerpNorm()
{
   switch (prob_)
   {
      case 1:
         return M_PI * M_SQRT1_2 * chi_ratio_;
      case 2:
         return sqrt(M_PI * gamma_) * M_SQRT1_2 *
                sqrt(gamma1_2((unsigned int)gamma_-1) *
                     gamma1_2((unsigned int)gamma_)) /
                sqrt(gamma((unsigned int)gamma_) * gamma((unsigned int)gamma_+1));
   }
}

double qParaNorm()
{
   switch (prob_)
   {
      case 1:
         return 0.0;
      case 2:
         return chi_ratio_ * qPerpNorm();
   }
}

double qNorm()
{
   double para = qParaNorm();
   double perp = qPerpNorm();
   return sqrt(para * para + perp * perp);
}

class AnisoConductionCoefficient : public MatrixCoefficient
{
public:
   AnisoConductionCoefficient(ParGridFunction & b)
      : MatrixCoefficient(2), b_(&b), bCoef_(NULL), B_(2) {}

   AnisoConductionCoefficient(VectorCoefficient & bCoef)
      : MatrixCoefficient(2), b_(NULL), bCoef_(&bCoef), B_(2) {}

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      if (b_)
      {
         b_->GetVectorValue(T.ElementNo, ip, B_);
      }
      else
      {
         bCoef_->Eval(B_, T, ip);
      }

      K.SetSize(2);

      double B2 = B_ * B_;

      K(0,0) = 1.0 + (chi_ratio_ - 1.0) * B_[0] * B_[0] / B2;
      K(0,1) = (chi_ratio_ - 1.0) * B_[0] * B_[1] / B2;
      K(1,0) = (chi_ratio_ - 1.0) * B_[0] * B_[1] / B2;
      K(1,1) = 1.0 + (chi_ratio_ - 1.0) * B_[1] * B_[1] / B2;
   }


private:
   ParGridFunction * b_;
   VectorCoefficient * bCoef_;
   Vector B_;
};

class ParaCoefficient : public MatrixCoefficient
{
public:
   ParaCoefficient(void (*unit_b)(const Vector &, Vector &))
      : MatrixCoefficient(2), unit_b_(unit_b), B_(2) , x_(2) {}

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      (*unit_b_)(x_, B_);

      K.SetSize(2);

      K(0,0) = B_[0] * B_[0];
      K(0,1) = B_[0] * B_[1];
      K(1,0) = B_[1] * B_[0];
      K(1,1) = B_[1] * B_[1];
   }

private:
   void (*unit_b_)(const Vector &, Vector &);
   Vector B_;
   Vector x_;
};

class PerpCoefficient : public MatrixCoefficient
{
public:
   PerpCoefficient(void (*unit_b)(const Vector &, Vector &))
      : MatrixCoefficient(2), unit_b_(unit_b), B_(2) , x_(2) {}

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      (*unit_b_)(x_, B_);

      K.SetSize(2);

      K(0,0) = 1.0 - B_[0] * B_[0];
      K(0,1) = 0.0 - B_[0] * B_[1];
      K(1,0) = 0.0 - B_[1] * B_[0];
      K(1,1) = 1.0 - B_[1] * B_[1];
   }

private:
   void (*unit_b_)(const Vector &, Vector &);
   Vector B_;
   Vector x_;
};

void H1AnisoDiffusionSolve(int myid, const ParMesh &pmesh,
                           const IntegrationRule &ir,
                           Coefficient &QCoef,
                           MatrixCoefficient &ChiCoef,
                           // const ParGridFunction &unit_b,
                           int order,
                           ParGridFunction &t,
                           ParGridFunction &q,
                           ParGridFunction &qPara,
                           ParGridFunction &qPerp);

void H1L2AnisoDiffusionSolve(int myid, const ParMesh &pmesh,
                             ParFiniteElementSpace * fespace_l2,
                             const IntegrationRule &ir,
                             Coefficient &QCoef,
                             MatrixCoefficient &ChiCoef,
                             // const ParGridFunction &unit_b,
                             int order,
                             ParGridFunction &t,
                             ParGridFunction &q,
                             ParGridFunction &qPara,
                             ParGridFunction &qPerp);

void HDivAnisoDiffusionSolve(int myid, const ParMesh &pmesh,
                             const IntegrationRule &ir,
                             Coefficient &QCoef,
                             MatrixCoefficient &ChiCoef,
                             // const ParGridFunction &unit_b,
                             int order,
                             ParGridFunction &t,
                             ParGridFunction &q,
                             ParGridFunction &qPara,
                             ParGridFunction &qPerp);

void shiftUnitSquare(const Vector &x, Vector &p)
{
   p[0] = x[0] - 0.5;
   p[1] = x[1] - 0.5;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int p_min = 2;
   int p_max = 4;
   int order = 1;
   int irOrder = -1;
   int el_type = Element::QUADRILATERAL;
   int B_type = 2;
   int sol_type = 0;
   bool zero_start = true;
   bool static_cond = false;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&p_min, "-p0", "--power-min",
                  "Number of elements in x and y directions.  "
                  "Total number of elements is 2^p.");
   args.AddOption(&p_max, "-p", "--power",
                  "Number of elements in x and y directions.  "
                  "Total number of elements is 2^p.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&irOrder, "-iro", "--int-rule-order",
                  "Integration Rule Order.");
   args.AddOption(&prob_, "-prob", "--problem",
                  "Specify problem type: 1 - Square, 2 - Ellipse.");
   args.AddOption(&chi_ratio_, "-chi", "--chi-ratio",
                  "Ratio of chi_parallel/chi_perp.");
   args.AddOption(&alpha_, "-alpha", "--constant-angle",
                  "Angle for constant B field (in degrees)");
   args.AddOption(&gamma_, "-gamma", "--exponent",
                  "Exponent used in problem 2");
   args.AddOption(&el_type, "-e", "--element-type",
                  "Element type: 2-Triangle, 3-Quadrilateral.");
   args.AddOption(&B_type, "-b", "--b-field-type",
                  "B field type: 0 - H1, 1 - HCurl, 2-HDiv.");
   args.AddOption(&sol_type, "-s", "--solver-type",
                  "Solver type: 0 - H1, 1 - H1/L2 Hybrid, 2-HDiv.");
   args.AddOption(&zero_start, "-z", "--zero-start", "-no-z",
                  "--no-zero-start",
                  "Initial guess of zero or exact solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   if (irOrder < 0)
   {
      irOrder = std::max(4, 2 * order - 2);
   }

   if (isnan(alpha_))
   {
      alpha_ = 0.0;
   }
   else
   {
      alpha_ *= M_PI / 180.0;
   }

   ostringstream oss;
   oss << "aniso_H1_o" << order << "b" << B_type
       << "e" << (int)floor(log10(chi_ratio_)) << ".dat";
   ofstream ofs;

   if ( myid == 0 )
   {
      ofs.open(oss.str().c_str());
   }

   socketstream src_sock;
   socketstream b_sock;
   int posx = 360, posy = 0;
   int dposx = 20, dposy = 20;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      src_sock.open(vishost, visport);
      src_sock.precision(8);
      b_sock.open(vishost, visport);
      b_sock.precision(8);
   }

   for (int p=p_min; p<=p_max; p++)
   {
      int n = pow(2, p);

      // 3. Construct a (serial) mesh of the given size on all processors.  We
      //    can handle triangular and quadrilateral surface meshes with the
      //    same code.
      Mesh *mesh = new Mesh(n, n, (Element::Type)el_type, 1);
      int dim = mesh->Dimension();

      mesh->Transform(shiftUnitSquare);

      // 4. This step is no longer needed

      // 5. Define a parallel mesh by a partitioning of the serial mesh. Once the
      //    parallel mesh is defined, the serial mesh can be deleted.
      ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      delete mesh;

      // 6. Define a parallel finite element space on the parallel mesh. Here we
      //    use continuous Lagrange finite elements of the specified order. If
      //    order < 1, we instead use an isoparametric/isogeometric space.
      FiniteElementCollection *fec_T;
      if (order > 0)
      {
         fec_T = new H1_FECollection(order, dim);
      }
      else if (pmesh->GetNodes())
      {
         fec_T = pmesh->GetNodes()->OwnFEC();
         if (myid == 0)
         {
            cout << "Using isoparametric FEs: " << fec_T->Name() << endl;
         }
      }
      else
      {
         fec_T = new H1_FECollection(order = 1, dim);
      }
      ParFiniteElementSpace *fespace_T =
         new ParFiniteElementSpace(pmesh, fec_T);
      HYPRE_Int size_T = fespace_T->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Number of finite element unknowns (T): " << size_T << endl;
      }

      FiniteElementCollection * fec_q = new RT_FECollection(order - 1, dim);
      ParFiniteElementSpace * fespace_q =
         new ParFiniteElementSpace(pmesh, fec_q);
      HYPRE_Int size_q = fespace_q->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Number of finite element unknowns (q): " << size_q << endl;
      }

      FiniteElementCollection * fec_l2 = new L2_FECollection(order - 1, dim);
      ParFiniteElementSpace * fespace_l2 =
         new ParFiniteElementSpace(pmesh, fec_l2);
      HYPRE_Int size_l2 = fespace_l2->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Number of finite element unknowns (L2): " << size_l2 << endl;
      }

      FiniteElementCollection *fec_B = NULL;
      switch (B_type)
      {
         case 0:
            if (myid == 0)
            {
               cout << "Using vector H1 for B" << endl;
            }

            fec_B = new H1_FECollection(order, dim);
            break;
         case 1:
            if (myid == 0)
            {
               cout << "Using vector H(Curl) for B" << endl;
            }
            fec_B = new ND_FECollection(order, dim);
            break;
         case 2:
            if (myid == 0)
            {
               cout << "Using vector H(Div) for B" << endl;
            }
            fec_B = new RT_FECollection(order - 1, dim);
            break;
      }
      ParFiniteElementSpace *fespace_B = new ParFiniteElementSpace(pmesh, fec_B,
                                                                   (B_type==0)?2:1);
      HYPRE_Int size_B = fespace_B->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Number of finite element unknowns (B): " << size_B << endl;
      }


      // 7. Determine the list of true (i.e. parallel conforming) essential
      //    boundary dofs. In this example, the boundary conditions are defined
      //    by marking all the boundary attributes from the mesh as essential
      //    (Dirichlet) and converting them to a list of true dofs.

      // 8. Set up the parallel linear form b(.) which corresponds to the
      //    right-hand side of the FEM linear system, which in this case is
      //    (2pi^2 sin(pi x) sin(pi y),phi_i) where phi_i are the basis
      //    functions in fespace.
      ConstantCoefficient zeroCoef(0.0);
      Vector zeroVec(2); zeroVec = 0.0;
      VectorConstantCoefficient zeroVecCoef(zeroVec);

      FunctionCoefficient QCoef(QFunc);
      ParGridFunction Q(fespace_T);
      Q.ProjectCoefficient(QCoef);

      // 9. Define the solution vector x as a parallel finite element grid function
      //    corresponding to fespace. Initialize x with initial guess of zero,
      //    which satisfies the boundary conditions.
      ParGridFunction t(fespace_T);
      FunctionCoefficient TExact(TFunc);
      if (zero_start)
      {
         t = 0.0;
      }
      else
      {
         t.ProjectCoefficient(TExact);
      }

      double err0 = t.ComputeL2Error(TExact);
      if (myid == 0)
      {
         cout << "Initial Error: " << err0 << endl;
      }

      ParGridFunction q(fespace_q);
      ParGridFunction qPara(fespace_q);
      ParGridFunction qPerp(fespace_q);
      VectorFunctionCoefficient qExact(2, qFunc);
      VectorFunctionCoefficient qParaExact(2, qParaFunc);
      VectorFunctionCoefficient qPerpExact(2, qPerpFunc);

      ParGridFunction unit_b(fespace_B);
      VectorFunctionCoefficient BExact(2, BFunc);
      unit_b.ProjectCoefficient(BExact);

      double errB0 = unit_b.ComputeL2Error(BExact);
      if (myid == 0)
      {
         cout << "Initial Error in B: " << errB0 << endl;
      }

      if (visualization)
      {
         // char vishost[] = "localhost";
         //  int  visport   = 19916;
         // socketstream src_sock(vishost, visport);
         src_sock << "parallel " << num_procs << " " << myid << "\n";
         src_sock << "solution\n" << *pmesh << Q << flush;
         if ( p == 2 )
         {
            src_sock << "window_geometry 0 0 350 350\n";
            src_sock << "keys maaAc\n";
            src_sock << "window_title '"
                     << "Heat Source'\n" << flush;
         }

         // socketstream b_sock(vishost, visport);
         b_sock << "parallel " << num_procs << " " << myid << "\n";
         b_sock << "solution\n" << *pmesh << unit_b << flush;
         if ( p == 2 )
         {
            b_sock << "window_geometry 0 395 350 350\n";
            b_sock << "keys maaAvvv\n";
            b_sock << "window_title '"
                   << "Field Alignment'\n" << flush;
         }
      }

      // 10. Set up the parallel bilinear form a(.,.) on the finite element space
      //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
      //     domain integrator.

      Geometry::Type geom = (el_type==2)?Geometry::TRIANGLE:Geometry::SQUARE;
      const IntegrationRule * ir = &IntRules.Get(geom, irOrder);

      if (myid == 0)
      {
         cout << "Integration Rule Order and Number of Points: "
              << irOrder << "\t" << ir->GetNPoints() << endl;
      }

      // AnisoConductionCoefficient ChiCoef(unit_b);
      AnisoConductionCoefficient ChiCoef(BExact);

      switch (sol_type)
      {
         case 0:
            H1AnisoDiffusionSolve(myid, *pmesh, *ir, QCoef, ChiCoef,
                                  //unit_b,
                                  order,
                                  t, q, qPara, qPerp);
            break;
         case 1:
            H1L2AnisoDiffusionSolve(myid, *pmesh, fespace_l2, *ir, QCoef, ChiCoef,
                                    //unit_b,
                                    order,
                                    t, q, qPara, qPerp);
            break;
         case 2:
            HDivAnisoDiffusionSolve(myid, *pmesh, *ir, QCoef, ChiCoef,
                                    // unit_b,
                                    order,
                                    t, q, qPara, qPerp);
            break;
      }

      cout << "T norm: " << t.ComputeL2Error(zeroCoef) << " " << TNorm() << endl;
      cout << "q norm: " << q.ComputeL2Error(zeroVecCoef) << " " <<
           qNorm() << endl;
      cout << "qPerp norm: " << qPerp.ComputeL2Error(zeroVecCoef) << " " <<
           qPerpNorm() << endl;
      cout << "qPara norm: " << qPara.ComputeL2Error(zeroVecCoef) << " " <<
           qParaNorm() << endl;

      // 13b. Extract and report the solution value at (0.5, 0.5) which
      //      conveniently is the location of the maximum.
      double t_max = t.Normlinf();
      double err_T = t.ComputeL2Error(TExact);
      double err_pt = fabs(1.0/t_max - 1.0);
      double err_q = q.ComputeL2Error(qExact);
      double err_q_para = qPara.ComputeL2Error(qParaExact);
      double err_q_perp = qPerp.ComputeL2Error(qPerpExact);
      if (myid == 0)
      {
         cout << "Relative L2 Error of T: " << err_T / TNorm() << endl;
         cout << "Relative L2 Error of q: " << err_q / qNorm() << endl;
         if (prob_ == 1)
         {
            cout << "L2 Error of q Para: " << err_q_para << endl;
         }
         else
         {
            cout << "Relative L2 Error of q Para: " << err_q_para / qParaNorm() << endl;
         }
         cout << "Relative L2 Error of q Perp: " << err_q_perp / qPerpNorm() << endl;
         cout << "Maximum Temperature: " << t_max << endl;
         cout << "| chi_eff - 1 | = " << err_pt << endl;
      }

      if ( myid == 0 )
      {
         ofs << 1.0/n << "\t" << "1/" << n << "\t" << fabs(1.0/t_max - 1.0)
             << "\t" << err_T / TNorm() << "\t" << err_q / qNorm();
         if (prob_ == 1)
         {
            ofs << "\t" << err_q_para;
         }
         else
         {
            ofs << "\t" << err_q_para / qParaNorm();
         }
         ofs << "\t" << err_q_perp / qPerpNorm()
             << endl;
      }

      // 14. Save the refined mesh and the solution in parallel. This output can
      //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
      {
         ostringstream mesh_name, sol_name;
         mesh_name << "mesh." << setfill('0') << setw(6) << myid;
         sol_name << "sol." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);

         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(8);
         t.Save(sol_ofs);
      }

      // 15. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << t;
         sol_sock << "window_geometry " << posx << " " << posy << " 350 350\n";
         sol_sock << "keys mmaaAc\n";
         sol_sock << "window_title '"
                  << "Chi Ratio 10^" << (int)floor(log10(chi_ratio_)) << ", "
                  << "Order " << order << ", "
                  << "Num Elems " << pow(2,p) << "^2, "
                  << "Error " << err_pt << "'\n" << flush;

         socketstream q_para_sock(vishost, visport);
         q_para_sock << "parallel " << num_procs << " " << myid << "\n";
         q_para_sock.precision(8);
         q_para_sock << "solution\n" << *pmesh << qPara;
         q_para_sock << "window_geometry " << posx << " " << posy+395 << " 350 350\n";
         q_para_sock << "keys mmaaAcvvv\n";
         q_para_sock << "window_title '"
                     << "q Para "
                     << "Chi Ratio 10^" << (int)floor(log10(chi_ratio_)) << ", "
                     << "Order " << order << ", "
                     << "Num Elems " << pow(2,p) << "^2, "
                     << "Error " << err_q_para << "'\n" << flush;

         socketstream q_perp_sock(vishost, visport);
         q_perp_sock << "parallel " << num_procs << " " << myid << "\n";
         q_perp_sock.precision(8);
         q_perp_sock << "solution\n" << *pmesh << qPerp;
         q_perp_sock << "window_geometry " << posx+360 << " " << posy+395 <<
                     " 350 350\n";
         q_perp_sock << "keys mmaaAcvvv\n";
         q_perp_sock << "window_title '"
                     << "q Perp "
                     << "Chi Ratio 10^" << (int)floor(log10(chi_ratio_)) << ", "
                     << "Order " << order << ", "
                     << "Num Elems " << pow(2,p) << "^2, "
                     << "Error " << err_q_perp << "'\n" << flush;

         posx += dposx;
         posy += dposy;
      }

      delete fespace_T;
      if (order > 0) { delete fec_T; }
      delete fespace_q;
      delete fec_q;
      delete fespace_B;
      delete fec_B;
      delete pmesh;
   }
   if ( myid == 0 )
   {
      ofs.close();
   }

   MPI_Finalize();

   return 0;
}

void H1AnisoDiffusionSolve(int myid, const ParMesh &pmesh,
                           const IntegrationRule &ir,
                           Coefficient &QCoef,
                           MatrixCoefficient &ChiCoef,
                           // const ParGridFunction &unit_b,
                           int order,
                           ParGridFunction &t,
                           ParGridFunction &q,
                           ParGridFunction &qPara,
                           ParGridFunction &qPerp)
{
   ParFiniteElementSpace * fespace_T = t.ParFESpace();
   ParFiniteElementSpace * fespace_q = q.ParFESpace();
   {
      // Solve for T

      Array<int> ess_tdof_list;
      if (pmesh.bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         fespace_T->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      BilinearFormIntegrator * diffInteg = new DiffusionIntegrator(ChiCoef);
      diffInteg->SetIntRule(&ir);

      ParBilinearForm *a = new ParBilinearForm(fespace_T);
      a->AddDomainIntegrator(diffInteg);
      a->Assemble();

      ParLinearForm *rhs = new ParLinearForm(fespace_T);
      rhs->AddDomainIntegrator(new DomainLFIntegrator(QCoef));
      rhs->Assemble();

      HypreParMatrix A;
      Vector RHS, T;
      a->FormLinearSystem(ess_tdof_list, t, *rhs, A, T, RHS, true);

      if (myid == 0)
      {
         cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
      }

      // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      //     preconditioner from hypre.
      HypreSolver *amg = new HypreBoomerAMG(A);
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(10000);
      pcg->SetPrintLevel(2);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(RHS, T);

      // 13. Recover the parallel grid function corresponding to X. This is the
      //     local finite element solution on each processor.
      a->RecoverFEMSolution(T, *rhs, t);

      // 16. Free the used memory.
      delete pcg;
      delete amg;
      delete a;
      delete rhs;
   }
   {
      // Solve for components of q
      ParaCoefficient ParaCoef(BFunc);
      PerpCoefficient PerpCoef(BFunc);
      MatrixSumCoefficient FullCoef(ParaCoef, PerpCoef, chi_ratio_);

      ParBilinearForm m2(fespace_q);
      m2.AddDomainIntegrator(new VectorFEMassIntegrator());
      m2.Assemble();

      HypreParMatrix M2;
      Array<int> ess_tdof(0);
      m2.FormSystemMatrix(ess_tdof, M2);
      HyprePCG M2Inv(M2);
      M2Inv.SetTol(1e-12);
      M2Inv.SetMaxIter(200);
      M2Inv.SetPrintLevel(0);
      HypreDiagScale M2Diag(M2);
      M2Inv.SetPreconditioner(M2Diag);

      ParMixedBilinearForm gFull(fespace_T, fespace_q);
      gFull.AddDomainIntegrator(new MixedVectorGradientIntegrator(FullCoef));
      gFull.Assemble();

      ParMixedBilinearForm gPara(fespace_T, fespace_q);
      gPara.AddDomainIntegrator(new MixedVectorGradientIntegrator(ParaCoef));
      gPara.Assemble();

      ParMixedBilinearForm gPerp(fespace_T, fespace_q);
      gPerp.AddDomainIntegrator(new MixedVectorGradientIntegrator(PerpCoef));
      gPerp.Assemble();

      gFull.Mult(t, q);
      gPara.Mult(t, qPara);
      gPerp.Mult(t, qPerp);

      int q_size = fespace_q->GetTrueVSize();
      Vector RHS(q_size);
      Vector X(q_size); X = 0.0;

      q.ParallelAssemble(RHS);
      M2Inv.Mult(RHS, X);
      q.Distribute(X);
      q *= -1.0;

      qPara.ParallelAssemble(RHS);
      M2Inv.Mult(RHS, X);
      qPara.Distribute(X);
      qPara *= -chi_ratio_;

      qPerp.ParallelAssemble(RHS);
      M2Inv.Mult(RHS, X);
      qPerp.Distribute(X);
      qPerp *= -1.0;
   }
}

void H1L2AnisoDiffusionSolve(int myid, const ParMesh &pmesh,
                             ParFiniteElementSpace * fespace_l2,
                             const IntegrationRule &ir,
                             Coefficient &QCoef,
                             MatrixCoefficient &ChiCoef,
                             // const ParGridFunction &unit_b,
                             int order,
                             ParGridFunction &t,
                             ParGridFunction &q,
                             ParGridFunction &qPara,
                             ParGridFunction &qPerp)
{
   ParFiniteElementSpace * fespace_T = t.ParFESpace();
   ParFiniteElementSpace * fespace_q = q.ParFESpace();

   q = 0.0;
   {
      // Solve for T
      Array<int> ess_bdr;
      Array<int> ess_tdof_list;

      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         fespace_T->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      BilinearFormIntegrator * diffInteg = new DiffusionIntegrator();
      diffInteg->SetIntRule(&ir);

      ParBilinearForm a(fespace_T);
      a.AddDomainIntegrator(diffInteg);
      a.Assemble();

      // VectorGridFunctionCoefficient unitBCoef(const_cast<ParGridFunction*>(&unit_b));
      VectorFunctionCoefficient unitBCoef(2, BFunc);
      ScalarVectorProductCoefficient scaledBCoef(1.0 - chi_ratio_, unitBCoef);
      ScalarVectorProductCoefficient negUnitBCoef(-1.0, unitBCoef);

      ParMixedBilinearForm wd(fespace_T, fespace_l2);
      wd.AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(scaledBCoef));
      wd.Assemble();
      wd.EliminateTestDofs(ess_bdr);

      ParMixedBilinearForm dd(fespace_l2, fespace_T);
      dd.AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(negUnitBCoef));
      dd.Assemble();

      Vector X0(1), RHS3(1);
      dd.EliminateTrialDofs(ess_bdr, X0, RHS3);

      ParBilinearForm m3(fespace_l2);
      m3.AddDomainIntegrator(new MassIntegrator());
      m3.Assemble();

      ParLinearForm rhs(fespace_T);
      rhs.AddDomainIntegrator(new DomainLFIntegrator(QCoef));
      rhs.Assemble();

      HypreParMatrix A;
      Vector RHS, T;
      a.FormLinearSystem(ess_tdof_list, t, rhs, A, T, RHS, true);

      if (myid == 0)
      {
         cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
      }

      // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      //     preconditioner from hypre.
      HypreSolver *amg = new HypreBoomerAMG(A);
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(10000);
      pcg->SetPrintLevel(2);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(RHS, T);

      // 13. Recover the parallel grid function corresponding to X. This is the
      //     local finite element solution on each processor.
      a.RecoverFEMSolution(T, rhs, t);

      // 16. Free the used memory.
      delete pcg;
      delete amg;
      // delete a;
      // delete rhs;
   }
   {
      // Solve for components of q
      ParaCoefficient ParaCoef(BFunc);
      PerpCoefficient PerpCoef(BFunc);

      ParBilinearForm m2(fespace_q);
      m2.AddDomainIntegrator(new VectorFEMassIntegrator());
      m2.Assemble();

      HypreParMatrix M2;
      Array<int> ess_tdof(0);
      m2.FormSystemMatrix(ess_tdof, M2);
      HyprePCG M2Inv(M2);
      M2Inv.SetTol(1e-12);
      M2Inv.SetMaxIter(200);
      M2Inv.SetPrintLevel(0);
      HypreDiagScale M2Diag(M2);
      M2Inv.SetPreconditioner(M2Diag);

      ParMixedBilinearForm gPara(fespace_T, fespace_q);
      gPara.AddDomainIntegrator(new MixedVectorGradientIntegrator(ParaCoef));
      gPara.Assemble();

      ParMixedBilinearForm gPerp(fespace_T, fespace_q);
      gPerp.AddDomainIntegrator(new MixedVectorGradientIntegrator(PerpCoef));
      gPerp.Assemble();

      gPara.Mult(t, qPara);
      gPerp.Mult(t, qPerp);

      int q_size = fespace_q->GetTrueVSize();
      Vector RHS(q_size);
      Vector X(q_size);

      qPara.ParallelAssemble(RHS);
      M2Inv.Mult(RHS, X);
      qPara.Distribute(X);
      qPara *= -chi_ratio_;

      qPerp.ParallelAssemble(RHS);
      M2Inv.Mult(RHS, X);
      qPerp.Distribute(X);
      qPerp *= -1.0;
   }
}

class VectorFEDivLFIntegrator : public LinearFormIntegrator
{
public:
   VectorFEDivLFIntegrator(Coefficient & q) : Q(&q) {}

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect)
   {
      int dof = el.GetDof();

      divshape.SetSize(dof);

      elvect.SetSize(dof);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         // int intorder = 2*el.GetOrder() - 1; // ok for O(h^{k+1}) conv. in L2
         int intorder = 2*el.GetOrder();
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         Tr.SetIntPoint (&ip);
         el.CalcPhysDivShape(Tr, divshape);

         elvect.Add(ip.weight * Tr.Weight() * Q->Eval(Tr, ip), divshape);
      }

   }

private:
   Coefficient * Q;
   Vector divshape;
};

void HDivAnisoDiffusionSolve(int myid, const ParMesh &pmesh,
                             const IntegrationRule &ir,
                             Coefficient &QCoef,
                             MatrixCoefficient &ChiCoef,
                             // const ParGridFunction &unit_b,
                             int order,
                             ParGridFunction &t,
                             ParGridFunction &q,
                             ParGridFunction &qPara,
                             ParGridFunction &qPerp)
{
   ParFiniteElementSpace * fespace_T = t.ParFESpace();
   ParFiniteElementSpace * fespace_q = qPara.ParFESpace();

   // ParGridFunction q(fespace_q);
   q = 0.0;

   {
      // Solve for q
      int dim = pmesh.Dimension();

      Array<int> ess_tdof_list;
      if (pmesh.bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         fespace_q->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // BilinearFormIntegrator * divdivInteg = new DivDivIntegrator();
      // divdivInteg->SetIntRule(&ir);

      double dt = 0.1;
      ConstantCoefficient dtCoef(dt);
      // ConstantCoefficient dtInvCoef(1.0 / dt);
      // double dt = 100.0 / chi_ratio_;
      // InverseMatrixCoefficient ChiInvCoef(ChiCoef);
      //ScalarMatrixProductCoefficient dtChiInvCoef(1.0 / dt, ChiInvCoef);
      //  ConstantCoefficient epsCoef(1e-6);

      ParBilinearForm a(fespace_q);
      //a.AddDomainIntegrator(new VectorFEMassIntegrator(dtChiInvCoef));
      a.AddDomainIntegrator(new VectorFEMassIntegrator());
      // a.AddDomainIntegrator(divdivInteg);
      a.AddDomainIntegrator(new DivDivIntegrator(dtCoef));
      a.Assemble();

      ParBilinearForm s(fespace_q);
      // s.AddDomainIntegrator(divdivInteg);
      s.AddDomainIntegrator(new DivDivIntegrator());
      s.Assemble();

      ParLinearForm Qs(fespace_q);
      Qs.AddDomainIntegrator(new VectorFEDivLFIntegrator(QCoef));
      Qs.Assemble();

      ParLinearForm rhs(fespace_q);

      ParGridFunction dqdt(fespace_q);
      dqdt = 0.0;

      HypreParMatrix A;
      Vector RHS, X;

      double tol  = 1e-5;

      double nrm = sqrt(InnerProduct(Qs.ParFESpace()->GetComm(), Qs, Qs));
      while (true)
      {
         rhs.Set(1.0, Qs);

         s.AddMult(q, rhs, -1.0);

         double nrm_rhs = sqrt(InnerProduct(rhs.ParFESpace()->GetComm(),
                                            rhs, rhs));
         if ( nrm_rhs < nrm * tol ) { break; }
         cout << "Correction: " << nrm_rhs / nrm << endl;

         a.FormLinearSystem(ess_tdof_list, dqdt, rhs, A, X, RHS, true);

         //if (myid == 0)
         //{
         //  cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
         //}

         // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
         //     preconditioner from hypre.
         HypreSolver *ads = (dim == 2) ?
                            (HypreSolver*)(new HypreAMS(A, fespace_q)) :
                            (HypreSolver*)(new HypreADS(A, fespace_q));
         HyprePCG *pcg = new HyprePCG(A);
         pcg->SetTol(1e-12);
         pcg->SetMaxIter(10000);
         pcg->SetPrintLevel(0);
         pcg->SetPreconditioner(*ads);
         pcg->Mult(RHS, X);

         // 13. Recover the parallel grid function corresponding to X. This is the
         //     local finite element solution on each processor.
         a.RecoverFEMSolution(X, rhs, dqdt);

         q.Add(dt, dqdt);

         // 16. Free the used memory.
         delete pcg;
         delete ads;
      }
      cout << "Done with pseudo-time-stepping" << endl;
      // delete a;
      // delete rhs;
   }
   cout << "Done with q calc" << endl;
   {
      // Solve for components of q
      ParaCoefficient ParaCoef(BFunc);
      PerpCoefficient PerpCoef(BFunc);

      ParBilinearForm m2(fespace_q);
      m2.AddDomainIntegrator(new VectorFEMassIntegrator());
      m2.Assemble();

      HypreParMatrix M2;
      Array<int> ess_tdof(0);
      m2.FormSystemMatrix(ess_tdof, M2);
      HyprePCG M2Inv(M2);
      M2Inv.SetTol(1e-12);
      M2Inv.SetMaxIter(200);
      M2Inv.SetPrintLevel(0);
      HypreDiagScale M2Diag(M2);
      M2Inv.SetPreconditioner(M2Diag);

      ParBilinearForm mPara(fespace_q);
      mPara.AddDomainIntegrator(new VectorFEMassIntegrator(ParaCoef));
      mPara.Assemble();

      ParBilinearForm mPerp(fespace_q);
      mPerp.AddDomainIntegrator(new VectorFEMassIntegrator(PerpCoef));
      mPerp.Assemble();

      mPara.Mult(q, qPara);
      mPerp.Mult(q, qPerp);

      int q_size = fespace_q->GetTrueVSize();
      Vector RHS(q_size);
      Vector X(q_size);

      qPara.ParallelAssemble(RHS);
      M2Inv.Mult(RHS, X);
      qPara.Distribute(X);
      qPara *= chi_ratio_;

      qPerp.ParallelAssemble(RHS);
      M2Inv.Mult(RHS, X);
      qPerp.Distribute(X);
   }
   cout << "Leaving function call" << endl;
}
