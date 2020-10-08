#include <string>
#include <fstream>
#include <iostream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;

#define DBG(...) { printf("\033[33m");printf(__VA_ARGS__);printf("\033[m\n");}

namespace mfem
{

namespace xfl
{

class TestFunction;
static ConstantCoefficient one(1.0);

/**
 * @brief The Problem struct
 */
struct Problem
{
   mfem::LinearForm *b;
   mfem::BilinearForm *a;
   Problem(mfem::BilinearForm *a, mfem::LinearForm *b): b(b), a(a) { }
};

/** ****************************************************************************
 * @brief The Integral classes
 ******************************************************************************/
struct Form { };

struct LinearForm
{
   mfem::LinearForm *b;

   LinearForm(FiniteElementSpace *fes): b(new mfem::LinearForm(fes)) {}

   LinearForm &operator *(Form dx) { return *this;}
};

struct ScalarForm
{
   FiniteElementSpace *fes;
   mfem::ConstantCoefficient cst;

   ScalarForm(double cst, FiniteElementSpace *fes): fes(fes), cst(cst) {}

   LinearForm operator*(Form dx)
   {
      assert(fes);
      LinearForm linear_form(fes);
      linear_form.b->AddDomainIntegrator(new DomainLFIntegrator(cst));
      return linear_form;
   }
};

struct BilinearForm
{
   mfem::BilinearForm *a;

   BilinearForm(FiniteElementSpace *fes): a(new mfem::BilinearForm(fes)) {}

   Problem operator ==(LinearForm &li) { return Problem(a, li.b); }

   // BilinearForm * dx
   BilinearForm &operator *(Form dx) { return *this;}

   // BilinearForm + BilinearForm
   BilinearForm &operator +(BilinearForm rhs)
   {
      Array<BilinearFormIntegrator*> *dbfi = rhs.a->GetDBFI();
      assert(dbfi->operator [](0));
      BilinearFormIntegrator *bfi = dbfi->operator [](0);
      a->AddDomainIntegrator(bfi);
      return *this;
   }

   BilinearForm &operator -(BilinearForm rhs) { return *this + rhs; /*!*/ }
};

/** ****************************************************************************
 * @brief The Function class
 ******************************************************************************/
class Function : public GridFunction
{
public:
   FiniteElementSpace *fes;
   Function(FiniteElementSpace *f): GridFunction(f), fes(f) { }
};

/**
 * @brief The GradFunction class
 */
class GradFunction: public Function
{
public:
   GradFunction(FiniteElementSpace *fes): Function(fes) { }

   // ∇u * ∇v
   BilinearForm operator*(GradFunction &v)
   {
      //DBG("\033[32m[Diffusion]");
      BilinearForm bf(fes);
      bf.a->AddDomainIntegrator(new DiffusionIntegrator(xfl::one));
      return bf;
   }
};

/**
 * @brief The DivFunction class
 */
class DivFunction: public Function
{
public:
   DivFunction(FiniteElementSpace *fes): Function(fes) { }

   BilinearForm operator*(GradFunction &v)
   {
      //DBG("\033[32m[Div*Grad]");
      BilinearForm bf(fes);
      return bf;
   }

   BilinearForm operator*(Function &v)
   {
      //DBG("\033[32m[Div*u]");
      BilinearForm bf(fes);
      return bf;
   }
};

/**
 * @brief The TrialFunction class
 */
struct TrialFunction: public Function
{
   TrialFunction &u;
public:
   TrialFunction(FiniteElementSpace *fes): Function(fes), u(*this) { }

   GradFunction Grad() { return GradFunction(fes); }
   DivFunction Div() { return DivFunction(fes); }

   // u * v
   BilinearForm operator*(TestFunction &v)
   {
      BilinearForm bf(fes);
      //DBG("\033[32m[Mass]");
      bf.a->AddDomainIntegrator(new MassIntegrator(xfl::one));
      return bf;
   }
};

/**
 * @brief The TestFunction class
 */
class TestFunction: public Function
{
   TestFunction &v;
public:
   TestFunction(FiniteElementSpace *fes): Function(fes), v(*this) { }

   GradFunction Grad() { return GradFunction(fes); }
   DivFunction Div() { return DivFunction(fes); }

   // v * u
   BilinearForm operator*(TrialFunction &u) { DBG("v*u"); return u * v;}

   // f * v
   ScalarForm operator*(double alpha)
   {
      ScalarForm scalar_form(alpha, fes);
      //DBG("\033[32m[Scalar]");
      return scalar_form;
   }
};

/** ****************************************************************************
 * @brief The Constant class
 ******************************************************************************/
class Constant: public ConstantCoefficient
{
public:
   Constant(double constant): ConstantCoefficient(constant) { }
   // T can be a Trial or a Test function
   template <typename T> ScalarForm operator*(T &gf) { return gf * constant; }
   double operator *(Form dx) { return constant;}
};

/** ****************************************************************************
 * @brief The FunctionSpace class
 ******************************************************************************/
class FunctionSpace: public FiniteElementSpace { };

/**
 * @brief UnitSquareMesh
 * @param nx
 * @param ny
 * @return
 */
Mesh *UnitSquareMesh(int nx, int ny)
{
   Element::Type quad = Element::Type::QUADRILATERAL;
   const bool generate_edges = false, sfc_ordering = true;
   const double sx = 1.0, sy = 1.0;
   return new Mesh(nx, ny, quad, generate_edges, sx, sy, sfc_ordering);
   //return new Mesh("/Users/camier1/home/sawmill/master/data/star.mesh",1,1);
}

FiniteElementSpace *FunctionSpace(Mesh *mesh, std::string family, int order)
{
   const int dim = mesh->Dimension();
   MFEM_VERIFY(family == "P", "Unsupported FE!");
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   assert(fec);
   return new FiniteElementSpace(mesh, fec);
}

//Constant* Expression(string, int degree) { return new Constant(1.0); }
//int DirichletBC(FiniteElementSpace&, Constant *u, bool on_boundary) { return 0; }
Array<int> DirichletBC(FiniteElementSpace *fes)
{
   assert(fes);
   Array<int> ess_tdof_list;
   Mesh &mesh = *fes->GetMesh();
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   return ess_tdof_list;
}

/**
 * Grad, Div
 */
template<typename T> GradFunction grad(T w) { return w.Grad(); }
template<typename T> DivFunction div(T w) { return w.Div(); }

/**
 * Dot/Inner product
 */
//template<typename T, typename U> BilinearForm dot(T u, U v) { return u * v; }
template<typename G> BilinearForm dot(G u, G v) { return u * v; }

/**
 * Math namespace
 */
namespace math
{

Constant Pow(Function &gf, double exp)
{
   return Constant(gf.Vector::Normlp(exp));
}
//double Pow(double base, double exp) { return std::pow(base, exp); }

} // namespace math

} // namespace xfl

/**
 * @brief solve with boundary conditions
 */
int solve(xfl::Problem pb, xfl::Function &x, Array<int> ess_tdof_list)
{
   x.GridFunction::operator =(0.0);
   FiniteElementSpace *fes = x.FESpace();
   assert(fes);
   Vector B, X;
   OperatorPtr A;
   LinearForm &b = *pb.b;
   b.Assemble();
   BilinearForm &a = *pb.a;
   a.Assemble();
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
   a.RecoverFEMSolution(X, b, x);
   std::cout << "Size of linear system: " << A->Height() << std::endl;
   std::cout << "L1norm:  " << x.Norml1() << std::endl;
   std::cout << "L2norm:  " << x.Norml2() << std::endl;
   return 0;
}

void plot(xfl::Function &x)
{
   FiniteElementSpace *fes = x.FESpace(); assert(fes);
   Mesh *mesh = fes->GetMesh(); assert(mesh);
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << *mesh << x << flush;
}

void plot(Mesh *mesh)
{
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "mesh\n" << *mesh << flush;
}

template<typename... Args>
void print(const char *fmt, Args... args) { printf(fmt,args...); }

} // namespace mfem
