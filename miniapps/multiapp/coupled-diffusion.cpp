#include "mfem.hpp"
#include "multiapp.hpp"
#include <fstream>
using namespace mfem;
using namespace std;

struct CaseContext
{
   int ser_ref = 1;         // Serial mesh refinement
   int order = 3;           // Finite element order
   bool visualization = true;// Visualization on/off
   int grad_mode = 1;       // Gradient mode for the coupled operator - 0: finite difference,
                            //                                          1: back/forward propagation
   bool coupled = true;     // Coupled (true) vs. uncoupled (false) solves
   int nl_iter = 50;        // Maximum number of nonlinear iterations
   int lin_iter = 2000;     // Maximum number of linear iterations

#if defined(MFEM_USE_DOUBLE)
   real_t tol_nsolve = 1e-4;
   real_t tol_lsolve = 1e-6;
#elif defined(MFEM_USE_SINGLE)
   real_t tol_nsolve = 1e-3;
   real_t tol_lsolve = 1e-3;
#else
#error "Only single and double precision are supported!"
   real_t tol_nsolve = 0;
   real_t tol_lsolve = 0;
#endif
} ctx;

void SetSolverParameters(IterativeSolver *solver, real_t rtol, real_t atol , int max_it,
                         int print_level, bool iterative_mode);


/// A functional diffusion coefficient (i.e., k(T))
class FunctionalCoefficient : public Coefficient
{
public:
   enum Mode { FUNC = 0, GRAD = 1};

protected:
   ParGridFunction *T_gf = nullptr;
   real_t kref = 1.0;
   real_t a0 = 0.0, a1 = 0.0, a2 = 0.0;
   int findex = 0;
   Mode mode = Mode::FUNC; // otherwise, grad

public:
   FunctionalCoefficient(ParGridFunction *T_gf, real_t kref): 
                         T_gf(T_gf), kref(kref) { }

   FunctionalCoefficient(ParGridFunction *T_gf, real_t kref, real_t a0):
                         T_gf(T_gf), kref(kref), a0(a0) { findex = 1; }

   FunctionalCoefficient(ParGridFunction *T_gf, real_t kref,
                         real_t a0, real_t a1, real_t a2): T_gf(T_gf),
                         kref(kref), a0(a0), a1(a1), a2(a2) { findex = 2; }

    real_t Exponential(real_t x, bool eval_f) const
    { 
      real_t f = kref*exp(a0*x);
      return (eval_f ? f : a0*f);
    }
    real_t Polynomial(real_t x, bool eval_f) const
    { 
      return (eval_f ? kref*(a0 + a1*x + a2*x*x) : kref*(a1 + 2*a2*x));
    }

    void SetMode(Mode mode) { this->mode = mode; }
    Mode GetMode() const { return mode; }
    
    void UpdateGridFunction(ParGridFunction *gf) { T_gf = gf; }

    real_t Eval(real_t x, bool eval_f) const
    {
        switch (findex)
        {
        case 1:
         return Exponential(x, eval_f);
        case 2:
         return Polynomial(x, eval_f);
        default:
         return kref;
        }
    }

    real_t Eval(ElementTransformation &Tr,
              const IntegrationPoint &ip) override
    {
        real_t T = T_gf ? T_gf->GetValue(Tr, ip) : 0.0;
        bool eval_f = (mode == Mode::FUNC);
        return Eval(T, eval_f);
    }
};


/// A coefficient defined by the product of grid functions, e.g. k(T) = prod_i x_i
class GridFunctionProductCoefficient : public Coefficient
{
protected:
   std::vector<ParGridFunction*> &x;

public:
   GridFunctionProductCoefficient(std::vector<ParGridFunction*> &x) : x(x) { }

   real_t Eval(ElementTransformation &Tr, const IntegrationPoint &ip) override
   {
      real_t prod = 1.0;
      for(size_t i = 0; i < x.size(); i++)
      {
         real_t val = x[i]->GetValue(Tr, ip);
         prod *= val;
      }
      return prod;
   }
};

class CoefficientIntegrator : public NonlinearFormIntegrator
{
protected:
   FunctionalCoefficient *func = nullptr;
   Vector shape;

public:
   CoefficientIntegrator(FunctionalCoefficient *func) : func(func) { }


   void SetCoefficient(FunctionalCoefficient *f) { func = f; }

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect)
   {
      int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;

      const IntegrationRule *ir = &el.GetNodes();

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         el.CalcShape(ip, shape);
         Tr.SetIntPoint(&ip);
         real_t x = elfun * shape; // Evaluate the function at the integration point
         real_t fval = func->Eval(x, true);
         for (int j = 0; j < dof; j++)
         {
            elvect(j) += fval * shape(j);
         }
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat)
   {
      int dof = el.GetDof();
      shape.SetSize(dof);
      elmat.SetSize(dof);
      elmat = 0.0;

      const IntegrationRule *ir = &el.GetNodes();

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         el.CalcShape(ip, shape);
         Tr.SetIntPoint(&ip);
         real_t x = elfun * shape; // Evaluate the function at the integration point
         real_t dfdx = func->Eval(x, false); // Evaluate the derivative of the function at the integration point
         for (int j = 0; j < dof; j++)
         {
            elmat(j,j) += dfdx * shape(j); // Diagonal contribution to the Jacobian
         }
      }
   }
};


class NonlinearDiffusionIntegrator : public NonlinearFormIntegrator
{
protected:
   Coefficient *k;
   Coefficient *dk;

   Vector u, vec, shape;
   DenseMatrix dshape, dshapedxt, adjJ;
public:
   NonlinearDiffusionIntegrator(Coefficient *kappa, Coefficient *dkappa) :
                                k(kappa), dk(dkappa) { }

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvec)
   {
      int dim = el.GetDim();
      int dof = el.GetDof();
      real_t w;

      elvec.SetSize(dof);
      elvec = 0.0;

      const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(el.GetGeomType(), 2*el.GetOrder());
      u.SetSize(dim);
      vec.SetSize(dim);
      dshape.SetSize(dof, dim);
      adjJ.SetSize(dim, dim);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         el.CalcDShape(ip, dshape);

         Tr.SetIntPoint(&ip);
         CalcAdjugate(Tr.Jacobian(), adjJ);
         w = ip.weight / Tr.Weight();

         dshape.MultTranspose(elfun, u);
         adjJ.MultTranspose(u, vec);
         if(k)
         {
            w *= k->Eval(Tr, ip);
         }

         vec *= w;
         adjJ.Mult(vec, u);
         dshape.AddMult(u, elvec);
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat)
   {
      int dim = el.GetDim();
      int dof = el.GetDof();
      real_t w, k0 = 0.0, dk0 = 0.0;

      elmat.SetSize(dof);
      elmat = 0.0;

      const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(el.GetGeomType(), 2*el.GetOrder());
      u.SetSize(dim);
      shape.SetSize(dof);
      vec.SetSize(dof);
      dshape.SetSize(dof, dim);
      dshapedxt.SetSize(dof, dim);

      // f = grad(psi) * k(u) * grad(T)
      // df/dT = grad(psi) ( k(u0) * grad(T) + k'(u0) * grad(u0) * T )
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         el.CalcShape(ip, shape);
         el.CalcDShape(ip, dshape);

         Tr.SetIntPoint(&ip);
         w = ip.weight / Tr.Weight();

         Mult(dshape, Tr.AdjugateJacobian(), dshapedxt);

         k0  = k ? k->Eval(Tr, ip) : 0.0;
         dk0 = dk ? dk->Eval(Tr, ip) : 0.0;

         if(k0 != 0.0) // grad(psi) * k(u0) * grad(T)
         {
            real_t kdT = w*k0;
            AddMult_a_AAt(kdT, dshapedxt, elmat);
         }

         if(dk0 != 0.0) // grad(psi) * (k'(T0) * grad(T0)) * T
         {
            dk0 = w*dk->Eval(Tr, ip);
            dshapedxt.MultTranspose(elfun, u); // grad(T0) in physical space
            u *= dk0; // k'(T0) * grad(T0)
            dshapedxt.Mult(u, vec); // grad(psi) * k'(T0) * grad(T0)
            AddMultVWt(vec, shape, elmat); // grad(psi) * k'(T0) * grad(T0) * T
         }
      }
   }
};

/// An application that takes an input field T, and computes an output field k(T)
// represented by the FunctionalCoefficient class.
class DiffusionCoefficient : public GraphNode
{
public:
   using Mode = FunctionalCoefficient::Mode;

protected:
   ParFiniteElementSpace &fes;
   mutable ParGridFunction T, k;
   mutable FunctionalCoefficient *kc;
   // mutable Vector tdof, kdof, dk_dof, dT_dof;
   mutable Mode mode = Mode::FUNC;

   mutable ParNonlinearForm Nform;
   mutable Operator *J = nullptr; // Jacobian for the nonlinear form

   CoefficientIntegrator *coeff_integrator = nullptr;

public:
   DiffusionCoefficient(ParFiniteElementSpace &fes) :
                        GraphNode(fes.GetTrueVSize()), fes(fes), T(&fes), k(&fes),
                        kc(new FunctionalCoefficient(&T, 1.0, 5.0e-2)),
                        Nform(&fes),
                        coeff_integrator(new CoefficientIntegrator(kc))
   {
      k = 0.0;
      T = 0.0;
      k.ProjectCoefficient(*kc);

      // Testing with the nonlinear form framework to compute k(T) and dk/dT
      Nform.AddDomainIntegrator(coeff_integrator); // Transfer ownership
      Nform.SetGradientType(Operator::Type::Hypre_ParCSR);
      Nform.Setup();

      SetInputOffsets(Array<int>({0, fes.GetTrueVSize()}));
      SetOutputOffsets(Array<int>({0, fes.GetTrueVSize()}));
   }

   void SetMode(Mode mode) { this->mode = mode; }

   FunctionalCoefficient* GetCoefficient() { return kc; }

   void SetCoefficient(FunctionalCoefficient *fc)
   {
      if(kc) delete kc;
      kc = fc;
      kc->SetMode(mode);
      kc->UpdateGridFunction(&T);
      coeff_integrator->SetCoefficient(kc);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      BlockVector xb(x.GetData(), InputOffsets());
      BlockVector yb(y.GetData(), OutputOffsets());

      MultiVector xmv(1), ymv(1);
      xmv.MakeRef(0, xb.GetBlock(0));
      ymv.MakeRef(0, yb.GetBlock(0));

      MultMV(xmv, ymv);
   }

   void MultMV(const MultiVector &x, MultiVector &y) const override
   {
      const Vector &tdof = x[0];
      Vector &kdof = y[0];

      Nform.Mult(tdof, kdof);
      if(exec_mode == GraphNode::GRADIENT_MODE)
      {
         J = &Nform.GetGradient(tdof); // Store jacobian for JVP
      }
      else
      {
         J = nullptr; // Clear the Jacobian if not in gradient mode
      }
   }

   void GradientMult(const MultiVector &x, const MultiVector &dx, MultiVector &dy) const override
   {
      const Vector &tdof = x[0];
      const Vector &xadj = dx[0];
      Vector &yadj = dy[0];

      if(J)
      {
         J->Mult(xadj, yadj);
      }
      else
      {
         J = &Nform.GetGradient(tdof); // Store jacobian for JVP
         J->Mult(xadj, yadj);
      }
   }

   ~DiffusionCoefficient() override
   {
      if(kc) delete kc;
   }
};

/// An application that takes n input fields x_i, and computes an output 
/// field prod(x) := y = prod_i x_i.
/// Also provides the derivative dy/dx_i = prod_{j!=i} x_j * dx_i/dx for i = 0,...,n-1.
class ProductGridFunctions : public GraphNode
{
protected:

   ParFiniteElementSpace &fes;
   mutable std::vector<ParGridFunction*> x_gf;
   mutable Vector dfdx;
   mutable ParGridFunction y_gf;
   mutable GridFunctionProductCoefficient prod_coeff;

public:
   ProductGridFunctions(ParFiniteElementSpace &fes, int n) :
                     //   GraphNode(fes.GetTrueVSize()),
                       GraphNode(fes.GetTrueVSize(), fes.GetTrueVSize() * n),
                       fes(fes), x_gf(n),
                       y_gf(&fes), prod_coeff(x_gf)
   {
      Array<int> offsets(n+1);
      offsets[0] = 0;
      for (int i = 0; i < n; i++)
      {
         x_gf[i]  = new ParGridFunction(&fes);
         *x_gf[i] = 0.0;
         offsets[i+1] = offsets[i] + fes.GetTrueVSize();
      }
      y_gf = 0.0;
      y_gf.ProjectCoefficient(prod_coeff);

      SetInputOffsets(offsets);
      SetOutputOffsets(Array<int>({0, fes.GetTrueVSize()}));
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      BlockVector xb(x.GetData(), InputOffsets());
      BlockVector yb(y.GetData(), OutputOffsets());

      MultiVector xmv(x_gf.size()), ymv(1);
      for (size_t i = 0; i < x_gf.size(); i++)
      {
         xmv.MakeRef(i, xb.GetBlock(i));
      }
      ymv.MakeRef(0, yb.GetBlock(0));
      MultMV(xmv, ymv);
   }

   void MultMV(const MultiVector &x, MultiVector &y) const override
   {
      for (size_t i = 0; i < x_gf.size(); i++)
      {
         const Vector &x_dof = x[i];
         x_gf[i]->SetFromTrueDofs(x_dof);
      }

      Field *out_field = OutputField(0);
      Vector &y_dof = y[0];
      y_gf.ProjectCoefficient(prod_coeff);
      y_gf.GetTrueDofs(y_dof);
   }

   void GradientMult(const MultiVector &x, const MultiVector &dx, MultiVector &dy) const override
   {
      // Jacobian vector product for y = prod_i x_i is:
      // dy/dx = sum_i (prod_{j!=i} x_j * dx_i/dx)
      for (size_t i = 0; i < x_gf.size(); i++)
      {
         const Vector &x_dof = x[i];
         x_gf[i]->SetFromTrueDofs(x_dof); // Set all x_i
      }

      Vector &jvp = dy[0];
      jvp = 0.0;
      for (size_t i = 0; i < x_gf.size(); i++)
      {
         const Vector &x_dof = x[i];
         const Vector &dx_dof = dx[i]; // Get dx_i/dx

         x_gf[i]->SetFromTrueDofs(dx_dof); // Set x_i = dx_i/dx for i-th term in the product
         y_gf.ProjectCoefficient(prod_coeff); // Recompute product with x_i replaced by dx_i/dx
         y_gf.GetTrueDofs(dfdx); // Get prod_{j!=i} x_j * dx_i/dx for i-th term
         jvp += dfdx; // Accumulate contribution from i-th term
         x_gf[i]->SetFromTrueDofs(x_dof); // reset to original value for next iteration
      }
   }

   ~ProductGridFunctions() override
   {
      for (size_t i = 0; i < x_gf.size(); i++)
      {
         if(x_gf[i]) delete x_gf[i];
      }
   }
};


/// An application that represents the nonlinear diffusion operator: f(T) = -Div(k(u) grad(T)) 
/// with input field T and k, and output field f(T).
class DiffusionOperator : public GraphNode
{
public:

   // Mesh and finite element space
   ParMesh &mesh;
   ParFiniteElementSpace &fes;

   /// Essential dof array.
   Array<int> ess_tdofs;

   /// Grid functions for the temperature and heat flux
   mutable ParGridFunction T, k, dk;
   mutable GridFunctionCoefficient k_gfc, dk_gfc;
   mutable ParNonlinearForm Nform;
   mutable ParLinearForm bform;
   mutable Vector b;

   ConstantCoefficient zero_coeff, one_coeff;

   mutable FunctionalCoefficient *kc = nullptr;
   mutable HypreParMatrix *dfdk_mat = nullptr, *dfdT_mat = nullptr;

public:

   DiffusionOperator(ParFiniteElementSpace &fes_) :
                     // GraphNode(fes_.GetTrueVSize()),
                     GraphNode(fes_.GetTrueVSize(),2*fes_.GetTrueVSize()),
                     mesh(*fes_.GetParMesh()), fes(fes_),
                     T(&fes), k(&fes), dk(&fes),
                     k_gfc(&k), dk_gfc(&dk),
                     Nform(&fes), bform(&fes),
                     zero_coeff(0.0), one_coeff(1.0)
   {
      fes.GetBoundaryTrueDofs(ess_tdofs);
      T = 0.0;
      k = 0.0;
      dk = 0.0;

      bform.AddDomainIntegrator(new DomainLFIntegrator(one_coeff));
      Nform.AddDomainIntegrator(new NonlinearDiffusionIntegrator(&k_gfc, &dk_gfc));
      Nform.SetGradientType(Operator::Type::Hypre_ParCSR);

      b.SetSize(fes.GetTrueVSize()); b = 0.0;
      Assemble();

      SetInputOffsets(Array<int>({0, fes.GetTrueVSize(), 2*fes.GetTrueVSize()}));
      SetOutputOffsets(Array<int>({0, fes.GetTrueVSize()}));
   }

   void SetCoefficient(FunctionalCoefficient *fc) { kc = fc; }

   void Assemble()
   {
      AssembleLinearForms();
      AssembleBilinearForms();
      AssembleNonlinearForms();
   }

   void AssembleBilinearForms()
   {}

   void AssembleNonlinearForms()
   {
      Nform.SetEssentialTrueDofs(ess_tdofs);
      Nform.Setup();
   }

   void AssembleLinearForms()
   {
      bform.Assemble();
      bform.ParallelAssemble(b);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      BlockVector xb(x.GetData(), InputOffsets());
      BlockVector yb(y.GetData(), OutputOffsets());

      MultiVector xmv(2), ymv(1);
      xmv.MakeRef(0, xb.GetBlock(0));
      xmv.MakeRef(1, xb.GetBlock(1));
      ymv.MakeRef(0, yb.GetBlock(0));

      MultMV(xmv, ymv);
   }

   void MultMV(const MultiVector &x, MultiVector &y) const override
   {
      const Vector &tdofs = x[0];
      const Vector &kdofs = x[1];
      Vector &fdofs = y[0];

      k.SetFromTrueDofs(kdofs); // update for use in k_gfc

      if(exec_mode == GraphNode::GRADIENT_MODE)
      {
         if(dfdT_mat) delete dfdT_mat;
         if(dfdk_mat) delete dfdk_mat;

         dk = 0.0;
         k.SetFromTrueDofs(kdofs);
         Operator* grad = &Nform.GetGradient(tdofs);
         dfdT_mat = new HypreParMatrix(dynamic_cast<const HypreParMatrix&>(*grad)); // deep copy

         dk = 1.0;
         k  = 0.0;
         grad = &Nform.GetGradient(tdofs);
         dfdk_mat = new HypreParMatrix(dynamic_cast<const HypreParMatrix&>(*grad)); // deep copy
      }
      else
      {
         if(dfdT_mat) { delete dfdT_mat; dfdT_mat = nullptr; }
         if(dfdk_mat) { delete dfdk_mat; dfdk_mat = nullptr; }
      }

      Nform.Mult(tdofs, fdofs);
      fdofs.SetSubVector(ess_tdofs, 0.0);
   }

   // Exact block jacobian [df/dT, df/dk]
   Operator& GetGradient(const Vector &x) const override
   {
      MFEM_ABORT("GetGradient not implemented for DiffusionOperator");
   }

   void GradientMult(const MultiVector &x, const MultiVector &dx, MultiVector &dy) const override
   {
      const Vector &Tadj = dx[0];
      const Vector &kadj = dx[1];
      Vector &yadj = dy[0];

      const Vector &tdofs = x[0];
      const Vector &kdofs = x[1];

      dfdT_mat->Mult(Tadj, yadj);
      dfdk_mat->AddMult(kadj, yadj);
   }

   /// @brief Destroy the DiffusionOperator object
   ~DiffusionOperator() override
   {
      if(dfdT_mat) delete dfdT_mat;
      if(dfdk_mat) delete dfdk_mat;
   }
};


int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   using GradMode = DAGraph::GradMode;

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");

   args.AddOption(&ctx.grad_mode, "-gm", "--grad-mode",
                  "Gradient mode for the coupled operator (0: exact, 1: finite difference, 2: algorithmic differentiation)");
   args.AddOption(&ctx.coupled, "-cp", "--coupled", "-ucp", "--uncoupled",
                  "Coupled (true) vs. uncoupled (false) solves.");
   args.ParseCheck();


   int order = ctx.order;
   std::string mesh_file = "../../data/star.mesh";
   Mesh *serial_mesh = new Mesh(mesh_file);
   int dim = serial_mesh->Dimension();

   for (int i = 0; i < ctx.ser_ref; ++i) { serial_mesh->UniformRefinement(); }
   serial_mesh->SetCurvature(order, false, dim, Ordering::byNODES);

   ParMesh pmesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   pmesh.UniformRefinement();

   // Finite element spaces
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec);

   // Build all operator nodes
   DiffusionCoefficient diff_coeff_1(fes);
   diff_coeff_1.SetName("k(T1)");
   diff_coeff_1.SetCoefficient(new FunctionalCoefficient(nullptr, 1.0, 3.5e-2));
   // diff_coeff_1.SetCoefficient(new FunctionalCoefficient(nullptr, 1.0, 1.0, 0.1, 0.0));

   DiffusionCoefficient diff_coeff_2(fes);
   diff_coeff_2.SetName("k(T2)");
   diff_coeff_2.SetCoefficient(new FunctionalCoefficient(nullptr, 1.0, 1.0, 2.0, 0.0));
   // diff_coeff_2.SetCoefficient(new FunctionalCoefficient(nullptr, 1.5, 2.5e-2));

   ProductGridFunctions prod_coeff(fes, 2);
   prod_coeff.SetName("k(T1,T2)");

   DiffusionOperator diff_op1(fes);
   diff_op1.SetName("Div(k(T1,T2) grad(T1))");
   diff_op1.SetCoefficient(diff_coeff_1.GetCoefficient());

   DiffusionOperator diff_op2(fes);
   diff_op2.SetName("Div(k(T1,T2) grad(T2))");
   diff_op2.SetCoefficient(diff_coeff_2.GetCoefficient());


   // Build the DAG in any order, and then sort it to ensure the correct execution order
   DAGraph dag(5);
   dag.AddOperator(&diff_coeff_1);
   dag.AddOperator(&diff_op1, fes.GetTrueVSize());
   dag.AddOperator(&diff_op2, fes.GetTrueVSize());
   dag.AddOperator(&diff_coeff_2);
   dag.AddOperator(&prod_coeff);

   Vector k1vec(fes.GetTrueVSize()); k1vec = 0.0;
   Vector k2vec(fes.GetTrueVSize()); k2vec = 0.0;
   Vector kpvec(fes.GetTrueVSize()); kpvec = 0.0;

   Vector k1adj(fes.GetTrueVSize()); k1adj = 0.0;
   Vector k2adj(fes.GetTrueVSize()); k2adj = 0.0;
   // Vector kpadj(fes.GetTrueVSize()); kpadj = 0.0;

   // Input fields get data from 'x' in DAGraph::Mult(x, y)
   Field T1_field(nullptr, nullptr);
   Field T2_field(nullptr, nullptr);

   // Write space for data and adjoint only needed
   // for the intermediate fields k1, k2, and k_prod
   Field k1_field(&k1vec, &k1adj);
   Field k2_field(&k2vec, &k2adj);
   Field kp_field(&kpvec, &kpvec); // can use same space for data & adjoint

   // Output fields get data from 'y' in DAGraph::Mult(x, y)
   Field f1_field(nullptr, nullptr);
   Field f2_field(nullptr, nullptr);


   // Add input and output to the DAG
   int sz = fes.GetTrueVSize();
   dag.AddInput(&T1_field, sz);
   dag.AddInput(&T2_field, sz);
   dag.AddOutput(&f1_field, sz);
   dag.AddOutput(&f2_field, sz);

   // Form connections between the nodes in the DAG
   diff_coeff_1.AddInput(&T1_field);
   diff_coeff_1.AddOutput(&k1_field);

   diff_coeff_2.AddInput(&T2_field);
   diff_coeff_2.AddOutput(&k2_field);

   prod_coeff.AddInputs(&k1_field, &k2_field);
   prod_coeff.AddOutput(&kp_field);

   diff_op1.AddInput(&T1_field);
   diff_op1.AddOutput(&f1_field);

   diff_op2.AddInput(&T2_field);
   diff_op2.AddOutput(&f2_field);

   if(ctx.coupled)
   {
      diff_op1.AddInput(&kp_field); // kp_field
      diff_op2.AddInput(prod_coeff.OutputField(0)); // Can also use kp_field directly
   }
   else
   {
      diff_op1.AddInput(&k1_field); // Can also use diff_coeff_1.OutputField(0)
      diff_op2.AddInput(&k2_field); // Can also use diff_coeff_2.OutputField(0)
   }

   // Assemble DAG: topological sort, validate nodes, etc.
   dag.Assemble();

   std::string output_prefix = ctx.coupled ? "Coupled_Diffusion" : "Uncoupled_Diffusion";

   if(Mpi::Root())
   {
      std::ofstream fout(output_prefix+"-dag.txt");
      fout << "{\n";
      dag.Save(fout);
      fout << "}\n";
      fout << std::flush;
      fout.close();
   }

   // Set initial guess and boundary conditions for T1 and T2
   Array<int> ess_tdofs;
   fes.GetBoundaryTrueDofs(ess_tdofs);

   int T1_idx = 0;
   int T2_idx = 1;

   BlockVector xb(dag.InputOffsets());
   BlockVector yb(dag.OutputOffsets());

   xb.GetBlock(T1_idx).Randomize();
   xb.GetBlock(T2_idx).Randomize();
   xb.GetBlock(T1_idx).SetSubVector(ess_tdofs, 0.0);
   xb.GetBlock(T2_idx).SetSubVector(ess_tdofs, 0.0);

   // Build the nonlinear solver and linear solver for the DAG
   NewtonSolver newton_solver(pmesh.GetComm());
   GMRESSolver linear_solver(pmesh.GetComm());
   linear_solver.SetKDim(500);
   SetSolverParameters(&newton_solver, ctx.tol_nsolve, 0.0, ctx.nl_iter, 1, true);
   SetSolverParameters(&linear_solver, ctx.tol_lsolve, 0.0, ctx.lin_iter, 1, false);

   newton_solver.SetPreconditioner(linear_solver);
   linear_solver.SetPrintLevel(1);

   // Set the gradient mode for the DAG and solve the coupled system
   GradMode gm = static_cast<GradMode>(ctx.grad_mode);
   dag.SetGradientMode(gm);
   newton_solver.SetOperator(dag);
   newton_solver.Mult(xb, yb);

   ParaViewDataCollection *pv = nullptr;
   if (ctx.visualization)
   {
      std::string pv_prefix;
      switch (ctx.grad_mode)
      {
         case 0: pv_prefix = "FD"; break;
         case 1: pv_prefix = "MF"; break;
         default: pv_prefix = "Unknown"; break;
      }

      pv = new ParaViewDataCollection(output_prefix+"-"+pv_prefix, &pmesh);
      pv->SetLevelsOfDetail(order);
      pv->SetDataFormat(VTKFormat::BINARY);
      pv->SetHighOrderOutput(true);

      ParGridFunction T1_gf(&fes);
      ParGridFunction T2_gf(&fes);
      T1_gf.SetFromTrueDofs(yb.GetBlock(T1_idx));
      T2_gf.SetFromTrueDofs(yb.GetBlock(T2_idx));

      pv->RegisterField("T1", &T1_gf);
      pv->RegisterField("T2", &T2_gf);
      pv->Save();
      delete pv;
   }

   std::cout << "Finished solving the coupled diffusion problem." << std::endl;
   return 0;
}

void SetSolverParameters(IterativeSolver *solver, real_t rtol, real_t atol,
                         int max_it, int print_level, bool iterative_mode)
{
    solver->SetRelTol(rtol);
    solver->SetAbsTol(atol);
    solver->SetMaxIter(max_it);
    solver->SetPrintLevel(print_level);
    solver->iterative_mode = iterative_mode;
}