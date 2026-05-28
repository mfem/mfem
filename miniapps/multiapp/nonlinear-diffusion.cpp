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
   int grad_mode = 2;       // Gradient mode for the coupled operator - 0: exact,
                            //                                          1: finite difference,
                            //                                          2: back/forward propagation
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

    real_t Eval(ElementTransformation &Tr,
              const IntegrationPoint &ip) override
    {
        real_t T = T_gf ? T_gf->GetValue(Tr, ip) : 0.0;
        bool eval_f = (mode == Mode::FUNC);
        switch (findex)
        {
        case 1:
         return Exponential(T, eval_f);
        case 2:
         return Polynomial(T, eval_f);
        default:
         return kref;
        }

        return kref;
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


class NonlinearDiffusionIntegrator : public NonlinearFormIntegrator
{
protected:
   Coefficient *k;
   Coefficient *dk;
   ParGridFunction *du;

   Vector u, vec, grad_du, shape;
   DenseMatrix dshape, dshapedxt, adjJ;
public:
   NonlinearDiffusionIntegrator(Coefficient *kappa, Coefficient *dkappa, ParGridFunction *du) :
                                k(kappa), dk(dkappa), du(du) { }

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
      real_t w, k0 = 0.0, dk0 = 0.0, du0 = 0.0;

      elmat.SetSize(dof);
      elmat = 0.0;

      const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(el.GetGeomType(), 2*el.GetOrder());
      u.SetSize(dim);
      shape.SetSize(dof);
      vec.SetSize(dof);
      dshape.SetSize(dof, dim);
      dshapedxt.SetSize(dof, dim);
      grad_du.SetSize(dim);

      // grad(psi) k(u0) grad(u) + grad(psi) k'(u0) grad(u0) u 
      // k and dk can be set to 0 or non-zero to assemble either 
      // term or both terms in the gradient.
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         el.CalcShape(ip, shape);
         el.CalcDShape(ip, dshape);

         Tr.SetIntPoint(&ip);
         w = ip.weight / Tr.Weight();

         Mult(dshape, Tr.AdjugateJacobian(), dshapedxt);

         du0 = du ? du->GetValue(Tr, ip) : 1.0;
         k0  = k ? k->Eval(Tr, ip) : 0.0;
         dk0 = dk ? dk->Eval(Tr, ip) : 0.0;
         
         if(k0 != 0.0 && du0 != 0.0) // grad(psi) k(u0) dT/du grad(u)
         {
            real_t kdu = w*k0*du0;
            AddMult_a_AAt(kdu, dshapedxt, elmat);
         }

         if(dk0 != 0.0) // grad(psi) (dk(u0) grad(u0)) u
         {
            dk0 = w*dk->Eval(Tr, ip);
            dshapedxt.MultTranspose(elfun, u); // grad(u0) in physical space
            u *= dk0; // k'(u0) grad(u0)
            dshapedxt.Mult(u, vec); // grad(psi) k'(u0) grad(u0)
            AddMultVWt(vec, shape, elmat); // grad(psi) k'(u0) grad(u0) u
         }

         if(du) // grad(psi) (k0 grad(dT/du)) u
         {
            du->GetGradient(Tr, grad_du); // grad(dT/du) in physical space
            grad_du *= (w * k0); // k(u0) grad(dT/du)
            dshapedxt.Mult(grad_du, vec); // grad(psi) k(u0) grad(dT/du)
            AddMultVWt(vec, shape, elmat); // grad(psi) k(u0) grad(dT/du) u
         }

         // if(k) // grad(psi) k(u0) grad(u)
         // {
         //    k0 = w*k->Eval(Tr, ip);
         //    AddMult_a_AAt(k0, dshapedxt, elmat);
         // }
      }
   }
};

/// An application that takes an input field T, and computes an output field k(T)
// represented by the FunctionalCoefficient class.
// This application also provides the derivative dk/dT.
class DiffusionCoefficient : public Application
{
public:
   using Mode = FunctionalCoefficient::Mode;

protected:
   ParFiniteElementSpace &fes;
   mutable ParGridFunction T, k;
   mutable FunctionalCoefficient *kc;
   mutable Vector tdof, kdof, dk_dof;
   mutable Mode mode = Mode::FUNC;

public:
   DiffusionCoefficient(ParFiniteElementSpace &fes) :
                        Application(0), fes(fes), T(&fes), k(&fes),
                        kc(new FunctionalCoefficient(&T, 1.0, 5.0e-2))
   {
      k = 0.0;
      T = 0.0;
      // fields.AddSourceField("k(T)", &kdof); // TODO: This shoud work but doesn't?
      fields.AddSourceField("k(T)", new Field(&kdof, Field::Type::SOURCE), true);
      fields.AddField("T", &tdof);
      k.ProjectCoefficient(*kc);
      k.GetTrueDofs(kdof);
      dk_dof = kdof;
   }

   void SetMode(Mode mode) { this->mode = mode; }

   FunctionalCoefficient* GetCoefficient() { return kc; }

   void SetCoefficient(FunctionalCoefficient *fc)
   {
      if(kc) delete kc;
      kc = fc;
      kc->SetMode(mode);
      kc->UpdateGridFunction(&T);
   }

   void Update() override
   {
      k.Update();
      T.Update();
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      T.SetFromTrueDofs(tdof);
      auto md = kc->GetMode();
      kc->SetMode(Mode::FUNC);
      k.ProjectCoefficient(*kc);
      k.GetTrueDofs(kdof);
      kc->SetMode(md); // reset mode
   }

   void Execute(const Vector &x, Vector &y) override
   {
      Mult(x, y);
   }

   void GetDerivative(Field* y, Field* x, Vector &x0, Vector &dydx)
   {
      if(y->GetID() == Fields("k(T)")->GetID() &&
         x->GetID() == Fields("T")->GetID())
      {
         T.SetFromTrueDofs(tdof); // Compute dk/dT at T = tdof
         auto md = kc->GetMode();
         kc->SetMode(Mode::GRAD);
         k.ProjectCoefficient(*kc);
         k.GetTrueDofs(dydx);
         kc->SetMode(md); // reset mode
      }
      else
      {
         dydx = 0.0;
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
class ProductGridFunctions : public Application
{
protected:

   ParFiniteElementSpace &fes;
   mutable std::vector<Vector*> x_dof;
   mutable std::vector<ParGridFunction*> x_gf;
   mutable Vector y_dof, dx, dfdx;
   mutable ParGridFunction y_gf;
   mutable GridFunctionProductCoefficient prod_coeff;
public:
   ProductGridFunctions(ParFiniteElementSpace &fes, int n) :
                       Application(0),
                       fes(fes), x_dof(n), x_gf(n),
                       y_gf(&fes), prod_coeff(x_gf)
   {
      y_gf = 0.0;
      for (int i = 0; i < n; i++)
      {
         x_gf[i]  = new ParGridFunction(&fes);
         *x_gf[i] = 0.0;
         x_dof[i] = new Vector(0);
         x_gf[i]->GetTrueDofs(*x_dof[i]);
         fields.AddField("x"+std::to_string(i), x_dof[i]);
      }
      y_gf.ProjectCoefficient(prod_coeff);
      y_gf.GetTrueDofs(y_dof);
      fields.AddSourceField("prod(x)", new Field(&y_dof, Field::Type::SOURCE), true);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      for (size_t i = 0; i < x_gf.size(); i++)
      {
         x_gf[i]->SetFromTrueDofs(*x_dof[i]);
      }
      y_gf.ProjectCoefficient(prod_coeff);
      y_gf.GetTrueDofs(y_dof);
   }

   void Execute(const Vector &x, Vector &y) override
   {
      Mult(x, y);
   }

   void GetDerivative(Field* y, Field* x, Vector &x0, Vector &dydx) override
   {
      if(y->GetID() == Fields("prod(x)")->GetID())
      {
         //     y = x_0 * x_1 * ... * x_{n-1}
         // dy/dx = sum_i (dy/dx_i * dx_i/dx)
         //       = sum_i ( x_0 * ... * x_{i-1} * dx_i/dx * x_{i+1} * ... * x_{n-1} ) 
         for (size_t i = 0; i < x_gf.size(); i++)
         {
            x_gf[i]->SetFromTrueDofs(*x_dof[i]); // Set all x_i
         }

         dydx.SetSize(x_dof[0]->Size());
         dydx = 0.0;
         for (size_t i = 0; i < x_gf.size(); i++)
         {
            std::string x_name = "x" + std::to_string(i);

            // auto x_src = Fields(x_name)->GetSource();
            // MFEM_ASSERT(x_src != nullptr, "Target field: " << x_name
            //             << " does not have an associated source field.");

            // auto x_op = x_src->GetNode();
            // MFEM_ASSERT(x_op != nullptr, "Source field: " << x_src->GetID() 
            //             << " for target field: " << x_name 
            //             << " does not have an associated GraphNode.");

            // x_op->GetDerivative(Fields(x_name), x, x0, dx); // Get dx_i/dx
            Fields(x_name)->GetDerivative(x,x0, dx); // Get dx_i/dx for i-th term in the product
            x_gf[i]->SetFromTrueDofs(dx); // Set x_i = dx_i/dx for i-th term in the product

            y_gf.ProjectCoefficient(prod_coeff); // Recompute product with x_i replaced by dx_i/dx
            y_gf.GetTrueDofs(dfdx); // Get dy/dx_i * dx_i/dx for i-th term
            dydx += dfdx; // Accumulate contribution from i-th term
            x_gf[i]->SetFromTrueDofs(*x_dof[i]); // reset to original value for next iteration
         }
      }
      else
      {
         dydx = 0.0;
      }
   }

   ~ProductGridFunctions() override
   {
      for (size_t i = 0; i < x_dof.size(); i++)
      {
         if(x_dof[i]) delete x_dof[i];
         if(x_gf[i]) delete x_gf[i];
      }
   }
};


/// An application that represents the nonlinear diffusion operator: f(T) = -Div(k(u) grad(T)) 
/// with input field T and k, and output field f(T).
class DiffusionOperator : public Application
{
public:
   enum Mode { MULTIAPP = 0, EXACT = 1 };

public:

   // Mesh and finite element space
   ParMesh &mesh;
   ParFiniteElementSpace &fes;

   /// Essential dof array.
   Array<int> ess_tdofs;

   /// Grid functions for the temperature and heat flux
   mutable ParGridFunction T, k, dk, dT;
   mutable GridFunctionCoefficient k_gfc, dk_gfc, dT_gfc;
   mutable ParNonlinearForm Nform;
   mutable ParLinearForm bform;
   mutable Vector tdofs, kdofs, dk_dofs, dT_dofs;

   mutable Vector b;
   mutable Vector output;

   ConstantCoefficient zero_coeff, one_coeff;

   mutable FunctionalCoefficient *kc = nullptr;
   Mode mode = Mode::MULTIAPP;

public:

   DiffusionOperator(ParFiniteElementSpace &fes_) :
                     Application(fes_.GetTrueVSize()),
                     mesh(*fes_.GetParMesh()), fes(fes_),
                     T(&fes), k(&fes), dk(&fes), dT(&fes),
                     k_gfc(&k), dk_gfc(&dk), dT_gfc(&dT),
                     Nform(&fes), bform(&fes),
                     zero_coeff(0.0), one_coeff(1.0)
   {

      fields.AddField("input", new Field(nullptr, Field::Type::TARGET), true);
      fields.AddSourceField("output", new Field(nullptr, Field::Type::SOURCE), true);

      fes.GetBoundaryTrueDofs(ess_tdofs);
      T = 0.0;
      k = 0.0;
      dk = 0.0;
      dT = 0.0;

      fields.AddField("T",&tdofs);
      fields.AddField("k",&kdofs);
      // fields.AddField("dk/dT",&dk_dofs); // Field not used in forming graph edges

      bform.AddDomainIntegrator(new DomainLFIntegrator(one_coeff));
      Nform.AddDomainIntegrator(new NonlinearDiffusionIntegrator(&k_gfc, &dk_gfc, &dT));
      Nform.SetGradientType(Operator::Type::Hypre_ParCSR);

      b.SetSize(fes.GetTrueVSize()); b = 0.0;
      output.SetSize(fes.GetTrueVSize());
      Assemble();
   }

   void SetMode(Mode mode) { this->mode = mode; }
   void SetCoefficient(FunctionalCoefficient *fc) { kc = fc; }

   void Assemble()
   {
      AssembleLinearForms();
      AssembleBilinearForms();
      AssembleNonlinearForms();
   }

   void AssembleBilinearForms()
   { }

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

   /// Update finite element space and re-assemble forms
   /// if the mesh has changed
   void Update() override 
   {
        fes.Update();
        T.Update(); k.Update(); dk.Update(); // grid functions
        Nform.Update(); bform.Update(); // Forms
        Assemble();
   }

   void BuildSolvers() {}

   void Mult(const Vector &xx, Vector &yy) const override
   {
      if(mode == Mode::EXACT)
      {
         MFEM_VERIFY(kc != nullptr, "FunctionalCoefficient must be set for exact mode");
         tdofs = xx;
         T.SetFromTrueDofs(tdofs);
         kc->UpdateGridFunction(&T);
         kc->SetMode(FunctionalCoefficient::Mode::FUNC);
         k.ProjectCoefficient(*kc);
      }
      else
      {
         T.SetFromTrueDofs(tdofs);
         k.SetFromTrueDofs(kdofs);
      }

      Nform.Mult(tdofs, output);
      yy = output;
      yy -= b;

      yy.SetSubVector(ess_tdofs, 0.0);
   }

   // Exact jacobian for uncoupled solve
   Operator& GetGradient(const Vector &x) const override
   {
      MFEM_VERIFY(kc != nullptr, "FunctionalCoefficient must be set to compute gradient");

      T.SetFromTrueDofs(x);
      kc->UpdateGridFunction(&T);
      kc->SetMode(FunctionalCoefficient::Mode::FUNC);
      k.ProjectCoefficient(*kc);
      kc->SetMode(FunctionalCoefficient::Mode::GRAD);
      dk.ProjectCoefficient(*kc);
      return Nform.GetGradient(x);
   }

   void Execute(const Vector &xx, Vector &yy) override
   {
      Mult(xx, yy);
   }

   [[nodiscard]] bool GetDerivative(Field *y, Vector &x, Operator* &dydx) override
   {
      // f = Div(k(y) grad(T))
      // [df/dq](T0,q0) = Div(dk/dq grad(T0)) + Div(k(q0) grad(q')dT/dq)
      // T0,q0 are from the forward pass. If q = T, dT/dq = I, and if q != T, dT/dq = 0. So,
      // we can disable terms in the Jacobian and assembling the Jacobian with only the relevant 
      // terms depending on whether we are differentiating w.r.t. T or or some other field.

      // auto k_src = Fields("k")->GetSource();
      // MFEM_ASSERT(k_src != nullptr, "Target field: k does not have an associated source field!");

      // auto k_op = k_src->GetNode();
      // MFEM_ASSERT(k_op != nullptr, "Source field: " << k_src->GetID() 
      //             << " for target field: k does not have an associated GraphNode.");

      // k_op->GetDerivative(Fields("k"), y, x, dk_dofs); // Get dk/dy
      // dk.SetFromTrueDofs(dk_dofs);

      Fields("k")->GetDerivative(y, x, dk_dofs); // Get dk/dy
      dk.SetFromTrueDofs(dk_dofs);

      Fields("T")->GetDerivative(y, x, dT_dofs); // Get dT/dy
      dT.SetFromTrueDofs(dT_dofs);

      // if(y->GetID() == Fields("T")->GetID())
      // {
      //    // Set k(q0) for the second term in the product rule
      //    k.SetFromTrueDofs(kdofs);
      // }
      // else
      // {
      //    // If not differentiating w.r.t. T, then disable the
      //    // second term in the product rule by setting k(y0) = 0
      //    // Here we have assumed that dT/dy = 0. Techinically, we should do
      //    // include this by requesting dT/dy from the source operator and
      //    // including it in the Jacobian, but for this case we assume dT/dy = 0 for y != T.
      //    k = 0.0;
      // }

      // Get the Jacobain with either/both terms in product rule...
      // depending on whether dk/dy and k are zero or non-zero.
      Operator* grad = &Nform.GetGradient(tdofs);
      dydx = new HypreParMatrix(const_cast<const HypreParMatrix&>(
                                dynamic_cast<const HypreParMatrix&>(*grad)));

      bool ownership = true; // Caller is responsible for deleting the operator
      return ownership;
   }

    /// @brief Destroy the DiffusionOperator object
   ~DiffusionOperator() override
   { }
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

   DiffusionCoefficient kappa(fes);
   kappa.SetName("k(T1)");
   kappa.SetCoefficient(new FunctionalCoefficient(nullptr, 1.0, 3.5e-2));
   // kappa.SetCoefficient(new FunctionalCoefficient(nullptr, 1.0, 1.0, 0.1, 0.0));

   DiffusionCoefficient gamma(fes);
   gamma.SetName("k(T2)");
   gamma.SetCoefficient(new FunctionalCoefficient(nullptr, 1.0, 1.0, 2.0, 0.0));
   // gamma.SetCoefficient(new FunctionalCoefficient(nullptr, 1.5, 2.5e-2));

   ProductGridFunctions kap_gamma(fes, 2);
   kap_gamma.SetName("k(T1,T2)");

   DiffusionOperator diffusion_op(fes);
   diffusion_op.SetName("Div(k(T1,T2) grad(T1))");
   diffusion_op.SetCoefficient(kappa.GetCoefficient());

   DiffusionOperator diffusion_op2(fes);
   diffusion_op2.SetName("Div(k(T1,T2) grad(T2))");
   diffusion_op2.SetCoefficient(gamma.GetCoefficient());


   DAGraph dag(5);
   dag.AddOperator(&kappa);
   dag.AddOperator(&gamma);
   dag.AddOperator(&kap_gamma);
   dag.AddOperator(&diffusion_op, fes.GetTrueVSize());
   dag.AddOperator(&diffusion_op2, fes.GetTrueVSize());
   dag.OwnGradientOperators(true);

   std::string input_name = "x" + std::to_string(diffusion_op.GetNodeIndex());
   dag.Fields().AddTargetField(input_name, diffusion_op.Fields("T"));
   dag.Fields().AddTargetField(input_name, kappa.Fields("T"));

   input_name = "x" + std::to_string(diffusion_op2.GetNodeIndex());
   dag.Fields().AddTargetField(input_name, diffusion_op2.Fields("T"));
   dag.Fields().AddTargetField(input_name, gamma.Fields("T"));

   kappa.Fields().AddTargetField("k(T)", kap_gamma.Fields("x0"));
   gamma.Fields().AddTargetField("k(T)", kap_gamma.Fields("x1"));

   if(ctx.coupled)
   {
      kap_gamma.Fields().AddTargetField("prod(x)", diffusion_op.Fields("k"));
      kap_gamma.Fields().AddTargetField("prod(x)", diffusion_op2.Fields("k"));
   }
   else
   {
      kappa.Fields().AddTargetField("k(T)", diffusion_op.Fields("k"));
      gamma.Fields().AddTargetField("k(T)", diffusion_op2.Fields("k"));
   }

   std::string output_prefix = ctx.coupled ? "CoupledDiffusion" : "UncoupledDiffusion";

   if(Mpi::Root())
   {
      std::ofstream fout(output_prefix+"-dag.txt");
      fout << "{\n";
      dag.Save(fout);
      fout << "}\n";
      fout << std::flush;
      fout.close();
   }

   // int dc_id = kappa.GetNodeIndex();
   // int do_id = diffusion_op.GetNodeIndex();

   int dk_id = kappa.GetInputIndex();
   int dg_id = gamma.GetInputIndex();
   int kg_id = kap_gamma.GetInputIndex();
   int do_id = diffusion_op.GetInputIndex();
   int do2_id = diffusion_op2.GetInputIndex();

   std::cout << "Node Indices: " << std::endl;
   std::cout << "Diffusion Coefficient: kappa " << dk_id << std::endl;
   std::cout << "Diffusion Coefficient: gamma " << dg_id << std::endl;
   std::cout << "Coupled Coefficient: kap_gamma " << kg_id << std::endl;
   std::cout << "Diffusion Operator: " << do_id << std::endl;
   std::cout << "Diffusion Operator: " << do2_id << std::endl;

   BlockVector xb(dag.InputOffsets());
   BlockVector yb(dag.OutputOffsets());
   // xb.GetBlock(dc_id) = 0.0;
   xb.GetBlock(do_id) = 1.0;
   xb.GetBlock(do_id).Randomize();
   xb.GetBlock(do2_id) = 1.0;
   xb.GetBlock(do2_id).Randomize();


   Array<int> ess_tdofs;
   fes.GetBoundaryTrueDofs(ess_tdofs);
   xb.GetBlock(do_id).SetSubVector(ess_tdofs, 0.0);
   xb.GetBlock(do2_id).SetSubVector(ess_tdofs, 0.0);


   NewtonSolver newton_solver(pmesh.GetComm());
   GMRESSolver linear_solver(pmesh.GetComm());
   linear_solver.SetKDim(500);
   SetSolverParameters(&newton_solver, ctx.tol_nsolve, 0.0, ctx.nl_iter, 1, true);
   SetSolverParameters(&linear_solver, ctx.tol_lsolve, 0.0, ctx.lin_iter, 1, false);

   newton_solver.SetPreconditioner(linear_solver);
   linear_solver.SetPrintLevel(1);

   bool use_exact = (ctx.grad_mode == 0);

   if(use_exact)
   {
      diffusion_op.SetMode(DiffusionOperator::Mode::EXACT);
      newton_solver.SetOperator(diffusion_op);
      newton_solver.Mult(xb.GetBlock(do_id), yb.GetBlock(do_id));
   }
   else
   {
      GradMode gm = static_cast<GradMode>(ctx.grad_mode - 1);
      dag.SetGradientMode(gm);
      newton_solver.SetOperator(dag);
      newton_solver.Mult(xb, yb);
   }


   ParaViewDataCollection *pv = nullptr;
   if (ctx.visualization)
   {
      std::string pv_prefix;
      switch (ctx.grad_mode)
      {
         case 0: pv_prefix = "Exact"; break;
         case 1: pv_prefix = "FD"; break;
         case 2: pv_prefix = "FWD"; break;
         case 3: pv_prefix = "BWD"; break;
         default: pv_prefix = "Unknown"; break;
      }

      pv = new ParaViewDataCollection(output_prefix+"-"+pv_prefix, &pmesh);
      pv->SetLevelsOfDetail(order);
      pv->SetDataFormat(VTKFormat::BINARY);
      pv->SetHighOrderOutput(true);

      ParGridFunction T1_gf(&fes);
      ParGridFunction T2_gf(&fes);
      T1_gf.SetFromTrueDofs(yb.GetBlock(do_id));
      T2_gf.SetFromTrueDofs(yb.GetBlock(do2_id));

      pv->RegisterField("T1", &T1_gf);
      pv->RegisterField("T2", &T2_gf);
      // pv->RegisterField("k(T)", &diffusion_op.Fields()["k"]->Data());
      pv->Save();
      delete pv;
   }

   // diffusion_op.SetMode(true);
   // diffusion_op.Mult(xb.GetBlock(do_id), yb.GetBlock(do_id));
   // std::cout << "Initial residual norm: " << yb.GetBlock(do_id).Norml2() << "\n";


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