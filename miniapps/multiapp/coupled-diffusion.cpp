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
class LambdaCoefficient : public Coefficient
{
public:
   enum Mode { FUNC = 0, GRAD = 1};

protected:
   ParGridFunction *gf = nullptr;
   std::function<real_t(real_t, bool)> func;
   Mode mode = Mode::FUNC; // otherwise, grad

public:
   LambdaCoefficient(ParGridFunction *gf, std::function<real_t(real_t, bool)> func) :
                     gf(gf), func(func) { }

    void SetMode(Mode mode) { this->mode = mode; }
    Mode GetMode() const { return mode; }

    real_t Eval(ElementTransformation &Tr,
              const IntegrationPoint &ip) override
    {
        real_t x = gf ? gf->GetValue(Tr, ip) : 0.0;
        bool eval_f = (mode == Mode::FUNC);
        return func(x, eval_f);
    }
};


/// A coefficient defined by the product over the ndim gridfunctions (i.e., prod_i x_i)
class VectorProductCoefficient : public Coefficient
{
protected:
   ParGridFunction &gf;
   Vector values;

public:
   VectorProductCoefficient(ParGridFunction &gf) : gf(gf) { }

   real_t Eval(ElementTransformation &Tr, const IntegrationPoint &ip) override
   {
      real_t prod = 1.0;
      gf.GetVectorValue(Tr, ip, values);
      for(int i = 0; i < values.Size(); i++)
      {
         prod *= values[i];
      }
      return prod;
   }
};


class CoefficientIntegrator : public NonlinearFormIntegrator
{
protected:
   std::function<real_t(real_t, bool)> func;
   Vector shape;

public:
   CoefficientIntegrator(std::function<real_t(real_t, bool)> func) : func(func) { }

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
         real_t x = elfun * shape; // Evaluate the state at the integration point
         real_t f = func(x, true); // Evaluate the function value
         for (int j = 0; j < dof; j++)
         {
            elvect(j) += f * shape(j);
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
         real_t x = elfun * shape; // Evaluate the state at the integration point
         real_t dfdx = func(x, false); // Evaluate the derivative value
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
class DiffusionCoefficient : public GraphNode
{
protected:
   ParFiniteElementSpace &fes;
   Array<int> in_offsets, out_offsets;

   mutable ParNonlinearForm Nform;
   mutable Operator *J = nullptr; // Jacobian for the nonlinear form

   CoefficientIntegrator *coeff_integrator = nullptr;

public:
   DiffusionCoefficient(ParFiniteElementSpace &fes, 
                        std::function<real_t(real_t, bool)> func) :
                        GraphNode(fes.GetTrueVSize()), fes(fes),
                        Nform(&fes), coeff_integrator(new CoefficientIntegrator(func))
   {
      // Testing with the nonlinear form framework to compute k(T) and dk/dT
      Nform.AddDomainIntegrator(coeff_integrator); // Transfer ownership
      Nform.SetGradientType(Operator::Type::Hypre_ParCSR);
      Nform.Setup();

      in_offsets = Array<int>({0, fes.GetTrueVSize()});
      out_offsets = Array<int>({0, fes.GetTrueVSize()});
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      BlockVector xb(x.GetData(), in_offsets);
      BlockVector yb(y.GetData(), out_offsets);

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
   { }
};

/// An application that takes n input fields x_i, and computes an output 
/// field prod(x) := y = prod_i x_i.
class FieldProduct : public GraphNode
{
protected:
   int ninputs;
   ParFiniteElementSpace *nd_fes;
   Array<int> in_offsets, out_offsets;

   mutable Vector dfdx, xdof;
   mutable ParGridFunction x_gf;
   mutable ParGridFunction y_gf;
   mutable VectorProductCoefficient prod_coeff;

public:
   FieldProduct(ParFiniteElementSpace &fes, int n) :
                GraphNode(fes.GetTrueVSize(), fes.GetTrueVSize() * n), ninputs(n),
                nd_fes(new ParFiniteElementSpace(fes.GetParMesh(), fes.FEColl(), n)),
                x_gf(nd_fes), y_gf(&fes), prod_coeff(x_gf)
   {
      in_offsets.SetSize(n+1);
      in_offsets[0] = 0;
      for (int i = 0; i < n; i++)
      {
         in_offsets[i+1] = in_offsets[i] + fes.GetTrueVSize();
      }
      x_gf = 0.0;
      y_gf = 0.0;
      x_gf.GetTrueDofs(xdof);
      y_gf.ProjectCoefficient(prod_coeff);

      out_offsets = Array<int>({0, fes.GetTrueVSize()});
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      BlockVector yb(y.GetData(), out_offsets);

      x_gf.SetFromTrueDofs(x);
      y_gf.ProjectCoefficient(prod_coeff);
      y_gf.GetTrueDofs(yb.GetBlock(0));
   }

   void MultMV(const MultiVector &x, MultiVector &y) const override
   {
      for (int i = 0; i < ninputs; i++)
      {
         xdof.SetVector(x[i], in_offsets[i]); // Set all x_i
      }

      Vector &y_dof = y[0];
      x_gf.SetFromTrueDofs(xdof);
      y_gf.ProjectCoefficient(prod_coeff);
      y_gf.GetTrueDofs(y_dof);
   }

   void GradientMult(const MultiVector &x, const MultiVector &dx, MultiVector &dy) const override
   {
      // Jacobian vector product for y = prod_i x_i is:
      // dy/dx = sum_i (prod_{j!=i} x_j * dx_i/dx)
      for (int i = 0; i < ninputs; i++)
      {
         xdof.SetVector(x[i], in_offsets[i]); // Set all x_i
      }

      Vector &jvp = dy[0];
      jvp = 0.0;
      for (int i = 0; i < ninputs; i++)
      {
         xdof.SetVector(dx[i], in_offsets[i]); // Set x_i = dx_i/dx for i-th term in the product
         x_gf.SetFromTrueDofs(xdof);
         y_gf.ProjectCoefficient(prod_coeff); // Recompute product with x_i replaced by dx_i/dx
         y_gf.GetTrueDofs(dfdx); // Get prod_{j!=i} x_j * dx_i/dx for i-th term
         jvp += dfdx; // Accumulate contribution from i-th term
         xdof.SetVector(x[i], in_offsets[i]); // Reset x_i to original value
      }
   }

   ~FieldProduct() override
   {
      if(nd_fes) delete nd_fes;
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

   // Offsets if calling Mult
   Array<int> in_offsets, out_offsets;

   /// Grid functions for the temperature and heat flux
   mutable ParGridFunction T, k, dk;
   mutable GridFunctionCoefficient k_gfc, dk_gfc;
   mutable ParNonlinearForm Nform;
   mutable ParLinearForm bform;
   mutable Vector b;

   ConstantCoefficient zero_coeff, one_coeff;

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

      in_offsets = Array<int>({0, fes.GetTrueVSize(), 2*fes.GetTrueVSize()});
      out_offsets = Array<int>({0, fes.GetTrueVSize()});
   }

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
      BlockVector xb(x.GetData(), in_offsets);
      BlockVector yb(y.GetData(), out_offsets);

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
   Operator& GetGradientMV(const MultiVector &x) const override
   {
      MFEM_ABORT("GetGradientMV not implemented for DiffusionOperator");
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

   // using GradMode = DAGraph::GradMode;

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
   auto exp_func = [kref=1.0, a = 3.5e-2](real_t x, bool eval_f) -> real_t
   {
      real_t k = kref * exp(a*x);
      real_t dk = a * k;
      return eval_f ? k : dk;
   };
   DiffusionCoefficient diff_coeff_1(fes, exp_func);
   diff_coeff_1.SetName("k(T1)");

   auto poly_func = [kref=1.0, a0=1.0, a1=2.0, a2=0.0](real_t x, bool eval_f) -> real_t
   {
      return eval_f ? kref * (a0 + a1*x + a2*x*x) : kref * (a1 + 2*a2*x);
   };
   DiffusionCoefficient diff_coeff_2(fes, poly_func);
   diff_coeff_2.SetName("k(T2)");

   FieldProduct prod_coeff(fes, 2);
   prod_coeff.SetName("k(T1,T2)");

   DiffusionOperator diff_op1(fes);
   diff_op1.SetName("Div(k(T1,T2) grad(T1))");

   DiffusionOperator diff_op2(fes);
   diff_op2.SetName("Div(k(T1,T2) grad(T2))");

   Vector k1vec(fes.GetTrueVSize()); k1vec = 0.0;
   Vector k1adj(fes.GetTrueVSize()); k1adj = 0.0;
   Vector kpvec(fes.GetTrueVSize()); kpvec = 0.0;

   // Input fields get data from 'x' in DAGraph::Mult(x, y)
   Field *T1_field = new Field();
   Field *T2_field = new Field();

   // Intermediate fields for the diffusion coefficients
   Field *k1_field = new Field();
   Field *k2_field = new Field();
   Field *kp_field = new Field(); // Only needed for the coupled case

   // Output fields get data from 'y' in DAGraph::Mult(x, y)
   Field *f1_field = new Field();
   Field *f2_field = new Field();

   // Write space for data and adjoint only needed
   // for the intermediate fields k1, k2, and k_prod (if coupled)
   k1_field->SetData(&k1vec, &k1adj); // Use different provided memory for data and adjoint
   k2_field->AllocateData(k1vec); // Allocate memory for data and adjoint of same size and type
   kp_field->SetData(&kpvec); // Use same, provided memory for data and adjoint

   // Define the DAG
   DAGraph dag;
   dag.Watch({T1_field, T2_field}); // Track fields that are inputs to the DAG
   dag.StartRecording();

   diff_coeff_1.RegisterFields({T1_field}, {k1_field});
   diff_coeff_2.RegisterFields({T2_field}, {k2_field});

   if(ctx.coupled)
   {
      prod_coeff.RegisterFields({k1_field, k2_field}, {kp_field});
      diff_op1.RegisterFields({T1_field, kp_field}, {f1_field});
      diff_op2.RegisterFields({T2_field, kp_field}, {f2_field}, // Possible to specify action lambdas
  /* force const if needed */   [&op=std::as_const(diff_op2)](const MultiVector &x, MultiVector &y) { op.MultMV(x, y); },
  /* Default for GraphNode */   [&op=diff_op2](const MultiVector &x, const MultiVector &dx, MultiVector &dy) { op.GradientMult(x, dx, dy); },
///* If using mfem::Operator */ [&op=diff_op2](const MultiVector &x, const MultiVector &dx, MultiVector &dy) { op.GetGradientMV(x).MultMV(dx, dy); },
                                [&op=diff_op2](const MultiVector &x, const MultiVector &dx, MultiVector &dy) { op.GradientMultTranspose(x, dx, dy); }
                              );

   /* // If you want to use state-dependent operators
      StateContainer app_state; // This is a user-defined struct that holds the state of the system
      state_op.RegisterFields({in1, in2}, {out1, out2}, app_state,
                                 [&op=std::as_const(state_op)](StateContainer &state, const MultiVector &x,
                                                               const MultiVector &dx, MultiVector &dy)
                                                               {
                                                                  op.FunctionThatNeedsState(state, x, dx, dy);
                                                                  // Or
                                                                  // op.SetState(state);
                                                                  // op.MultMV(x, y);
                                                                  // op.GetState(state);
                                                               },
                                 // Can provide lambdas for GradientMult, etc.
                                 );
   */
   }
   else
   {
      diff_op1.RegisterFields({T1_field, k1_field}, {f1_field});
      diff_op2.RegisterFields({T2_field, k2_field}, {f2_field});
   }

   // Stop recording and specify the output fields of the DAG
   // Outputs can also be intermediate fields.
   dag.StopRecording({f1_field, f2_field});

   int sz = fes.GetTrueVSize();
   // Needed to construct MultiVector from Vector in Mult()
   Array<int> dag_offsets({0, sz, 2*sz});
   dag.SetOffsets(dag_offsets, dag_offsets);

   // Assemble DAG: topological sort (if needed), validate nodes, etc.
   dag.Assemble();

   std::string output_prefix = ctx.coupled ? "Coupled_Diffusion" : "Uncoupled_Diffusion";

   if(Mpi::Root())
   {
      std::ofstream fout(output_prefix+"-dag.txt");
      fout << "{\n";
      // dag.Save(fout);
      fout << "}\n";
      fout << std::flush;
      fout.close();
   }

   // Set initial guess and boundary conditions for T1 and T2
   Array<int> ess_tdofs;
   fes.GetBoundaryTrueDofs(ess_tdofs);

   int T1_idx = 0;
   int T2_idx = 1;

   BlockVector xb(dag_offsets);
   BlockVector yb(dag_offsets);

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

   // Delete fields
   if(T1_field) delete T1_field;
   if(T2_field) delete T2_field;
   if(k1_field) delete k1_field;
   if(k2_field) delete k2_field;
   if(kp_field) delete kp_field;
   if(f1_field) delete f1_field;
   if(f2_field) delete f2_field;

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