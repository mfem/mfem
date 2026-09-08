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
protected:
   ParGridFunction *gf = nullptr;
   std::function<real_t(real_t)> func;

public:
   LambdaCoefficient(ParGridFunction *gf, std::function<real_t(real_t)> func) :
                     gf(gf), func(func) { }

    real_t Eval(ElementTransformation &Tr,
              const IntegrationPoint &ip) override
    {
        real_t x = gf ? gf->GetValue(Tr, ip) : 0.0;
         return func(x);
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

class VectorProductDerivativeCoefficient : public Coefficient
{
protected:
   ParGridFunction &gf, &gf_dx;
   Vector vals, dvals;

public:
   VectorProductDerivativeCoefficient(ParGridFunction &gf, ParGridFunction &gf_dx) :
                                      gf(gf), gf_dx(gf_dx) { }

   real_t Eval(ElementTransformation &Tr, const IntegrationPoint &ip) override
   {
      gf.GetVectorValue(Tr, ip, vals);
      gf_dx.GetVectorValue(Tr, ip, dvals);
      real_t prod = 1.0;
      for(int i = 0; i < vals.Size(); i++)
      {
         prod *= vals[i];
      }
      real_t sum = 0.0;
      for(int i = 0; i < vals.Size(); i++)
      {
         sum += dvals[i] * (prod / vals[i]); 
      }
      return sum;
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
            real_t kdT = w*dk0;
            dshapedxt.MultTranspose(elfun, u); // grad(T0) in physical space
            u *= kdT; // k'(T0) * grad(T0)
            dshapedxt.Mult(u, vec); // grad(psi) * k'(T0) * grad(T0)
            AddMultVWt(vec, shape, elmat); // grad(psi) * k'(T0) * grad(T0) * T
         }
      }
   }
};

/// An application that takes an input field T, and computes an output field k(T)
class DiffusionCoefficient : public GraphOperator
{
public:
   class GradientOperator : public Operator
   {
   protected:
      mutable Vector xdofs;
      mutable ParGridFunction x_gf;
      ParGridFunction &y_gf;
      mutable LambdaCoefficient dk_coeff;

   public:
      GradientOperator(int sz, ParGridFunction &x, ParGridFunction &y,
                       std::function<real_t(real_t)> func) : Operator(sz,sz),
                       x_gf(x), y_gf(y), dk_coeff(&x_gf, func) { }

      Operator &GetGradientMV(const MultiVector &x) const
      {
         x_gf.SetFromTrueDofs(x[0]);
         y_gf.ProjectCoefficient(dk_coeff);
         y_gf.GetTrueDofs(xdofs); // J = dk/dT
         return const_cast<GradientOperator&>(*this);
      }

      void MultMV(const MultiVector &x, MultiVector &y) const override
      { y[0] = x[0]; y[0] *= xdofs; } // y = J * x

      void Mult(const Vector &x, Vector &y) const override
      { MFEM_ABORT("Mult not implemented for GradientOperator."); }
   };
protected:

   ParFiniteElementSpace &fes;
   Array<int> in_offsets, out_offsets;
   std::function<real_t(real_t, bool)> lambda_func;

   mutable ParGridFunction T_gf, k_gf;
   mutable LambdaCoefficient k_coeff;
   GradientOperator gradient;

public:
   DiffusionCoefficient(ParFiniteElementSpace &fes, 
                        std::function<real_t(real_t, bool)> func) :
                        GraphOperator(fes.GetTrueVSize()), fes(fes),
                        lambda_func(func), T_gf(&fes), k_gf(&fes),
                        k_coeff(&T_gf, [&](real_t x) { return lambda_func(x, true); }),
                        gradient(fes.GetTrueVSize(), T_gf, k_gf, [&](real_t x) { return lambda_func(x, false); })
   {
      in_offsets = Array<int>({0, fes.GetTrueVSize()});
      out_offsets = Array<int>({0, fes.GetTrueVSize()});
   }

   void Mult(const Vector &x, Vector &y) const override
   { MultiVector xmv(x), ymv(y); MultMV(xmv, ymv); }

   void MultMV(const MultiVector &x, MultiVector &y) const override
   {
      const Vector &tdof = x[0];
      Vector &kdof = y[0];

      T_gf.SetFromTrueDofs(tdof);
      k_gf.ProjectCoefficient(k_coeff);
      k_gf.GetTrueDofs(kdof);
   }

   Operator& GetGradientMV(const MultiVector &x) const override
   { return gradient.GetGradientMV(x); }

   ~DiffusionCoefficient() override
   { }
};

/// An application that takes n input fields x_i, and computes an output 
/// field prod(x) := y = prod_i x_i.
class FieldProduct : public GraphOperator
{
class GradientOperator : public Operator
{
protected:
   Array<int> &in_offsets, &out_offsets;
   mutable ParGridFunction x_gf, dx_gf;
   ParGridFunction &y_gf;
   mutable Vector dfdx;
   mutable VectorProductDerivativeCoefficient ddx_coeff;

public:
   GradientOperator(Array<int> &in_off, Array<int> &out_off,
                    ParGridFunction &x, ParGridFunction &y) :
                    Operator(out_off.Last(), in_off.Last()),
                    in_offsets(in_off), out_offsets(out_off),
                    x_gf(x), dx_gf(x), y_gf(y), ddx_coeff(x_gf, dx_gf)
                    {
                     dx_gf = 0.0;
                     dx_gf.GetTrueDofs(dfdx);
                    }

   Operator& GetGradientMV(const MultiVector &x) const override
   {
      for (int i = 0; i < in_offsets.Size()-1; i++)
      { dfdx.SetVector(x[i], in_offsets[i]); } // Set all x_i
      x_gf.SetFromTrueDofs(dfdx);
      return const_cast<GradientOperator&>(*this);
   }
   void MultMV(const MultiVector &x, MultiVector &y) const override
   {
      Vector &y_dof = y[0];
      for (int i = 0; i < in_offsets.Size()-1; i++)
      { dfdx.SetVector(x[i], in_offsets[i]); } // Set all dx_i
      dx_gf.SetFromTrueDofs(dfdx);
      y_gf.ProjectCoefficient(ddx_coeff); // Compute sum_i (dx_i * d(prod)/dx_i)
      y_gf.GetTrueDofs(y_dof);
   }

   void Mult(const Vector &x, Vector &y) const override
   { MFEM_ABORT("Mult is not implemented for GradientOperator."); }
};
protected:
   int ninputs;
   ParFiniteElementSpace *nd_fes;
   Array<int> in_offsets, out_offsets;

   mutable Vector xdof;
   mutable ParGridFunction x_gf, y_gf;
   mutable VectorProductCoefficient prod_coeff;
   mutable GradientOperator *gradient;

public:
   FieldProduct(ParFiniteElementSpace &fes, int n) :
                GraphOperator(fes.GetTrueVSize(), fes.GetTrueVSize() * n), ninputs(n),
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
      gradient = new GradientOperator(in_offsets, out_offsets, x_gf, y_gf);
   }

   void Mult(const Vector &x, Vector &y) const override
   { MFEM_ABORT("Mult is not implemented for FieldProduct."); }

   void MultMV(const MultiVector &x, MultiVector &y) const override
   {
      for (int i = 0; i < ninputs; i++)
      { xdof.SetVector(x[i], in_offsets[i]); } // Set all x_i

      Vector &y_dof = y[0];
      x_gf.SetFromTrueDofs(xdof);
      y_gf.ProjectCoefficient(prod_coeff);
      y_gf.GetTrueDofs(y_dof);
   }

   Operator& GetGradientMV(const MultiVector &x) const override
   { return gradient->GetGradientMV(x); }

   ~FieldProduct() override
   {
      if(nd_fes) delete nd_fes;
      if(gradient) delete gradient;
   }
};


/// An application that represents the nonlinear diffusion operator: f(T) = -Div(k(u) grad(T)) 
/// with input field T and k, and output field f(T).
class DiffusionOperator : public GraphOperator
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
   BlockOperator *gradient = nullptr;

public:

   DiffusionOperator(ParFiniteElementSpace &fes_) :
                     // GraphOperator(fes_.GetTrueVSize()),
                     GraphOperator(fes_.GetTrueVSize(),2*fes_.GetTrueVSize()),
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
      gradient = new BlockOperator(out_offsets, in_offsets);
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
      b.SetSubVector(ess_tdofs, 0.0);
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
      Nform.Mult(tdofs, fdofs);
      fdofs += b; // Add the source term
      fdofs.SetSubVector(ess_tdofs, 0.0);
   }

   // Exact block jacobian [df/dT, df/dk]
   Operator& GetGradientMV(const MultiVector &x) const override
   {
      const Vector &tdofs = x[0];
      const Vector &kdofs = x[1];

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

      gradient->SetBlock(0, 0, dfdT_mat);
      gradient->SetBlock(0, 1, dfdk_mat);
      return *gradient;
   }

   /// @brief Destroy the DiffusionOperator object
   ~DiffusionOperator() override
   {
      if(dfdT_mat) delete dfdT_mat;
      if(dfdk_mat) delete dfdk_mat;
      if(gradient) delete gradient;
   }
};


int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");

   args.AddOption(&ctx.grad_mode, "-gm", "--grad-mode",
                  "Gradient mode for the coupled operator (0: finite difference, 1: algorithmic differentiation)");
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
   auto exp_func = [kref=1.0, a = 3.5e-1](real_t x, bool eval_f) -> real_t
   {
      real_t k = kref * exp(a*x);
      return eval_f ? k : a * k;
   };

   auto poly_func = [kref=1.0, a0=1.0, a1=2.0, a2=3.0](real_t x, bool eval_f) -> real_t
   {
      return eval_f ? kref * (a0 + a1*x + a2*x*x) : kref * (a1 + 2*a2*x);
   };

   DiffusionCoefficient diff_coeff_1(fes, exp_func);
   diff_coeff_1.SetName("k(T1)");

   DiffusionCoefficient diff_coeff_2(fes, poly_func);
   diff_coeff_2.SetName("k(T2)");

   FieldProduct prod_coeff(fes, 2);
   prod_coeff.SetName("k(T1,T2)");

   DiffusionOperator diff_op1(fes);
   diff_op1.SetName("Div(k(T1,T2) grad(T1))");

   DiffusionOperator diff_op2(fes);
   diff_op2.SetName("Div(k(T1,T2) grad(T2))");

   // Size and memory type for the fields
   int fsize = fes.GetTrueVSize();
   MemoryType mem_type = MemoryType::HOST;

   // Input fields get memory from 'x' in DAGraph::Mult(x, y)
   Field *T1_field = new VectorField(fsize, mem_type);
   Field *T2_field = new VectorField(fsize, mem_type);

   // Intermediate fields for the diffusion coefficients
   // Memory allocated internally in dag using Field::MakeNew()
   Field *k1_field = new VectorField(fsize, mem_type);
   Field *k2_field = new VectorField(fsize, mem_type);
   Field *kp_field = new VectorField(fsize, mem_type);; // Only needed for the coupled case

   // // Output fields get memory from 'y' in DAGraph::Mult(x, y)
   Field *f1_field = new VectorField(fsize, mem_type);
   Field *f2_field = new VectorField(fsize, mem_type);

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
  /* Force const if needed */     [&op=std::as_const(diff_op2)](const MultiVector &x, MultiVector &y) { op.MultMV(x, y); }
  /* For matrix-free grad_mult */ //[&op=diff_op2](const MultiVector &x, const MultiVector &dx, MultiVector &dy) { op.GradientMultMV(x, dx, dy); },
                              //   [&op=diff_op2](const MultiVector &x, const MultiVector &dx, MultiVector &dy) { op.GradientMultTransposeMV(x, dx, dy); }
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

   // Offsets needed to construct MultiVector from Vector in Mult()
   int sz = fes.GetTrueVSize();
   Array<int> dag_offsets({0, sz, 2*sz});
   dag.SetOffsets(dag_offsets, dag_offsets);

   // Assemble DAG: topological sort (if needed), validate nodes, etc.
   dag.Assemble();

   dag.SetGradientMode(static_cast<DAGraph::GradientMode>(ctx.grad_mode));

   std::string output_prefix = ctx.coupled ? "Coupled_Diffusion" : "Uncoupled_Diffusion";

   // Set initial guess and boundary conditions for T1 and T2
   Array<int> ess_tdofs;
   fes.GetBoundaryTrueDofs(ess_tdofs);

   BlockVector xb(dag_offsets);
   BlockVector yb(dag_offsets);
   xb = 0.0; yb = 0.0;

   xb.Randomize(10);
   xb.GetBlock(0).SetSubVector(ess_tdofs, 1.0);
   xb.GetBlock(1).SetSubVector(ess_tdofs, 1.0);

   // Build the nonlinear solver and linear solver for the DAG
   NewtonSolver newton_solver(pmesh.GetComm());
   GMRESSolver linear_solver(pmesh.GetComm());
   linear_solver.SetKDim(500);

   // Set the gradient mode for the DAG and solve the coupled system
   if(ctx.coupled)
   {
      SetSolverParameters(&newton_solver, ctx.tol_nsolve, 0.0, ctx.nl_iter, 1, true);
      SetSolverParameters(&linear_solver, ctx.tol_lsolve, 0.0, ctx.lin_iter, 1, false);

      newton_solver.SetPreconditioner(linear_solver);
      linear_solver.SetPrintLevel(1);
      newton_solver.SetOperator(dag);
      newton_solver.Mult(yb, xb);
   }
   else
   {
      SetSolverParameters(&linear_solver, ctx.tol_lsolve, 0.0, ctx.lin_iter, 1, true);
      linear_solver.SetOperator(dag);
      linear_solver.Mult(yb, xb);
   }

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
      ParGridFunction k1_gf(&fes);
      ParGridFunction k2_gf(&fes);
      Vector kvec(fes.GetTrueVSize());

      T1_gf.SetFromTrueDofs(xb.GetBlock(0));
      T2_gf.SetFromTrueDofs(xb.GetBlock(1));

      diff_coeff_1.Mult(xb.GetBlock(0), kvec);
      k1_gf.SetFromTrueDofs(kvec);

      diff_coeff_2.Mult(xb.GetBlock(1), kvec);
      k2_gf.SetFromTrueDofs(kvec);

      pv->RegisterField("T1", &T1_gf);
      pv->RegisterField("T2", &T2_gf);
      pv->RegisterField("k1", &k1_gf);
      pv->RegisterField("k2", &k2_gf);
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