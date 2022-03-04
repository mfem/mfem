//                        MFEM Example 42 - Parallel version
//
// Compile with: make ex42p
//

#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/tensor.hpp"
#include "mfem.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::internal;

/// This example only works in 3D.
constexpr int dimension = 3;

#ifdef MFEM_USE_ENZYME
/*
 * Variables prefixed with enzyme_* or function types prefixed with __enzyme_*,
 * are variables which will get preprocessed in the LLVM intermediate
 * representation when the Enzyme LLVM plugin is loaded. See the Enzyme
 * documentation (https://enzyme.mit.edu) for more information.
 */

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... Args>
return_type __enzyme_autodiff(Args...);

template <typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);
#endif

/// Define the identity tensor in three dimensions.
MFEM_HOST_DEVICE static auto I = Identity<dimension>();

/** @brief Linear elastic material.
 *
 * Defines a linear elastic material response. It satisfies the material_type
 * interface for ElasticityOperator::SetMaterial.
 */
template <int dim> struct LinearElasticMaterial
{
   /**
    * @brief Compute the stress response.
    *
    * @param dudx derivative of the displacement
    * @return tensor<double, dim, dim>
    */
   tensor<double, dim, dim>
   MFEM_HOST_DEVICE stress(const tensor<double, dim, dim> &dudx) const
   {
      auto epsilon = sym(dudx);
      return lambda * tr(epsilon) * I + 2.0 * mu * epsilon;
   }

   /**
    * @brief Apply the gradient of the stress.
    *
    */
   tensor<double, dim, dim> MFEM_HOST_DEVICE
   action_of_gradient(const tensor<double, dim, dim> & /* dudx */,
                      const tensor<double, dim, dim> &ddudx) const
   {
      return stress(ddudx);
   }

   /**
    * @brief Compute the gradient.
    *
    * This method is used in the ElasticityDiagonalPreconditioner type to
    * compute the gradient matrix entries of the current quadrature point,
    * instead of the action.
    *
    * @param dudx
    * @return tensor<double, dim, dim, dim, dim>
    */
   tensor<double, dim, dim, dim, dim>
   MFEM_HOST_DEVICE gradient(tensor<double, dim, dim> /* dudx */) const
   {
      return make_tensor<dim, dim, dim, dim>([&](int i, int j, int k, int l)
      {
         return lambda * (i == j) * (k == l) +
                mu * ((i == l) * (j == k) + (i == k) * (j == l));
      });
   }

   /// First Lame parameter
   double lambda = 100;
   /// Second Lame parameter
   double mu = 50;
};

enum class GradientType
{
   Symbolic,
   EnzymeFwd,
   EnzymeRev,
   FiniteDiff,
   DualNumbers
};

/**
 * @brief Neo-Hookean material
 *
 * Defines a Neo-Hookean material response. It satisfies the material_type
 * interface for ElasticityOperator::SetMaterial. This material type allows
 * choosing the method of derivative calculation in `action_of_gradient`.
 * Choices include methods derived by hand using symbolic calculation and a
 * variety of automatically computed gradient applications, like
 * - Enzyme forward mode
 * - Enzyme reverse mode
 * - Dual number type forward mode
 * - Finite difference mode
 *
 * @tparam dim
 * @tparam gradient_type
 */
template <int dim = 3, GradientType gradient_type = GradientType::Symbolic>
struct NeoHookeanMaterial
{
   static_assert(dim == 3, "NeoHookean model only defined in 3D");

   /**
    * @brief Compute the stress response.
    *
    * @param dudx derivative of the displacement
    * @return tensor<double, dim, dim>
    */
   template <typename T>
   MFEM_HOST_DEVICE tensor<T, dim, dim>
   stress(const tensor<T, dim, dim> &__restrict__ dudx) const
   {
      T J = det(I + dudx);
      T p = -2.0 * D1 * J * (J - 1);
      auto devB = dev(dudx + transpose(dudx) + dot(dudx, transpose(dudx)));
      auto sigma = -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
      return sigma;
   }

   /**
    * @brief A method to wrap the stress calculation into a static function.
    *
    * This is necessary for Enzyme to access the class pointer (self).
    *
    * @param self
    * @param dudx
    * @param sigma
    * @return MFEM_HOST_DEVICE
    */
   MFEM_HOST_DEVICE static void
   stress_wrapper(NeoHookeanMaterial<dim, gradient_type> *self,
                  tensor<double, dim, dim> &dudx,
                  tensor<double, dim, dim> &sigma)
   {
      sigma = self->stress(dudx);
   }

   /**
    * @brief Compute the gradient.
    *
    * This method is used in the ElasticityDiagonalPreconditioner type to
    * compute the gradient matrix entries of the current quadrature point,
    * instead of the action.
    *
    * @param dudx
    * @return tensor<double, dim, dim, dim, dim>
    */
   MFEM_HOST_DEVICE tensor<double, dim, dim, dim, dim>
   gradient(tensor<double, dim, dim> dudx) const
   {
      tensor<double, dim, dim> F = I + dudx;
      tensor<double, dim, dim> invF = inv(F);
      tensor<double, dim, dim> devB =
         dev(dudx + transpose(dudx) + dot(dudx, transpose(dudx)));
      double J = det(F);
      double coef = (C1 / pow(J, 5.0 / 3.0));
      return make_tensor<dim, dim, dim, dim>([&](int i, int j, int k, int l)
      {
         return 2.0 * (D1 * J * (i == j) - (5.0 / 3.0) * coef * devB[i][j]) *
                invF[l][k] +
                2.0 * coef *
                ((i == k) * F[j][l] + F[i][l] * (j == k) -
                 (2.0 / 3.0) * ((i == j) * F[k][l]));
      });
   }

   /**
    * @brief Apply the gradient of the stress.
    *
    */
   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient(const tensor<double, dim, dim> &dudx,
                      const tensor<double, dim, dim> &ddudx) const
   {
      if (gradient_type == GradientType::Symbolic)
      {
         return action_of_gradient_symbolic(dudx, ddudx);
      }
#ifdef MFEM_USE_ENZYME
      else if (gradient_type == GradientType::EnzymeFwd)
      {
         return action_of_gradient_enzyme_fwd(dudx, ddudx);
      }
      else if (gradient_type == GradientType::EnzymeRev)
      {
         return action_of_gradient_enzyme_rev(dudx, ddudx);
      }
#endif
      else if (gradient_type == GradientType::FiniteDiff)
      {
         return action_of_gradient_fd(dudx, ddudx);
      }
      else if (gradient_type == GradientType::DualNumbers)
      {
         return action_of_gradient_dual(dudx, ddudx);
      }
   }

   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_dual(const tensor<double, dim, dim> &dudx,
                           const tensor<double, dim, dim> &ddudx) const
   {
      auto sigma = stress(make_tensor<dim, dim>([&](int i, int j)
      {
         return dual<double, double> {dudx[i][j], ddudx[i][j]};
      }));
      return make_tensor<dim, dim>(
      [&](int i, int j) { return sigma[i][j].gradient; });
   }

#ifdef MFEM_USE_ENZYME
   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_enzyme_fwd(const tensor<double, dim, dim> &dudx,
                                 const tensor<double, dim, dim> &ddudx) const
   {
      tensor<double, dim, dim> sigma{};
      tensor<double, dim, dim> dsigma{};

      __enzyme_fwddiff<void>(stress_wrapper, enzyme_const, this, enzyme_dup,
                             &dudx, &ddudx, enzyme_dupnoneed, &sigma, &dsigma);
      return dsigma;
   }

   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_enzyme_rev(const tensor<double, dim, dim> &dudx,
                                 const tensor<double, dim, dim> &ddudx) const
   {
      tensor<double, dim, dim, dim, dim> gradient{};
      tensor<double, dim, dim> sigma{};
      tensor<double, dim, dim> dir{};

      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            dir[i][j] = 1;
            __enzyme_autodiff<void>(stress_wrapper, enzyme_const, this, enzyme_dup,
                                    &dudx, &gradient[i][j], enzyme_dupnoneed,
                                    &sigma, &dir);
            dir[i][j] = 0;
         }
      }
      return ddot(gradient, ddudx);
   }
#endif

   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_fd(const tensor<double, dim, dim> &dudx,
                         const tensor<double, dim, dim> &ddudx) const
   {
      return (stress(dudx + 1.0e-8 * ddudx) - stress(dudx - 1.0e-8 * ddudx)) /
             2.0e-8;
   }

   // d(stress)_{ij} := (d(stress)_ij / d(du_dx)_{kl}) * d(du_dx)_{kl}
   // Only works with 3D stress
   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_symbolic(const tensor<double, dim, dim> &du_dx,
                               const tensor<double, dim, dim> &ddu_dx) const
   {
      tensor<double, dim, dim> F = I + du_dx;
      tensor<double, dim, dim> invFT = inv(transpose(F));
      tensor<double, dim, dim> devB =
         dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
      double J = det(F);
      double coef = (C1 / pow(J, 5.0 / 3.0));
      double a1 = ddot(invFT, ddu_dx);
      double a2 = ddot(F, ddu_dx);

      return (2.0 * D1 * J * a1 - (4.0 / 3.0) * coef * a2) * I -
             ((10.0 / 3.0) * coef * a1) * devB +
             (2 * coef) * (dot(ddu_dx, transpose(F)) + dot(F, transpose(ddu_dx)));
   }

   // Parameters
   double D1 = 100.0;
   double C1 = 50.0;
};

class ElasticityGradientOperator;

//
class ElasticityOperator : public Operator
{
public:
   ElasticityOperator(ParMesh &mesh, const int order);

   /**
    * @brief Compute the residual Y = R(U) representing the elasticity equation
    * with a material model chosen by calling SetMaterial.
    *
    * The output vector @a Y has essential degrees of freedom applied by setting
    * them to zero. This ensures R(U)_i = 0 being satisfied for each essential
    * dof i.
    *
    * @param U U
    * @param Y Residual R(U)
    */
   virtual void Mult(const Vector &U, Vector &Y) const override;

   /**
    * @brief Get the Gradient object
    *
    * Update and cache the state vector @a U, used to compute the linearization
    * dR(U)/dU.
    *
    * @param U
    * @return Operator&
    */
   Operator &GetGradient(const Vector &U) const override;

   /**
    * @brief Multiply the linearization of the residual R(U) wrt to the current
    * state U by a perturbation @a dX.
    *
    * Y = dR(U)/dU * dX = K(U) dX
    *
    * @param dX
    * @param Y
    */
   void GradientMult(const Vector &dX, Vector &Y) const;

   /**
    * @brief Assemble the linearization of the residual K = dR(U)/dU.
    *
    * This method needs three input vectors which also act as output vectors.
    * They don't have to be the right size on the first call, but it is advised
    * that memory is kept alive during successive call. The data layout of the
    * outputs will be
    *
    * @a Ke_diag: #dofs x #dofs x #dofs x dim x ne x dim
    *
    * @a K_diag_local: width(H1_Restriction) x dim
    *
    * @a K_diag: width(H1_Prolongation) x dim
    *
    * This data layout is needed due to the Ordering::byNODES. See method
    * implementation comments for more details. The output @a K_diag has
    * modified entries when essential boundaries are defined. Each essential dof
    * row and column are set to zero with it's diagonal entry set to 1.
    *
    * @param Ke_diag
    * @param K_diag_local
    * @param K_diag
    */
   void AssembleGradientDiagonal(Vector &Ke_diag, Vector &K_diag_local,
                                 Vector &K_diag) const;

   ~ElasticityOperator();

   ParMesh &mesh_;
   /// Polynomial order of the FE space
   const int order_;
   const int dim_;
   const int vdim_;
   /// Number of elements in the mesh (rank local)
   const int ne_;
   /// H1 finite element collection
   H1_FECollection h1_fec_;
   ParFiniteElementSpace h1_fes_;
   IntegrationRule *ir_ = nullptr;
   /// Number of degrees of freedom in 1D
   int d1d_;
   /// Number of quadrature points in 1D
   int q1d_;
   const Operator *h1_element_restriction_;
   const Operator *h1_prolongation_;
   Array<int> ess_tdof_list_;
   Array<int> displaced_tdof_list_;
   ElasticityGradientOperator *gradient_;
   const GeometricFactors *geometric_factors_;
   const DofToQuad *maps_;
   /// Input state L-vector
   mutable Vector X_local_;
   /// Input state E-vector
   mutable Vector X_el_;
   /// Output state L-vector
   mutable Vector Y_local_;
   /// Output state E-Vector
   mutable Vector Y_el_;
   /// Cached current state. Used to determine the state on which to compute the
   /// linearization on during the Newton method.
   mutable Vector current_state;
   mutable Vector cstate_local_;
   mutable Vector cstate_el_;
   /// Temporary vector for the pertubation of the solution with essential
   /// boundaries eliminated. Defined as a T-vector.
   mutable Vector dX_ess_;

   /**
    * @brief Wrapper for the application of the residual R(U).
    *
    * The wrapper is used in SetMaterial to instantiate the chosen kernel and
    * erase the material type kernel. This is purely an interface design choice
    * and could be replaced by an abstract base class for the material including
    * virtual function calls.
    */
   std::function<void(const int, const Array<double> &, const Array<double> &,
                      const Array<double> &, const Vector &, const Vector &,
                      const Vector &, Vector &)>
   element_apply_kernel_wrapper;

   /**
    * @brief Wrapper for the application of the gradient of the residual
    *
    *  K(U) dX = dR(U)/dU dX
    */
   std::function<void(const int, const Array<double> &, const Array<double> &,
                      const Array<double> &, const Vector &, const Vector &,
                      const Vector &, Vector &, const Vector &)>
   element_apply_gradient_kernel_wrapper;

   /**
    * @brief Wrapper for the assembly of the gradient on each diagonal element
    *
    * Ke_ii(U) = dRe_ii(U)/dU
    */
   std::function<void(const int, const Array<double> &, const Array<double> &,
                      const Array<double> &, const Vector &, const Vector &,
                      const Vector &, Vector &)>
   element_kernel_assemble_diagonal_wrapper;

   /**
    * @brief Set the material type.
    *
    * This method sets the material type by instantiating the kernels with a
    * material_type object.
    *
    * @tparam material_type
    * @param material
    */
   template <typename material_type>
   void SetMaterial(const material_type &material);

   /**
    * @brief Set the essential attributes which mark degrees of freedom for the
    * solving process.
    *
    * Can be either a fixed boundary or a prescribed displacement.
    *
    * @param attr
    */
   void SetEssentialAttributes(const Array<int> attr)
   {
      h1_fes_.GetEssentialTrueDofs(attr, ess_tdof_list_);
   }

   /**
    * @brief Set the attributes which mark the degrees of freedom that have a
    * fixed displacement.
    *
    * @param attr
    */
   void SetDisplacedAttributes(const Array<int> attr)
   {
      h1_fes_.GetEssentialTrueDofs(attr, displaced_tdof_list_);
   }

   /**
    * @brief Return the T-vector degrees of freedom that have been marked as
    * displaced.
    *
    * @return const Array<int>&
    */
   const Array<int> &GetDisplacedTDofs() { return displaced_tdof_list_; };
};

/**
 * @brief ElasticityGradientOperator is a wrapper class to pass
 * ElasticityOperator::AssembleGradientDiagonal and
 * ElasticityOperator::GradientMult as a separate object through NewtonSolver.
 */
class ElasticityGradientOperator : public Operator
{
public:
   ElasticityGradientOperator(ElasticityOperator &op);

   void AssembleGradientDiagonal(Vector &Ke_diag, Vector &K_diag_local,
                                 Vector &K_diag) const
   {
      elasticity_op_.AssembleGradientDiagonal(Ke_diag, K_diag_local, K_diag);
   }

   void Mult(const Vector &x, Vector &y) const override;

   ElasticityOperator &elasticity_op_;
};

/**
 * @brief ElasticityDiagonalPreconditioner acts as a matrix-free preconditioner
 * for ElasticityOperator.
 *
 * @note There are two types to choose from
 * - Diagonal: A classic Jacobi type preconditioner
 * - BlockDiagonal: A Jacobi type preconditioner which calculates the diagonal
 *   contribution of ElasticityOperator on each diagonal element and applies
 *   it's inverted submatrix.
 */
class ElasticityDiagonalPreconditioner : public Solver
{
public:
   static constexpr int dim = dimension;

   enum Type { Diagonal, BlockDiagonal };

   ElasticityDiagonalPreconditioner(Type type = Type::Diagonal)
      : Solver(), type_(type) {}

   void SetOperator(const Operator &op) override
   {
      gradient_operator_ = dynamic_cast<const ElasticityGradientOperator *>(&op);
      MFEM_ASSERT(gradient_operator_ != nullptr,
                  "Operator is not ElasticityGradientOperator");

      width = height = op.Height();

      gradient_operator_->AssembleGradientDiagonal(Ke_diag_, K_diag_local_,
                                                   K_diag_);

      submat_height_ = gradient_operator_->elasticity_op_.h1_fes_.GetVDim();
      num_submats_ = gradient_operator_->elasticity_op_.h1_fes_.GetTrueVSize() /
                     gradient_operator_->elasticity_op_.h1_fes_.GetVDim();
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      if (type_ == Type::Diagonal)
      {
         x.HostRead();
         y.HostReadWrite();
         auto K_diag_submats =
            Reshape(K_diag_.HostRead(), num_submats_, submat_height_, submat_height_);
         // TODO: This could be MFEM_FORALL
         // Assuming Y and X are ordered byNODES. K_diag is ordered byVDIM.
         for (int s = 0; s < num_submats_; s++)
         {
            for (int i = 0; i < submat_height_; i++)
            {
               int idx = s + i * num_submats_;
               y(idx) = x(idx) / K_diag_submats(s, i, i);
            }
         }
      }
      else if (type_ == Type::BlockDiagonal)
      {
         auto K_diag_submats =
            Reshape(K_diag_.Read(), num_submats_, submat_height_, submat_height_);

         for (int s = 0; s < num_submats_; s++)
         {
            auto submat_inv = inv(make_tensor<dim, dim>(
            [&](int i, int j) { return K_diag_submats(s, i, j); }));

            auto x_block =
            make_tensor<dim>([&](int i) { return x(s + i * num_submats_); });

            tensor<double, dim> y_block;

            y_block = submat_inv * x_block;

            for (int i = 0; i < dim; i++)
            {
               int idx = s + i * num_submats_;
               y(idx) = y_block(i);
            }
         }
      }
      else
      {
         MFEM_ABORT("Unknwon ElasticityDiagonalPreconditioner::Type");
      }
   }

private:
   const ElasticityGradientOperator *gradient_operator_;
   int num_submats_, submat_height_;
   Vector Ke_diag_, K_diag_local_, K_diag_;
   Type type_;
};

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int myid = mpi.WorldRank();

   int order = 1;
   const char *device_config = "cpu";
   int diagpc_type = ElasticityDiagonalPreconditioner::Type::Diagonal;
   int serial_refinement_levels = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&diagpc_type, "-pc", "--pctype",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&serial_refinement_levels, "-rs", "--ref-serial",
                  "Number of uniform refinements on the serial mesh.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   if (myid == 0)
   {
      device.Print();
   }

   auto mesh =
      Mesh::MakeCartesian3D(8, 1, 1, Element::HEXAHEDRON, 8.0, 1.0, 1.0);
   if (mesh.Dimension() != 3)
   {
      MFEM_ABORT("This example only works in 3D.");
   }
   mesh.EnsureNodes();

   for (int l = 0; l < serial_refinement_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Create the elasticity operator on the parallel mesh.
   ElasticityOperator elasticity_op(pmesh, order);

   // Create and set the material type. We define it's GradientType during
   // instantiation.
   const NeoHookeanMaterial<dimension, GradientType::DualNumbers> material{};
   elasticity_op.SetMaterial(material);

   // Define all essential boundaries. In this specific example, this includes
   // all fixed and statically displaced degrees of freedom on mesh entities in
   // the defined attributes.
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_attr(pmesh.bdr_attributes.Max());
      ess_attr = 0;
      ess_attr[4] = 1;
      ess_attr[2] = 1;
      elasticity_op.SetEssentialAttributes(ess_attr);
   }

   // Define all statically displaced mesh attributes. On these degrees of
   // freedom (determined from the mesh attributes), a fixed displacement is
   // prescribed.
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> displaced_attr(pmesh.bdr_attributes.Max());
      displaced_attr = 0;
      displaced_attr[2] = 1;
      elasticity_op.SetDisplacedAttributes(displaced_attr);
   }

   ParGridFunction U_gf(&elasticity_op.h1_fes_);
   U_gf = 0.0;

   Vector U;
   U_gf.GetTrueDofs(U);

   // Prescribe a fixed displacement to the displaced degrees of freedom.
   U.SetSubVector(elasticity_op.GetDisplacedTDofs(), 1.0e-2);

   // Define the type of preconditioner to use for the linear solver.
   ElasticityDiagonalPreconditioner diagonal_pc(
      static_cast<ElasticityDiagonalPreconditioner::Type>(diagpc_type));

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-1);
   cg.SetMaxIter(10000);
   cg.SetPrintLevel(2);
   cg.SetPreconditioner(diagonal_pc);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetSolver(cg);
   newton.SetOperator(elasticity_op);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(10);
   newton.SetPrintLevel(1);

   Vector zero;
   newton.Mult(zero, U);

   U_gf.Distribute(U);

   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("ex42_output", &pmesh);
   pd->RegisterField("solution", &U_gf);
   pd->SetLevelsOfDetail(order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();

   delete pd;

   return 0;
}

namespace KernelHelpers
{

// Kernel helper functions
void CheckMemoryRestriction(int d1d, int q1d)
{
   MFEM_VERIFY(d1d <= MAX_D1D,
               "Maximum number of degrees of freedom in 1D reached."
               "This number can be increased globally in general/forall.hpp if "
               "device memory allows.");
   MFEM_VERIFY(q1d <= MAX_Q1D, "Maximum quadrature points 1D reached."
               "This number can be increased globally in "
               "general/forall.hpp if device memory allows.");
}

/**
 * @brief Multi-component gradient evaluation from DOFs to quadrature points in
 * reference coordinates.
 *
 * @note DeviceTensor<2> means RANK=2
 * TODO: Does not make use of shared memory on the GPU.
 *
 * @tparam dim
 * @tparam d1d
 * @tparam q1d
 * @param B
 * @param G
 * @param U
 * @param dUdxi
 */
template <int dim, int d1d, int q1d>
static inline void MFEM_HOST_DEVICE
CalcGrad(const DeviceTensor<2, const double> &B, // q1d x d1d
         const DeviceTensor<2, const double> &G, // q1d x d1d
         const DeviceTensor<4, const double> &U, // d1d x d1d x d1d x dim
         tensor<double, q1d, q1d, q1d, dim, dim> &dUdxi)
{
   for (int c = 0; c < dim; ++c)
   {
      for (int dz = 0; dz < d1d; ++dz)
      {
         tensor<double, q1d, q1d, dim> gradXY{};
         for (int dy = 0; dy < d1d; ++dy)
         {
            tensor<double, q1d, 2> gradX{};
            for (int dx = 0; dx < d1d; ++dx)
            {
               const double s = U(dx, dy, dz, c);
               for (int qx = 0; qx < q1d; ++qx)
               {
                  gradX[qx][0] += s * B(qx, dx);
                  gradX[qx][1] += s * G(qx, dx);
               }
            }
            for (int qy = 0; qy < q1d; ++qy)
            {
               const double wy = B(qy, dy);
               const double wDy = G(qy, dy);
               for (int qx = 0; qx < q1d; ++qx)
               {
                  const double wx = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx * wDy;
                  gradXY[qy][qx][2] += wx * wy;
               }
            }
         }
         for (int qz = 0; qz < q1d; ++qz)
         {
            const double wz = B(qz, dz);
            const double wDz = G(qz, dz);
            for (int qy = 0; qy < q1d; ++qy)
            {
               for (int qx = 0; qx < q1d; ++qx)
               {
                  dUdxi[qz][qy][qx][c][0] += gradXY[qy][qx][0] * wz;
                  dUdxi[qz][qy][qx][c][1] += gradXY[qy][qx][1] * wz;
                  dUdxi[qz][qy][qx][c][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }
   }
}

/**
 * @brief Multi-component transpose gradient evaluation from DOFs to quadrature
 * points in reference coordinates with contraction of the D vector.
 *
 * @note TODO: Does not make use of shared memory on the GPU.
 *
 * @tparam dim
 * @tparam d1d
 * @tparam q1d
 * @param B
 * @param G
 * @param U
 * @param F
 */
template <int dim, int d1d, int q1d>
static inline void MFEM_HOST_DEVICE CalcGradTSum(
   const DeviceTensor<2, const double> &B,           // q1d x d1d
   const DeviceTensor<2, const double> &G,           // q1d x d1d
   const tensor<double, q1d, q1d, q1d, dim, dim> &U, // q1d x q1d x q1d x dim
   DeviceTensor<4, double> &F)                       // d1d x d1d x d1d x dim
{
   for (int c = 0; c < dim; ++c)
   {
      for (int qz = 0; qz < q1d; ++qz)
      {
         tensor<double, d1d, d1d, dim> gradXY{};
         for (int qy = 0; qy < q1d; ++qy)
         {
            tensor<double, d1d, dim> gradX{};
            for (int qx = 0; qx < q1d; ++qx)
            {
               const double gX = U[qx][qy][qz][0][c];
               const double gY = U[qx][qy][qz][1][c];
               const double gZ = U[qx][qy][qz][2][c];
               for (int dx = 0; dx < d1d; ++dx)
               {
                  const double wx = B(qx, dx);
                  const double wDx = G(qx, dx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }
            for (int dy = 0; dy < d1d; ++dy)
            {
               const double wy = B(qy, dy);
               const double wDy = G(qy, dy);
               for (int dx = 0; dx < d1d; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }
         for (int dz = 0; dz < d1d; ++dz)
         {
            const double wz = B(qz, dz);
            const double wDz = G(qz, dz);
            for (int dy = 0; dy < d1d; ++dy)
            {
               for (int dx = 0; dx < d1d; ++dx)
               {
                  F(dx, dy, dz, c) +=
                     ((gradXY[dy][dx][0] * wz) + (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }
   }
}

/**
 * @brief Compute the gradient of all shape functions.
 *
 * @note TODO: Does not make use of shared memory on the GPU.
 *
 * @tparam dim
 * @tparam d1d
 * @tparam q1d
 * @param qx
 * @param qy
 * @param qz
 * @param B
 * @param G
 * @param invJ
 */
template <int dim, int d1d, int q1d>
static inline MFEM_HOST_DEVICE tensor<double, d1d, d1d, d1d, dim>
GradAllPhis(int qx, int qy, int qz, const DeviceTensor<2, const double> &B,
            const DeviceTensor<2, const double> &G,
            const tensor<double, dim, dim> &invJ)
{
   MFEM_SHARED tensor<double, d1d, d1d, d1d, dim> dphi_dx;
   // G (x) B (x) B
   // B (x) G (x) B
   // B (x) B (x) G
   MFEM_FOREACH_THREAD(dx, x, d1d)
   {
      MFEM_FOREACH_THREAD(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD(dz, z, d1d)
         {

            dphi_dx[dx][dy][dz] =
               transpose(invJ) *
               tensor<double, dim> {G(qx, dx) * B(qy, dy) * B(qz, dz),
                                    B(qx, dx) * G(qy, dy) * B(qz, dz),
                                    B(qx, dx) * B(qy, dy) * G(qz, dz)
                                   };
         }
      }
   }
   MFEM_SYNC_THREAD;
   return dphi_dx;
}
} // namespace KernelHelpers

namespace ElasticityKernels
{
template <int d1d, int q1d, typename material_type>
static inline void
Apply3D(const int ne, const Array<double> &B_, const Array<double> &G_,
        const Array<double> &W_, const Vector &Jacobian_, const Vector &detJ_,
        const Vector &X_, Vector &Y_, const material_type &material)
{
   constexpr int dim = dimension;
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   // 1D Basis functions in column-major layout
   // q1d x d1d
   const auto B = Reshape(B_.Read(), q1d, d1d);
   // Gradients of 1D basis functions evaluated at quadrature points in
   // column-major layout q1d x d1d
   const auto G = Reshape(G_.Read(), q1d, d1d);
   const auto qweights = Reshape(W_.Read(), q1d, q1d, q1d);
   // Jacobians of the element transformations at all quadrature points in
   // column-major layout q1d x q1d x q1d x sdim x dim x ne
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, q1d, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto U = Reshape(X_.Read(), d1d, d1d, d1d, dim, ne);
   // Output vector
   // d1d x d1d x d1d x vdim x ne
   auto force = Reshape(Y_.ReadWrite(), d1d, d1d, d1d, dim, ne);

   MFEM_FORALL_3D(e, ne, q1d, q1d, q1d,
   {
      // cauchy stress
      MFEM_SHARED tensor<double, q1d, q1d, q1d, dim, dim> invJ_sigma_detJw;

      // du/dxi
      tensor<double, q1d, q1d, q1d, dim, dim> dudxi{};
      const auto U_el = Reshape(&U(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad<dim, d1d, q1d>(B, G, U_el, dudxi);

      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               auto invJqp = inv(make_tensor<dim, dim>(
               [&](int i, int j) { return J(qx, qy, qz, i, j, e); }));

               auto dudx = dudxi(qz, qy, qx) * invJqp;

               auto sigma = material.stress(dudx);

               invJ_sigma_detJw(qx, qy, qz) =
                  invJqp * sigma * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
            }
         }
      }
      MFEM_SYNC_THREAD;
      auto F = Reshape(&force(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGradTSum<dim, d1d, q1d>(B, G, invJ_sigma_detJw, F);
   }); // for each element
}

template <int d1d, int q1d, typename material_type>
static inline void
ApplyGradient3D(const int ne, const Array<double> &B_, const Array<double> &G_,
                const Array<double> &W_, const Vector &Jacobian_,
                const Vector &detJ_, const Vector &dU_, Vector &dF_,
                const Vector &U_, const material_type &material)
{
   constexpr int dim = dimension;
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   // 1D Basis functions in column-major layout
   // q1d x d1d
   const auto B = Reshape(B_.Read(), q1d, d1d);
   // Gradients of 1D basis functions evaluated at quadrature points in
   // column-major layout q1d x d1d
   const auto G = Reshape(G_.Read(), q1d, d1d);
   const auto qweights = Reshape(W_.Read(), q1d, q1d, q1d);
   // Jacobians of the element transformations at all quadrature points in
   // column-major layout q1d x q1d x q1d x sdim x dim x ne
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, q1d, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto dU = Reshape(dU_.Read(), d1d, d1d, d1d, dim, ne);
   // Output vector
   // d1d x d1d x d1d x vdim x ne
   auto force = Reshape(dF_.ReadWrite(), d1d, d1d, d1d, dim, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto U = Reshape(U_.Read(), d1d, d1d, d1d, dim, ne);

   MFEM_FORALL_3D(e, ne, q1d, q1d, q1d,
   {
      // cauchy stress
      MFEM_SHARED tensor<double, q1d, q1d, q1d, dim, dim> invJ_dsigma_detJw;

      // du/dxi
      tensor<double, q1d, q1d, q1d, dim, dim> dudxi{};
      const auto U_el = Reshape(&U(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad<dim, d1d, q1d>(B, G, U_el, dudxi);

      // ddu/dxi
      tensor<double, q1d, q1d, q1d, dim, dim> ddudxi{};
      const auto dU_el = Reshape(&dU(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad<dim, d1d, q1d>(B, G, dU_el, ddudxi);

      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               auto invJqp = inv(make_tensor<dim, dim>(
               [&](int i, int j) { return J(qx, qy, qz, i, j, e); }));

               auto dudx = dudxi(qz, qy, qx) * invJqp;
               auto ddudx = ddudxi(qz, qy, qx) * invJqp;

               auto dsigma = material.action_of_gradient(dudx, ddudx);

               invJ_dsigma_detJw(qx, qy, qz) =
                  invJqp * dsigma * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
            }
         }
      }
      MFEM_SYNC_THREAD;
      auto F = Reshape(&force(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGradTSum<dim, d1d, q1d>(B, G, invJ_dsigma_detJw, F);
   }); // for each element
}

template <int d1d, int q1d, typename material_type>
static inline void AssembleGradientDiagonal3D(
   const int ne, const Array<double> &B_, const Array<double> &G_,
   const Array<double> &W_, const Vector &Jacobian_, const Vector &detJ_,
   const Vector &X_, Vector &Ke_diag_memory, const material_type &material)
{
   constexpr int dim = dimension;
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   // 1D Basis functions in column-major layout
   // q1d x d1d
   const auto B = Reshape(B_.Read(), q1d, d1d);
   // Gradients of 1D basis functions evaluated at quadrature points in
   // column-major layout q1d x d1d
   const auto G = Reshape(G_.Read(), q1d, d1d);
   const auto qweights = Reshape(W_.Read(), q1d, q1d, q1d);
   // Jacobians of the element transformations at all quadrature points. This
   // array uses a column-major layout
   // (q1d x q1d x q1d x sdim x dim x ne)
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, q1d, ne);
   // Input vector
   // d1d x d1d x d1d x vdim x ne
   const auto U = Reshape(X_.Read(), d1d, d1d, d1d, dim, ne);
   // Output vector
   // d1d x d1d x d1d x vdim x ne
   auto Ke_diag_m =
      Reshape(Ke_diag_memory.ReadWrite(), d1d, d1d, d1d, dim, ne, dim);

   MFEM_FORALL(e, ne,
   {
      tensor<double, d1d, d1d, d1d, dim, dim> Ke_diag{};

      // du/dxi
      tensor<double, q1d, q1d, q1d, dim, dim> dudxi{};
      const auto U_el = Reshape(&U(0, 0, 0, 0, e), d1d, d1d, d1d, dim);
      KernelHelpers::CalcGrad<dim, d1d, q1d>(B, G, U_el, dudxi);

      for (int qx = 0; qx < q1d; qx++)
      {
         for (int qy = 0; qy < q1d; qy++)
         {
            for (int qz = 0; qz < q1d; qz++)
            {
               auto invJqp = inv(make_tensor<dim, dim>(
               [&](int i, int j) { return J(qx, qy, qz, i, j, e); }));

               auto dudx = dudxi(qz, qy, qx) * invJqp;

               auto dsigma_ddudx = material.gradient(dudx);

               double JxW = detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
               auto dphidx = KernelHelpers::GradAllPhis<dim, d1d, q1d>(qx, qy, qz, B,
                                                                       G, invJqp);

               for (int dx = 0; dx < d1d; dx++)
               {
                  for (int dy = 0; dy < d1d; dy++)
                  {
                     for (int dz = 0; dz < d1d; dz++)
                     {
                        // phi_i * f(...) * phi_i
                        // dphidx_i dsigma_ddudx_ijkl dphidx_l
                        Ke_diag[dx][dy][dz] +=
                           (dphidx[dx][dy][dz] * dsigma_ddudx * dphidx[dx][dy][dz]) *
                           JxW;
                     }
                  }
               }
            }
         }
      }
      for (int i = 0; i < d1d; i++)
      {
         for (int j = 0; j < d1d; j++)
         {
            for (int k = 0; k < d1d; k++)
            {
               for (int l = 0; l < dim; l++)
               {
                  for (int m = 0; m < dim; m++)
                  {
                     Ke_diag_m(i, j, k, l, e, m) = Ke_diag[i][j][k][l][m];
                  }
               }
            }
         }
      }
   }); // for each element
}
} // namespace ElasticityKernels

ElasticityGradientOperator::ElasticityGradientOperator(ElasticityOperator &op)
   : Operator(op.Height()), elasticity_op_(op) {}

void ElasticityGradientOperator::Mult(const Vector &x, Vector &y) const
{
   elasticity_op_.GradientMult(x, y);
}

ElasticityOperator::ElasticityOperator(ParMesh &mesh, const int order)
   : Operator(), mesh_(mesh), order_(order), dim_(mesh_.SpaceDimension()),
     vdim_(mesh_.SpaceDimension()), ne_(mesh_.GetNE()), h1_fec_(order_, dim_),
     h1_fes_(&mesh_, &h1_fec_, vdim_, Ordering::byNODES)
{
   this->height = h1_fes_.GetTrueVSize();
   this->width = this->height;

   int global_tdof_size = h1_fes_.GlobalTrueVSize();
   if (mesh.GetMyRank() == 0)
   {
      cout << "#dofs: " << global_tdof_size << endl;
   }

   h1_element_restriction_ =
      h1_fes_.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   h1_prolongation_ = h1_fes_.GetProlongationMatrix();

   ir_ = const_cast<IntegrationRule *>(
            &IntRules.Get(mfem::Element::HEXAHEDRON, 2 * h1_fes_.GetOrder(0) + 1));

   geometric_factors_ = h1_fes_.GetParMesh()->GetGeometricFactors(
                           *ir_, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
   maps_ = &h1_fes_.GetFE(0)->GetDofToQuad(*ir_, DofToQuad::TENSOR);
   d1d_ = maps_->ndof;
   q1d_ = maps_->nqpt;

   dX_ess_.UseDevice(true);
   dX_ess_.SetSize(h1_fes_.GetTrueVSize());

   X_el_.UseDevice(true);
   X_el_.SetSize(h1_element_restriction_->Height());

   Y_el_.UseDevice(true);
   Y_el_.SetSize(h1_element_restriction_->Height());

   cstate_el_.UseDevice(true);
   cstate_el_.SetSize(h1_element_restriction_->Height());

   X_local_.UseDevice(true);
   X_local_.SetSize(h1_prolongation_->Height());

   Y_local_.UseDevice(true);
   Y_local_.SetSize(h1_prolongation_->Height());

   cstate_local_.UseDevice(true);
   cstate_local_.SetSize(h1_prolongation_->Height());

   gradient_ = new ElasticityGradientOperator(*this);
}

void ElasticityOperator::Mult(const Vector &X, Vector &Y) const
{
   ess_tdof_list_.Read();

   // T-vector to L-vector
   h1_prolongation_->Mult(X, X_local_);
   // L-vector to E-vector
   h1_element_restriction_->Mult(X_local_, X_el_);

   // Reset output vector
   Y_el_ = 0.0;

   // Apply operator
   element_apply_kernel_wrapper(ne_, maps_->B, maps_->G, ir_->GetWeights(),
                                geometric_factors_->J, geometric_factors_->detJ,
                                X_el_, Y_el_);

   // E-vector to L-vector
   h1_element_restriction_->MultTranspose(Y_el_, Y_local_);
   // L-vector to T-vector
   h1_prolongation_->MultTranspose(Y_local_, Y);

   // Set the residual at Dirichlet dofs on the T-vector to zero
   Y.SetSubVector(ess_tdof_list_, 0.0);
}

Operator &ElasticityOperator::GetGradient(const Vector &x) const
{
   h1_prolongation_->Mult(x, cstate_local_);
   h1_element_restriction_->Mult(cstate_local_, cstate_el_);
   return *gradient_;
}

void ElasticityOperator::GradientMult(const Vector &dX, Vector &Y) const
{
   ess_tdof_list_.Read();

   // Column elimination for essential dofs
   dX_ess_ = dX;
   dX_ess_.SetSubVector(ess_tdof_list_, 0.0);

   // T-vector to L-vector
   h1_prolongation_->Mult(dX_ess_, X_local_);
   // L-vector to E-vector
   h1_element_restriction_->Mult(X_local_, X_el_);

   // Reset output vector
   Y_el_ = 0.0;

   // Apply operator
   element_apply_gradient_kernel_wrapper(
      ne_, maps_->B, maps_->G, ir_->GetWeights(), geometric_factors_->J,
      geometric_factors_->detJ, X_el_, Y_el_, cstate_el_);

   // E-vector to L-vector
   h1_element_restriction_->MultTranspose(Y_el_, Y_local_);
   // L-vector to T-vector
   h1_prolongation_->MultTranspose(Y_local_, Y);

   // Re-assign the essential degrees of freedom on the final output vector.
   {
      const auto d_dX = dX.Read();
      auto d_Y = Y.ReadWrite();
      const auto d_ess_tdof_list = ess_tdof_list_.Read();
      MFEM_FORALL(i, ess_tdof_list_.Size(),
                  d_Y[d_ess_tdof_list[i]] = d_dX[d_ess_tdof_list[i]];);
   }
}

void ElasticityOperator::AssembleGradientDiagonal(Vector &Ke_diag,
                                                  Vector &K_diag_local,
                                                  Vector &K_diag) const
{
   Ke_diag.SetSize(d1d_ * d1d_ * d1d_ * dim_ * ne_ * dim_);
   K_diag_local.SetSize(h1_element_restriction_->Width() * dim_);
   K_diag.SetSize(h1_prolongation_->Width() * dim_);

   element_kernel_assemble_diagonal_wrapper(
      ne_, maps_->B, maps_->G, ir_->GetWeights(), geometric_factors_->J,
      geometric_factors_->detJ, cstate_el_, Ke_diag);

   // For each dimension, the H1 element restriction and H1 prolongation
   // transpose actions are applied separately.
   for (int i = 0; i < dim_; i++)
   {
      // Scalar component E-size
      int sce_sz = d1d_ * d1d_ * d1d_ * dim_ * ne_;
      // Scalar component L-size
      int scl_sz = h1_element_restriction_->Width();

      Vector vin_local, vout_local;
      vin_local.MakeRef(Ke_diag, i * sce_sz, sce_sz);
      vout_local.MakeRef(K_diag_local, i * scl_sz, scl_sz);
      h1_element_restriction_->MultTranspose(vin_local, vout_local);

      // Scalar component T-size
      int sct_sz = h1_prolongation_->Width();
      Vector vout;
      vout.MakeRef(K_diag, i * sct_sz, sct_sz);
      h1_prolongation_->MultTranspose(vout_local, vout);
   }

   // Each essential dof row and column are set to zero with it's diagonal entry
   // set to 1, i.e. (Ke)_ii = 1.0.
   ess_tdof_list_.HostRead();
   int num_submats = h1_fes_.GetTrueVSize() / h1_fes_.GetVDim();
   auto K_diag_submats = Reshape(K_diag.Write(), num_submats, dim_, dim_);
   for (int i = 0; i < ess_tdof_list_.Size(); i++)
   {
      int ess_idx = ess_tdof_list_[i];
      int submat = ess_idx % num_submats;
      int row = ess_idx / num_submats;
      for (int j = 0; j < dim_; j++)
      {
         if (row == j)
         {
            K_diag_submats(submat, row, j) = 1.0;
         }
         else
         {
            K_diag_submats(submat, row, j) = 0.0;
            K_diag_submats(submat, j, row) = 0.0;
         }
      }
   }
}

template <typename material_type>
void ElasticityOperator::SetMaterial(const material_type &material)
{
   if (dim_ != dimension)
   {
      MFEM_ABORT("dim != 3 not implemented");
   }

   element_apply_kernel_wrapper =
      [=](const int ne, const Array<double> &B_, const Array<double> &G_,
          const Array<double> &W_, const Vector &Jacobian_, const Vector &detJ_,
          const Vector &X_, Vector &Y_)
   {
      const int id = (d1d_ << 4) | q1d_;
      switch (id)
      {
         case 0x22:
            ElasticityKernels::Apply3D<2, 2, material_type>(
               ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
            break;
         case 0x33:
            ElasticityKernels::Apply3D<3, 3, material_type>(
               ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
            break;
         default:
            MFEM_ABORT("not implemented");
      }
   };

   element_apply_gradient_kernel_wrapper =
      [=](const int ne, const Array<double> &B_, const Array<double> &G_,
          const Array<double> &W_, const Vector &Jacobian_, const Vector &detJ_,
          const Vector &dU_, Vector &dF_, const Vector &U_)
   {
      const int id = (d1d_ << 4) | q1d_;
      switch (id)
      {
         case 0x22:
            ElasticityKernels::ApplyGradient3D<2, 2, material_type>(
               ne, B_, G_, W_, Jacobian_, detJ_, dU_, dF_, U_, material);
            break;
         case 0x33:
            ElasticityKernels::ApplyGradient3D<3, 3, material_type>(
               ne, B_, G_, W_, Jacobian_, detJ_, dU_, dF_, U_, material);
            break;
         default:
            MFEM_ABORT("not implemented");
      }
   };

   element_kernel_assemble_diagonal_wrapper =
      [=](const int ne, const Array<double> &B_, const Array<double> &G_,
          const Array<double> &W_, const Vector &Jacobian_, const Vector &detJ_,
          const Vector &X_, Vector &Y_)
   {
      const int id = (d1d_ << 4) | q1d_;
      switch (id)
      {
         case 0x22:
            ElasticityKernels::AssembleGradientDiagonal3D<2, 2, material_type>(
               ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
            break;
         case 0x33:
            ElasticityKernels::AssembleGradientDiagonal3D<3, 3, material_type>(
               ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
            break;
         default:
            MFEM_ABORT("not implemented");
      }
   };
}

ElasticityOperator::~ElasticityOperator() { delete gradient_; }
