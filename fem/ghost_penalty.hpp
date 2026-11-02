/** @file ghost_penalty.hpp
    @brief SVD-based projection ghost penalties for full H1 assembly.

    The API is available when MFEM_USE_LAPACK is enabled. It supports affine
    triangles in 2D and tetrahedra in 3D, with MPI shared-face support.
    See doc/projection-ghost-penalty.md for the mathematical formulation and
    miniapps/mtop/projection_ghost_penalty.cpp for a complete example. */

#ifndef MFEM_GHOST_PENALTY_HPP
#define MFEM_GHOST_PENALTY_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_LAPACK

#include "bilininteg.hpp"
#include "fespace.hpp"
#include <map>
#include <memory>
#include <string>

namespace mfem
{

/** @brief Parameters for the two-cell projection-residual penalty.

    The patch matrix is @f$G=\gamma h_F^{-2}R^TR@f$, where @f$R@f$ contains
    quadrature-weighted residuals after projection onto a common physical
    total-degree polynomial space on the two complete cells. */
struct ProjectionGhostPenaltyOptions
{
   /// Finite, nonnegative penalty strength; zero still performs validation/SVD.
   real_t gamma = 1.0;
   /// Common polynomial degree >= 0; -1 uses the minimum of the two FE orders.
   int projection_degree = -1;
   /** @brief Nonnegative extra order for generated volume quadrature rules.
       On side i, the target order is 2*max(FE order, projection degree) plus
       this value. Caller-supplied rules need only meet the order without it. */
   int quadrature_extra = 2;
   /** @brief Relative singular-value threshold in [0,1).
       Zero selects 32*max(number of quadrature points, number of polynomial
       basis functions)*epsilon(real_t). Rank deficiency is an error, not a
       request for truncated SVD or regularization. */
   real_t svd_rtol = 0.0;
   /// Print per-patch SVD rank, conditioning, and geometry data on each rank.
   bool verbose = false;
};

/// Diagnostics for one local weighted projection or two-cell patch assembly.
struct ProjectionGhostPenaltyDiagnostics
{
   /// Singular values of the weighted polynomial matrix, in descending order.
   Vector singular_values;
   /// Number of singular values strictly above the relative cutoff.
   int rank = 0;
   /// Number of weighted evaluation rows, summed over both cells for a patch.
   int quadrature_points = 0;
   /// Relative cutoff, applied to the largest singular value when testing rank.
   real_t relative_threshold = 0.0;
   /// Largest divided by smallest singular value for a full-rank projection.
   real_t condition_number = 0.0;
   /// Penalty length h_F = max(diameter(first cell), diameter(second cell)).
   real_t face_length = 0.0;
   /// Polynomial normalization length (volume(first)+volume(second))^(1/dim).
   real_t basis_length = 0.0;
};

/** @brief Local dense kernels, independent of face selection and MPI.

    Projection uses a thin LAPACK SVD rather than normal equations. All
    workspaces are local to a call; callers must not share mutable element
    transformations between concurrent Assemble() calls. */
class ProjectionGhostPenaltyKernel
{
 public:
   /** @brief Solve a full-column-rank weighted least-squares projection.

       For weighted polynomial evaluations @f$Q@f$ and weighted FE evaluations
       @f$H@f$, compute @f$C=V\Sigma^{-1}U^TH@f$ from @f$Q=U\Sigma V^T@f$.

       @param[in] weighted_polynomials Finite nq-by-m matrix, nq >= m > 0.
       @param[in] weighted_shapes Finite nq-by-ndof matrix, ndof > 0.
       @param[in] relative_tolerance Cutoff in [0,1); zero uses the automatic
                  threshold from ProjectionGhostPenaltyOptions::svd_rtol.
       @param[out] coefficients Resized to m-by-ndof and overwritten with C.
       @param[in,out] diagnostics Receives spectrum, rank, quadrature count,
                      relative threshold, and condition number. The geometry
                      fields face_length and basis_length are left unchanged.
       @param[in] context Label appended to validation and rank diagnostics.

       The input matrices are not modified. Invalid dimensions, nonfinite
       entries/results, and deficient numerical rank trigger MFEM_VERIFY;
       no singular values are truncated and no degree reduction is attempted. */
   static void
   ComputeProjection(const DenseMatrix &weighted_polynomials,
                     const DenseMatrix &weighted_shapes,
                     real_t relative_tolerance, DenseMatrix &coefficients,
                     ProjectionGhostPenaltyDiagnostics &diagnostics,
                     const std::string &context = "standalone patch");

   /** @brief Assemble a scalar penalty over two complete physical cells.

       @param[in] first_fe Scalar, VALUE-mapped FE on the first cell.
       @param[in] second_fe Compatible scalar FE; its order may differ.
       @param[in,out] first_trans First cell transformation; its integration
                      point is modified during evaluation.
       @param[in,out] second_trans Second cell transformation, also modified.
       @param[in] options Projection, quadrature, and penalty parameters.
       @param[out] matrix Resized square matrix, with first-cell scalar DOFs
                   followed by second-cell scalar DOFs; prior entries are lost.
       @param[out] diagnostics Optional destination for all patch diagnostics.
       @param[in] first_rule Optional first-cell reference-volume quadrature.
       @param[in] second_rule Optional second-cell reference-volume quadrature.
       @param[in] context Label included in error and verbose diagnostics.

       Both cells must be nondegenerate affine triangles in 2D or tetrahedra
       in 3D, with matching reference geometries and no embedding. The caller
       supplies the cell pair; adjacency and active/cut flags are not checked.
       Rules are borrowed only for this call. A null rule is generated with
       positive weights; a supplied rule must truthfully advertise order at
       least 2*max(its FE order, projection degree). Every physical quadrature
       weight must be finite and strictly positive.

       Internally the cells are canonically ordered by physical vertices, but
       output DOFs always follow the caller's first/second ordering. Weighted
       residual outer products produce a symmetric positive-semidefinite
       matrix up to roundoff. There is no face-surface or cut-cell integration,
       and no MPI communication. Mixed-order pairs are supported here even
       though GhostFaceSelection requires uniform-order assembled spaces. */
   static void
   Assemble(const FiniteElement &first_fe, const FiniteElement &second_fe,
            ElementTransformation &first_trans,
            ElementTransformation &second_trans,
            const ProjectionGhostPenaltyOptions &options, DenseMatrix &matrix,
            ProjectionGhostPenaltyDiagnostics *diagnostics = nullptr,
            const IntegrationRule *first_rule = nullptr,
            const IntegrationRule *second_rule = nullptr,
            const std::string &context = "standalone patch");
};

/** @brief Prepared selection of local and MPI-shared interior faces.

    A face is selected when both cells are active and at least one lies in the
    cut/extension band. The mesh must be conforming and consist of affine
    triangles or tetrahedra; the space must use a uniform-order H1_FECollection.
    Curved/embedded cells and ambiguous multiple-face element pairings are
    unsupported. Use a ParFiniteElementSpace when the mesh is a ParMesh.

    The FE space and mesh are borrowed and must outlive this object and all
    integrators that use it. Flags are consumed during construction, not kept
    by reference. Rebuild selection and forms after refinement, repartitioning,
    FE-space updates, or geometry/flag changes. Sequence checks cannot detect
    raw coordinate edits or subsequent changes to the input flag arrays. */
class GhostFaceSelection
{
 private:
   /// Borrowed FE space whose element numbering defines the face keys.
   FiniteElementSpace *space;
   /// Mesh sequence captured when the lookup data were prepared.
   long mesh_sequence;
   /// FE-space sequence captured when the lookup data were prepared.
   long space_sequence;
   /// Local element count captured at construction.
   int elements;
   /// FE-space communicator rank, or zero in serial.
   int rank = 0;
   /// Number of selected, fully local interior faces.
   int local_faces = 0;
   /// Number of selected shared-face visits on this rank.
   int shared_faces = 0;
   /// Selection flags keyed by unordered local/shifted-neighbor element pairs.
   std::map<std::pair<int, int>, bool> faces;

 public:
   /** @brief Prepare face lookup data, collectively for a parallel FE space.

       @param[in,out] fes Borrowed space; parallel construction exchanges
                      neighbor metadata on its communicator.
       @param[in] active One binary (0/1) flag per locally owned element.
       @param[in] cut One binary flag per local element; inactive cut cells
                  do not seed the band.
       @param[in] extension_layers Nonnegative number of active-cell adjacency
                  steps used to expand the initial cut band before selection.

       All MPI ranks, including empty/inactive ones, must participate with the
       same polynomial order and extension_layers. Two L2/P0 grid functions
       exchange active and current-band markers. Construction prepares lookup
       data for both selected and unselected interior faces; subsequent face
       queries and integrator callbacks perform no metadata communication. */
   GhostFaceSelection(FiniteElementSpace &fes, const Array<int> &active,
                      const Array<int> &cut, int extension_layers = 0);

   /// Reject stale mesh/space sequences or a changed local element count.
   void Validate() const;
   /** @brief Query a prepared local or shared interior face after Validate().
       Boundary faces and unprepared pairs are errors, not false results.
       Shared faces use MFEM's local/shifted-neighbor element numbering. */
   bool IsSelected(const FaceElementTransformations &trans) const;
   /// Vector dimension of the borrowed FE space.
   int GetVDim() const { return space->GetVDim(); }
   /// Rank in the FE-space communicator, or zero for a serial space.
   int GetRank() const { return rank; }
   /// Selected faces whose two cells are owned by this rank, counted once.
   int GetNumLocalFaces() const { return local_faces; }
   /** @brief Selected shared-face visits on this rank; zero for a serial space.
       Each physical shared face is counted on both incident ranks. The global
       physical count is sum(GetNumLocalFaces() + 0.5*GetNumSharedFaces()). */
   int GetNumSharedFaces() const { return shared_faces; }
};

/** @brief Scalar H1 projection ghost penalty for LEGACY full assembly.

    Register with BilinearForm::AddInteriorFaceIntegrator() or its parallel
    counterpart, not as a boundary or domain integrator. The space must have
    vdim=1. Unselected faces return zero matrices. Other assembly levels are
    not implemented.

    The form owns the integrator; the integrator shares immutable selection
    ownership and copies options. Selection still borrows the mesh and space.
    In MPI, use identical options across ranks, process both sides of shared
    faces, and retain ParBilinearForm's default KeepNbrBlock(false). Call
    Assemble() through the concrete ParBilinearForm type: the base-class
    BilinearForm::Assemble() is nonvirtual and skips shared-face assembly.
    ParallelAssemble() produces the true-DOF matrix. */
class ScalarProjectionGhostPenaltyIntegrator : public BilinearFormIntegrator
{
 private:
   /// Shared immutable face selection; its mesh and space remain borrowed.
   std::shared_ptr<const GhostFaceSelection> selection;
   /// Owned copy of the kernel configuration.
   ProjectionGhostPenaltyOptions options;

 public:
   /** @brief Construct an integrator with validated parameters.
       @param[in] selection Nonnull selection for a scalar (vdim=1) space.
       @param[in] options Configuration copied into the integrator. */
   ScalarProjectionGhostPenaltyIntegrator(
      std::shared_ptr<const GhostFaceSelection> selection,
      const ProjectionGhostPenaltyOptions &options = {});

   /** @brief Overwrite matrix in first-cell/second-cell scalar DOF ordering.
       The result has size first_fe.GetDof()+second_fe.GetDof(). Selection is
       validated on every face; selected patches use the local kernel.
       This callback performs no MPI communication.
       @param[in] first_fe Scalar FE on the first cell.
       @param[in] second_fe Scalar FE on the second cell.
       @param[in,out] trans Prepared face and mutable cell transformations.
       @param[out] matrix Resized and overwritten scalar patch matrix. */
   void AssembleFaceMatrix(const FiniteElement &first_fe,
                           const FiniteElement &second_fe,
                           FaceElementTransformations &trans,
                           DenseMatrix &matrix) override;
};

/** @brief Componentwise H1 ghost penalty for any vector dimension >= 2.

    Each component receives the scalar patch penalty, with no cross-component
    coupling. The vector dimension need not equal the spatial dimension. Local
    matrices use element-local component blocks: all components of the first
    cell, followed by all components of the second. MFEM's VDof mapping supports
    both Ordering::byNODES and Ordering::byVDIM global orderings.

    The registration, ownership, LEGACY assembly, and MPI requirements are the
    same as for ScalarProjectionGhostPenaltyIntegrator. */
class VectorProjectionGhostPenaltyIntegrator : public BilinearFormIntegrator
{
 private:
   /// Shared immutable face selection; its mesh and space remain borrowed.
   std::shared_ptr<const GhostFaceSelection> selection;
   /// Owned copy of the componentwise kernel configuration.
   ProjectionGhostPenaltyOptions options;

 public:
   /** @brief Construct an integrator with validated parameters.
       @param[in] selection Nonnull selection for a space with vdim >= 2.
       @param[in] options Configuration copied into the integrator. */
   VectorProjectionGhostPenaltyIntegrator(
      std::shared_ptr<const GhostFaceSelection> selection,
      const ProjectionGhostPenaltyOptions &options = {});

   /** @brief Overwrite a componentwise patch matrix, zero on unselected faces.
       Its size is vdim*(first_fe.GetDof()+second_fe.GetDof()), with the local
       ordering described above. No metadata exchange occurs in this callback.
       @param[in] first_fe Scalar FE underlying the first cell's components.
       @param[in] second_fe Scalar FE underlying the second cell's components.
       @param[in,out] trans Prepared face and mutable cell transformations.
       @param[out] matrix Resized and overwritten componentwise patch matrix. */
   void AssembleFaceMatrix(const FiniteElement &first_fe,
                           const FiniteElement &second_fe,
                           FaceElementTransformations &trans,
                           DenseMatrix &matrix) override;
};

}

#endif
#endif
