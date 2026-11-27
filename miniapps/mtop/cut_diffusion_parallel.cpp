#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

/** @brief Run-time options for the parallel cut-domain diffusion example.

    ReadOptions() populates and validates these values before mesh construction.
    All ranks must use the same settings. Defaults select a 2D straight band,
    order-one H1 elements, projection stabilization, and unpreconditioned CG.
    The solution order does not set the level-set order: main() uses a separate
    degree-two or degree-four field to represent the chosen geometry.

    The string pointers borrow literals or command-line storage and must remain
    valid throughout the solve. MUMPS and ParaView are selected independently;
    neither changes the discretization or supplies a preconditioner. */
struct Settings
{
   int dimension = 2;
   int elements = 8;
   int order = 1;
   int quadrature_extra = 4;
   int max_iterations = 5000;
   real_t lower = 0.23;
   real_t upper = 0.77;
   real_t bend = 0.08;
   real_t diffusivity = 1.0;
   real_t gamma = 1.0;
   real_t relative_tolerance = 1.e-11;
   const char *geometry = "band";
   const char *stabilization = "projection";
   const char *solver = "cg";
   const char *problem = "diffusion";
   const char *backend = "algoim-auto";
   const char *output_directory = "ParaView";
   bool ghost_penalty = true;
   bool paraview = false;
   bool self_test = false;
};

/** @brief Compare an option value with a supported keyword, case-sensitively.
    @param[in] value Non-null, null-terminated option value.
    @param[in] expected Non-null, null-terminated keyword.
    @return True only for an exact string match. */
bool Is(const char *value, const char *expected)
{
   return std::string(value) == expected;
}

/** @brief Parse command-line options and check supported problem combinations.
    @param[in] argc Number of command-line arguments.
    @param[in] argv Argument strings, retained by the string-valued options.
    @param[in,out] settings Defaults on entry; parsed settings on success.
    @return False when parsing fails or help is requested; true on success.

    Call on every MPI rank after initialization, with identical arguments.
    Only rank zero prints usage or the accepted options. Invalid numerical
    parameters, unsupported geometry/problem/stabilization combinations, and
    unavailable MUMPS support fail through MFEM_VERIFY rather than returning
    false. Backend availability is checked later when the backend is created.
    The argument storage must outlive any settings that refer to it. */
bool ReadOptions(int argc, char *argv[], Settings &settings)
{
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&settings.dimension, "-dim", "--dimension",
                       "Dimension: 2 or 3.");
   arguments.AddOption(&settings.elements, "-n", "--elements",
                       "Background elements per coordinate.");
   arguments.AddOption(&settings.order, "-o", "--order",
                       "H1 solution order, at least one.");
   arguments.AddOption(&settings.geometry, "-ls", "--level-set",
                       "band or curved-channel.");
   arguments.AddOption(&settings.lower, "-a", "--lower",
                       "Lower Dirichlet-side intersection.");
   arguments.AddOption(&settings.upper, "-b", "--upper",
                       "Upper Dirichlet-side intersection.");
   arguments.AddOption(&settings.bend, "-bend", "--bend",
                       "Curved-channel bending amplitude.");
   arguments.AddOption(&settings.stabilization, "-stab", "--stabilization",
                       "projection (default) or classical (order one only).");
   arguments.AddOption(&settings.gamma, "-gamma", "--gamma",
                       "Positive ghost-penalty strength.");
   arguments.AddOption(
       &settings.ghost_penalty, "-gp", "--ghost-penalty", "-no-gp",
       "--no-ghost-penalty",
       "Enable ghost stabilization; disable only for diagnostics.");
   arguments.AddOption(&settings.diffusivity, "-k", "--diffusivity",
                       "Positive constant diffusivity.");
   arguments.AddOption(
       &settings.problem, "-problem", "--problem",
       "diffusion, patch, or manufactured; last two require band geometry.");
   arguments.AddOption(&settings.backend, "-backend", "--backend",
                       "algoim-auto, algoim-general, or algoim-polynomial.");
   arguments.AddOption(
       &settings.quadrature_extra, "-qe", "--quadrature-extra",
       "Extra cut quadrature order above twice the solution order.");
   arguments.AddOption(&settings.solver, "-solver", "--solver",
                       "cg (unpreconditioned) or mumps.");
   arguments.AddOption(&settings.relative_tolerance, "-rtol",
                       "--relative-tolerance", "CG relative tolerance.");
   arguments.AddOption(&settings.max_iterations, "-mi", "--max-iterations",
                       "Maximum CG iterations.");
   arguments.AddOption(&settings.paraview, "-pv", "--paraview", "-no-pv",
                       "--no-paraview",
                       "Write level_set and solution to ParaView.");
   arguments.AddOption(&settings.output_directory, "-out", "--output-dir",
                       "ParaView output parent directory.");
   arguments.AddOption(
       &settings.self_test, "-test", "--self-test", "-no-test",
       "--no-self-test",
       "Run the example-local classical face-kernel tests instead of a solve.");
   arguments.Parse();
   if (!arguments.Good())
   {
      if (Mpi::Root())
      {
         arguments.PrintUsage(mfem::out);
      }
      return false;
   }
   MFEM_VERIFY(settings.dimension == 2 || settings.dimension == 3,
               "Dimension must be 2 or 3.");
   MFEM_VERIFY(settings.elements >= 2 && settings.order >= 1 &&
                   settings.order <= 16 && settings.quadrature_extra >= 0 &&
                   settings.quadrature_extra <= 32,
               "Require elements >= 2, order in [1,16], and quadrature-extra "
               "in [0,32].");
   MFEM_VERIFY(Is(settings.geometry, "band") ||
                   Is(settings.geometry, "curved-channel"),
               "Unknown level-set geometry.");
   MFEM_VERIFY(std::isfinite(settings.lower) && std::isfinite(settings.upper) &&
                   settings.lower > 0.0 && settings.lower < settings.upper &&
                   settings.upper < 1.0 && std::isfinite(settings.bend),
               "Require finite 0 < lower < upper < 1 and finite bend.");
   MFEM_VERIFY(
       Is(settings.geometry, "band") ||
           (settings.lower - std::abs(settings.bend) > 0.0 &&
            settings.upper + std::abs(settings.bend) < 1.0),
       "Curved channel must remain inside the transverse background bounds.");
   MFEM_VERIFY(Is(settings.stabilization, "projection") ||
                   Is(settings.stabilization, "classical"),
               "Unknown stabilization.");
   MFEM_VERIFY(!Is(settings.stabilization, "classical") || settings.order == 1,
               "Classical stabilization supports only order one.");
   MFEM_VERIFY(
       std::isfinite(settings.gamma) && settings.gamma > 0.0 &&
           std::isfinite(settings.diffusivity) && settings.diffusivity > 0.0 &&
           std::isfinite(settings.gamma * settings.diffusivity),
       "Require finite positive gamma and diffusivity with finite product.");
   MFEM_VERIFY(Is(settings.solver, "cg") || Is(settings.solver, "mumps"),
               "Unknown solver.");
#ifndef MFEM_USE_MUMPS
   MFEM_VERIFY(!Is(settings.solver, "mumps"),
               "MUMPS requested but MFEM_USE_MUMPS is disabled.");
#endif
   MFEM_VERIFY(std::isfinite(settings.relative_tolerance) &&
                   settings.relative_tolerance > 0.0 &&
                   settings.relative_tolerance < 1.0 &&
                   settings.max_iterations > 0,
               "Invalid solver tolerance or iteration limit.");
   MFEM_VERIFY(Is(settings.problem, "diffusion") ||
                   Is(settings.problem, "patch") ||
                   Is(settings.problem, "manufactured"),
               "Unknown problem.");
   MFEM_VERIFY(Is(settings.geometry, "band") ||
                   Is(settings.problem, "diffusion"),
               "Patch/manufactured flux conditions require band geometry.");
   MFEM_VERIFY(Is(settings.backend, "algoim-auto") ||
                   Is(settings.backend, "algoim-general") ||
                   Is(settings.backend, "algoim-polynomial"),
               "An ALGOIM backend is required.");
   if (Mpi::Root())
   {
      arguments.PrintOptions(mfem::out);
   }
   return true;
}

/** @brief Evaluate the analytic polynomial defining the physical channel.
    @param[in] settings Validated dimension, intersections, and bending options.
    @param[in] position Physical coordinates in the unit background box.
    @return A positive value inside the channel, zero on its implicit boundary,
            and a negative value outside; this is not a signed distance.

    With w=(upper-lower)/2, return w*w-(y-c)*(y-c). For the band,
    c=(lower+upper)/2. The curved channel adds bend*q(x) in 2D or
    bend*q(x)*q(z) in 3D, where q(t)=4*t*(1-t). The shift vanishes on the
    two Dirichlet sides, preserving their intersection interval. This analytic
    function is projected into the independent level-set space before ALGOIM
    reads the geometry; quadrature does not call this function directly. */
real_t LevelSet(const Settings &settings, const Vector &position)
{
   real_t center = 0.5 * (settings.lower + settings.upper);
   if (Is(settings.geometry, "curved-channel"))
   {
      real_t shift = 4.0 * position(0) * (1.0 - position(0));
      if (settings.dimension == 3)
      {
         shift *= 4.0 * position(2) * (1.0 - position(2));
      }
      center += settings.bend * shift;
   }
   const real_t radius = 0.5 * (settings.upper - settings.lower);
   return (radius - (position(1) - center)) * (radius + (position(1) - center));
}

/** @brief Oscillatory part of the straight-band manufactured solution.
    @param[in] settings Dimension and band endpoints a=lower, b=upper.
    @param[in] position Physical coordinates with settings.dimension entries.
    @return sin(pi*x)*cos(pi*(y-a)/(b-a)), times cos(pi*z) in 3D.

    This function vanishes on x=0 and x=1 and has zero normal derivative on
    y=a, y=b, and, in 3D, z=0 and z=1. Thus adding it to x preserves the
    prescribed Dirichlet values and the band's homogeneous natural flux.
    Those flux conditions do not apply to the curved channel. */
real_t Oscillation(const Settings &settings, const Vector &position)
{
   const real_t pi = std::acos(-1.0);
   real_t value = std::sin(pi * position(0)) *
                  std::cos(pi * (position(1) - settings.lower) /
                           (settings.upper - settings.lower));
   if (settings.dimension == 3)
   {
      value *= std::cos(pi * position(2));
   }
   return value;
}

/** @brief Evaluate the selected straight-band reference solution.
    @param[in] settings Validated problem selection and positive diffusivity.
    @param[in] position Physical coordinates with settings.dimension entries.
    @return x for the patch test, x+Oscillation() for the manufactured problem,
            or x+x*(1-x)/(2*kappa) for unit forcing.

    Each reference has Dirichlet data g_D=x and zero flux on the band's
    natural boundaries. Use this function for error measurements only with
    band geometry; the curved-channel solve has no supplied exact solution. */
real_t ExactSolution(const Settings &settings, const Vector &position)
{
   if (Is(settings.problem, "patch"))
   {
      return position(0);
   }
   if (Is(settings.problem, "manufactured"))
   {
      return position(0) + Oscillation(settings, position);
   }
   return position(0) +
          position(0) * (1.0 - position(0)) / (2.0 * settings.diffusivity);
}

/** @brief Evaluate the analytic physical gradient of ExactSolution().
    @param[in] settings Same straight-band problem data as ExactSolution().
    @param[in] position Physical evaluation point.
    @param[out] gradient Resized to settings.dimension and fully overwritten.

    This differentiates the analytic expression, not its finite element
    interpolant, and supplies the reference for the H1-seminorm error. */
void ExactGradient(const Settings &settings, const Vector &position,
                   Vector &gradient)
{
   gradient.SetSize(settings.dimension);
   gradient = 0.0;
   gradient(0) = 1.0;
   if (Is(settings.problem, "patch"))
   {
      return;
   }
   if (Is(settings.problem, "diffusion"))
   {
      gradient(0) += (1.0 - 2.0 * position(0)) / (2.0 * settings.diffusivity);
      return;
   }
   const real_t pi = std::acos(-1.0);
   const real_t width = settings.upper - settings.lower;
   const real_t first = pi * position(0);
   const real_t second = pi * (position(1) - settings.lower) / width;
   const real_t third = settings.dimension == 3 ? pi * position(2) : 0.0;
   gradient(0) += pi * std::cos(first) * std::cos(second) * std::cos(third);
   gradient(1) =
       -pi / width * std::sin(first) * std::sin(second) * std::cos(third);
   if (settings.dimension == 3)
   {
      gradient(2) = -pi * std::sin(first) * std::cos(second) * std::sin(third);
   }
}

/** @brief Build the serial Cartesian mesh and mark Dirichlet-side faces.
    @param[in] settings Validated dimension, elements per axis, and band limits.
    @return A unit-square quad mesh or unit-cube hex mesh with boundary
            attributes 1 (left), 2 (right), and 3 (all remaining faces).

    Attributes 1 and 2 are assigned only to faces on x=0 and x=1 whose
    y-interval has positive-measure overlap with (lower, upper). Both supplied
    geometries have this same intersection on those sides. A partially cut
    background face is marked as a whole, allowing standard MFEM essential
    boundary handling with the prescribed data extended along that face.
    Each rank builds the same serial mesh before main() partitions it. */
Mesh BackgroundMesh(const Settings &settings)
{
   Mesh mesh =
       settings.dimension == 2
           ? Mesh::MakeCartesian2D(settings.elements, settings.elements,
                                   Element::QUADRILATERAL, true, 1.0, 1.0)
           : Mesh::MakeCartesian3D(settings.elements, settings.elements,
                                   settings.elements, Element::HEXAHEDRON, 1.0,
                                   1.0, 1.0);
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      auto *transformation = mesh.GetBdrElementTransformation(boundary);
      Vector center(settings.dimension);
      transformation->Transform(
          Geometries.GetCenter(transformation->GetGeometryType()), center);
      Array<int> vertices;
      mesh.GetBdrElementVertices(boundary, vertices);
      real_t minimum = 1.0, maximum = 0.0;
      for (int vertex : vertices)
      {
         minimum = std::min(minimum, mesh.GetVertex(vertex)[1]);
         maximum = std::max(maximum, mesh.GetVertex(vertex)[1]);
      }
      const bool intersects =
          maximum > settings.lower && minimum < settings.upper;
      const int attribute = intersects && center(0) == 0.0
                                ? 1
                                : (intersects && center(0) == 1.0 ? 2 : 3);
      mesh.GetBdrElement(boundary)->SetAttribute(attribute);
   }
   mesh.SetAttributes();
   return mesh;
}

/** @brief Owned cut-volume quadrature cache and parallel cell classification.

    ALGOIM integrates the positive region of the discrete level set in each
    locally owned background element. Rules use reference-cell coordinates and
    weights; the physical Jacobian is applied later by the MFEM integrators or
    Measure(). On the unit reference square/cube, the sum of weights is the
    physical-domain fraction of the corresponding Cartesian element.

    The constructor is collective on the mesh communicator, including ranks
    with no elements. It checks nonnegative finite weights, a full-cell anchor,
    and the known channel volume. Local arrays use the mesh element numbering;
    the scalar statistics are globally reduced and identical on all ranks.

    The cache owns copies of the integration rules and retains no references to
    the constructor inputs. It is nevertheless tied to their geometry and
    numbering: rebuild it and the dependent forms/selection after mesh,
    partition, level-set, or quadrature-setting changes. */
struct CellRules
{
   /// Reference-volume rules indexed by locally owned element number.
   std::vector<IntegrationRule> rules;
   /// Binary flags: positive quadrature measure, and active but not full.
   Array<int> active, cut;
   /// Global counts of active, cut, and fully interior elements.
   long long global_active = 0, global_cut = 0, global_full = 0;
   /// Smallest reference-volume fraction among active elements on all ranks.
   real_t minimum_fraction = 1.0;
   /// Global physical volume, including element Jacobian weights.
   real_t volume = 0.0;

   /** @brief Generate rules and classify elements collectively.
       @param[in,out] mesh Background mesh; its transformations are evaluated.
       @param[in] level_set Discrete geometry on mesh; positive is inside.
       @param[in] backend Selected ALGOIM quadrature implementation.
       @param[in] settings Validated geometry and quadrature options.

       The requested quadrature order is twice the solution order plus
       settings.quadrature_extra. A quadrature failure or unresolved channel
       is an error rather than an implicit change of backend or geometry. */
   CellRules(ParMesh &mesh, const ParGridFunction &level_set,
             const CutQuadratureBackend &backend, const Settings &settings)
       : rules(mesh.GetNE()), active(mesh.GetNE()), cut(mesh.GetNE())
   {
      active = 0;
      cut = 0;
      const int order = 2 * settings.order + settings.quadrature_extra;
      CutQuadratureGenerator generator(backend);
      long long local_counts[3] = {0, 0, 0}, global_counts[3];
      real_t local_volume = 0.0, local_minimum = 1.0;
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
         PolynomialLevelSet polynomial;
         polynomial.Set(level_set, element);
         auto *transformation = mesh.GetElementTransformation(element);
         CutQuadrature quadrature;
         const auto report = generator.Generate(
             transformation->GetGeometryType(), CutDomain::Positive(polynomial),
             order, quadrature);
         MFEM_VERIFY(report.Succeeded(), "Cut quadrature failed on rank "
                                             << Mpi::WorldRank() << " element "
                                             << element << ": "
                                             << report.message);
         rules[element] = quadrature.rule;
         real_t fraction = 0.0;
         for (int point = 0; point < quadrature.GetNPoints(); ++point)
         {
            const auto &integration_point = rules[element].IntPoint(point);
            MFEM_VERIFY(std::isfinite(integration_point.weight) &&
                            integration_point.weight >= 0.0,
                        "ALGOIM returned an invalid volume weight.");
            fraction += integration_point.weight;
            transformation->SetIntPoint(&integration_point);
            local_volume += integration_point.weight * transformation->Weight();
         }
         MFEM_VERIFY(fraction <= 1.0 + 1.e-9,
                     "Cut fraction exceeds the reference cell volume.");
         active[element] = fraction > 0.0;
         cut[element] = active[element] &&
                        quadrature.region != CutQuadrature::Region::Full;
         local_counts[0] += active[element];
         local_counts[1] += cut[element];
         local_counts[2] += active[element] && !cut[element];
         if (active[element])
         {
            local_minimum = std::min(local_minimum, fraction);
         }
      }
      MPI_Allreduce(local_counts, global_counts, 3, MPI_LONG_LONG, MPI_SUM,
                    mesh.GetComm());
      MPI_Allreduce(&local_volume, &volume, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, mesh.GetComm());
      MPI_Allreduce(&local_minimum, &minimum_fraction, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN, mesh.GetComm());
      global_active = global_counts[0];
      global_cut = global_counts[1];
      global_full = global_counts[2];
      MFEM_VERIFY(global_active > 0 && global_full > 0,
                  "Unresolved physical channel: refine until a full interior "
                  "cell anchors the cut band.");
      MFEM_VERIFY(std::abs(volume - (settings.upper - settings.lower)) < 1.e-7,
                  "Cut quadrature failed the exact channel-volume check; "
                  "increase quadrature order or change ALGOIM backend.");
   }
};

/** @brief Scalar diffusion restricted to the physical part of each element.

    For physical domain Omega = {level_set > 0}, assemble
    @f$ A^K_{ij} = \int_{K\cap\Omega}
        \kappa\nabla\varphi_i\cdot\nabla\varphi_j\,dx @f$.
    The implementation supplies a cached cut-volume rule to MFEM's standard
    DiffusionIntegrator, which evaluates the coefficient and physical gradients.
    Inactive elements contribute an explicitly sized zero matrix.

    Only the legacy element-matrix assembly path is adapted; this is not a
    partial-assembly or matrix-free cut integrator. The coefficient and cache
    are borrowed and must outlive the integrator. The cache must not be moved or
    changed while in use. The bilinear form owns the registered integrator. */
class CutDiffusionIntegrator : public DiffusionIntegrator
{
   /// Borrowed cache whose numbering matches the form's mesh.
   const CellRules &cells;

public:
   /** @brief Bind the borrowed diffusivity and physical-domain rule cache.
       @param[in] coefficient Diffusivity, including any physical scaling.
       @param[in] rules Previously prepared cut-volume rules. */
   CutDiffusionIntegrator(Coefficient &coefficient, const CellRules &rules)
       : DiffusionIntegrator(coefficient), cells(rules)
   {
   }
   /** @brief Assemble the cut-element matrix, or zero for an inactive element.
       @param[in] element Scalar solution finite element on this cell.
       @param[in,out] transformation Cell mapping and local element number.
       @param[out] matrix Dense local stiffness matrix, fully overwritten.

       transformation.ElementNo must index a locally owned cached rule.
       The output is resized to element.GetDof() by element.GetDof(). */
   void AssembleElementMatrix(const FiniteElement &element,
                              ElementTransformation &transformation,
                              DenseMatrix &matrix) override
   {
      if (!cells.active[transformation.ElementNo])
      {
         matrix.SetSize(element.GetDof());
         matrix = 0.0;
         return;
      }
      SetIntegrationRule(cells.rules[transformation.ElementNo]);
      DiffusionIntegrator::AssembleElementMatrix(element, transformation,
                                                 matrix);
   }
};

/** @brief Volume load restricted to the same physical region as diffusion.

    Assemble @f$ b^K_i = \int_{K\cap\Omega} f\varphi_i\,dx @f$ by passing the
    cached rule to DomainLFIntegrator. Inactive elements return a zero vector.
    Sharing CellRules with CutDiffusionIntegrator keeps the load and stiffness
    on the same discrete geometry and quadrature.

    Assembly is host-only: SupportsDevice() prevents the linear form from
    bypassing this per-element override through the device assembly path.
    The forcing coefficient and cache are borrowed and must outlive this
    integrator; the linear form owns the registered integrator. */
class CutLoadIntegrator : public DomainLFIntegrator
{
   /// Borrowed, immutable cut-volume cache for the linear form's mesh.
   const CellRules &cells;

public:
   /** @brief Bind the borrowed volume forcing and physical-domain rule cache.
       @param[in] coefficient Volume forcing f, not boundary flux data.
       @param[in] rules Previously prepared cut-volume rules. */
   CutLoadIntegrator(Coefficient &coefficient, const CellRules &rules)
       : DomainLFIntegrator(coefficient), cells(rules)
   {
   }
   /// Require the host assembly path that selects a rule for each element.
   bool SupportsDevice() const override { return false; }
   /** @brief Assemble a cut-element load with element.GetDof() entries.
       @param[in] element Scalar solution finite element on this cell.
       @param[in,out] transformation Cell mapping and local element number.
       @param[out] vector Dense local load vector, fully overwritten.

       transformation.ElementNo indexes the locally owned quadrature cache. */
   void AssembleRHSElementVect(const FiniteElement &element,
                               ElementTransformation &transformation,
                               Vector &vector) override
   {
      if (!cells.active[transformation.ElementNo])
      {
         vector.SetSize(element.GetDof());
         vector = 0.0;
         return;
      }
      SetIntegrationRule(cells.rules[transformation.ElementNo]);
      DomainLFIntegrator::AssembleRHSElementVect(element, transformation,
                                                 vector);
   }
   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** @brief Classical first-order ghost penalty on selected background faces.

    For a face shared by K1 and K2, assemble over the entire background face,
    not just its physical-domain intersection:
    @f$ g_F(u,v) = \gamma\kappa h_F\int_F
        [\partial_n u][\partial_n v]\,ds @f$.
    Both derivatives use the same physical unit normal, with the jump defined
    as the K1 derivative minus the K2 derivative. Normal reversal leaves the
    matrix unchanged; exchanging elements permutes the corresponding DOF blocks.

    This example assumes scalar Q1 fields on affine Cartesian quadrilaterals or
    hexahedra. With volume Jacobians J_K and face Jacobian J_F, the scale
    h_F = (J_K1 + J_K2)/(2*J_F) is the mean normal cell height. Face quadrature
    of order two integrates the products of the tangentially varying Q1 normal
    derivatives exactly. Higher-order, curved, and non-Cartesian extensions are
    not implemented; no projection or SVD is used by this kernel.

    Register with ParBilinearForm::AddInteriorFaceIntegrator() for legacy full
    assembly and keep KeepNbrBlock(false). GhostFaceSelection prepares local
    and MPI-shared face lookup data collectively; this callback itself performs
    no MPI communication. Unselected faces contribute zero matrices.

    The form owns the integrator, which shares immutable selection ownership.
    The selection still borrows the mesh and FE space; both must outlive it.
    Rebuild selection and forms after geometry, space, or active/cut changes. */
class NormalDerivativeGhostPenaltyIntegrator : public BilinearFormIntegrator
{
   /// Prepared active-active face selection, with at least one cut cell.
   std::shared_ptr<const GhostFaceSelection> selection;
   /// Product gamma*kappa; the face-dependent h_F is applied during assembly.
   real_t strength;

public:
   /** @brief Share prepared face selection and copy the penalty coefficient.
       @param[in] faces Non-null prepared selection on the form's FE space.
       @param[in] coefficient Positive finite gamma*kappa, copied by value. */
   NormalDerivativeGhostPenaltyIntegrator(
       std::shared_ptr<const GhostFaceSelection> faces, real_t coefficient)
       : selection(std::move(faces)), strength(coefficient)
   {
   }
   /** @brief Assemble the two-element face matrix in [K1 DOFs, K2 DOFs] order.
       @param[in] first Scalar finite element on K1.
       @param[in] second Scalar finite element on K2.
       @param[in,out] transformation Face mapping and both adjacent cell maps.
       @param[out] matrix Dense two-element matrix, fully overwritten.

       The transformation must describe a prepared local or MPI-shared interior
       face. The output is resized and cleared even when the face is unselected.
       Selected faces require both elements to have order one. */
   void AssembleFaceMatrix(const FiniteElement &first,
                           const FiniteElement &second,
                           FaceElementTransformations &transformation,
                           DenseMatrix &matrix) override
   {
      const int dimension = first.GetDim();
      const int first_dofs = first.GetDof(), second_dofs = second.GetDof();
      matrix.SetSize(first_dofs + second_dofs);
      matrix = 0.0;
      if (!selection->IsSelected(transformation))
      {
         return;
      }
      MFEM_VERIFY(first.GetOrder() == 1 && second.GetOrder() == 1,
                  "Classical stabilization supports only order one.");
      DenseMatrix first_derivatives(first_dofs, dimension),
          second_derivatives(second_dofs, dimension);
      Vector normal(dimension), first_normal(first_dofs),
          second_normal(second_dofs);
      Vector jump(first_dofs + second_dofs);
      const auto &rule = IntRules.Get(transformation.GetGeometryType(), 2);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &integration_point = rule.IntPoint(point);
         transformation.SetAllIntPoints(&integration_point);
         CalcOrtho(transformation.Jacobian(), normal);
         const real_t face_measure = normal.Norml2();
         MFEM_VERIFY(face_measure > 0.0 && std::isfinite(face_measure),
                     "Degenerate background face.");
         normal /= face_measure;
         first.CalcPhysDShape(*transformation.Elem1, first_derivatives);
         second.CalcPhysDShape(*transformation.Elem2, second_derivatives);
         first_derivatives.Mult(normal, first_normal);
         second_derivatives.Mult(normal, second_normal);
         for (int dof = 0; dof < first_dofs; ++dof)
         {
            jump(dof) = first_normal(dof);
         }
         for (int dof = 0; dof < second_dofs; ++dof)
         {
            jump(first_dofs + dof) = -second_normal(dof);
         }
         const real_t normal_size =
             (transformation.Elem1->Weight() + transformation.Elem2->Weight()) /
             (2.0 * face_measure);
         const real_t weight =
             strength * normal_size * integration_point.weight * face_measure;
         for (int column = 0; column < jump.Size(); ++column)
            for (int row = 0; row < jump.Size(); ++row)
            {
               matrix(row, column) += weight * jump(row) * jump(column);
            }
      }
   }
};

/** @brief Register the selected ghost penalty with a parallel bilinear form.
    @param[in,out] form Legacy form that takes ownership of the new integrator.
    @param[in] selection Prepared face selection on the form's FE space.
    @param[in] settings Validated stabilization choice, gamma, and diffusivity.

    This is a no-op when ghost_penalty is false. Otherwise the penalty strength
    includes gamma*kappa; the projection degree equals the solution order.
    Classical stabilization is restricted to order one by ReadOptions().
    Registration does not assemble a matrix: the caller subsequently invokes
    ParBilinearForm::Assemble() with KeepNbrBlock(false) on every rank. */
void AddStabilization(
    ParBilinearForm &form,
    const std::shared_ptr<const GhostFaceSelection> &selection,
    const Settings &settings)
{
   if (!settings.ghost_penalty)
   {
      return;
   }
   if (Is(settings.stabilization, "classical"))
   {
      form.AddInteriorFaceIntegrator(new NormalDerivativeGhostPenaltyIntegrator(
          selection, settings.gamma * settings.diffusivity));
   }
   else
   {
      ProjectionGhostPenaltyOptions options;
      options.gamma = settings.gamma * settings.diffusivity;
      options.projection_degree = settings.order;
      form.AddInteriorFaceIntegrator(
          new ScalarProjectionGhostPenaltyIntegrator(selection, options));
   }
}

/** @brief Collect Dirichlet and unsupported true DOFs for strong elimination.
    @param[in,out] space Solution space; shared support flags are synchronized.
    @param[in] cells Active-element flags in the space's local mesh numbering.
    @param[in] boundary_marker MFEM attribute marker for Dirichlet boundaries.
    @param[in,out] solution Field with boundary data already projected; entries
                          without any active-element support are set to zero.
    @param[out] essential Sorted, unique local true-DOF indices to eliminate.
    @return Global number of inactive true DOFs, not the Dirichlet DOF count.

    Collective on the space communicator, including empty/inactive ranks.
    A DOF is supported if any incident active cell on any sharing rank needs it;
    its nodal position or level-set sign is not an elimination criterion.
    Standard GetEssentialTrueDofs() supplies the boundary indices. Only the
    owner appends each inactive true DOF, avoiding duplicate global counts.
    The caller uses DIAG_ONE to give eliminated zero rows a valid diagonal. */
long long EssentialDofs(ParFiniteElementSpace &space, const CellRules &cells,
                        const Array<int> &boundary_marker,
                        ParGridFunction &solution, Array<int> &essential)
{
   Array<int> supported(space.GetVSize()), element_dofs;
   supported = 0;
   for (int element = 0; element < space.GetNE(); ++element)
   {
      if (!cells.active[element])
      {
         continue;
      }
      space.GetElementDofs(element, element_dofs);
      for (int dof : element_dofs)
      {
         supported[dof >= 0 ? dof : -1 - dof] = 1;
      }
   }
   space.Synchronize(supported);
   space.GetEssentialTrueDofs(boundary_marker, essential);
   long long local_inactive = 0, global_inactive;
   for (int dof = 0; dof < space.GetVSize(); ++dof)
   {
      if (supported[dof])
      {
         continue;
      }
      solution(dof) = 0.0;
      const int true_dof = space.GetLocalTDofNumber(dof);
      if (true_dof >= 0)
      {
         essential.Append(true_dof);
         ++local_inactive;
      }
   }
   essential.Sort();
   essential.Unique();
   MPI_Allreduce(&local_inactive, &global_inactive, 1, MPI_LONG_LONG, MPI_SUM,
                 space.GetComm());
   return global_inactive;
}

/** @brief Globally reduced solution measurements on the physical domain.

    Measure() fills these values using the same cut-volume cache as assembly;
    inactive background regions and ghost-penalty terms are excluded. For the
    straight band, the errors use the selected problem's exact solution. For
    the curved channel no exact solution is claimed, so both errors stay -1.
    Algebraic residual and ghost energy are computed separately in main(). */
struct Diagnostics
{
   /// Physical-domain L2 norm of the recovered finite element solution.
   real_t norm = 0.0;
   /// Signed integral of the solution over the physical domain.
   real_t integral = 0.0;
   /// Integral of kappa*|grad(u)|^2, without a factor of one half.
   real_t diffusion_energy = 0.0;
   /// L2 error against the exact solution, or -1 when unavailable.
   real_t l2_error = -1.0;
   /// H1-seminorm error (gradient only), or -1 when unavailable.
   real_t h1_error = -1.0;
};

/** @brief Integrate solution diagnostics over the physical domain collectively.
    @param[in,out] mesh Background mesh; cell transformations are evaluated.
    @param[in] solution Recovered finite element solution on this mesh.
    @param[in] cells Physical-domain rules used for diffusion/load assembly.
    @param[in] settings Diffusivity, geometry, and exact-problem selection.
    @return Globally reduced Diagnostics, identical on every rank.

    All ranks in mesh.GetComm() participate. Element Jacobians convert cached
    reference weights to physical volume. L2 norms and gradient errors are
    square-rooted only after summing squared contributions globally; integral
    and diffusion energy are not square-rooted. Exact errors are evaluated
    only for the straight band and otherwise retain the -1 sentinel. */
Diagnostics Measure(ParMesh &mesh, const ParGridFunction &solution,
                    const CellRules &cells, const Settings &settings)
{
   const bool has_exact = Is(settings.geometry, "band");
   real_t local[5] = {0.0, 0.0, 0.0, 0.0, 0.0}, global[5];
   Vector position(settings.dimension), gradient(settings.dimension),
       exact_gradient(settings.dimension);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      auto *transformation = mesh.GetElementTransformation(element);
      const auto &rule = cells.rules[element];
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &integration_point = rule.IntPoint(point);
         transformation->SetIntPoint(&integration_point);
         const real_t weight =
             integration_point.weight * transformation->Weight();
         const real_t value =
             solution.GetValue(*transformation, integration_point);
         solution.GetGradient(*transformation, gradient);
         local[0] += weight * value * value;
         local[1] += weight * value;
         local[2] += settings.diffusivity * weight * (gradient * gradient);
         if (has_exact)
         {
            transformation->Transform(integration_point, position);
            const real_t difference = value - ExactSolution(settings, position);
            ExactGradient(settings, position, exact_gradient);
            gradient -= exact_gradient;
            local[3] += weight * difference * difference;
            local[4] += weight * (gradient * gradient);
         }
      }
   }
   MPI_Allreduce(local, global, 5, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 mesh.GetComm());
   Diagnostics result;
   result.norm = std::sqrt(global[0]);
   result.integral = global[1];
   result.diffusion_energy = global[2];
   if (has_exact)
   {
      result.l2_error = std::sqrt(global[3]);
      result.h1_error = std::sqrt(global[4]);
   }
   return result;
}

/** @brief Verify the classical face kernel on a serial two-cell patch.
    @param[in] dimension Spatial dimension, either 2 or 3.

    Checks symmetry, annihilation of every global Q1 basis monomial, exact
    energy for constant and tangentially varying normal-derivative jumps,
    sampled nonnegative quadratic forms, and neighbor-order invariance after
    the corresponding DOF permutation. Failures terminate through MFEM_VERIFY.
    No MPI collectives occur here: --self-test calls it independently on every
    rank. Distributed assembly is covered by the regression driver. */
void CheckClassicalKernel(int dimension)
{
   Mesh mesh = dimension == 2
                   ? Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, true)
                   : Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
   H1_FECollection collection(1, dimension);
   FiniteElementSpace space(&mesh, &collection);
   Array<int> active(mesh.GetNE()), cut(mesh.GetNE());
   active = 1;
   cut = 1;
   auto selection = std::make_shared<GhostFaceSelection>(space, active, cut);
   NormalDerivativeGhostPenaltyIntegrator integrator(selection, 1.0);
   for (int face = 0; face < mesh.GetNumFaces(); ++face)
   {
      auto *transformation = mesh.GetInteriorFaceTransformations(face);
      if (!transformation)
      {
         continue;
      }
      const auto &first = *space.GetFE(transformation->Elem1No);
      const auto &second = *space.GetFE(transformation->Elem2No);
      DenseMatrix matrix;
      integrator.AssembleFaceMatrix(first, second, *transformation, matrix);
      const real_t tolerance = 1.e-11 * std::max(1.0, matrix.FNorm());
      Vector values(matrix.Height()), image(matrix.Height()),
          physical(dimension);
      for (int column = 0; column < matrix.Width(); ++column)
         for (int row = 0; row < matrix.Height(); ++row)
         {
            MFEM_VERIFY(std::abs(matrix(row, column) - matrix(column, row)) <
                            tolerance,
                        "Classical matrix is not symmetric.");
         }
      for (int polynomial = 0; polynomial < (1 << dimension); ++polynomial)
      {
         int offset = 0;
         for (int side = 0; side < 2; ++side)
         {
            const auto &element = side == 0 ? first : second;
            auto *mapping =
                side == 0 ? transformation->Elem1 : transformation->Elem2;
            for (int dof = 0; dof < element.GetDof(); ++dof)
            {
               mapping->Transform(element.GetNodes().IntPoint(dof), physical);
               real_t value = 1.0;
               for (int axis = 0; axis < dimension; ++axis)
               {
                  if (polynomial & (1 << axis))
                  {
                     value *= physical(axis);
                  }
               }
               values(offset + dof) = value;
            }
            offset += element.GetDof();
         }
         matrix.Mult(values, image);
         MFEM_VERIFY(
             image.Normlinf() < tolerance,
             "Classical penalty does not annihilate a global Q1 field.");
      }
      for (int tangential = 0; tangential < (1 << (dimension - 1));
           ++tangential)
      {
         values = 0.0;
         real_t expected_energy = 0.5;
         for (int axis = 1; axis < dimension; ++axis)
         {
            if (tangential & (1 << (axis - 1)))
            {
               expected_energy /= 3.0;
            }
         }
         for (int dof = 0; dof < first.GetDof(); ++dof)
         {
            transformation->Elem1->Transform(first.GetNodes().IntPoint(dof),
                                             physical);
            values(dof) = physical(0) - 0.5;
            for (int axis = 1; axis < dimension; ++axis)
            {
               if (tangential & (1 << (axis - 1)))
               {
                  values(dof) *= physical(axis);
               }
            }
         }
         matrix.Mult(values, image);
         MFEM_VERIFY(std::abs(values * image - expected_energy) < tolerance,
                     "Incorrect classical jump energy/scaling.");
      }
      for (int sample = 0; sample < 8; ++sample)
      {
         for (int dof = 0; dof < values.Size(); ++dof)
         {
            values(dof) = std::sin((sample + 1.0) * (dof + 0.7));
         }
         matrix.Mult(values, image);
         MFEM_VERIFY(values * image >= -tolerance,
                     "Classical matrix is not positive semidefinite.");
      }
      std::swap(transformation->Elem1, transformation->Elem2);
      std::swap(transformation->Elem1No, transformation->Elem2No);
      std::swap(transformation->Loc1, transformation->Loc2);
      DenseMatrix reversed;
      integrator.AssembleFaceMatrix(second, first, *transformation, reversed);
      // Map an original [K1, K2] index into the reversed [K2, K1] ordering.
      auto reverse_index = [&](int index)
      {
         return index < first.GetDof() ? second.GetDof() + index
                                       : index - first.GetDof();
      };
      for (int column = 0; column < matrix.Width(); ++column)
         for (int row = 0; row < matrix.Height(); ++row)
         {
            MFEM_VERIFY(std::abs(matrix(row, column) -
                                 reversed(reverse_index(row),
                                          reverse_index(column))) < tolerance,
                        "Classical face orientation changed the matrix.");
         }
   }
}

}

/** @brief Run the parallel cut-diffusion solve or the classical kernel checks.
    @param[in] argc Number of command-line arguments.
    @param[in] argv MPI/application command-line arguments.
    @return Zero after a successful solve or self-test, or two when option
            parsing fails or help is requested. Verification failures abort.

    Every MPI rank follows the same setup, assembly, solve, and reduction path;
    ParaView output is collective, while the final RESULT record is root-only.
    The numbered steps keep geometry, standard strong boundary treatment,
    stabilization, and solver selection separate. MFEM manages MPI/Hypre
    finalization through the initialization helpers. */
int main(int argc, char *argv[])
{
   // 1. Initialize MPI/Hypre and validate options on every rank.
   //    The optional kernel checks exit before any cut-domain solve is built.
   Mpi::Init(argc, argv);
   Hypre::Init();
   Settings settings;
   if (!ReadOptions(argc, argv, settings))
   {
      return 2;
   }
   if (settings.self_test)
   {
      CheckClassicalKernel(2);
      CheckClassicalKernel(3);
      if (Mpi::Root())
      {
         mfem::out << "Classical kernel tests passed in 2D and 3D.\n";
      }
      return 0;
   }

   // 2. Mark the serial background boundaries, then partition by y-coordinate.
   //    This deterministic split also exercises shared cuts and inactive or
   //    empty ranks. Release the replicated mesh after constructing ParMesh.
   Mesh background = BackgroundMesh(settings);
   Array<int> partition(background.GetNE());
   Vector center(settings.dimension);
   for (int element = 0; element < background.GetNE(); ++element)
   {
      background.GetElementCenter(element, center);
      partition[element] =
          std::min(Mpi::WorldSize() - 1, int(center(1) * Mpi::WorldSize()));
   }
   ParMesh mesh(MPI_COMM_WORLD, background, partition);
   background.Clear();

   // 3. Represent the geometry independently of the solution order, then cache
   //    ALGOIM volume rules and active/cut flags from that discrete level set.
   //    CellRules performs global volume and full-cell-anchor checks.
   const int level_set_order = Is(settings.geometry, "band") ? 2 : 4;
   H1_FECollection level_set_collection(level_set_order, settings.dimension);
   ParFiniteElementSpace level_set_space(&mesh, &level_set_collection);
   ParGridFunction level_set(&level_set_space);
   // Adapt the analytic level-set evaluation to MFEM's coefficient interface.
   FunctionCoefficient level_set_coefficient(
       [&](const Vector &position) { return LevelSet(settings, position); });
   level_set.ProjectCoefficient(level_set_coefficient);
   auto backend = CreateCutQuadratureBackend(settings.backend);
   MFEM_VERIFY(backend, "Requested ALGOIM backend is unavailable.");
   CellRules cells(mesh, level_set, *backend, settings);

   // 4. Create the scalar H1 solution space and prepare ghost-face selection.
   //    Preparation includes MPI neighbors; later callbacks use these lookup
   //    data to select active-active faces with at least one cut neighbor.
   H1_FECollection collection(settings.order, settings.dimension);
   ParFiniteElementSpace space(&mesh, &collection);
   auto selection =
       std::make_shared<GhostFaceSelection>(space, cells.active, cells.cut);

   // 5. Impose g_D=x through standard MFEM boundary projection on attributes
   //    1 and 2. Add only globally unsupported DOFs to the elimination list;
   //    exterior nodes that support active cells remain part of the solve.
   Array<int> boundary_marker(3);
   boundary_marker = 0;
   boundary_marker[0] = boundary_marker[1] = 1;
   ParGridFunction solution(&space);
   solution = 0.0;
   // Supply Dirichlet data as a physical-coordinate coefficient, not a penalty.
   FunctionCoefficient boundary_value([](const Vector &position)
                                      { return position(0); });
   solution.ProjectBdrCoefficient(boundary_value, boundary_marker);
   Array<int> essential;
   const long long inactive_dofs =
       EssentialDofs(space, cells, boundary_marker, solution, essential);

   // 6. Assemble the physical-volume load with the same cached cut rules.
   //    The forcing callback returns 0 for the patch, 1 for diffusion, and
   //    -kappa*Delta(u_exact) for the straight-band manufactured problem.
   FunctionCoefficient forcing(
       [&](const Vector &position)
       {
          if (Is(settings.problem, "patch"))
          {
             return real_t(0.0);
          }
          if (Is(settings.problem, "diffusion"))
          {
             return real_t(1.0);
          }
          const real_t pi = std::acos(-1.0),
                       width = settings.upper - settings.lower;
          return settings.diffusivity * pi * pi *
                 (1.0 + 1.0 / (width * width) +
                  (settings.dimension == 3 ? 1.0 : 0.0)) *
                 Oscillation(settings, position);
       });
   ConstantCoefficient diffusivity(settings.diffusivity);
   ParLinearForm load(&space);
   load.AddDomainIntegrator(new CutLoadIntegrator(forcing, cells));
   load.Assemble();

   // 7. Assemble physical diffusion plus the selected full-face ghost penalty.
   //    DIAG_ONE is needed for inactive rows, whose original diagonal is zero.
   //    Concrete ParBilinearForm assembly includes shared faces without keeping
   //    the neighbor-neighbor block on each rank.
   ParBilinearForm form(&space);
   form.SetAssemblyLevel(AssemblyLevel::LEGACY);
   form.SetDiagonalPolicy(Operator::DIAG_ONE);
   form.KeepNbrBlock(false);
   form.AddDomainIntegrator(new CutDiffusionIntegrator(diffusivity, cells));
   AddStabilization(form, selection, settings);
   form.Assemble();
   form.Finalize();

   // 8. Build the true-DOF linear system and eliminate prescribed values using
   //    MFEM's standard strong-BC path, including the associated RHS lifting.
   OperatorPtr system_matrix;
   Vector system_solution, system_rhs;
   form.FormLinearSystem(essential, solution, load, system_matrix,
                         system_solution, system_rhs);

   // 9. Solve the same distributed system with MUMPS or unpreconditioned CG.
   //    MUMPS is a direct reference solve; no preconditioner is installed.
   //    The iteration count remains zero for the direct-solver path.
   int iterations = 0;
   if (Is(settings.solver, "mumps"))
   {
#ifdef MFEM_USE_MUMPS
      MUMPSSolver solver(mesh.GetComm());
      solver.SetPrintLevel(0);
      solver.SetMatrixSymType(MUMPSSolver::SYMMETRIC_POSITIVE_DEFINITE);
      solver.SetOperator(*system_matrix);
      solver.Mult(system_rhs, system_solution);
#endif
   }
   else
   {
      CGSolver solver(mesh.GetComm());
      solver.SetRelTol(settings.relative_tolerance);
      solver.SetAbsTol(0.0);
      solver.SetMaxIter(settings.max_iterations);
      solver.SetPrintLevel(0);
      solver.SetOperator(*system_matrix);
      solver.Mult(system_rhs, system_solution);
      iterations = solver.GetNumIterations();
      MFEM_VERIFY(solver.GetConverged(),
                  "Unpreconditioned CG did not converge; use MUMPS for the "
                  "reference solve.");
   }

   // 10. Independently check the true-system residual before recovering the
   //     grid function. For zero RHS, the residual check is absolute instead
   //     of relative. Recovery restores the prescribed boundary values.
   Vector residual(system_rhs.Size());
   system_matrix->Mult(system_solution, residual);
   residual -= system_rhs;
   const real_t rhs_norm =
       std::sqrt(InnerProduct(mesh.GetComm(), system_rhs, system_rhs));
   const real_t relative_residual =
       std::sqrt(InnerProduct(mesh.GetComm(), residual, residual)) /
       (rhs_norm > 0.0 ? rhs_norm : 1.0);
   MFEM_VERIFY(std::isfinite(relative_residual) &&
                   relative_residual <
                       std::max(1.e-8, 20 * settings.relative_tolerance),
               "Independent linear-system residual check failed.");
   form.RecoverFEMSolution(system_solution, load, solution);

   // 11. Measure physical-domain quantities and assemble a separate, unmodified
   //     ghost matrix for its energy. Do not eliminate boundary rows from this
   //     diagnostic matrix. The linear patch must also reproduce its exact
   //     solution and have negligible stabilization energy.
   const Diagnostics diagnostics = Measure(mesh, solution, cells, settings);
   real_t ghost_energy = 0.0;
   if (settings.ghost_penalty)
   {
      ParBilinearForm ghost_form(&space);
      ghost_form.KeepNbrBlock(false);
      AddStabilization(ghost_form, selection, settings);
      ghost_form.Assemble();
      ghost_form.Finalize();
      std::unique_ptr<HypreParMatrix> ghost_matrix(
          ghost_form.ParallelAssemble());
      Vector true_solution, image(space.GetTrueVSize());
      solution.GetTrueDofs(true_solution);
      ghost_matrix->Mult(true_solution, image);
      ghost_energy = InnerProduct(mesh.GetComm(), true_solution, image);
   }
   if (Is(settings.problem, "patch"))
   {
      MFEM_VERIFY(diagnostics.l2_error < 1.e-7 &&
                      diagnostics.h1_error < 1.e-6 &&
                      std::abs(ghost_energy) < 1.e-8,
                  "Linear patch reproduction failed.");
   }

   // 12. Optionally save both fields collectively on the background mesh.
   //     Clip by level_set > 0 in ParaView to hide the exterior extension.
   //     Reduce output errors so a failed rank cannot be reported as success.
   if (settings.paraview)
   {
      ParaViewDataCollection output("cut_diffusion", &mesh);
      output.SetPrefixPath(settings.output_directory);
      output.SetLevelsOfDetail(std::max(settings.order, level_set_order));
      output.SetDataFormat(VTKFormat::BINARY);
      output.SetHighOrderOutput(true);
      output.RegisterField("level_set", &level_set);
      output.RegisterField("solution", &solution);
      output.SetCycle(0);
      output.SetTime(0.0);
      output.Save();
      int local_error = output.Error(), global_error;
      MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX,
                    mesh.GetComm());
      MFEM_VERIFY(global_error == DataCollection::No_Error,
                  "ParaView output failed.");
   }

   // 13. Reduce ownership/constraint statistics on every rank, then print one
   //     machine-readable RESULT record. Essential counts include inactive
   //     constraints; physical norms exclude them. Keep collective queries
   //     outside the root-only printing branch.
   long long local_essential = essential.Size(), global_essential;
   MPI_Allreduce(&local_essential, &global_essential, 1, MPI_LONG_LONG, MPI_SUM,
                 mesh.GetComm());
   const int locally_inactive = cells.active.Find(1) < 0 ? 1 : 0;
   int inactive_ranks;
   MPI_Allreduce(&locally_inactive, &inactive_ranks, 1, MPI_INT, MPI_SUM,
                 mesh.GetComm());
   const long long global_dofs = space.GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << std::setprecision(16)
                << "RESULT {\"dimension\":" << settings.dimension
                << ",\"order\":" << settings.order
                << ",\"ranks\":" << Mpi::WorldSize() << ",\"solver\":\""
                << settings.solver << "\",\"stabilization\":\""
                << (settings.ghost_penalty ? settings.stabilization : "none")
                << "\",\"volume\":" << cells.volume
                << ",\"active_elements\":" << cells.global_active
                << ",\"cut_elements\":" << cells.global_cut
                << ",\"full_elements\":" << cells.global_full
                << ",\"minimum_fraction\":" << cells.minimum_fraction
                << ",\"true_dofs\":" << global_dofs
                << ",\"active_dofs\":" << global_dofs - inactive_dofs
                << ",\"inactive_dofs\":" << inactive_dofs
                << ",\"essential_dofs\":" << global_essential
                << ",\"inactive_ranks\":" << inactive_ranks
                << ",\"iterations\":" << iterations
                << ",\"relative_residual\":" << relative_residual
                << ",\"solution_norm\":" << diagnostics.norm
                << ",\"solution_integral\":" << diagnostics.integral
                << ",\"diffusion_energy\":" << diagnostics.diffusion_energy
                << ",\"ghost_energy\":" << ghost_energy
                << ",\"l2_error\":" << diagnostics.l2_error
                << ",\"h1_error\":" << diagnostics.h1_error << "}\n";
   }
   return 0;
}
