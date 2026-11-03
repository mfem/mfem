/** @file projection_ghost_penalty.cpp
    @brief Serial/MPI example of a fully assembled projection ghost penalty.

    This miniapp assembles only the stabilization matrix, not a complete PDE.
    It marks the physical domain x[0] < test_level_set on a polynomial
    background mesh, selects active interior cell pairs near the cut, and
    penalizes residuals from a common two-cell polynomial projection. Both
    complete cells are integrated; no cut-cell quadrature is performed.

    With vdim=1 it uses ScalarProjectionGhostPenaltyIntegrator; larger vector
    dimensions use independent componentwise penalties. The reported energy
    is the assembled quadratic form of the H1 interpolant of the smooth field
    (component+1)*sin(3.1*x[0]+x[1]). This is an assembly diagnostic, not a
    solution error or a coercivity test: the penalty has a polynomial nullspace.

    @par Build and sample runs
    Run from the workspace root, using the existing configured build. LAPACK
    enables the integrators automatically; MPI runs also require MFEM_USE_MPI.
    The environment script selects the intended compiler, libraries, and MPI.
    @code{.sh}
    source ./set_up_local.sh
    cmake -S mfem -B build -DMFEM_USE_LAPACK=ON
    cmake --build build --parallel 16 --target projection_ghost_penalty
    build/miniapps/mtop/projection_ghost_penalty --serial --order 2 --vdim 3
    build/miniapps/mtop/projection_ghost_penalty --serial --dimension 3 \
      --element-type tensor --order 2 --vdim 3 --ordering byVDIM
    build/miniapps/mtop/projection_ghost_penalty --serial --dimension 3 \
      --element-type tensor --geometry-order 2 --warp 0.05 --order 2 \
      --projection-space p --projection-degree 1
    mpiexec -n 4 build/miniapps/mtop/projection_ghost_penalty \
      --parallel --order 2 --vdim 3
    @endcode
    On this host, local Hydra runs may additionally require
    `mpiexec -launcher fork -rmk user -n 4 ...`; use site/scheduler settings
    instead for multi-node runs. Without --parallel, each invocation is serial,
    even if launched by mpiexec.

    @par Inputs and defaults
    With no --mesh, generate an 8-by-2 triangle mesh in 2D or an 8-by-1-by-1
    tetrahedral mesh in 3D. --element-type tensor instead generates quads/hexes.
    A supplied mesh determines its dimension and ignores --element-type.
    Supported cells are conforming polynomial triangles, quads, tets, and hexes.
    Embedded, rational/NURBS, and nonconforming geometries are excluded.
    Defaults are serial mode, dimension=2, order=2, vdim=1, ordering=byNODES,
    test-level-set=0.43, gamma=1, projection-degree=-1 (use the FE order),
    quad-extra=2, svd-rtol=0 (automatic threshold), layers=0, geometry-order=1,
    warp=0, and quiet output. Geometry order is independent of solution order.
    --projection-space auto (default) uses P on simplices and Q on tensor
    cells. Explicit p/q select either space. Q requires compatible affine
    tensor axes; request p explicitly for incompatible quad/hex frames and all
    non-affine/higher-order geometry. There is no automatic fallback to P.
    --geometry-order and --warp apply only to generated meshes, before MPI
    partitioning. Warp adds a small polynomial deformation with amplitude at
    most 0.1 in magnitude; order-one tensor maps can already be non-affine.
    Loaded meshes retain their geometry; nondefault generation controls fail.
    Curved level-set flags sample a reference lattice with 2*k subdivisions:
    they are approximate markers, not certified cell-intersection tests.
    --layers expands the cut band through active-cell adjacency; --verbose
    prints per-patch SVD diagnostics from each rank. See the OptionsParser
    registrations below for all command-line switches.

    @par Output and limitations
    Rank zero prints mode, ranks, dimension, order, geometry_order, vdim, true_dofs,
    selected_faces, and energy. The MPI counts and energy are global; shared
    faces are counted once physically. true_dofs includes the whole background
    FE space, not only active cells. Changing partition or vector ordering
    should preserve the energy to floating-point tolerance. A real CutFEM
    solver must add physical-domain terms, boundary conditions, and appropriate
    treatment of inactive DOFs. See doc/projection-ghost-penalty.md and
    fem/ghost_penalty.hpp for the formulation and public API contracts. */

#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <string>

using namespace mfem;

/** @brief Assemble and measure a scalar or vector ghost-penalty operator.
    @param[in] argc Number of command-line arguments.
    @param[in] argv Options documented in the file overview and parser below.
    @return Zero for finite energy, one when option parsing stops, or two for
            nonfinite energy. Invalid setups also trigger MFEM_VERIFY. */
int main(int argc, char *argv[])
{
   // Parse mesh, FE, selection, and SVD controls before creating MFEM data.
   const char *mesh_file = "";
   const char *ordering_name = "byNODES";
   const char *element_type = "simplex";
   const char *projection_name = "auto";
   int order = 2, components = 1, dimension = 2, layers = 0;
   int geometry_order = 1;
   real_t level_set = 0.43, warp = 0.0;
   bool parallel = false;
   ProjectionGhostPenaltyOptions options;
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&mesh_file, "-m", "--mesh",
                       "Mesh path; empty generates a grid.");
   arguments.AddOption(&element_type, "-et", "--element-type",
                       "Generated cells: simplex or tensor; "
                       "ignored with --mesh.");
   arguments.AddOption(&projection_name, "-ps", "--projection-space",
                       "auto, p (total degree), or q (common tensor frame).");
   arguments.AddOption(&order, "-o", "--order", "Uniform H1 polynomial order.");
   arguments.AddOption(&geometry_order, "-go", "--geometry-order",
                       "Generated geometry order; independent of --order.");
   arguments.AddOption(&warp, "-w", "--warp",
                       "Generated polynomial deformation amplitude, |warp|<=0.1.");
   arguments.AddOption(&components, "-v", "--vdim",
                       "Number of independent components.");
   arguments.AddOption(&dimension, "-d", "--dimension",
                       "Generated mesh dimension (2 or 3).");
   arguments.AddOption(&ordering_name, "-ord", "--ordering",
                       "byNODES or byVDIM.");
   arguments.AddOption(&options.gamma, "-g", "--gamma",
                       "Nonnegative stabilization strength.");
   arguments.AddOption(&options.projection_degree, "-p", "--projection-degree",
                       "Patch degree; -1 uses H1 order.");
   arguments.AddOption(&options.svd_rtol, "-tol", "--svd-rtol",
                       "Relative rank threshold; zero is automatic.");
   arguments.AddOption(&options.quadrature_extra, "-q", "--quad-extra",
                       "Additional quadrature order.");
   arguments.AddOption(&layers, "-l", "--layers",
                       "Connectivity extension layers.");
   arguments.AddOption(&level_set, "-ls", "--test-level-set",
                       "Physical domain x[0] < this value.");
   arguments.AddOption(&parallel, "-par", "--parallel", "-ser", "--serial",
                       "Use MPI or serial assembly.");
   arguments.AddOption(&options.verbose, "-verb", "--verbose", "-quiet",
                       "--quiet", "Print per-patch SVD diagnostics.");
   arguments.Parse();
   if (!arguments.Good())
   {
      arguments.PrintUsage(std::cerr);
      return 1;
   }
   MFEM_VERIFY(
      order >= 1 && components >= 1 && (dimension == 2 || dimension == 3) &&
         std::isfinite(level_set) && geometry_order >= 1 &&
         std::isfinite(warp) && std::abs(warp) <= 0.1,
      "Invalid order, vector dimension, mesh dimension, or level set.");
   MFEM_VERIFY(!mesh_file[0] || (geometry_order == 1 && warp == 0.0),
               "Geometry generation options cannot be used with --mesh.");
   const std::string ordering_string(ordering_name);
   MFEM_VERIFY(ordering_string == "byNODES" || ordering_string == "byVDIM",
               "Ordering must be byNODES or byVDIM.");
   const auto ordering =
      ordering_string == "byNODES" ? Ordering::byNODES : Ordering::byVDIM;
   const std::string element_string(element_type),
         projection_string(projection_name);
   MFEM_VERIFY(element_string == "simplex" || element_string == "tensor",
               "Element type must be simplex or tensor.");
   MFEM_VERIFY(projection_string == "auto" || projection_string == "p" ||
                  projection_string == "q",
               "Projection space must be auto, p, or q.");
   options.projection_space = projection_string == "p"
      ? ProjectionGhostPenaltySpace::TOTAL_DEGREE : projection_string == "q"
      ? ProjectionGhostPenaltySpace::TENSOR_PRODUCT
      : ProjectionGhostPenaltySpace::AUTO;
   const bool tensor = element_string == "tensor";
   int rank = 0, ranks = 1;
   // Initialize MPI only when requested; the same executable can run serially.
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      Mpi::Init(argc, argv);
      Hypre::Init();
      rank = Mpi::WorldRank();
      ranks = Mpi::WorldSize();
   }
#else
   MFEM_VERIFY(!parallel, "This MFEM build has no MPI support.");
#endif
   Mesh serial_mesh = mesh_file[0] ? Mesh(mesh_file) : dimension == 2
      ? Mesh::MakeCartesian2D(8, 2, tensor ? Element::QUADRILATERAL
                             : Element::TRIANGLE, true)
      : Mesh::MakeCartesian3D(8, 1, 1, tensor ? Element::HEXAHEDRON
                             : Element::TETRAHEDRON);
   dimension = serial_mesh.Dimension();
   if (!mesh_file[0] && (geometry_order > 1 || warp != 0.0))
   {
      serial_mesh.SetCurvature(geometry_order);
      serial_mesh.Transform([=](const Vector &position, Vector &value)
      {
         value = position;
         const real_t first = position(0), second = position(1);
         value(0) += warp * first * second;
         value(1) += warp * first * second;
         if (geometry_order >= 2)
         {
            value(0) += warp * second * (1.0 - second);
            value(1) += 0.3 * warp * first * first;
         }
         if (geometry_order >= 3)
         { value(0) += 0.1 * warp * second * second * second; }
         if (dimension == 3)
         {
            value(2) += warp * second * position(2);
            if (geometry_order >= 2)
            { value(2) += 0.2 * warp * first * first; }
         }
      });
   }
   if (serial_mesh.GetNE())
   { geometry_order = serial_mesh.GetElementTransformation(0)->Order(); }
   Mesh *mesh = &serial_mesh;
   std::unique_ptr<Mesh> distributed_mesh;
   // Partition the replicated background mesh by element index for this demo.
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      Array<int> partition(serial_mesh.GetNE());
      for (int element = 0; element < serial_mesh.GetNE(); ++element)
      {
         partition[element] =
            std::min(ranks - 1, element * ranks / serial_mesh.GetNE());
      }
      distributed_mesh.reset(
         new ParMesh(MPI_COMM_WORLD, serial_mesh, partition));
      mesh = distributed_mesh.get();
   }
#endif
   // Keep the mesh, collection, and space alive until forms/selection are gone.
   H1_FECollection collection(order, dimension);
   std::unique_ptr<FiniteElementSpace> space;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      space.reset(new ParFiniteElementSpace(static_cast<ParMesh *>(mesh),
                                            &collection, components, ordering));
   }
   else
#endif
   {
      space.reset(
         new FiniteElementSpace(mesh, &collection, components, ordering));
   }
   // Curved level-set flags are sampled approximations, not cut-cell quadrature.
   // The original vertex classification is retained for unwarped affine grids.
   Array<int> active(mesh->GetNE()), cut(mesh->GetNE());
   GeometryRefiner geometry_refiner;
   for (int element = 0; element < mesh->GetNE(); ++element)
   {
      auto *trans = mesh->GetElementTransformation(element);
      const auto &samples = !mesh_file[0] && warp == 0.0 && geometry_order == 1 ?
         *Geometries.GetVertices(trans->GetGeometryType()) :
         geometry_refiner.Refine(trans->GetGeometryType(),
                                 2 * trans->Order())->RefPts;
      Vector position(dimension);
      real_t minimum = infinity(), maximum = -infinity();
      for (int sample = 0; sample < samples.GetNPoints(); ++sample)
      {
         trans->Transform(samples.IntPoint(sample), position);
         minimum = std::min(minimum, position(0));
         maximum = std::max(maximum, position(0));
      }
      active[element] = minimum < level_set;
      cut[element] = minimum < level_set && maximum > level_set;
   }
   // Construct collectively in MPI, including ranks with no active elements.
   // Neighbor marker exchange finishes here, before any face integrator calls.
   auto selection =
      std::make_shared<GhostFaceSelection>(*space, active, cut, layers);
   std::unique_ptr<BilinearForm> form;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      form.reset(new ParBilinearForm(
         static_cast<ParFiniteElementSpace *>(space.get())));
   }
   else
#endif
   {
      form.reset(new BilinearForm(space.get()));
   }
   // Full assembly uses the existing local/shared-face sparse assembly path.
   // The form owns its integrator, which shares selection ownership.
   form->SetAssemblyLevel(AssemblyLevel::LEGACY);
   if (components == 1)
   {
      form->AddInteriorFaceIntegrator(
         new ScalarProjectionGhostPenaltyIntegrator(selection, options));
   }
   else
   {
      form->AddInteriorFaceIntegrator(
         new VectorProjectionGhostPenaltyIntegrator(selection, options));
   }
   // Assemble is nonvirtual: dispatch to ParBilinearForm for shared faces.
#ifdef MFEM_USE_MPI
   if (parallel) { static_cast<ParBilinearForm *>(form.get())->Assemble(); }
   else
#endif
   {
      form->Assemble();
   }
   form->Finalize();
   // Interpolate a smooth, non-polynomial field to exercise nonzero penalties.
   VectorFunctionCoefficient coefficient(
      components,
      [components](const Vector &position, Vector &value)
      {
         for (int component = 0; component < components; ++component)
         {
            value(component) =
               (component + 1) * std::sin(3.1 * position(0) + position(1));
         }
      });
   real_t energy = 0.0;
   // Each MPI-shared face is visited twice, but counts as one physical face.
   real_t selected_faces =
      selection->GetNumLocalFaces() + 0.5 * selection->GetNumSharedFaces();
   long long true_dofs = space->GetTrueVSize();
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto *parallel_space = static_cast<ParFiniteElementSpace *>(space.get());
      ParGridFunction field(parallel_space);
      field.ProjectCoefficient(coefficient);
      Vector values, action(parallel_space->GetTrueVSize());
      field.GetTrueDofs(values);
      // Assemble the true-DOF operator, then sum local dot products.
      std::unique_ptr<HypreParMatrix> matrix(
         static_cast<ParBilinearForm *>(form.get())->ParallelAssemble());
      matrix->Mult(values, action);
      const real_t local_energy = values * action;
      MPI_Allreduce(&local_energy, &energy, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      const real_t local_faces = selected_faces;
      MPI_Allreduce(&local_faces, &selected_faces, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
      true_dofs = parallel_space->GlobalTrueVSize();
   }
   else
#endif
   {
      // In conforming serial H1, evaluate with the assembled local matrix.
      GridFunction field(space.get());
      field.ProjectCoefficient(coefficient);
      Vector action(space->GetVSize());
      form->SpMat().Mult(field, action);
      energy = field * action;
   }
   // Report one global summary; an empty selection has zero energy.
   if (rank == 0)
   {
      std::cout << std::setprecision(16)
                << "mode=" << (parallel ? "MPI" : "serial")
                << " ranks=" << ranks << " dimension=" << dimension
                << " order=" << order << " vdim=" << components
                << " geometry_order=" << geometry_order
                << " projection_space=" << projection_name
                << " true_dofs=" << true_dofs
                << " selected_faces=" << selected_faces << " energy=" << energy
                << '\n';
   }
   return std::isfinite(energy) ? 0 : 2;
}
