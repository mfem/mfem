/** @file projection_ghost_penalty_parallel.cpp
    @brief MPI example of a fully assembled projection ghost penalty.

    This is the parallel counterpart of projection_ghost_penalty_serial.cpp.
    Both files use the same eight application stages and numerical settings;
    this file makes collective selection, shared-face assembly, and true-DOF
    algebra explicit. It always initializes MPI, including on one rank.

    The physical domain is x[0] < test_level_set. The ghost penalty integrates
    complete background-cell patches, not cut-cell intersections. This example
    measures stabilization of an interpolated field, not a PDE solution. A real
    CutFEM application also needs physical-domain terms, boundary conditions,
    and constraints/elimination for inactive DOFs.

    @par Sample runs (from a build with MFEM_USE_LAPACK and MFEM_USE_MPI)
    @code{.sh}
    cmake --build . --target projection_ghost_penalty_parallel
    mpiexec -n 4 miniapps/mtop/projection_ghost_penalty_parallel --order 2
    mpiexec -n 4 miniapps/mtop/projection_ghost_penalty_parallel \
      --vdim 3 --ordering byVDIM
    mpiexec -n 4 miniapps/mtop/projection_ghost_penalty_parallel \
      --element-type tensor --dimension 3 --geometry-order 2 --warp 0.05 \
      --order 2 --projection-space p --projection-degree 1
    @endcode

    There are no --serial/--parallel switches. Use your site's MPI launcher
    settings and --help for application options. Curved/higher-order geometry
    requires explicit p projection; AUTO does not fall back from affine Q.
    See doc/projection-ghost-penalty.md for the formulation and API contracts.
 */

#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <string>

using namespace mfem;

struct DemoOptions
{
   const char *mesh_file = "";
   const char *element_type = "simplex";
   const char *ordering = "byNODES";
   const char *projection_space = "auto";
   int dimension = 2;
   int order = 2;
   int components = 1;
   int geometry_order = 1;
   int layers = 0;
   int projection_degree = -1;
   int quadrature_extra = 2;
   real_t gamma = 1.0;
   real_t svd_rtol = 0.0;
   real_t level_set = 0.43;
   real_t warp = 0.0;
   bool verbose = false;
};

bool ReadOptions(int argc, char *argv[], DemoOptions &settings);
Mesh CreateBackgroundMesh(const DemoOptions &settings);
void MarkActiveAndCutElements(Mesh &mesh, const DemoOptions &settings,
                              Array<int> &active, Array<int> &cut);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and create identical geometry before partitioning it.
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int rank = Mpi::WorldRank(), ranks = Mpi::WorldSize();
   DemoOptions settings;
   if (!ReadOptions(argc, argv, settings)) { return 1; }
   Mesh background_mesh = CreateBackgroundMesh(settings);
   const int dimension = background_mesh.Dimension();
   int geometry_order = settings.geometry_order;
   if (background_mesh.GetNE())
   { geometry_order = background_mesh.GetElementTransformation(0)->Order(); }
   Array<int> partitioning(background_mesh.GetNE());
   for (int element = 0; element < background_mesh.GetNE(); ++element)
   {
      partitioning[element] = static_cast<int>(
         static_cast<long long>(element) * ranks / background_mesh.GetNE());
   }
   ParMesh mesh(MPI_COMM_WORLD, background_mesh, partitioning);

   // 2. Use a concrete parallel space. Mesh/collection/space lifetimes enclose
   //    those of the selection, form, grid function, and true-DOF matrix.
   H1_FECollection collection(settings.order, dimension);
   const auto ordering = std::string(settings.ordering) == "byNODES" ?
                         Ordering::byNODES : Ordering::byVDIM;
   ParFiniteElementSpace space(&mesh, &collection, settings.components,
                               ordering);

   // 3. Each rank supplies flags for its locally owned elements only.
   Array<int> active(mesh.GetNE()), cut(mesh.GetNE());
   MarkActiveAndCutElements(mesh, settings, active, cut);

   // 4. Every rank must use the same projection and stabilization settings.
   ProjectionGhostPenaltyOptions options;
   options.gamma = settings.gamma;
   options.projection_degree = settings.projection_degree;
   options.quadrature_extra = settings.quadrature_extra;
   options.svd_rtol = settings.svd_rtol;
   options.verbose = settings.verbose;
   if (std::string(settings.projection_space) == "p")
   { options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE; }
   else if (std::string(settings.projection_space) == "q")
   { options.projection_space = ProjectionGhostPenaltySpace::TENSOR_PRODUCT; }

   // 5. Selection is collective, even on inactive/empty ranks. It exchanges
   //    neighbor flags and geometry before any face-integrator callbacks.
   //    Rebuild both after changing mesh, geometry, space, or flags.
   auto selection =
      std::make_shared<GhostFaceSelection>(space, active, cut, settings.layers);

   // 6. The form owns the integrator, which shares selection ownership.
   //    Visit both sides of shared faces, but discard neighbor-row blocks.
   ParBilinearForm form(&space);
   form.SetAssemblyLevel(AssemblyLevel::LEGACY);
   form.KeepNbrBlock(false);
   if (settings.components == 1)
   {
      form.AddInteriorFaceIntegrator(
         new ScalarProjectionGhostPenaltyIntegrator(selection, options));
   }
   else
   {
      form.AddInteriorFaceIntegrator(
         new VectorProjectionGhostPenaltyIntegrator(selection, options));
   }

   // 7. Call Assemble() on ParBilinearForm directly: the base-class method is
   //    nonvirtual and would omit shared faces. Then assemble on true DOFs.
   form.Assemble();
   form.Finalize();
   std::unique_ptr<HypreParMatrix> matrix(form.ParallelAssemble());

   // 8. Evaluate the same interpolant as the serial example, using owned true
   //    DOFs and global reductions. The ghost operator alone is not a PDE.
   VectorFunctionCoefficient coefficient(settings.components,
      [components = settings.components](const Vector &position, Vector &value)
   {
      for (int component = 0; component < components; ++component)
      {
         value(component) =
            (component + 1) * std::sin(3.1 * position(0) + position(1));
      }
   });
   ParGridFunction field(&space);
   field.ProjectCoefficient(coefficient);
   Vector values, action(space.GetTrueVSize());
   field.GetTrueDofs(values);
   matrix->Mult(values, action);
   const real_t local_energy = values * action;
   real_t energy;
   MPI_Allreduce(&local_energy, &energy, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
   const real_t local_faces = selection->GetNumLocalFaces() +
                             0.5 * selection->GetNumSharedFaces();
   real_t selected_faces;
   MPI_Allreduce(&local_faces, &selected_faces, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
   const auto true_dofs = space.GlobalTrueVSize();
   if (rank == 0)
   {
      std::cout << std::setprecision(16)
                << "mode=MPI ranks=" << ranks << " dimension=" << dimension
                << " order=" << settings.order
                << " vdim=" << settings.components
                << " geometry_order=" << geometry_order
                << " projection_space=" << settings.projection_space
                << " true_dofs=" << true_dofs
                << " selected_faces=" << selected_faces
                << " energy=" << energy << '\n';
   }
   return std::isfinite(energy) ? 0 : 2;
}

bool ReadOptions(int argc, char *argv[], DemoOptions &settings)
{
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&settings.mesh_file, "-m", "--mesh",
                       "Mesh path; empty generates a grid.");
   arguments.AddOption(&settings.element_type, "-et", "--element-type",
                       "Generated cells: simplex or tensor; "
                       "ignored with --mesh.");
   arguments.AddOption(&settings.projection_space, "-ps", "--projection-space",
                       "auto, p (physical P), or q (common affine Q frame).");
   arguments.AddOption(&settings.order, "-o", "--order", "Uniform H1 order.");
   arguments.AddOption(&settings.geometry_order, "-go", "--geometry-order",
                       "Generated geometry order; independent of --order.");
   arguments.AddOption(&settings.warp, "-w", "--warp",
                       "Generated polynomial deformation, |warp|<=0.1.");
   arguments.AddOption(&settings.components, "-v", "--vdim",
                       "Number of independent components.");
   arguments.AddOption(&settings.dimension, "-d", "--dimension",
                       "Generated mesh dimension (2 or 3).");
   arguments.AddOption(&settings.ordering, "-ord", "--ordering",
                       "byNODES or byVDIM.");
   arguments.AddOption(&settings.gamma, "-g", "--gamma",
                       "Nonnegative stabilization strength.");
   arguments.AddOption(&settings.projection_degree, "-p", "--projection-degree",
                       "Common patch degree; -1 uses H1 order.");
   arguments.AddOption(&settings.svd_rtol, "-tol", "--svd-rtol",
                       "Relative rank threshold; zero is automatic.");
   arguments.AddOption(&settings.quadrature_extra, "-q", "--quad-extra",
                       "Additional quadrature order.");
   arguments.AddOption(&settings.layers, "-l", "--layers",
                       "Active-cell adjacency layers extending the cut band.");
   arguments.AddOption(&settings.level_set, "-ls", "--test-level-set",
                       "Physical domain x[0] < this value.");
   arguments.AddOption(&settings.verbose, "-verb", "--verbose", "-quiet",
                       "--quiet", "Print per-patch SVD diagnostics.");
   arguments.Parse();
   if (!arguments.Good())
   {
      if (Mpi::Root()) { arguments.PrintUsage(std::cerr); }
      return false;
   }
   MFEM_VERIFY(settings.order >= 1 && settings.components >= 1 &&
               (settings.dimension == 2 || settings.dimension == 3) &&
               std::isfinite(settings.level_set) &&
               settings.geometry_order >= 1 &&
               std::isfinite(settings.warp) && std::abs(settings.warp) <= 0.1,
               "Invalid order, dimensions, level set, or geometry settings.");
   MFEM_VERIFY(!settings.mesh_file[0] ||
               (settings.geometry_order == 1 && settings.warp == 0.0),
               "Geometry generation options cannot be used with --mesh.");
   const std::string ordering(settings.ordering);
   const std::string element(settings.element_type);
   const std::string projection(settings.projection_space);
   MFEM_VERIFY(ordering == "byNODES" || ordering == "byVDIM",
               "Ordering must be byNODES or byVDIM.");
   MFEM_VERIFY(element == "simplex" || element == "tensor",
               "Element type must be simplex or tensor.");
   MFEM_VERIFY(projection == "auto" || projection == "p" || projection == "q",
               "Projection space must be auto, p, or q.");
   return true;
}

Mesh CreateBackgroundMesh(const DemoOptions &settings)
{
   const bool tensor = std::string(settings.element_type) == "tensor";
   Mesh mesh;
   if (settings.mesh_file[0]) { mesh = Mesh(settings.mesh_file); }
   else if (settings.dimension == 2)
   {
      mesh = Mesh::MakeCartesian2D(8, 2, tensor ? Element::QUADRILATERAL :
                                   Element::TRIANGLE, true);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(8, 1, 1, tensor ? Element::HEXAHEDRON :
                                   Element::TETRAHEDRON);
   }
   if (!settings.mesh_file[0] &&
       (settings.geometry_order > 1 || settings.warp != 0.0))
   {
      mesh.SetCurvature(settings.geometry_order);
      mesh.Transform([&settings](const Vector &position, Vector &value)
      {
         value = position;
         const real_t first = position(0), second = position(1);
         value(0) += settings.warp * first * second;
         value(1) += settings.warp * first * second;
         if (settings.geometry_order >= 2)
         {
            value(0) += settings.warp * second * (1.0 - second);
            value(1) += 0.3 * settings.warp * first * first;
         }
         if (settings.geometry_order >= 3)
         { value(0) += 0.1 * settings.warp * second * second * second; }
         if (position.Size() == 3)
         {
            value(2) += settings.warp * second * position(2);
            if (settings.geometry_order >= 2)
            { value(2) += 0.2 * settings.warp * first * first; }
         }
      });
   }
   return mesh;
}

void MarkActiveAndCutElements(Mesh &mesh, const DemoOptions &settings,
                              Array<int> &active, Array<int> &cut)
{
   // Toy classification: sampled extrema are not certified curved-cell
   // intersections. A real application supplies its own geometry markers.
   GeometryRefiner refiner;
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      auto *trans = mesh.GetElementTransformation(element);
      const bool affine_grid = !settings.mesh_file[0] && settings.warp == 0.0 &&
                               settings.geometry_order == 1;
      const auto &samples = affine_grid ?
         *Geometries.GetVertices(trans->GetGeometryType()) :
         refiner.Refine(trans->GetGeometryType(), 2 * trans->Order())->RefPts;
      Vector position(mesh.SpaceDimension());
      real_t minimum = infinity(), maximum = -infinity();
      for (int sample = 0; sample < samples.GetNPoints(); ++sample)
      {
         trans->Transform(samples.IntPoint(sample), position);
         minimum = std::min(minimum, position(0));
         maximum = std::max(maximum, position(0));
      }
      active[element] = minimum < settings.level_set;
      cut[element] = minimum < settings.level_set &&
                     maximum > settings.level_set;
   }
}
