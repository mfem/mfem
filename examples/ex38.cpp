//                                MFEM Example 38
//
// Compile with: make ex38
//
// Sample runs:
// (since all sample runs require LAPACK or ALGOIM, the * symbol is used to
//  exclude them from the automatically generated internal MFEM tests).
//              * ex38
//              * ex38 -i volumetric1d
//              * ex38 -i surface2d
//              * ex38 -i surface2d -o 4 -r 5 -m 1
//              * ex38 -i volumetric2d
//              * ex38 -i volumetric2d -o 4 -r 5 -m 1
//              * ex38 -i surface3d
//              * ex38 -i surface3d -o 3 -r 4 -m 1
//              * ex38 -i volumetric3d
//              * ex38 -i volumetric3d -o 3 -r 4 -m 1
//
// Description: Integrate functions over implicit interfaces and subdomains
//              bounded by implicit interfaces using cut quadrature.

#include "mfem.hpp"
#include <cmath>
#include <cstring>
#include <iomanip>

using namespace mfem;
using namespace std;

namespace
{

enum class IntegrationType
{
   Volumetric1D,
   Surface2D,
   Volumetric2D,
   Surface3D,
   Volumetric3D
};

bool ParseIntegrationType(const char *name, IntegrationType &type)
{
   if (!strcmp(name, "volumetric1d") || !strcmp(name, "Volumetric1D"))
   { type = IntegrationType::Volumetric1D; }
   else if (!strcmp(name, "surface2d") || !strcmp(name, "Surface2D"))
   { type = IntegrationType::Surface2D; }
   else if (!strcmp(name, "volumetric2d") || !strcmp(name, "Volumetric2D"))
   { type = IntegrationType::Volumetric2D; }
   else if (!strcmp(name, "surface3d") || !strcmp(name, "Surface3D"))
   { type = IntegrationType::Surface3D; }
   else if (!strcmp(name, "volumetric3d") || !strcmp(name, "Volumetric3D"))
   { type = IntegrationType::Volumetric3D; }
   else { return false; }
   return true;
}

int Dimension(IntegrationType type)
{
   if (type == IntegrationType::Volumetric1D) { return 1; }
   return type == IntegrationType::Surface2D ||
          type == IntegrationType::Volumetric2D ? 2 : 3;
}

bool HasVolumeIntegral(IntegrationType type)
{
   return type == IntegrationType::Volumetric1D ||
          type == IntegrationType::Volumetric2D ||
          type == IntegrationType::Volumetric3D;
}

real_t LevelSet(IntegrationType type, const Vector &point)
{
   switch (type)
   {
      case IntegrationType::Volumetric1D:
         return 0.55 - point(0);
      case IntegrationType::Surface2D:
         return 1.0 - point(0)*point(0) - point(1)*point(1);
      case IntegrationType::Volumetric2D:
         return 1.0 - std::pow(point(0)/1.5, 2) -
                std::pow(point(1)/0.75, 2);
      case IntegrationType::Surface3D:
         return 1.0 - point*point;
      case IntegrationType::Volumetric3D:
         return 1.0 - std::pow(point(0)/1.5, 2) -
                std::pow(point(1)/0.75, 2) - std::pow(point(2)/0.5, 2);
   }
   return 0.0;
}

real_t Integrand(IntegrationType type, const Vector &point)
{
   switch (type)
   {
      case IntegrationType::Volumetric1D:
         return point(0)*point(0);
      case IntegrationType::Surface2D:
         return 3.0*point(0)*point(0) - point(1)*point(1);
      case IntegrationType::Surface3D:
         return 4.0 - 3.0*point(0)*point(0) +
                2.0*point(1)*point(1) - point(2)*point(2);
      case IntegrationType::Volumetric2D:
      case IntegrationType::Volumetric3D:
         return 1.0;
   }
   return 0.0;
}

real_t ExactSurfaceIntegral(IntegrationType type)
{
   const real_t pi = std::acos(-1.0);
   switch (type)
   {
      case IntegrationType::Volumetric1D: return 0.3025;
      case IntegrationType::Surface2D: return 2.0*pi;
      case IntegrationType::Volumetric2D: return 7.26633616541076;
      case IntegrationType::Surface3D: return 40.0*pi/3.0;
      case IntegrationType::Volumetric3D: return 9.90182151329315;
   }
   return 0.0;
}

real_t ExactVolumeIntegral(IntegrationType type)
{
   const real_t pi = std::acos(-1.0);
   if (type == IntegrationType::Volumetric1D)
   { return std::pow(0.55, 3)/3.0; }
   return type == IntegrationType::Volumetric2D ? 9.0*pi/8.0 : 3.0*pi/4.0;
}

}

int main(int argc, char *argv[])
{
   int refinements = 3, order = 2, method = 0;
   const char *integration_type = "surface2d";
   bool visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Nominal cut quadrature order.");
   args.AddOption(&refinements, "-r", "--refine", "Uniform refinement levels.");
   args.AddOption(&method, "-m", "--method",
                  "Cut integration method: 0 for moments, 1 for ALGOIM.");
   args.AddOption(&integration_type, "-i", "--integration-type",
                  "Integration type: volumetric1d, surface2d, volumetric2d, "
                  "surface3d, or volumetric3d.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Display the polynomial level set.");
   args.Parse();
   IntegrationType type = IntegrationType::Surface2D;
   if (!args.Good() || refinements < 0 || order < 0 ||
       (method != 0 && method != 1) ||
       !ParseIntegrationType(integration_type, type))
   {
      args.PrintUsage(cout);
      return 1;
   }

   auto backend = CreateCutQuadratureBackend(method ? "algoim-auto" : "moment");
   if (!backend)
   {
      cerr << "The selected cut-quadrature backend is unavailable.\n";
      return MFEM_SKIP_RETURN_VALUE;
   }

   const int dimension = Dimension(type);
   if (dimension == 1 && method == 1)
   {
      MFEM_ABORT("The ALGOIM cut-quadrature backend does not support "
                 "one-dimensional elements.");
   }
   Mesh mesh = dimension == 1 ? Mesh::MakeCartesian1D(4) :
               (dimension == 2 ?
                Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                                      true, 3.2, 3.2) :
                Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                                      3.2, 3.2, 3.2));
   for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
   {
      for (int axis = 0; axis < dimension; ++axis)
      { mesh.GetVertex(vertex)[axis] -= dimension == 1 ? 0.0 : 1.6; }
   }
   for (int level = 0; level < refinements; ++level)
   { mesh.UniformRefinement(); }

   FunctionCoefficient coefficient([type](const Vector &point)
   { return LevelSet(type, point); });
   H1_FECollection collection(2, dimension);
   FiniteElementSpace space(&mesh, &collection);
   GridFunction field(&space);
   field.ProjectCoefficient(coefficient);

   CutQuadratureGenerator generator(*backend);
   real_t surface_integral = 0.0, volume_integral = 0.0;
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      auto &transformation = *mesh.GetElementTransformation(element);
      PolynomialLevelSet level_set;
      level_set.Set(field, element);
      const int integrals = HasVolumeIntegral(type) ? 2 : 1;
      for (int integral = 0; integral < integrals; ++integral)
      {
         const bool surface = integral == 0;
         const auto domain = surface ? CutDomain::Surface(level_set) :
                             CutDomain::Positive(level_set);
         CutQuadrature quadrature;
         auto report = generator.Generate(transformation.GetGeometryType(),
                                          domain, order, quadrature);
         CutQuadratureMapping mapping;
         if (report.Succeeded())
         {
            report = MapCutQuadrature(
                        transformation, quadrature,
                        CutQuadratureMapping::PHYSICAL_POINTS |
                        CutQuadratureMapping::MEASURE_SCALING, mapping);
         }
         if (!report.Succeeded())
         {
            cerr << report.message << '\n';
            return 2;
         }
         Vector point(dimension);
         for (int index = 0; index < quadrature.GetNPoints(); ++index)
         {
            mapping.physical_points.GetColumn(index, point);
            const real_t physical_weight =
               quadrature.rule.IntPoint(index).weight *
               mapping.measure_scaling(index);
            (surface ? surface_integral : volume_integral) +=
               physical_weight * Integrand(type, point);
         }
      }
   }

   const real_t exact_surface = ExactSurfaceIntegral(type);
   cout << setprecision(16)
        << "Surface: " << surface_integral
        << ", error: " << abs(surface_integral - exact_surface) << '\n';
   if (HasVolumeIntegral(type))
   {
      const real_t exact_volume = ExactVolumeIntegral(type);
      cout << "Volume: " << volume_integral
           << ", error: " << abs(volume_integral - exact_volume) << '\n';
   }

   if (visualization)
   {
      socketstream socket("localhost", 19916);
      socket << "solution\n" << mesh << field << flush;
   }
   return 0;
}
