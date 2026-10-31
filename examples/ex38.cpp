//                                MFEM Example 38
//
// Compile with: make ex38
//
// Sample runs:
// (since all sample runs require ALGOIM, the * symbol is used to
//  exclude them from the automatically generated internal MFEM tests).
//              * ex38
//              * ex38 -dim 3
//              * ex38 -dim 2 -o 14 -r 2
//              * ex38 -dim 3 -o 14 -r 2

#include "mfem.hpp"
#include <cmath>
#include <iomanip>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   int dimension = 2, refinements = 1, order = 10;
   bool visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&dimension, "-dim", "--dimension", "Spatial dimension: 2 or 3.");
   args.AddOption(&refinements, "-r", "--refinements", "Uniform refinement levels.");
   args.AddOption(&order, "-o", "--order", "Nominal cut quadrature order.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Display the polynomial level set.");
   args.Parse();
   if (!args.Good() || (dimension != 2 && dimension != 3) || refinements < 0)
   {
      args.PrintUsage(cout);
      return 1;
   }
   auto backend = CreateCutQuadratureBackend("algoim");
   if (!backend)
   {
      cerr << "Example 38 requires MFEM_USE_ALGOIM.\n";
      return MFEM_SKIP_RETURN_VALUE;
   }
   Mesh mesh = dimension == 2 ?
               Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 2.4, 2.4) :
               Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 2.4, 2.4, 2.4);
   for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
   {
      for (int axis = 0; axis < dimension; ++axis)
      { mesh.GetVertex(vertex)[axis] -= 1.2; }
   }
   for (int level = 0; level < refinements; ++level) { mesh.UniformRefinement(); }
   FunctionCoefficient coefficient([](const Vector &point)
   { return 1.0 - point * point; });
   H1_FECollection collection(2, dimension);
   FiniteElementSpace space(&mesh, &collection);
   GridFunction field(&space);
   field.ProjectCoefficient(coefficient);
   CutQuadratureGenerator generator(*backend);
   CutMappingOptions mapping;
   mapping.weight_type = CutWeightType::Physical;
   real_t volume = 0, area = 0;
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      PolynomialLevelSet level_set;
      auto report = level_set.Set(field, element);
      if (!report.Succeeded()) { cerr << report.message << '\n'; return 2; }
      Array<const ReferenceLevelSet *> level_sets(1);
      level_sets[0] = &level_set;
      for (int surface = 0; surface < 2; ++surface)
      {
         CutQuadratureRequest request;
         request.order = order;
         request.domain = surface ? CutDomain::Surface() : CutDomain::Positive();
         MappedCutQuadrature quadrature;
         report = generator.Generate(*mesh.GetElementTransformation(element),
                                     level_sets, request, mapping, quadrature);
         if (!report.Succeeded()) { cerr << report.message << '\n'; return 2; }
         for (int index = 0; index < quadrature.rule.GetNPoints(); ++index)
         {
            (surface ? area : volume) += quadrature.rule.IntPoint(index).weight;
         }
      }
   }
   const real_t pi = std::acos(-1.0);
   const real_t exact_volume = dimension == 2 ? pi : 4 * pi / 3;
   const real_t exact_area = dimension == 2 ? 2 * pi : 4 * pi;
   cout << setprecision(16)
        << "Volume: " << volume << ", error: " << abs(volume - exact_volume) << '\n'
        << "Surface: " << area << ", error: " << abs(area - exact_area) << '\n';
   if (visualization)
   {
      socketstream socket("localhost", 19916);
      socket << "solution\n" << mesh << field << flush;
   }
   return 0;
}
