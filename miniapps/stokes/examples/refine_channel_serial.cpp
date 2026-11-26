#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

void Check(bool valid, const char *message)
{
   if (!valid) { throw std::runtime_error(message); }
}

int Run(int argc, char *argv[])
{
   const char *input = "meshes/channel-3d/channel_hex.mesh", *output = "channel-refined.mesh";
   int factor_x = 1, factor_y = 1, factor_z = 1;
   OptionsParser arguments(argc, argv);
   arguments.AddOption(&input, "-m", "--mesh", "Input uniform Cartesian hex channel with boundary tags 1-6.");
   arguments.AddOption(&output, "-o", "--output", "New serial MFEM mesh; existing files are rejected.");
   arguments.AddOption(&factor_x, "-fx", "--factor-x", "Positive integer subdivision factor along x.");
   arguments.AddOption(&factor_y, "-fy", "--factor-y", "Positive integer subdivision factor along y.");
   arguments.AddOption(&factor_z, "-fz", "--factor-z", "Positive integer subdivision factor along z.");
   arguments.Parse();
   if (!arguments.Good()) { arguments.PrintUsage(std::cout); return arguments.Help() ? 0 : 2; }
   Check(factor_x > 0 && factor_y > 0 && factor_z > 0, "Subdivision factors must be positive");
   Check(std::ifstream(input).good(), "Cannot read input channel mesh");
   Check(!std::ifstream(output).good(), "Refusing to overwrite an existing mesh");
   Mesh original(input, 1, 1, false);
   Check(original.Dimension() == 3 && original.SpaceDimension() == 3 &&
         !original.GetNodes() && !original.Nonconforming() && original.GetNE() > 0,
         "Require a first-order conforming Cartesian hex mesh");
   Check(original.CheckElementOrientation(false) == 0 && original.CheckBdrElementOrientation(false) == 0,
         "Invalid input mesh orientation");
   Vector lower, upper;
   original.GetBoundingBox(lower, upper);
   std::array<int, 3> counts{}, factors{factor_x, factor_y, factor_z};
   std::array<real_t, 3> spacing{};
   const real_t tolerance = 1e-10 * std::max(real_t(1.0), upper.Normlinf() + lower.Normlinf());
   for (int axis = 0; axis < 3; ++axis)
   {
      std::vector<real_t> coordinates;
      for (int vertex = 0; vertex < original.GetNV(); ++vertex) { coordinates.push_back(original.GetVertex(vertex)[axis]); }
      std::sort(coordinates.begin(), coordinates.end());
      coordinates.erase(std::unique(coordinates.begin(), coordinates.end(),
                                   [&](real_t first, real_t second) { return std::abs(first - second) <= tolerance; }), coordinates.end());
      counts[axis] = static_cast<int>(coordinates.size()) - 1;
      Check(counts[axis] > 0 && std::isfinite(upper(axis) - lower(axis)) && upper(axis) > lower(axis),
            "Invalid Cartesian channel extent");
      spacing[axis] = (upper(axis) - lower(axis)) / counts[axis];
      int index = 0;
      for (real_t coordinate : coordinates)
      {
         Check(std::abs(coordinate - lower(axis) - index * spacing[axis]) <= tolerance,
               "Input channel must have uniform coordinate lines");
         ++index;
      }
   }
   std::set<std::array<int, 3>> cells;
   for (int element = 0; element < original.GetNE(); ++element)
   {
      Check(original.GetElementBaseGeometry(element) == Geometry::CUBE && original.GetAttribute(element) == 1,
            "Input channel must contain only domain-1 hexahedra");
      Array<int> vertices;
      original.GetElementVertices(element, vertices);
      std::array<int, 3> minimum = counts, maximum{};
      std::set<std::array<int, 3>> corners;
      for (int vertex : vertices)
      {
         std::array<int, 3> coordinate{};
         for (int axis = 0; axis < 3; ++axis)
         {
            coordinate[axis] = static_cast<int>(std::lround((original.GetVertex(vertex)[axis] - lower(axis)) / spacing[axis]));
            minimum[axis] = std::min(minimum[axis], coordinate[axis]);
            maximum[axis] = std::max(maximum[axis], coordinate[axis]);
         }
         corners.insert(coordinate);
      }
      for (int axis = 0; axis < 3; ++axis) { Check(maximum[axis] == minimum[axis] + 1, "Input cells must match the Cartesian grid"); }
      Check(corners.size() == 8 && cells.insert(minimum).second, "Invalid or duplicated Cartesian cell");
   }
   Check(static_cast<long long>(counts[0]) * counts[1] * counts[2] == original.GetNE(), "Incomplete Cartesian grid");
   std::array<real_t, 6> areas{};
   Vector physical(3);
   for (int boundary = 0; boundary < original.GetNBE(); ++boundary)
   {
      const int attribute = original.GetBdrAttribute(boundary);
      Check(attribute >= 1 && attribute <= 6, "Expected boundary tags 1-6");
      auto &transformation = *original.GetBdrElementTransformation(boundary);
      const auto &rule = IntRules.Get(original.GetBdrElementBaseGeometry(boundary), 3);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const auto &ip = rule.IntPoint(point);
         transformation.SetIntPoint(&ip); transformation.Transform(ip, physical);
         const int axis = (attribute - 1) / 2;
         Check(std::abs(physical(axis) - (attribute % 2 ? lower(axis) : upper(axis))) <= tolerance,
               "Input boundary tag does not match its channel face");
         areas[attribute - 1] += ip.weight * transformation.Weight();
      }
   }
   const real_t volume = (upper(0) - lower(0)) * (upper(1) - lower(1)) * (upper(2) - lower(2));
   for (int attribute = 1; attribute <= 6; ++attribute)
   {
      const int axis = (attribute - 1) / 2;
      const real_t expected = volume / (upper(axis) - lower(axis));
      Check(std::abs(areas[attribute - 1] - expected) < 1e-9 * expected, "Incomplete input boundary coverage");
   }
   long long total = 1;
   for (int axis = 0; axis < 3; ++axis)
   {
      Check(counts[axis] <= std::numeric_limits<int>::max() / factors[axis], "Subdivision exceeds integer capacity");
      counts[axis] *= factors[axis];
      Check(total <= std::numeric_limits<int>::max() / counts[axis], "Refined mesh exceeds integer capacity");
      total *= counts[axis];
   }
   Mesh refined = Mesh::MakeCartesian3D(counts[0], counts[1], counts[2], Element::HEXAHEDRON,
                                       upper(0) - lower(0), upper(1) - lower(1), upper(2) - lower(2));
   for (int vertex = 0; vertex < refined.GetNV(); ++vertex)
   {
      for (int axis = 0; axis < 3; ++axis) { refined.GetVertex(vertex)[axis] += lower(axis); }
   }
   for (int boundary = 0; boundary < refined.GetNBE(); ++boundary)
   {
      auto &transformation = *refined.GetBdrElementTransformation(boundary);
      transformation.Transform(Geometries.GetCenter(transformation.GetGeometryType()), physical);
      int attribute = 0;
      for (int axis = 0; axis < 3; ++axis)
      {
         if (std::abs(physical(axis) - lower(axis)) <= tolerance) { attribute = 2 * axis + 1; }
         if (std::abs(physical(axis) - upper(axis)) <= tolerance) { attribute = 2 * axis + 2; }
      }
      Check(attribute > 0, "Cannot classify refined channel boundary");
      refined.SetBdrAttribute(boundary, attribute);
   }
   refined.SetAttributes();
   Check(refined.GetNE() == total && refined.CheckElementOrientation(false) == 0 &&
         refined.CheckBdrElementOrientation(false) == 0, "Invalid refined channel mesh");
   std::ofstream stream(output);
   stream.precision(17); refined.Print(stream); stream.close();
   Check(!stream.fail(), "Cannot write refined mesh");
   Mesh reloaded(output, 1, 1, false);
   Check(reloaded.GetNE() == total && reloaded.GetNV() == refined.GetNV() &&
         reloaded.bdr_attributes.Size() == 6, "Refined mesh reload failed");
   std::cout << "serial_refinement input_elements=" << original.GetNE() << " elements=" << total
             << " nx=" << counts[0] << " ny=" << counts[1] << " nz=" << counts[2]
             << " vertices=" << refined.GetNV() << " volume=" << volume << '\n';
   return 0;
}

}

int main(int argc, char *argv[])
{
   try { return Run(argc, argv); }
   catch (const std::exception &error)
   {
      std::cerr << "refine_channel_serial: " << error.what() << '\n';
      return 2;
   }
}
