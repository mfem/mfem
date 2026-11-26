#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

void Require(bool condition, const std::string &message)
{
   if (!condition) { throw std::runtime_error(message); }
}

double Number(const std::string &text)
{
   std::size_t consumed = 0;
   const double value = std::stod(text, &consumed);
   Require(consumed == text.size() && std::isfinite(value),
           "Invalid finite number: " + text);
   return value;
}

bool Close(double actual, double expected, double relative, double absolute)
{
   Require(relative >= 0.0 && absolute >= 0.0, "Negative comparison tolerance");
   return std::abs(actual - expected) <=
          std::max(absolute, relative * std::max(std::abs(actual), std::abs(expected)));
}

void CheckVisualization(const std::filesystem::path &directory, int ranks)
{
   int collections = 0, indices = 0, pieces = 0;
   double maximum_indicator = 0.0;
   for (const auto &entry : std::filesystem::recursive_directory_iterator(directory))
   {
      if (!entry.is_regular_file()) { continue; }
      const auto extension = entry.path().extension();
      if (extension == ".pvd") { ++collections; }
      if (extension == ".pvtu") { ++indices; }
      if (extension != ".vtu") { continue; }
      ++pieces;
      std::ifstream stream(entry.path());
      Require(stream.good(), "Cannot read visualization piece");
      const std::string contents((std::istreambuf_iterator<char>(stream)), {});
      for (const std::string name : {"velocity", "pressure", "inclusion_indicator", "brinkman_penalty"})
      {
         const auto field = contents.find("Name=\"" + name + "\"");
         Require(field != std::string::npos, "Missing output field " + name);
         const auto begin = contents.find('>', field);
         const auto end = contents.find("</DataArray>", field);
         Require(begin != std::string::npos && end != std::string::npos && begin < end,
                 "Invalid output field " + name);
         std::istringstream values(contents.substr(begin + 1, end - begin - 1));
         std::string token;
         while (values >> token)
         {
            const double value = Number(token);
            if (name == "inclusion_indicator")
            {
               Require(value >= -1e-12 && value <= 1.0 + 1e-12, "Invalid cell-average mask");
               maximum_indicator = std::max(maximum_indicator, value);
            }
         }
      }
   }
   Require(collections == 1 && indices == 1, "Missing parallel visualization index");
   Require(pieces == ranks, "Wrong number of visualization pieces");
   Require(maximum_indicator > 0.0, "Visualization missed the sampled inclusion");
}

void SelfTest()
{
   Require(Close(1.0, 1.0 + 1e-8, 2e-7, 2e-9), "Relative tolerance failed");
   Require(Close(1e-10, 0.0, 2e-7, 2e-9), "Absolute tolerance failed");
   Require(!Close(1.0, 2.0, 2e-7, 2e-9), "Unequal values were accepted");
   Require(Number("-1.25e-3") == -0.00125, "Scientific notation failed");
   for (const std::string token : {"nan", "inf", "-inf", "1e999", "1.0junk", ""})
   {
      bool rejected = false;
      try { Number(token); }
      catch (const std::exception &) { rejected = true; }
      Require(rejected, "Invalid number was accepted: " + token);
   }
}

}

int main(int argc, char *argv[])
{
   try
   {
      Require(argc >= 2, "Expected a driver-check mode");
      const std::string mode = argv[1];
      if (mode == "self-test" && argc == 2) { SelfTest(); }
      else if (mode == "finite" && argc > 2)
      {
         for (int argument = 2; argument < argc; ++argument) { Number(argv[argument]); }
      }
      else if ((mode == "close" || mode == "difference") && argc == 7)
      {
         const double actual = mode == "difference" ? Number(argv[2]) - Number(argv[3]) : Number(argv[2]);
         const double expected = mode == "difference" ? Number(argv[6]) : Number(argv[3]) + Number(argv[6]);
         Require(std::isfinite(actual) && std::isfinite(expected), "Nonfinite comparison value");
         Require(Close(actual, expected, Number(argv[4]), Number(argv[5])),
                 std::string("Comparison failed: ") + argv[2] + ", " + argv[3] + ", " + argv[6]);
      }
      else if (mode == "visualization" && argc == 4)
      {
         CheckVisualization(argv[2], std::stoi(argv[3]));
      }
      else { throw std::runtime_error("Invalid driver-check arguments"); }
      return 0;
   }
   catch (const std::exception &error)
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
}
