#pragma once
#include "dfem_refactor.hpp"

#define DFEM_TEST_MAIN(function)                                               \
   int main(int argc, char* argv[])                                            \
   {                                                                           \
      Mpi::Init();                                                             \
                                                                               \
      const char* device_config = "cpu";                                       \
      const char* mesh_file = "../data/fichera-q2.mesh";                      \
      int polynomial_order = 2;                                                \
      int ir_order = 4;                                                        \
      int refinements = 0;                                                     \
                                                                               \
      OptionsParser args(argc, argv);                                          \
      args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");         \
      args.AddOption(&polynomial_order, "-o", "--order", "");                  \
      args.AddOption(&refinements, "-r", "--r", "");                           \
      args.AddOption(&ir_order, "-iro", "--iro", "");                          \
      args.AddOption(&device_config, "-d", "--device",                         \
         "Device configuration string, see Device::Configure().");             \
      args.ParseCheck();                                                       \
                                                                               \
      Device device(device_config);                                            \
      if (Mpi::Root() == 0)                                                    \
      {                                                                        \
         device.Print();                                                       \
      }                                                                        \
                                                                               \
      out << std::setprecision(12);                                            \
                                                                               \
      int ret;                                                                 \
                                                                               \
      ret = function(mesh_file, refinements, polynomial_order);                \
      out << #function;                                                        \
      ret ? out << " FAILURE\n" : out << " OK\n";                              \
                                                                               \
      return ret;                                                              \
   }\
