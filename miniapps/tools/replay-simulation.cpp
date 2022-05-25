#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   const int myid = mpi.WorldRank();

   // 1. Parse command-line options
   const char *simulation_name = "../../examples/internal/Example9P-internal";
   const char *field_name = "solution";
   double t0 = 0.0;
   double T = std::numeric_limits<double>::infinity();
   const char* vishost = "localhost";
   int visport = 19916;
   int precision = 8;

   OptionsParser args(argc, argv);
   args.AddOption(&simulation_name, "-sn", "--simulation-name",
                  "Path with name of the MFEMDataCollection.");
   args.AddOption(&field_name, "-fn", "--field-name",
                  "Name of the field in the MFEMDataCollection.");
   args.AddOption(&t0, "-t0", "--first-time-point",
                  "Time point to begin the replay at.");
   args.AddOption(&T, "-T", "--last-time-point",
                  "Time point to end the replay at.");
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

   // Load data collection
   auto dc = std::make_shared<MFEMDataCollection>(simulation_name);
   auto metainfo = dc->ReloadMetaInfo();
   if (!metainfo)
   {
      mfem_error("Failed to load meta info.");
   }

   socketstream vis;
   vis.open(vishost, visport);
   vis.precision(precision);

   for (auto [cycle, t] : metainfo.value())
   {
      if (t < t0 || t > T) { continue; }

      dc->Load(cycle);
      auto pmesh = dc->GetParMesh();
      auto u = dc->GetParField(field_name);
      vis << "parallel " << pmesh->GetNRanks() << " " << pmesh->GetMyRank() << "\n";
      vis << "solution\n" << *pmesh << *u << std::flush;
      vis << "plot_caption 't=" << t << "'\n";
   }

   return 0;
}
