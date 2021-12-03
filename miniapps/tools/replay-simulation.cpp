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
    const char *dc_folder = "../../examples/internal/";
    const char *simulation_name = "Example9P-internal";
    const char *field_name = "solution";
    // const auto t0 = 0.0;
    // const auto T = std::numeric_limits<double>::infinity();
    const char* vishost = "localhost";
    int visport = 19916;
    int precision = 8;

    OptionsParser args(argc, argv);
    args.AddOption(&dc_folder, "-dc", "--data-collection-folder",
                    "Folder containing the DataCollections in MFEM internal format.");
    args.AddOption(&simulation_name, "-sn", "--simulation-name",
                    "Name of the DataCollection.");
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
    dc->SetPrefixPath(dc_folder);
    auto metainfo = dc->ReloadMetaInfo();
    if(!metainfo)
    {
        mfem_error("Failed to load meta info.");
    }

    socketstream vis;
    vis.open(vishost, visport);
    vis.precision(precision);

    for(auto [cycle, t, Δt] : metainfo.value())
    {
        dc->Load(cycle);
        auto pmesh = dc->GetParMesh();
        auto u = dc->GetParField(field_name);
        vis << "parallel " << pmesh->GetNRanks() << " " << pmesh->GetMyRank() << "\n";
        vis << "solution\n" << *pmesh << *u << std::flush;
        vis << "plot_caption 't=" << t << "'\n";
    }

    return 0;
}
