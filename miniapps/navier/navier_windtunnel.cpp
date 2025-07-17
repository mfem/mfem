// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//

// Navier Wind Tunnel
//
// The problem domain is set up like this (flow in +x direction):
//
//    Back (y=max)        Front (y=0)
//     +--------+        +--------+
//    /|       /|       /|       /|
//   / |      / |      / |      / |
//  +--------+  |     +--------+  |
//  |  |     |  |     |  |     |  |
//  |  +-----|--+     |  +-----|--+
//  | /      | /      | /      | /
//  |/       |/       |/       |/
//  +--------+        +--------+
// Left     Right    Left     Right
// (x=0)   (x=max)  (x=0)   (x=max)
//
//        Bottom (z=0)    Top (z=max)
//
// Boundary conditions (MFEM attribute assignment):
// - Left (attr 5, x=0): Inlet - Prescribed velocity Dirichlet BC
// - Bottom (attr 1, z=0): Ground - No-slip BC (all velocity components = 0)
// - Front (attr 2, y=0): Front wall - No-penetration BC (zero y-velocity component)
// - Right (attr 3, x=max): Outlet - Do nothing (no BC)
// - Back (attr 4, y=max): Back wall - No-penetration BC (zero y-velocity component)
// - Top (attr 6, z=max): Top wall - No-penetration BC (zero z-velocity component)


#include "navier_solver.hpp"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <limits>

using namespace mfem;
using namespace navier;

/// @brief Context struct for Navier solver
struct s_NavierContext
{
   int ser_ref_levels = 1; // Number of times to uniformly refine the mesh
   int order = 2; // Order (degree) of the finite elements.

   real_t kinvis = 1.0 / 10.0; // Kinematic viscosity \approx 1 / Reynolds number
   real_t t_final = 2.0; // Final time
   real_t dt = 0.01; // Time step

   // MFEM options
   bool pa = true; // Partial assembly
   bool ni = false; // Numerical integration
   bool visualization = false; // GLVis visualization
   bool checkres = false; // Check residual

   // Inlet profile selection
   //   0: Constant
   //   1: Power law
   //   2: Logarithmic
   int inlet_profile_type = 0;
   bool add_fluct =
      false; // Use provided fluctuation field on top of prescribed mean profile
   std::string turb_field_file = "inputs/IEC_simple.vti"; // Turbulence field file

   // Common parameters
   real_t inlet_velocity = 1.0; // Inlet velocity magnitude
   real_t ref_height = 0.4; // Reference height for wind profile

   // Power law parameters
   real_t power_alpha = 0.15;    // Power law exponent

   // Logarithmic parameters
   real_t u_star = 0.1;          // Friction velocity
   real_t z0 = 0.001;            // Roughness length
   real_t kappa = 0.4;           // von Karman constant
   real_t f = 0.0001;            // Coriolis parameter

   // Exponential parameters
   real_t exp_decay = 2.0; // Decay rate
} ctx;

/// @brief Turbulence field data structure, which is meant for directly loading in
/// the data from a VTI file, assumed to be formatted as DRDMannTurb does.
/// @note We should always have Nc = 3
struct TurbField
{
   int Nx, Ny, Nz, Nc;
   std::vector<double> data;
   std::vector<double> spacing;  // x, y, z spacing
   std::vector<double> origin;   // x, y, z origin
};

namespace InletProfile
{
/// Constant velocity profile
void constant(const Vector &x, real_t t, Vector &u)
{
   u(0) = ctx.inlet_velocity;
   u(1) = 0.0;
   u(2) = 0.0;
}

/// Power law profile
void power_law(const Vector &x, real_t t, Vector &u)
{
   real_t z = x(2);
   real_t u_z = ctx.inlet_velocity * pow(z / ctx.ref_height, ctx.power_alpha);

   u(0) = u_z > 0.0 ? u_z : 0.0;
   u(1) = 0.0;
   u(2) = 0.0;
}

/// Uniform logarithmic wind profile
void logarithmic(const Vector &x, real_t t, Vector &u)
{
   real_t z = x(2);

   // NOTE: To avoid log(0) = NaN, we enforce x(2) = z > ctx.z0
   real_t safe_z = std::max(z, ctx.z0 * real_t(1.01));

   real_t u_z = (ctx.u_star / ctx.kappa) * (log(safe_z/ctx.z0) + 34.5 * ctx.f *
                                            safe_z);

   u(0) = u_z > 0.0 ? u_z : 0.0;
   u(1) = 0.0;
   u(2) = 0.0;
}
} // END InletProfile namespace

// Global variables
real_t mean_inlet_velocity = 0.0; // Mean of the inlet profile
TurbField turb_field;

// Profile selection and computation functions

/// @brief Profile selection function
/// @return Pointer to the selected profile function
void (*get_inlet_profile())(const Vector &, real_t, Vector &)
{
   switch (ctx.inlet_profile_type)
   {
      case 0: return InletProfile::constant;
      case 1: return InletProfile::power_law;
      case 2: return InletProfile::logarithmic;
      default: return InletProfile::constant;
   }
}

/// @brief Computes the mean of the inlet profile over a given height.
/// @param height Height over which to compute the mean.
/// @return Mean of the inlet profile.
real_t compute_mean_of_inlet_profile(real_t height)
{
    const int n_points = 1000;
    real_t sum_u = 0.0;
    real_t dz = height / (n_points - 1);

    for (int i = 0; i < n_points; ++i)
    {
        real_t z = i * dz;

        Vector x(3);
        x(0) = 0.0;
        x(1) = 0.0;
        x(2) = z;
        Vector u(3);
        real_t t = 0.0;

        get_inlet_profile()(x, t, u);
        sum_u += u(0);
    }

    return sum_u / n_points;
}

// Turbulence handling functions

/// @brief Adds an interpolation of the provided turbulent fluctuation field to the mean velocity profile
void turb_add(const Vector &x, real_t t, Vector &u)
{
    real_t effective_x = mean_inlet_velocity * t;
    real_t pos[3] = {effective_x, x(1), x(2)};

    // Compute normalized coordinates in [0,1]
    real_t nx[3];
    for (int d = 0; d < 3; ++d)
    {
        nx[d] = (pos[d] - turb_field.origin[d]) / (turb_field.spacing[d] * turb_field.Nx);
        nx[d] = fmod(nx[d], 1.0); // Periodic wrapping
        if (nx[d] < 0.0) 
        {
            nx[d] += 1.0;
        }
    }

    real_t fx = nx[0] * (turb_field.Nx - 1);
    real_t fy = nx[1] * (turb_field.Ny - 1);
    real_t fz = nx[2] * (turb_field.Nz - 1);

    int ix = floor(fx);
    int iy = floor(fy);
    int iz = floor(fz);

    real_t dx = fx - ix;
    real_t dy = fy - iy;
    real_t dz = fz - iz;

    // Clamp indices to valid range
    ix = std::max(0, std::min(ix, turb_field.Nx - 2));
    iy = std::max(0, std::min(iy, turb_field.Ny - 2));
    iz = std::max(0, std::min(iz, turb_field.Nz - 2));

    Vector fluct(3); fluct = 0.0;
    for (int c = 0; c < 3; ++c)
    {
        auto get_val = [&](int dx_off, int dy_off, int dz_off) -> real_t
        {
            int gx = ix + dx_off, gy = iy + dy_off, gz = iz + dz_off;
            size_t idx = (gx + turb_field.Nx * (gy + turb_field.Ny * gz)) * 3 + c;
            return turb_field.data[idx];
        };

        real_t v000 = get_val(0, 0, 0), v001 = get_val(0, 0, 1);
        real_t v010 = get_val(0, 1, 0), v011 = get_val(0, 1, 1);
        real_t v100 = get_val(1, 0, 0), v101 = get_val(1, 0, 1);
        real_t v110 = get_val(1, 1, 0), v111 = get_val(1, 1, 1);

        real_t vxy0 = (1-dz) * ((1-dy) * ((1 - dx) * v000 + dx * v100) + dy * ((1 - dx) * v010 + dx*v110)) +
            dz * ((1-dy) * ((1-dx) * v001 + dx * v101) + dy * ((1-dx) * v011 + dx * v111));

        fluct(c) = vxy0;
    }

    u += fluct;
}

void turbulent_inlet(const Vector &x, real_t t, Vector &u)
{
   get_inlet_profile()(x, t, u);
   turb_add(x, t, u);
}

/// @brief Loads turbulence field from a VTI file.
/// @param filename Name of the VTI file to load.
/// @return Turbulence field data structure.
TurbField load_turb_field(const std::string &filename)
{
   TurbField turb_field;
   std::ifstream infile(filename, std::ios::binary);
   if (!infile)
   {
      throw std::runtime_error("Failed to open VTI file: " + filename);
   }

   // Read entire file into memory for parsing
   infile.seekg(0, std::ios::end);
   size_t file_size = infile.tellg();
   infile.seekg(0, std::ios::beg);
   std::vector<char> buffer(file_size);
   infile.read(buffer.data(), file_size);

   std::string content(buffer.begin(), buffer.end());

   // Parse WholeExtent
   size_t pos = content.find("WholeExtent=\"");
   if (pos == std::string::npos)
   {
      throw std::runtime_error("Invalid VTI: No WholeExtent");
   }
   std::istringstream ext(content.substr(pos + 13, 50));
   int ex_min, ex_max, ey_min, ey_max, ez_min, ez_max;
   ext >> ex_min >> ex_max >> ey_min >> ey_max >> ez_min >> ez_max;
   turb_field.Nx = ex_max - ex_min;  // Number of cells
   turb_field.Ny = ey_max - ey_min;
   turb_field.Nz = ez_max - ez_min;
   turb_field.Nc = 3;

   // Parse Origin
   pos = content.find("Origin=\"");
   if (pos != std::string::npos)
   {
      std::istringstream org(content.substr(pos + 8, 50));
      turb_field.origin.resize(3);
      org >> turb_field.origin[0] >> turb_field.origin[1] >> turb_field.origin[2];
   }

   // Parse Spacing
   pos = content.find("Spacing=\"");
   if (pos != std::string::npos)
   {
      std::istringstream spc(content.substr(pos + 9, 50));
      turb_field.spacing.resize(3);
      spc >> turb_field.spacing[0] >> turb_field.spacing[1] >> turb_field.spacing[2];
   }

   // Find start of AppendedData
   pos = content.find("<AppendedData encoding=\"raw\">");
   if (pos == std::string::npos)
   {
      throw std::runtime_error("Invalid VTI: No AppendedData tag");
   }
   size_t appended_start = pos +
                           std::string("<AppendedData encoding=\"raw\">").length();

   // Find the '_' marker after the tag
   pos = content.find("_", appended_start);
   if (pos == std::string::npos)
   {
      throw std::runtime_error("Invalid VTI: No raw data marker '_'");
   }
   size_t data_start = pos + 1;  // Binary data starts after '_'

   // Hardcoded offsets from your XML (adjust if needed or parse dynamically)
   size_t grid_offset = 0;
   size_t wind_offset = 300568;

   // Ensure data_start is relative to buffer[0]
   data_start = data_start;  // Already absolute in content

   // Skip grid array: uint64 prefix + data
   uint64_t grid_prefix;
   std::memcpy(&grid_prefix, buffer.data() + data_start + grid_offset,
               sizeof(uint64_t));
   size_t grid_size_bytes = static_cast<size_t>(grid_prefix);
   data_start += grid_offset + sizeof(uint64_t) + grid_size_bytes;

   // Now at wind array: read prefix
   uint64_t wind_prefix;
   std::memcpy(&wind_prefix, buffer.data() + data_start, sizeof(uint64_t));
   data_start += sizeof(uint64_t);

   // Validate size
   size_t expected_bytes = turb_field.Nx * turb_field.Ny * turb_field.Nz *
                           turb_field.Nc * sizeof(double);
   if (wind_prefix != expected_bytes)
   {
      throw std::runtime_error("Wind data size mismatch: expected " + std::to_string(
                                  expected_bytes) + ", got " + std::to_string(wind_prefix));
   }

   // Read wind data
   size_t wind_size = turb_field.Nx * turb_field.Ny * turb_field.Nz *
                      turb_field.Nc;
   turb_field.data.resize(wind_size);
   std::memcpy(turb_field.data.data(), buffer.data() + data_start, expected_bytes);

   return turb_field;
}


/// @brief Main function
/// @param argc Number of command line arguments
/// @param argv Command line arguments
/// @return 0 if successful, 1 otherwise
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   int visport = 19916;

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.inlet_velocity, "-u0", "--inlet-velocity",
                  "Inlet velocity magnitude.");
   args.AddOption(&ctx.pa,
                  "-pa",
                  "--enable-pa",
                  "-no-pa",
                  "--disable-pa",
                  "Enable partial assembly.");
   args.AddOption(&ctx.ni,
                  "-ni",
                  "--enable-ni",
                  "-no-ni",
                  "--disable-ni",
                  "Enable numerical integration rules.");
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&ctx.inlet_profile_type, "-profile", "--inlet-profile",
                  "Inlet profile: 0=constant, 1=power, 2=logarithmic");
   args.AddOption(&ctx.ref_height, "-href", "--reference-height",
                  "Reference height for wind profile");
   args.AddOption(&ctx.power_alpha, "-alpha", "--power-alpha",
                  "Power law exponent (0.1-0.4)");
   args.AddOption(&ctx.z0, "-z0", "--roughness-length",
                  "Surface roughness length");
   args.AddOption(&ctx.u_star, "-ustar", "--friction-velocity",
                  "Friction velocity for log profile");
   args.AddOption(&ctx.add_fluct, "-fluct", "--add-fluct", "-no-fluct",
                  "--no-add-fluct",
                  "Add turbulent fluctuation to mean velocity profile");
   args.AddOption(&ctx.turb_field_file, "-turb", "--turb-field",
                  "Turbulence field file path (relative to PWD)");

   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }

   // Load turbulence field in
   turb_field = load_turb_field(ctx.turb_field_file);
   if (Mpi::Root())
   {
      mfem::out << "Turbulence field loaded successfully" << std::endl;
      mfem::out << "Nx: " << turb_field.Nx << ", Ny: " << turb_field.Ny << ", Nz: " <<
                turb_field.Nz << std::endl;
   }


//    // Verify loaded data (print first few values for comparison with NumPy)
//    if (Mpi::Root())
//    {
//       mfem::out <<
//                 "Verifying turbulence data (first 9 values, assuming (x,y,z,c) order):" <<
//                 std::endl;
//       for (size_t i = 0; i < 9 && i < turb_field.data.size(); ++i)
//       {
//          mfem::out << turb_field.data[i] << " ";
//       }
//       mfem::out << std::endl;

//       // Optional: Print min/max per component for summary
//       double min_u = std::numeric_limits<double>::max(),
//              max_u = std::numeric_limits<double>::lowest();
//       double min_v = std::numeric_limits<double>::max(),
//              max_v = std::numeric_limits<double>::lowest();
//       double min_w = std::numeric_limits<double>::max(),
//              max_w = std::numeric_limits<double>::lowest();
//       size_t size = turb_field.data.size();
//       for (size_t i = 0; i < size; i += 3)
//       {
//          min_u = std::min(min_u, turb_field.data[i]);
//          max_u = std::max(max_u, turb_field.data[i]);
//          min_v = std::min(min_v, turb_field.data[i+1]);
//          max_v = std::max(max_v, turb_field.data[i+1]);
//          min_w = std::min(min_w, turb_field.data[i+2]);
//          max_w = std::max(max_w, turb_field.data[i+2]);
//       }
//       mfem::out << "Component U: min=" << min_u << ", max=" << max_u << std::endl;
//       mfem::out << "Component V: min=" << min_v << ", max=" << max_v << std::endl;
//       mfem::out << "Component W: min=" << min_w << ", max=" << max_w << std::endl;
//    }

   real_t length = 300.0;
   real_t width = 100.0;
   real_t height = 100.0;

   real_t mean_inlet_velocity = compute_mean_of_inlet_profile(height);

   // Domain: [0, 3] x [0, 1] x [0, 1] (Length x Width x Height)
   Mesh mesh = Mesh::MakeCartesian3D(6, 2, 2, Element::HEXAHEDRON,
                                     length, width, height);

   for (int i = 0; i < ctx.ser_ref_levels; ++i)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      mfem::out << "Number of elements: " << mesh.GetNE() << std::endl;
      mfem::out << "Mesh dimension: " << mesh.Dimension() << std::endl;
      mfem::out << "Number of boundary attributes: " << mesh.bdr_attributes.Max() <<
                std::endl;

      // Print boundary attribute assignments for verification
      mfem::out << "\nBoundary attribute assignments:" << std::endl;
      mfem::out << "  Bottom (z=0): attr 1" << std::endl;
      mfem::out << "  Front (y=0):  attr 2" << std::endl;
      mfem::out << "  Right (x=max): attr 3" << std::endl;
      mfem::out << "  Back (y=max):  attr 4" << std::endl;
      mfem::out << "  Left (x=0):   attr 5" << std::endl;
      mfem::out << "  Top (z=max):  attr 6" << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);
   flowsolver.EnablePA(ctx.pa);
   flowsolver.EnableNI(ctx.ni);

   // Set the initial condition (quiescent flow)
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorConstantCoefficient u_init(Vector({0.0, 0.0, 0.0}));
   u_ic->ProjectCoefficient(u_init);

   // BOUNDARY CONDITIONS

   // 1. GROUND: No-slip
   Array<int> attr_ground(pmesh->bdr_attributes.Max());
   attr_ground = 0;
   attr_ground[0] = 1;  // attr 1
   flowsolver.AddVelDirichletBC(new VectorConstantCoefficient(Vector({0.0, 0.0, 0.0})),
                                attr_ground);

   // 2. "FRONT" WALL: No-penetration
   Array<int> attr_front(pmesh->bdr_attributes.Max());
   attr_front = 0;
   attr_front[1] = 1;  // attr 2
   flowsolver.AddVelDirichletBC(new ConstantCoefficient(0.0), attr_front, 1);

   // 3. "BACK" WALL: No-penetration
   Array<int> attr_back(pmesh->bdr_attributes.Max());
   attr_back = 0;
   attr_back[3] = 1;  // attr 4
   flowsolver.AddVelDirichletBC(new ConstantCoefficient(0.0), attr_back, 1);

   // 4. "TOP" WALL: No-penetration
   Array<int> attr_top(pmesh->bdr_attributes.Max());
   attr_top = 0;
   attr_top[5] = 1;  // attr 6
   flowsolver.AddVelDirichletBC(new ConstantCoefficient(0.0), attr_top, 2);

   // 5. INLET: Prescribed velocity
   Array<int> attr_inlet(pmesh->bdr_attributes.Max());
   attr_inlet = 0;
   attr_inlet[4] = 1;  // attr 5
   flowsolver.AddVelDirichletBC(get_inlet_profile(), attr_inlet);

   // 6. OUTLET: Do nothing

   real_t t = 0.0;
   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;

   if (Mpi::Root())
   {
      printf("%5s %8s %8s %8s %11s\n",
             "Step", "Time", "dt", "CFL", "||u||_max");
   }

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      u_gf = flowsolver.GetCurrentVelocity();
      p_gf = flowsolver.GetCurrentPressure();

      ParaViewDataCollection pvdc("navier_windtunnel_output", pmesh);
      pvdc.SetDataFormat(VTKFormat::BINARY32);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetLevelsOfDetail(ctx.order);
      pvdc.SetCycle(step);
      pvdc.SetTime(t);
      pvdc.RegisterField("velocity", u_gf);
      pvdc.RegisterField("pressure", p_gf);
      pvdc.Save();

      real_t cfl = flowsolver.ComputeCFL(*u_gf, dt);
      real_t max_vel = u_gf->Max();

      if (Mpi::Root() && step % 10 == 0)
      {
         printf("%5d %8.3f %8.2E %8.2E %11.4E\n",
                step, t, dt, cfl, max_vel);
         fflush(stdout);
      }
   }

   // GLVis visualization of final time step
   if (ctx.visualization)
   {
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << Mpi::WorldSize() << " "
               << Mpi::WorldRank() << "\n";
      sol_sock << "solution\n" << *pmesh << *u_gf << std::endl;
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}
