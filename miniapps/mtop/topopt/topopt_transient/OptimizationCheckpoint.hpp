// =============================================================================
// Minimal Optimization Checkpoint for Transient Topology Optimization
// =============================================================================
//
// Purpose: survive HPC wall-clock limits. Saves ONLY the raw control density
// (true-dof vector, one binary file per MPI rank) plus a small human-readable
// metadata file. On restart the density is used as the initial guess for a
// fresh MMA run - no optimizer internals (asymptotes, design history) are
// saved; MMA rebuilds them within a couple of iterations.
//
// Crash safety: every file is written to "<name>.tmp" and atomically renamed;
// metadata is written LAST, so a job killed mid-save leaves either the
// previous consistent checkpoint or the new one. (Worst case a kill between
// the rank renames mixes two consecutive designs - harmless, since the
// payload is only an initial guess.)
//
// Constraints (validated on load): same MPI rank count, same mesh refinement
// and FE order as the run that wrote the checkpoint.
//
// DISTINCT FROM trajectory checkpointing (TrajectoryCheckpointing.hpp), which
// handles RK4 states inside one forward/adjoint sweep.
//
// =============================================================================

#ifndef OPTIMIZATION_CHECKPOINT_HPP
#define OPTIMIZATION_CHECKPOINT_HPP

#include "mfem.hpp"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace mfem
{

struct OptimizationCheckpointMetadata
{
   int iteration = 0;             // Last completed MMA iteration
   real_t objective = 0.0;        // J at that iteration (informational)
   real_t volume_fraction = 0.0;  // Volume fraction at that iteration
   int n_mpi_ranks = 1;           // Must match on restart
   int refinement_level = 0;      // Must match on restart
   int fe_order = 1;              // Must match on restart
};

class OptimizationCheckpoint
{
private:
   static constexpr int32_t design_magic_ = 0x52484F31;   // "RHO1"

   std::string dir_;
   MPI_Comm comm_;
   int myid_;

   std::string MetadataPath() const { return dir_ + "/metadata.txt"; }

   std::string DesignPath(int rank) const
   {
      std::ostringstream name;
      name << dir_ << "/design." << std::setfill('0') << std::setw(6) << rank;
      return name.str();
   }

   bool CreateDirectoryIfNeeded() const
   {
      bool ok = true;
      if (myid_ == 0)
      {
         struct stat st;
         if (stat(dir_.c_str(), &st) != 0)
         {
            ok = (mkdir(dir_.c_str(), 0755) == 0);
            if (!ok)
            {
               std::cerr << "Checkpoint: failed to create directory: "
                         << dir_ << std::endl;
            }
         }
      }
      MPI_Bcast(&ok, 1, MPI_C_BOOL, 0, comm_);
      return ok;
   }

   // Reduce a local success flag to a global all-ranks-succeeded flag.
   bool AllOk(bool local_ok) const
   {
      int loc = local_ok ? 1 : 0, glob = 0;
      MPI_Allreduce(&loc, &glob, 1, MPI_INT, MPI_MIN, comm_);
      return glob == 1;
   }

public:
   OptimizationCheckpoint(const std::string &dir, MPI_Comm comm)
      : dir_(dir), comm_(comm)
   {
      MPI_Comm_rank(comm_, &myid_);
   }

   /// Save the local control-density true-dof vector + metadata.
   /// Call at the end of each completed MMA iteration; overwrites in place
   /// (atomic per file, metadata last).
   bool Save(const OptimizationCheckpointMetadata &meta, const Vector &rho_tv)
   {
      if (!CreateDirectoryIfNeeded()) { return false; }

      // 1. Every rank: write its design piece to .tmp and rename.
      bool ok = true;
      {
         const std::string path = DesignPath(myid_);
         const std::string tmp = path + ".tmp";
         std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
         const int64_t n = rho_tv.Size();
         ok = ofs.good();
         if (ok)
         {
            ofs.write(reinterpret_cast<const char *>(&design_magic_),
                      sizeof(design_magic_));
            ofs.write(reinterpret_cast<const char *>(&meta.iteration),
                      sizeof(meta.iteration));
            ofs.write(reinterpret_cast<const char *>(&n), sizeof(n));
            ofs.write(reinterpret_cast<const char *>(rho_tv.GetData()),
                      n * sizeof(real_t));
            ofs.close();
            ok = ofs.good() && (std::rename(tmp.c_str(), path.c_str()) == 0);
         }
      }
      if (!AllOk(ok))
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint: design write failed on some rank; "
                      << "keeping previous metadata." << std::endl;
         }
         return false;
      }

      // 2. Rank 0: metadata last (acts as the commit marker), atomically.
      bool meta_ok = true;
      if (myid_ == 0)
      {
         int nranks = 1;
         MPI_Comm_size(comm_, &nranks);
         const std::string tmp = MetadataPath() + ".tmp";
         std::ofstream ofs(tmp, std::ios::trunc);
         meta_ok = ofs.good();
         if (meta_ok)
         {
            ofs << "iteration " << meta.iteration << "\n"
                << "objective " << std::setprecision(17) << meta.objective << "\n"
                << "volume_fraction " << meta.volume_fraction << "\n"
                << "n_mpi_ranks " << nranks << "\n"
                << "refinement_level " << meta.refinement_level << "\n"
                << "fe_order " << meta.fe_order << "\n";
            ofs.close();
            meta_ok = ofs.good() &&
                      (std::rename(tmp.c_str(), MetadataPath().c_str()) == 0);
         }
         if (!meta_ok)
         {
            std::cerr << "Checkpoint: metadata write failed." << std::endl;
         }
      }
      MPI_Bcast(&meta_ok, 1, MPI_C_BOOL, 0, comm_);
      return meta_ok;
   }

   bool Exists() const
   {
      bool exists = false;
      if (myid_ == 0)
      {
         std::ifstream test(MetadataPath());
         exists = test.good();
      }
      MPI_Bcast(&exists, 1, MPI_C_BOOL, 0, comm_);
      return exists;
   }

   /// Read + broadcast the metadata and check it matches this run.
   bool ValidateCompatibility(int expected_ref_level, int expected_order,
                              OptimizationCheckpointMetadata &meta) const
   {
      bool read_ok = true;
      if (myid_ == 0)
      {
         std::ifstream ifs(MetadataPath());
         read_ok = ifs.good();
         std::string key;
         while (read_ok && ifs >> key)
         {
            if (key == "iteration")             { ifs >> meta.iteration; }
            else if (key == "objective")        { ifs >> meta.objective; }
            else if (key == "volume_fraction")  { ifs >> meta.volume_fraction; }
            else if (key == "n_mpi_ranks")      { ifs >> meta.n_mpi_ranks; }
            else if (key == "refinement_level") { ifs >> meta.refinement_level; }
            else if (key == "fe_order")         { ifs >> meta.fe_order; }
            else { std::string skip; ifs >> skip; }
            read_ok = !ifs.fail();
         }
      }
      MPI_Bcast(&read_ok, 1, MPI_C_BOOL, 0, comm_);
      if (!read_ok)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint: cannot parse " << MetadataPath()
                      << std::endl;
         }
         return false;
      }
      MPI_Bcast(&meta.iteration, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.objective, 1, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(&meta.volume_fraction, 1, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(&meta.n_mpi_ranks, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.refinement_level, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.fe_order, 1, MPI_INT, 0, comm_);

      int nranks = 1;
      MPI_Comm_size(comm_, &nranks);
      bool compatible = true;
      if (meta.n_mpi_ranks != nranks)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint incompatible: written with "
                      << meta.n_mpi_ranks << " ranks, running with "
                      << nranks << ". Resubmit with the original count.\n";
         }
         compatible = false;
      }
      if (meta.refinement_level != expected_ref_level)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint incompatible: refinement level "
                      << meta.refinement_level << " vs " << expected_ref_level
                      << ".\n";
         }
         compatible = false;
      }
      if (meta.fe_order != expected_order)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint incompatible: FE order "
                      << meta.fe_order << " vs " << expected_order << ".\n";
         }
         compatible = false;
      }
      return compatible;
   }

   /// Load this rank's design piece into rho_tv (must be pre-sized to the
   /// local control true-dof count). Call after ValidateCompatibility.
   bool Load(Vector &rho_tv) const
   {
      bool ok = true;
      {
         std::ifstream ifs(DesignPath(myid_), std::ios::binary);
         ok = ifs.good();
         int32_t magic = 0;
         int iteration = 0;
         int64_t n = 0;
         if (ok)
         {
            ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
            ifs.read(reinterpret_cast<char *>(&iteration), sizeof(iteration));
            ifs.read(reinterpret_cast<char *>(&n), sizeof(n));
            ok = ifs.good() && magic == design_magic_ && n == rho_tv.Size();
         }
         if (ok)
         {
            ifs.read(reinterpret_cast<char *>(rho_tv.GetData()),
                     n * sizeof(real_t));
            ok = ifs.good();
         }
         if (!ok)
         {
            std::cerr << "Checkpoint: rank " << myid_
                      << " failed to load " << DesignPath(myid_)
                      << " (magic/size mismatch or read error)." << std::endl;
         }
      }
      return AllOk(ok);
   }
};

} // namespace mfem

#endif // OPTIMIZATION_CHECKPOINT_HPP
