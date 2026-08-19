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
// Constraints (validated on load): same MPI rank count, same mesh refinement,
// and same design/filter FE order. The state (forward/adjoint) FE order is
// informational: raw rho can seed a new physics discretization when its own
// L2 control layout is unchanged.
//
// DISTINCT FROM trajectory checkpointing (TrajectoryCheckpointing.hpp), which
// handles RK4 states inside one forward/adjoint sweep.
//
// =============================================================================

#ifndef OPTIMIZATION_CHECKPOINT_HPP
#define OPTIMIZATION_CHECKPOINT_HPP

#include "mfem.hpp"
#include <cmath>
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
   // Version 2 gives the payload an unambiguous design index: rho^K has
   // design_iteration=K after K completed MMA updates. Version 1 metadata used
   // a zero-based update index and attached pre-update objective data to the
   // post-update density; the loader translates that legacy convention.
   int format_version = 2;
   int design_iteration = 0;
   bool objective_valid_for_design = false;
   real_t objective = 0.0;
   real_t volume_fraction = 0.0;  // Always evaluated on the saved density
   int n_mpi_ranks = 1;           // Must match on restart
   int refinement_level = 0;      // Must match on restart
   int fe_order = 1;              // State H1 order (informational on restart)
   // H1 order of rho_tilde. rho uses paired L2 degree max(0, p_d - 1).
   // -1 marks legacy metadata predating independent design order.
   int design_fe_order = -1;
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

   static bool MetadataIsFinite(const OptimizationCheckpointMetadata &meta)
   {
      return std::isfinite(meta.objective) &&
             std::isfinite(meta.volume_fraction);
   }

   static bool DesignIsFinite(const Vector &rho_tv)
   {
      for (int i = 0; i < rho_tv.Size(); i++)
      {
         if (!std::isfinite(rho_tv[i])) { return false; }
      }
      return true;
   }

public:
   OptimizationCheckpoint(const std::string &dir, MPI_Comm comm)
      : dir_(dir), comm_(comm)
   {
      MPI_Comm_rank(comm_, &myid_);
   }

   /// Save the local control-density true-dof vector + metadata.
   /// Save rho^K with design_iteration=K; overwrites in place (atomic per
   /// file, metadata last). objective_valid_for_design states whether the
   /// objective field was evaluated on this exact payload.
   bool Save(const OptimizationCheckpointMetadata &meta, const Vector &rho_tv)
   {
      // Validate collectively before touching the existing checkpoint. Every
      // rank participates in both reductions so a bad local design cannot leave
      // other ranks waiting in the subsequent file-write collectives.
      const bool metadata_finite = AllOk(
         meta.format_version == 2 && meta.design_iteration >= 0 &&
         MetadataIsFinite(meta));
      const bool design_finite = AllOk(DesignIsFinite(rho_tv));
      if (!metadata_finite || !design_finite)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint: refusing to save invalid or non-finite "
                      << (!metadata_finite && !design_finite ?
                          "metadata and design." :
                          (!metadata_finite ? "metadata." : "design."))
                      << std::endl;
         }
         return false;
      }

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
            ofs.write(reinterpret_cast<const char *>(&meta.design_iteration),
                      sizeof(meta.design_iteration));
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
            ofs << "format_version " << meta.format_version << "\n"
                << "design_iteration " << meta.design_iteration << "\n"
                << "objective_valid_for_design "
                << (meta.objective_valid_for_design ? 1 : 0) << "\n"
                << "objective " << std::setprecision(17) << meta.objective << "\n"
                << "volume_fraction " << meta.volume_fraction << "\n"
                << "n_mpi_ranks " << nranks << "\n"
                << "refinement_level " << meta.refinement_level << "\n"
                << "fe_order " << meta.fe_order << "\n"
                << "design_fe_order " << meta.design_fe_order << "\n";
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

   /// Read + broadcast the metadata and check the raw-control layout matches
   /// this run. State order intentionally need not match: only rho is loaded.
   bool ValidateCompatibility(int expected_ref_level, int expected_design_order,
                              OptimizationCheckpointMetadata &meta) const
   {
      bool read_ok = true;
      bool metadata_finite = true;
      if (myid_ == 0)
      {
         std::ifstream ifs(MetadataPath());
         read_ok = ifs.good();
         int legacy_iteration = -1;
         int objective_valid = 0;
         bool saw_format_version = false;
         bool saw_design_iteration = false;
         bool saw_objective_valid = false;
         std::string key;
         while (read_ok && ifs >> key)
         {
            if (key == "format_version")
            {
               ifs >> meta.format_version;
               saw_format_version = true;
            }
            else if (key == "design_iteration")
            {
               ifs >> meta.design_iteration;
               saw_design_iteration = true;
            }
            else if (key == "objective_valid_for_design")
            {
               ifs >> objective_valid;
               saw_objective_valid = true;
            }
            else if (key == "iteration")        { ifs >> legacy_iteration; }
            else if (key == "objective")        { ifs >> meta.objective; }
            else if (key == "volume_fraction")  { ifs >> meta.volume_fraction; }
            else if (key == "n_mpi_ranks")      { ifs >> meta.n_mpi_ranks; }
            else if (key == "refinement_level") { ifs >> meta.refinement_level; }
            else if (key == "fe_order")         { ifs >> meta.fe_order; }
            else if (key == "design_fe_order")  { ifs >> meta.design_fe_order; }
            else { std::string skip; ifs >> skip; }
            read_ok = !ifs.fail();
         }
         if (read_ok && !saw_format_version)
         {
            meta.format_version = 1;
            read_ok = legacy_iteration >= 0;
            if (read_ok)
            {
               meta.design_iteration = legacy_iteration + 1;
               meta.objective_valid_for_design = false;
               std::cout
                  << "Checkpoint: legacy iteration metadata; interpreting "
                  << "the payload as rho^" << meta.design_iteration
                  << " and marking its stored objective as pre-update.\n";
            }
         }
         else if (read_ok)
         {
            read_ok = meta.format_version == 2 && saw_design_iteration &&
                      saw_objective_valid &&
                      meta.design_iteration >= 0 &&
                      (objective_valid == 0 || objective_valid == 1);
            meta.objective_valid_for_design = objective_valid == 1;
         }
         // Legacy checkpoints coupled filter order to fe_order, with the
         // paired L2 control degree. Infer that design space when the field is
         // absent so existing checkpoints remain restartable.
         if (read_ok && meta.design_fe_order < 0)
         {
            meta.design_fe_order = meta.fe_order;
            std::cout << "Checkpoint: legacy metadata; inferring design FE order "
                      << meta.design_fe_order << " from fe_order.\n";
         }
         if (read_ok)
         {
            metadata_finite = MetadataIsFinite(meta);
         }
      }
      MPI_Bcast(&read_ok, 1, MPI_C_BOOL, 0, comm_);
      MPI_Bcast(&metadata_finite, 1, MPI_C_BOOL, 0, comm_);
      if (!read_ok)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint: cannot parse " << MetadataPath()
                      << std::endl;
         }
         return false;
      }
      if (!metadata_finite)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint: metadata contains a non-finite objective "
                      << "or volume fraction; refusing restart." << std::endl;
         }
         return false;
      }
      MPI_Bcast(&meta.format_version, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.design_iteration, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.objective_valid_for_design, 1, MPI_C_BOOL, 0, comm_);
      MPI_Bcast(&meta.objective, 1, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(&meta.volume_fraction, 1, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(&meta.n_mpi_ranks, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.refinement_level, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.fe_order, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.design_fe_order, 1, MPI_INT, 0, comm_);

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
      if (meta.design_fe_order != expected_design_order)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint incompatible: design FE order "
                      << meta.design_fe_order << " vs "
                      << expected_design_order << ".\n";
         }
         compatible = false;
      }
      return compatible;
   }

   /// Load this rank's design piece into rho_tv (must be pre-sized to the
   /// local control true-dof count). Call after ValidateCompatibility.
   bool Load(Vector &rho_tv,
             const OptimizationCheckpointMetadata *meta = nullptr) const
   {
      bool io_ok = true;
      {
         std::ifstream ifs(DesignPath(myid_), std::ios::binary);
         io_ok = ifs.good();
         int32_t magic = 0;
         int embedded_design_index = 0;
         int64_t n = 0;
         if (io_ok)
         {
            ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
            ifs.read(reinterpret_cast<char *>(&embedded_design_index),
                     sizeof(embedded_design_index));
            ifs.read(reinterpret_cast<char *>(&n), sizeof(n));
            io_ok = ifs.good() && magic == design_magic_ && n == rho_tv.Size();
            if (io_ok && meta)
            {
               const int expected_embedded_index =
                  meta->format_version >= 2 ? meta->design_iteration :
                  meta->design_iteration - 1;
               io_ok = embedded_design_index == expected_embedded_index;
            }
         }
         if (io_ok)
         {
            ifs.read(reinterpret_cast<char *>(rho_tv.GetData()),
                     n * sizeof(real_t));
            io_ok = ifs.good();
         }
      }

      // Keep both reductions unconditional: all ranks must reach them even when
      // one rank encountered an I/O error or a non-finite payload.
      const bool design_finite = io_ok ? DesignIsFinite(rho_tv) : true;
      const bool global_io_ok = AllOk(io_ok);
      const bool global_design_finite = AllOk(design_finite);
      if (myid_ == 0 && !global_io_ok)
      {
         std::cerr << "Checkpoint: a rank failed to load its design piece "
                   << "(magic/size/design-index mismatch or read error)."
                   << std::endl;
      }
      if (myid_ == 0 && !global_design_finite)
      {
         std::cerr << "Checkpoint: design contains non-finite values; "
                   << "refusing restart." << std::endl;
      }
      return global_io_ok && global_design_finite;
   }
};

} // namespace mfem

#endif // OPTIMIZATION_CHECKPOINT_HPP
