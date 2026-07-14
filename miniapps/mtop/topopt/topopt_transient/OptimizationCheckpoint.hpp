// =============================================================================
// Optimization Checkpoint System for Transient Topology Optimization
// =============================================================================
//
// Saves/loads MMA outer-loop state at each iteration for:
//   1. Restart after HPC time limit (24h jobs)
//   2. Recovery from job cancellation (lose only current incomplete iteration)
//
// DISTINCT FROM trajectory checkpointing (mtop-chkpt/) which handles RK4 states.
//
// Strategy: Fixed filenames (3 files), overwritten each iteration to respect
//           HPC filesystem inode limits.
//
// MPI Compatibility: MUST restart with same number of MPI ranks.
//
// =============================================================================

#ifndef OPTIMIZATION_CHECKPOINT_HPP
#define OPTIMIZATION_CHECKPOINT_HPP

#include "mfem.hpp"
#include <fstream>
#include <string>
#include <sys/stat.h>

namespace mfem
{

// =============================================================================
// OPTIMIZATION CHECKPOINT METADATA
// =============================================================================
struct OptimizationCheckpointMetadata
{
   // Iteration state
   int iteration;                // Last completed MMA iteration
   real_t objective;             // J(rho) at this iteration
   real_t volume_fraction;       // Current volume fraction
   real_t convergence_error;     // L1 design change ||rho - rho_old||

   // Problem configuration (must match on restart)
   int n_active_dofs;            // Size of active design vector
   int n_mpi_ranks;              // CRITICAL: Must match when restarting
   int refinement_level;         // Mesh refinement level
   int fe_order;                 // Finite element order

   // Output directory tracking (for ParaView continuity)
   char output_dir_path[256];    // ParaView output directory used in this run

   OptimizationCheckpointMetadata()
      : iteration(0), objective(0.0), volume_fraction(0.0),
        convergence_error(1e10), n_active_dofs(0), n_mpi_ranks(1),
        refinement_level(0), fe_order(1)
   {
      std::strncpy(output_dir_path, "ParaView", 255);
      output_dir_path[255] = '\0';
   }

   void SetOutputDir(const char* dir)
   {
      std::strncpy(output_dir_path, dir, 255);
      output_dir_path[255] = '\0';
   }

   std::string GetOutputDir() const
   {
      return std::string(output_dir_path);
   }
};

// =============================================================================
// MMA OPTIMIZER STATE (what's needed for restart)
// =============================================================================
// The MMA algorithm needs design history to compute asymptotes.
// If we don't save this, restart would reset asymptotes → poor convergence.
struct MMACheckpointState
{
   Vector xval;      // Current design (active DOFs only)
   Vector xold1;     // Design from previous iteration
   Vector xold2;     // Design from 2 iterations ago
   Vector low;       // Lower asymptotes
   Vector upp;       // Upper asymptotes

   void SetSize(int n)
   {
      xval.SetSize(n);
      xold1.SetSize(n);
      xold2.SetSize(n);
      low.SetSize(n);
      upp.SetSize(n);
   }

   // Initialize for first iteration (no history yet)
   void InitializeFromDesign(const Vector& rho_initial)
   {
      int n = rho_initial.Size();
      SetSize(n);
      xval = rho_initial;
      xold1 = rho_initial;
      xold2 = rho_initial;
      low = 0.0;
      upp = 1.0;
   }
};

// =============================================================================
// OPTIMIZATION CHECKPOINT MANAGER
// =============================================================================
class OptimizationCheckpoint
{
private:
   std::string checkpoint_dir_;
   MPI_Comm comm_;
   int myid_;

   std::string MetadataPath() const { return checkpoint_dir_ + "/metadata.bin"; }
   std::string DesignPath() const { return checkpoint_dir_ + "/design.gf"; }
   std::string MMAPath() const { return checkpoint_dir_ + "/mma_state.bin"; }

   bool CreateDirectoryIfNeeded()
   {
      if (myid_ == 0)
      {
         struct stat st;
         if (stat(checkpoint_dir_.c_str(), &st) != 0)
         {
            // Directory doesn't exist, create it
            if (mkdir(checkpoint_dir_.c_str(), 0755) != 0)
            {
               std::cerr << "Failed to create checkpoint directory: "
                         << checkpoint_dir_ << std::endl;
               return false;
            }
         }
      }
      MPI_Barrier(comm_);
      return true;
   }

public:
   OptimizationCheckpoint(const std::string& dir, MPI_Comm comm)
      : checkpoint_dir_(dir), comm_(comm)
   {
      myid_ = Mpi::WorldRank();
   }

   // =========================================================================
   // SAVE CHECKPOINT (overwrites previous)
   // =========================================================================
   // Call this at the END of each successful MMA iteration (after MMA update).
   // If job crashes during save, the .tmp file is incomplete and we restart
   // from the previous complete checkpoint.
   bool Save(const OptimizationCheckpointMetadata& meta,
             const ParGridFunction& rho,
             const MMACheckpointState& mma_state)
   {
      if (!CreateDirectoryIfNeeded()) return false;

      // 1. Save metadata (rank 0 only)
      if (myid_ == 0)
      {
         std::ofstream meta_file(MetadataPath(), std::ios::binary | std::ios::trunc);
         if (!meta_file)
         {
            std::cerr << "Failed to open metadata file for writing\n";
            return false;
         }

         meta_file.write(reinterpret_cast<const char*>(&meta.iteration), sizeof(int));
         meta_file.write(reinterpret_cast<const char*>(&meta.objective), sizeof(real_t));
         meta_file.write(reinterpret_cast<const char*>(&meta.volume_fraction), sizeof(real_t));
         meta_file.write(reinterpret_cast<const char*>(&meta.convergence_error), sizeof(real_t));
         meta_file.write(reinterpret_cast<const char*>(&meta.n_active_dofs), sizeof(int));
         meta_file.write(reinterpret_cast<const char*>(&meta.n_mpi_ranks), sizeof(int));
         meta_file.write(reinterpret_cast<const char*>(&meta.refinement_level), sizeof(int));
         meta_file.write(reinterpret_cast<const char*>(&meta.fe_order), sizeof(int));
         meta_file.write(meta.output_dir_path, 256);  // Save output directory path
         meta_file.close();
      }

      // 2. Save density field (MPI-parallel, atomic via temp file)
      //    Each rank writes its local portion to the file.
      std::string design_temp = DesignPath() + ".tmp";
      std::ofstream rho_ofs(design_temp, std::ios::binary);
      if (!rho_ofs)
      {
         if (myid_ == 0)
         {
            std::cerr << "Failed to open design file for writing\n";
         }
         return false;
      }
      rho.Save(rho_ofs);
      rho_ofs.close();

      // Atomic rename (only rank 0, after all ranks finished writing)
      MPI_Barrier(comm_);
      if (myid_ == 0)
      {
         std::rename(design_temp.c_str(), DesignPath().c_str());
      }
      MPI_Barrier(comm_);

      // 3. Save MMA state (rank 0 only, these vectors are replicated)
      if (myid_ == 0)
      {
         std::ofstream mma_file(MMAPath(), std::ios::binary | std::ios::trunc);
         if (!mma_file)
         {
            std::cerr << "Failed to open MMA state file for writing\n";
            return false;
         }

         int n = mma_state.xval.Size();
         mma_file.write(reinterpret_cast<const char*>(&n), sizeof(int));
         mma_file.write(reinterpret_cast<const char*>(mma_state.xval.GetData()), n * sizeof(real_t));
         mma_file.write(reinterpret_cast<const char*>(mma_state.xold1.GetData()), n * sizeof(real_t));
         mma_file.write(reinterpret_cast<const char*>(mma_state.xold2.GetData()), n * sizeof(real_t));
         mma_file.write(reinterpret_cast<const char*>(mma_state.low.GetData()), n * sizeof(real_t));
         mma_file.write(reinterpret_cast<const char*>(mma_state.upp.GetData()), n * sizeof(real_t));
         mma_file.close();
      }

      MPI_Barrier(comm_);
      return true;
   }

   // =========================================================================
   // LOAD CHECKPOINT
   // =========================================================================
   // Returns false if checkpoint doesn't exist or is incompatible.
   // If incompatible (wrong #ranks, refinement, etc), prints error and fails.
   bool Load(OptimizationCheckpointMetadata& meta,
             ParGridFunction& rho,
             MMACheckpointState& mma_state)
   {
      // 1. Load metadata (rank 0 reads, broadcasts)
      if (myid_ == 0)
      {
         std::ifstream meta_file(MetadataPath(), std::ios::binary);
         if (!meta_file)
         {
            std::cerr << "Checkpoint metadata not found: " << MetadataPath() << std::endl;
            return false;
         }

         meta_file.read(reinterpret_cast<char*>(&meta.iteration), sizeof(int));
         meta_file.read(reinterpret_cast<char*>(&meta.objective), sizeof(real_t));
         meta_file.read(reinterpret_cast<char*>(&meta.volume_fraction), sizeof(real_t));
         meta_file.read(reinterpret_cast<char*>(&meta.convergence_error), sizeof(real_t));
         meta_file.read(reinterpret_cast<char*>(&meta.n_active_dofs), sizeof(int));
         meta_file.read(reinterpret_cast<char*>(&meta.n_mpi_ranks), sizeof(int));
         meta_file.read(reinterpret_cast<char*>(&meta.refinement_level), sizeof(int));
         meta_file.read(reinterpret_cast<char*>(&meta.fe_order), sizeof(int));
         meta_file.read(meta.output_dir_path, 256);  // Load output directory path
         meta_file.close();
      }

      // Broadcast metadata to all ranks
      MPI_Bcast(&meta.iteration, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.objective, 1, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(&meta.volume_fraction, 1, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(&meta.convergence_error, 1, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(&meta.n_active_dofs, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.n_mpi_ranks, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.refinement_level, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.fe_order, 1, MPI_INT, 0, comm_);

      // CRITICAL SAFETY CHECKS
      int current_ranks = Mpi::WorldSize();
      if (meta.n_mpi_ranks != current_ranks)
      {
         if (myid_ == 0)
         {
            std::cerr << "\n*** CHECKPOINT ERROR: MPI rank mismatch ***\n"
                      << "Checkpoint was created with " << meta.n_mpi_ranks << " ranks\n"
                      << "Current job is using " << current_ranks << " ranks\n"
                      << "Cannot restart with different number of MPI ranks!\n"
                      << "Please resubmit with: mpirun -np " << meta.n_mpi_ranks << " ...\n"
                      << std::endl;
         }
         return false;
      }

      // 2. Load density field (MPI-parallel)
      //    Each rank reads its local portion from the file.
      std::ifstream rho_ifs(DesignPath(), std::ios::binary);
      if (!rho_ifs)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint design file not found: " << DesignPath() << std::endl;
         }
         return false;
      }
      rho.Load(rho_ifs, rho.Size());
      rho_ifs.close();

      // 3. Load MMA state (rank 0 reads, broadcasts)
      int n = meta.n_active_dofs;
      mma_state.SetSize(n);

      if (myid_ == 0)
      {
         std::ifstream mma_file(MMAPath(), std::ios::binary);
         if (!mma_file)
         {
            std::cerr << "Checkpoint MMA state not found: " << MMAPath() << std::endl;
            return false;
         }

         int n_read;
         mma_file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
         if (n_read != n)
         {
            std::cerr << "MMA state size mismatch: expected " << n
                      << ", got " << n_read << std::endl;
            return false;
         }

         mma_file.read(reinterpret_cast<char*>(mma_state.xval.GetData()), n * sizeof(real_t));
         mma_file.read(reinterpret_cast<char*>(mma_state.xold1.GetData()), n * sizeof(real_t));
         mma_file.read(reinterpret_cast<char*>(mma_state.xold2.GetData()), n * sizeof(real_t));
         mma_file.read(reinterpret_cast<char*>(mma_state.low.GetData()), n * sizeof(real_t));
         mma_file.read(reinterpret_cast<char*>(mma_state.upp.GetData()), n * sizeof(real_t));
         mma_file.close();
      }

      // Broadcast MMA state to all ranks
      MPI_Bcast(mma_state.xval.GetData(), n, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(mma_state.xold1.GetData(), n, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(mma_state.xold2.GetData(), n, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(mma_state.low.GetData(), n, MPITypeMap<real_t>::mpi_type, 0, comm_);
      MPI_Bcast(mma_state.upp.GetData(), n, MPITypeMap<real_t>::mpi_type, 0, comm_);

      return true;
   }

   // =========================================================================
   // CHECK IF CHECKPOINT EXISTS
   // =========================================================================
   bool Exists() const
   {
      bool exists = false;
      if (myid_ == 0)
      {
         std::ifstream test(MetadataPath());
         exists = test.good();
         test.close();
      }
      MPI_Bcast(&exists, 1, MPI_C_BOOL, 0, comm_);
      return exists;
   }

   // =========================================================================
   // VALIDATE COMPATIBILITY (before attempting load)
   // =========================================================================
   // Quick check if checkpoint is compatible with current run configuration.
   // Returns true + fills meta if compatible, false otherwise.
   bool ValidateCompatibility(int expected_ref_level, int expected_order,
                              OptimizationCheckpointMetadata& meta) const
   {
      if (!Exists()) return false;

      // Load metadata to check compatibility
      if (myid_ == 0)
      {
         std::ifstream meta_file(MetadataPath(), std::ios::binary);
         if (!meta_file) return false;

         meta_file.read(reinterpret_cast<char*>(&meta.iteration), sizeof(int));
         meta_file.read(reinterpret_cast<char*>(&meta.objective), sizeof(real_t));
         meta_file.read(reinterpret_cast<char*>(&meta.volume_fraction), sizeof(real_t));
         meta_file.read(reinterpret_cast<char*>(&meta.convergence_error), sizeof(real_t));
         meta_file.read(reinterpret_cast<char*>(&meta.n_active_dofs), sizeof(int));
         meta_file.read(reinterpret_cast<char*>(&meta.n_mpi_ranks), sizeof(int));
         meta_file.read(reinterpret_cast<char*>(&meta.refinement_level), sizeof(int));
         meta_file.read(reinterpret_cast<char*>(&meta.fe_order), sizeof(int));
         meta_file.read(meta.output_dir_path, 256);  // Load output directory path
         meta_file.close();
      }

      MPI_Bcast(&meta.iteration, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.n_mpi_ranks, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.refinement_level, 1, MPI_INT, 0, comm_);
      MPI_Bcast(&meta.fe_order, 1, MPI_INT, 0, comm_);

      bool compatible = true;
      int current_ranks = Mpi::WorldSize();

      if (meta.n_mpi_ranks != current_ranks)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint incompatible: MPI ranks mismatch ("
                      << meta.n_mpi_ranks << " vs " << current_ranks << ")\n";
         }
         compatible = false;
      }

      if (meta.refinement_level != expected_ref_level)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint incompatible: refinement level mismatch ("
                      << meta.refinement_level << " vs " << expected_ref_level << ")\n";
         }
         compatible = false;
      }

      if (meta.fe_order != expected_order)
      {
         if (myid_ == 0)
         {
            std::cerr << "Checkpoint incompatible: FE order mismatch ("
                      << meta.fe_order << " vs " << expected_order << ")\n";
         }
         compatible = false;
      }

      return compatible;
   }
};

} // namespace mfem

#endif // OPTIMIZATION_CHECKPOINT_HPP
