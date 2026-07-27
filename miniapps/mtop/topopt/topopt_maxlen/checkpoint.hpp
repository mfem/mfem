// Checkpoint: save/restore of topology-optimization design variables (the
// `rho` density true-dof vector plus one `alpha[r]` true-dof vector per ray
// direction, and optionally MMA optimizer state: xold1, xold2, lower/upper
// asymptotes) so a long-running MPI run can survive HPC wall-clock limits and
// resume instead of restarting from scratch.
//
// Each Save() writes one binary file per rank for rho, one per (ray, rank)
// for alpha, optionally 4 files for MMA state, and a metadata.txt (iteration,
// n_dir, rank count, refinement level, FE order, epsilon, plus the objective/
// residual normalization constants init_comp and init_thickness_res so a
// resumed run rescales the same way as the original) written last so its
// presence marks a complete checkpoint. All writes go through a ".tmp" +
// atomic rename so a crash mid-write never leaves a corrupt checkpoint that
// looks complete. ValidateCompatibility() checks rank count/refinement/order/
// n_dir before Load(). Typical usage: call Exists() and ValidateCompatibility()
// for verification, then Load() to restore state.

#pragma once

#include "mfem.hpp"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace mfem
{

class Checkpoint
{
private:
    static constexpr int32_t magic_rho_ = 0x52484F31;   // rho file tag
    static constexpr int32_t magic_alpha_ = 0x414C5048;  // alpha file tag
    static constexpr int32_t magic_xold1_ = 0x584F4C31;  // xold1 file tag
    static constexpr int32_t magic_xold2_ = 0x584F4C32;  // xold2 file tag
    static constexpr int32_t magic_low_ = 0x4C4F5731;    // lower asymptote tag
    static constexpr int32_t magic_upp_ = 0x55505031;    // upper asymptote tag

    std::string dir_;   // checkpoint directory
    MPI_Comm comm_;      // MPI communicator
    int myid_;           // rank id
    int nranks_;         // rank count

    int iteration_;          // last saved/loaded iteration
    int n_dir_;               // number of ray directions
    int refinement_level_;    // mesh refinement level
    int fe_order_;             // FE order
    real_t epsilon_;           // last saved/loaded epsilon
    real_t init_comp_;                       // objective normalization constant
    std::vector<real_t> init_thickness_res_; // per-ray residual normalization constants

    std::string MetadataPath() const;
    std::string RhoPath(int rank) const;
    std::string AlphaPath(int ray, int rank) const;
    std::string XOld1Path(int rank) const;
    std::string XOld2Path(int rank) const;
    std::string LowerAsymptotePath(int rank) const;
    std::string UpperAsymptotePath(int rank) const;

    // Rank 0 creates the checkpoint directory if it doesn't exist.
    bool CreateDirectoryIfNeeded() const;
    // Reduces a per-rank success flag to a single global result.
    bool AllOk(bool local_ok) const;

    // Writes one vector to disk via a .tmp file + atomic rename.
    bool SaveVector(const std::string& path, int32_t magic, const Vector& vec,
                    int iteration) const;
    // Reads one vector from disk, checking magic/size before copying data.
    bool LoadVector(const std::string& path, int32_t expected_magic, Vector& vec,
                    int& iteration) const;

    // MMA state vectors for proper restart (stored alongside rho/alpha)
    Vector xold1_tv_, xold2_tv_, lower_asymptotes_tv_, upper_asymptotes_tv_;

public:
    Checkpoint(const char* dir, MPI_Comm comm);

    // Saves rho and alpha true-dof vectors plus run metadata.
    bool Save(const Vector& rho_tv, const std::vector<Vector>& alpha_tv,
              int iteration = 0, int n_dir = -1, int ref_level = 0, int order = 0,
              real_t epsilon = 0, real_t init_comp = 1,
              const std::vector<real_t>& init_thickness_res = {});

    // Saves rho, alpha, and MMA state vectors plus run metadata (for proper restart).
    bool Save(const Vector& rho_tv, const std::vector<Vector>& alpha_tv,
              const Vector& xold1_tv, const Vector& xold2_tv,
              const Vector& lower_asymptotes_tv, const Vector& upper_asymptotes_tv,
              int iteration = 0, int n_dir = -1, int ref_level = 0, int order = 0,
              real_t epsilon = 0, real_t init_comp = 1,
              const std::vector<real_t>& init_thickness_res = {});

    // Whether a complete checkpoint (metadata.txt) is present.
    bool Exists() const;

    // Checks a saved checkpoint's rank/refinement/order/n_dir against the
    // current run before Load() is attempted.
    bool ValidateCompatibility(int expected_ref_level, int expected_order,
                               int expected_n_dir) const;

    // Loads rho/alpha into pre-sized vectors; call after ValidateCompatibility().
    bool Load(Vector& rho_tv, std::vector<Vector>& alpha_tv);

    int GetIteration() const { return iteration_; }
    int GetNDir() const { return n_dir_; }
    real_t GetEpsilon() const { return epsilon_; }
    real_t GetInitComp() const { return init_comp_; }
    const std::vector<real_t>& GetInitThicknessRes() const { return init_thickness_res_; }

    // MMA state accessors (for restoring optimizer state after load)
    const Vector& GetXOld1() const { return xold1_tv_; }
    const Vector& GetXOld2() const { return xold2_tv_; }
    const Vector& GetLowerAsymptotes() const { return lower_asymptotes_tv_; }
    const Vector& GetUpperAsymptotes() const { return upper_asymptotes_tv_; }
};

inline std::string Checkpoint::MetadataPath() const
{
    return dir_ + "/metadata.txt";
}

inline std::string Checkpoint::RhoPath(int rank) const
{
    std::ostringstream name;
    name << dir_ << "/rho." << std::setfill('0') << std::setw(6) << rank;
    return name.str();
}

inline std::string Checkpoint::AlphaPath(int ray, int rank) const
{
    std::ostringstream name;
    name << dir_ << "/alpha_" << ray << "."
         << std::setfill('0') << std::setw(6) << rank;
    return name.str();
}

inline std::string Checkpoint::XOld1Path(int rank) const
{
    std::ostringstream name;
    name << dir_ << "/xold1." << std::setfill('0') << std::setw(6) << rank;
    return name.str();
}

inline std::string Checkpoint::XOld2Path(int rank) const
{
    std::ostringstream name;
    name << dir_ << "/xold2." << std::setfill('0') << std::setw(6) << rank;
    return name.str();
}

inline std::string Checkpoint::LowerAsymptotePath(int rank) const
{
    std::ostringstream name;
    name << dir_ << "/lower_asymptote." << std::setfill('0') << std::setw(6) << rank;
    return name.str();
}

inline std::string Checkpoint::UpperAsymptotePath(int rank) const
{
    std::ostringstream name;
    name << dir_ << "/upper_asymptote." << std::setfill('0') << std::setw(6) << rank;
    return name.str();
}

inline bool Checkpoint::CreateDirectoryIfNeeded() const
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
                MFEM_WARNING("Checkpoint: failed to create directory: " << dir_);
            }
        }
    }
    MPI_Bcast(&ok, 1, MPI_C_BOOL, 0, comm_);
    return ok;
}

inline bool Checkpoint::AllOk(bool local_ok) const
{
    int loc = local_ok ? 1 : 0, glob = 0;
    MPI_Allreduce(&loc, &glob, 1, MPI_INT, MPI_MIN, comm_);
    return glob == 1;
}

inline bool Checkpoint::SaveVector(const std::string& path,
                                   int32_t magic, const Vector& vec,
                                   int iteration) const
{
    const std::string tmp = path + ".tmp";
    std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
    const int64_t n = vec.Size();
    bool ok = ofs.good();
    if (ok)
    {
        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&iteration), sizeof(iteration));
        ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
        ofs.write(reinterpret_cast<const char*>(vec.GetData()),
                  n * sizeof(real_t));
        ofs.close();
        ok = ofs.good() && (std::rename(tmp.c_str(), path.c_str()) == 0);
    }
    return ok;
}

inline bool Checkpoint::LoadVector(const std::string& path,
                                   int32_t expected_magic, Vector& vec,
                                   int& iteration) const
{
    std::ifstream ifs(path, std::ios::binary);
    bool ok = ifs.good();
    int32_t magic = 0;
    int64_t n = 0;
    if (ok)
    {
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char*>(&iteration), sizeof(iteration));
        ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
        ok = ifs.good() && magic == expected_magic && n == vec.Size();
    }
    if (ok)
    {
        ifs.read(reinterpret_cast<char*>(vec.GetData()), n * sizeof(real_t));
        ok = ifs.good();
    }
    if (!ok && myid_ == 0)
    {
        MFEM_WARNING("Checkpoint: rank " << myid_ << " failed to load "
                     << path << " (magic/size mismatch or read error).");
    }
    return ok;
}

inline Checkpoint::Checkpoint(const char* dir, MPI_Comm comm)
    : dir_(dir), comm_(comm), iteration_(0), n_dir_(0),
      refinement_level_(0), fe_order_(0), epsilon_(0), init_comp_(1)
{
    MPI_Comm_rank(comm_, &myid_);
    MPI_Comm_size(comm_, &nranks_);
}

inline bool Checkpoint::Save(const Vector& rho_tv,
                             const std::vector<Vector>& alpha_tv,
                             int iteration, int n_dir, int ref_level,
                             int order, real_t epsilon, real_t init_comp,
                             const std::vector<real_t>& init_thickness_res)
{
    // Legacy Save without MMA state - delegate to new version with empty MMA vectors
    static const Vector empty;
    return Save(rho_tv, alpha_tv, empty, empty, empty, empty,
                iteration, n_dir, ref_level, order, epsilon, init_comp,
                init_thickness_res);
}

inline bool Checkpoint::Save(const Vector& rho_tv,
                             const std::vector<Vector>& alpha_tv,
                             const Vector& xold1_tv, const Vector& xold2_tv,
                             const Vector& lower_asymptotes_tv,
                             const Vector& upper_asymptotes_tv,
                             int iteration, int n_dir, int ref_level,
                             int order, real_t epsilon, real_t init_comp,
                             const std::vector<real_t>& init_thickness_res)
{
    if (!CreateDirectoryIfNeeded()) { return false; }

    iteration_ = iteration;
    n_dir_ = (n_dir < 0) ? static_cast<int>(alpha_tv.size()) : n_dir;
    refinement_level_ = ref_level;
    fe_order_ = order;
    epsilon_ = epsilon;
    init_comp_ = init_comp;
    init_thickness_res_ = init_thickness_res;

    bool ok = SaveVector(RhoPath(myid_), magic_rho_, rho_tv, iteration);

    for (int r = 0; r < n_dir_ && ok; r++)
    {
        if (r < static_cast<int>(alpha_tv.size()))
        {
            ok = SaveVector(AlphaPath(r, myid_), magic_alpha_,
                           alpha_tv[r], iteration);
        }
    }

    // Save MMA state if provided (non-empty vectors)
    if (ok && xold1_tv.Size() > 0)
    {
        ok = SaveVector(XOld1Path(myid_), magic_xold1_, xold1_tv, iteration);
    }
    if (ok && xold2_tv.Size() > 0)
    {
        ok = SaveVector(XOld2Path(myid_), magic_xold2_, xold2_tv, iteration);
    }
    if (ok && lower_asymptotes_tv.Size() > 0)
    {
        ok = SaveVector(LowerAsymptotePath(myid_), magic_low_,
                       lower_asymptotes_tv, iteration);
    }
    if (ok && upper_asymptotes_tv.Size() > 0)
    {
        ok = SaveVector(UpperAsymptotePath(myid_), magic_upp_,
                       upper_asymptotes_tv, iteration);
    }

    if (!AllOk(ok))
    {
        if (myid_ == 0)
        {
            MFEM_WARNING("Checkpoint: design write failed on some rank.");
        }
        return false;
    }

    bool meta_ok = true;
    if (myid_ == 0)
    {
        const std::string tmp = MetadataPath() + ".tmp";
        std::ofstream ofs(tmp, std::ios::trunc);
        meta_ok = ofs.good();
        if (meta_ok)
        {
            ofs << "iteration " << iteration_ << "\n"
                << "n_dir " << n_dir_ << "\n"
                << "n_mpi_ranks " << nranks_ << "\n"
                << "refinement_level " << refinement_level_ << "\n"
                << "fe_order " << fe_order_ << "\n"
                << "epsilon " << std::setprecision(17) << epsilon_ << "\n"
                << "init_comp " << std::setprecision(17) << init_comp_ << "\n";
            for (int r = 0; r < static_cast<int>(init_thickness_res_.size()); r++)
            {
                ofs << "init_thickness_res_" << r << " "
                    << std::setprecision(17) << init_thickness_res_[r] << "\n";
            }
            ofs.close();
            meta_ok = ofs.good() &&
                      (std::rename(tmp.c_str(), MetadataPath().c_str()) == 0);
        }
        if (!meta_ok)
        {
            MFEM_WARNING("Checkpoint: metadata write failed.");
        }
    }
    MPI_Bcast(&meta_ok, 1, MPI_C_BOOL, 0, comm_);
    if (myid_ == 0)
    {
        if (meta_ok)
        {
            mfem::out << "Checkpoint saved at iteration " << iteration_ << "\n";
        }
        else
        {
            MFEM_WARNING("Failed to save checkpoint at iteration " << iteration_);
        }
    }
    return meta_ok;
}

inline bool Checkpoint::Exists() const
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

inline bool Checkpoint::ValidateCompatibility(int expected_ref_level,
                                              int expected_order,
                                              int expected_n_dir) const
{
    int iteration = 0, n_dir = 0, n_ranks = 0, ref_level = 0, order = 0;
    bool read_ok = true;

    if (myid_ == 0)
    {
        std::ifstream ifs(MetadataPath());
        read_ok = ifs.good();
        std::string key;
        while (read_ok && ifs >> key)
        {
            if (key == "iteration")             { ifs >> iteration; }
            else if (key == "n_dir")            { ifs >> n_dir; }
            else if (key == "n_mpi_ranks")      { ifs >> n_ranks; }
            else if (key == "refinement_level") { ifs >> ref_level; }
            else if (key == "fe_order")         { ifs >> order; }
            else { std::string skip; ifs >> skip; }
            read_ok = !ifs.fail();
        }
    }

    MPI_Bcast(&read_ok, 1, MPI_C_BOOL, 0, comm_);
    if (!read_ok)
    {
        if (myid_ == 0)
        {
            MFEM_WARNING("Checkpoint: cannot parse " << MetadataPath());
        }
        return false;
    }

    MPI_Bcast(&iteration, 1, MPI_INT, 0, comm_);
    MPI_Bcast(&n_dir, 1, MPI_INT, 0, comm_);
    MPI_Bcast(&n_ranks, 1, MPI_INT, 0, comm_);
    MPI_Bcast(&ref_level, 1, MPI_INT, 0, comm_);
    MPI_Bcast(&order, 1, MPI_INT, 0, comm_);

    bool compatible = true;
    if (n_ranks != nranks_)
    {
        if (myid_ == 0)
        {
            MFEM_WARNING("Checkpoint incompatible: written with " << n_ranks
                         << " ranks, running with " << nranks_ << ".");
        }
        compatible = false;
    }
    if (ref_level != expected_ref_level)
    {
        if (myid_ == 0)
        {
            MFEM_WARNING("Checkpoint incompatible: refinement level "
                         << ref_level << " vs " << expected_ref_level << ".");
        }
        compatible = false;
    }
    if (order != expected_order)
    {
        if (myid_ == 0)
        {
            MFEM_WARNING("Checkpoint incompatible: FE order " << order
                         << " vs " << expected_order << ".");
        }
        compatible = false;
    }
    if (n_dir != expected_n_dir)
    {
        if (myid_ == 0)
        {
            MFEM_WARNING("Checkpoint incompatible: n_dir " << n_dir
                         << " vs " << expected_n_dir << ".");
        }
        compatible = false;
    }

    return compatible;
}

inline bool Checkpoint::Load(Vector& rho_tv,
                             std::vector<Vector>& alpha_tv)
{
    init_thickness_res_.clear();
    if (myid_ == 0)
    {
        std::ifstream ifs(MetadataPath());
        if (!ifs.good()) { return false; }
        std::string key;
        while (ifs >> key)
        {
            if (key == "iteration")        { ifs >> iteration_; }
            else if (key == "n_dir")       { ifs >> n_dir_; }
            else if (key == "epsilon")     { ifs >> epsilon_; }
            else if (key == "init_comp")   { ifs >> init_comp_; }
            else if (key.rfind("init_thickness_res_", 0) == 0)
            {
                real_t val; ifs >> val;
                init_thickness_res_.push_back(val);
            }
            else { std::string skip; ifs >> skip; }
        }
    }
    MPI_Bcast(&iteration_, 1, MPI_INT, 0, comm_);
    MPI_Bcast(&n_dir_, 1, MPI_INT, 0, comm_);
    MPI_Bcast(&epsilon_, 1, MPITypeMap<real_t>::mpi_type, 0, comm_);
    MPI_Bcast(&init_comp_, 1, MPITypeMap<real_t>::mpi_type, 0, comm_);
    if (myid_ == 0) { init_thickness_res_.resize(n_dir_, real_t(1)); }
    else { init_thickness_res_.assign(n_dir_, real_t(1)); }
    if (n_dir_ > 0)
    {
        MPI_Bcast(init_thickness_res_.data(), n_dir_, MPITypeMap<real_t>::mpi_type, 0, comm_);
    }

    int iter_check = 0;
    bool ok = LoadVector(RhoPath(myid_), magic_rho_, rho_tv, iter_check);

    for (int r = 0; r < n_dir_ && ok; r++)
    {
        if (r < static_cast<int>(alpha_tv.size()))
        {
            ok = LoadVector(AlphaPath(r, myid_), magic_alpha_,
                           alpha_tv[r], iter_check);
        }
    }

    // Try to load MMA state (optional - may not exist in older checkpoints)
    // Allocate storage for MMA state vectors (same size as rho)
    xold1_tv_.SetSize(rho_tv.Size());
    xold2_tv_.SetSize(rho_tv.Size());
    lower_asymptotes_tv_.SetSize(rho_tv.Size());
    upper_asymptotes_tv_.SetSize(rho_tv.Size());

    bool mma_ok = LoadVector(XOld1Path(myid_), magic_xold1_, xold1_tv_, iter_check);
    mma_ok = mma_ok && LoadVector(XOld2Path(myid_), magic_xold2_, xold2_tv_, iter_check);
    mma_ok = mma_ok && LoadVector(LowerAsymptotePath(myid_), magic_low_,
                                   lower_asymptotes_tv_, iter_check);
    mma_ok = mma_ok && LoadVector(UpperAsymptotePath(myid_), magic_upp_,
                                   upper_asymptotes_tv_, iter_check);

    if (!mma_ok && myid_ == 0)
    {
        MFEM_WARNING("Checkpoint: MMA state not found (old checkpoint format). "
                     "MMA will restart fresh - expect conservative first steps.");
        // Zero out the vectors so GetXOld1() etc return empty state
        xold1_tv_.SetSize(0);
        xold2_tv_.SetSize(0);
        lower_asymptotes_tv_.SetSize(0);
        upper_asymptotes_tv_.SetSize(0);
    }

    return AllOk(ok);
}

} // namespace mfem
