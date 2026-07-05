#include "mfem.hpp"


#include <cmath>
#include <iomanip>
#include <string>

using namespace mfem;


// the forward state consists of vector and 
// several additional values
struct State
{
   mfem::real_t time = 0.0;
   mfem::real_t obj  = 0.0;
   mfem::Vector v;
};

/// class to manage the Snaphot Layout
class StateSnapshotLayout
{
public:
   // Constructor: n - size of the vector in the state 
   explicit StateSnapshotLayout(int n) : n_(n)
   {
      MFEM_VERIFY(n_ > 0, "StateSnapshotLayout: n must be > 0.");
   }

   // returns the total size of the snapshot in mfem::real_t
   int N() const { return n_; }

   // returns the total number of bytes necessary to store the state
   int Bytes() const
   {
      return (2 + n_) * (int)sizeof(mfem::real_t);
   }

   // Pack State -> bytes (host bytes)
   void Pack(const State &s, unsigned char *dst, int bytes) const
   {
      MFEM_VERIFY(dst != nullptr, "Pack: dst is null.");
      MFEM_VERIFY(bytes == Bytes(), "Pack: snapshot byte size mismatch.");
      MFEM_VERIFY(s.v.Size() == n_, "Pack: State.v size mismatch.");

      // Ensure we read from host even if MFEM uses device memory.
      const mfem::real_t *vh = s.v.HostRead();

      std::memcpy(dst + 0*sizeof(mfem::real_t), &s.time, sizeof(mfem::real_t));
      std::memcpy(dst + 1*sizeof(mfem::real_t), &s.obj,  sizeof(mfem::real_t));
      std::memcpy(dst + 2*sizeof(mfem::real_t),
                  vh, (std::size_t)n_ * sizeof(mfem::real_t));
   }

   // Unpack bytes -> State (host write)
   void Unpack(const unsigned char *src, int bytes, State &out) const
   {
      MFEM_VERIFY(src != nullptr, "Unpack: src is null.");
      MFEM_VERIFY(bytes == Bytes(), "Unpack: snapshot byte size mismatch.");

      if (out.v.Size() != n_) { out.v.SetSize(n_); }
      mfem::real_t *vh = out.v.HostWrite();

      std::memcpy(&out.time, src + 0*sizeof(mfem::real_t), sizeof(mfem::real_t));
      std::memcpy(&out.obj,  src + 1*sizeof(mfem::real_t), sizeof(mfem::real_t));
      std::memcpy(vh,
                  src + 2*sizeof(mfem::real_t),
                  (std::size_t)n_ * sizeof(mfem::real_t));
   }

private:
   int n_ = 0;
};

/* The REVOLVE manager (as provided earlier) uses this callback style:

        * make_snapshot(const State&, uint8_t* out, size_t bytes)
        * restore_snapshot(State&, const uint8_t* in, size_t bytes)
        * primal_step(int step, State&)
        * adjoint_step(int step, const State&, AdjointState&)

        and a storage backend with:
        * Save(slot, bytes)
        * Load(slot, bytes)
    
    REVOLVE manager will:
        *request storing snapshots into checkpoint slots (takeshot)
        *request restoring a checkpoint slot (restore)
        *request recomputation forward (advance)
        *then request the next adjoint step (firsturn / youturn)
        all while using only (Ncheck) stored checkpoints.

    Fixed-step REVOLVE manager:
        * Snapshot is raw bytes packed/unpacked by your callbacks.
        * Storage is a fixed indexed array (Save(slot) / Load(slot)), 
               because REVOLVE addresses checkpoints by slot index.

    Best use when Nsteps is known in the begining of the simulation.
*/

int main(int argc, char *argv[])
{
   // Backend selection:
   //   0 = fixed-slot memory (single RAM block)
   //   1 = fixed-slot file   (single file with fixed offsets)
   int backend = 0;

    const int n = 100/* fixed State.v size */;
    StateSnapshotLayout layout(n);
    const std::size_t snapshot_bytes = (std::size_t)layout.Bytes();

    const int Nsteps = 20 /* known number of time steps */;
    const int Ncheck = 5 /* number of checkpoints (snaps) */;

    // Memory backend (single block)
    // FixedSlotMemoryStorage storage(Ncheck, snapshot_bytes);
    // FixedStepRevolveCheckpointing<FixedSlotMemoryStorage>  
    //                        cktp(Nsteps, Ncheck, snapshot_bytes, storage);

    // or file backend (single file)
    FixedSlotFileStorage storage("revolve_ckpts.bin", Ncheck, snapshot_bytes);
    FixedStepRevolveCheckpointing<FixedSlotFileStorage>  
                            cktp(Nsteps, Ncheck, snapshot_bytes, storage);

    
    auto make_snapshot = [&](const State &s, uint8_t *outb, std::size_t bytes)
    {
        MFEM_VERIFY(bytes == snapshot_bytes, "make_snapshot: byte size mismatch");
        layout.Pack(s, reinterpret_cast<unsigned char*>(outb), (int)bytes);
    };

    auto restore_snapshot = [&](State &s, const uint8_t *inb, std::size_t bytes)
    {
        MFEM_VERIFY(bytes == snapshot_bytes, "restore_snapshot: byte size mismatch");
        layout.Unpack(reinterpret_cast<const unsigned char*>(inb), (int)bytes, s);
    };

    // set the work state
    State u_work;
    u_work.v.SetSize(n); u_work.v=0.0; 

    State u;
    u.v.SetSize(n); u.v=0.0; u.obj=0.0; u.time=0.0;
    // init u.time, u.obj, u.v ...

    auto primal_step = [&](int step, State &u_inout)
    {
        (void)step;
        // advance u_inout -> u_{step+1}
    };

    struct AdjointState { /* ... */ };
    AdjointState lambda;

    auto adjoint_step = [&](int step, const State &u_step, AdjointState &lambda_inout)
    {
        (void)step;
        // update lambda_{step+1} -> lambda_step using u_step
    };

    // Forward: i = 0..Nsteps-1
    for (int i = 0; i < Nsteps; ++i)
    {
        cktp.ForwardStep(i, u, primal_step, make_snapshot);
    }

    // Reverse: i = Nsteps-1..0
    for (int i = Nsteps - 1; i >= 0; --i)
    {
        cktp.BackwardStep(i, lambda, u_work,
                        primal_step, adjoint_step,
                        make_snapshot, restore_snapshot);
    }

   mfem::out << "\nDone.\n";
   return 0;
};
