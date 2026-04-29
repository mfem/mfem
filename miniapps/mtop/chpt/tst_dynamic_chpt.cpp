#include "mfem.hpp"

#include <cmath>
#include <iomanip>
#include <string>

using namespace mfem;


/************************************************************************
 * The forward state consists of vector and several additional values.
 * The goal is to demontrate how different storages can be used together
 * with DynamicCheckpointing in order to avoid unnecessary memory 
 * allocations, data copies, and deallocations.
 * *********************************************************************/
struct State
{
   mfem::real_t time = 0.0;
   mfem::real_t obj  = 0.0;
   mfem::Vector v;
};

// ---------------------------
// Snapshot type used by storage:
// a lightweight view (non-owning).
// ---------------------------
struct StateSnapshotView
{
   mfem::real_t time = 0.0;
   mfem::real_t obj  = 0.0;

   // Points to n*sizeof(real_t) bytes.
   // - during Store(): points to State::v host data
   // - during Read(): points into storage's internal scratch buffer
   const unsigned char *v_bytes = nullptr;
};

// ---------------------------
// Packer for fixed-slot storage
// Layout in slot bytes:
//   [ time | obj | v[0..n-1] ]
// ---------------------------
class StateSnapshotViewPacker
{
public:
    explicit StateSnapshotViewPacker(int n) : n_(n)
    {
        MFEM_VERIFY(n_ > 0, "StateSnapshotViewPacker: n must be > 0.");
    }

    std::size_t SlotBytes() const
    {
        return (std::size_t)(2 + n_) * sizeof(mfem::real_t);
    }

    void Pack(const StateSnapshotView &s, void *dst) const
    {
        MFEM_VERIFY(dst != nullptr, "Pack: dst is null.");
        MFEM_VERIFY(s.v_bytes != nullptr, "Pack: v_bytes is null.");

        unsigned char *b = static_cast<unsigned char*>(dst);

        std::memcpy(b + 0*sizeof(mfem::real_t), &s.time, sizeof(mfem::real_t));
        std::memcpy(b + 1*sizeof(mfem::real_t), &s.obj,  sizeof(mfem::real_t));
        std::memcpy(b + 2*sizeof(mfem::real_t),
                    s.v_bytes,
                    (std::size_t)n_ * sizeof(mfem::real_t));
    }

    // IMPORTANT:
    // Unpack produces a view into the provided src buffer.
    // The resulting pointer is only valid as long as src remains unchanged.
    void Unpack(const void *src, StateSnapshotView &out) const
    {
        MFEM_VERIFY(src != nullptr, "Unpack: src is null.");

        const unsigned char *b = static_cast<const unsigned char*>(src);

        std::memcpy(&out.time, b + 0*sizeof(mfem::real_t), sizeof(mfem::real_t));
        std::memcpy(&out.obj,  b + 1*sizeof(mfem::real_t), sizeof(mfem::real_t));
        out.v_bytes = b + 2*sizeof(mfem::real_t);
    }

    int N() const { return n_; }

private:
    int n_ = 0;
};


/*****************************************************************************
 * Provides a recipe for runing dynamic checkpointing with  memory storage 
 * alocated as one single big block.
 * **************************************************************************/
void run_fixed_slot_memory_storage( mfem::real_t Tfinal /*Final time*/, 
                                    mfem::real_t dtime  /*time step*/,
                                    int s /* checkpoint budget (real checkpoints) */, 
                                    int n  /*State vector size*/,
                                    bool print=true)
{
   StateSnapshotViewPacker packer(n);
   
   using Storage = mfem::FixedSlotMemoryCheckpointStorage<StateSnapshotView, 
                                                            StateSnapshotViewPacker>;
   Storage storage(s, packer);

   // Snapshot type is StateSnapshotView
   using Checkpointing = mfem::DynamicCheckpointing<StateSnapshotView, Storage>;
   Checkpointing ckpt(s, storage);

    // Returns view of the State and avoids data transfer
    auto make_snapshot = [&](const State &u) -> StateSnapshotView
    {
        MFEM_VERIFY(u.v.Size() == n, "make_snapshot: State.v size changed!");
        const mfem::real_t *vh = u.v.HostRead(); // host pointer

        StateSnapshotView snap;
        snap.time = u.time;
        snap.obj  = u.obj;
        snap.v_bytes = reinterpret_cast<const unsigned char*>(vh);
        return snap;
    };

    //Transfers data from the snaphot view to the State u_out.
    auto restore_snapshot = [&](const StateSnapshotView &snap, State &u_out)
    {
        u_out.time = snap.time;
        u_out.obj  = snap.obj;

        if (u_out.v.Size() != n) { u_out.v.SetSize(n); }
        mfem::real_t *vh = u_out.v.HostWrite();

        std::memcpy(vh,
                snap.v_bytes,
                (std::size_t)n * sizeof(mfem::real_t));
    };

    using Step = Checkpointing::Step;

    State u;
    u.v.SetSize(n); u.v=0.0;

    auto primal_step = [&](State &u_inout, Step i)
    {
        // advance u_inout.v, u_inout.time, u_inout.obj
        if(print)
        std::cout<<"Forward step: "<<i<<" time="<<u_inout.time<<" obj="<<u_inout.obj<<std::endl;

        u_inout.time=i*0.1;
        u_inout.obj=i*0.2;
    };

    struct AdjointState { /* ... */ };
    AdjointState q; 

    auto adjoint_step = [&](AdjointState &q_ip1_inout, const State &u_i, Step i)
    {
        // use u_i.time/u_i.obj/u_i.v to update adjoint
        if(print)
        std::cout<<"Adjoint step: "<<i<<" time="<<u_i.time<<" obj="<<u_i.obj<<std::endl;

    };

    // Forward sweep (unknown number of steps)
    mfem::real_t t=0.0;
    Step i=0;
    while(t<Tfinal)
    {
        // Store snapshot of u_i when manager decides; then primal_step u_i -> u_{i+1}
        ckpt.ForwardStep(i, u, primal_step, make_snapshot);
        t+=dtime;
        ++i;
    }

    mfem::out << "Total number of steps m="<<i-1<<"\n";
    mfem::out << "\nBackward sweep.\n";

    const Step m=i;
    // Backward sweep
    State u_work;
    u_work.v.SetSize(n); // allocate once
    
    for (Step i = m - 1; i >= 0; --i)
    {
        ckpt.BackwardStep(i, q, u_work,
                         primal_step, adjoint_step,
                         make_snapshot, restore_snapshot);
        if (i == 0) { break; }
    }
}

/****************************************************************************
 * Provides a recipe for runing dynamic checkpointing with  file storage 
 * alocated as one single big file.
 ****************************************************************************/

/****************************************************************************
 * The dynamic algorithm conceptually maintains s + 1 checkpoints, but the 
 * last one is a placeholder checkpoint that “stores no solution and takes 
 * little memory”; only s are real stored solutions.
 * So FixedSlotFileCheckpointStorage needs only s slots.
 * File behavior for FixedSlotFileCheckpointStorage:
        - Writes a small header + s * slot_bytes payload region,
        - Erase(handle) just returns the slot to the free list 
          (file contents remain, will be overwritten later),
          no filesystem metadata churn.

 * Device memory: Runs on GPU -packing to a host file requires a host pointer. 
    Using:
        u.v.HostRead() for packing,
        u_out.v.HostWrite() for restore,
    keeps memmory access correct.
 * Snapshot pointer lifetime: In the read path, StateSnapshotViewPacker::Unpack 
   sets snap.v_bytes to point into the storage’s internal read buffer. That 
   pointer is only valid until the storage performs another Read/Store 
   (and in general, one should treat it as valid only during restore_snapshot).
 *****************************************************************************/

void run_fixed_slot_file_storage(mfem::real_t Tfinal /*Final time*/, 
                                    mfem::real_t dtime  /*time step*/,
                                    int s /* checkpoint budget (real checkpoints) */, 
                                    int n  /*State vector size*/,
                                    const std::string &file_path,
                                    bool print=true)
{
    // s = number of REAL checkpoints (placeholder doesn't store a snapshot) 
    MFEM_VERIFY(s > 0, "Need s > 0.");
    MFEM_VERIFY(n > 0, "Need n > 0.");    

    StateSnapshotViewPacker packer(n);

    // Fixed-slot FILE storage:
    // - single file (file_path)
    // - max_slots = s (real checkpoints)
    // - slot_bytes = packer.SlotBytes()
    // - truncate=true to start fresh
    // - flush_on_store=false for performance (set true if you want extra safety)
    using Storage = mfem::FixedSlotFileCheckpointStorage<StateSnapshotView, StateSnapshotViewPacker>;
    Storage storage(file_path,
                    /*max_slots=*/s,
                    /*packer=*/packer,
                    /*truncate=*/true,
                    /*flush_on_store=*/false);

    // Dynamic checkpointing manager using this storage.
    using Checkpointing = mfem::DynamicCheckpointing<StateSnapshotView, Storage>;
    Checkpointing ckpt(s, storage);

    // ---------------------------
    // Callbacks
    // ---------------------------

    // Convert State -> SnapshotView (no allocation).
    auto make_snapshot = [&](const State &u) -> StateSnapshotView
    {
        MFEM_VERIFY(u.v.Size() == n, "make_snapshot: State.v size changed!");

        // Ensure host pointer valid even when MFEM is using device memory.
        const mfem::real_t *vh = u.v.HostRead();

        StateSnapshotView snap;
        snap.time = u.time;
        snap.obj  = u.obj;
        snap.v_bytes = reinterpret_cast<const unsigned char*>(vh);
        return snap;
    };

    // Convert SnapshotView -> State (must COPY out of v_bytes).
    auto restore_snapshot = [&](const StateSnapshotView &snap, State &u_out)
    {
        u_out.time = snap.time;
        u_out.obj  = snap.obj;

        if (u_out.v.Size() != n) { u_out.v.SetSize(n); }

        mfem::real_t *vh = u_out.v.HostWrite();
        std::memcpy(vh,
                    snap.v_bytes,
                    (std::size_t)n * sizeof(mfem::real_t));
    };

    // Example primal/adjoint step signatures (replace with your own):
    using Step = Checkpointing::Step;

    auto primal_step = [&](State &u_inout, Step i)
    {
        if(print)
        std::cout<<"Forward step: "<<i<<" time="<<u_inout.time<<" obj="<<u_inout.obj<<std::endl;

        u_inout.time=i*0.1;
        u_inout.obj=i*0.2;
        // update u_inout.time, u_inout.obj, u_inout.v
    };

    struct AdjointState
    {
        // your adjoint variables, e.g., mfem::Vector lambda;
    };

    auto adjoint_step = [&](AdjointState &q_ip1_inout, const State &u_i, Step i)
    {
        // update adjoint using u_i
        if(print)
        std::cout<<"Adjoint step: "<<i<<" time="<<u_i.time<<" obj="<<u_i.obj<<std::endl;
    };

    // ---------------------------
    // Forward/backward skeleton
    // ---------------------------
    State u;
    u.v.SetSize(n);
    u.time = 0.0;
    u.obj  = 0.0;
    u.v = 0.0;    

    // Forward sweep (unknown number of steps) 
    mfem::real_t t=0.0;
    Step i=0;
    while(t<Tfinal)
    {
        ckpt.ForwardStep(i, u, primal_step, make_snapshot);
        t+=dtime;
        ++i;
    }

    mfem::out << "Total number of steps m="<<i-1<<"\n";
    mfem::out << "\nBackward sweep.\n";

    // Backward sweep
    AdjointState q;
    const Step m=i;
    State u_work;
    u_work.v.SetSize(n);

    for (Step i = m - 1; i >= 0; --i)
    {
        ckpt.BackwardStep(i, q, u_work,
                            primal_step, adjoint_step,
                            make_snapshot, restore_snapshot);
        if (i == 0) { break; }
    }   
}


/******************************************************************************
   The following classes implement the IO and the data necessery for dynamic
   checkpointing with mfem::FileCheckpointStorage (one file per stored snapshot).
   With FileCheckpointStorage, the Snapshot must own the data which will be 
   written, i.e. it cannot be a “view” containing pointers into State::v 
   as those pointers would be meaningless when read back.
*******************************************************************************/

// The StateSnapshot is just a copy of the State
using StateSnapshot=State;

struct StateSnapshotBinaryIO
{
   static void Write(std::ostream &os, const StateSnapshot &s)
   {
      os.write(reinterpret_cast<const char*>(&s.time), sizeof(mfem::real_t));
      os.write(reinterpret_cast<const char*>(&s.obj),  sizeof(mfem::real_t));
      MFEM_VERIFY(os.good(), "StateSnapshotBinaryIO: write time/obj failed.");

      const std::int64_t n = (std::int64_t)s.v.Size();
      os.write(reinterpret_cast<const char*>(&n), sizeof(n));
      MFEM_VERIFY(os.good(), "StateSnapshotBinaryIO: write vector size failed.");

      if (n > 0)
      {
         const mfem::real_t *vh = s.v.HostRead(); // ensure host pointer
         os.write(reinterpret_cast<const char*>(vh),
                  (std::streamsize)(n * (std::int64_t)sizeof(mfem::real_t)));
         MFEM_VERIFY(os.good(), "StateSnapshotBinaryIO: write vector payload failed.");
      }
   }

   static StateSnapshot Read(std::istream &is)
   {
      StateSnapshot s;

      is.read(reinterpret_cast<char*>(&s.time), sizeof(mfem::real_t));
      is.read(reinterpret_cast<char*>(&s.obj),  sizeof(mfem::real_t));
      MFEM_VERIFY(is.good(), "StateSnapshotBinaryIO: read time/obj failed.");

      std::int64_t n = 0;
      is.read(reinterpret_cast<char*>(&n), sizeof(n));
      MFEM_VERIFY(is.good(), "StateSnapshotBinaryIO: read vector size failed.");
      MFEM_VERIFY(n >= 0, "StateSnapshotBinaryIO: invalid negative vector size.");

      s.v.SetSize((int)n);
      if (n > 0)
      {
         mfem::real_t *vh = s.v.HostWrite(); // ensure host pointer
         is.read(reinterpret_cast<char*>(vh),
                 (std::streamsize)(n * (std::int64_t)sizeof(mfem::real_t)));
         MFEM_VERIFY(is.good(), "StateSnapshotBinaryIO: read vector payload failed.");
      }
      return s;
   }
};

/***************************************************************************
    Important differences vs FixedSlotFileCheckpointStorage
    Metadata overhead

    FileCheckpointStorage typically:
        *creates a new file for each Store() (plus a temp file rename),
        *deletes a file on each Erase() (unless keep_files=true).

    Dynamic checkpointing can perform many store/erase operations during 
    forward and during recomputation in reverse, so this can hammer filesystem 
    metadata on parallel filesystems.

    If that’s a concern, prefer:
        *fixed-slot single-file storage, or
        *segmented “range file” storage (few files + offsets).

    Correctness note about pointers:
    This is why we used an owning snapshot (StateSnapshot): storing a “view” 
    with v_bytes into file storage would store pointer values, which are 
    meaningless when read back.
*****************************************************************************/

void run_file_storage(mfem::real_t Tfinal /*Final time*/, 
                                    mfem::real_t dtime  /*time step*/,
                                    int s /* checkpoint budget (real checkpoints) */, 
                                    int n  /*State vector size*/,
                                    const std::string &directory,
                                    bool print=true)
{
    MFEM_VERIFY(s > 0, "Need s > 0.");
    MFEM_VERIFY(n > 0, "Need n > 0.");

    // One file per snapshot (create_dir=true). keep_files=false means Erase() removes files.
    using Storage = mfem::FileCheckpointStorage<StateSnapshot, StateSnapshotBinaryIO>;
    Storage storage(directory, "ckpt_", ".bin", /*create_dir=*/true, /*keep_files=*/false);   

    // Snapshot type = StateSnapshot (owning)
    using Checkpointing = mfem::DynamicCheckpointing<StateSnapshot, Storage>;
    Checkpointing ckpt(s, storage);

    using Step = Checkpointing::Step;

    // ---- Callbacks ----

    // Make an owning snapshot from the current State (deep copy of vector payload)
    auto make_snapshot = [&](const State &u) -> StateSnapshot
    {
        MFEM_VERIFY(u.v.Size() == n, "make_snapshot: State.v size changed!");

        StateSnapshot snap;
        snap.time = u.time;
        snap.obj  = u.obj;

        snap.v.SetSize(n);
        const mfem::real_t *src = u.v.HostRead();
        mfem::real_t *dst = snap.v.HostWrite();
        std::memcpy(dst, src, (std::size_t)n * sizeof(mfem::real_t));

        return snap; // move into storage
    };    

    // Restore from snapshot into an actual State (copy payload)
    auto restore_snapshot = [&](const StateSnapshot &snap, State &u_out)
    {
        u_out.time = snap.time;
        u_out.obj  = snap.obj;

        MFEM_VERIFY(snap.v.Size() == n, "restore_snapshot: snapshot vector size mismatch.");

        if (u_out.v.Size() != n) { u_out.v.SetSize(n); }

        const mfem::real_t *src = snap.v.HostRead();
        mfem::real_t *dst = u_out.v.HostWrite();
        std::memcpy(dst, src, (std::size_t)n * sizeof(mfem::real_t));
    };

    // Example primal step signature (replace with your integrator)
    auto primal_step = [&](State &u_inout, Step i)
    {
        // advance u_inout.time, u_inout.obj, u_inout.v
        if(print)
        std::cout<<"Forward step: "<<i<<" time="<<u_inout.time<<" obj="<<u_inout.obj<<std::endl;

        u_inout.time=i*0.1;
        u_inout.obj=i*0.2;
    };

    // Example adjoint state and adjoint step signature (replace with yours)
    struct AdjointState
    {
        // e.g., mfem::Vector lambda;
    };

    auto adjoint_step = [&](AdjointState &q_ip1_inout, const State &u_i, Step i)
    {
        // update adjoint using u_i
        if(print)
        std::cout<<"Adjoint step: "<<i<<" time="<<u_i.time<<" obj="<<u_i.obj<<std::endl;
    };

    // ---- Forward/backward skeleton ----

    State u;
    u.v.SetSize(n);
    u.time = 0.0;
    u.obj  = 0.0;
    u.v = 0.0;



    // Forward sweep (unknown number of steps) 
    mfem::real_t t=0.0;
    Step i=0;
    while(t<Tfinal)
    {
        ckpt.ForwardStep(i, u, primal_step, make_snapshot);
        t+=dtime;
        ++i;
    }

    mfem::out << "Total number of steps m="<<i-1<<"\n";
    mfem::out << "\nBackward sweep.\n";

    // backward
    const Step m=i;
    AdjointState q;
    State u_work;
    u_work.v.SetSize(n);

    for (Step i = m - 1; i >= 0; --i)
    {
        ckpt.BackwardStep(i, q, u_work,
                            primal_step, adjoint_step,
                            make_snapshot, restore_snapshot);
        if (i == 0) { break; }
    }    
}

/****************************************************************************
 * Using standard in memory storage. Works with variable snaphot sizes.
 * The size of the vector (n) is provided in order to compare the run to
 * the other implementations.
 ***************************************************************************/

void run_in_memory_storage(mfem::real_t Tfinal /*Final time*/, 
                                    mfem::real_t dtime  /*time step*/,
                                    int s /* checkpoint budget (real checkpoints) */, 
                                    int n  /*State vector size*/,
                                    bool print=true)
{
    InMemoryCheckpointStorage<StateSnapshot> storage;

    using Checkpointing = mfem::DynamicCheckpointing<StateSnapshot,
                                       InMemoryCheckpointStorage<StateSnapshot>>;

    Checkpointing ckpt(s, storage);    

    auto make_snapshot = [&](const State &u) -> StateSnapshot
    {
        StateSnapshot snap;
        snap.time = u.time;
        snap.obj  = u.obj;
        snap.v    = u.v;   // deep copy (MFEM handles memory)
        return snap;       // moved into storage by Store()
    };

    auto restore_snapshot = [&](const StateSnapshot &snap, State &u_out)
    {
        u_out.time = snap.time;
        u_out.obj  = snap.obj;
        u_out.v    = snap.v; // deep copy back into working state
    };

    using Step = Checkpointing::Step;

    // Primal step: u_i -> u_{i+1}
    auto primal_step = [&](State &u, Step i)
    {
        // ... update u.time, u.obj, u.v ...
        if(print)
        std::cout<<"Forward step: "<<i<<" time="<<u.time<<" obj="<<u.obj<<std::endl;

        u.time=i*0.1;
        u.obj=i*0.2;
    };

    // Adjoint step: q_{i+1} -> q_i using u_i
    struct AdjointState
    {
        // e.g. mfem::Vector lambda;
    };

    auto adjoint_step = [&](AdjointState &q, const State &u_i, Step i)
    {
        // ... update q using u_i ...
        if(print)
        std::cout<<"Adjoint step: "<<i<<" time="<<u_i.time<<" obj="<<u_i.obj<<std::endl;
    };

    State u;
    u.v.SetSize(n); 
    u.v=0.0;
    u.time = 0.0;
    u.obj  = 0.0;

    // Forward sweep (unknown number of steps)
    mfem::real_t t=0.0;
    Step i=0;
    while(t<Tfinal)
    {
        ckpt.ForwardStep(i, u, primal_step, make_snapshot);
        t+=dtime;
        ++i;
    }

    mfem::out << "Total number of steps m="<<i-1<<"\n";
    mfem::out << "\nBackward sweep.\n";

    // Backward
    const Step m=i;
    AdjointState q;
    State u_work;  // used to restore/recompute primal state at step i
    u_work.v.SetSize(u.v.Size());

    for (Step i = m - 1; i >= 0; --i)
    {
        ckpt.BackwardStep(i, q, u_work,
                        primal_step, adjoint_step,
                        make_snapshot, restore_snapshot);
        if (i == 0) { break; }
    }
}

int main(int argc, char *argv[])
{
    int s = 5/* checkpoint budget (real checkpoints) */;
    int n = 30 /* fixed State.v size */;    
    std::string filepath="dynamic_ckpts.bin"; /*file name for fixed slot single file*/
    std::string directory="dyn_ckpts"; 

    mfem::real_t Tfinal=1;
    mfem::real_t dt=0.1;
    int print=true;

    OptionsParser args(argc, argv);
    args.AddOption(&s,      "-s",   
                    "--checkpoints", "Checkpoint budget s (real checkpoints).");
    args.AddOption(&n,      "-n",   
                    "--size", "Size of the state vector.");
    args.AddOption(&Tfinal, "-T",   
                    "--tfinal",      "Terminate when accumulated time reaches Tfinal.");
    args.AddOption(&dt,    "-dt", "--dt",         "Time step.");    
    args.AddOption(&filepath, "-file", "--file", 
                    "File name for storing the checkpoints."); 
    args.AddOption(&directory, "-dir", "--dir",
                    "Directory for storing the checkpoints.");
    args.AddOption(&print, "-print", "--print",
                    "Turn on/off printing." );

    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(mfem::out);
        return 1;
    }
    args.PrintOptions(mfem::out);

    double run_time;

    mfem::out<<"\nFixedSlotMemoryStorage Start\n";
    mfem::tic();
    run_fixed_slot_memory_storage(Tfinal, dt, s, n, print);
    run_time=mfem::toc();
    mfem::out<<"FixedSlotMemoryStorage time= "<<run_time<<std::endl;

    mfem::out<<"\nFixedSlotFileStorage Start\n";
    mfem::tic();
    run_fixed_slot_file_storage(Tfinal,dt, s,n, filepath,print);
    run_time=mfem::toc();
    mfem::out<<"FixedSlotFileStorage time= "<<run_time<<std::endl;

    mfem::out<<"\nFileStorage Start\n";
    mfem::tic();
    run_file_storage(Tfinal,dt, s,n, directory, print);
    run_time=mfem::toc();
    mfem::out<<"FileStorage time= "<<run_time<<std::endl;

    mfem::out<<"\nInMemoryStorage\n";
    mfem::tic();
    run_in_memory_storage(Tfinal,dt, s,n, print);
    run_time=mfem::toc();
    mfem::out<<"InMemoryStorage time= "<<run_time<<std::endl;

    mfem::out << "\nDone.\n";
    return 0;   
}
