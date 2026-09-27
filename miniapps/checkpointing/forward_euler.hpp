// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CHECKPOINT_DEMO_FORWARD_EULER
#define MFEM_CHECKPOINT_DEMO_FORWARD_EULER

#include "mfem.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace mfem
{
namespace checkpoint_demo
{
namespace forward_euler_detail
{

/// Complete ODE continuation encoded by the miniapp adapter.
struct ODECheckpointData
{
   Vector state;
   TimePoint time;
   real_t dt = 0.0;
   Snapshot restart;
};

/// Internal encoder/decoder for the stable exact ODE checkpoint payload.
/** The 64-byte header contains, in order, an eight-byte magic value, version,
    byte-order and scalar-width markers, reserved bits, logical ID, trajectory
    step, physical time, continuation step size, Vector length, and restart
    length. Vector entries and restart bytes follow the header. */
class ODECheckpointSerializer
{
public:
   /// Current persisted payload version.
   static constexpr std::uint32_t FormatVersion = 1;

   /// Exact encoded header size in bytes.
   static constexpr std::size_t HeaderSize = 64;

   /// Serialize @a checkpoint and record its logical @a id.
   static Snapshot Encode(CheckpointId id, const ODECheckpointData &checkpoint);

   /// Decode @a snapshot and verify that it contains @a expected_id.
   static ODECheckpointData Decode(CheckpointId expected_id,
                                   const Snapshot &snapshot);
};

const std::uint64_t checkpoint_magic = ///< Stable persisted format magic.
   UINT64_C(0x4d46454d43503100);
const unsigned char little_endian_encoding = 1; ///< Byte-order marker.
const unsigned char binary64_encoding = 8; ///< Persisted scalar width.

/// Append unsigned integer @a value in canonical little-endian order.
template <typename T>
void AppendLittleEndian(std::vector<unsigned char> &bytes, T value)
{
   for (std::size_t i = 0; i < sizeof(T); i++)
   {
      bytes.push_back(static_cast<unsigned char>(value & T(0xff)));
      value = static_cast<T>(value >> 8);
   }
}

/// Read an unsigned little-endian integer and advance @a offset.
template <typename T>
T ReadLittleEndian(const Snapshot &snapshot, std::size_t &offset)
{
   if (offset > snapshot.Size() || sizeof(T) > snapshot.Size() - offset)
   {
      throw InvalidCheckpointFormat("checkpoint header is truncated");
   }
   T value = 0;
   for (std::size_t i = 0; i < sizeof(T); i++)
   {
      value |= static_cast<T>(snapshot.Data()[offset + i]) << (8 * i);
   }
   offset += sizeof(T);
   return value;
}

/// Return the exact IEEE binary64 bit representation of @a value.
inline std::uint64_t DoubleBits(double value)
{
   std::uint64_t bits = 0;
   static_assert(sizeof(bits) == sizeof(value), "binary64 size mismatch");
   std::memcpy(&bits, &value, sizeof(bits));
   return bits;
}

/// Reconstruct an IEEE binary64 value from @a bits.
inline double BitsDouble(std::uint64_t bits)
{
   double value = 0.0;
   std::memcpy(&value, &bits, sizeof(value));
   return value;
}

/// Convert a persisted size after checking platform capacity.
inline std::size_t CheckedSize(std::uint64_t value, const char *description)
{
   if constexpr (sizeof(std::size_t) < sizeof(value))
   {
      if (value > std::numeric_limits<std::size_t>::max())
      {
         throw InvalidCheckpointFormat(std::string(description) +
                                       " exceeds platform capacity");
      }
   }
   return static_cast<std::size_t>(value);
}

/// Return total serialized size after checking every addition/multiplication.
inline std::size_t CheckedPayloadSize(std::size_t states, std::size_t restart)
{
   if (states > (std::numeric_limits<std::size_t>::max() -
                 ODECheckpointSerializer::HeaderSize) / sizeof(double))
   {
      throw InvalidCheckpointState("checkpoint state byte count overflows");
   }
   const std::size_t state_bytes = states * sizeof(double);
   if (restart > std::numeric_limits<std::size_t>::max() -
       ODECheckpointSerializer::HeaderSize - state_bytes)
   {
      throw InvalidCheckpointState("checkpoint restart byte count overflows");
   }
   return ODECheckpointSerializer::HeaderSize + state_bytes + restart;
}

inline Snapshot ODECheckpointSerializer::Encode(
   CheckpointId id, const ODECheckpointData &checkpoint)
{
   if (checkpoint.state.Size() < 0)
   {
      throw InvalidCheckpointState("checkpoint Vector has a negative size");
   }
   if (checkpoint.time.step < 0)
   {
      throw InvalidCheckpointState("checkpoint step must be non-negative");
   }

   const std::size_t state_size =
      static_cast<std::size_t>(checkpoint.state.Size());
   const std::size_t total_size =
      CheckedPayloadSize(state_size, checkpoint.restart.Size());
   std::vector<unsigned char> encoded;
   encoded.reserve(total_size);
   AppendLittleEndian(encoded, checkpoint_magic);
   AppendLittleEndian(encoded, FormatVersion);
   encoded.push_back(little_endian_encoding);
   encoded.push_back(binary64_encoding);
   AppendLittleEndian(encoded, std::uint16_t(0));
   AppendLittleEndian(encoded, id);

   std::uint64_t step_bits = 0;
   static_assert(sizeof(step_bits) == sizeof(checkpoint.time.step),
                 "StateId size mismatch");
   std::memcpy(&step_bits, &checkpoint.time.step, sizeof(step_bits));
   AppendLittleEndian(encoded, step_bits);
   AppendLittleEndian(encoded,
                      DoubleBits(static_cast<double>(checkpoint.time.time)));
   AppendLittleEndian(encoded, DoubleBits(static_cast<double>(checkpoint.dt)));
   AppendLittleEndian(encoded, static_cast<std::uint64_t>(state_size));
   AppendLittleEndian(encoded,
                      static_cast<std::uint64_t>(checkpoint.restart.Size()));

   const real_t *values = checkpoint.state.HostRead();
   for (std::size_t i = 0; i < state_size; i++)
   {
      AppendLittleEndian(encoded, DoubleBits(static_cast<double>(values[i])));
   }
   if (checkpoint.restart.Size() != 0)
   {
      encoded.insert(encoded.end(), checkpoint.restart.Data(),
                     checkpoint.restart.Data() + checkpoint.restart.Size());
   }

   if (encoded.size() != total_size)
   {
      throw CheckpointConsistencyError(
         "serialized checkpoint size disagrees with its header");
   }
   Snapshot result(encoded.size());
   if (!encoded.empty())
   {
      std::memcpy(result.Data(), encoded.data(), encoded.size());
   }
   return result;
}

inline ODECheckpointData ODECheckpointSerializer::Decode(
   CheckpointId expected_id, const Snapshot &snapshot)
{
   if (snapshot.Size() < HeaderSize)
   {
      throw InvalidCheckpointFormat("checkpoint payload is truncated");
   }
   std::size_t offset = 0;
   if (ReadLittleEndian<std::uint64_t>(snapshot, offset) != checkpoint_magic)
   {
      throw InvalidCheckpointFormat("invalid checkpoint magic");
   }
   if (ReadLittleEndian<std::uint32_t>(snapshot, offset) != FormatVersion)
   {
      throw InvalidCheckpointFormat("unsupported checkpoint format version");
   }
   if (ReadLittleEndian<unsigned char>(snapshot, offset) !=
       little_endian_encoding)
   {
      throw InvalidCheckpointFormat("unsupported checkpoint byte order");
   }
   if (ReadLittleEndian<unsigned char>(snapshot, offset) != binary64_encoding)
   {
      throw InvalidCheckpointFormat("unsupported checkpoint scalar encoding");
   }
   if (ReadLittleEndian<std::uint16_t>(snapshot, offset) != 0)
   {
      throw InvalidCheckpointFormat("checkpoint reserved header bits are set");
   }
   if (ReadLittleEndian<std::uint64_t>(snapshot, offset) != expected_id)
   {
      throw InvalidCheckpointFormat(
         "checkpoint logical ID disagrees with requested ID");
   }

   const std::uint64_t step_bits =
      ReadLittleEndian<std::uint64_t>(snapshot, offset);
   StateId step = 0;
   std::memcpy(&step, &step_bits, sizeof(step));
   if (step < 0)
   {
      throw InvalidCheckpointFormat("checkpoint step is negative");
   }
   const double time = BitsDouble(
                          ReadLittleEndian<std::uint64_t>(snapshot, offset));
   const double dt = BitsDouble(
                        ReadLittleEndian<std::uint64_t>(snapshot, offset));
   const std::uint64_t encoded_state_size =
      ReadLittleEndian<std::uint64_t>(snapshot, offset);
   const std::uint64_t encoded_restart_size =
      ReadLittleEndian<std::uint64_t>(snapshot, offset);
   const std::size_t state_size =
      CheckedSize(encoded_state_size, "checkpoint Vector length");
   const std::size_t restart_size =
      CheckedSize(encoded_restart_size, "checkpoint restart length");
   if (state_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
   {
      throw InvalidCheckpointFormat("checkpoint Vector length exceeds INT_MAX");
   }
   std::size_t expected_size = 0;
   try
   {
      expected_size = CheckedPayloadSize(state_size, restart_size);
   }
   catch (const InvalidCheckpointState &error)
   {
      throw InvalidCheckpointFormat(error.what());
   }
   if (snapshot.Size() != expected_size)
   {
      throw InvalidCheckpointFormat(
         "checkpoint byte count disagrees with its header");
   }

   ODECheckpointData checkpoint;
   checkpoint.time.step = step;
   checkpoint.time.time = static_cast<real_t>(time);
   checkpoint.dt = static_cast<real_t>(dt);
   checkpoint.state.SetSize(static_cast<int>(state_size));
   real_t *values = checkpoint.state.HostWrite();
   for (std::size_t i = 0; i < state_size; i++)
   {
      const auto bits = ReadLittleEndian<std::uint64_t>(snapshot, offset);
      values[i] = static_cast<real_t>(BitsDouble(bits));
   }
   checkpoint.restart.SetSize(restart_size);
   if (restart_size != 0)
   {
      std::memcpy(checkpoint.restart.Data(), snapshot.Data() + offset,
                  restart_size);
   }
   return checkpoint;
}

} // namespace forward_euler_detail

/// Exact state adapter for MFEM's fixed-step ForwardEulerSolver.
class ForwardEulerCheckpointAdapter : public ODECheckpointStateAdapter
{
private:
   ForwardEulerSolver &solver;   ///< Borrowed solver to reinitialize.
   TimeDependentOperator &oper;  ///< Borrowed time-dependent operator.

public:
   /// Borrow ODE state, solver, and operator for the adapter lifetime.
   ForwardEulerCheckpointAdapter(ForwardEulerSolver &solver_,
                                 TimeDependentOperator &oper_, Vector &state_,
                                 TimePoint &time_, real_t &dt_)
      : ODECheckpointStateAdapter(state_, time_, dt_), solver(solver_),
        oper(oper_) { }

   /// @copydoc CheckpointStateAdapter::Capture()
   Snapshot Capture(
      StateId state_id,
      std::optional<CheckpointId> checkpoint = std::nullopt) const override;

   /// @copydoc CheckpointStateAdapter::Restore()
   void Restore(
      StateId state_id, const Snapshot &snapshot,
      std::optional<CheckpointId> checkpoint = std::nullopt) override;
};

inline Snapshot ForwardEulerCheckpointAdapter::Capture(
   StateId state_id, std::optional<CheckpointId> checkpoint) const
{
   if (state_id < 0 || time.step != state_id || !(dt > 0.0) ||
       !std::isfinite(time.time))
   {
      throw InvalidCheckpointState("Forward Euler checkpoint requires a "
                                   "matching non-negative state ID and "
                                   "positive dt and finite time");
   }
   const forward_euler_detail::ODECheckpointData data
   {state, time, dt, Snapshot()};
   return forward_euler_detail::ODECheckpointSerializer::Encode(
             checkpoint.value_or(0), data);
}

inline void ForwardEulerCheckpointAdapter::Restore(
   StateId state_id, const Snapshot &snapshot,
   std::optional<CheckpointId> checkpoint)
{
   auto restored = forward_euler_detail::ODECheckpointSerializer::Decode(
                      checkpoint.value_or(0), snapshot);
   if (restored.time.step != state_id)
   {
      throw InvalidCheckpointFormat(
         "checkpoint state ID disagrees with requested state");
   }
   if (!(restored.dt > 0.0) || !std::isfinite(restored.time.time) ||
       restored.restart.Size() != 0)
   {
      throw InvalidCheckpointState(
         "Forward Euler restart requires positive dt, finite time, "
         "and no solver payload");
   }
   state = restored.state;
   time = restored.time;
   dt = restored.dt;
   oper.SetTime(time.time);
   solver.Init(oper);
}

} // namespace checkpoint_demo
} // namespace mfem

#endif // MFEM_CHECKPOINT_DEMO_FORWARD_EULER
