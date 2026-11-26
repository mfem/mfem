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

#ifndef MFEM_HP_PARALLEL_CONSTRAINTS_HPP
#define MFEM_HP_PARALLEL_CONSTRAINTS_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../general/communication.hpp"
#include "../linalg/hypre_parcsr.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace mfem
{
namespace internal
{

using HpRootRow = std::map<HYPRE_BigInt, real_t>;

struct HpRootConstraint
{
   HpRootRow terms;
   // Sum of absolute contributions before cancellation in the expansion C*P0.
   real_t scale = 0.0;
};

namespace hp_root
{

using Id = HYPRE_BigInt;

inline int Owner(const std::vector<Id> &offsets, Id id)
{
   MFEM_ASSERT(id >= offsets.front() && id < offsets.back(),
               "Hp root outside its global partition");
   return int(std::upper_bound(offsets.begin(), offsets.end(), id)
              - offsets.begin() - 1);
}

inline std::vector<int> Displacements(const std::vector<int> &counts)
{
   std::vector<int> displs(1, 0);
   for (int count : counts)
   {
      displs.push_back(displs.back() + count);
   }
   return displs;
}

class ExchangePlan
{
   MPI_Comm comm;
   std::vector<int> send, recv, sdisp, rdisp;

public:
   ExchangePlan(MPI_Comm comm_, const std::vector<int> &counts)
      : comm(comm_), send(counts), recv(counts.size()),
        sdisp(Displacements(send))
   {
      MPI_Alltoall(send.data(), 1, MPI_INT, recv.data(), 1, MPI_INT, comm);
      rdisp = Displacements(recv);
   }

   template <typename T>
   std::vector<T> Exchange(const std::vector<T> &values,
                           bool reverse = false) const
   {
      const auto &from = reverse ? rdisp : sdisp;
      const auto &to = reverse ? sdisp : rdisp;
      MFEM_ASSERT(values.size() == size_t(from.back()), "");
      std::vector<T> result(to.back());
      auto type = MPITypeMap<T>::mpi_type;
      MPI_Alltoallv(values.data(), (reverse ? recv : send).data(), from.data(),
                    type, result.data(), (reverse ? send : recv).data(),
                    to.data(), type, comm);
      return result;
   }
};

inline std::vector<int> RootCounts(const std::vector<Id> &offsets,
                                   const std::vector<Id> &ids)
{
   std::vector<int> counts(offsets.size() - 1, 0);
   for (Id id : ids) { ++counts[Owner(offsets, id)]; }
   return counts;
}

// Route variable-length sparse rows. Root IDs are nonnegative; a negative ID
// starts a row and carries the component or pivot key, paired with its scale.
class RowMessages
{
   std::vector<std::vector<Id>> ids;
   std::vector<std::vector<real_t>> values;

   template <typename T>
   static std::vector<T> Flatten(std::vector<std::vector<T>> &parts)
   {
      size_t size = 0;
      for (const auto &part : parts) { size += part.size(); }
      std::vector<T> flat;
      flat.reserve(size);
      for (const auto &part : parts)
      {
         flat.insert(flat.end(), part.begin(), part.end());
      }
      parts.clear();
      return flat;
   }

public:
   explicit RowMessages(int nranks) : ids(nranks), values(nranks) {}

   void Add(int dest, Id key, const HpRootRow &terms, real_t scale)
   {
      ids[dest].push_back(-1 - key);
      values[dest].push_back(scale);
      for (const auto &term : terms)
      {
         if (term.second == 0.0) { continue; }
         ids[dest].push_back(term.first);
         values[dest].push_back(term.second);
      }
   }

   template <typename Consume>
   void Deliver(MPI_Comm comm, Consume consume)
   {
      std::vector<int> counts;
      for (const auto &part : ids) { counts.push_back(int(part.size())); }
      ExchangePlan plan(comm, counts);
      const auto keys = plan.Exchange(Flatten(ids));
      const auto data = plan.Exchange(Flatten(values));
      for (size_t i = 0; i < keys.size(); )
      {
         MFEM_ASSERT(keys[i] < 0, "Missing hp constraint row header");
         const Id key = -(keys[i] + 1);
         HpRootConstraint row;
         row.scale = data[i++];
         while (i < keys.size() && keys[i] >= 0)
         {
            row.terms.emplace(keys[i], data[i]);
            ++i;
         }
         consume(key, std::move(row));
      }
   }
};

// Same incremental strongest-pivot rule as CompleteHpDependencies, followed by
// back-substitution so callers can build Q without chasing remote pivots.
inline std::map<Id, HpRootRow> Solve(std::vector<HpRootConstraint> &rows)
{
   const real_t tol = 128 * std::numeric_limits<real_t>::epsilon();
   std::map<Id, size_t> pivot_number;
   std::vector<Id> pivots;
   std::vector<HpRootRow> echelon;
   for (auto &constraint : rows)
   {
      HpRootRow &row = constraint.terms;
      real_t scale = constraint.scale;
      std::set<size_t> active;
      for (const auto &term : row)
      {
         auto pos = pivot_number.find(term.first);
         if (pos != pivot_number.end()) { active.insert(pos->second); }
      }
      while (!active.empty())
      {
         const size_t i = *active.begin();
         active.erase(active.begin());
         const real_t factor = row.at(pivots[i]);
         row.erase(pivots[i]);
         for (const auto &entry : echelon[i])
         {
            const real_t value = factor * entry.second;
            row[entry.first] -= value;
            scale += std::abs(value);
            auto pos = pivot_number.find(entry.first);
            if (pos != pivot_number.end()) { active.insert(pos->second); }
         }
      }
      auto pivot = row.end();
      real_t maximum = 0.0;
      for (auto it = row.begin(); it != row.end(); ++it)
      {
         if (std::abs(it->second) > maximum)
         {
            maximum = std::abs(it->second);
            pivot = it;
         }
      }
      if (maximum <= tol * scale) { continue; }
      const Id id = pivot->first;
      const real_t diagonal = pivot->second;
      row.erase(pivot);
      for (auto it = row.begin(); it != row.end(); )
      {
         if (std::abs(it->second) <= tol * scale) { it = row.erase(it); }
         else { it->second /= diagonal; ++it; }
      }
      pivot_number.emplace(id, pivots.size());
      pivots.push_back(id);
      echelon.push_back(std::move(row));
   }

   std::map<Id, HpRootRow> solved;
   for (size_t i = pivots.size(); i > 0; --i)
   {
      HpRootRow &rhs = solved[pivots[i-1]];
      for (const auto &entry : echelon[i-1])
      {
         auto dep = solved.find(entry.first);
         if (dep == solved.end()) { rhs[entry.first] -= entry.second; }
         else
         {
            for (const auto &free : dep->second)
            {
               rhs[free.first] -= entry.second * free.second;
            }
         }
      }
      for (auto it = rhs.begin(); it != rhs.end(); )
      {
         if (it->second == 0.0) { it = rhs.erase(it); }
         else { ++it; }
      }
   }
   return solved;
}

} // namespace hp_root

// Returns owned eliminated roots expressed in the original global free-root
// IDs. A present key with an empty row denotes a root constrained to zero.
inline std::map<HYPRE_BigInt, HpRootRow> CompleteParallelHpRoots(
   MPI_Comm comm, const std::vector<HYPRE_BigInt> &offsets,
   const std::vector<HpRootConstraint> &rows)
{
   using namespace hp_root;
   int nranks;
   MPI_Comm_size(comm, &nranks);

   std::map<Id, Id> parent;
   auto representative = [&](Id id)
   {
      while (parent.at(id) != id)
      {
         parent[id] = parent.at(parent.at(id));
         id = parent.at(id);
      }
      return id;
   };
   std::vector<Id> anchors;
   for (const auto &row : rows)
   {
      Id anchor = -1;
      for (const auto &term : row.terms)
      {
         if (term.second == 0.0) { continue; }
         const Id id = term.first;
         parent.emplace(id, id);
         const Id root = representative(id);
         if (anchor < 0) { anchor = root; }
         else
         {
            parent[std::max(anchor, root)] = std::min(anchor, root);
            anchor = std::min(anchor, root);
         }
      }
      anchors.push_back(anchor);
   }

   std::map<Id, HpRootRow> pivots;
   int any = !parent.empty();
   MPI_Allreduce(MPI_IN_PLACE, &any, 1, MPI_INT, MPI_MAX, comm);
   if (!any) { return pivots; }

   std::vector<Id> ids;
   std::map<Id, Id> labels;
   for (auto &entry : parent)
   {
      entry.second = representative(entry.first);
      ids.push_back(entry.first);
      labels.emplace(entry.second, entry.second);
   }
   ExchangePlan plan(comm, RootCounts(offsets, ids));
   const auto owner_ids = plan.Exchange(ids);
   int changed;
   do
   {
      std::vector<Id> send;
      send.reserve(ids.size());
      for (const auto &entry : parent) { send.push_back(labels.at(entry.second)); }
      auto received = plan.Exchange(send);
      std::map<Id, Id> minima;
      for (size_t i = 0; i < received.size(); ++i)
      {
         auto it = minima.emplace(owner_ids[i], received[i]).first;
         it->second = std::min(it->second, received[i]);
      }
      for (size_t i = 0; i < received.size(); ++i)
      {
         received[i] = minima.at(owner_ids[i]);
      }
      const auto replies = plan.Exchange(received, true);
      changed = 0;
      for (size_t i = 0; i < ids.size(); ++i)
      {
         Id &label = labels.at(parent.at(ids[i]));
         if (replies[i] < label) { label = replies[i]; changed = 1; }
      }
      MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, comm);
   }
   while (changed);

   RowMessages equations(nranks);
   for (size_t i = 0; i < rows.size(); ++i)
   {
      if (anchors[i] < 0) { continue; }
      const Id component = labels.at(parent.at(anchors[i]));
      equations.Add(Owner(offsets, component), component,
                    rows[i].terms, rows[i].scale);
   }
   std::map<Id, std::vector<HpRootConstraint>> components;
   equations.Deliver(comm, [&](Id id, HpRootConstraint row)
   {
      components[id].push_back(std::move(row));
   });

   RowMessages solutions(nranks);
   for (auto &component : components)
   {
      for (const auto &pivot : Solve(component.second))
      {
         solutions.Add(Owner(offsets, pivot.first), pivot.first, pivot.second, 0.0);
      }
   }
   solutions.Deliver(comm, [&](Id id, HpRootConstraint row)
   {
      pivots.emplace(id, std::move(row.terms));
   });
   return pivots;
}

inline std::map<HYPRE_BigInt, HYPRE_BigInt> ResolveParallelHpFreeRoots(
   MPI_Comm comm, const std::vector<HYPRE_BigInt> &old_offsets,
   const std::vector<HYPRE_BigInt> &local_free_ids,
   const std::set<HYPRE_BigInt> &queries)
{
   using namespace hp_root;
   int rank;
   MPI_Comm_rank(comm, &rank);
   const std::vector<Id> ids(queries.begin(), queries.end());
   ExchangePlan plan(comm, RootCounts(old_offsets, ids));
   auto requests = plan.Exchange(ids);
   for (auto &id : requests)
   {
      id = local_free_ids[id - old_offsets[rank]];
      MFEM_ASSERT(id >= 0, "Requested hp root is not free");
   }
   const auto replies = plan.Exchange(requests, true);
   std::map<Id, Id> result;
   for (size_t i = 0; i < ids.size(); ++i) { result.emplace(ids[i], replies[i]); }
   return result;
}

} // namespace internal
} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_HP_PARALLEL_CONSTRAINTS_HPP
