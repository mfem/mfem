// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_HANDLE_HPP
#define MFEM_HANDLE_HPP

#include "../config/config.hpp"
#include <memory>

namespace mfem
{

/// @brief A smart pointer class that may represent either shared ownership, or
/// a non-owning borrow.
///
/// A Handle may either be owning or non-owning. Non-owning Handle%s point to
/// externally owned data; it is the responsibility of the user both to ensure
/// that the data remains valid as long as the Handle is alive and to delete the
/// pointer when it is no longer needed. Owning Handle%s use <a
/// href="https://en.cppreference.com/w/cpp/memory/shared_ptr">
/// std::shared_ptr</a> to implement reference counting. The underlying data
/// will be valid as long as there is at least one live copy. When the last
/// Handle is destroyed, the pointer is deleted.
///
/// Both types of Handle%s can be copied, moved, stored in standard containers,
/// etc.
///
/// A non-owning Handle may assume ownership over its data, but an owning Handle
/// cannot release ownership over its data.
///
/// It is an invariant of this class that **at most** one of the data members
/// @a not_owned and @a owned will be non-null.
template <typename T>
class Handle
{
   /// If this is a non-owning handle, @a not_owned will point to the data.
   T *not_owned = nullptr;

   /// If this is an owning handle, @a owned will point to the data.
   std::shared_ptr<T> owned = nullptr;

   /// @brief Types @a Handle<T> and @a %Handle\<U\> are friends to allow
   /// construction of one from another when @a T and @a U are convertible
   /// types.
   template <typename U> friend class Handle;

public:
   /// Create an empty (null) Handle.
   Handle() = default;

   /// @brief Create a Handle pointing to @a t.
   ///
   /// If @a take_ownership is true, then the Handle assumes ownership over the
   /// pointer, and it should not be deleted externally. Otherwise, the Handle
   /// will be non-owning, and it is the user's responsibility to ensure the
   /// correct lifetime of @a t.
   Handle(T *t, bool take_ownership)
   {
      if (take_ownership) { owned.reset(t); }
      else { not_owned = t; }
   }

   /// Create a Handle from a std::shared_ptr (sharing ownership with @a t).
   Handle(const std::shared_ptr<T> &t) : owned(t) { }

   /// @brief Copy constructor.
   ///
   /// Copying an owning Handle results in another owning handle. Copying a
   /// non-owning handle results in a non-owning handle.
   Handle(const Handle &other) = default;

   /// Move constructor (see Handle(const Handle&)).
   Handle(Handle &&other) = default;

   /// @brief Constructs a copy of @a u, where type @a U is convertible to @a T.
   ///
   /// This allows the construction of Handle<Base> from Handle<Derived>.
   template <typename U>
   Handle(const Handle<U> &u) : not_owned(u.not_owned), owned(u.owned) { }

   /// @brief Move-constructs from @a u, where type @a U is convertible to @a T.
   ///
   /// See @ref Handle(const Handle<U>&).
   template <typename U>
   Handle(Handle<U> &&u) : not_owned(u.not_owned), owned(u.owned) { }

   /// Destructor. If the Handle is owning, decrement the reference count.
   ~Handle() = default;

   /// Copy assignment (see Handle(const Handle&)).
   Handle &operator=(const Handle &other) = default;

   /// Move assignment (see Handle(const Handle&)).
   Handle &operator=(Handle &&other) = default;

   /// Returns the contained pointer (may be null).
   T *Get() const
   {
      if (not_owned) { return not_owned; }
      else { return owned.get(); }
   }

   /// @brief If the Handle is owning, return a copy of the underlying shared
   /// pointer.
   ///
   /// @warning If the Handle is non-owning (even if non-null), this will return
   /// and empty (null) shared pointer.
   std::shared_ptr<T> GetSharedPtr() const { return owned; }

   /// Dereference operator. The Handle must be non-null.
   T &operator*() const { return *Get(); }

   /// Member access (arrow) operator. The Handle must be non-null.
   T *operator->() const { return Get(); }

   /// @brief Returns true if the Handle is owning, false if it is non-owning.
   ///
   /// Returns false if the Handle is null (empty).
   bool IsOwner() const { return owned; }

   /// Returns true if the Handle is non-null.
   explicit operator bool() const { return not_owned || owned; }

   /// @brief Assume owernship of the data.
   ///
   /// If the Handle is already owning, this does nothing.
   void MakeOwner()
   {
      if (owned) { return; }
      owned.reset(not_owned);
      not_owned = nullptr;
   }

   /// @brief Reset the Handle to be empty.
   ///
   /// If the Handle is owning, this will decrement the reference count.
   void Reset()
   {
      owned.reset();
      not_owned = nullptr;
   }

   /// @brief Reset the Handle to point to @a t.
   ///
   /// The Handle may assume ownership of the pointer according to @a
   /// take_ownership (see @ref Handle(T*, bool)).
   void Reset(T *t, bool take_ownership)
   {
      if (take_ownership)
      {
         owned.reset(t);
         not_owned = nullptr;
      }
      else
      {
         owned.reset();
         not_owned = t;
      }
   }

   /// Reset the Handle to share ownership with @a t.
   void Reset(const std::shared_ptr<T> &t)
   {
      owned = t;
      not_owned = nullptr;
   }
};

/// @brief Return a new owning Handle, where the pointed-to object is a new
/// object constructed using the given arguments.
///
/// This is analogous to <a
/// href="https://en.cppreference.com/w/cpp/memory/shared_ptr/make_shared">
/// std::make_shared</a>.
template <typename T, typename... Args>
Handle<T> MakeOwning(Args&&... args)
{
   T *t = new T(std::forward<Args>(args)...);
   return Handle<T>(t, true);
}

/// Return a new owning Handle pointing to @a t.
template <typename T>
Handle<T> Owning(T *t) { return Handle<T>(t, true); }

/// Return a new non-owning Handle pointing to @a t.
template <typename T>
Handle<T> NonOwning(T *t) { return Handle<T>(t, false); }

} // namespace mfem

#endif
