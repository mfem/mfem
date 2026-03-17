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
#pragma once

#include <memory>
#include <thread>

#include "glvis_data.hpp"


// wait/signal for READY
inline void wait_for_ready(const std::shared_ptr<GLVisData>& data)
{
   std::unique_lock<std::mutex> lock(data->mutex);
   data->cond.wait(lock, [&data] { return data->ready.load(); });
}

inline void signal_for_ready(const std::shared_ptr<GLVisData>& data)
{
   std::lock_guard<std::mutex> lock(data->mutex);
   data->ready.store(true);  // Set atomic flag
   data->cond.notify_one();  // Notify the waiter
}

// wait/signal for RUNNING
inline void wait_for_running(const std::shared_ptr<GLVisData>& data)
{
   std::unique_lock<std::mutex> lock(data->mutex);
   data->cond.wait(lock, [&data] { return data->running.load(); });
}

inline void signal_for_running(const std::shared_ptr<GLVisData>& data)
{
   std::lock_guard<std::mutex> lock(data->mutex);
   data->running.store(true);
   data->cond.notify_one();
}

// wait/signal for UPDATE
inline void wait_for_update(const std::shared_ptr<GLVisData>& data)
{
   std::unique_lock<std::mutex> lock(data->mutex);
   data->cond.wait(lock, [&data] { return data->update.load(); });
   data->update.store(false); // Reset for next
}

inline void signal_for_update(const std::shared_ptr<GLVisData>& data)
{
   std::lock_guard<std::mutex> lock(data->mutex);
   data->update.store(true);
   data->cond.notify_one();
}

///////////////////////////////////////////////////////////////////////////////
class GLVisServer
{
   std::shared_ptr<GLVisData> data;
   std::unique_ptr<std::thread> glvis_thread {nullptr};

public:
   GLVisServer(std::shared_ptr<GLVisData>);

   ~GLVisServer();

   int Wait();

   int Stop();
};
