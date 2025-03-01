#ifndef MFEM_ROCTX_HPP
#define MFEM_ROCTX_HPP

#include <string>

#include "globals.hpp"
#include "backends.hpp"

#if defined(MFEM_USE_HIP)
#include <rocprofiler-sdk-roctx/roctx.h>
#else  // MFEM_USE_HIP
inline int roctxRangePush(const char *) { return 0; }
inline int roctxRangePop(void) { return 0; }
#endif // MFEM_USE_HIP

namespace mfem
{

class RoctxConfig
{
   bool roctx_enabled;
   bool enforce_kernel_sync;
   RoctxConfig()
   {
      roctx_enabled = GetEnv("MFEM_ROCTX") != nullptr;
      enforce_kernel_sync = GetEnv("MFEM_EKS") != nullptr;
   }
   static RoctxConfig &Instance()
   {
      static RoctxConfig instance;
      return instance;
   }
public:
   static bool Enabled() { return Instance().roctx_enabled; }
   static bool EnforceKernelSync() {return Instance().enforce_kernel_sync; }
};

class Roctx
{
public:
   Roctx(const std::string &message)
   {
      if (!RoctxConfig::Enabled()) { return; }
      roctxRangePush(message.c_str());
      mfem::out << "ROCTX: " << message << '\n';
   }

   Roctx(const std::string &filename,
         const int line_number,
         const std::string &function,
         const std::string &message = "")
   {
      if (!RoctxConfig::Enabled()) { return; }

      size_t start_pos = 0;
      const size_t i1 = filename.rfind('/');
      if (i1 != std::string::npos)
      {
         const size_t i2 = filename.substr(0, i1).rfind('/');
         if (i2 != std::string::npos) { start_pos = i2 + 1; }
      }
      std::string full_message = filename.substr(start_pos) + ":"
                                 + std::to_string(line_number) + ":["
                                 + function + "]";
      if (!message.empty()) { full_message += " " + message; }
      roctxRangePush(message.c_str());
      mfem::out << message << '\n';
   }

   ~Roctx()
   {
      if (!RoctxConfig::Enabled()) { return; }
      if (RoctxConfig::EnforceKernelSync())
      {
         roctxRangePush("Sync");
         MFEM_STREAM_SYNC;
         roctxRangePop();
      }
      roctxRangePop();
      mfem::out << "pop\n";
   }
};

// Helpers for generating unique variable names
#define ROCTX_PRIVATE_NAME(name) ROCTX_PRIVATE_CONCAT(name, __LINE__)
#define ROCTX_PRIVATE_CONCAT(a, b) ROCTX_PRIVATE_CONCAT2(a, b)
#define ROCTX_PRIVATE_CONCAT2(a, b) a##b

#define ROCTX(msg) \
   mfem::Roctx ROCTX_PRIVATE_NAME(roctx)(__FILE__, __LINE__, __FUNCTION__, msg);

} // namespace mfem

#endif // MFEM_NVVP_HPP
