/**
This library helps in finding the sources of memory validity errors especially
with the "debug" device backend. It works by preloading a modified mprotect
which tracks every address that mprotect is called, and stores the stack
trace of that call. When a SIGSEGV with code

This should be built as a shared library with position independent code.

To use, build this as a shared library "mprotect_trace.so" and set LD_PRELOAD=mprotect_trace.so

Requires libbacktrace which can be obtained from
https://github.com/ianlancetaylor/libbacktrace
or
https://github.com/gcc-mirror/gcc/tree/master/libbacktrace
or from a package manager like spack.
 */

#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <map>
#include <mutex>
#include <iostream>
#include <backtrace.h>
#include <backtrace-supported.h>
#include <cstring>
#include <vector>
#include <cxxabi.h>
#include <sstream>
#include <chrono>

#if BACKTRACE_SUPPORTED != 1
#error "Backtrace not supported! See output file backtrace-supported.h for details."
#endif

/// Try to undo the name mangling. Return mangled name if unsuccesful.
std::string demangle(const char* mangled)
{
   int status = 0;
   char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
   std::string result = (status == 0 && demangled) ? demangled : mangled;
   free(demangled);
   return result;
}

/// Fallback filename to library if source wasn't available
std::string fallback_file_name(uintptr_t pc)
{
   Dl_info info;
   if (dladdr((void*)pc, &info))
   {
      // use ostringstream because it's easier to convert to from hex.
      return std::string (info.dli_fname ? info.dli_fname : "??");
   }
   else
   {
      return "??";
   }
}

/// Fallback function name from dladdr when nicer name from backtrace fails.
std::string fallback_func_info(uintptr_t pc)
{
   Dl_info info;
   std::ostringstream ss;
   if (dladdr((void*)pc, &info))
   {
      if (info.dli_sname)
      {
         ss << " in " << abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, nullptr);
      }
      else
      {
         ss << "pc " << (void*)pc;
      }
   }
   else
   {
      ss << "pc " << (void*)pc;
   }
   return ss.str();
}

/// Tracks information about mprotect.
struct ProtectionInfo
{
   /// Address that mprotect was called on.
   void* addr;
   /// The length of the protected buffer.
   size_t len;
   /// The protection code used in mprotect.
   int prot;
   /// The stack trace.
   std::vector<std::string> trace;
   /// The timespace of the mprotect call, used to differentiate calls to overlapping memory spans.
   std::chrono::steady_clock::time_point timestamp;
};

static std::mutex g_mutex;

/// the keys of this map are the addresses that mprotect was called on.s
static std::map<void*, ProtectionInfo> g_protected;

void error_callback(void*, const char* msg, int errnum)
{
   std::cerr << "libbacktrace error: " << msg << " (" << errnum << ")\n";
}

/// Get the string representation of the trace.
int mprotect_backtrace_full_callback(void* data, uintptr_t pc,
                                     const char* filename, int lineno, const char* function)
{
   char buf[512];
   std::string demangled_name;
   std::string fname_str;
   if (!function)
   {
      demangled_name = fallback_func_info(pc);
   }
   else
   {
      demangled_name = demangle( function);
   }
   if (!filename)
   {
      fname_str = fallback_file_name(pc);
   }
   else
   {
      fname_str = filename;
   }

   snprintf(buf, sizeof(buf), "%s:%d in %s",fname_str.c_str(), lineno,
            demangled_name.c_str());
   ((std::vector<std::string>*)data)->emplace_back(buf);
   return 0;
}

/// This gets called if libbacktrace ran into a problem.
void mprotect_backtrace_error_callback(void* data, const char* msg, int)
{
   ((std::vector<std::string>*)data)->emplace_back(std::string("???: ") + msg);
}

/// Call this to try to generate a trace.
void collect_trace(std::vector<std::string>& out)
{
   static backtrace_state* state = backtrace_create_state(nullptr, 1,
                                                          error_callback, nullptr);
   backtrace_full(state, 0,
                  mprotect_backtrace_full_callback,
                  mprotect_backtrace_error_callback,
                  &out);
}

/// Our custom mprotect
extern "C" int mprotect(void* addr, size_t len, int prot)
{
   static auto real_mprotect = (int(*)(void*, size_t, int))dlsym(RTLD_NEXT,
                                                                 "mprotect");

   std::vector<std::string> stack;
   collect_trace(stack);
   auto timepoint = std::chrono::steady_clock::now();

   {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_protected[addr] = { addr, len, prot, stack, timepoint};
   }

   return real_mprotect(addr, len, prot);
}

/// Return human readable protection code
/// There are some codes I didn't write a conversion for.
std::string prot_to_string(int prot)
{
   std::string prot_string;
   if (prot == PROT_NONE) { prot_string += " PROT_NONE "; }
   if (prot & PROT_READ) { prot_string += " PROT_READ "; }
   if (prot & PROT_WRITE) { prot_string += " PROT_WRITE "; }
   if (prot & PROT_EXEC) { prot_string += " PROT_EXEC "; }
   return prot_string;
}

void segv_handler(int /*sig*/, siginfo_t* info, void*)
{
   void* fault_addr = info->si_addr;
   auto code = info->si_code;
   std::cerr << "Caught SIGSEGV with code " << code <<  " at address " <<
             fault_addr << ".\n";
   if (code == SEGV_ACCERR)
   {
      std::cerr <<
                "This is a permission violation which was likely caused by accessing memory protected by mprotect.\n";
   }
   else
   {
      std::cerr <<
                "This is not a permission violation which means the mprotect trace is likely not useful.\n";
   }

   std::vector<std::string> access_trace;
   collect_trace(access_trace);

   std::cerr << "\nACCESS STACK TRACE (this caused the violation):\n";
   for (const auto& s : access_trace) { std::cerr << "  " << s << "\n"; }
   {
      std::lock_guard<std::mutex> lock(g_mutex);
      // loop through addresses that m_protect was called on and see
      // if the offending address fell in any of their ranges.
      // It's possible that this has multiple hits if data was freed
      // and another mprotect was called in a shifted location. This is dealt
      // with here using timestamps from a high resolution clock. The most
      // recent call is used.
      static std::map<std::chrono::steady_clock::time_point, ProtectionInfo>
      faulty_candidates;
      static std::map<std::chrono::steady_clock::time_point, bool>
      multiple_candidates;
      for (const auto& [base, pi] : g_protected)
      {
         if (fault_addr >= base && fault_addr < (char*)base + pi.len)
         {
            multiple_candidates[pi.timestamp] = faulty_candidates.count(pi.timestamp) > 0;
            faulty_candidates[pi.timestamp] = pi;
         }
      }
      if (!faulty_candidates.empty())
      {
         // reverse the map from newest to oldest time. Extract most recent time.
         const auto most_recent_time = faulty_candidates.rbegin()->first;
         const auto &pi = faulty_candidates[most_recent_time];
         const auto multiple_matches_found = multiple_candidates[most_recent_time];
         std::cerr <<
                   "\nPROTECTION STACK TRACE (this altered the access permissions most recently):\n";
         std::cerr << "mprotect set the following permissions: " << prot_to_string(
                      pi.prot) << "\n";
         std::cerr << "Full protection number: " << pi.prot << "\n";
         for (const auto& s : pi.trace) { std::cerr << "  " << s << "\n"; }
         if (multiple_matches_found)
         {
            std::cerr <<
                      "WARNING : Multiple protection stack traces found. Only showing one, but it may not be right. Maybe this tool needs a finer resolution clock.\n";
         }
      }
      else
      {
         std::cerr << "No offending mprotect stack trace could be found\n";
      }
   }

   exit(1);
}

// This code gets run before main.
__attribute__((constructor))
void install_handler()
{
   struct sigaction sa {};
   sa.sa_flags = SA_SIGINFO;
   sa.sa_sigaction = segv_handler;
   sigaction(SIGSEGV, &sa, nullptr);
}
