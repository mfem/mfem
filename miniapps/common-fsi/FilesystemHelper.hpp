#ifndef FILESYSTEM_HELPER_HPP
#define FILESYSTEM_HELPER_HPP

#if __cplusplus >= 201703L
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __cplusplus >= 201402L
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#else
    #error "C++14 or later is required for filesystem support."
#endif

#endif // FILESYSTEM_HELPER_HPP
