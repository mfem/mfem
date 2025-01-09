//
#include "mfem.hpp"
#include <ctime>
#include <string>
#include <sstream>

std::string GetTimestamp();
void WriteParametersToFile(const mfem::OptionsParser& args,
                           const std::string& output_dir);
void CreateParaViewPath(const char* mesh_file, std::string& output_dir);

std::string GetFilename(const std::string& filePath);
