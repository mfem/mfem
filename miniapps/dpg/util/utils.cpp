//
#include "utils.hpp"

std::string GetTimestamp()
{
   std::time_t now = std::time(nullptr);
   std::tm* local_time = std::localtime(&now);
   std::ostringstream oss;
   oss << (local_time->tm_year + 1900) << "-"
       << (local_time->tm_mon + 1) << "-"
       << local_time->tm_mday << "_"
       << local_time->tm_hour << "-"
       << local_time->tm_min << "-"
       << local_time->tm_sec;
   return oss.str();
}

// Write parsed options to a file in the ParaView directory
void WriteParametersToFile(const mfem::OptionsParser& args,
                           const std::string& output_dir)
{
   // Ensure directory exists
   std::string mkdir_command = "mkdir -p " + output_dir;
   int ret = system(mkdir_command.c_str());
   if (ret != 0)
   {
      std::cerr << "Warning: Failed to create ParaView output directory.\n";
   }

   std::string filename = output_dir + "/run_parameters.txt";

   std::ofstream param_file(filename);
   if (param_file.is_open())
   {
      param_file << "Simulation Parameters \n";
      param_file << "------------------------------------\n";

      // Use OptionsParser's Print method to output parameters to the file
      args.PrintOptions(param_file);

      param_file.close();
      std::cout << "Parameters saved to " << filename << "\n";
   }
   else
   {
      std::cerr << "Error: Unable to open file to save parameters.\n";
   }
}

void CreateParaViewPath(const char* mesh_file, std::string& output_dir)
{
   std::string timestamp = GetTimestamp();
   std::string paraview_file = timestamp;

   output_dir = output_dir + paraview_file;


}