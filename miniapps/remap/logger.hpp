#pragma once

#include <iomanip>
#include <iostream>
#include <vector>
#include "mfem.hpp"


namespace mfem
{

class TableLogger
{
public:
   enum dtype { DOUBLE, INT };

protected:
   // Double data to be printed.
   std::vector<double *> data_double;
   // Int data to be printed
   std::vector<int *> data_int;
   // Data type for each column
   std::vector<dtype> data_order;
   // Name of each monitored data
   std::vector<std::string> names;
   // Output stream
   std::ostream &os;
   // Column width
   int w;
   // Whether the variable name row has been printed or not
   bool var_name_printed;
   bool isRoot; // true if serial or root in parallel
   std::unique_ptr<std::fstream> file;

private:
public:
   // Create a logger that prints a row of variables for each call of Print
   TableLogger(std::ostream &os = std::cout);
   // Set column width of the table to be printed
   void setw(const int column_width) { w = column_width; }
   // Add double data to be monitored
   void Append(const std::string name, double &val);
   // Add double data to be monitored
   void Append(const std::string name, int &val);
   // Print a row of currently monitored data. If it is called
   void Print(bool print_valname=false);
   // Save data to a file whenever Print is called.
   void SaveWhenPrint(std::string filename,
                      std::ios::openmode mode = std::ios::out);
   // Close file manually.
   void CloseFile() { if (file) { file.reset(nullptr); } }
};

class GLVis
{
   std::vector<std::unique_ptr<socketstream>> sockets;
   // Array<mfem::socketstream *> sockets;
   Array<mfem::GridFunction *> gfs;
   Array<mfem::QuadratureFunction *> qfs;
   Array<Mesh *> meshes;
   Array<bool> parallel;
   const char *hostname;
   const int port;
   int w, h, nrWinPerRow;
   bool secure;

public:
#ifdef MFEM_USE_GNUTLS
   static const bool secure_default = true;
#else
   static const bool secure_default = false;
#endif
   GLVis(const char hostname[], int port, int w=400, int h=350,
         int nrWinPerRow=1,
         bool secure = secure_default)
      : sockets(0), gfs(0), meshes(0), parallel(0), hostname(hostname),
        port(port), w(w), h(h), nrWinPerRow(nrWinPerRow),
        secure(secure_default) {}

   void Append(GridFunction &gf, const char window_title[] = nullptr,
               const char keys[] = nullptr);
   void Append(QuadratureFunction &qf, const char window_title[] = nullptr,
               const char keys[] = nullptr);
   void Update();

   GridFunction& GetGridFunction(int i)
   {
      MFEM_VERIFY(i < gfs.Size(), "Index out of range");
      return *gfs[i];
   }

   socketstream &GetSocket(int i)
   {
      MFEM_VERIFY(i < sockets.size(), "Index out of range");
      return *sockets[i];
   }
};

} // namespace mfem
