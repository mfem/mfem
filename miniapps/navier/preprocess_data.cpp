// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// 3D flow over a cylinder benchmark example

//#include "navier_solver.hpp"
#include "navier_3d_brink_workflow.hpp"
#include "ascii.hpp"
#include <fstream>
#include <ctime>
#include <cstdlib> 
#include <vector> 

using namespace mfem;
using namespace navier;





int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   const char *mesh_file = "bar3d.msh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");

   args.Parse();

   {
      std::string tStringOut = "./A_OutputFile_";

      int tNumFiles = 1000;

      std::vector< std::string > tVec(tNumFiles, "");

      int tCounter = 0;

      for( int Ik = 0; Ik < tNumFiles; Ik ++)
      {
         std::string tString = "./OutputFile_" + std::to_string(Ik);

         if( !file_exists(tString) )
         {
             std::cout<<"file does not exist"<<std::endl;
         }
         else
         {

            Ascii tAsciiWriter( tString, FileMode::OPEN_RDONLY );

            int tNumLines = tAsciiWriter.length();
     
            for( int Ii = 0; Ii < tNumLines; Ii ++)
            {
               std::string tLine = tAsciiWriter.line( Ii );

              tVec[tCounter] = tVec[tCounter] + " " + tLine;
            }

            tCounter++;
         }

      }

      Ascii tAsciiWriter( tStringOut, FileMode::NEW );

      for( int Ik = 0; Ik < tNumFiles; Ik ++)
      {
          tAsciiWriter.print(tVec[Ik]);

          std::cout<<tVec[Ik]<<std::endl;
      }
         
      tAsciiWriter.save();
   }
  


   return 0;
}
