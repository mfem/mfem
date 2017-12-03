// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//            -----------------------------------------------------
//            Hertz Miniapp:  Simple Magnetostatics Simulation Code
//            -----------------------------------------------------
//

#include "hertz_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::electromagnetics;

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   if ( mpi.Root() ) { display_banner(cout); }

   return 0;
}

// Print the Volta ascii logo to the given ostream
void display_banner(ostream & os)
{
  /*
   os << "  _    _           _        " << endl
      << " | |  | |         | |       " << endl
      << " | |__| | ___ _ __| |_ ____ " << endl
      << " |  __  |/ _ \\ '__| __|_  / " << endl
      << " | |  | |  __/ |  | |_ / /  " << endl
      << " |_|  |_|\\___|_|   \\__/___| " << endl
      << "                            " << endl << flush;
  */
  /*
   os << "  ___ ___                 __           " << endl 
      << " /   |   \\   ____________/  |_________ " << endl
      << "/    ~    \\_/ __ \\_  __ \\   __\\___   / " << endl
      << "\\    Y    /\\  ___/|  | \\/|  |  /    /  " << endl
      << " \\___|_  /  \\___  >__|   |__| /_____ \\ " << endl
      << "       \\/       \\/                  \\/ " << endl << flush;
  */
   os << "     ____  ____              __           " << endl 
      << "    /   / /   / ____________/  |_________ " << endl
      << "   /   /_/   /_/ __ \\_  __ \\   __\\___   / " << endl
      << "  /   __    / \\  ___/|  | \\/|  |  /   _/  " << endl
      << " /___/ /_  /   \\___  >__|   |__| /_____ \\ " << endl
      << "         \\/        \\/                  \\/ " << endl << flush;
}
