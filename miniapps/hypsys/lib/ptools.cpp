#include "ptools.hpp"

void ParVisualizeField(socketstream &sock, const char *vishost, int visport,
                       string ProblemName, ParGridFunction &gf, bool vec)
{
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int myid;
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;

   if (myid == 0)
   {
      if (!sock.is_open() && sock)
      {
         sock.open(vishost, visport);
         sock.precision(8);
         newly_opened = true;
      }
      sock << "solution\n";
   }

   pmesh.PrintAsOne(sock);
   gf.SaveAsOne(sock);

   if (myid == 0 && newly_opened)
   {
      sock << "window_title '" << ProblemName << "'\n"
           << "window_geometry "
           << 0 << " " << 0 << " " << 1080 << " " << 1080 << "\n"
           << "keys mcjlppppppppppppppppppppppppppp66666666666666666666666"
           << "66666666666666666666666666666666666666666666666662222222222";
      if ( vec ) { sock << "vvv"; }
      sock << endl;
   }
}
