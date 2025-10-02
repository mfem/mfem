#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "mtop_solvers.hpp"
#include "ascii.hpp"

using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   mfem::Mpi::Init(argc, argv);
   int myrank = mfem::Mpi::WorldRank();
   mfem::Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "./mini_flow2d_ball.msh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   real_t newton_rel_tol = 1e-7;
   real_t newton_abs_tol = 1e-12;
   int newton_iter = 10;
   int print_level = 1;
   bool visualization = false;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&newton_rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter,
                  "-it",
                  "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   // mfem::Mesh mesh(mesh_file, 1, 1);

   double meshOffsetX = -12.0;
   double meshOffsetY = -12.0;
   double meshOffsetZ = -0.5;


   double Lx = 24.0; double Ly = 24.0; double Lz = 150.5;
   int NX = 48;      int NY = 48;      int NZ = 301;
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(NX, NY, NZ, mfem::Element::HEXAHEDRON, Lx, Ly, Lz, true);
   int dim = mesh.Dimension();

   int tNumVertices  = mesh.GetNV();
   for (int i = 0; i < tNumVertices; ++i)
   {
      double * Coords = mesh.GetVertex(i);

      Coords[ 0 ] = Coords[ 0 ] + meshOffsetX;
      Coords[ 1 ] = Coords[ 1 ] + meshOffsetY;
      Coords[ 2 ] = Coords[ 2 ] + meshOffsetZ;
   }



   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   std::cout<<"My rank="<<pmesh.GetMyRank()<<std::endl;

   std::string tStringWeight = "./singleparticlesdiameter.txt";
   mfem::Ascii tAsciiReader( tStringWeight, FileMode::OPEN_RDONLY );

   int tNumLines = tAsciiReader.length();

   std::vector<int> particalType;  particalType.reserve(9*tNumLines);
   std::vector<real_t> xPos;       xPos.reserve(9*tNumLines);
   std::vector<real_t> yPos;       yPos.reserve(9*tNumLines);
   std::vector<real_t> zPos;       zPos.reserve(9*tNumLines);
   std::vector<real_t> rad;        rad.reserve(9*tNumLines);

   real_t maxRad = 0.0;

   for( int Ik = 0; Ik < tNumLines; Ik++ )
   {
      const std::string & tFileLine = tAsciiReader.line( Ik );

      std::vector<std::string> ListOfStrings = split_string( tFileLine, " " );

      particalType.push_back(std::stod( ListOfStrings[1] ));
      xPos        .push_back(std::stod( ListOfStrings[2] ));
      yPos        .push_back(std::stod( ListOfStrings[3] ));
      zPos        .push_back(std::stod( ListOfStrings[4] ));
      rad         .push_back(std::stod( ListOfStrings[5] ) / 2.0);

      maxRad = std::max( rad[Ik], maxRad );
   }

   maxRad += 1e-6;

   for( int Ik = 0; Ik < tNumLines; Ik++ )
   {
      int pType = particalType[Ik];
      real_t xCopy = xPos[Ik];
      real_t yCopy = yPos[Ik];
      real_t zCopy = zPos[Ik];
      real_t radCopy = rad[Ik];

      real_t xC;
      real_t yC;

      bool isCopyX = false;
      bool isCopyY = false;
      bool isCopyCorner = false;

      if(xPos[Ik]  < (meshOffsetX + maxRad)){ 
         xC = xCopy + Lx; 
         isCopyX = true; }
      else if(xPos[Ik]  > (meshOffsetX + Lx - maxRad)) { 
         xC = xCopy - Lx; 
         isCopyX = true; }

      if(isCopyX)
      {
         particalType.push_back(pType);
         xPos        .push_back(xC);
         yPos        .push_back(yCopy);
         zPos        .push_back(zCopy);
         rad         .push_back(radCopy);
      }

      if(yPos[Ik] < (meshOffsetY + maxRad))      { 
         yC = yCopy + Ly;
         isCopyY = true; }
      else if(yPos[Ik]  > (meshOffsetY + Ly - maxRad)) {
         yC = yCopy - Ly;
         isCopyY = true; }

      if(isCopyY)
      {
         particalType.push_back(pType);
         xPos        .push_back(xCopy);
         yPos        .push_back(yC);
         zPos        .push_back(zCopy);
         rad         .push_back(radCopy);
      }

      if(xPos[Ik]  < (meshOffsetX + maxRad) && yPos[Ik]  < (meshOffsetY + maxRad))     { 
         xC = xCopy + Lx;  
         yC = yCopy + Ly;
         isCopyCorner = true; }
      else if(xPos[Ik]  < (meshOffsetX + maxRad) && yPos[Ik]  > (meshOffsetY + Ly - maxRad)) { 
         xC = xCopy + Lx;    
         yC = yCopy - Ly;
         isCopyCorner = true; }

      else if(xPos[Ik]  > (meshOffsetX + Lx - maxRad) && yPos[Ik]  < (meshOffsetY + maxRad))     { 
         xC = xCopy - Lx;  
         yC = yCopy + Ly;
         isCopyCorner = true; }
      else if(xPos[Ik]  > (meshOffsetX + Lx - maxRad) && yPos[Ik]  > (meshOffsetY + Ly - maxRad)) { 
         xC = xCopy - Lx;  
         yC = yCopy - Ly;
         isCopyCorner = true; }

      if(isCopyCorner)
      {
         particalType.push_back(pType);
         xPos        .push_back(xC);
         yPos        .push_back(yC);
         zPos        .push_back(zCopy);
         rad         .push_back(radCopy);
      }
   }

   particalType.shrink_to_fit();
   xPos        .shrink_to_fit();
   yPos        .shrink_to_fit();
   zPos        .shrink_to_fit();
   rad         .shrink_to_fit();

   int numParticles = particalType.size();

   ::mfem::H1_FECollection FECol_H1(order, dim);
   ::mfem::ParFiniteElementSpace FESpace_H1(&pmesh, &FECol_H1, 1, mfem::Ordering::byNODES);

   ::mfem::ParGridFunction grainLSField(&FESpace_H1);

   int numNodes   = grainLSField.Size();
   mfem::Vector locationVector(dim);

   for ( int Ik = 0; Ik<numNodes; Ik++)
   {
      pmesh.GetNode(Ik, &locationVector[0]);
      const double * pCoords(locationVector.GetData());

      double LSVal = -1000.0;
      for (int ii = 0; ii < numParticles; ii++)
      {
         double val = rad[ii] - pow(pow(std::abs(pCoords[0] - xPos[ii]), 2) 
              + pow(std::abs(pCoords[1] - yPos[ii]), 2) + pow(std::abs(pCoords[2] - zPos[ii]), 2), 0.5);

         LSVal = std::max(val, LSVal);
      }
      grainLSField[Ik]= LSVal;


   }

   //dump the solution
   {
      ParaViewDataCollection paraview_dc("grain_test", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("level_set",&grainLSField);

      paraview_dc.Save();
   }


   MPI::Finalize();
   return 0;
}


