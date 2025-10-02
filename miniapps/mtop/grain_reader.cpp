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

#include <fstream>
#include <iostream>

#include "grain_reader.hpp"
#include "mfem.hpp"


namespace mfem
{

    //------------------------------------------------------------------------------

    bool
    file_exists( const std::string & aPath )
    {
        // test if file exists
        std::ifstream tFile( aPath );

        // save result into output variable
        bool aFileExists;

        if( tFile )
        {
            // close file
            tFile.close();
            aFileExists = true;
        }
        else
        {
            aFileExists = false;
        }

        return aFileExists;
    }

    //------------------------------------------------------------------------------

    std::vector<std::string> split_string(
            const std::string  & aString,
            const std::string  & aDelim)
    {
        // create empty cell of strings
        std::vector<std::string> VectorOfStrings;

        size_t start;
        size_t end = 0;

        while ((start = aString.find_first_not_of(aDelim, end)) != std::string::npos)
        {
            end = aString.find(aDelim, start);
            VectorOfStrings.push_back(aString.substr(start, end - start));
        }

        return VectorOfStrings;
    }

    //------------------------------------------------------------------------------

    Ascii::Ascii( const std::string & aPath, const FileMode & aMode ) :
                mMode( aMode )
    {
        // test if path is absolute
        if( aPath.substr( 0,1 ) == "/" )
        {
            mPath = aPath;
        }
        // test if path is relative
        else if( aPath.substr( 0,2 ) == "./" )
        {
            mPath = aPath;
        }
        else
        {
            MFEM_ASSERT( false, "");
            //mPath = std::sprint( "%s/%s", std::getenv( "PWD" ), aPath.c_str() );
        }

        switch ( aMode )
        {
            case( FileMode::OPEN_RDONLY ) :
            {
                this->load_buffer();
                break;
            }
            case( FileMode::NEW ) :
            {
                mBuffer.clear();
                break;
            }
            default:
            {
                MFEM_ASSERT( false, "Unknown file mode for ASCII file" );
            }
        }
    }

    //------------------------------------------------------------------------------

    Ascii::~Ascii()
    {
        MFEM_ASSERT( ! mChangedSinceLastSave, "File  was changed but never saved." );

        mBuffer.clear();
    }
    //------------------------------------------------------------------------------

    bool Ascii::save()
    {
        MFEM_ASSERT( mMode != FileMode::OPEN_RDONLY,
                "File can't be saved since it is opened in write protected mode." );

        // open file
        std::ofstream tFile( mPath.c_str(),  std::ofstream::trunc );

        if( tFile )
        {
            // save buffer to file
            for( std::string & tLine : mBuffer )
            {
                tFile << tLine << std::endl;
            }

            tFile.close();
        }
        else
        {
            MFEM_ASSERT( false,
                    "Something went wrong while trying to save." );
        }

        mChangedSinceLastSave = false;

        return mChangedSinceLastSave;
    }

    //------------------------------------------------------------------------------

    int Ascii::length() const
    {
        return mBuffer.size();
    }

    //------------------------------------------------------------------------------

    std::string & Ascii::line( const int aLineNumber )
    {
        return mBuffer[aLineNumber];
    }

    //------------------------------------------------------------------------------

    const std::string & Ascii::line( const int aLineNumber ) const
    {
        return mBuffer[aLineNumber];
    }

    //------------------------------------------------------------------------------

    void Ascii::print( const std::string & aLine )
    {
        mBuffer.push_back( aLine );

        mChangedSinceLastSave = true;
    }

    //------------------------------------------------------------------------------

    void Ascii::load_buffer()
    {
        // tidy up buffer
        mBuffer.clear();

        // make sure that file exists
        MFEM_ASSERT( file_exists( mPath ),
                "File does not exist." );

        // open file
        std::ifstream tFile( mPath );

        // test if file can be opened
        if( tFile )
        {
            // temporary container for string
            std::string tLine;

            while ( std::getline( tFile, tLine ) )
            {
                mBuffer.push_back( tLine );
            }

            // close file
            tFile.close();
        }
        else
        {
            MFEM_ASSERT( false, "Someting went wrong while opening file\n " );
        }
    }

    //------------------------------------------------------------------------------

    GrainReader::GrainReader( mfem::ParMesh * mesh, std::string & name ) 
      : mesh_(mesh), name_(name)
   {
      dim = mesh->Dimension();

      int tNumVertices  = mesh->GetNV();
      for (int i = 0; i < tNumVertices; ++i)
      {
         double * Coords = mesh->GetVertex(i);

         xMax = std::max(xMax, Coords[ 0 ]);
         yMax = std::max(yMax, Coords[ 1 ]);
         zMax = std::max(zMax, Coords[ 2 ]);

         xMin = std::min(xMin, Coords[ 0 ]);
         yMin = std::min(yMin, Coords[ 1 ]);
         zMin = std::min(zMin, Coords[ 2 ]);
      }

      MPI_Allreduce(MPI_IN_PLACE, &xMax, 1, MPI_DOUBLE, MPI_MAX, mesh_->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &yMax, 1, MPI_DOUBLE, MPI_MAX, mesh_->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &zMax, 1, MPI_DOUBLE, MPI_MAX, mesh_->GetComm());

      MPI_Allreduce(MPI_IN_PLACE, &xMin, 1, MPI_DOUBLE, MPI_MIN, mesh_->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &yMin, 1, MPI_DOUBLE, MPI_MIN, mesh_->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &zMin, 1, MPI_DOUBLE, MPI_MIN, mesh_->GetComm());

      Lx = xMax - xMin;
      Ly = yMax - yMin;
      Lz = zMax - zMin;

      this->readGrainFile();
   }

   void GrainReader::readGrainFile()
   {
   mfem::Ascii tAsciiReader( name_, FileMode::OPEN_RDONLY );

   int tNumLines = tAsciiReader.length();

   particalType.reserve(9*tNumLines);
   xPos.reserve(9*tNumLines);
   yPos.reserve(9*tNumLines);
   zPos.reserve(9*tNumLines);
   rad.reserve(9*tNumLines);

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

      if(xPos[Ik]  < (xMin + maxRad)){ 
         xC = xCopy + Lx; 
         isCopyX = true; }
      else if(xPos[Ik]  > (xMin + Lx - maxRad)) { 
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

      if(yPos[Ik] < (yMin + maxRad))      { 
         yC = yCopy + Ly;
         isCopyY = true; }
      else if(yPos[Ik]  > (yMin + Ly - maxRad)) {
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

      if(xPos[Ik]  < (xMin + maxRad) && yPos[Ik]  < (yMin + maxRad))     { 
         xC = xCopy + Lx;  
         yC = yCopy + Ly;
         isCopyCorner = true; }
      else if(xPos[Ik]  < (xMin + maxRad) && yPos[Ik]  > (yMin + Ly - maxRad)) { 
         xC = xCopy + Lx;    
         yC = yCopy - Ly;
         isCopyCorner = true; }

      else if(xPos[Ik]  > (xMin + Lx - maxRad) && yPos[Ik]  < (yMin + maxRad))     { 
         xC = xCopy - Lx;  
         yC = yCopy + Ly;
         isCopyCorner = true; }
      else if(xPos[Ik]  > (xMin + Lx - maxRad) && yPos[Ik]  > (yMin + Ly - maxRad)) { 
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

   numParticles = particalType.size();
   }

   void GrainReader::computeGridFunction( ::mfem::ParFiniteElementSpace& feSpace)
   {
      grainLSField.SetSpace(&feSpace);

      int numNodes   = grainLSField.Size();
      mfem::Vector locationVector(dim);

      int numEle = mesh_->GetNE();

      for ( int e = 0; e<numEle; e++)
      {
         const IntegrationRule &ir = feSpace.GetFE(e)->GetNodes();

         // Transformation of the element with the pos_mesh coordinates.
         mfem::IsoparametricTransformation Tr;
         feSpace.GetElementTransformation(e, &Tr);

         mfem::DenseMatrix pos_nodes;
         Tr.Transform(ir, pos_nodes);
         mfem::Vector valVec(pos_nodes.NumCols());

         for ( int Ik = 0; Ik< pos_nodes.NumCols(); Ik++)
         {
            double LSVal = -1000.0;
            for (int ii = 0; ii < numParticles; ii++)
            {
               double val = rad[ii] - pow(pow(std::abs(pos_nodes(0,Ik) - xPos[ii]), 2) 
                  + pow(std::abs(pos_nodes(1,Ik) - yPos[ii]), 2) + pow(std::abs(pos_nodes(2,Ik) - zPos[ii]), 2), 0.5);

               LSVal = std::max(val, LSVal);
            }
            valVec[Ik]= LSVal;
         }

         mfem::Array< int > dofs;
         feSpace.GetElementDofs( e, dofs );

         grainLSField.SetSubVector(dofs, valVec);
      }

      // for ( int Ik = 0; Ik<numNodes; Ik++)
      // {
      //    mesh_->GetNode(Ik, &locationVector[0]);
      //    const double * pCoords(locationVector.GetData());

      //    double LSVal = -1000.0;
      //    for (int ii = 0; ii < numParticles; ii++)
      //    {
      //       double val = rad[ii] - pow(pow(std::abs(pCoords[0] - xPos[ii]), 2) 
      //            + pow(std::abs(pCoords[1] - yPos[ii]), 2) + pow(std::abs(pCoords[2] - zPos[ii]), 2), 0.5);

      //       LSVal = std::max(val, LSVal);
      //    }
      //    grainLSField[Ik]= LSVal;
      // }
   }

}