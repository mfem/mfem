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

#include "ascii.hpp"
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

}