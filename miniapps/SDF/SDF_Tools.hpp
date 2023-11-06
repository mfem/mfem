// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SDF_TOOLS_HPP
#define MFEM_SDF_TOOLS_HPP

#include "mfem.hpp"

namespace mfem
{
namespace sdf
{
const double gSDFepsilon = 1e-7;
//-------------------------------------------------------------------------------

inline void
TrianglePermutation( const unsigned int aZ,
                     unsigned int & aX, unsigned int & aY )
{
   if ( aZ == 0 )
   {
      // y-z
      aX = 1;
      aY = 2;
   }
   else if ( aZ == 1 )
   {
      // x-z
      aX = 2;
      aY = 0;
   }
   else
   {
      // x-y
      aX = 0;
      aY = 1;
   }
}

//-------------------------------------------------------------------------------

inline mfem::Vector
cross( const mfem::Vector & aA, const mfem::Vector & aB )
{
   mfem::Vector aOut( 3 );

   aOut[ 0 ] = aA[ 1 ] * aB[ 2 ] - aA[ 2 ] * aB[ 1 ];
   aOut[ 1 ] = aA[ 2 ] * aB[ 0 ] - aA[ 0 ] * aB[ 2 ];
   aOut[ 2 ] = aA[ 0 ] * aB[ 1 ] - aA[ 1 ] * aB[ 0 ];

   return aOut;
}

// =============================================================================
// LEXICAL FUNCTIONS
// =============================================================================

template< typename T >
inline T
min( const T& aA, const T& aB )
{
   return ( aA < aB ) ? ( aA ) : ( aB );
}

// -----------------------------------------------------------------------------

template< typename T >
inline T
max( const T& aA, const T& aB )
{
   return ( aA > aB ) ? ( aA ) : ( aB );
}

// -----------------------------------------------------------------------------

template< typename T >
inline T
max( const T& aA, const T& aB, const T& aC )
{
   return max( max( aA, aB ), aC );
}

// -----------------------------------------------------------------------------

template< typename T >
inline T
min( const T& aA, const T& aB, const T& aC )
{
   return min( min( aA, aB ), aC );
}

// -----------------------------------------------------------------------------

bool inline string_to_bool( const std::string& aString )
{
   // locale
   std::locale loc;

   // lower string of aString
   std::string tLowerString( aString );
   for ( unsigned int i = 0; i < aString.length(); ++i )
   {
      tLowerString[ i ] = std::tolower( aString[ i ] );
   }

   return ( tLowerString == "true"
            || tLowerString == "on"
            || tLowerString == "yes"
            || tLowerString == "1" );
}

// =============================================================================
// Linear Algebra
// =============================================================================

inline mfem::DenseMatrix
rotation_matrix( const mfem::Vector & aAxis, const double & aAngle )
{
   mfem::DenseMatrix aT( 3, 3 );

   double tCos          = std::cos( aAngle );
   double tSin          = std::sin( aAngle );
   double tCos_minusOne = tCos - 1.0;

   aT( 0, 0 ) = tCos - aAxis[ 0 ] * aAxis[ 0 ] * tCos_minusOne;
   aT( 1, 0 ) = aAxis[ 2 ] * tSin - aAxis[ 0 ] * aAxis[ 1 ] * tCos_minusOne;
   aT( 2, 0 ) = -aAxis[ 1 ] * tSin - aAxis[ 0 ] * aAxis[ 2 ] * tCos_minusOne;
   aT( 0, 1 ) = -aAxis[ 2 ] * tSin - aAxis[ 0 ] * aAxis[ 1 ] * tCos_minusOne;
   aT( 1, 1 ) = tCos - aAxis[ 1 ] * aAxis[ 1 ] * tCos_minusOne;
   aT( 2, 1 ) = aAxis[ 0 ] * tSin - aAxis[ 1 ] * aAxis[ 2 ] * tCos_minusOne;
   aT( 0, 2 ) = aAxis[ 1 ] * tSin - aAxis[ 0 ] * aAxis[ 2 ] * tCos_minusOne;
   aT( 1, 2 ) = -aAxis[ 0 ] * tSin - aAxis[ 1 ] * aAxis[ 2 ] * tCos_minusOne;
   aT( 2, 2 ) = tCos - aAxis[ 2 ] * aAxis[ 2 ] * tCos_minusOne;
   return aT;
}

// =============================================================================
// Random Stuff
// =============================================================================
inline unsigned int
random_seed()
{
   std::ifstream file( "/dev/urandom", std::ios::binary );
   unsigned int          tSeed;
   if ( file.is_open() )
   {
      char* memblock;
      int   size = sizeof( unsigned int );
      memblock   = new char[ size ];
      file.read( memblock, size );
      file.close();
      tSeed = *reinterpret_cast< int* >( memblock );
      delete[] memblock;
   }
   else
   {
      tSeed = time( NULL );
   }
   return tSeed;
}

// -----------------------------------------------------------------------------

/**
 * @brief returns a normalized pseudorandom vector
 *
 * @return    The random vector. Must be initialized already.
 */

inline mfem::Vector
random_axis()
{
   mfem::Vector aVector( 3 );

   std::srand( random_seed() );

   for ( unsigned int i = 0; i < 3; ++i )
   {
      aVector[ i ] = std::rand();
   }
   double tNorm = aVector.Norml2();

   for ( unsigned int i = 0; i < 3; ++i )
   {
      aVector[ i ] /= tNorm;
   }

   return aVector;
}

// -----------------------------------------------------------------------------

/**
 * @brief returns a pseudorandom angle between -Pi and Pi
 *
 * @return Angle: The returned angle in rad.
 *
 */
inline double
random_angle()
{
   std::srand( random_seed() );

   return ( ( (double)std::rand() ) / RAND_MAX - 0.5 )
          * 4.0 * std::acos( 0.0 );
}

// =============================================================================
// String tools
// =============================================================================

/**
 * this funcitons removes leading and double spaces and tabs from a string
 */
inline std::string
clean( const std::string& aString )
{
   // length of string
   unsigned int tLength = aString.size();

   bool tFlag = true;    // flag telling if last char is space or tab

   std::string aResult = "";

   for ( unsigned int k = 0; k < tLength; ++k )
   {
      if ( (int)aString[ k ] == 9 || (int)aString[ k ] == 32 )
      {
         if ( !tFlag )
         {
            aResult = aResult + " ";
         }
         tFlag = true;
      }
      else
      {
         tFlag   = false;
         aResult = aResult + aString[ k ];
      }
   }

   // trim last space
   if ( tFlag )
   {
      aResult = aResult.substr( 0, aResult.find_last_of( " " ) );
   }
   return aResult;
}

//-------------------------------------------------------------------------------

inline std::vector< std::string >
string_to_words( const std::string& aString )
{
   // output cell
   std::vector< std::string > aWords;

   // cleanup string
   std::string tClean = clean( aString );

   // get length of string
   int tLength = tClean.size();

   // extract words from string
   if ( tLength > 0 )
   {
      int tEnd   = -1;
      int tStart = 0;
      for ( int k = 0; k < tLength; ++k )
      {
         if ( tClean[ k ] == ' ' )
         {
            tStart = tEnd + 1;
            tEnd   = k;
            aWords.push_back( tClean.substr( tStart, tEnd - tStart ) );
         }
      }

      // last word
      tStart = tEnd + 1;
      tEnd   = tLength;
      aWords.push_back( tClean.substr( tStart, tEnd - tStart ) );
   }

   return aWords;
}
} /* namespace sdf */
} /* namespace mfem */

#endif
