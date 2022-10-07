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

#ifndef PROJECTS_ASCII_HPP_
#define PROJECTS_ASCII_HPP_

#include <vector>
#include <iomanip>
#include <iostream>
#include <sstream>



namespace mfem
{
    //------------------------------------------------------------------------------

    enum class FileMode
    {
            NEW,
            OPEN_RDONLY,
            OPEN_RDWR
    };

    //------------------------------------------------------------------------------

    /**
     * this function tests if a file exists
     * @param aPath
     * @return
     */
    bool file_exists( const std::string & aPath );


    std::vector<std::string> split_string(
            const std::string  & aString,
            const std::string  & aDelim);

    class Ascii
    {
            //------------------------------------------------------------------------------
        protected:
            //------------------------------------------------------------------------------

            std::string          mPath;
            const FileMode       mMode;
            std::vector< std::string >       mBuffer;

            bool                 mChangedSinceLastSave = false;

            //------------------------------------------------------------------------------
        public:
            //------------------------------------------------------------------------------

            Ascii(
                    const std::string   & aPath,
                    const enum FileMode & aMode );

            //------------------------------------------------------------------------------

            ~Ascii();

            //------------------------------------------------------------------------------

            // save the buffer to the file
            bool save();

            //------------------------------------------------------------------------------

            void print( const std::string & aLine );

            //------------------------------------------------------------------------------

            /**
             * return the number of lines
             */
            int length() const;

            //------------------------------------------------------------------------------

            std::string & line( const int aLineNumber );

            //------------------------------------------------------------------------------

            const std::string & line( const int aLineNumber ) const;

            //------------------------------------------------------------------------------
        private:
            //------------------------------------------------------------------------------

            void load_buffer();

            //------------------------------------------------------------------------------
    };

    //------------------------------------------------------------------------------

        template<typename T>
        inline std::string stringify(T aValue)
        {
            std::ostringstream out;
            out << aValue;
            return out.str();
        }

        template<>
        inline std::string stringify<bool>(bool aValue)
        {
            std::ostringstream out;
            out << std::boolalpha << aValue;
            return out.str();
        }

        template<>
        inline std::string stringify<double>(double aValue)
        {
            std::ostringstream out;
            out << std::setprecision(14) << std::scientific << aValue;
            return out.str();
        }

        template<>
        inline std::string stringify<long double>(long double aValue)
        {
            std::ostringstream out;
            out << std::setprecision(14) << std::scientific << aValue;
            return out.str();
        }

        template<>
        inline std::string stringify<float>(float aValue)
        {
            std::ostringstream out;
            out << std::setprecision(14) << std::scientific << aValue;
            return out.str();
        }
}

#endif /* PROJECTS_ASCII_HPP_ */
