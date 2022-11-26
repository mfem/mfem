#include "hpc4mat.hpp"
#include "ascii.hpp"



namespace mfem {

    void SurrogateNLDiffusionCoefficient::MyNeuralNet::readSurrogateModel()
    {
        std::string tStringWeight1 = "./NeuralNet/vel_weights1_Ellips.txt";
        std::string tStringWeight2 = "./NeuralNet/vel_weights2_Ellips.txt";
        std::string tStringBias1   = "./NeuralNet/vel_bias1_Ellips.txt";
        std::string tStringBias2 = "./NeuralNet/vel_bias2_Ellips.txt";

        mfem::Ascii tAsciiReader1( tStringWeight1, FileMode::OPEN_RDONLY );
        mfem::Ascii tAsciiReader2( tStringWeight2, FileMode::OPEN_RDONLY );
        mfem::Ascii tAsciiReader3( tStringBias1,   FileMode::OPEN_RDONLY );
        mfem::Ascii tAsciiReader4( tStringBias2,   FileMode::OPEN_RDONLY );

        int tNumLines1 = tAsciiReader1.length();        Weight1Rows = tNumLines1;
        int tNumLines2 = tAsciiReader2.length();        Weight2Rows = tNumLines2;
        int tNumLines3 = tAsciiReader3.length();
        int tNumLines4 = tAsciiReader4.length();

        int Weight1Cols = split_string(tAsciiReader1.line( 0 ), " ").size();
        int Weight2Cols = split_string(tAsciiReader2.line( 0 ), " ").size();

        A1.SetSize(tNumLines1, Weight1Cols);
        A2.SetSize(tNumLines2, Weight2Cols);
        b1  .SetSize(tNumLines3);
        b2  .SetSize(tNumLines4);   

        for( int Ik = 0; Ik < tNumLines1; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader1.line( Ik );

            std::vector<std::string> ListOfStrings = split_string( tFileLine, " " );

            for( int Ii = 0; Ii < ListOfStrings.size(); Ii++ )
            {
                A1( Ik, Ii ) = std::stod( ListOfStrings[Ii] );
            }
        }

        for( int Ik = 0; Ik < tNumLines2; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader2.line( Ik );

            std::vector<std::string> ListOfStrings = split_string( tFileLine, " " );

            for( int Ii = 0; Ii < ListOfStrings.size(); Ii++ )
            {
                A2( Ik, Ii ) = std::stod( ListOfStrings[Ii] );
            }
        }

        for( int Ik = 0; Ik < tNumLines3; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader3.line( Ik );
                
            b1( Ik ) = std::stod( tFileLine );

        }

        for( int Ik = 0; Ik < tNumLines4; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader4.line( Ik );

            b2( Ik ) = std::stod( tFileLine );
        }
    }

    void SurrogateNLDiffusionCoefficient::MyNeuralNet_MatTensor::readSurrogateModel()
    {
        std::string tStringWeight1 = "./NeuralNet/tensor_weights1_Ellips.txt";
        std::string tStringWeight2 = "./NeuralNet/tensor_weights2_Ellips.txt";
        std::string tStringBias1   = "./NeuralNet/tensor_bias1_Ellips.txt";
        std::string tStringBias2 = "./NeuralNet/tensor_bias2_Ellips.txt";

        mfem::Ascii tAsciiReader1( tStringWeight1, FileMode::OPEN_RDONLY );
        mfem::Ascii tAsciiReader2( tStringWeight2, FileMode::OPEN_RDONLY );
        mfem::Ascii tAsciiReader3( tStringBias1,   FileMode::OPEN_RDONLY );
        mfem::Ascii tAsciiReader4( tStringBias2,   FileMode::OPEN_RDONLY );

        int tNumLines1 = tAsciiReader1.length();        Weight1Rows = tNumLines1;
        int tNumLines2 = tAsciiReader2.length();        Weight2Rows = tNumLines2;
        int tNumLines3 = tAsciiReader3.length();
        int tNumLines4 = tAsciiReader4.length();

        int Weight1Cols = split_string(tAsciiReader1.line( 0 ), " ").size();
        int Weight2Cols = split_string(tAsciiReader2.line( 0 ), " ").size();

        A1.SetSize(tNumLines1, Weight1Cols);
        A2.SetSize(tNumLines2, Weight2Cols);
        b1  .SetSize(tNumLines3);
        b2  .SetSize(tNumLines4);   

        for( int Ik = 0; Ik < tNumLines1; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader1.line( Ik );

            std::vector<std::string> ListOfStrings = split_string( tFileLine, " " );

            for( int Ii = 0; Ii < ListOfStrings.size(); Ii++ )
            {
                A1( Ik, Ii ) = std::stod( ListOfStrings[Ii] );
            }
        }

        for( int Ik = 0; Ik < tNumLines2; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader2.line( Ik );

            std::vector<std::string> ListOfStrings = split_string( tFileLine, " " );

            for( int Ii = 0; Ii < ListOfStrings.size(); Ii++ )
            {
                A2( Ik, Ii ) = std::stod( ListOfStrings[Ii] );
            }
        }

        for( int Ik = 0; Ik < tNumLines3; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader3.line( Ik );
                
            b1( Ik ) = std::stod( tFileLine );

        }

        for( int Ik = 0; Ik < tNumLines4; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader4.line( Ik );

            b2( Ik ) = std::stod( tFileLine );
        }
    }

}
