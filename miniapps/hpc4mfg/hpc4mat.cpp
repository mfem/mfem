#include "hpc4mat.hpp"
#include "ascii.hpp"



namespace mfem {

    void SurrogateNLDiffusionCoefficient::MyNeuralNet::readSurrogateModel()
    {
        std::string tStringWeight1 = "./NeuralNet/vel_weights1_Circle.txt";
        std::string tStringWeight2 = "./NeuralNet/vel_weights2_Circle.txt";
        std::string tStringBias1   = "./NeuralNet/vel_bias1_Circle.txt";
        std::string tStringBias2 = "./NeuralNet/vel_bias2_Circles.txt";

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
        // std::string tStringWeight1 = "./NeuralNet/Circle_weights1_Darcy.txt";
        // std::string tStringWeight2 = "./NeuralNet/Circle_weights2_Darcy.txt";
        // std::string tStringBias1   = "./NeuralNet/Circle_bias1_Darcy.txt";
        // std::string tStringBias2   = "./NeuralNet/Circle_bias2_Darcy.txt";

        // std::string tStringWeight1 = "./NeuralNet/weights1_Darcy_FD_Circle_FD_triangle.txt";
        // std::string tStringWeight2 = "./NeuralNet/weights2_Darcy_FD_Circle_FD_triangle.txt";
        // std::string tStringBias1   = "./NeuralNet/bias1_Darcy_FD_Circle_FD_triangle.txt";
        // std::string tStringBias2   = "./NeuralNet/bias2_Darcy_FD_Circle_FD_triangle.txt";

        // std::string tStringWeight1 = "./NeuralNet/weights1_FD_Labrynth_FD_100.txt";
        // std::string tStringWeight2 = "./NeuralNet/weights2_FD_Labrynth_FD_100.txt";
        // std::string tStringBias1   = "./NeuralNet/bias1_FD_Labrynth_FD_100.txt";
        // std::string tStringBias2   = "./NeuralNet/bias2_FD_Labrynth_FD_100.txt";

        // std::string tStringWeight1 = "./NeuralNet/weights1_Gyroid_FD_3D.txt";
        // std::string tStringWeight2 = "./NeuralNet/weights2_Gyroid_FD_3D.txt";
        // std::string tStringBias1   = "./NeuralNet/bias1_Gyroid_FD_3D.txt";
        // std::string tStringBias2   = "./NeuralNet/bias2_Gyroid_FD_3D.txt";

        // std::string tStringWeight1 = "./NeuralNet/weights1_Darcy_FD_Labrynth_FD.txt";
        // std::string tStringWeight2 = "./NeuralNet/weights2_Darcy_FD_Labrynth_FD.txt";
        // std::string tStringBias1   = "./NeuralNet/bias1_Darcy_FD_Labrynth_FD.txt";
        // std::string tStringBias2   = "./NeuralNet/bias2_Darcy_FD_Labrynth_FD.txt";

        std::string tStringWeight1 = "./NeuralNet/weights1_Darcy_Circle_FD.txt";
        std::string tStringWeight2 = "./NeuralNet/weights2_Darcy_Circle_FD.txt";
        std::string tStringBias1   = "./NeuralNet/bias1_Darcy_Circle_FD.txt";
        std::string tStringBias2   = "./NeuralNet/bias2_Darcy_Circle_FD.txt";

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
