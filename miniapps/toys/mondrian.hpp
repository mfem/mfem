// MFEM Mondrian Miniapp - Code for Parsing PGM files

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

// Simple class to parse portable graymap format (PGM) image files, see
// http://netpbm.sourceforge.net/doc/pgm.html
class ParsePGM
{
public:
   ParsePGM(const char *filename);
   ~ParsePGM();

   int Height() const { return N; }
   int Width() const { return M; }

   int operator()(int i, int j) const
   { return int((pgm8) ? pgm8[M*i+j] : pgm16[M*i+j]); }

private:
   int M, N;
   int depth;

   char *pgm8;
   unsigned short int *pgm16;

   void ReadMagicNumber(istream &in);
   void ReadComments(istream &in);
   void ReadDimensions(istream &in);
   void ReadDepth(istream &in);
   void ReadPGM(istream &in);
};

ParsePGM::ParsePGM(const char *filename)
   : M(-1), N(-1), depth(-1), pgm8(NULL), pgm16(NULL)
{
   ifstream in(filename);
   if (!in)
   {
      // Abort with an error message
      MFEM_ABORT("Image file not found: " << filename << '\n');
   }

   ReadMagicNumber(in);
   ReadDimensions(in);
   ReadDepth(in);
   ReadPGM(in);

   in.close();
}

ParsePGM::~ParsePGM()
{
   if (pgm8  != NULL) { delete [] pgm8; }
   if (pgm16 != NULL) { delete [] pgm16; }
}

void ParsePGM::ReadMagicNumber(istream &in)
{
   char c;
   int p;
   in >> c >> p; // Read magic number which should be P2 or P5
   MFEM_VERIFY(c == 'P' && (p == 2 || p == 5),
               "Invalid PGM file! Unrecognized magic number\""
               << c << p << "\".");
   ReadComments(in);
}

void ParsePGM::ReadComments(istream &in)
{
   string buf;
   in >> std::ws; // absorb any white space
   while (in.peek() == '#')
   {
      std::getline(in,buf);
   }
   in >> std::ws; // absorb any white space
}

void ParsePGM::ReadDimensions(istream &in)
{
   in >> M;
   ReadComments(in);
   in >> N;
   ReadComments(in);
}

void ParsePGM::ReadDepth(istream &in)
{
   in >> depth;
   ReadComments(in);
}

void ParsePGM::ReadPGM(istream &in)
{
   if (depth < 16)
   {
      pgm8 = new char[M*N];
   }
   else
   {
      pgm16 = new unsigned short int[M*N];
   }

   if (pgm8)
   {
      for (int i=0; i<M*N; i++)
      {
         in >> pgm8[i];
      }
   }
   else
   {
      for (int i=0; i<M*N; i++)
      {
         in >> pgm16[i];
      }
   }
}

// Given a point x, return its "material" specification defined by the grayscale
// pixel values from the pgm image using NC different colors.
int material(const ParsePGM &pgm, int NC, Vector &x, Vector &xmin, Vector &xmax)
{
   // Rescaling to [0,1]^sdim
   for (int i = 0; i < x.Size(); i++)
   {
      x(i) = (x(i)-xmin(i))/(xmax(i)-xmin(i));
   }

   int M = pgm.Width();
   int N = pgm.Height();

   int i = x(1)*N, j = x(0)*M;
   if (i == N) { i = N-1; }
   if (j == M) { j = M-1; }
   i = N-1-i;

   return pgm(i,j)/NC+1;
}
