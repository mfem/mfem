
#include "ToroidST.hpp"

ToroidST::ToroidST(SesquilinearForm * bf_, const Vector & aPmlThickness_, 
       double omega_, int nrsubdomains_)
: bf(bf_), aPmlThickness(aPmlThickness_), omega(omega_), nrsubdomains(nrsubdomains_)       
{
   
   cout << "In ToroidST" << endl;
   cin.get();


}


void ToroidST::Mult(const Vector & r, Vector & z) const 
{

}


ToroidST::~ToroidST()
{
   
}