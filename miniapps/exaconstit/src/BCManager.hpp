
#ifndef BCMANAGER
#define BCMANAGER

#include "BCData.hpp"

// C/C++ includes
#include <unordered_map> // for std::unordered_map

namespace mfem
{

class BCManager
{
public:
   BCManager();
   ~BCManager();

   static BCManager & getInstance()
   {
      static BCManager bcManager;
      return bcManager;
   }

   BCData & GetBCInstance( int bcID )
   {
      return m_bcInstances.at(bcID);
   }

   BCData const & GetBCInstance( int bcID) const
   {
      return m_bcInstances.at(bcID);
   }

   BCData & CreateBCs( int bcID )
   {
      return m_bcInstances[bcID];
   }
  
   std::unordered_map< int, BCData >&GetBCInstances()
   {
      return m_bcInstances;
   }

private:
   std::unordered_map< int, BCData > m_bcInstances;
};
}

#endif
