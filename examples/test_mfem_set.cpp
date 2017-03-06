
#include "mfem.hpp"
#include <ctime>
#include <set>
#include <unordered_set>

using namespace std;
using namespace mfem;

int main()
{
   // const int num_entries = 4*1024*1024; // avg bin size = 2 (with defaults)
   // const int num_entries = 6*1024*1024; // avg bin size = 1.5 (with defaults)
   const int num_entries = 5*1024*1024; // avg bin size = 1.25 (with defaults)
   // const int num_entries = 50*1024*1024;

   // const int range = 300*1000;
   struct func
   {
      // int operator()(int n) { return n%range; }
      int operator()(int n) { return n; }
   };
   func fn;

   time_t seed = time(NULL);

   Set<int> my_set;
   // cheating ... speeds up Insert(), slows down Find():
   // Set<int> my_set(num_entries/2);
   // cheating ... speeds up Insert():
   // Set<int> my_set((num_entries/4)*3);
   // cheating ... speeds up both Insert() and Find():
   // Set<int> my_set(num_entries);

   cout << "Inserting " << num_entries << " random entries in a mfem::Set ..."
        << flush;
   tic_toc.Clear();
   tic_toc.Start();
   // cheating ... does not affect speed much, just the final MemoryUsage():
   // my_set.Reserve(num_entries);
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      my_set.Insert(fn(rand()));
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   cout << "Inserting the same random entries again ..."
        << flush;
   tic_toc.Clear();
   tic_toc.Start();
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      my_set.Insert(fn(rand()));
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   cout << "Verifying all entries in the Set ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      MFEM_VERIFY(my_set.Find(fn(rand())) != -1, "entry not found");
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   my_set.PrintStats();

   cout << "\nOptimizing the Set ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   my_set.Optimize();
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   cout << "Inserting the same random entries again ..."
        << flush;
   tic_toc.Clear();
   tic_toc.Start();
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      my_set.Insert(fn(rand()));
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   cout << "Verifying all entries in the Set ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      MFEM_VERIFY(my_set.Find(fn(rand())) != -1, "entry not found");
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   my_set.PrintStats();

   cout << "Sorting the set ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   my_set.Sort();
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   cout << "Verifying all entries in the Set ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      MFEM_VERIFY(my_set.Find(fn(rand())) != -1, "entry not found");
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   my_set.Clear();


   //----------------------------------------------------------------
   //  std::set
   //----------------------------------------------------------------

   std::set<int> std_set;
   cout << "\nInserting " << num_entries
        << " random entries in an std::set ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      std_set.insert(fn(rand()));
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   cout << "Verifying all entries in the std::set ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      MFEM_VERIFY(std_set.find(fn(rand())) != std_set.end(), "entry not found");
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;
   std_set.clear();


   //----------------------------------------------------------------
   //  std::unordered_set
   //----------------------------------------------------------------

   std::unordered_set<int> uo_set;
   cout << "\nInserting " << num_entries
        << " random entries in an std::unordered_set ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      uo_set.insert(fn(rand()));
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;

   cout << "Verifying all entries in the std::unordered_set ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   srand(seed);
   for (int i = 0; i < num_entries; i++)
   {
      MFEM_VERIFY(uo_set.find(fn(rand())) != uo_set.end(), "entry not found");
   }
   tic_toc.Stop();
   cout << " done (" << tic_toc.RealTime() << " sec).\n" << endl;
   uo_set.clear();


   return 0;
}
