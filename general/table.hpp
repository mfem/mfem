// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TABLE
#define MFEM_TABLE

// Data types for Table.

#include "mem_alloc.hpp"
#include "array.hpp"
#include "globals.hpp"
#include <ostream>
#include <istream>

namespace mfem
{

/// Helper struct for defining a connectivity table, see Table::MakeFromList.
struct Connection
{
   int from, to;
   Connection() = default;
   Connection(int from, int to) : from(from), to(to) {}

   bool operator== (const Connection &rhs) const
   { return (from == rhs.from) && (to == rhs.to); }
   bool operator< (const Connection &rhs) const
   { return (from == rhs.from) ? (to < rhs.to) : (from < rhs.from); }
};


/** Data type Table. Table stores the connectivity of elements of TYPE I
    to elements of TYPE II, for example, it may be Element-To-Face
    connectivity table, etc. */
class Table
{
protected:
   /// size is the number of TYPE I elements.
   int size;

   /** Arrays for the connectivity information in the CSR storage.
       I is of size "size+1", J is of size the number of connections
       between TYPE I to TYPE II elements (actually stored I[size]). */
   Memory<int> I, J;

public:
   /// Creates an empty table
   Table() { size = -1; I.Reset(); J.Reset(); }

   /// Copy constructor
   Table(const Table &);

   /// Assignment operator: deep copy
   Table& operator=(const Table &rhs);

   /// Create a table with an upper limit for the number of connections.
   explicit Table (int dim, int connections_per_row = 3);

   /** Create a table from a list of connections, see MakeFromList(). */
   Table(int nrows, Array<Connection> &list) : size(-1)
   { I.Reset(); J.Reset(); MakeFromList(nrows, list); }

   /** Create a table with one entry per row with column indices given
       by 'partitioning'. */
   Table (int nrows, int *partitioning);

   /// Next 7 methods are used together with the default constructor
   void MakeI (int nrows);
   void AddAColumnInRow (int r) { I[r]++; }
   void AddColumnsInRow (int r, int ncol) { I[r] += ncol; }
   void MakeJ();
   void AddConnection (int r, int c) { J[I[r]++] = c; }
   void AddConnections (int r, const int *c, int nc);
   void ShiftUpI();

   /// Set the size and the number of connections for the table.
   void SetSize(int dim, int connections_per_row);

   /** Set the rows and the number of all connections for the table.
       Does NOT initialize the whole array I ! (I[0]=0 and I[rows]=nnz only) */
   void SetDims(int rows, int nnz);

   /// Returns the number of TYPE I elements.
   inline int Size() const { return size; }

   /** Returns the number of connections in the table. If Finalize() is
       not called, it returns the number of possible connections established
       by the used constructor. Otherwise, it is exactly the number of
       established connections before calling Finalize(). */
   inline int Size_of_connections() const { return I[size]; }

   /** Returns index of the connection between element i of TYPE I and
       element j of TYPE II. If there is no connection between element i
       and element j established in the table, then the return value is -1. */
   int operator() (int i, int j) const;

   /// Return row i in array row (the Table must be finalized)
   void GetRow(int i, Array<int> &row) const;

   int RowSize(int i) const { return I[i+1]-I[i]; }

   const int *GetRow(int i) const { return J+I[i]; }
   int *GetRow(int i) { return J+I[i]; }

   int *GetI() { return I; }
   int *GetJ() { return J; }
   const int *GetI() const { return I; }
   const int *GetJ() const { return J; }

   Memory<int> &GetIMemory() { return I; }
   Memory<int> &GetJMemory() { return J; }
   const Memory<int> &GetIMemory() const { return I; }
   const Memory<int> &GetJMemory() const { return J; }

   /// @brief Sort the column (TYPE II) indices in each row.
   void SortRows();

   /// Replace the #I and #J arrays with the given @a newI and @a newJ arrays.
   /** If @a newsize < 0, then the size of the Table is not modified. */
   void SetIJ(int *newI, int *newJ, int newsize = -1);

   /** Establish connection between element i and element j in the table.
       The return value is the index of the connection. It returns -1 if it
       fails to establish the connection. Possibilities are there is not
       enough memory on row i to establish connection to j, an attempt to
       establish new connection after calling Finalize(). */
   int Push( int i, int j );

   /** Finalize the table initialization. The function may be called
       only once, after the table has been initialized, in order to compress
       array J (by getting rid of -1's in array J). Calling this function
       will "freeze" the table and function Push will work no more.
       Note: The table is functional even without calling Finalize(). */
   void Finalize();

   /** Create the table from a list of connections {(from, to)}, where 'from'
       is a TYPE I index and 'to' is a TYPE II index. The list is assumed to be
       sorted and free of duplicities, i.e., you need to call Array::Sort and
       Array::Unique before calling this method. */
   void MakeFromList(int nrows, const Array<Connection> &list);

   /// Returns the number of TYPE II elements (after Finalize() is called).
   int Width() const;

   /// Call this if data has been stolen.
   void LoseData() { size = -1; I.Reset(); J.Reset(); }

   /// Prints the table to stream out.
   void Print(std::ostream & out = mfem::out, int width = 4) const;
   void PrintMatlab(std::ostream & out) const;

   void Save(std::ostream &out) const;
   void Load(std::istream &in);

   void Copy(Table & copy) const;
   void Swap(Table & other);

   void Clear();

   long MemoryUsage() const;

   /// Destroys Table.
   ~Table();
};

/// Specialization of the template function Swap<> for class Table
template <> inline void Swap<Table>(Table &a, Table &b)
{
   a.Swap(b);
}

///  Transpose a Table
void Transpose (const Table &A, Table &At, int _ncols_A = -1);
Table * Transpose (const Table &A);

///  Transpose an Array<int>
void Transpose(const Array<int> &A, Table &At, int _ncols_A = -1);

///  C = A * B  (as boolean matrices)
void Mult (const Table &A, const Table &B, Table &C);
Table * Mult (const Table &A, const Table &B);


/** Data type STable. STable is similar to Table, but it's for symmetric
    connectivity, i.e. TYPE I is equivalent to TYPE II. In the first
    dimension we put the elements with smaller index. */

class STable : public Table
{
public:
   /// Creates table with fixed number of connections.
   STable (int dim, int connections_per_row = 3);

   /** Returns index of the connection between element i of TYPE I and
       element j of TYPE II. If there is no connection between element i
       and element j established in the table, then the return value is -1. */
   int operator() (int i, int j) const;

   /** Establish connection between element i and element j in the table.
       The return value is the index of the connection. It returns -1 if it
       fails to establish the connection. Possibilities are there is not
       enough memory on row i to establish connection to j, an attempt to
       establish new connection after calling Finalize(). */
   int Push( int i, int j );

   /// Destroys STable.
   ~STable() {}
};


class DSTable
{
private:
   class Node
   {
   public:
      Node *Prev;
      int  Column, Index;
   };

   int  NumRows, NumEntries;
   Node **Rows;
#ifdef MFEM_USE_MEMALLOC
   MemAlloc <Node, 1024> NodesMem;
#endif

   int Push_(int r, int c);
   int Index(int r, int c) const;

public:
   DSTable(int nrows);
   int NumberOfRows() const { return (NumRows); }
   int NumberOfEntries() const { return (NumEntries); }
   int Push(int a, int b)
   { return ((a <= b) ? Push_(a, b) : Push_(b, a)); }
   int operator()(int a, int b) const
   { return ((a <= b) ? Index(a, b) : Index(b, a)); }
   ~DSTable();

   class RowIterator
   {
   private:
      Node *n;
   public:
      RowIterator (const DSTable &t, int r) { n = t.Rows[r]; }
      int operator!() { return (n != NULL); }
      void operator++() { n = n->Prev; }
      int Column() { return (n->Column); }
      int Index() { return (n->Index); }
      void SetIndex(int new_idx) { n->Index = new_idx; }
   };
};

}

#endif
