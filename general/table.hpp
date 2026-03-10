// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
   Connection(int from, int to) : from(from), to(to) { }

   bool operator==(const Connection &rhs) const
   { return (from == rhs.from) && (to == rhs.to); }
   bool operator<(const Connection &rhs) const
   { return (from == rhs.from) ? (to < rhs.to) : (from < rhs.from); }
};


/** @brief Table stores the connectivity of elements of TYPE I to elements of
    TYPE II. For example, it may be the element-to-face connectivity table. */
class Table
{
protected:
   int size; ///< The number of TYPE I elements.

   /// @name Arrays for the connectivity information in the CSR storage.
   /// @{

   /// The length of the I array is 'size + 1',
   Array<int> I;

   /// @brief The length of the J array is equal to the number of connections
   /// between TYPE I and TYPE II elements.
   Array<int> J;

   /// @}

public:
   /// Creates an empty table
   Table() { size = -1; }

   /// Merge constructor: combine two tables into one table.
   Table(const Table &table1,
         const Table &table2, int offset2);

   /// Merge constructor: combine three tables into one table.
   Table(const Table &table1,
         const Table &table2, int offset2,
         const Table &table3, int offset3);

   /// Create a table with an upper limit for the number of connections.
   explicit Table(int dim, int connections_per_row = 3);

   /// Create a table from a list of connections, see MakeFromList().
   Table(int nrows, Array<Connection> &list) : size(-1)
   { MakeFromList(nrows, list); }

   /// @brief Create a table with one entry per row with column indices given by
   /// @a partitioning.
   Table(int nrows, int *partitioning);

   /// @name Used together with the default constructor
   /// @{
   void MakeI(int nrows);
   void AddAColumnInRow(int r) { I[r]++; }
   void AddColumnsInRow(int r, int ncol) { I[r] += ncol; }
   void MakeJ();
   void AddConnection(int r, int c) { J[I[r]++] = c; }
   void AddConnections(int r, const int *c, int nc);
   void ShiftUpI();
   /// @}

   /// Set the size and the number of connections for the table.
   void SetSize(int dim, int connections_per_row);

   /// @brief Set the rows and the number of all connections for the table.
   ///
   /// Does NOT initialize the whole array I ! (I[0]=0 and I[rows]=nnz only)
   void SetDims(int rows, int nnz);

   /// Returns the number of TYPE I elements.
   inline int Size() const { return size; }

   /// @brief Returns the number of connections in the table.
   ///
   /// If Finalize() is not called, it returns the number of possible
   /// connections established by the used constructor. Otherwise, it is exactly
   /// the number of established connections after calling Finalize(). */
   inline int Size_of_connections() const { return J.Size(); }

   /// @brief Returns index of the connection between element i of TYPE I and
   /// element j of TYPE II.
   ///
   /// If there is no connection between element i and element j established in
   /// the table, then the return value is -1.
   int operator() (int i, int j) const;

   /// Return row i in array row (the Table must be finalized)
   void GetRow(int i, Array<int> &row) const;

   int RowSize(int i) const { return I[i+1] - I[i]; }

   const int *GetRow(int i) const { return J.GetMemory() + I[i]; }
   int *GetRow(int i) { return J.GetMemory() + I[i]; }

   int *GetI() { return I.GetData(); }
   int *GetJ() { return J.GetData(); }
   const int *GetI() const { return I.GetData(); }
   const int *GetJ() const { return J.GetData(); }

   Memory<int> &GetIMemory() { return I.GetMemory(); }
   Memory<int> &GetJMemory() { return J.GetMemory(); }
   const Memory<int> &GetIMemory() const { return I.GetMemory(); }
   const Memory<int> &GetJMemory() const { return J.GetMemory(); }

   const int *ReadI(bool on_dev = true) const { return I.Read(on_dev); }
   int *WriteI(bool on_dev = true) { return I.Write(on_dev); }
   int *ReadWriteI(bool on_dev = true) { return I.ReadWrite(on_dev); }
   const int *HostReadI() const { return I.HostRead(); }
   int *HostWriteI() { return I.HostWrite(); }
   int *HostReadWriteI() { return I.HostReadWrite(); }

   const int *ReadJ(bool on_dev = true) const { return J.Read(on_dev); }
   int *WriteJ(bool on_dev = true) { return J.Write(on_dev); }
   int *ReadWriteJ(bool on_dev = true) { return J.ReadWrite(on_dev); }
   const int *HostReadJ() const { return J.HostRead(); }
   int *HostWriteJ() { return J.HostWrite(); }
   int *HostReadWriteJ() { return J.HostReadWrite(); }

   /// Sort the column (TYPE II) indices in each row.
   void SortRows();

   /// Replace the #I and #J arrays with the given @a newI and @a newJ arrays.
   /** If @a newsize < 0, then the size of the Table is not modified. */
   void SetIJ(int *newI, int *newJ, int newsize = -1);

   /// Establish connection between element i and element j in the table.
   /** The return value is the index of the connection. It returns -1 if it
       fails to establish the connection. Possibilities are there is not enough
       memory on row i to establish connection to j, an attempt to establish new
       connection after calling Finalize(). */
   int Push( int i, int j );

   /// Finalize the table initialization.
   /** The function may be called only once, after the table has been
       initialized, in order to compress array J (by getting rid of -1's in
       array J). Calling this function will "freeze" the table and function Push
       will work no more. Note: The table is functional even without calling
       Finalize(). */
   void Finalize();

   /// @brief Create the table from a list of connections {(from, to)}, where
   /// 'from' is a TYPE I index and 'to' is a TYPE II index.
   ///
   /// The list is assumed to be sorted and free of duplicities, i.e., you need
   /// to call Array::Sort and Array::Unique before calling this method. */
   void MakeFromList(int nrows, const Array<Connection> &list);

   /// Returns the number of TYPE II elements (after Finalize() is called).
   int Width() const;

   /// Releases ownership of and null-ifies the data.
   void LoseData() { size = -1; I.LoseData(); J.LoseData(); }

   /// Prints the table to the stream @a out.
   void Print(std::ostream & out = mfem::out, int width = 4) const;
   void PrintMatlab(std::ostream & out) const;

   void Save(std::ostream &out) const;
   void Load(std::istream &in);

   void Copy(Table & copy) const;
   void Swap(Table & other);

   void Clear();

   std::size_t MemoryUsage() const;
};

///  Transpose a Table
void Transpose (const Table &A, Table &At, int ncols_A_ = -1);
Table * Transpose (const Table &A);

///  @brief Transpose an Array<int>.
///
/// The array @a A represents a table where each row @a i has exactly one
/// connection to the column (TYPE II) index specified by @a A[i].
///
/// @note The column (TYPE II) indices in each row of @a At will be sorted.
void Transpose(const Array<int> &A, Table &At, int ncols_A_ = -1);

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
