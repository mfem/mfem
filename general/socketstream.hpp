// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SOCKETSTREAM
#define MFEM_SOCKETSTREAM

#include "../config/config.hpp"
#include "error.hpp"
#include "globals.hpp"

#ifdef MFEM_USE_GNUTLS
#include <gnutls/gnutls.h>
#if GNUTLS_VERSION_NUMBER < 0x020800
#error "MFEM requires GnuTLS version >= 2.8.0"
#endif
// Use X.509 certificates: (comment out to use OpenPGP keys)
#define MFEM_USE_GNUTLS_X509
#endif

namespace mfem
{

class socketbuf : public std::streambuf
{
protected:
   int socket_descriptor;
   static const int buflen = 1024;
   char ibuf[buflen], obuf[buflen];

public:
   socketbuf()
   {
      socket_descriptor = -1;
   }

   explicit socketbuf(int sd)
   {
      socket_descriptor = sd;
      setp(obuf, obuf + buflen);
   }

   socketbuf(const char hostname[], int port)
   {
      socket_descriptor = -1;
      open(hostname, port);
   }

   /** @brief Attach a new socket descriptor to the socketbuf. Returns the old
       socket descriptor which is NOT closed. */
   virtual int attach(int sd);

   /// Detach the current socket descriptor from the socketbuf.
   int detach() { return attach(-1); }

   /** @brief Open a socket on the 'port' at 'hostname' and store the socket
       descriptor. Returns 0 if there is no error, otherwise returns -1. */
   virtual int open(const char hostname[], int port);

   /// Close the current socket descriptor.
   virtual int close();

   /// Returns the attached socket descriptor.
   int getsocketdescriptor() { return socket_descriptor; }

   /** @brief Returns true if the socket is open and has a valid socket
       descriptor. Otherwise returns false. */
   bool is_open() { return (socket_descriptor >= 0); }

   virtual ~socketbuf() { close(); }

protected:
   virtual int sync();

   virtual int_type underflow();

   virtual int_type overflow(int_type c = traits_type::eof());

   virtual std::streamsize xsgetn(char_type *s__, std::streamsize n__);

   virtual std::streamsize xsputn(const char_type *s__, std::streamsize n__);
};


#ifdef MFEM_USE_GNUTLS

class GnuTLS_status
{
protected:
   int res;

public:
   GnuTLS_status() : res(GNUTLS_E_SUCCESS) { }

   bool good() const { return (res == GNUTLS_E_SUCCESS); }

   void set_result(int result) { res = result; }

   int get_result() const { return res; }

   void print_on_error(const char *msg) const
   {
      if (good()) { return; }
      mfem::err << "Error in " << msg << ": " << gnutls_strerror(res)
                << std::endl;
   }
};

class GnuTLS_global_state
{
protected:
   gnutls_dh_params_t dh_params;
   bool glob_init_ok;

   void generate_dh_params();

public:
   GnuTLS_global_state();
   ~GnuTLS_global_state();

   GnuTLS_status status;

   void set_log_level(int level)
   { if (status.good()) { gnutls_global_set_log_level(level); } }

   gnutls_dh_params_t get_dh_params()
   {
      if (!dh_params) { generate_dh_params(); }
      return dh_params;
   }
};

class GnuTLS_session_params
{
protected:
   gnutls_certificate_credentials_t my_cred;
   unsigned int my_flags;

public:
   GnuTLS_global_state &state;
   GnuTLS_status status;

   GnuTLS_session_params(GnuTLS_global_state &state,
                         const char *pubkey_file,
                         const char *privkey_file,
                         const char *trustedkeys_file,
                         unsigned int flags);
   ~GnuTLS_session_params()
   {
      if (my_cred) { gnutls_certificate_free_credentials(my_cred); }
   }

   gnutls_certificate_credentials_t get_cred() const { return my_cred; }
   unsigned int get_flags() const { return my_flags; }
};

class GnuTLS_socketbuf : public socketbuf
{
protected:
   GnuTLS_status status;
   gnutls_session_t session;
   bool session_started;

   const GnuTLS_session_params &params;
   gnutls_certificate_credentials_t my_cred; // same as params.my_cred

   void handshake();
   void start_session();
   void end_session();

public:
   GnuTLS_socketbuf(const GnuTLS_session_params &p)
      : session_started(false), params(p), my_cred(params.get_cred())
   { status.set_result(params.status.get_result()); }

   virtual ~GnuTLS_socketbuf() { close(); }

   bool gnutls_good() const { return status.good(); }

   /** Attach a new socket descriptor to the socketbuf. Returns the old socket
       descriptor which is NOT closed. */
   int attach(int sd) override;

   int open(const char hostname[], int port) override;

   int close() override;

protected:
   int sync() override;

   int_type underflow() override;

   // Same as in the base class:
   // virtual int_type overflow(int_type c = traits_type::eof());

   std::streamsize xsgetn(char_type *s__, std::streamsize n__) override;

   std::streamsize xsputn(const char_type *s__, std::streamsize n__) override;
};

#endif // MFEM_USE_GNUTLS

class socketstream : public std::iostream
{
protected:
   socketbuf *buf__;
   bool glvis_client;

   void set_socket(bool secure);
   inline void check_secure_socket();
#ifdef MFEM_USE_GNUTLS
   static int num_glvis_sockets;
   static GnuTLS_global_state *state;
   static GnuTLS_session_params *params;
   static GnuTLS_session_params &add_socket();
   static void remove_socket();
   inline void set_secure_socket(const GnuTLS_session_params &p);
#endif

public:
#ifdef MFEM_USE_GNUTLS
   static const bool secure_default = true;
#else
   static const bool secure_default = false;
#endif

   /** @brief Create a socket stream without connecting to a host.

       If 'secure' is true, (GnuTLS support must be enabled) then the connection
       will use GLVis client session keys from ~/.config/glvis/client for GnuTLS
       identification. If you want to use other GnuTLS session keys or
       parameters, use the constructor from GnuTLS_session_params. */
   socketstream(bool secure = secure_default);

   /** @brief Create a socket stream associated with the given socket buffer.
       The new object takes ownership of 'buf'. */
   explicit socketstream(socketbuf *buf)
      : std::iostream(buf), buf__(buf), glvis_client(false) { }

   /** @brief Create a socket stream and associate it with the given socket
       descriptor 's'. The treatment of the 'secure' flag is similar to that in
       the default constructor. */
   explicit socketstream(int s, bool secure = secure_default);

   /** @brief Create a socket stream and connect to the given host and port.
       The treatment of the 'secure' flag is similar to that in the default
       constructor. */
   socketstream(const char hostname[], int port, bool secure = secure_default)
      : std::iostream(0) { set_socket(secure); open(hostname, port); }

#ifdef MFEM_USE_GNUTLS
   /// Create a secure socket stream using the given GnuTLS_session_params.
   explicit socketstream(const GnuTLS_session_params &p);
#endif

   socketbuf *rdbuf() { return buf__; }

   /// Open the socket stream on 'port' at 'hostname'.
   int open(const char hostname[], int port);

   /// Close the socketstream.
   int close() { return buf__->close(); }

   /// True if the socketstream is open, false otherwise.
   bool is_open() { return buf__->is_open(); }

   virtual ~socketstream();
};


class socketserver
{
private:
   int listen_socket;

public:
   explicit socketserver(int port, int backlog=4);

   bool good() { return (listen_socket >= 0); }

   int close();

   int accept();

   int accept(socketstream &sockstr);

   ~socketserver() { close(); }
};

} // namespace mfem

#endif
