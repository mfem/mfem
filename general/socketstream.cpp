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

#ifdef _WIN32
// Turn off CRT deprecation warnings for strerror
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "socketstream.hpp"

#include <cstring>      // memset, memcpy, strerror
#include <cerrno>       // errno
#ifndef _WIN32
#include <netdb.h>      // getaddrinfo
#include <arpa/inet.h>  // htons
#include <sys/types.h>  // socket, setsockopt, connect, recv, send
#include <sys/socket.h> // socket, setsockopt, connect, recv, send
#include <unistd.h>     // close
#define closesocket (::close)
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#ifdef _MSC_VER
typedef int ssize_t;
typedef int socklen_t;
// Link with ws2_32.lib
#pragma comment(lib, "ws2_32.lib")
#endif
#endif

#ifdef MFEM_USE_GNUTLS
#ifndef MFEM_USE_GNUTLS_X509
#include <gnutls/openpgp.h>
#endif
// Enable debug messages from GnuTLS_* classes
// #define MFEM_USE_GNUTLS_DEBUG
#endif

namespace mfem
{

// Helper class for handling Winsock initialization calls.
class WinsockWrapper
{
public:
   WinsockWrapper()
   {
#ifdef _WIN32
      WSADATA wsaData;
      int err_flag = WSAStartup(MAKEWORD(2,2), &wsaData);
      if (err_flag != 0)
      {
         mfem::err << "Error occurred during initialization of WinSock."
                   << std::endl;
         return;
      }
#endif
      initialized = true;
   }

#ifdef _WIN32
   ~WinsockWrapper() { WSACleanup(); }
#endif

   WinsockWrapper(const WinsockWrapper&) = delete;
   WinsockWrapper& operator=(const WinsockWrapper&) = delete;

   bool Initialized() { return initialized; }
private:
   bool initialized = false;
};

// If available, Winsock is initialized when this object is constructed.
static WinsockWrapper wsInit_;

int socketbuf::attach(int sd)
{
   int old_sd = socket_descriptor;
   pubsync();
   socket_descriptor = sd;
   setg(NULL, NULL, NULL);
   setp(obuf, obuf + buflen);
   return old_sd;
}

int socketbuf::open(const char hostname[], int port)
{
   struct addrinfo     hints, *res, *rp;

   if (!wsInit_.Initialized())
   {
      mfem_error("Attempting to open socket, but Winsock not initialized.");
   }

   close();
   setg(NULL, NULL, NULL);
   setp(obuf, obuf + buflen);

   hints.ai_family = AF_UNSPEC;
   hints.ai_socktype = SOCK_STREAM;
   hints.ai_flags = 0;
   hints.ai_protocol = 0;
   // On Windows, the following need to be set to 0; also required by POSIX.
   hints.ai_addrlen = 0;
   hints.ai_canonname = NULL;
   hints.ai_addr = NULL;
   hints.ai_next = NULL;

   std::string portStr = std::to_string(port);
   int s = getaddrinfo(hostname, portStr.c_str(), &hints, &res);
   if (s != 0)
   {
#ifdef MFEM_DEBUG
      mfem::err << "Error in getaddrinfo(): code = " << s << std::endl;
#endif
      socket_descriptor = -3;
      return -1;
   }

   for (rp = res; rp != NULL; rp = rp->ai_next)
   {
      socket_descriptor = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if (socket_descriptor < 0)
      {
         continue;
      }

#if defined __APPLE__
      // OS X does not support the MSG_NOSIGNAL option of send().
      // Instead we can use the SO_NOSIGPIPE socket option.
      int on = 1;
      if (setsockopt(socket_descriptor, SOL_SOCKET, SO_NOSIGPIPE,
                     &on, sizeof(on)) < 0)
      {
         closesocket(socket_descriptor);
         socket_descriptor = -2;
         continue;
      }
#endif

      if (connect(socket_descriptor, rp->ai_addr,
                  static_cast<socklen_t>(rp->ai_addrlen)) < 0)
      {
         closesocket(socket_descriptor);
         socket_descriptor = -2;
         continue;
      }
      break;
   }

   freeaddrinfo(res);
   return (socket_descriptor < 0) ? -1 : 0;
}

int socketbuf::close()
{
   if (is_open())
   {
      pubsync();
      int err_flag = closesocket(socket_descriptor);
      socket_descriptor = -1;
      return err_flag;
   }
   return 0;
}

int socketbuf::sync()
{
   ssize_t bw, n = pptr() - pbase();
   // mfem::out << "[socketbuf::sync n=" << n << ']' << std::endl;
   while (n > 0)
   {
#ifdef MSG_NOSIGNAL
      bw = send(socket_descriptor, pptr() - n, n, MSG_NOSIGNAL);
#else
      bw = send(socket_descriptor, pptr() - n, n, 0);
#endif
      if (bw < 0)
      {
#ifdef MFEM_DEBUG
         mfem::err << "Error in send(): " << strerror(errno) << std::endl;
#endif
         setp(pptr() - n, obuf + buflen);
         pbump(n);
         return -1;
      }
      n -= bw;
   }
   setp(obuf, obuf + buflen);
   return 0;
}

socketbuf::int_type socketbuf::underflow()
{
   // assuming (gptr() < egptr()) is false
   ssize_t br = recv(socket_descriptor, ibuf, buflen, 0);
   // mfem::out << "[socketbuf::underflow br=" << br << ']'
   //           << std::endl;
   if (br <= 0)
   {
#ifdef MFEM_DEBUG
      if (br < 0)
      {
         mfem::err << "Error in recv(): " << strerror(errno) << std::endl;
      }
#endif
      setg(NULL, NULL, NULL);
      return traits_type::eof();
   }
   setg(ibuf, ibuf, ibuf + br);
   return traits_type::to_int_type(*ibuf);
}

socketbuf::int_type socketbuf::overflow(int_type c)
{
   if (sync() < 0)
   {
      return traits_type::eof();
   }
   if (traits_type::eq_int_type(c, traits_type::eof()))
   {
      return traits_type::not_eof(c);
   }
   *pptr() = traits_type::to_char_type(c);
   pbump(1);
   return c;
}

std::streamsize socketbuf::xsgetn(char_type *s__, std::streamsize n__)
{
   // mfem::out << "[socketbuf::xsgetn n__=" << n__ << ']'
   //           << std::endl;
   const std::streamsize bn = egptr() - gptr();
   if (n__ <= bn)
   {
      traits_type::copy(s__, gptr(), n__);
      gbump(n__);
      return n__;
   }
   traits_type::copy(s__, gptr(), bn);
   setg(NULL, NULL, NULL);
   std::streamsize remain = n__ - bn;
   char_type *end = s__ + n__;
   ssize_t br;
   while (remain > 0)
   {
      br = recv(socket_descriptor, end - remain, remain, 0);
      if (br <= 0)
      {
#ifdef MFEM_DEBUG
         if (br < 0)
         {
            mfem::err << "Error in recv(): " << strerror(errno) << std::endl;
         }
#endif
         return (n__ - remain);
      }
      remain -= br;
   }
   return n__;
}

std::streamsize socketbuf::xsputn(const char_type *s__, std::streamsize n__)
{
   // mfem::out << "[socketbuf::xsputn n__=" << n__ << ']'
   //           << std::endl;
   if (pptr() + n__ <= epptr())
   {
      traits_type::copy(pptr(), s__, n__);
      pbump(n__);
      return n__;
   }
   if (sync() < 0)
   {
      return 0;
   }
   ssize_t bw;
   std::streamsize remain = n__;
   const char_type *end = s__ + n__;
   while (remain > buflen)
   {
#ifdef MSG_NOSIGNAL
      bw = send(socket_descriptor, end - remain, remain, MSG_NOSIGNAL);
#else
      bw = send(socket_descriptor, end - remain, remain, 0);
#endif
      if (bw < 0)
      {
#ifdef MFEM_DEBUG
         mfem::err << "Error in send(): " << strerror(errno) << std::endl;
#endif
         return (n__ - remain);
      }
      remain -= bw;
   }
   if (remain > 0)
   {
      traits_type::copy(pptr(), end - remain, remain);
      pbump(remain);
   }
   return n__;
}


socketserver::socketserver(int port, int backlog)
{
   listen_socket = socket(PF_INET, SOCK_STREAM, 0); // tcp socket
   if (listen_socket < 0)
   {
      return;
   }
   int on = 1;
   if (setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR,
                  (char *)(&on), sizeof(on)) < 0)
   {
      closesocket(listen_socket);
      listen_socket = -2;
      return;
   }
   struct sockaddr_in sa;
   memset(&sa, 0, sizeof(sa));
   sa.sin_family = AF_INET;
   sa.sin_port = htons(port);
   sa.sin_addr.s_addr = INADDR_ANY;
   if (bind(listen_socket, (const struct sockaddr *)&sa, sizeof(sa)))
   {
      closesocket(listen_socket);
      listen_socket = -3;
      return;
   }

   if (listen(listen_socket, backlog) < 0)
   {
      closesocket(listen_socket);
      listen_socket = -4;
      return;
   }
}

int socketserver::close()
{
   if (!good())
   {
      return 0;
   }
   int err_flag = closesocket(listen_socket);
   listen_socket = -1;
   return err_flag;
}

int socketserver::accept()
{
   return good() ? ::accept(listen_socket, NULL, NULL) : -1;
}

int socketserver::accept(socketstream &sockstr)
{
   if (!good())
   {
      return -1;
   }
   int socketd = ::accept(listen_socket, NULL, NULL);
   if (socketd >= 0)
   {
      sockstr.rdbuf()->close();
      sockstr.rdbuf()->attach(socketd);
      return sockstr.rdbuf()->getsocketdescriptor();
   }
   return socketd;
}

#ifdef MFEM_USE_GNUTLS

static void mfem_gnutls_log_func(int level, const char *str)
{
   mfem::out << "GnuTLS <" << level << "> " << str << std::flush;
}

GnuTLS_global_state::GnuTLS_global_state()
{
   status.set_result(gnutls_global_init());
   status.print_on_error("gnutls_global_init");
   glob_init_ok = status.good();

   if (status.good())
   {
      gnutls_global_set_log_function(mfem_gnutls_log_func);
   }

   dh_params = NULL;
}

GnuTLS_global_state::~GnuTLS_global_state()
{
   gnutls_dh_params_deinit(dh_params);
   if (glob_init_ok) { gnutls_global_deinit(); }
}

void GnuTLS_global_state::generate_dh_params()
{
   if (status.good())
   {
      status.set_result(gnutls_dh_params_init(&dh_params));
      status.print_on_error("gnutls_dh_params_init");
      if (!status.good()) { dh_params = NULL; }
      else
      {
#if GNUTLS_VERSION_NUMBER >= 0x021200
         unsigned bits =
            gnutls_sec_param_to_pk_bits(
               GNUTLS_PK_DH, GNUTLS_SEC_PARAM_LEGACY);
#else
         unsigned bits = 1024;
#endif
         mfem::out << "Generating DH params (" << bits << " bits) ..."
                   << std::flush;
         status.set_result(gnutls_dh_params_generate2(dh_params, bits));
         mfem::out << " done." << std::endl;
         status.print_on_error("gnutls_dh_params_generate2");
         if (!status.good())
         {
            gnutls_dh_params_deinit(dh_params);
            dh_params = NULL;
         }
      }
   }
}

static int mfem_gnutls_verify_callback(gnutls_session_t session)
{
   unsigned int status;
#if GNUTLS_VERSION_NUMBER >= 0x030104
   const char *hostname = (const char *) gnutls_session_get_ptr(session);
   int ret = gnutls_certificate_verify_peers3(session, hostname, &status);
   if (ret < 0)
   {
      mfem::err << "Error in gnutls_certificate_verify_peers3:"
                << gnutls_strerror(ret) << std::endl;
      return GNUTLS_E_CERTIFICATE_ERROR;
   }

#ifdef MFEM_DEBUG
   gnutls_datum_t status_str;
   gnutls_certificate_type_t type = gnutls_certificate_type_get(session);
   ret = gnutls_certificate_verification_status_print(
            status, type, &status_str, 0);
   if (ret < 0)
   {
      mfem::err << "Error in gnutls_certificate_verification_status_print:"
                << gnutls_strerror(ret) << std::endl;
      return GNUTLS_E_CERTIFICATE_ERROR;
   }
   mfem::out << status_str.data << std::endl;
   gnutls_free(status_str.data);
#endif
#else // --> GNUTLS_VERSION_NUMBER < 0x030104
   int ret = gnutls_certificate_verify_peers2(session, &status);
   if (ret < 0)
   {
      mfem::err << "Error in gnutls_certificate_verify_peers2:"
                << gnutls_strerror(ret) << std::endl;
      return GNUTLS_E_CERTIFICATE_ERROR;
   }
#ifdef MFEM_DEBUG
   mfem::out << (status ?
                 "The certificate is NOT trusted." :
                 "The certificate is trusted.") << std::endl;
#endif
#endif

   return status ? GNUTLS_E_CERTIFICATE_ERROR : 0;
}

GnuTLS_session_params::GnuTLS_session_params(
   GnuTLS_global_state &state, const char *pubkey_file,
   const char *privkey_file, const char *trustedkeys_file, unsigned int flags)
   : state(state)
{
   status.set_result(state.status.get_result());
   my_flags = status.good() ? flags : 0;

   // allocate my_cred
   if (status.good())
   {
      status.set_result(
         gnutls_certificate_allocate_credentials(&my_cred));
      status.print_on_error("gnutls_certificate_allocate_credentials");
   }
   if (!status.good()) { my_cred = NULL; }
   else
   {
#ifndef MFEM_USE_GNUTLS_X509
      status.set_result(
         gnutls_certificate_set_openpgp_key_file(
            my_cred, pubkey_file, privkey_file, GNUTLS_OPENPGP_FMT_RAW));
      status.print_on_error("gnutls_certificate_set_openpgp_key_file");
#else
      status.set_result(
         gnutls_certificate_set_x509_key_file(
            my_cred, pubkey_file, privkey_file, GNUTLS_X509_FMT_PEM));
      // e.g. pubkey_file, privkey_file == "cert.pem", "key.pem"
      status.print_on_error("gnutls_certificate_set_x509_key_file");
#endif
   }

   if (status.good())
   {
      /*
      gnutls_certificate_set_pin_function(
         my_cred,
         (gnutls_pin_callback_t) fn,
         (void *) userdata);
      */
   }

   if (status.good())
   {
#ifndef MFEM_USE_GNUTLS_X509
      status.set_result(
         gnutls_certificate_set_openpgp_keyring_file(
            my_cred, trustedkeys_file, GNUTLS_OPENPGP_FMT_RAW));
      status.print_on_error("gnutls_certificate_set_openpgp_keyring_file");
#else
      int num_certs =
         gnutls_certificate_set_x509_trust_file(
            my_cred, trustedkeys_file, GNUTLS_X509_FMT_PEM);
      // e.g. trustedkeys_file == "trusted-certs.pem"
#ifdef MFEM_USE_GNUTLS_DEBUG
      mfem::out << "[GnuTLS_session_params::GnuTLS_session_params] "
                "number of trusted certificates = " << num_certs << std::endl;
#endif
      status.set_result(num_certs > 0 ?
                        GNUTLS_E_SUCCESS : GNUTLS_E_CERTIFICATE_ERROR);
      status.print_on_error("gnutls_certificate_set_x509_trust_file");
#endif
   }

#if GNUTLS_VERSION_NUMBER >= 0x021000
   if (status.good())
   {
      gnutls_certificate_set_verify_function(
         my_cred, mfem_gnutls_verify_callback);
   }
#endif

   if (status.good() && (flags & GNUTLS_SERVER))
   {
      gnutls_dh_params_t dh_params = state.get_dh_params();
      status.set_result(state.status.get_result());
      if (status.good())
      {
         gnutls_certificate_set_dh_params(my_cred, dh_params);
      }
   }
}

void GnuTLS_socketbuf::handshake()
{
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::handshake]" << std::endl;
#endif

   // Called at the end of start_session.
   int err_flag;
   do
   {
      err_flag = gnutls_handshake(session);
      status.set_result(err_flag);
      if (status.good())
      {
#if 0
         mfem::out << "handshake successful, TLS version is "
                   << gnutls_protocol_get_name(
                      gnutls_protocol_get_version(session)) << std::endl;
#endif
         return;
      }
   }
   while (err_flag == GNUTLS_E_INTERRUPTED || err_flag == GNUTLS_E_AGAIN);
#ifdef MFEM_DEBUG
   status.print_on_error("gnutls_handshake");
#endif
}

#if (defined(MSG_NOSIGNAL) && !defined(_WIN32) && !defined(__APPLE__))
#define MFEM_USE_GNUTLS_PUSH_FUNCTION

static ssize_t mfem_gnutls_push_function(
   gnutls_transport_ptr_t fd_ptr, const void *data, size_t datasize)
{
   return send((int)(long)fd_ptr, data, datasize, MSG_NOSIGNAL);
}
#endif

void GnuTLS_socketbuf::start_session()
{
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::start_session]" << std::endl;
#endif

   // check for valid 'socket_descriptor' and inactive session
   if (!is_open() || session_started) { return; }

   status.set_result(params.status.get_result());
   if (status.good())
   {
#if GNUTLS_VERSION_NUMBER >= 0x030102
      status.set_result(gnutls_init(&session, params.get_flags()));
#else
      status.set_result(
         gnutls_init(&session, (gnutls_connection_end_t) params.get_flags()));
#endif
      status.print_on_error("gnutls_init");
   }

   session_started = status.good();
   if (status.good())
   {
      const char *priorities;
      // what is the right version here?
      if (gnutls_check_version("2.12.0") != NULL)
      {
         // This works for version 2.12.23 (0x020c17) and above
#ifndef MFEM_USE_GNUTLS_X509
         priorities = "NONE:+VERS-TLS1.2:+CIPHER-ALL:+MAC-ALL:+SIGN-ALL:"
                      "+COMP-ALL:+KX-ALL:+CTYPE-OPENPGP:+CURVE-ALL";
#else
         priorities = "NONE:+VERS-TLS1.2:+CIPHER-ALL:+MAC-ALL:+SIGN-ALL:"
                      "+COMP-ALL:+KX-ALL:+CTYPE-X509:+CURVE-ALL";
#endif
      }
      else
      {
         // This works for version 2.8.5 (0x020805) and below
#ifndef MFEM_USE_GNUTLS_X509
         priorities = "NORMAL:-CTYPE-X.509";
#else
         priorities = "NORMAL:";
#endif
      }
      const char *err_ptr;
      status.set_result(
         gnutls_priority_set_direct(session, priorities, &err_ptr));
      status.print_on_error("gnutls_priority_set_direct");
      if (!status.good())
      {
         mfem::err << "Error ptr = \"" << err_ptr << '"' << std::endl;
      }
   }

   if (status.good())
   {
      // set session credentials
      status.set_result(
         gnutls_credentials_set(
            session, GNUTLS_CRD_CERTIFICATE, my_cred));
      status.print_on_error("gnutls_credentials_set");
   }

   if (status.good())
   {
      const char *hostname = NULL; // no hostname verification
      gnutls_session_set_ptr(session, (void*)hostname);
      if (params.get_flags() & GNUTLS_SERVER)
      {
         // require clients to send certificate:
         gnutls_certificate_server_set_request(session, GNUTLS_CERT_REQUIRE);
      }
#if GNUTLS_VERSION_NUMBER >= 0x030100
      gnutls_handshake_set_timeout(
         session, GNUTLS_DEFAULT_HANDSHAKE_TIMEOUT);
#endif
   }

   if (status.good())
   {
#if GNUTLS_VERSION_NUMBER >= 0x030109
      gnutls_transport_set_int(session, socket_descriptor);
#else
      gnutls_transport_set_ptr(session,
                               (gnutls_transport_ptr_t) socket_descriptor);
#endif

      handshake();
   }

#if GNUTLS_VERSION_NUMBER < 0x021000
   if (status.good())
   {
      status.set_result(mfem_gnutls_verify_callback(session));
      if (!status.good())
      {
         int err_flag;
         do
         {
            // Close the connection without waiting for close reply, i.e. we
            // use GNUTLS_SHUT_WR.
            err_flag = gnutls_bye(session, GNUTLS_SHUT_WR);
         }
         while (err_flag == GNUTLS_E_AGAIN || err_flag == GNUTLS_E_INTERRUPTED);
      }
   }
#endif

#ifdef MFEM_USE_GNUTLS_PUSH_FUNCTION
   if (status.good())
   {
      gnutls_transport_set_push_function(session, mfem_gnutls_push_function);
   }
#endif

   if (!status.good())
   {
      if (session_started) { gnutls_deinit(session); }
      session_started = false;
      socketbuf::close();
   }
}

void GnuTLS_socketbuf::end_session()
{
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::end_session]" << std::endl;
#endif

   // check for valid 'socket_descriptor'
   if (!session_started) { return; }

   if (is_open() && status.good())
   {
      pubsync();
#ifdef MFEM_USE_GNUTLS_DEBUG
      mfem::out << "[GnuTLS_socketbuf::end_session: gnutls_bye]" << std::endl;
#endif
      int err_flag;
      do
      {
         // err_flag = gnutls_bye(session, GNUTLS_SHUT_RDWR);
         err_flag = gnutls_bye(session,
                               GNUTLS_SHUT_WR); // does not wait for reply
         status.set_result(err_flag);
      }
      while (err_flag == GNUTLS_E_AGAIN || err_flag == GNUTLS_E_INTERRUPTED);
      status.print_on_error("gnutls_bye");
   }

   gnutls_deinit(session);
   session_started = false;
}

int GnuTLS_socketbuf::attach(int sd)
{
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::attach]" << std::endl;
#endif

   end_session();

   int old_sd = socketbuf::attach(sd);

   start_session();

   return old_sd;
}

int GnuTLS_socketbuf::open(const char hostname[], int port)
{
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::open]" << std::endl;
#endif

   int err_flag = socketbuf::open(hostname, port); // calls close()
   if (err_flag) { return err_flag; }

   start_session();

   return status.good() ? 0 : -100;
}

int GnuTLS_socketbuf::close()
{
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::close]" << std::endl;
#endif

   end_session();

   int err_flag = socketbuf::close();

   return status.good() ? err_flag : -100;
}

int GnuTLS_socketbuf::sync()
{
   ssize_t bw, n = pptr() - pbase();
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::sync n=" << n << ']' << std::endl;
#endif
   if (!session_started || !status.good()) { return -1; }
   while (n > 0)
   {
      bw = gnutls_record_send(session, pptr() - n, n);
      if (bw == GNUTLS_E_INTERRUPTED || bw == GNUTLS_E_AGAIN) { continue; }
      if (bw < 0)
      {
         status.set_result((int)bw);
#ifdef MFEM_DEBUG
         status.print_on_error("gnutls_record_send");
#endif
         setp(pptr() - n, obuf + buflen);
         pbump(n);
         return -1;
      }
      n -= bw;
   }
   setp(obuf, obuf + buflen);
   return 0;
}

GnuTLS_socketbuf::int_type GnuTLS_socketbuf::underflow()
{
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::underflow ...]" << std::endl;
#endif
   if (!session_started || !status.good()) { return traits_type::eof(); }

   ssize_t br;
   do
   {
      br = gnutls_record_recv(session, ibuf, buflen);
      if (br == GNUTLS_E_REHANDSHAKE)
      {
         continue; // TODO: replace with re-handshake
      }
   }
   while (br == GNUTLS_E_INTERRUPTED || br == GNUTLS_E_AGAIN);
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::underflow br=" << br << ']' << std::endl;
#endif

   if (br <= 0)
   {
      if (br < 0)
      {
         status.set_result((int)br);
#ifdef MFEM_DEBUG
         status.print_on_error("gnutls_record_recv");
#endif
      }
      setg(NULL, NULL, NULL);
      return traits_type::eof();
   }
   setg(ibuf, ibuf, ibuf + br);
   return traits_type::to_int_type(*ibuf);
}

std::streamsize GnuTLS_socketbuf::xsgetn(char_type *s__, std::streamsize n__)
{
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::xsgetn n__=" << n__ << ']' << std::endl;
#endif
   if (!session_started || !status.good()) { return 0; }

   const std::streamsize bn = egptr() - gptr();
   if (n__ <= bn)
   {
      traits_type::copy(s__, gptr(), n__);
      gbump(n__);
      return n__;
   }
   traits_type::copy(s__, gptr(), bn);
   setg(NULL, NULL, NULL);
   std::streamsize remain = n__ - bn;
   char_type *end = s__ + n__;
   ssize_t br;
   while (remain > 0)
   {
      do
      {
         br = gnutls_record_recv(session, end - remain, remain);
         if (br == GNUTLS_E_REHANDSHAKE)
         {
            continue; // TODO: replace with re-handshake
         }
      }
      while (br == GNUTLS_E_INTERRUPTED || br == GNUTLS_E_AGAIN);
      if (br <= 0)
      {
         if (br < 0)
         {
            status.set_result((int)br);
#ifdef MFEM_DEBUG
            status.print_on_error("gnutls_record_recv");
#endif
         }
         return (n__ - remain);
      }
      remain -= br;
   }
   return n__;
}

std::streamsize GnuTLS_socketbuf::xsputn(const char_type *s__,
                                         std::streamsize n__)
{
#ifdef MFEM_USE_GNUTLS_DEBUG
   mfem::out << "[GnuTLS_socketbuf::xsputn n__=" << n__ << ']' << std::endl;
#endif
   if (!session_started || !status.good()) { return 0; }

   if (pptr() + n__ <= epptr())
   {
      traits_type::copy(pptr(), s__, n__);
      pbump(n__);
      return n__;
   }
   if (sync() < 0)
   {
      return 0;
   }
   ssize_t bw;
   std::streamsize remain = n__;
   const char_type *end = s__ + n__;
   while (remain > buflen)
   {
      bw = gnutls_record_send(session, end - remain, remain);
      if (bw == GNUTLS_E_INTERRUPTED || bw == GNUTLS_E_AGAIN) { continue; }
#ifdef MFEM_USE_GNUTLS_DEBUG
      mfem::out << "[GnuTLS_socketbuf::xsputn bw=" << bw << ']' << std::endl;
#endif
      if (bw < 0)
      {
         status.set_result((int)bw);
#ifdef MFEM_DEBUG
         status.print_on_error("gnutls_record_send");
#endif
         return (n__ - remain);
      }
      remain -= bw;
   }
   if (remain > 0)
   {
      traits_type::copy(pptr(), end - remain, remain);
      pbump(remain);
   }
   return n__;
}


int socketstream::num_glvis_sockets = 0;
GnuTLS_global_state *socketstream::state = NULL;
GnuTLS_session_params *socketstream::params = NULL;

// static method
GnuTLS_session_params &socketstream::add_socket()
{
   if (num_glvis_sockets == 0)
   {
      state = new GnuTLS_global_state;
      // state->set_log_level(1000);
      std::string home_dir(GetEnv("HOME"));
      std::string client_dir = home_dir + "/.config/glvis/client/";
#ifndef MFEM_USE_GNUTLS_X509
      std::string pubkey  = client_dir + "pubring.gpg";
      std::string privkey = client_dir + "secring.gpg";
      std::string trustedkeys = client_dir + "trusted-servers.gpg";
#else
      std::string pubkey  = client_dir + "cert.pem";
      std::string privkey = client_dir + "key.pem";
      std::string trustedkeys = client_dir + "trusted-servers.pem";
#endif
      params = new GnuTLS_session_params(
         *state, pubkey.c_str(), privkey.c_str(), trustedkeys.c_str(),
         GNUTLS_CLIENT);
      if (!params->status.good())
      {
         mfem::err << "  public key   = " << pubkey << '\n'
                   << "  private key  = " << privkey << '\n'
                   << "  trusted keys = " << trustedkeys << std::endl;
         mfem::err << "Error setting GLVis client parameters.\n"
                   "Use the following GLVis script to create your GLVis keys:\n"
                   "   bash glvis-keygen.sh [\"Your Name\"] [\"Your Email\"]"
                   << std::endl;
      }
   }
   num_glvis_sockets++;
   return *params;
}

// static method
void socketstream::remove_socket()
{
   if (num_glvis_sockets > 0)
   {
      num_glvis_sockets--;
      if (num_glvis_sockets == 0)
      {
         delete params; params = NULL;
         delete state; state = NULL;
      }
   }
}

inline void socketstream::set_secure_socket(const GnuTLS_session_params &p)
{
   buf__ = new GnuTLS_socketbuf(p);
   std::iostream::rdbuf(buf__);
}

socketstream::socketstream(const GnuTLS_session_params &p)
   : std::iostream(0), glvis_client(false)
{
   set_secure_socket(p);
   check_secure_socket();
}

#endif // MFEM_USE_GNUTLS

void socketstream::set_socket(bool secure)
{
   glvis_client = secure;
   if (secure)
   {
#ifdef MFEM_USE_GNUTLS
      set_secure_socket(add_socket());
#else
      mfem_error("The secure option in class mfem::socketstream can only\n"
                 "be used when GnuTLS support is enabled.");
#endif
   }
   else
   {
      buf__ = new socketbuf;
      std::iostream::rdbuf(buf__);
   }
}

inline void socketstream::check_secure_socket()
{
#ifdef MFEM_USE_GNUTLS
   if (((GnuTLS_socketbuf*)buf__)->gnutls_good()) { clear(); }
   else { setstate(std::ios::failbit); }
#endif
}

socketstream::socketstream(bool secure) : std::iostream(0)
{
   set_socket(secure);
   if (secure) { check_secure_socket(); }
}

socketstream::socketstream(int s, bool secure) : std::iostream(0)
{
   set_socket(secure);
   buf__->attach(s);
   if (secure) { check_secure_socket(); }
}

int socketstream::open(const char hostname[], int port)
{
   int err_flag = buf__->open(hostname, port);
   if (err_flag)
   {
      setstate(std::ios::failbit);
   }
   else
   {
      clear();
   }
   return err_flag;
}

socketstream::~socketstream()
{
   delete buf__;
#ifdef MFEM_USE_GNUTLS
   if (glvis_client) { remove_socket(); }
#endif
}

} // namespace mfem
