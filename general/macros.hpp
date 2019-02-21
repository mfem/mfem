// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_FOR_EACH_MACROS_HPP
#define MFEM_FOR_EACH_MACROS_HPP

// *****************************************************************************
#define VOID()
#define DROP(...)
#define CAT_ARGS(a, ...) a ## __VA_ARGS__
#define CAT(a, ...) CAT_ARGS(a, __VA_ARGS__)

// *****************************************************************************
// * Deferred __VA_ARGS__ evaluation
// *****************************************************************************
#define DEFER(...) __VA_ARGS__ VOID()
#define DEFER2(...) __VA_ARGS__ DEFER(VOID) ()
#define DEFER3(...) __VA_ARGS__ DEFER2(VOID) ()
#define DEFER4(...) __VA_ARGS__ DEFER3(VOID) ()

// *****************************************************************************
// * __VA_ARGS__ evaluation stack
// *****************************************************************************
#define EVAL1(...) __VA_ARGS__
#define EVAL2(...) EVAL1(EVAL1(__VA_ARGS__))
#define EVAL3(...) EVAL2(EVAL2(__VA_ARGS__))
#define EVAL4(...) EVAL3(EVAL3(__VA_ARGS__))
#define EVAL(...) EVAL4(__VA_ARGS__)

// *****************************************************************************
// * Macros existence test
// *****************************************************************************
#define DOES_NOT_EXIST 0
#define DOES_EXISTS(...) 1
#define EXIST(x) CAT(DOES_, x)

#define EXPAND_IF_EXISTS(...) EXPAND, EXISTS(__VA_ARGS__) ) DROP (
#define GET_IF_EXIST_RESULT(x) (CAT(EXPAND_IF_,x), NOT_EXIST)
#define GET_IF_EXIST_EXPAND(expand, value) value
#define GET_IF_EXIST(x) GET_IF_EXIST_EXPAND  x 
#define IF_EXISTS(x) GET_IF_EXIST(GET_IF_EXIST_RESULT(x))

// *****************************************************************************
// * IF tests
// *****************************************************************************
#define IF_1(true, ...) true
#define IF_0(true, ...) __VA_ARGS__
#define IF(value) CAT(IF_, value)

// *****************************************************************************
#define EXTRACT_VALUE_EXISTS(...) __VA_ARGS__
#define EXTRACT_VALUE(value) CAT(EXTRACT_VALUE_, value)
#define EXTRACT_EXISTS(value, ...)                                      \
   IF (EXIST(IF_EXISTS(value))) (EXTRACT_VALUE(value), __VA_ARGS__ )

// *****************************************************************************
#define NOT_0 EXISTS(1)
#define NOT(x) EXTRACT_EXISTS (CAT(NOT_,x),0)

// *****************************************************************************
// * List manipulation
// *****************************************************************************
#define HEAD(x, ...) x
#define TAIL(x, ...) __VA_ARGS__

// *****************************************************************************
#define IS_LIST_VOID(...) EXTRACT_EXISTS(DEFER(HEAD) (__VA_ARGS__ EXISTS(1)), 0)
#define IS_LIST_NOT_VOID(...) NOT(IS_LIST_VOID(__VA_ARGS__))

// *****************************************************************************
#define ENCLOSE(...) ( __VA_ARGS__ )
#define REM_ENCLOSE_(...) __VA_ARGS__
#define REM_ENCLOSE(...) REM_ENCLOSE_ __VA_ARGS__
   
// *****************************************************************************
#define IS_ENCLOSED_TEST(...) EXISTS(1)
#define IS_ENCLOSED(x, ...) EXTRACT_EXISTS ( IS_ENCLOSED_TEST x, 0 )
#define IF_ENCLOSED(...) CAT(IF_, IS_ENCLOSED(__VA_ARGS__))
#define ENCLOSED(...)                                                   \
   IF_ENCLOSED (__VA_ARGS__) ( REM_ENCLOSE(__VA_ARGS__), __VA_ARGS__ )

// *****************************************************************************
// * FOR_EACH_1D
// *****************************************************************************
#define FOR_EACH_INDIRECT() FOR_EACH_NO_EVAL
#define FOR_EACH_NO_EVAL(f,expr, ...)                                   \
   IF (IS_LIST_NOT_VOID( __VA_ARGS__ ))                                 \
   (expr(f, ENCLOSED(HEAD(__VA_ARGS__)))                                \
    DEFER2 ( FOR_EACH_INDIRECT ) () (f, expr, TAIL(__VA_ARGS__)))
#define FOR_EACH_1D(expr, f, ...) EVAL(FOR_EACH_NO_EVAL(f,expr, __VA_ARGS__)) 

// *****************************************************************************
// * FOR_EACH_2D
// *****************************************************************************
#define FOR_EACH_2D_INDIRECT() FOR_EACH_2D_
#define FOR_EACH_2D_(f, expr, A, orgA, B, orgB )                        \
   IF ( IS_LIST_NOT_VOID B )                                            \
   ( expr(f,HEAD A, HEAD B )                                            \
     DEFER2(FOR_EACH_2D_INDIRECT) ()                                    \
     (f, expr, A, A, (TAIL B), orgB ),                                  \
     IF (IS_LIST_NOT_VOID( TAIL A ) )                                   \
     (DEFER3(FOR_EACH_2D_INDIRECT) ()                                   \
      (f, expr, (TAIL orgA), (TAIL orgA), orgB, orgB )))
#define FOR_EACH_2D_NO_EVAL(f,expr, A, B) FOR_EACH_2D_(f, expr, A, A, B, B)
#define FOR_EACH_2D(expr, f, A, B) EVAL(FOR_EACH_2D_NO_EVAL(f,expr, A, B))

// *****************************************************************************
// * FOR_EACH_3D
// *****************************************************************************
#define FOR_EACH_3D_INDIRECT() FOR_EACH_3D_
#define FOR_EACH_3D_(f, expr, A, orgA, B, orgB, C, orgC )               \
   IF ( IS_LIST_NOT_VOID C )                                            \
   (expr(f, HEAD A, HEAD B, HEAD C )                                    \
    DEFER2(FOR_EACH_3D_INDIRECT) ()                                     \
    (f, expr, A, A, B, B, (TAIL C), orgC ),                             \
    IF (IS_LIST_NOT_VOID( TAIL B ) )                                    \
    (DEFER3(FOR_EACH_3D_INDIRECT) ()                                    \
     (f, expr, orgA, orgA, (TAIL orgB), (TAIL orgB), orgC, orgC )       \
     IF (IS_LIST_NOT_VOID( TAIL A ) )                                   \
     (DEFER4(FOR_EACH_3D_INDIRECT) ()                                   \
      (f, expr, (TAIL orgA), (TAIL orgA), orgB, orgB, orgC, orgC ))))
#define FOR_EACH_3D_NO_EVAL(f,expr, A, B, C)                            \
   FOR_EACH_3D_(f,expr, A, A, B, B, C, C)
#define FOR_EACH_3D(expr, f,A, B, C)                                    \
   EVAL(FOR_EACH_3D_NO_EVAL(f,expr, A, B, C))

// *****************************************************************************
// * Map prefix
// *****************************************************************************
#define MAP_PREFIX_(map,fct_p)                                          \
   static std::unordered_map<unsigned int, fct_p> map = {

// *****************************************************************************
// * Map postfix
// *****************************************************************************
#define MAP_POSTFIX_(map) }

// *****************************************************************************
// * MFEM_TEMPLATES_FOREACH_1D, where the user defined MFEM_TEMPLATES_ID is used
// *****************************************************************************
#define MFEM_MAP_FCT_1D(f,a) {MFEM_TEMPLATES_ID(a), &f<a>},
#define MFEM_TEMPLATES_FOREACH_1D(map,fct_p,fct,...)                    \
   MAP_PREFIX_(map,fct_p)                                               \
   FOR_EACH_1D(MFEM_MAP_FCT_1D,fct,__VA_ARGS__)                         \
   MAP_POSTFIX_(map)

// *****************************************************************************
// * MFEM_TEMPLATES_FOREACH_2D, where the user defined MFEM_TEMPLATES_ID is used
// *****************************************************************************
#define MFEM_MAP_FCT_2D_ID(f,a,b) {MFEM_TEMPLATES_ID(a,b), &f<a,b>},
#define MFEM_TEMPLATES_FOREACH_2D_ID(map,fct_p,fct,...)                 \
   MAP_PREFIX_(map,fct_p)                                               \
   FOR_EACH_2D(MFEM_MAP_FCT_2D_ID,fct,__VA_ARGS__)                      \
   MAP_POSTFIX_(map)

// *****************************************************************************
// * MFEM_TEMPLATES_FOREACH_2D
// *****************************************************************************
#define MFEM_TEMPLATES_2D(a,b) ((a*100) + (b))
#define MFEM_MAP_FCT_2D(f,a,b) {MFEM_TEMPLATES_ID(a,b), &f<a,b>},
#define MFEM_TEMPLATES_FOREACH_2D(map,fct_p,fct,...)                    \
   MAP_PREFIX_(map,fct_p)                                               \
   FOR_EACH_2D(MFEM_MAP_FCT_2D,fct,__VA_ARGS__)                         \
   MAP_POSTFIX_(map)

// *****************************************************************************
// * MFEM_TEMPLATES_FOREACH_3D_ID, where the user defined MFEM_TEMPLATES_ID is used
// *****************************************************************************
#define MFEM_MAP_FCT_3D_ID(f,a,b,c) {MFEM_TEMPLATES_ID(a,b,c), &f<a,b,c>},
#define MFEM_TEMPLATES_FOREACH_3D_ID(map,fct_p,fct,...)                 \
   MAP_PREFIX_(map,fct_p)                                               \
   FOR_EACH_3D(MFEM_MAP_FCT_3D_ID,fct,__VA_ARGS__)                      \
   MAP_POSTFIX_(map)

// *****************************************************************************
// * MFEM_TEMPLATES_FOREACH_3D
// *****************************************************************************
#define MFEM_TEMPLATES_3D(a,b,c) ((a*100*100) + (b*100) + (c))
#define MFEM_MAP_FCT_3D(f,a,b,c) {MFEM_TEMPLATES_3D(a,b,c), &f<a,b,c>},
#define MFEM_TEMPLATES_FOREACH_3D(map,id,a,b,c,fct_p,fct,...)           \
   const unsigned int id = MFEM_TEMPLATES_3D(a,b,c);                    \
   MAP_PREFIX_(map,fct_p)                                               \
   FOR_EACH_3D(MFEM_MAP_FCT_3D,fct,__VA_ARGS__)                         \
   MAP_POSTFIX_(map)

#endif // MFEM_FOR_EACH_MACROS_HPP
