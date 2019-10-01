/**
 * @file BCs.h
 * @brief Declarations for BCs and definition of the BC fn (ptr) array
 * 
 * The boundary functions for each of the boundaries are written and combined into an array of
 * functions (or function pointers). See sad2D::set_boundary_ids() for the boundary
 * definitions. All functions take owner value as parameter
 */

#ifndef BCs_h
#define BCs_h

#include <array>

double b0(const double o_value);
double b1(const double o_value);
double b2(const double o_value);

#endif