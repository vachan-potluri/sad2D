/**
 * @file IC.cc
 * @brief Contains definition for initial condition
 */

#include "IC.h"

double IC::value(const Point<2> &p, const uint component) const
{
      return p.norm();
}