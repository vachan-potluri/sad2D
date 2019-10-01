/**
 * @file BCs.cc
 * @brief Definition of BC functions
 */

#include "BCs.h"

double b0(const double o_value)
{
        return 1.0;
}
double b1(const double o_value)
{
        return 2.0;
}
double b2(const double o_value)
{
        return o_value;
}