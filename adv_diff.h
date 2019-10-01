/**
 * @file adv_diff.h
 * @brief Namespace for functions related to advection diffusion equation
 */

#include <deal.II/base/point.h>

#include "common.h"

#ifndef adv_diff_h
#define adv_diff_h

namespace adv_diff
{
        Tensor<1,2> wind(const Point<2> &p);
}

#endif