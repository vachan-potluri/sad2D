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
        const double nu = 1.0; // diffusivity

        Tensor<1,2> wind(const Point<2> &loc);
        Tensor<1,2> exact_aflux(const double value, const Point<2> &loc);
        double rusanov_aflux(const double o_value, const double n_value, const Tensor<1,2> &normal,
                const Point<2> &loc);
}

#endif