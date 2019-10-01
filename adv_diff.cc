/**
 * @file adv_diff.cc
 * @brief Namespace for functions related to advection diffusion equation
 */

#include "adv_diff.h"

Tensor<1,2> adv_diff::wind(const Point<2> &p)
{
        Tensor<1,2> w;
        w[0] = 1.0;
        w[2] = 1.0;
        w /= w.norm();
        return w;
}