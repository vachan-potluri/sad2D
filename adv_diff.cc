/**
 * @file adv_diff.cc
 * @brief Namespace for functions related to advection diffusion equation
 */

#include "adv_diff.h"

/**
 * @brief Returns wind value at a location @p loc
 */
Tensor<1,2> adv_diff::wind(const Point<2> &loc)
{
        Tensor<1,2> w;
        w[0] = 1.0;
        w[2] = 1.0;
        w /= w.norm();
        return w;
}

/**
 * @brief Returns exact advective flux corresponding to @p value and location @p loc
 */
Tensor<1,2> adv_diff::exact_aflux(const double value, const Point<2> &loc)
{
        return value*wind(loc);
}

/**
 * @brief Returns normal numerical flux at a location on face using Rusanov's scheme
 * @param[in] o_value The owner state value
 * @param[in] n_value The neighbour state value
 * @param[in] normal The face normal
 * @param[in] loc The quad point location
 * 
 * See https://www.cfd-online.com/Forums/blogs/praveen/315-flux-computation-unstructured-grids.html
 * and normal_numerical_flux function of step-33
 * @note The flux vector dotted with normal vector is returned. This is what is required for assembly
 * @warning The @p normal must be a unit vector and must point from owner to neighbor
 */
double adv_diff::rusanov_aflux(const double o_value, const double n_value,
        const Tensor<1,2> &normal, const Point<2> &loc)
{
        const double num_visc = fabs(wind(loc)*normal); // artificial or numerical viscosity
        return 0.5*( exact_aflux(o_value, loc) + exact_aflux(n_value, loc) ) * normal +
               0.5*num_visc*(o_value - n_value);
}