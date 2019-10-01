/**
 * @file IC.h
 * @brief Contains declaration for initial condition
 */

#ifndef IC_h
#define IC_h

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include "common.h"

class IC: public Function<2>
{
        public:
        IC() = default;
        virtual double value(const Point<2> &p, const uint component=0) const override;
};

#endif