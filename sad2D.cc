/**
 * @file sad2D.cc
 * @brief Contains definition of sad2D class
 */

#include "sad2D.h"

/**
 * @brief Constructor with @p order of polynomial approx as arg
 * 
 * sad2D::mapping, sad2D::fe and sad2D::fe_face are initialised.
 * sad2D::dof_handler is associated to sad2D::triang.
 * Based on order, face_first_dof and face_dof_increment containers are set here. See
 * https://www.dealii.org/current/doxygen/deal.II/structGeometryInfo.html and
 * https://www.dealii.org/current/doxygen/deal.II/classFE__DGQ.html for face and dof ordering
 * respectively in a cell. According to GeometryInfo, the direction of face lines is along the
 * positive axes. See DG notes dated 24-09-19.
 * 
 * Eg: for order=2, on 1-th face, the first cell dof is 2 and the next dof is obtained after
 * increment of 3
 */
advection2D::advection2D(const uint order)
: mapping(), fe(order), fe_face(order), dof_handler(triang),
        face_first_dof{0, order, 0, (order+1)*order},
        face_dof_increment{order+1, order+1, 1, 1}
{}