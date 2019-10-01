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
sad2D::sad2D(const uint order)
: mapping(), fe(order), fe_face(order), dof_handler(triang),
        face_first_dof{0, order, 0, (order+1)*order},
        face_dof_increment{order+1, order+1, 1, 1}
{}

/**
 * @brief Sets up the system
 * 
 * 1. Mesh is setup and stored in sad2D::triang
 * 2. sad2D::dof_handler is linked to sad2D::fe
 * 3. sad2D::g_solution and sad2D::l_rhs sizes are set
 * 4. Sizes of sad2D::stiff_mats, sad2D::lift_mats and sad2D::l_rhs containers are
 * set
 */
void sad2D::setup_system()
{
        deallog << "Setting up the system" << std::endl;
        // initialise the triang variable
        GridGenerator::hyper_cube(triang);
        triang.refine_global(5); // 2^5=32 cells in each direction, total length 1m

        // set dof_handler
        dof_handler.distribute_dofs(fe);
        dof_locations.resize(dof_handler.n_dofs());
        DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_locations);

        // no system_matrix because the solution is updated cell wise
        g_solution.reinit(dof_handler.n_dofs());
        gold_solution.reinit(dof_handler.n_dofs());

        // set user flags for cell
        // for a face, cell with lower user index will be treated owner
        // is this reqd? can't we just use cell->index()?
        // uint i=0;
        // for(auto &cell: dof_handler.active_cell_iterators()){
        //         cell->set_user_index(i++);
        // } // loop over cells

        // set sizes of stiffness and lifting matrix containers
        stiff_mats.resize(triang.n_active_cells());
        lift_mats.resize(triang.n_active_cells());

        l_rhs.resize(triang.n_active_cells());
        for(auto &cur_rhs: l_rhs) cur_rhs.reinit(fe.dofs_per_cell);
}