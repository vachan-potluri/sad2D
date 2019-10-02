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
 * Based on order, sad2D::face_first_dof and sad2D::face_dof_increment containers are set here. See
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
 * 4. Sizes of all matrices and rhs
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
        // reinit is not required for matrix containers because their elements are directly equated
        // to matrices
        stiff_mats.resize(triang.n_active_cells());
        alift_mats.resize(triang.n_active_cells());
        dlift_mats.resize(triang.n_active_cells());
        damp_mats.resize(triang.n_active_cells());

        // since rhs elements are subjected to addition, the size of elements of rhs container is
        // also set
        l_rhs.resize(triang.n_active_cells());
        for(auto &cur_rhs: l_rhs) cur_rhs.reinit(fe.dofs_per_cell);
}

/**
 * @brief Assembles matrices for all cells
 */
void sad2D::assemble_system()
{
        deallog << "Assembling system ... " << std::flush;
        // allocate all local matrices
        FullMatrix<double> l_mass(fe.dofs_per_cell),
                l_mass_inv(fe.dofs_per_cell),
                l_diff(fe.dofs_per_cell),
                l_lap(fe.dofs_per_cell),
                l_dflux(fe.dofs_per_cell),
                l_aflux(fe.dofs_per_cell),
                temp(fe.dofs_per_cell); // initialise with square matrix size
        QGauss<2> cell_quad_formula(fe.degree+1); // (N+1) gauss quad for cell
        QGauss<1> face_quad_formula(fe.degree+1); // for face
        FEValues<2> fe_values(fe, cell_quad_formula,
                update_values | update_gradients | update_JxW_values | update_quadrature_points);
        FEFaceValues<2> fe_face_values(fe, face_quad_formula,
                update_values | update_gradients | update_JxW_values | update_quadrature_points
                | update_normal_vectors);
        
        uint i, j, i_face, j_face, qid, face_id;
        // compute mass and diff matrices
        for(auto &cell: dof_handler.active_cell_iterators()){
                fe_values.reinit(cell);
                l_mass = 0;
                l_diff = 0;
                l_lap = 0;
                for(qid=0; qid<fe_values.n_quadrature_points; qid++){
                        for(i=0; i<fe.dofs_per_cell; i++){
                                for(j=0; j<fe.dofs_per_cell; j++){
                                        l_mass(i,j) += fe_values.shape_value(i, qid) *
                                                fe_values.shape_value(j, qid) *
                                                fe_values.JxW(qid);
                                        l_diff(i,j) += fe_values.shape_grad(i, qid) *
                                                adv_diff::wind(fe_values.quadrature_point(qid)) *
                                                fe_values.shape_value(j, qid) *
                                                fe_values.JxW(qid);
                                        l_lap(i,j) += fe_values.shape_grad(i, qid) *
                                                fe_values.shape_grad(j, qid) *
                                                fe_values.JxW(qid);
                                } // inner loop cell shape fns
                        } // outer loop cell shape fns
                } // loop over cell quad points
                l_mass_inv.invert(l_mass);
                l_mass_inv.mmult(temp, l_diff); // store mass_inv * diff into temp
                stiff_mats[cell->index()] = temp;
                l_mass_inv.mmult(temp, l_lap);
                damp_mats[cell->index()] = temp;

                // each face will have separate flux matrices
                for(face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        fe_face_values.reinit(cell, face_id);
                        l_aflux = 0;
                        l_dflux = 0;
                        for(qid=0; qid<fe_face_values.n_quadrature_points; qid++){
                                for(i_face=0; i_face<fe_face.dofs_per_face; i_face++){
                                        for(j_face=0; j_face<fe_face.dofs_per_face; j_face++){
                                                // mapping
                                                i = face_first_dof[face_id] +
                                                        i_face*face_dof_increment[face_id];
                                                j = face_first_dof[face_id] +
                                                        j_face*face_dof_increment[face_id];
                                                l_aflux(i,j) +=
                                                        fe_face_values.shape_value(i, qid) *
                                                        fe_face_values.shape_value(j, qid) *
                                                        fe_face_values.JxW(qid);
                                                l_dflux(i,j) +=
                                                        fe_face_values.shape_grad(j, qid) *
                                                        fe_face_values.normal_vector(qid) *
                                                        fe_face_values.shape_value(i,qid) *
                                                        fe_face_values.JxW(qid);
                                        } // inner loop over face shape fns
                                } // outer loop over face shape fns
                        } // loop over face quad points
                        l_mass_inv.mmult(temp, l_aflux);
                        alift_mats[cell->index()][face_id] = temp;
                        l_mass_inv.mmult(temp, l_dflux);
                        dlift_mats[cell->index()][face_id] = temp;
                }// loop over faces
        } // loop over cells
        deallog << "Completed assembly" << std::endl;
}

/**
 * @brief Sets initial condition
 * 
 * Since nodal basis is being used, initial condition is easy to set. interpolate function of
 * VectorTools namespace is used with IC class and sad2D::g_solution. See IC::value()
 */
void sad2D::set_IC()
{
        VectorTools::interpolate(dof_handler, IC(), g_solution);
}

/**
 * @brief Boundary ids are set here
 * 
 * @f$x=0@f$ forms boundary 0 with @f$\phi@f$ value prescribed as @f$1@f$<br/>
 * @f$y=0@f$ forms boundary 1 with @f$\phi@f$ value prescribed as @f$0@f$<br/>
 * @f$x=1 \bigcup y=1@f$ forms boundary 2 with zero gradient
 * @note Ghost cell approach will be used
 */
void sad2D::set_boundary_ids()
{
        for(auto &cell: dof_handler.active_cell_iterators()){
                for(uint face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        if(cell->face(face_id)->at_boundary()){
                                Point<2> fcenter = cell->face(face_id)->center(); // face center
                                if(fabs(fcenter(0)) < 1e-6)
                                        cell->face(face_id)->set_boundary_id(0);
                                else if(fabs(fcenter(1)) < 1e-6)
                                        cell->face(face_id)->set_boundary_id(1);
                                else
                                        cell->face(face_id)->set_boundary_id(2);
                        }
                } // loop over faces
        } // loop over cells
}

/**
 * @brief Calculates stable time step based on the Courant number @p co
 * 
 * If @f$r@f$ is the "radius" of the cell, then
 * @f[
 * \Delta t = \text{Co}\frac{1}{2N+1}\min\left[ \frac{r^2}{\alpha}, \frac{r}{u}, \frac{r}{v} \right]
 * @f]
 */
double sad2D::calc_time_step(const double co) const
{
        double min=0.0, cur, radius;
        Tensor<1,2> cur_wind;
        for(auto &cell: dof_handler.active_cell_iterators()){
                radius = 0.5*cell->diameter();
                cur_wind = adv_diff::wind(cell->center());
                cur = std::min({
                        radius*radius/adv_diff::nu,
                        radius/(cur_wind[0]+1e-6),
                        radius/(cur_wind[1]+1e-6),
                });
                if(cur < min) min = cur;
        }
        return co*min/(2*fe.degree + 1);
}

/**
 * @brief Updates the solution with the given @p time_step
 * 
 * Algorithm:
 * - For every cell:
 *   - Add contribution of stiffness and damping matrices to rhs
 *   - For every face:
 *     - Get neighbor id
 *     - if neighbor id > cell id, continue
 *     - else:
 *       - Get face id wrt owner and neighbor (using neighbor_of_neighbor)
 *       - Get global dofs on owner and neighbor
 *       - Using face ids and global dofs of owner and neighbor, get global dofs on this face on
 * owner and neighbor side
 *       - Compute the numerical flux (advective and diffusive)
 *       - Use (advective and diffusive) lifting matrices to update owner and neighbor rhs
 * 
 * <code>cell->get_dof_indices()</code> will return the dof indices in the order shown in
 * https://www.dealii.org/current/doxygen/deal.II/classFE__DGQ.html. This fact is mentioned in
 * https://www.dealii.org/current/doxygen/deal.II/classDoFCellAccessor.html.
 * To get the dof location, sad2D::dof_locations has been obtained using
 * <code>DoFTools::map_dofs_to_support_points()</code>. To get normal vectors, an FEFaceValues
 * object is created with Gauss-Lobatto quadrature of order <code>fe.degree+1</code>.
 * 
 * The face normal flux vector must be mapped to owner- and neighbor- local dofs for multplication
 * with lifting matrices. The mapped vectors will be of size <code>dof_per_cell</code>.
 * 
 * @pre @p time_step must be a stable one, any checks on this value are not done
 * @todo Some code repitition exists in the loop over faces
 */
void sad2D::update(const double time_step)
{
        // update old solution
        uint i;
        for(i=0; i<dof_handler.n_dofs(); i++) gold_solution(i) = g_solution(i);

        // set rhs to zero
        for(auto &cur_rhs: l_rhs) cur_rhs=0.0;

        uint face_id, face_id_neighbor; // id of face wrt owner and neighbor
        uint l_dof_id, l_dof_id_neighbor; // dof id (on a face) dof wrt owner and neighbor
        // global dof ids of owner and neighbor
        std::vector<uint> dof_ids(fe.dofs_per_cell), dof_ids_neighbor(fe.dofs_per_cell);
        double phi, phi_neighbor; // owner and neighbor side values of phi
        double cur_anormal_flux; // advective normal flux at current dof
        double cur_dvalue; // (diffusive) average value on face at current dof
        // the -ve of advective normal num flux vector of face wrt owner and neighbor
        Vector<double> neg_anormal_flux(fe.dofs_per_cell),
                neg_anormal_flux_neighbor(fe.dofs_per_cell);
        // (diffusive) average values of phi on current face wrt owner and neighbor
        Vector<double> dvalues(fe.dofs_per_cell), dvalues_neighbor(fe.dofs_per_cell);
        Point<2> dof_loc; // dof coordinates (on a face)
        Tensor<1,2> normal; // face normal from away from owner at current dof
        FEFaceValues<2> fe_face_values(fe, QGaussLobatto<1>(fe.degree+1), update_normal_vectors);

        for(auto &cell: dof_handler.active_cell_iterators()){
                cell->get_dof_indices(dof_ids);
                // compute stiffness and damping term
                Vector<double> lold_solution(fe.dofs_per_cell); // old phi values of cell
                for(i=0; i<fe.dofs_per_cell; i++) lold_solution[i] = gold_solution[dof_ids[i]];
                stiff_mats[cell->index()].vmult_add(
                        l_rhs[cell->index()],
                        lold_solution
                );
                lold_solution *= -1.0; // change sign of old soln
                damp_mats[cell->index()].vmult_add(
                        l_rhs[cell->index()],
                        lold_solution
                );
                // advective and diffusive face terms
                for(face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        if(cell->face(face_id)->at_boundary()){
                                // this face is part of boundary, set phi_neighbor appropriately
                                fe_face_values.reinit(cell, face_id);
                                for(i=0; i<fe_face.dofs_per_face; i++){
                                        l_dof_id = face_first_dof[face_id] +
                                                i*face_dof_increment[face_id];
                                        
                                        normal = fe_face_values.normal_vector(i);
                                        // owner and neighbor side dof locations will match
                                        dof_loc = dof_locations[
                                                dof_ids[ l_dof_id ]
                                        ];

                                        phi = gold_solution[
                                                dof_ids[ l_dof_id ]
                                        ];
                                        // use array of functions (or func ptrs) to set BC
                                        phi_neighbor =
                                                bc_fns[cell->face(face_id)->boundary_id()](phi);

                                        cur_anormal_flux = adv_diff::rusanov_aflux(phi,
                                                phi_neighbor, normal, dof_loc);
                                        neg_anormal_flux(l_dof_id) = -cur_anormal_flux;
                                        cur_dvalue = 0.5*(phi+phi_neighbor);
                                        dvalues(l_dof_id) = cur_dvalue;
                                } // loop over face dofs

                                // multiply normal flux with advective lift matrx and store add to rhs
                                alift_mats[cell->index()][face_id].vmult_add(
                                        l_rhs[cell->index()],
                                        neg_anormal_flux
                                );
                                // multiply average value with diffusive life matrix and add to rhs
                                dlift_mats[cell->index()][face_id].vmult_add(
                                        l_rhs[cell->index()],
                                        dvalues
                                );
                        }
                        else if(cell->neighbor_index(face_id) > cell->index()) continue;
                        else{
                                // internal face
                                fe_face_values.reinit(cell, face_id);
                                face_id_neighbor = cell->neighbor_of_neighbor(face_id);
                                cell->neighbor(face_id)->get_dof_indices(dof_ids_neighbor);
                                for(i=0; i<fe_face.dofs_per_face; i++){
                                        l_dof_id = face_first_dof[face_id] +
                                                i*face_dof_increment[face_id];
                                        l_dof_id_neighbor = face_first_dof[face_id_neighbor] +
                                                i*face_dof_increment[face_id_neighbor];
                                        
                                        normal = fe_face_values.normal_vector(i);
                                        // owner and neighbor side dof locations will match
                                        dof_loc = dof_locations[
                                                dof_ids[ l_dof_id ]
                                        ];

                                        phi = gold_solution[
                                                dof_ids[ l_dof_id ]
                                        ];
                                        phi_neighbor = gold_solution[
                                                dof_ids_neighbor[ l_dof_id_neighbor ]
                                        ];

                                        cur_anormal_flux = adv_diff::rusanov_aflux(phi,
                                                phi_neighbor, normal, dof_loc);
                                        cur_dvalue = 0.5*(phi+phi_neighbor);
                                        neg_anormal_flux(l_dof_id) = -cur_anormal_flux;
                                        neg_anormal_flux_neighbor(l_dof_id_neighbor) = cur_anormal_flux;
                                        dvalues(l_dof_id) = cur_dvalue;
                                        dvalues_neighbor(l_dof_id_neighbor) = cur_dvalue;
                                } // loop over face dofs

                                // multiply normal flux with advective lift matrx and add to rhs
                                // for both owner and neighbor
                                alift_mats[cell->neighbor_index(face_id)][face_id_neighbor].vmult_add(
                                        l_rhs[cell->neighbor_index(face_id)],
                                        neg_anormal_flux_neighbor
                                );
                                alift_mats[cell->index()][face_id].vmult_add(
                                        l_rhs[cell->index()],
                                        neg_anormal_flux
                                );
                                // multiply average values with diffusive lift matrix and add to rhs
                                dlift_mats[cell->neighbor_index(face_id)][face_id_neighbor].vmult_add(
                                        l_rhs[cell->neighbor_index(face_id)],
                                        dvalues_neighbor
                                );
                                dlift_mats[cell->index()][face_id].vmult_add(
                                        l_rhs[cell->index()],
                                        dvalues
                                );
                        }
                } // loop over faces
        } // loop over cells for assembling rhs

        // Now, update
        for(auto &cell: dof_handler.active_cell_iterators()){
                cell->get_dof_indices(dof_ids);
                for(i=0; i<fe.dofs_per_cell; i++){
                        g_solution[dof_ids[i]] = gold_solution[dof_ids[i]] +
                                l_rhs[cell->index()][i] * time_step;
                }
        }
}

/**
 * @brief Prints stifness, damping and the 8 lifting matrices of 0-th element
 */
void sad2D::print_matrices() const
{
        deallog << "Stiffness matrix" << std::endl;
        stiff_mats[0].print(deallog, 10, 2);
        deallog << "Damping matrix" << std::endl;
        damp_mats[0].print(deallog, 10, 2);
        for(uint i=0; i<GeometryInfo<2>::faces_per_cell; i++){
                deallog << "Advection lifting matrix, face " << i << std::endl;
                alift_mats[0][i].print(deallog, 15, 4);
        }
        for(uint i=0; i<GeometryInfo<2>::faces_per_cell; i++){
                deallog << "Diffusion lifting matrix, face " << i << std::endl;
                dlift_mats[0][i].print(deallog, 15, 4);
        }
}

/**
 * @brief Outputs the global solution in vtk format taking the filename as argument
 */
void sad2D::output(const std::string &filename) const
{
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(g_solution, "phi");

        data_out.build_patches();

        std::ofstream ofile(filename);
        data_out.write_vtk(ofile);
}



// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
// Test function
// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#ifdef DEBUG
void sad2D::test()
{
        deallog << "---------------------------------------------" << std::endl;
        deallog << "Testing advection2D class" << std::endl;
        deallog << "---------------------------------------------" << std::endl;
        sad2D problem(1);
        problem.setup_system();
        problem.assemble_system();
        problem.print_matrices();
        problem.set_IC();
        problem.set_boundary_ids();
}
#endif