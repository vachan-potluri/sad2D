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