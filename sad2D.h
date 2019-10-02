/**
 * @file sad2D.h
 * @brief Declares sad2D class
 */

// Includes: most of them are from step-12 and dflo
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>

#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <functional>
#include <algorithm>

// #include <deal.II/numerics/derivative_approximation.h> // for adaptive mesh

#include "common.h"
#include "adv_diff.h"
#include "IC.h"
#include "BCs.h"

#ifndef sad2D_h
#define sad2D_h

/**
 * @class sad2D
 * @brief A class for solving scalar advection diffusion equations in 2D
 * 
 * The advection diffusion equation is solved
 * @f[
 * \frac{\partial \phi}{\partial t} + \nabla \cdot (\phi \vec{v}) = \nu\nabla^2\phi
 * @f]
 * The DG formulation would be
 * @f[
 * \int_{\Omega_h} l_i \left(\sum \frac{\partial\phi_j}{\partial t} l_j\right) \,d\Omega +
 * \sum_{\text{faces}} \int_{\text{face}} l_i \left(\sum\phi^*_j l_j\right)
 * \vec{v}\cdot\vec{n}\,dA -
 * \int_{\Omega_h}\nabla l_i\cdot\vec{v} \left(\sum\phi_j l_j\right) \,d\Omega =
 * \sum_{\text{faces}} \int_{\text{face}} l_i \left(\sum\overline{\phi}_j\nabla l_j\right)
 * \cdot \vec{n}\,dA -
 * \int_{\Omega_h} \nabla l_i \cdot \left(\sum \phi_j\nabla l_j\right)\,d\Omega
 * @f]
 * where @f$\phi^*_j\vec{v}\cdot\vec{n} \equiv f^*_j@f$ is the normal advective numerical flux and
 * @f$\overline{\phi}_j@f$ is the average of owner and neighbor values at a face dof.
 * 
 * With explicit time integration, we get
 * @f[
 * [M]\{\phi\}^{n+1} = [M]\{\phi\}^n + \Delta t \left[
 * [D]\{\phi\}^n - \sum_{\text{faces}}[F_a]\{f^*\}^n +
 * \sum_{\text{faces}}[F_d]\{\overline{\phi}\}^n - [L]\{\phi\}^n
 * \right]
 * @f]
 * where @f$[M]@f$ is the mass matrix, @f$[D]@f$ is the differentiation matrix, @f$[L]@f$ is the
 * Laplacian matrix, @f$[F_a]@f$ is the advective flux matrix and @f$[F_d]@f$ is the diffusive flux
 * matrix.
 * 
 * Multiplying with mass inverse on both sides, we get
 * @f[
 * \{\phi\}^{n+1} = \{\phi\}^n + \left[
 * [S]\{\phi\}^n - \sum_{\text{faces}}[L_a]\{f^*\}^n +
 * \sum_{\text{faces}}[L_d]\{\overline{\phi}\}^n - [A]\{\phi\}^n
 * \right]
 * @f]
 * where @f$[S]@f$ is the stiffness matrix, @f$[L_a]@f$ is the advective lifting matrix, @f$[L_d]@f$ is
 * the diffusive lifting matrix and @f$[A]@f$ is the damping matrix.
 * 
 * @todo Functions sad2D::assemble_system() and sad2D::update
 */
class sad2D
{
        public:
        sad2D(const uint order);

        // first cell dof on a face
        const std::array<uint, GeometryInfo<2>::faces_per_cell> face_first_dof;

        // increment of cell dof on a face
        const std::array<uint, GeometryInfo<2>::faces_per_cell> face_dof_increment;

        std::array< std::function<double(const double)>, 3 > bc_fns = {b0,b1,b2};

        private:
        void setup_system();
        void assemble_system();
        void set_IC();
        void set_boundary_ids();
        double calc_time_step(const double co) const;
        void update(const double time_step);
        void print_matrices() const;
        void output(const std::string &filename) const;

        // class variables
        Triangulation<2> triang;
        const MappingQ1<2> mapping;

        // By default, fe assumes all dofs to be inside cell. Thus, fe.dofs_per_face will return 0.
        // The variable fe_face can be thought as projection of a DG basis on a face
        FE_DGQ<2> fe;
        FE_FaceQ<2> fe_face; // face finite element
        DoFHandler<2> dof_handler;
        std::vector<Point<2>> dof_locations; // all dof locations

        // solution has to be global to enable results output, a local solution cannot used to
        // output results
        Vector<double> g_solution; // global solution
        Vector<double> gold_solution; // global old solution
        std::vector<Vector<double>> l_rhs; // local rhs of every cell

        // stiffness, (advection and diffusion) lifting and damping matrices
        std::vector<FullMatrix<double>> stiff_mats;
        std::vector< std::array<FullMatrix<double>, GeometryInfo<2>::faces_per_cell> > alift_mats;
        std::vector< std::array<FullMatrix<double>, GeometryInfo<2>::faces_per_cell> > dlift_mats;
        std::vector<FullMatrix<double>> damp_mats;



        public:
        #ifdef DEBUG
        static void test();
        #endif
};

#endif