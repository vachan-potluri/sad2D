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

// #include <deal.II/numerics/derivative_approximation.h> // for adaptive mesh

#include "common.h"
#include "adv_diff.h"
#include "IC.h"
#include "BCs.h"

#ifndef sad2D_h
#define sad2D_h

/**
 * @class sad2D
 * @brief A class for solvin scalar advection diffusion equations in 2D
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

        // stiffness and lifting matrices
        std::vector<FullMatrix<double>> stiff_mats;
        std::vector< std::array<FullMatrix<double>, GeometryInfo<2>::faces_per_cell> > lift_mats;



        public:
        #ifdef DEBUG
        static void test();
        #endif
};

#endif