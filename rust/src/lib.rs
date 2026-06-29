mod alg1;
mod grid;

use pyo3::prelude::*;

#[pymodule]
mod _scippneutron_algo {
    use numpy::{IntoPyArray, PyArray4, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::{prelude::*};
    use crate::grid::Grid;

    #[pyfunction]
    fn compute_q_de_norm_impl<'py>(
        py: Python<'py>,
        start: PyReadonlyArray2<'py, f64>,
        stop: PyReadonlyArray2<'py, f64>,
        solid_angle: PyReadonlyArray1<'py, f64>,
        grid: (
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
        ),
    ) -> PyResult<Bound<'py, PyArray4<f64>>> {
        let grid = Grid::new(
            grid.0.as_array(),
            grid.1.as_array(),
            grid.2.as_array(),
            grid.3.as_array(),
        )?;
        Ok(crate::alg1::compute_q_de_norm_impl(
            start.as_array(),
            stop.as_array(),
            solid_angle.as_array(),
            grid,
        )
        .into_pyarray(py))
    }
}
