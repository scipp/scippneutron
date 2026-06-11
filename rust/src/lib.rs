use ndarray::{Array1, Array2, Array4, ArrayView1, ArrayView2, Zip};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pymodule]
mod _scippneutron_algo {
    use numpy::{IntoPyArray, PyArray4, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::{exceptions::PyRuntimeError, prelude::*};

    #[pyfunction]
    fn compute_q_de_norm_impl<'py>(
        py: Python<'py>,
        start: PyReadonlyArray2<'py, f64>,
        stop: PyReadonlyArray2<'py, f64>,
        solid_angle: PyReadonlyArray1<'py, f64>,
        incident_energy: f64,
        grid: (
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
        ),
    ) -> PyResult<Bound<'py, PyArray4<f64>>> {
        let grid = super::Grid::new(
            grid.0.as_array(),
            grid.1.as_array(),
            grid.2.as_array(),
            grid.3.as_array(),
        )?;
        super::compute_q_de_norm_impl(
            start.as_array(),
            stop.as_array(),
            solid_angle.as_array(),
            incident_energy,
            grid,
        )
        .map(|a| a.into_pyarray(py))
        .map_err(|message| PyErr::new::<PyRuntimeError, _>(message))
    }
}

fn compute_q_de_norm_impl(
    start: ArrayView2<f64>,
    stop: ArrayView2<f64>,
    solid_angle: ArrayView1<f64>,
    incident_energy: f64,
    grid: Grid,
) -> Result<Array4<f64>, String> {
    let mut out = ndarray::Array4::zeros(grid.shape());
    out.fill(2.0);
    Zip::from(start.rows()).and(stop.rows()).for_each(|a, b| {
        println!("{a} | {b}");
    });
    Ok(out)
}

struct Grid<'g> {
    h: ArrayView1<'g, f64>,
    k: ArrayView1<'g, f64>,
    l: ArrayView1<'g, f64>,
    kf: ArrayView1<'g, f64>,
    kf_offset: (usize, usize),
}

impl<'g> Grid<'g> {
    pub fn new(
        h: ArrayView1<'g, f64>,
        k: ArrayView1<'g, f64>,
        l: ArrayView1<'g, f64>,
        kf: ArrayView1<'g, f64>,
    ) -> PyResult<Self> {
        let kf_offset = compute_kf_offset(kf)?;
        Ok(Self {
            h,
            k,
            l,
            kf,
            kf_offset,
        })
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.h.len(), self.k.len(), self.l.len(), self.kf.len())
    }
}

fn compute_kf_offset(kf: ArrayView1<f64>) -> PyResult<(usize, usize)> {
    let (initial, _) = kf
        .iter()
        .take_while(|x| !x.is_finite())
        .copied()
        .enumerate()
        .next()
        .unwrap_or((0, 0.0));
    if initial + 1 == kf.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "No finite values in kf".to_string(),
        ));
    }
    let (fin, _) = kf
        .iter()
        .rev()
        .take_while(|x| !x.is_finite())
        .copied()
        .enumerate()
        .next()
        .unwrap_or((0, 0.0));
    Ok((initial, fin))
}
