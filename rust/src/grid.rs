use ndarray::{ArrayView1};
use pyo3::{PyResult, PyErr, exceptions::PyValueError};

pub struct Grid<'g> {
    pub h: ArrayView1<'g, f64>,
    pub k: ArrayView1<'g, f64>,
    pub l: ArrayView1<'g, f64>,
    pub kf: ArrayView1<'g, f64>,
    pub kf_offset: (usize, usize), // TODO NaNs?
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

    pub fn n_cells(&self) -> (usize, usize, usize, usize) {
        (
            self.h.len() - 1,
            self.k.len() - 1,
            self.l.len() - 1,
            self.kf.len() - 1,
        )
    }

    // TODO unify
    pub fn n_cells_array(&self) -> [usize;4] {
        [
            self.h.len() - 1,
            self.k.len() - 1,
            self.l.len() - 1,
            self.kf.len() - 1,
        ]
    }

    // TODO check that edges are >=2 elems, then error never happens
    pub fn includes_h(&self, h: f64) -> bool {
        let Some(first) = self.h.first() else {
            return false;
        };
        let Some(last) = self.h.last() else {
            return false;
        };
        h >= *first && h < *last
    }

    pub fn includes_k(&self, k: f64) -> bool {
        let Some(first) = self.k.first() else {
            return false;
        };
        let Some(last) = self.k.last() else {
            return false;
        };
        k >= *first && k < *last
    }

    pub fn includes_l(&self, l: f64) -> bool {
        let Some(first) = self.l.first() else {
            return false;
        };
        let Some(last) = self.l.last() else {
            return false;
        };
        l >= *first && l < *last
    }

    pub fn includes_kf(&self, kf: f64) -> bool {
        let Some(first) = self.kf.first() else {
            return false;
        };
        let Some(last) = self.kf.last() else {
            return false;
        };
        kf >= *first && kf < *last
    }

    pub fn index_of(&self, point: &(f64, f64, f64, f64)) -> (usize, usize, usize, usize) {
        (
            bin_index_of(&self.h, point.0),
            bin_index_of(&self.k, point.1),
            bin_index_of(&self.l, point.2),
            bin_index_of(&self.kf, point.3),
        )
    }

    // TODO generic over axis?
    pub fn edges(&self, axis: usize) -> &ArrayView1<f64> {
        match axis {
            0 => &self.h,
            1 => &self.k,
            2 => &self.l,
            3 => &self.kf,
            _ => panic!(),
        }
    }
}

pub fn bin_index_of(array: &ArrayView1<'_, f64>, x: f64) -> usize {
    for (i, val) in array.iter().enumerate() {
        if *val > x {
            return i - 1;
        }
    }
    panic!("Element not in array"); // Should never happen to way the centers are constructed.
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
