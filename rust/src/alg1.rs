use ndarray::{Array4, ArrayView1, ArrayView2, Zip};
use super::grid::Grid;
use pyo3::{exceptions::PyValueError, prelude::*};

fn bin_index_of(array: &ArrayView1<'_, f64>, x: f64) -> usize {
    for (i, val) in array.iter().enumerate() {
        if *val > x {
            if i == 0 {
                dbg!(val, x, array);
            }
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

pub fn compute_q_de_norm_impl(
    start: ArrayView2<f64>,
    stop: ArrayView2<f64>,
    solid_angle: ArrayView1<f64>,
    grid: Grid,
) -> Array4<f64> {
    let mut out = Array4::zeros(grid.n_cells());
    Zip::from(start.rows())
        .and(stop.rows())
        .and(&solid_angle)
        .for_each(|start, stop, solid_angle| {
            let segment_endpoints = find_trajectory_segments(start, stop, &grid);
            add_strajectory_to_norm(&mut out, segment_endpoints, *solid_angle, &grid);
        });
    out
}

fn find_trajectory_segments(
    start: ArrayView1<f64>,
    stop: ArrayView1<f64>,
    grid: &Grid,
) -> Vec<(f64, f64, f64, f64)> {
    let mut segment_endpoints = Vec::new();
    const EPS: f64 = 1e-10;

    let dim = 0;
    if (start[dim] - stop[dim]).abs() > EPS {
        let delta = stop[dim] - start[dim];
        let fk = (stop[1] - start[1]) / delta;
        let fl = (stop[2] - start[2]) / delta;
        let fkf = (stop[3] - start[3]) / delta;
        for h in grid.h {
            if (start[dim] <= *h && *h < stop[dim]) || (stop[dim] < *h && *h <= start[dim]) {
                let k = fk * (h - start[dim]) + start[1];
                let l = fl * (h - start[dim]) + start[2];

                // TODO check that edges have >=2 elements
                if grid.includes_k(k) && grid.includes_l(l) {
                    let kf = fkf * (h - start[dim]) + start[3];
                    if kf >= *grid.kf.first().unwrap() && kf < *grid.kf.last().unwrap() {
                        segment_endpoints.push((*h, k, l, kf));
                    }
                }
            }
        }
    }

    let dim = 1;
    if (start[dim] - stop[dim]).abs() > EPS {
        let delta = stop[dim] - start[dim];
        let fh = (stop[0] - start[0]) / delta;
        let fl = (stop[2] - start[2]) / delta;
        let fkf = (stop[3] - start[3]) / delta;
        for k in grid.k {
            if (start[dim] <= *k && *k < stop[dim]) || (stop[dim] < *k && *k <= start[dim]) {
                let h = fh * (k - start[dim]) + start[0];
                let l = fl * (k - start[dim]) + start[2];

                if grid.includes_h(h) && grid.includes_l(l) {
                    let kf = fkf * (k - start[dim]) + start[3];
                    if grid.includes_kf(kf) {
                        segment_endpoints.push((h, *k, l, kf));
                    }
                }
            }
        }
    }

    let dim = 2;
    if (start[dim] - stop[dim]).abs() > EPS {
        let delta = stop[dim] - start[dim];
        let fh = (stop[0] - start[0]) / delta;
        let fk = (stop[1] - start[1]) / delta;
        let fkf = (stop[3] - start[3]) / delta;
        for l in grid.l {
            if (start[dim] <= *l && *l < stop[dim]) || (stop[dim] < *l && *l <= start[dim]) {
                let h = fh * (l - start[dim]) + start[0];
                let k = fk * (l - start[dim]) + start[1];

                if grid.includes_h(h) && grid.includes_k(k) {
                    let kf = fkf * (l - start[dim]) + start[3];
                    if grid.includes_kf(kf) {
                        segment_endpoints.push((h, k, *l, kf));
                    }
                }
            }
        }
    }

    let dim = 3;
    if (start[dim] - stop[dim]).abs() > EPS {
        let delta = stop[dim] - start[dim];
        let fh = (stop[0] - start[0]) / delta;
        let fk = (stop[1] - start[1]) / delta;
        let fl = (stop[2] - start[2]) / delta;
        for kf in grid.kf {
            if (start[dim] <= *kf && *kf < stop[dim]) || (stop[dim] < *kf && *kf <= start[dim]) {
                let h = fh * (kf - start[dim]) + start[0];
                let k = fk * (kf - start[dim]) + start[1];
                let l = fl * (kf - start[dim]) + start[2];

                if grid.includes_h(h) && grid.includes_k(k) && grid.includes_l(l) {
                    segment_endpoints.push((h, k, l, *kf));
                }
            }
        }
    }

    if grid.includes_h(start[0])
        && grid.includes_k(start[1])
        && grid.includes_l(start[2])
        && grid.includes_kf(start[3])
    {
        segment_endpoints.push((start[0], start[1], start[2], start[3]));
    }
    if grid.includes_h(stop[0])
        && grid.includes_k(stop[1])
        && grid.includes_l(stop[2])
        && grid.includes_kf(stop[3])
    {
        segment_endpoints.push((stop[0], stop[1], stop[2], stop[3]));
    }

    segment_endpoints
}

fn add_strajectory_to_norm(
    norm: &mut Array4<f64>,
    mut segment_endpoints: Vec<(f64, f64, f64, f64)>,
    solid_angle: f64,
    grid: &Grid,
) {
    sort_by_kf(&mut segment_endpoints);

    let left = segment_endpoints.iter();
    let mut right = left.clone();
    if right.next().is_none() {
        return; // should never happen
    }
    for (a, b) in left.zip(right) {
        let center = midpoint(a, b);
        let index = grid.index_of(&center);
        let delta_e = momentum_to_energy(b.3) - momentum_to_energy(a.3);
        norm[index] += delta_e * solid_angle;
    }
}

fn sort_by_kf(points: &mut [(f64, f64, f64, f64)]) {
    // Custom comparator to make this work with floats
    points.sort_by(|a, b| {
        if a.3 < b.3 {
            std::cmp::Ordering::Less
        } else if b.3 < a.3 {
            std::cmp::Ordering::Greater
        } else if !a.3.is_finite() || !b.3.is_finite() {
            std::cmp::Ordering::Greater // Move all NaNs and Infs to the end
        } else {
            std::cmp::Ordering::Equal
        }
    });
}

fn midpoint(a: &(f64, f64, f64, f64), b: &(f64, f64, f64, f64)) -> (f64, f64, f64, f64) {
    (
        a.0.midpoint(b.0),
        a.1.midpoint(b.1),
        a.2.midpoint(b.2),
        a.3.midpoint(b.3),
    )
}

fn momentum_to_energy(momentum: f64) -> f64 {
    // This factor is hbar ** 2 / (2 * m_n) combined with a conversion from J to meV:
    const FACTOR: f64 = 2.072124851989335;
    FACTOR * momentum * momentum
}
