use super::grid::Grid;
use ndarray::{Array4, ArrayView1, ArrayView2, Zip};

pub fn compute_q_de_norm_impl(
    start: ArrayView2<f64>,
    stop: ArrayView2<f64>,
    solid_angle: ArrayView1<f64>,
    grid: Grid,
) -> Array4<f64> {
    let mut out = Array4::<f64>::zeros(grid.n_cells_array());
    Zip::from(start.rows())
        .and(stop.rows())
        .and(&solid_angle)
        .for_each(|start, stop, solid_angle| {
            compute_norm_single(&mut out, &start, &stop, solid_angle, &grid);
        });
    out
}

type Point = [f64; 4];
type Direction = i8;
type Directions = [Direction; 4];

fn compute_norm_single(
    out: &mut Array4<f64>,
    start: &ArrayView1<f64>,
    stop: &ArrayView1<f64>,
    solid_angle: &f64,
    grid: &Grid,
) {
    let (start, stop) = start_stop_points(start, stop);
    let mut point = start.clone();
    let directions = compute_directions(&start, &stop);
    let Some(mut indices) = find_next_indices(&start, &directions, grid) else {
        return; // Trajectory outside the grid -> no contribution
    };
    let slopes = compute_slopes(&start, &stop);

    let mut next_intersections = compute_next_intersections(&indices, &start, &slopes, grid);

    loop {
        let next_axis = find_next_axis_to_intersect(&next_intersections, &stop);
        let next_intersection = if next_axis > 3 {
            stop
        } else {
            next_intersections[next_axis]
        };
        // don't need delta E because Ei = const -> goes away in diff
        let delta_e = momentum_to_energy(next_intersection[3]) - momentum_to_energy(point[3]);

        // TODO better check?
        // If the next index is 0 or N, we are currently outside the grid.
        // Advance to the next intersection point but don't write the result.
        let bi = bin_index(&indices, &directions);
        if !index_out_of_bounds(&bi, grid)
        {
            out[bin_index(&indices, &directions)] += delta_e * solid_angle;
        }

        if next_axis > 3 || reached_end_of_axis(next_axis, &indices, &directions, grid) {
            return; // we left the grid or reached stop
        }
        if directions[next_axis] == 1 {
            indices[next_axis] += 1;
        } else {
            indices[next_axis] -= 1;
        }

        point = next_intersection;
        next_intersections[next_axis] = compute_next_intersection(
            next_axis,
            indices[next_axis],
            &start,
            &slopes[next_axis],
            grid,
        );
    }
}

fn index_out_of_bounds(index: &(usize, usize, usize, usize), grid: &Grid) -> bool {
    // All indices are unsigned, so checking for >= n also covers underflow
    let n = grid.n_cells_array();
    index.0 >= n[0]
        || index.1 >= n[1]
        ||  index.2 >= n[2]
        ||  index.3 >= n[3]
}

fn reached_end_of_axis(
    axis: usize,
    indices: &[isize; 4],
    directions: &Directions,
    grid: &Grid,
) -> bool {
    if directions[axis] == 1 {
        indices[axis] == grid.n_cells_array()[axis] as isize
    } else {
        indices[axis] == 0
    }
}

fn bin_index(
    next_edge_index: &[isize; 4],
    directions: &Directions,
) -> (usize, usize, usize, usize) {
    // `- direction` because indices is the *next* edge
    (
        (next_edge_index[0] - if directions[0] == -1 { 0 } else { 1 }) as usize,
        (next_edge_index[1] - if directions[1] == -1 { 0 } else { 1 }) as usize,
        (next_edge_index[2] - if directions[2] == -1 { 0 } else { 1 }) as usize,
        (next_edge_index[3] - 1) as usize, // direction[3] == 1, always
    )
}

fn find_next_axis_to_intersect(intersections: &[Point; 4], stop: &Point) -> usize {
    let mut imin = 4; // imin = 4 => stop is the next 'intersection'
    let mut min = stop[3];
    for i in 0..4 {
        if intersections[i][3] < min {
            min = intersections[i][3];
            imin = i;
        }
    }
    imin
}

fn compute_next_intersections(
    indices: &[isize; 4],
    start: &Point,
    slopes: &[[f64; 4]; 4],
    grid: &Grid,
) -> [Point; 4] {
    [
        compute_next_intersection(0, indices[0], start, &slopes[0], grid),
        compute_next_intersection(1, indices[1], start, &slopes[1], grid),
        compute_next_intersection(2, indices[2], start, &slopes[2], grid),
        compute_next_intersection(3, indices[3], start, &slopes[3], grid),
    ]
}

fn compute_next_intersection(
    axis: usize,
    index: isize,
    start: &Point,
    slopes: &[f64; 4],
    grid: &Grid,
) -> Point {
    let edge = grid.edges(axis)[index as usize];
    let mut intersection = [0.0; 4];
    for i in 0..4 {
        intersection[i] = if i == axis {
            edge
        } else {
            slopes[i] * (edge - start[axis]) + start[i]
        };
    }
    intersection
}

fn find_next_indices(point: &Point, directions: &Directions, grid: &Grid) -> Option<[isize; 4]> {
    Some([
        find_next_index(point[0], directions[0], &grid.h)?,
        find_next_index(point[1], directions[1], &grid.k)?,
        find_next_index(point[2], directions[2], &grid.l)?,
        find_next_index(point[3], 1, &grid.kf)?,
    ])
}

// TODO binary search?
// TODO NaN? (only kf)
fn find_next_index(x: f64, direction: Direction, edges: &ArrayView1<f64>) -> Option<isize> {
    if direction > 0 {
        for (i, val) in edges.iter().enumerate() {
            if *val > x {
                return Some(i as isize);
            }
        }
    } else {
        for (i, val) in edges.iter().rev().enumerate() {
            if *val < x {
                return Some(edges.len() as isize - i as isize - 1);
            }
        }
    }
    None
}

fn compute_slopes(start: &Point, stop: &Point) -> [[f64; 4]; 4] {
    [
        compute_slope::<0>(start, stop),
        compute_slope::<1>(start, stop),
        compute_slope::<2>(start, stop),
        compute_slope::<3>(start, stop),
    ]
}

fn compute_slope<const I: usize>(start: &Point, stop: &Point) -> [f64; 4] {
    let denominator = stop[I] - start[I];
    let mut slope = [0.0; 4];

    // TODO do we need the checks?
    if I != 0 {
        slope[0] = (stop[0] - start[0]) / denominator;
    }
    if I != 1 {
        slope[1] = (stop[1] - start[1]) / denominator;
    }
    if I != 2 {
        slope[2] = (stop[2] - start[2]) / denominator;
    }
    if I != 3 {
        slope[3] = (stop[3] - start[3]) / denominator;
    }

    slope
}

fn compute_directions(start: &Point, stop: &Point) -> Directions {
    [
        if start[0] > stop[0] { -1 } else { 1 },
        if start[1] > stop[1] { -1 } else { 1 },
        if start[2] > stop[2] { -1 } else { 1 },
        1, // start, stop are ordered to guarantee this
    ]
}

// Make sure kf(start) <= kf(stop) to simplify comparisons
fn start_stop_points(start: &ArrayView1<f64>, stop: &ArrayView1<f64>) -> (Point, Point) {
    if start[3] > stop[3] {
        (
            [stop[0], stop[1], stop[2], stop[3]],
            [start[0], start[1], start[2], start[3]],
        )
    } else {
        (
            [start[0], start[1], start[2], start[3]],
            [stop[0], stop[1], stop[2], stop[3]],
        )
    }
}

fn momentum_to_energy(momentum: f64) -> f64 {
    // This factor is hbar ** 2 / (2 * m_n) combined with a conversion from J to meV:
    const FACTOR: f64 = 2.072124851989335;
    FACTOR * momentum * momentum
}
