//! # Hydro-analysis
//!
//! `hydro-analysis` provides functions for Hydrology DEM manipulation that are based on
//! [whitebox](https://github.com/jblindsay/whitebox-tools).  Whitebox is a command line tool, this
//! crate provides some (only a couple functions at present) of that functionality via functions so
//! can be called from your code.
//!
//! ## Example
//!
//! ```
//! use ndarray::Array2;
//! use hydro_analysis::fill_depressions;
//!
//! let mut dem = Array2::from_shape_vec(
//!     (3, 3),
//!     vec![
//!         10.0, 12.0, 10.0,
//!         12.0, 9.0,  12.0,
//!         10.0, 12.0, 10.0,
//!     ],
//! ).expect("Failed to create DEM");
//!
//! fill_depressions(&mut dem, -3.0, 8.0, 8.0, true);
//! ```
use rayon::prelude::*;
use std::collections::{BinaryHeap, VecDeque};
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use ndarray::Array2;


#[derive(PartialEq, Debug)]
struct GridCell {
    row: usize,
    column: usize,
    priority: f64,
}

impl Eq for GridCell {}

impl PartialOrd for GridCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.priority.partial_cmp(&self.priority)
    }
}

impl Ord for GridCell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(PartialEq, Debug)]
struct GridCell2 {
    row: usize,
    column: usize,
    z: f64,
    priority: f64,
}

impl Eq for GridCell2 {}

impl PartialOrd for GridCell2 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.priority.partial_cmp(&self.priority)
    }
}

impl Ord for GridCell2 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}


/// Fills depressions (sinks) in a digital elevation model (DEM).
///
/// More-or-less the contents of
/// [whitebox fill_depressions](https://github.com/jblindsay/whitebox-tools/blob/master/whitebox-tools-app/src/tools/hydro_analysis/fill_depressions.rs)
///
/// This function modifies the input `dem` to ensure that all depressions (local minima that do not
/// drain) are removed, making the surface hydrologically correct. It also considers no-data values
/// and can optionally fix flat areas.
///
/// # Parameters
///
/// - `dem`: A mutable reference to a 2D array (`Array2<f64>`) representing the elevation data.
/// - `nodata`: The value representing no-data cells in the DEM.
/// - `resx`: The horizontal resolution (grid spacing in the x-direction).
/// - `resy`: The vertical resolution (grid spacing in the y-direction).
/// - `fix_flats`: A boolean flag to determine whether flat areas should be slightly sloped.
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use hydro_analysis::fill_depressions;
///
/// let mut dem = Array2::from_shape_vec(
///     (3, 3),
///     vec![
///         10.0, 12.0, 10.0,
///         12.0, 9.0,  12.0,
///         10.0, 12.0, 10.0,
///     ],
/// ).expect("Failed to create DEM");
///
/// fill_depressions(&mut dem, -3.0, 8.0, 8.0, true);
/// ```
pub fn fill_depressions(
    dem: &mut Array2<f64>, nodata: f64, resx: f64, resy: f64, fix_flats: bool
)
{
    let (rows, columns) = (dem.nrows(), dem.ncols());
    let small_num = {
        let diagres = (resx * resx + resy * resy).sqrt();
        let elev_digits = (dem.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as i64).to_string().len();
        let elev_multiplier = 10.0_f64.powi((9 - elev_digits) as i32);
        1.0_f64 / elev_multiplier as f64 * diagres.ceil()
    };

    let dx = [1, 1, 1, 0, -1, -1, -1, 0];
    let dy = [-1, 0, 1, 1, 1, 0, -1, -1];

    // Find pit cells. This step is parallelizable.
	let mut pits: Vec<_> = (1..rows - 1)
		.into_par_iter()
		.flat_map(|row| {
			let mut local_pits = Vec::new();
			for col in 1..columns - 1 {
				let z = dem[[row, col]];
				if z == nodata {
					continue;
				}
				let mut apit = true;
            	// is anything lower than me?
				for n in 0..8 {
					let zn = dem[[(row as isize + dy[n]) as usize, (col as isize + dx[n]) as usize]];
					if zn < z || zn == nodata {
						apit = false;
						break;
					}
				}
				// no, so I am a pit
				if apit {
					local_pits.push((row, col, z));
				}
			}
			local_pits
		}).collect();

    // Now we need to perform an in-place depression filling
    let mut minheap = BinaryHeap::new();
    let mut minheap2 = BinaryHeap::new();
    let mut visited = Array2::<u8>::zeros((rows, columns));
    let mut flats = Array2::<u8>::zeros((rows, columns));
    let mut possible_outlets = vec![];
    let mut queue = VecDeque::new();

    // go through pits from highest to lowest
    pits.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Equal));
    while let Some(cell) = pits.pop() {
        let row: usize = cell.0;
        let col: usize = cell.1;

        // if it's already in a solved site, don't do it a second time.
        if flats[[row, col]] != 1 {
            // First there is a priority region-growing operation to find the outlets.
            minheap.clear();
            minheap.push(GridCell {
                row: row,
                column: col,
                priority: dem[[row, col]],
            });
            visited[[row, col]] = 1;
            let mut outlet_found = false;
            let mut outlet_z = f64::INFINITY;
            if !queue.is_empty() {
                queue.clear();
            }
            while let Some(cell2) = minheap.pop() {
                let z = cell2.priority;
                if outlet_found && z > outlet_z {
                    break;
                }
                if !outlet_found {
                    for n in 0..8 {
                        let cn: usize = (cell2.column as isize + dx[n]) as usize;
                        let rn: usize = (cell2.row as isize + dy[n]) as usize;
                        if rn < rows && cn < columns && visited[[rn, cn]] == 0 {
                            let zn = dem[[rn, cn]];
                            if !outlet_found {
                                if zn >= z && zn != nodata {
                                    minheap.push(GridCell {
                                        row: rn,
                                        column: cn,
                                        priority: zn,
                                    });
                                    visited[[rn, cn]] = 1;
                                } else if zn != nodata {
                                    // zn < z
                                    // 'cell' has a lower neighbour that hasn't already passed through minheap.
                                    // Therefore, 'cell' is a pour point cell.
                                    outlet_found = true;
                                    outlet_z = z;
                                    queue.push_back((cell2.row, cell2.column));
                                    possible_outlets.push((cell2.row, cell2.column));
                                }
                            } else if zn == outlet_z {
                                // We've found the outlet but are still looking for additional depression cells.
                                minheap.push(GridCell {
                                    row: rn,
                                    column: cn,
                                    priority: zn,
                                });
                                visited[[rn, cn]] = 1;
                            }
                        }
                    }
                } else {
                    // We've found the outlet but are still looking for additional depression cells and potential outlets.
                    if z == outlet_z {
                        let mut anoutlet = false;
                        for n in 0..8 {
                            let cn: usize = (cell2.column as isize + dx[n]) as usize;
                            let rn: usize = (cell2.row as isize + dy[n]) as usize;
                            if rn < rows && cn < columns && visited[[rn, cn]] == 0 {
                                let zn = dem[[rn, cn]];
                                if zn < z {
                                    anoutlet = true;
                                } else if zn == outlet_z {
                                    minheap.push(GridCell {
                                        row: rn,
                                        column: cn,
                                        priority: zn,
                                    });
                                    visited[[rn, cn]] = 1;
                                }
                            }
                        }
                        if anoutlet {
                            queue.push_back((cell2.row, cell2.column));
                            possible_outlets.push((cell2.row, cell2.column));
                        } else {
                            visited[[cell2.row, cell2.column]] = 1;
                        }
                    }
                }
            }

            if outlet_found {
                // Now that we have the outlets, raise the interior of the depression.
                // Start from the outlets.
                while let Some(cell2) = queue.pop_front() {
                    for n in 0..8 {
                        let rn: usize = (cell2.0 as isize + dy[n]) as usize;
                        let cn: usize = (cell2.1 as isize + dx[n]) as usize;
                        if rn < rows && cn < columns && visited[[rn, cn]] == 1 {
                            visited[[rn, cn]] = 0;
                            queue.push_back((rn, cn));
                            let z = dem[[rn, cn]];
                            if z < outlet_z {
                                dem[[rn, cn]] = outlet_z;
                                flats[[rn, cn]] = 1;
                            } else if z == outlet_z {
                                flats[[rn, cn]] = 1;
                            }
                        }
                    }
                }
            } else {
                queue.push_back((row, col)); // start at the pit cell and clean up visited
                while let Some(cell2) = queue.pop_front() {
                    for n in 0..8 {
                        let rn: usize = (cell2.0 as isize + dy[n]) as usize;
                        let cn: usize = (cell2.1 as isize + dx[n]) as usize;
                        if visited[[rn, cn]] == 1 {
                            visited[[rn, cn]] = 0;
                            queue.push_back((rn, cn));
                        }
                    }
                }
            }
        }

    }

    drop(visited);

    //let (mut col, mut row): (isize, isize);
    //let (mut rn, mut cn): (isize, isize);
    //let (mut z, mut zn): (f64, f64);
    //let mut flag: bool;

    if small_num > 0.0 && fix_flats {
        // Some of the potential outlets really will have lower cells.
        minheap.clear();
        while let Some(cell) = possible_outlets.pop() {
            let z = dem[[cell.0, cell.1]];
            let mut anoutlet = false;
            for n in 0..8 {
                let rn: usize = (cell.0 as isize + dy[n]) as usize;
                let cn: usize = (cell.1 as isize + dx[n]) as usize;
                if rn >= rows || cn >= columns {
                    break;
                }
                let zn = dem[[rn, cn]];
                if zn < z && zn != nodata {
                    anoutlet = true;
                    break;
                }
            }
            if anoutlet {
                minheap.push(GridCell {
                    row: cell.0,
                    column: cell.1,
                    priority: z,
                });
            }
        }

        let mut outlets = vec![];
        while let Some(cell) = minheap.pop() {
            if flats[[cell.row, cell.column]] != 3 {
                let z = dem[[cell.row, cell.column]];
                flats[[cell.row, cell.column]] = 3;
                if !outlets.is_empty() {
                    outlets.clear();
                }
                outlets.push(cell);
                // Are there any other outlet cells at the same elevation (likely for the same feature)
                let mut flag = true;
                while flag {
                    match minheap.peek() {
                        Some(cell2) => {
                            if cell2.priority == z {
                                flats[[cell2.row, cell2.column]] = 3;
                                outlets
                                    .push(minheap.pop().expect("Error during pop operation."));
                            } else {
                                flag = false;
                            }
                        }
                        None => {
                            flag = false;
                        }
                    }
                }
                if !minheap2.is_empty() {
                    minheap2.clear();
                }
                for cell2 in &outlets {
                    let z = dem[[cell2.row, cell2.column]];
                    for n in 0..8 {
                        let cn: usize = (cell2.column as isize + dx[n]) as usize;
                        let rn: usize = (cell2.row as isize + dy[n]) as usize;
                        if rn < rows && cn < columns && flats[[rn, cn]] != 3 {
                            let zn = dem[[rn, cn]];
                            if zn == z && zn != nodata {
                                minheap2.push(GridCell2 {
                                    row: rn,
                                    column: cn,
                                    z: z,
                                    priority: dem[[rn, cn]],
                                });
                                dem[[rn, cn]] = z + small_num;
                                flats[[rn, cn]] = 3;
                            }
                        }
                    }
                }
                // Now fix the flats
                while let Some(cell2) = minheap2.pop() {
                    let z = dem[[cell2.row, cell2.column]];
                    for n in 0..8 {
                        let cn: usize = (cell2.column as isize + dx[n]) as usize;
                        let rn: usize = (cell2.row as isize + dy[n]) as usize;
                        if rn < rows && cn < columns && flats[[rn, cn]] != 3 {
                            let zn = dem[[rn, cn]];
                            if zn < z + small_num && zn >= cell2.z && zn != nodata {
                                minheap2.push(GridCell2 {
                                    row: rn,
                                    column: cn,
                                    z: cell2.z,
                                    priority: dem[[rn, cn]],
                                });
                                dem[[rn, cn]] = z + small_num;
                                flats[[rn, cn]] = 3;
                            }
                        }
                    }
                }
            }

        }
    }
}



