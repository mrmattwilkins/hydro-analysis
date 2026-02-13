//! # Hydro-analysis
//!
//! `hydro-analysis` provides functions for Hydrology DEM manipulation.  There are
//! a couple generic functions for reading/writing raster files of any common
//! primative type (which surprizingly I couldn't find anywhere else, unless you
//! use GDAL which I am trying to avoid).  Also there are a couple functions based
//! on [whitebox](https://github.com/jblindsay/whitebox-tools).  Whitebox is a
//! command line tool, this provides functionality via functions so can be called
//! from your code.
//!
//! ```

use rayon::prelude::*;
use std::collections::{HashMap, BinaryHeap, VecDeque};
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::collections::hash_map::Entry::Vacant;
use ndarray::{Array2, par_azip};

use std::{fs::File, f64, path::PathBuf};
use thiserror::Error;
use bytemuck::cast_slice;

use tiff::decoder::DecodingResult;
use tiff::encoder::compression::Deflate;
use tiff::encoder::colortype::{Gray8,Gray16,Gray32,Gray64,Gray32Float,Gray64Float,GrayI8,GrayI16,GrayI32,GrayI64};
use tiff::tags::Tag;
use tiff::TiffFormatError;


#[derive(Debug, Error)]
pub enum RasterError {
    #[error("TIFF error: {0}")]
    Tiff(#[from] tiff::TiffError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("NDarray: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("Failed to parse nodata value")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("Failed to parse nodata value")]
    ParseFloatError(#[from] std::num::ParseFloatError),

    #[error("Unsupported type: {0}")]
    UnsupportedType(String)
}

/// Reads a single-band grayscale GeoTIFF raster file returning ndarry2 and metadata
///
/// # Type Parameters
/// - `T`: The pixel value type.  u8, u16, i16, f64 etc
///
/// # Parameters
/// - `fname`: Path to the input `.tif` GeoTIFF file.
///
/// # Returns
/// A `Result` with a tuple containing:
///  - `Array2<T>`: The raster data in a 2D array.
///  - `T`: nodata
///  - `u16`: CRS (e.g. 2193)
///  - `[f64; 6]`: The affine GeoTransform in the format:
///   `[origin_x, pixel_size_x, rotation_x, origin_y, rotation_y, pixel_size_y]`.
///  - `Vec<u64>`: raw GeoKeyDirectoryTag values (needed for writing to file)
///  - `String`: PROJ string (needed for writing to file)
///
/// # Errors
///  - Returns `RasterError` variants if reading fails, the type conversion for data or metadata
///   fails, or required tags are missing from the TIFF file.
///
/// # Example
///   See [examples/reading_and_writing_rasters.rs](examples/reading_and_writing_rasters.rs)
pub fn rasterfile_to_array<T>(fname: &PathBuf) -> Result<
    (
        Array2<T>,
        T,          // nodata
        u16,        // crs
        [f64; 6],   // geo transform [start_x, psize_x, rotation, starty, rotation, psize_y]
        Vec<u64>,   // geo dir, it has the crs in it
        String      // the projection string
    ),
    RasterError
>
    where T: std::str::FromStr + num::FromPrimitive,
          <T as std::str::FromStr>::Err: std::fmt::Debug,
          RasterError: std::convert::From<<T as std::str::FromStr>::Err> { // Open the file
    let file = File::open(fname)?;

    // Create a TIFF decoder
    let mut decoder = tiff::decoder::Decoder::new(file)?;
    decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());

    // Read the image dimensions
    let (width, height) = decoder.dimensions()?;

    fn estr<T>(etype: &'static str) -> RasterError {
        RasterError::Tiff(TiffFormatError::Format(format!("Raster is {}, I was expecting {}", etype, std::any::type_name::<T>()).into()).into())
    }
    let data: Vec<T> = match decoder.read_image()? {
        DecodingResult::I8(buf)  => buf.into_iter().map(|v| <T>::from_i8(v).ok_or(estr::<T>("I8"))).collect::<Result<_, _>>(),
        DecodingResult::I16(buf) => buf.into_iter().map(|v| <T>::from_i16(v).ok_or(estr::<T>("I16"))).collect::<Result<_, _>>(),
        DecodingResult::I32(buf) => buf.into_iter().map(|v| <T>::from_i32(v).ok_or(estr::<T>("I32"))).collect::<Result<_, _>>(),
        DecodingResult::I64(buf) => buf.into_iter().map(|v| <T>::from_i64(v).ok_or(estr::<T>("I64"))).collect::<Result<_, _>>(),
        DecodingResult::U8(buf)  => buf.into_iter().map(|v| <T>::from_u8(v).ok_or(estr::<T>("U8"))).collect::<Result<_, _>>(),
        DecodingResult::U16(buf) => buf.into_iter().map(|v| <T>::from_u16(v).ok_or(estr::<T>("U16"))).collect::<Result<_, _>>(),
        DecodingResult::U32(buf) => buf.into_iter().map(|v| <T>::from_u32(v).ok_or(estr::<T>("U32"))).collect::<Result<_, _>>(),
        DecodingResult::U64(buf) => buf.into_iter().map(|v| <T>::from_u64(v).ok_or(estr::<T>("U64"))).collect::<Result<_, _>>(),
        DecodingResult::F32(buf) => buf.into_iter().map(|v| <T>::from_f32(v).ok_or(estr::<T>("F32"))).collect::<Result<_, _>>(),
        DecodingResult::F64(buf) => buf.into_iter().map(|v| <T>::from_f64(v).ok_or(estr::<T>("F64"))).collect::<Result<_, _>>(),
    }?;

    // Convert the flat vector into an ndarray::Array2
    let array: Array2<T> = Array2::from_shape_vec((height as usize, width as usize), data)?;

    // nodata value
    let nodata: T = decoder.get_tag_ascii_string(Tag::GdalNodata)?.trim().parse::<T>()?;

    // pixel scale [pixel scale x, pixel scale y, ...]
    // NB pixel scale y is the absolute value, it is POSITIVE.  We have to make it negative later
    let pscale: Vec<f64> = decoder.get_tag_f64_vec(Tag::ModelPixelScaleTag)?.into_iter().collect();

    // tie point [0 0 0 startx starty 0]
    let tie: Vec<f64>  = decoder.get_tag_f64_vec(Tag::ModelTiepointTag)?.into_iter().collect();

    // transform, the zeros are the rotations [start x, x pixel size, 0, start y, 0, y pixel size]
    let geotrans: [f64; 6] = [tie[3], pscale[0], 0.0, tie[4], 0.0, -pscale[1]];

    let projection: String = decoder.get_tag_ascii_string(Tag::GeoAsciiParamsTag)?;
    let geokeydir: Vec<u64> = decoder .get_tag_u64_vec(Tag::GeoKeyDirectoryTag)?;

    // try and get the CRS out of the geokeydir, it is the bit after 3072
    let crs = geokeydir.windows(4).find(|w| w[0] == 3072).map(|w| w[3])
        .ok_or(RasterError::Tiff(tiff::TiffFormatError::InvalidTagValueType(Tag::GeoKeyDirectoryTag).into()))? as u16;

    Ok((array, nodata, crs, geotrans, geokeydir, projection))
}

/// Writes a 2D array of values to a GeoTIFF raster with geo metadata.
///
/// # Type Parameters
/// - `T`: The element type of the array, which must implement `bytemuck::Pod`
/// (for safe byte casting) and `ToString` (for writing NoData values to
/// metadata).
///
/// # Parameters
/// `data`: A 2D array (`ndarray::Array2<T>`) containing raster pixel values.
/// `nd`: NoData value
/// `geotrans`: A 6-element array defining the affine geotransform:
///   `[origin_x, pixel_size_x, rotation_x, origin_y, rotation_y, pixel_size_y]`.
/// `geokeydir`: &[u64] the GeoKeyDirectoryTag (best got from reading a raster)
/// `proj`: PROJ string (best got from reading a raster)
/// `outfile`: The path to the output `.tif` file.
///
/// # Returns
/// Ok() or a `RasterError`
///
/// # Errors
/// - Returns `RasterError::UnsupportedType` if `T` can't be mapped to a TIFF format.
/// - Propagates I/O and TIFF writing errors
///
/// # Example
///   See [examples/reading_and_writing_rasters.rs](examples/reading_and_writing_rasters.rs)
pub fn array_to_rasterfile<T>(
    data: &Array2<T>,
    nd: T,                      // nodata
    geotrans: &[f64; 6],        // geo transform [start_x, psize_x, rotation, starty, rotation, psize_y]
    geokeydir: &[u64],          // geo dir, it has the crs in it
    proj: &str,                 // the projection string
    outfile: &PathBuf
) -> Result<(), RasterError>
    where T: bytemuck::Pod + ToString
{
    let (nrows, ncols) = (data.nrows(), data.ncols());

    let fh = File::create(outfile)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(fh)?;

    // Because image doesn't have traits I couldn't figure out how to do this with generics
    // This macro takes the tiff colortype
    macro_rules! writit {
        ($pix:ty) => {{
            let mut image = encoder.new_image_with_compression::<$pix, Deflate>(ncols as u32, nrows as u32, Deflate::default())?;
            image.encoder().write_tag(Tag::GdalNodata, &nd.to_string()[..])?;
            // remember that geotrans is negative, but in tiff tags assumed to be positive
            image.encoder().write_tag(Tag::ModelPixelScaleTag, &[geotrans[1], -geotrans[5], 0.0][..])?;
            image.encoder().write_tag(Tag::ModelTiepointTag, &[0.0, 0.0, 0.0, geotrans[0], geotrans[3], 0.0][..])?;
            image.encoder().write_tag(Tag::GeoKeyDirectoryTag, geokeydir)?;
            image.encoder().write_tag(Tag::GeoAsciiParamsTag, &proj)?;
            image.write_data(cast_slice(data.as_slice().unwrap()))?;
        }};
    }

    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<u8>()  => writit!(Gray8),
        id if id == std::any::TypeId::of::<u16>() => writit!(Gray16),
        id if id == std::any::TypeId::of::<u32>() => writit!(Gray32),
        id if id == std::any::TypeId::of::<u64>() => writit!(Gray64),
        id if id == std::any::TypeId::of::<f32>() => writit!(Gray32Float),
        id if id == std::any::TypeId::of::<f64>() => writit!(Gray64Float),
        id if id == std::any::TypeId::of::<i8>()  => writit!(GrayI8),
        id if id == std::any::TypeId::of::<i16>() => writit!(GrayI16),
        id if id == std::any::TypeId::of::<i32>() => writit!(GrayI32),
        id if id == std::any::TypeId::of::<i64>() => writit!(GrayI64),
        _ => return Err(RasterError::UnsupportedType(format!("Cannot handle type {}", std::any::type_name::<T>())))
    };

    Ok(())
}


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
/// use ndarray::{array, Array2};
/// use hydro_analysis::fill_depressions;
///
/// let mut dem: Array2<f64> = array![
///         [10.0, 12.0, 10.0],
///         [12.0, 1.0, 12.0],
///         [10.0, 12.0, 10.0],
/// ];
/// let filled: Array2<f64> = array![
///         [10.0, 12.0, 10.0],
///         [12.0, 1.0, 12.0],
///         [10.0, 12.0, 10.0],
/// ];
/// fill_depressions(&mut dem, -3.0, 8.0, 8.0, true);
/// assert_eq!(dem, filled);
/// let mut dem: Array2<f64> = array![
///         [10.0, 12.0, 10.0, 10.0],
///         [12.0, 1.0, 10.0, 12.0],
///         [10.0, 12.0, 10.0, 9.0],
/// ];
/// let filled: Array2<f64> = array![
///         [10.0, 12.0, 10.0, 10.0],
///         [12.0, 10.0, 10.0, 12.0],
///         [10.0, 12.0, 10.0, 9.0],
/// ];
/// fill_depressions(&mut dem, -3.0, 8.0, 8.0, true);
/// for (x, y) in dem.iter().zip(filled.iter()) {
///    assert!((*x - *y).abs() < 1e-4);
/// }
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

    //let input = dem.clone();    // original whitebox used an input and output while doing fixing flats, don't think you really need it

    let dx = [1, 1, 1, 0, -1, -1, -1, 0];
    let dy = [-1, 0, 1, 1, 1, 0, -1, -1];

    // Find pit cells (we don't care about pits around edge, they won't be considered a pit). This step is parallelizable.
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
                        let cn = cell2.column as isize + dx[n];
                        let rn = cell2.row as isize + dy[n];
                        if rn < 0 || rn as usize >= rows || cn < 0 || cn as usize >= columns {
                            continue;
                        }
                        let cn = cn as usize;
                        let rn = rn as usize;
                        if visited[[rn, cn]] == 0 {
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
                            let cn = cell2.column as isize + dx[n];
                            let rn = cell2.row as isize + dy[n];
                            if rn < 0 || rn as usize >= rows || cn < 0 || cn as usize >= columns {
                                continue;
                            }
                            let cn = cn as usize;
                            let rn = rn as usize;
                            if visited[[rn, cn]] == 0 {
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
                        let cn = cell2.1 as isize + dx[n];
                        let rn = cell2.0 as isize + dy[n];
                        if rn < 0 || rn as usize >= rows || cn < 0 || cn as usize >= columns {
                            continue;
                        }
                        let cn = cn as usize;
                        let rn = rn as usize;
                        if visited[[rn, cn]] == 1 {
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
                        let cn = cell2.1 as isize + dx[n];
                        let rn = cell2.0 as isize + dy[n];
                        if rn < 0 || rn as usize >= rows || cn < 0 || cn as usize >= columns {
                            continue;
                        }
                        let cn = cn as usize;
                        let rn = rn as usize;
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

    if small_num > 0.0 && fix_flats {
        // Some of the potential outlets really will have lower cells.
        minheap.clear();
        while let Some(cell) = possible_outlets.pop() {
            let z = dem[[cell.0, cell.1]];
            let mut anoutlet = false;
            for n in 0..8 {
                let rn: isize = cell.0 as isize + dy[n];
                let cn: isize = cell.1 as isize + dx[n];
                if rn < 0 || rn as usize >= rows || cn < 0 || cn as usize >= columns {
                    continue;
                }
                let zn = dem[[rn as usize, cn as usize]];
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
                        let cn = cell2.column as isize + dx[n];
                        let rn = cell2.row as isize + dy[n];
                        if rn < 0 || rn as usize >= rows || cn < 0 || cn as usize >= columns {
                            continue;
                        }
                        let cn = cn as usize;
                        let rn = rn as usize;
                        if flats[[rn, cn]] != 3 {
                            let zn = dem[[rn, cn]];
                            if zn == z && zn != nodata {
                                minheap2.push(GridCell2 {
                                    row: rn,
                                    column: cn,
                                    z: z,
                                    priority: dem[[rn, cn]], // FIXME
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
                        let cn = cell2.column as isize + dx[n];
                        let rn = cell2.row as isize + dy[n];
                        if rn < 0 || rn as usize >= rows || cn < 0 || cn as usize >= columns {
                            continue;
                        }
                        let cn = cn as usize;
                        let rn = rn as usize;
                        if flats[[rn, cn]] != 3 {
                            let zn = dem[[rn, cn]];
                            if zn < z + small_num && zn >= cell2.z && zn != nodata {
                                minheap2.push(GridCell2 {
                                    row: rn,
                                    column: cn,
                                    z: cell2.z,
                                    priority: dem[[rn, cn]], // FIXME
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

/// Calculates the D8 flow direction from a digital elevation model (DEM).
///
/// More-or-less the contents of
/// [whitebox d8_pointer](https://github.com/jblindsay/whitebox-tools/blob/master/whitebox-tools-app/src/tools/hydro_analysis/d8_pointer.rs)
///
/// This function computes the D8 flow direction for each cell in the provided DEM:
///
/// | .  |  .  |  . |
/// |:--:|:---:|:--:|
/// | 64 | 128 | 1  |
/// | 32 |  0  | 2  |
/// | 16 |  8  | 4  |
///
/// Grid cells that have no lower neighbours are assigned a flow direction of zero. In a DEM that
/// has been pre-processed to remove all depressions and flat areas, this condition will only occur
/// along the edges of the grid.
///
/// Grid cells possessing the NoData value in the input DEM are assigned the NoData value in the
/// output image.
///
/// # Parameters
/// - `dem`: A 2D array representing the digital elevation model (DEM)
/// - `nodata`: The nodata in the DEM
/// - `resx`: The resolution of the DEM in the x-direction in meters
/// - `resy`: The resolution of the DEM in the y-direction in meters
///
/// # Returns
/// - A tuple containing:
/// - An `Array2<u8>` representing the D8 flow directions for each cell.
/// - A `u8` nodata value (255)
///
/// # Example
/// ```rust
/// use anyhow::Result;
/// use ndarray::{Array2, array};
/// use hydro_analysis::{d8_pointer};
/// let dem = Array2::from_shape_vec(
///     (3, 3),
///     vec![
///         11.0, 12.0, 10.0,
///         12.0, 13.0, 12.0,
///         10.5, 12.0, 11.0,
///     ],
/// ).expect("Failed to create DEM");
/// let d8: Array2<u8> = array![
///     [0, 2, 0],
///     [8, 1, 128],
///     [0, 32, 0],
/// ];
/// let nodata = -9999.0;
/// let resx = 8.0;
/// let resy = 8.0;
/// let (res, _nd) = d8_pointer(&dem, nodata, resx, resy);
/// assert_eq!(res, d8);
/// ```
pub fn d8_pointer(dem: &Array2<f64>, nodata: f64, resx: f64, resy: f64) -> (Array2<u8>, u8)
{
    let (nrows, ncols) = (dem.nrows(), dem.ncols());
    let out_nodata: u8 = 255;
    let mut d8: Array2<u8> = Array2::from_elem((nrows, ncols), out_nodata);

    let diag = (resx * resx + resy * resy).sqrt();
    let grid_lengths = [diag, resx, diag, resy, diag, resx, diag, resy];

    let dx = [1, 1, 1, 0, -1, -1, -1, 0];
    let dy = [-1, 0, 1, 1, 1, 0, -1, -1];

    d8.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(row, mut d8_row)| {
            for col in 0..ncols {
                let z = dem[[row, col]];
                if z == nodata {
                    continue;
                }

                let mut dir = 0;
                let mut max_slope = f64::MIN;
                for i in 0..8 {
                    let rn: isize = row as isize + dy[i];
                    let cn: isize = col as isize + dx[i];
                    if rn < 0 || rn >= nrows as isize || cn < 0 || cn >= ncols as isize {
                        continue;
                    }
                    let z_n = dem[[rn as usize, cn as usize]];
                    if z_n != nodata {
                        let slope = (z - z_n) / grid_lengths[i];
                        if slope > max_slope && slope > 0.0 {
                            max_slope = slope;
                            dir = i;
                        }
                    }
                }

                if max_slope >= 0.0 {
                    d8_row[col] = 1 << dir;
                } else {
                    d8_row[col] = 0u8;
                }
            }
        });

    return (d8, out_nodata);
}


/// Updates a D8 flow direction using a mask that indicates onland and ocean and return changed
/// flattened indices.
///
/// This function will ensure your downstream traces calculated from a D8 will stop at the coast.
/// Any cell whose direction leads us to the ocean will become a sink.  Any cell in the ocean will
/// be nodata
///
/// An alternative approach (which doesn't work) is to first mask the DEM with nodata and then
/// calculate the D8.  Unfortunately this results in downstream traces that get to the coast
/// (nodata) and then head along the coast, sometimes for ages before coming to a natural sink.
///
/// # Parameters
///   d8: A 2D u8 array representing the directions
///   nodata: The nodata for the d8 (probably 255)
///   onland: A 2D bool array representing the land
///
/// # Returns
/// nothing, but updates the d8
///
/// # Example
/// ```rust
/// use ndarray::Array2;
/// let mut d8 = Array2::from_shape_vec(
///     (3, 3),
///     vec![
///         4, 2, 1,
///         2, 2, 1,
///         1, 128, 4,
///     ],
/// ).expect("Failed to create D8");
/// let onland = Array2::from_shape_vec(
///     (3, 3),
///     vec![
///         true, true, false,
///         true, true, false,
///         true, true, false,
///     ],
/// ).expect("Failed to create onland");
/// let newd8 = Array2::from_shape_vec(
///     (3, 3),
///     vec![
///         4, 0, 255,
///         2, 0, 255,
///         1, 128, 255,
///     ],
/// ).expect("Failed to create D8");
/// let changed = hydro_analysis::d8_clipped(&mut d8, 255, &onland);
/// assert_eq!(d8, newd8);
/// assert_eq!(changed, vec![1, 4]);
/// ```
pub fn d8_clipped(d8: &mut Array2<u8>, nodata: u8, onland: &Array2<bool>) -> Vec<usize>
{
    let (nrows, ncols) = (d8.nrows(), d8.ncols());

    // in the ocean is nodata
    par_azip!((d in &mut *d8, m in onland){ if !m { *d = nodata } });

    // so we can index using 1D 
    let slice = d8.as_slice().unwrap();

    let tozero: Vec<usize> = (0..nrows).into_par_iter().flat_map(|r| {
        let mut local: Vec<usize> = Vec::new();
        for c in 0..ncols {
            let idx = r*ncols + c;
            if slice[idx] == nodata || slice[idx] == 0 {
                continue;
            }

            // get next cell downstream
            let (dr, dc) = match slice[idx] {
                1   => (-1,  1),
                2   => ( 0,  1),
                4   => ( 1,  1),
                8   => ( 1,  0),
                16  => ( 1, -1),
                32  => ( 0, -1),
                64  => (-1, -1),
                128 => (-1,  0),
                _ => unreachable!(),
            };
            let rn = r as isize + dr;
            let cn = c as isize + dc;

            // if next is outside, don't change idx
            if rn < 0 || rn >= nrows as isize || cn < 0 || cn >= ncols as isize {
                continue;
            }
            // next is inside and nodata, then set idx to a sink
            if d8[[rn as usize, cn as usize]] == nodata {
                local.push(idx);
            }
        }
        local
    }).collect();

    let slice = d8.as_slice_mut().unwrap();
    for idx in &tozero {
        slice[*idx] = 0;
    }

    tozero
}


/// Breach depressions least cost.  Implements
///
/// [whitebox breach_depressions_least_cost](https://github.com/jblindsay/whitebox-tools/blob/master/whitebox-tools-app/src/tools/hydro_analysis/breach_depressions_least_cost.rs)
///
/// with
///   max_cost set to infinity,
///   flat_increment is default, and
///   minimize_dist set to true.
///
/// with more modern and less memory intensive datastructures.
///
/// The only real parameter to tune is max_dist (measured in cells) and a default would be 20
///
/// # Returns
///  number of pits that remain
///
/// # Examples
///
/// See tests/breach_depressions.rs for lots of examples, here is just one
///
/// ```rust
/// use ndarray::{Array2, array};
/// use hydro_analysis::breach_depressions;
/// let resx = 8.0;
/// let resy = 8.0;
/// let max_dist = 100;
/// let ep = 0.00000012;
/// let mut dem: Array2<f64> = array![
///     [2.0, 2.0, 2.0, 2.0, 0.0],
///     [2.0, 1.0, 2.0, 0.5, 0.0],
///     [2.0, 2.0, 2.0, 2.0, 0.0],
/// ];
/// let breached: Array2<f64> = array![
///     [2.0, 2.0, 2.0, 2.0, 0.0],
///     [2.0, 2.0-ep, 2.0-2.0*ep, 0.5, 0.0],
///     [2.0, 2.0, 2.0, 2.0, 0.0],
/// ];
/// let n = breach_depressions(&mut dem, -1.0, resx, resy, max_dist);
/// assert_eq!(n, 0);
/// for (x, y) in dem.iter().zip(breached.iter()) {
///    assert!((*x - *y).abs() < 1e-10);
/// }
///
pub fn breach_depressions(dem: &mut Array2<f64>, nodata: f64, resx: f64, resy: f64, max_dist: usize) -> usize
{

    let diagres: f64 = (resx * resx + resy * resy).sqrt();
    let cost_dist = [diagres, resx, diagres, resy, diagres, resx, diagres, resy];

    let (nrows, ncols) = (dem.nrows(), dem.ncols());
    let small_num = {
        let diagres = (resx * resx + resy * resy).sqrt();
        let elev_digits = (dem.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as i64).to_string().len();
        let elev_multiplier = 10.0_f64.powi((9 - elev_digits) as i32);
        1.0_f64 / elev_multiplier as f64 * diagres.ceil()
    };

    // want to raise interior (ie not on boundary or with a nodata neighbour) pit cells up to
    // just below minimum neighbour height.
    //
    // so first find the pits
    let dx = [1, 1, 1, 0, -1, -1, -1, 0];
    let dy = [-1, 0, 1, 1, 1, 0, -1, -1];
	let mut pits: Vec<_> = (1..nrows - 1).into_par_iter().flat_map(|row| {
        let mut local_pits = Vec::new();
        for col in 1..ncols - 1 {
            let z = dem[[row, col]];
            if z == nodata {
                continue;
            }
            let mut apit = true;
            // is any neighbour lower than me or nodata?
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

    // set depth to just below min neighbour, can't do this in parallel, we update the dem and pits
    for &mut (row, col, ref mut z) in pits.iter_mut() {
        let min_zn: f64 = dx.iter().zip(&dy).map(|(&dxi, &dyi)|
            dem[[
                (row as isize + dyi) as usize,
                (col as isize + dxi) as usize,
            ]]
        ).fold(f64::INFINITY, f64::min);
        *z = min_zn - small_num;
        dem[[row, col]] = *z;
    }

    // Sort highest to lowest so poping off the end will get the lowest, should do that first
    // because might solve some higher ones
    pits.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Equal));

    // keep track of number we can't deal with to return
    let mut num_unsolved: usize = 0;

    // roughly how many cells we will have to look at
    let num_to_chk: usize = (max_dist * 2 + 1) * (max_dist * 2 + 1);

    // for keeping track of path be make
    #[derive(Debug)]
    struct NodeInfo {
        prev: Option<(usize, usize)>,
        length: usize,
    }

    // try and dig a channel from row, col
    while let Some((row, col, z)) = pits.pop() {

        // May have been solved during previous depression step, so can skip
        if dx.iter().zip(&dy).any(|(&dxi, &dyi)|
            dem[[(row as isize + dyi) as usize, (col as isize + dxi) as usize]] < z
        ) {
            continue;
        }

        // keep our path info in here
        let mut visited: HashMap<(usize,usize), NodeInfo> = HashMap::default();
        visited.insert((row,col), NodeInfo {
            prev: None,
            length: 0,
        });

        // cells we will check to see if lower than (row, col, z)
        let mut cells_to_chk = BinaryHeap::with_capacity(num_to_chk);
        cells_to_chk.push(GridCell {row: row, column: col, priority: 0.0 });

        let mut dugit = false;
        'search: while let Some(GridCell {row: chkrow, column: chkcol, priority: accum}) = cells_to_chk.pop() {

            let length: usize = visited[&(chkrow,chkcol)].length;
            let zn: f64 = dem[[chkrow, chkcol]];
            let cost1: f64 = zn - z + length as f64 * small_num;
            let mut breach: Option<(usize, usize)> = None;

            // lets look about chkrow, chkcol
            for n in 0..8 {
                let rn: isize = chkrow as isize + dy[n];
                let cn: isize = chkcol as isize + dx[n];

                // chkrow, chkcol is a breach if on the edge
                if rn < 0 || rn >= nrows as isize || cn < 0 || cn >= ncols as isize {
                    breach = Some((chkrow, chkcol));
                    // we need to force this boundary cell down
                    dem[[chkrow, chkcol]] = z - (length as f64 * small_num);
                    break;
                }
                let rn: usize = rn as usize;
                let cn: usize = cn as usize;
                let nextlen = length + 1;

                // insert if not visited
                if let Vacant(e) = visited.entry((rn, cn)) {
                    e.insert(NodeInfo {
                        prev: Some((chkrow, chkcol)),
                        length: nextlen
                    });
                } else {
                    continue;
                }

                let zn: f64 = dem[[rn, cn]];
                // an internal nodata cannot be breach point
                if zn == nodata {
                    continue;
                }

                // zout is lowered from z by nextlen * slope
                let zout: f64 = z - (nextlen as f64 * small_num);
                if zn <= zout {
                    breach = Some((rn, cn));
                    break;
                }

                // (rn, cn) no good, zn is too high, just push onto heap
                if zn > zout {
                    let cost2: f64 = zn - zout;
                    let new_cost: f64 = accum + (cost1 + cost2) / 2.0 * cost_dist[n];
                    // but haven't travelled too far, so we will scan from this cell
                    if nextlen <= max_dist {
                        cells_to_chk.push(GridCell {
                            row: rn,
                            column: cn,
                            priority: new_cost
                        });
                    }
                }
            }

            if let Some(mut cur) = breach {
                loop {
                    let node: &NodeInfo = &visited[&cur];
                    // back at the pit?
                    let Some(parent) = node.prev else {
                        break;
                    };

                    let zn = dem[[parent.0, parent.1]];
                    let length = visited[&parent].length;
                    let zout = z - (length as f64 * small_num);
                    if zn > zout {
                        dem[[parent.0, parent.1]] = zout;
                    }
                    cur = parent;
                }
                dugit = true;
                break 'search;
            }

        }

        if !dugit {
            // Didn't find any lower cells, tough luck
            num_unsolved += 1;
        }
    }

    num_unsolved
}

