# Hydro Analysis

`hydro-analysis` provides functions for Hydrology DEM manipulation.  There are
a couple generic functions for reading/writing raster files of any common
primative type (which surprizingly I couldn't find anywhere else, unless you
use GDAL which I am trying to avoid).  Also there are a couple functions based
on [whitebox](https://github.com/jblindsay/whitebox-tools).  Whitebox is a
command line tool, this provides functionality via functions so can be called
from your code.

## Example of reading and writing rasters

```
let ifn = PathBuf::from("input.tif");
let (d8, nd, crs, geo, gdir, proj) = rasterfile_to_array::<u8>(&ifn)?;
/* do something with d8, or make a new array2 */
let ofn = PathBuf::from("output.tif");
if let Err(e) = array_to_rasterfile::<u8>(&d8, nd, &geo, &gdir, &proj, &ofn) {
	eprintln!("Error occured while writing {}: {:?}", ofn.display(), e);
}
```

## Example of filling and d8

```
use ndarray::Array2;
use hydro_analysis::{fill_depressions, d8_pointer};

let mut dem = Array2::from_shape_vec(
    (3, 3),
    vec![
        10.0, 12.0, 10.0,
        12.0, 9.0,  12.0,
        10.0, 12.0, 10.0,
    ],
).expect("Failed to create DEM");

fill_depressions(&mut dem, -3.0, 8.0, 8.0, true);
let (d8, d8_nd) = d8_pointer(&dem, -1.0, 8.0, 8.0);
```


