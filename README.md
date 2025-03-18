# Hydro Analysis

`hydro-analysis` provides functions for Hydrology DEM manipulation that are based on
[whitebox](https://github.com/jblindsay/whitebox-tools).  Whitebox is a command line tool, this
crate provides some (only a couple functions at present) of that functionality via functions so
can be called from your code.

## Example

```
use ndarray::Array2;
use hydro_analysis::fill_depressions;

let mut dem = Array2::from_shape_vec(
    (3, 3),
    vec![
        10.0, 12.0, 10.0,
        12.0, 9.0,  12.0,
        10.0, 12.0, 10.0,
    ],
).expect("Failed to create DEM");

fill_depressions(&mut dem, -3.0, 8.0, 8.0, true);
```
