# Hydro Analysis

hydro-analysis provides functions for Hydrology DEM manipulation.  There are
a couple generic functions for reading/writing raster files of any common
primative type (which surprizingly I couldn't find anywhere else, unless you
use GDAL which I am trying to avoid).  There are a number of functions based
on [whitebox](https://github.com/jblindsay/whitebox-tools), but written using
more modern rust datastructures.   Whitebox is a
command line tool, this crate provides functionality via functions so can be called
from your code directly.  Functions implemented are:

* fill depressions
* breach depressions least cost
* d8 cliped
* d8 pointer

To get started see the [examples/](examples) folder

