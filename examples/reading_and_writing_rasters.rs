use anyhow::Result;
use tempfile::NamedTempFile;
use std::path::PathBuf;
use ndarray::{Array2, array};
use hydro_analysis::{rasterfile_to_array, array_to_rasterfile};

fn main() -> Result<()> {
    let d8: Array2<u8> = array![
        [0, 3, 3, 7],
        [1, 4, 9, 2],
        [5, 6, 8, 0],
    ];
    let nd: u8 = 255;
    let crs: u16 = 2193;
    let geo: [f64;6] = [1361171.0, 8.0, 0.0, 5006315.0, 0.0, -8.0];
    let gdir  = [1u64, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 1026, 34737, 48, 0, 2049, 34737, 9, 48, 2054, 0, 1, 9102, 3072, 0, 1, 2193, 3076, 0, 1, 9001];
    let proj: &str = "NZGD2000 / New Zealand Transverse Mercator 2000|NZGD2000|";
    let tmp = NamedTempFile::new()?;
    let ofn: PathBuf = tmp.path().to_path_buf();
    println!("Writing array to {:?}", ofn);
    array_to_rasterfile::<u8>(&d8, nd, &geo, &gdir, &proj, &ofn)?;

    // read file back in
    println!("Reading {:?} into new array and checking got same values", ofn);
    let (d8_new, nd_new, crs_new, geo_new, gdir_new, proj_new) = rasterfile_to_array::<u8>(&ofn)?;

    // checking got same values back
    assert_eq!(d8_new.shape(), d8.shape());
    assert_eq!(d8_new, d8);

    assert_eq!(nd_new, nd);
    assert_eq!(crs_new, crs);
    assert_eq!(geo_new, geo);
    assert_eq!(gdir_new, gdir);
    assert_eq!(proj_new, proj);


    tmp.close()?;
/*
    let dem: Array2<f64> = array![                                                                                                       
        [10.0, 12.0, 10.0, 10.0],                                                                                                            
        [12.0, 1.0,  10.0, 12.0],                                                                                                            
        [10.0, 12.0, 10.0, 9.0],                                                                                                            
    ];
    let nd: f64 = -100.0;
    let crs: u16 = 2193;
    let geo: [f64;6] = [1361171.0, 8.0, 0.0, 5006315.0, 0.0, -8.0];
    let gdir  = [1u64, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 1026, 34737, 48, 0, 2049, 34737, 9, 48, 2054, 0, 1, 9102, 3072, 0, 1, 2193, 3076, 0, 1, 9001];
    let proj: &str = "NZGD2000 / New Zealand Transverse Mercator 2000|NZGD2000|";
    let ofn: PathBuf = "/tmp/it.tif".into();
    array_to_rasterfile::<f64>(&dem, nd, &geo, &gdir, &proj, &ofn)?;
*/
    //let (blah, nd_new, crs_new, geo_new, gdir_new, proj_new) = rasterfile_to_array::<f64>(&"/tmp/out.tif".into())?;

    Ok(())
}
