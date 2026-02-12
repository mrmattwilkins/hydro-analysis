use anyhow::Result;
use ndarray::{Array2, array};
use hydro_analysis::{fill_depressions};

fn main() -> Result<()> {
    let mut dem: Array2<f64> = array![
        [3.0, 3.0, 3.0, 7.0],
        [3.0, 2.0, 4.0, 2.0],
        [5.0, 6.0, 8.0, 0.0],
    ];
    let nd: f64 = -100.0;
    let resx: f64 = 8.0;
    let resy: f64 = 8.0;
    println!("Filling depressions on {dem}");
    fill_depressions(&mut dem, nd, resx, resy, true);
    println!("now is {dem}");

    let mut dem: Array2<f64> = array![                                                                                                       
        [10.0, 12.0, 10.0, 10.0],                                                                                                              
        [12.0, 1.0,  10.0, 12.0],                                                                                                               
        [10.0, 12.0, 10.0, 11.0],                                                                                                              
    ];
    println!("Filling depressions on {dem}");
    fill_depressions(&mut dem, nd, resx, resy, true);
    println!("now is {dem}");

    Ok(())
}



