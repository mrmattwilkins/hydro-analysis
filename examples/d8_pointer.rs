use anyhow::Result;
use ndarray::{Array2, array};
use hydro_analysis::{d8_pointer};

fn main() -> Result<()> {
    let filled: Array2<f64> = array![
        [2.0, 3.0, 3.0, 7.0],
        [3.0, 5.0, 4.0, 2.0],
        [5.0, 6.0, 8.0, 0.0],
    ];
    let nd: f64 = -100.0;
    let resx: f64 = 8.0;
    let resy: f64 = 8.0;
    println!("Running d8 on {filled}");
    let (d8, _d8_nd) = d8_pointer(&filled, nd, resx, resy);
    println!("d8 is {d8}");

    Ok(())
}



