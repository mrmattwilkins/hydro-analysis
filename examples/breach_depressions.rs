use anyhow::Result;
use ndarray::{Array2, array};
use hydro_analysis::breach_depressions;

fn main() -> Result<()> {
    let mut dem: Array2<f64> = array![
        [5.0, 5.0, 4.0, 4.0, 4.0, 3.0],
        [5.0, 5.0, 4.5, 2.0, 3.0, 2.0],
        [5.0, 3.0, 4.5, 4.0, 4.0, 3.0],
        [5.0, 4.0, 4.5, 4.0, 4.0, 3.0],
        [5.0, 1.0, 3.0, 4.0, 4.0, 3.0],
    ];

    let n = breach_depressions(&mut dem, -10.0, 1.0, 1.0, 10);
    println!("Number of pits left is {n}");
    println!("Dem is now {:?}", dem);

    Ok(())
}

