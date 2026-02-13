#[cfg(test)]
mod tests {
    use hydro_analysis::breach_depressions;
    use ndarray::{Array2, array};

    #[test]
    fn test_single() {
        let nodata = -999.0;
        let resx = 8.0;
        let resy = 8.0;
        let max_dist = 100;

        let mut dem: Array2<f64> = array![
            [1.0],
        ];
        let n = breach_depressions(&mut dem, nodata, resx, resy, max_dist);
        assert_eq!(n, 0);
        assert_eq!(dem, array![[1.0]]);
    }

    #[test]
    fn test_no_pits() {
        let nodata = -999.0;
        let resx = 8.0;
        let resy = 8.0;
        let max_dist = 100;

        let mut dem: Array2<f64> = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let orig = dem.clone();
        let n = breach_depressions(&mut dem, nodata, resx, resy, max_dist);
        assert_eq!(n, 0);
        assert_eq!(dem, orig);
    }

    #[test]
    fn test_simple_single_pit() {
        let nodata = -999.0;
        let resx = 8.0;
        let resy = 8.0;
        let max_dist = 100;

        let ep = 0.00000012;
        let mut dem: Array2<f64> = array![
            [2.0, 2.0, 2.0],
            [2.0, 1.0, 1.9],
            [2.0, 2.0, 2.0],
        ];
        let breached: Array2<f64> = array![
            [2.0, 2.0, 2.0],
            [2.0, 1.9-ep, 1.9-2.0*ep],
            [2.0, 2.0, 2.0],
        ];
        let n = breach_depressions(&mut dem, nodata, resx, resy, max_dist);
        assert_eq!(n, 0);
        for (x, y) in dem.iter().zip(breached.iter()) {
           assert!((*x - *y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_single_pit_further_on_edge() {
        let nodata = -999.0;
        let resx = 8.0;
        let resy = 8.0;
        let max_dist = 100;
        let ep = 0.00000012;

        let mut dem: Array2<f64> = array![
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 1.0, 2.0, 0.5],
            [2.0, 2.0, 2.0, 2.0],
        ];
        let breached: Array2<f64> = array![
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0-ep, 2.0-2.0*ep, 0.5],
            [2.0, 2.0, 2.0, 2.0],
        ];
        let n = breach_depressions(&mut dem, nodata, resx, resy, max_dist);
        assert_eq!(n, 0);
        for (x, y) in dem.iter().zip(breached.iter()) {
           assert!((*x - *y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_single_pit_in_larger() {
        let nodata = -999.0;
        let resx = 8.0;
        let resy = 8.0;

        let ep = 0.0000012;
        let mut dem: Array2<f64> = array![
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0,  8.0,  8.0,  8.0, 10.0],
            [10.0,  8.0,  5.0,  8.0, 10.0],
            [10.0,  8.0,  8.0,  8.0, 10.0],
            [10.0, 10.0,  9.0, 10.0, 10.0],
        ];
        let breached: Array2<f64> = array![
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0,  8.0,  8.0,  8.0, 10.0],
            [10.0,  8.0,  8.0-ep,  8.0, 10.0],
            [10.0,  8.0,  8.0,  8.0, 10.0],
            [10.0, 10.0,  9.0, 10.0, 10.0],
        ];
        let n = breach_depressions(&mut dem, nodata, resx, resy, 1);
        assert_eq!(n, 1);
        for (x, y) in dem.iter().zip(breached.iter()) {
           assert!((*x - *y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_single_pit_in_larger_to_edge() {
        let nodata = -999.0;
        let resx = 8.0;
        let resy = 8.0;
        let max_dist = 100;

        let ep = 0.0000012;
        let mut dem: Array2<f64> = array![
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0,  8.0,  8.0,  8.0, 10.0],
            [10.0,  8.0,  5.0,  8.0, 10.0],
            [10.0,  8.0,  8.0,  8.0, 10.0],
            [10.0, 10.0,  9.0, 10.0, 10.0],
        ];
        let breached: Array2<f64> = array![
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0,  8.0,  8.0,  8.0, 10.0],
            [10.0,  8.0,  8.0-ep,  8.0, 10.0],
            [10.0,  8.0,  8.0-2.0*ep,  8.0, 10.0],
            [10.0, 10.0,  8.0-3.0*ep, 10.0, 10.0],
        ];
        let n = breach_depressions(&mut dem, nodata, resx, resy, max_dist);
        assert_eq!(n, 0);
        for (x, y) in dem.iter().zip(breached.iter()) {
           assert!((*x - *y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_two_pits() {
        let nodata = -999.0;
        let resx = 8.0;
        let resy = 8.0;
        let max_dist = 100;

        let ep = 0.0000012;
        let mut dem: Array2<f64> = array![
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0,  8.0,  8.0,  8.0, 10.0, 10.0, 10.0],
            [10.0,  8.0,  1.0,  8.0, 10.0,  8.0, 10.0],
            [10.0,  8.0,  8.0,  7.0, 10.0,  7.9, 10.0],
            [10.0, 10.0, 10.0, 10.0,  6.0,  7.8, 10.0],
            [10.0, 10.0, 11.0, 10.0,  8.0,  5.0,  9.0],
            [10.0,  9.0, 12.0, 10.0, 10.0, 10.0, 10.0],
        ];
        let breached: Array2<f64> = array![
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0,  8.0,  8.0,  8.0, 10.0, 10.0, 10.0],
            [10.0,  8.0,  7.0-ep, 8.0, 10.0,  8.0, 10.0],
            [10.0,  8.0,  8.0,  7.0-2.0*ep, 10.0,  7.9, 10.0],
            [10.0, 10.0, 10.0, 10.0,  6.0,  7.8, 10.0],
            [10.0, 10.0, 11.0, 10.0,  8.0,  6.0-ep, 6.0-2.0*ep],
            [10.0,  9.0, 12.0, 10.0, 10.0, 10.0, 10.0],
        ];
        let n = breach_depressions(&mut dem, nodata, resx, resy, max_dist);
        assert_eq!(n, 0);
        for (x, y) in dem.iter().zip(breached.iter()) {
           assert!((*x - *y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nodata_in_the_way() {
        let resx = 8.0;
        let resy = 8.0;
        let max_dist = 100;

        let ep = 0.00000012;
        let mut dem: Array2<f64> = array![
            [2.0, 2.0, 2.0, 3.0],
            [2.0, 1.0, 2.0, 3.0],
            [2.0, 2.0, 1.5, 3.0],
            [2.0, 1.8, -1.0,3.0],
        ];
        let breached: Array2<f64> = array![
            [2.0, 2.0, 2.0, 3.0],
            [2.0, 1.5-ep, 2.0, 3.0],
            [2.0, 2.0, 1.5-2.0*ep, 3.0],
            [2.0, 1.5-3.0*ep, -1.0,3.0],
        ];
        let n = breach_depressions(&mut dem, -1.0, resx, resy, max_dist);
        assert_eq!(n, 0);
        for (x, y) in dem.iter().zip(breached.iter()) {
           assert!((*x - *y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_more_nodata_inway() {
        let resx = 8.0;
        let resy = 8.0;
        let max_dist = 100;

        let ep = 0.00000012;
        let mut dem: Array2<f64> = array![
            [2.0, 2.0, 2.0, 3.0],
            [1.9, 1.0, 2.0, -1.0],
            [2.0, 2.0, 1.5, -1.0],
            [2.0, -1.0, -1.0, -1.0],
        ];
        let breached: Array2<f64> = array![
            [2.0, 2.0, 2.0, 3.0],
            [1.5-2.0*ep, 1.5-ep, 2.0, -1.0],
            [2.0, 2.0, 1.5, -1.0],
            [2.0, -1.0, -1.0, -1.0],
        ];
        let n = breach_depressions(&mut dem, -1.0, resx, resy, max_dist);
        assert_eq!(n, 0);
        println!("{:?}", dem);
        for (x, y) in dem.iter().zip(breached.iter()) {
           assert!((*x - *y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_no_pit_since_nodata() {
        let resx = 8.0;
        let resy = 8.0;
        let max_dist = 100;

        let mut dem: Array2<f64> = array![
            [2.0, 2.0, 2.0],
            [2.0, 1.0, 2.0],
            [2.0, 2.0, 2.0],
        ];
        let orig = dem.clone();
        let n = breach_depressions(&mut dem, 2.0, resx, resy, max_dist);
        assert_eq!(n, 0);
        println!("{:?}", dem);
        assert_eq!(dem, orig);
    }

}
