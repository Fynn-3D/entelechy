//! A value is made of its scalars: `matches_scalars` answers as comparing
//! `scalars()` does, for every kind of value.

use entelechy_abi::{Scalar, Value};

#[test]
fn a_value_matches_its_own_scalars_and_no_others() {
    let values = [
        Value::Float(1.5),
        Value::Int(-3),
        Value::Bool(true),
        Value::Double(2.25),
        Value::Long(7),
        Value::Vec3([1.0, 2.0, 3.0]),
        Value::IVec2([4, 5]),
        Value::DVec2([1.0, 2.0]),
        Value::Mat2([1.0, 0.0, 0.0, 1.0]),
        Value::Record(vec![
            Value::Float(1.0),
            Value::IVec2([1, 2]),
            Value::Long(3),
        ]),
    ];
    for value in &values {
        let own = value.scalars();
        assert!(value.matches_scalars(&own), "{value:?}");
        let mut longer = own.clone();
        longer.push(Scalar::Float);
        assert!(!value.matches_scalars(&longer), "{value:?}");
        assert!(!value.matches_scalars(&own[..own.len() - 1]), "{value:?}");
        let other = own
            .iter()
            .map(|s| {
                if *s == Scalar::Int {
                    Scalar::Float
                } else {
                    Scalar::Int
                }
            })
            .collect::<Vec<_>>();
        assert!(!value.matches_scalars(&other), "{value:?}");
    }
}
