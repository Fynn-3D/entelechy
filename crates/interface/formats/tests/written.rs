//! What a path may be written as is what `encode` writes: every extension
//! `WRITTEN` names is one it knows, and any other is refused before a
//! scene runs; a data table is written only knowing its columns' names.

use std::path::Path;

use entelechy_formats::{EncodeError, WRITTEN, encode, writable};
use entelechy_host::{Domain, FieldData, Scalar, Schema};

#[test]
fn every_written_extension_is_one_encode_knows() {
    let data = FieldData::from_storage(
        Schema::new(vec![Scalar::Float]),
        Domain::Rows(4),
        vec![0; 4],
    )
    .unwrap();
    let names = [String::new()];
    for extension in WRITTEN {
        let written = encode(&data, extension, None, Some(&names));
        assert!(!matches!(written, Err(EncodeError::Unknown)), "{extension}");
        assert!(
            writable(Path::new(&format!("out.{extension}"))).is_ok(),
            "{extension}"
        );
    }
    for other in ["txt", "jpg", ""] {
        assert!(
            matches!(encode(&data, other, None, None), Err(EncodeError::Unknown)),
            "{other}"
        );
    }
    for table in ["csv", "json"] {
        assert_eq!(encode(&data, table, None, None), Err(EncodeError::Unnamed));
    }
    let refused = writable(Path::new("/dev/null")).unwrap_err();
    assert!(
        refused.contains("/dev/null") && refused.contains(".png") && refused.contains(".csv"),
        "{refused}"
    );
}
