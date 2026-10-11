//! What a path may be written as is what `encode` writes: every extension
//! `WRITTEN` names is one it knows, and any other is refused before a
//! scene runs.

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
    for extension in WRITTEN {
        let written = encode(&data, extension, None);
        assert!(!matches!(written, Err(EncodeError::Unknown)), "{extension}");
        assert!(
            writable(Path::new(&format!("out.{extension}"))).is_ok(),
            "{extension}"
        );
    }
    for other in ["txt", "jpg", ""] {
        assert!(
            matches!(encode(&data, other, None), Err(EncodeError::Unknown)),
            "{other}"
        );
    }
    let refused = writable(Path::new("/dev/null")).unwrap_err();
    assert!(
        refused.contains("/dev/null") && refused.contains(".png"),
        "{refused}"
    );
}
