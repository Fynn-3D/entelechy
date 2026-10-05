//! A Kernel past the size a device's compiler takes in comfort is said to
//! be large, with its largest functions and the calls that reach each.

#[test]
fn a_large_kernel_names_its_largest_functions_and_how_they_are_reached() {
    // A function too large to be written out where it is called.
    let steps: String = (1..60)
        .map(|i| {
            format!(
                "    let a{i} = sin(a{}) * {i}.0 + cos(a{}) * x;\n",
                i - 1,
                i - 1
            )
        })
        .collect();
    let source = format!(
        "fn heavy(x: Float) -> Float {{
    let a0 = x;
{steps}    return a59;
}}
fn middle(x: Float) -> Float {{ return heavy(x) + heavy(x * 2.0); }}
kernel t.k@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ let x = float(element()); return middle(x) + middle(x + 1.0); }} }}"
    );
    let module = orla_compiler::compile(&source).unwrap();
    assert!(
        orla_session::large(&module).is_none(),
        "a small Kernel is not large"
    );
    let said = orla_session::large_past(&module, 50).expect("past fifty instructions");
    assert!(said.contains("`t.k@1` is"), "{said}");
    // `middle` is written out where it is called; `heavy` is not.
    assert!(
        said.contains("`heavy` (") && said.contains("called by the Kernel itself"),
        "{said}"
    );
}
