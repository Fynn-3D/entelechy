//! A builder's shader, without a device: a long body is written once, however
//! many elements it builds, and a short array is written at named places.

use entelechy_compiler::compile;
use entelechy_gpu::wgsl::emit;

fn shader(count: u32) -> String {
    let source = format!(
        "kernel test.built@1 {{ requires {{ field.write; }} output result: Image;
        compute {{
            let c = coord();
            let a = [each {count} {{ let k = float(index()); let w = sin(c.x * k + 0.5) * cos(c.y - k) + k * k * 0.01; let z = mix(w, k, 0.25) + abs(w - c.x) * 2.0 - sqrt(abs(k - c.y)); z * z + w * 0.5 + k }}];
            return vec4(a[int(c.x) % {count}], a[0], 0.0, 1.0);
        }} }}"
    );
    let module = compile(&source).unwrap_or_else(|error| panic!("{error}"));
    emit(&module)
}

#[test]
fn a_long_body_is_written_once_however_many_elements_it_builds() {
    let (sixteen, sixty_four, a_hundred) = (shader(16), shader(64), shader(100));
    assert!(
        sixty_four.len() < sixteen.len() + 4096,
        "16 elements: {} bytes, 64: {} bytes",
        sixteen.len(),
        sixty_four.len()
    );
    assert!(
        sixty_four.contains("switch") && sixty_four.contains("[63] = "),
        "a short array is written at named places"
    );
    assert!(
        !a_hundred.contains("switch") && a_hundred.len() < sixteen.len() + 4096,
        "a long one at the counter's place, its body once"
    );
}
