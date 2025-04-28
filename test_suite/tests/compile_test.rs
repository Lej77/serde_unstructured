/// Tests that check if some code fails to compile.
#[test]
#[cfg_attr(miri, ignore)]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/**/*.rs");
}
