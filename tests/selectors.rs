//! Selector identity and canonicalization properties.

use weftcodec::error::ErrorClass;
use weftcodec::limits::Limits;
use weftcodec::selector::{self, Expr, Literal};

fn id(source: &str) -> weftcodec::Hash32 {
    selector::parse(source).unwrap().selector_id().unwrap()
}

#[test]
fn whitespace_hex_case_and_offset_zero_are_identity_preserving() {
    assert_eq!(id("select attrs.x == 1"), id("  select   attrs.x==1  "));
    assert_eq!(
        id("select attrs.x == 1 offset 0"),
        id("select attrs.x == 1")
    );
    assert_eq!(
        id("select attrs.x == 5 order by attrs.y"),
        id("select attrs.x == 05 order by attrs.y asc")
    );
    assert_eq!(
        id("select relation(p, AB112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)"),
        id("select relation(p, obj:ab112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)")
    );
}

#[test]
fn operand_reordering_changes_identity() {
    assert_ne!(
        id("select a == 1 and b == 2"),
        id("select b == 2 and a == 1")
    );
    assert_ne!(id("select a == 1 or b == 2"), id("select b == 2 or a == 1"));
}

#[test]
fn grammar_forms_and_normalization() {
    let selector = selector::parse("select not not (a == 1 and (b == 2 and c == 3))").unwrap();
    match selector.predicate {
        Expr::And(items) => assert_eq!(items.len(), 3),
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(
        selector::parse("select not true").unwrap().predicate,
        Expr::False
    );
    assert_eq!(
        selector::parse("select not false").unwrap().predicate,
        Expr::True
    );

    let selector = selector::parse(
        "select a != -1 and b <= null and c >= \"x\" order by a.b.c desc limit 3 offset 7",
    )
    .unwrap();
    let Expr::And(items) = &selector.predicate else {
        panic!("expected and")
    };
    assert!(matches!(
        &items[0],
        Expr::Compare {
            value: Literal::I64(-1),
            ..
        }
    ));
    assert_eq!(selector.order.len(), 1);
    assert!(selector.order[0].desc);
    assert_eq!(selector.limit, Some(3));
    assert_eq!(selector.offset, Some(7));

    for source in [
        "select has(attrs.x)",
        "select tag(\"demo\")",
        "select tag(demo)",
        "select relation(p, 00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)",
        "select true",
        "select false",
        "select a == null",
        "select a == true",
        "select a == false",
    ] {
        selector::parse(source).unwrap_or_else(|error| panic!("{source}: {error}"));
    }
}

#[test]
fn parse_errors_carry_positions() {
    for source in [
        "select",
        "select a = 1",
        "select a == ",
        "select and",
        "select a == \"unterminated",
        "select a == \"\\uD800\"",
        "select a == \"\\uDC00\"",
        "select a == \"raw\ncontrol\"",
        "select a == \"\\q\"",
        "select a == 1 order by a, a",
        "select a == 1 limit 1 limit 2",
        "select a == 1 offset",
        "select a == 1 garbage",
        "select a == -",
        "select a == 99999999999999999999999999",
        "SELECT a == 1",
    ] {
        let error = selector::parse(source).unwrap_err();
        assert!(
            matches!(error.class(), ErrorClass::Parse | ErrorClass::LimitExceeded),
            "{source}: {:?}",
            error.class()
        );
        assert!(error.position().is_some(), "{source} has no position");
    }
    let long_field = (0..17)
        .map(|index| format!("f{index}"))
        .collect::<Vec<_>>()
        .join(".");
    let source = format!("select {long_field} == 1");
    assert_eq!(
        selector::parse(&source).unwrap_err().class(),
        ErrorClass::LimitExceeded
    );
}

#[test]
fn escapes_are_byte_exact_and_not_normalized() {
    let composed = selector::parse("select a == \"\\u00e9\"").unwrap();
    let decomposed = selector::parse("select a == \"e\\u0301\"").unwrap();
    assert_ne!(
        composed.selector_id().unwrap(),
        decomposed.selector_id().unwrap()
    );
    let escaped = selector::parse("select a == \"\\uD83D\\uDE00\\n\\t\\\"\\\\\\/\"").unwrap();
    match escaped.predicate {
        Expr::Compare {
            value: Literal::Str(value),
            ..
        } => assert_eq!(value, "😀\n\t\"\\/"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn canonical_bytes_decode_back_to_same_ast() {
    let sources = [
        "select true",
        "select attrs.x == 1 and attrs.y == 2",
        "select attrs.x == 1 or (attrs.y == 2 and attrs.z == 3)",
        "select not has(attrs.x)",
        "select relation(derived_from, obj:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)",
        "select attrs.x == 1 order by attrs.a desc, attrs.b limit 10 offset 4",
        "select attrs.x == \"\\u00e9\"",
    ];
    for source in sources {
        let ast = selector::parse(source).unwrap();
        let bytes = ast.canonical_bytes().unwrap();
        let decoded = selector::Selector::from_canonical_bytes(&bytes, &Limits::default()).unwrap();
        assert_eq!(decoded, ast, "{source}");
    }
}

#[test]
fn selector_limits() {
    let limits = Limits::default().with_max_selector_bytes(8);
    assert_eq!(
        selector::parse_with("select true", selector::DEFAULT_NAMESPACE, &limits)
            .unwrap_err()
            .class(),
        ErrorClass::LimitExceeded
    );
    let limits = Limits::default().with_max_selector_operands(2);
    assert_eq!(
        selector::parse_with(
            "select a == 1 and b == 2 and c == 3",
            selector::DEFAULT_NAMESPACE,
            &limits
        )
        .unwrap_err()
        .class(),
        ErrorClass::LimitExceeded
    );
    let limits = Limits::default().with_max_selector_depth(2);
    assert_eq!(
        selector::parse_with("select (((a == 1)))", selector::DEFAULT_NAMESPACE, &limits)
            .unwrap_err()
            .class(),
        ErrorClass::LimitExceeded
    );
}
