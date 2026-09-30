"""SPEC §2 tool-calling contract: every returned tool call is validated
client-side before execution — name in the offered set, deduped, arguments
JSON-parseable, required params present, per-tool checks pass. Failures are
collected as bounce-back messages for the model, never surfaced to the user.
"""

from phd_helper.toolcall import validate_tool_calls


def call(name, arguments, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": arguments}}


OFFERED = {"section_read": {"section": "string"},
           "section_write": {"section": "string", "find": "string",
                             "replace": "string"}}


def test_wrongly_typed_param_rejected_with_a_bounce():
    # {"query": 5} must bounce to the model, not reach the tool and raise
    # out of the turn (review 97ea4d0: silent turn death).
    valid, errors = validate_tool_calls(
        [call("section_read", '{"section": 5}')], OFFERED)
    assert valid == []
    assert len(errors) == 1
    assert "section" in errors[0] and "string" in errors[0]


def test_null_param_rejected():
    valid, errors = validate_tool_calls(
        [call("section_read", '{"section": null}')], OFFERED)
    assert valid == []
    assert len(errors) == 1


def test_valid_call_passes_through():
    valid, errors = validate_tool_calls(
        [call("section_read", '{"section": "intro.tex"}')], OFFERED)
    assert errors == []
    assert len(valid) == 1
    assert valid[0].name == "section_read"
    assert valid[0].args == {"section": "intro.tex"}


def test_unknown_name_rejected():
    valid, errors = validate_tool_calls(
        [call("delete_everything", "{}")], OFFERED)
    assert valid == []
    assert len(errors) == 1
    assert "delete_everything" in errors[0]


def test_unparseable_arguments_rejected():
    valid, errors = validate_tool_calls(
        [call("section_read", '{"section": "intro.tex')], OFFERED)
    assert valid == []
    assert len(errors) == 1


def test_missing_required_param_rejected():
    valid, errors = validate_tool_calls(
        [call("section_write", '{"section": "intro.tex", "find": "a"}')],
        OFFERED)
    assert valid == []
    assert len(errors) == 1
    assert "replace" in errors[0]


def test_non_object_arguments_rejected():
    valid, errors = validate_tool_calls(
        [call("section_read", '"just a string"')], OFFERED)
    assert valid == []
    assert len(errors) == 1


def test_duplicate_calls_deduped():
    dupes = [call("section_read", '{"section": "intro.tex"}', "c1"),
             call("section_read", '{"section": "intro.tex"}', "c2")]
    valid, errors = validate_tool_calls(dupes, OFFERED)
    assert len(valid) == 1
    assert errors == []


def test_same_name_different_args_both_kept():
    calls = [call("section_write",
                  '{"section": "a.tex", "find": "x", "replace": "y"}', "c1"),
             call("section_write",
                  '{"section": "b.tex", "find": "x", "replace": "y"}', "c2")]
    valid, errors = validate_tool_calls(calls, OFFERED)
    assert len(valid) == 2


def test_per_tool_validator_rejects_and_reports_reason():
    def find_must_exist(args):
        if args.get("find") not in "the live section text":
            return "find anchor not present in the section"
        return None

    valid, errors = validate_tool_calls(
        [call("section_write",
              '{"section": "intro.tex", "find": "gone", "replace": "y"}')],
        OFFERED, validators={"section_write": find_must_exist})
    assert valid == []
    assert "find anchor not present" in errors[0]


def test_errors_are_bounce_ready_one_per_rejection():
    calls = [call("nope", "{}"),
             call("section_read", "not json")]
    valid, errors = validate_tool_calls(calls, OFFERED)
    assert valid == []
    assert len(errors) == 2
