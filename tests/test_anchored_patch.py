"""Anchored patch: the default write into a section (SPEC §5).

Anchored patch = exact find/replace whose untouched prose stays byte-identical.
"""

from phd_helper.patches import AnchoredPatch, apply_patch, inverse_patch, section_hash

SECTION = (
    "\\section{Method}\n"
    "\n"
    "We train the model on the full corpus.\n"
    "Validation uses the held-out split.\n"
    "\n"
    "\\begin{equation}\n"
    "  y = Wx + b\n"
    "\\end{equation}\n"
)


def test_patch_applies_to_unchanged_section_and_untouched_prose_is_byte_identical():
    patch = AnchoredPatch(
        find="We train the model on the full corpus.",
        replace="We train the model on the filtered corpus.",
        base_hash=section_hash(SECTION),
    )

    result = apply_patch(SECTION, patch)

    assert result.applied is True
    assert result.text == (
        "\\section{Method}\n"
        "\n"
        "We train the model on the filtered corpus.\n"
        "Validation uses the held-out split.\n"
        "\n"
        "\\begin{equation}\n"
        "  y = Wx + b\n"
        "\\end{equation}\n"
    )


def test_patch_whose_find_is_not_in_the_section_is_rejected_with_a_reason():
    patch = AnchoredPatch(
        find="We train the model on the whole corpus.",  # never in this section
        replace="We train the model on the filtered corpus.",
        base_hash=section_hash(SECTION),
    )

    result = apply_patch(SECTION, patch)

    assert result.applied is False
    assert result.text is None
    assert result.reason


def test_patch_whose_find_matches_twice_is_rejected_as_ambiguous():
    section = (
        "\\section{Method}\n"
        "\n"
        "We use a fixed seed. This keeps runs comparable.\n"
        "We use a fixed seed. This keeps runs comparable.\n"
    )
    patch = AnchoredPatch(
        find="We use a fixed seed.",
        replace="We use a fixed random seed.",
        base_hash=section_hash(section),
    )

    result = apply_patch(section, patch)

    assert result.applied is False
    assert result.text is None
    assert result.reason


def test_patch_reanchors_and_applies_when_the_user_edited_elsewhere_since_it_was_computed():
    stale_base = "We train the model on the full corpus.\n"
    edited_by_user = (
        "We train the model on the full corpus.\n"
        "\n"
        "Added by the user after the patch was computed.\n"
    )
    patch = AnchoredPatch(
        find="We train the model on the full corpus.",
        replace="We train the model on the filtered corpus.",
        base_hash=section_hash(stale_base),
    )

    result = apply_patch(edited_by_user, patch)

    assert result.applied is True
    assert result.text == (
        "We train the model on the filtered corpus.\n"
        "\n"
        "Added by the user after the patch was computed.\n"
    )


def test_patch_reanchors_to_a_unique_close_match_and_keeps_the_users_edit_in_the_span():
    stale_base = "We train the model on the full corpus.\n"
    user_edited_span = "We train the model quickly on the full corpus.\n"
    patch = AnchoredPatch(
        find="We train the model on the full corpus.",
        replace="We train the model on the filtered corpus.",
        base_hash=section_hash(stale_base),
    )

    result = apply_patch(user_edited_span, patch)

    assert result.applied is True
    assert result.text == "We train the model quickly on the filtered corpus.\n"


def test_patch_is_rejected_when_the_user_edited_exactly_what_the_patch_changes():
    stale_base = "We train the model on the full corpus.\n"
    user_edited_the_same_words = "We train the model on the entire corpus.\n"
    patch = AnchoredPatch(
        find="We train the model on the full corpus.",
        replace="We train the model on the filtered corpus.",
        base_hash=section_hash(stale_base),
    )

    result = apply_patch(user_edited_the_same_words, patch)

    assert result.applied is False
    assert result.text is None
    assert result.reason


def test_undo_of_an_apply_restores_the_section_byte_identically():
    patch = AnchoredPatch(
        find="We train the model on the full corpus.",
        replace="We train the model on the filtered corpus.",
        base_hash=section_hash(SECTION),
    )
    applied = apply_patch(SECTION, patch)
    assert applied.applied is True

    restored = apply_patch(applied.text, inverse_patch(applied.text, patch))

    assert restored.applied is True
    assert restored.text == SECTION


def test_undo_reanchors_when_the_user_edited_elsewhere_after_the_apply():
    patch = AnchoredPatch(
        find="We train the model on the full corpus.",
        replace="We train the model on the filtered corpus.",
        base_hash=section_hash(SECTION),
    )
    applied = apply_patch(SECTION, patch)
    user_note = "\nUser note added after the apply.\n"
    section_now = applied.text + user_note

    restored = apply_patch(section_now, inverse_patch(applied.text, patch))

    assert restored.applied is True
    assert restored.text == SECTION + user_note
