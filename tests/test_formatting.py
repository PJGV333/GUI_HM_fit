import textwrap


def test_expandtabs_uses_four_space_columns():
    line = "Name\tValue\tUnit"
    expanded = line.expandtabs(4)

    expected = "Name    Value   Unit"
    assert expanded == expected
    # Ensure the alignment works with additional content.
    wrapped = textwrap.dedent(
        """
        A\t1\talpha
        AB\t23\tbeta
        """
    ).strip().splitlines()

    expanded_wrapped = [entry.expandtabs(4) for entry in wrapped]
    assert expanded_wrapped[0].index("1") == expanded_wrapped[1].index("23")
