from safe.scenarios import synthetic_scenarios


def test_generators_are_reproducible_and_cover_required_scenarios():
    first = synthetic_scenarios()
    assert first == synthetic_scenarios()
    assert first != synthetic_scenarios(1730)
    assert len(first) == 18
    assert len({s.name for s in first}) == 18
    assert all(s.end - s.start == 7 * 86400 for s in first)

