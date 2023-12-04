"""Test cases for 'An Investigation into Fuzzy Systems' Q2."""

# pylint: disable=missing-function-docstring, missing-class-docstring

from pytest import approx, raises  # type: ignore

from main import FuzzySet, apply_elementwise, fuzzy_cond_prob_dist

# A set of integer elements from 1 to 11.
test_elements = set(range(1, 12))

# A membership function that maps the above elements to unique values.
test_membership = {
    element: min(element / 10, 1.0) for element in range(1, 12)
}

# A membership function that maps the above elements to non-unique values.
test_membership_duplicates = {
    1: 0.1,
    2: 0.1,
    3: 0.3,
    4: 0.3,
    5: 0.5,
    6: 0.5,
    7: 0.7,
    8: 0.7,
    9: 0.9,
    10: 0.9,
    11: 1.0,
}


class TestFuzzySet:
    def test_alpha_cut(self):
        fuzzy_set = FuzzySet(test_elements, test_membership)
        assert fuzzy_set.alpha_cut(0.1) == set(range(1, 12))
        assert fuzzy_set.alpha_cut(0.2) == set(range(2, 12))
        assert fuzzy_set.alpha_cut(0.3) == set(range(3, 12))
        assert fuzzy_set.alpha_cut(0.4) == set(range(4, 12))
        assert fuzzy_set.alpha_cut(0.5) == set(range(5, 12))
        assert fuzzy_set.alpha_cut(0.6) == set(range(6, 12))
        assert fuzzy_set.alpha_cut(0.7) == set(range(7, 12))
        assert fuzzy_set.alpha_cut(0.8) == set(range(8, 12))
        assert fuzzy_set.alpha_cut(0.9) == set(range(9, 12))
        assert fuzzy_set.alpha_cut(1.0) == set(range(10, 12))

    def test_alpha_cut_duplicates(self):
        fuzzy_set = FuzzySet(test_elements, test_membership_duplicates)
        assert fuzzy_set.alpha_cut(0.1) == set(range(1, 12))
        assert fuzzy_set.alpha_cut(0.3) == set(range(3, 12))
        assert fuzzy_set.alpha_cut(0.5) == set(range(5, 12))
        assert fuzzy_set.alpha_cut(0.7) == set(range(7, 12))
        assert fuzzy_set.alpha_cut(0.9) == set(range(9, 12))

    def test_alpha_cut_lecture_notes(self):
        """Example 5.3.2 from the lecture notes."""
        fuzzy_set = FuzzySet({4, 5, 6}, {4: 0.3, 5: 0.7, 6: 1.0})
        assert fuzzy_set.alpha_cut(0.3) == {4, 5, 6}
        assert fuzzy_set.alpha_cut(0.7) == {5, 6}
        assert fuzzy_set.alpha_cut(1.0) == {6}

    def test_alpha_cuts(self):
        assert FuzzySet(test_elements, test_membership).alpha_cuts() == {
            frozenset(range(1, 12)): (0.0, 0.1),
            frozenset(range(2, 12)): (0.1, 0.2),
            frozenset(range(3, 12)): (0.2, 0.3),
            frozenset(range(4, 12)): (0.3, 0.4),
            frozenset(range(5, 12)): (0.4, 0.5),
            frozenset(range(6, 12)): (0.5, 0.6),
            frozenset(range(7, 12)): (0.6, 0.7),
            frozenset(range(8, 12)): (0.7, 0.8),
            frozenset(range(9, 12)): (0.8, 0.9),
            frozenset(range(10, 12)): (0.9, 1.0),
        }

    def test_alpha_cuts_duplicates(self):
        assert FuzzySet(
            test_elements, test_membership_duplicates
        ).alpha_cuts() == {
            frozenset(range(1, 12)): (0.0, 0.1),
            frozenset(range(3, 12)): (0.1, 0.3),
            frozenset(range(5, 12)): (0.3, 0.5),
            frozenset(range(7, 12)): (0.5, 0.7),
            frozenset(range(9, 12)): (0.7, 0.9),
            frozenset(range(11, 12)): (0.9, 1.0),
        }

    def test_alpha_cuts_lecture_notes(self):
        """Example 5.3.2 from the lecture notes."""
        assert FuzzySet(
            {4, 5, 6}, {4: 0.3, 5: 0.7, 6: 1.0}
        ).alpha_cuts() == {
            frozenset({4, 5, 6}): (0.0, 0.3),
            frozenset({5, 6}): (0.3, 0.7),
            frozenset({6}): (0.7, 1.0),
        }

    def test_from_alpha_cuts(self):
        fuzzy_set = FuzzySet.from_alpha_cuts(
            {
                frozenset(range(1, 12)): (0.0, 0.1),
                frozenset(range(2, 12)): (0.1, 0.2),
                frozenset(range(3, 12)): (0.2, 0.3),
                frozenset(range(4, 12)): (0.3, 0.4),
                frozenset(range(5, 12)): (0.4, 0.5),
                frozenset(range(6, 12)): (0.5, 0.6),
                frozenset(range(7, 12)): (0.6, 0.7),
                frozenset(range(8, 12)): (0.7, 0.8),
                frozenset(range(9, 12)): (0.8, 0.9),
                frozenset(range(10, 12)): (0.9, 1.0),
            }
        )
        assert fuzzy_set.elements == test_elements
        assert fuzzy_set.membership_function == test_membership

    def test_from_alpha_cuts_duplicates(self):
        fuzzy_set = FuzzySet.from_alpha_cuts(
            {
                frozenset(range(1, 12)): (0.0, 0.1),
                frozenset(range(3, 12)): (0.1, 0.3),
                frozenset(range(5, 12)): (0.3, 0.5),
                frozenset(range(7, 12)): (0.5, 0.7),
                frozenset(range(9, 12)): (0.7, 0.9),
                frozenset(range(11, 12)): (0.9, 1.0),
            }
        )
        assert fuzzy_set.elements == test_elements
        assert fuzzy_set.membership_function == test_membership_duplicates

    def test_from_alpha_cuts_lecture_notes(self):
        """Example 5.3.3 from the lecture notes."""
        fuzzy_set = FuzzySet.from_alpha_cuts(
            {
                frozenset({4, 5, 6}): (0.0, 0.3),
                frozenset({5, 6}): (0.3, 0.7),
                frozenset({6}): (0.7, 1.0),
            }
        )
        assert fuzzy_set.elements == {4, 5, 6}
        assert fuzzy_set.membership_function == {4: 0.3, 5: 0.7, 6: 1.0}

    def test_apply_elementwise_one_to_one(self):
        fuzzy_set = FuzzySet(
            test_elements, test_membership
        ).apply_elementwise(lambda element: element**2)
        assert fuzzy_set.elements == {
            1,
            4,
            9,
            16,
            25,
            36,
            49,
            64,
            81,
            100,
            121,
        }
        assert fuzzy_set.membership_function == {
            1: 0.1,
            4: 0.2,
            9: 0.3,
            16: 0.4,
            25: 0.5,
            36: 0.6,
            49: 0.7,
            64: 0.8,
            81: 0.9,
            100: 1.0,
            121: 1.0,
        }

    def test_apply_elementwise_many_to_one(self):
        fuzzy_set = FuzzySet(
            test_elements, test_membership
        ).apply_elementwise(lambda element: element // 2)
        assert fuzzy_set.elements == {0, 1, 2, 3, 4, 5}
        assert fuzzy_set.membership_function == {
            0: 0.1,
            1: 0.3,
            2: 0.5,
            3: 0.7,
            4: 0.9,
            5: 1.0,
        }

    def test_apply_elementwise_lecture_notes(self):
        """Example 5.4.2 from the lecture notes."""
        fuzzy_set = FuzzySet(
            {4, 5, 6}, {4: 0.3, 5: 0.7, 6: 1.0}
        ).apply_elementwise(
            lambda element: 6 if element == 1 else element - 1
        )
        assert fuzzy_set.elements == {3, 4, 5}
        assert fuzzy_set.membership_function == {
            3: 0.3,
            4: 0.7,
            5: 1.0,
        }

    def test_apply_numeric_lecture_notes(self):
        """Example 5.4.1 from the lecture notes."""
        assert (
            FuzzySet({4, 5, 6}, {4: 0.3, 5: 0.7, 6: 1.0}).apply_numeric(
                len
            )
            == 2.0
        )


def test_apply_elementwise():
    assert apply_elementwise(
        set(range(1, 11)), lambda element: element**2
    ) == {1, 4, 9, 16, 25, 36, 49, 64, 81, 100}
    assert apply_elementwise(
        set(range(1, 11)), lambda element: element // 2
    ) == {0, 1, 2, 3, 4, 5}


class TestFuzzyConditional:
    def test_valid(self):
        p = fuzzy_cond_prob_dist(
            {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
            FuzzySet(
                {1, 2, 3, 4},
                {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7},
            ),
        )
        assert sum(p.values()) == 1.0
        assert approx(p) == {
            1: 0.22,
            2: 0.24,
            3: 0.26,
            4: 0.28,
        }

    def test_invalid_empty(self):
        with raises(ValueError) as excinfo:
            fuzzy_cond_prob_dist(
                dict[int, float](),
                FuzzySet(
                    {1, 2, 3, 4},
                    {1: 0.9, 2: 0.9, 3: 0.8, 4: 0.7},
                ),
            )
        assert (
            str(excinfo.value)
            == "prob_dist and fuzzy_prop must be non-empty."
        )

        with raises(ValueError) as excinfo:
            fuzzy_cond_prob_dist(
                {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
                FuzzySet(set[int](), dict[int, float]()),
            )
        assert (
            str(excinfo.value)
            == "prob_dist and fuzzy_prop must be non-empty."
        )

    def test_invalid_elements(self):
        with raises(ValueError) as excinfo:
            fuzzy_cond_prob_dist(
                {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
                FuzzySet(
                    {1, 2, 3},
                    {1: 0.9, 2: 0.9, 3: 0.8},
                ),
            )
        assert str(excinfo.value) == (
            "prob_dist and fuzzy_prop must be defined for the same "
            "possible worlds."
        )

    def test_invalid_membership(self):
        with raises(ValueError) as excinfo:
            fuzzy_cond_prob_dist(
                {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
                FuzzySet(
                    {1, 2, 3, 4},
                    {1: 0.9, 2: 0.9, 3: 0.8, 4: 0.7},
                ),
            )
        assert str(excinfo.value) == (
            "fuzzy_prop must contain a possible world with membership"
            "value 1."
        )

    def test_invalid_prob_dist(self):
        with raises(ValueError) as excinfo:
            fuzzy_cond_prob_dist(
                {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.3},
                FuzzySet(
                    {1, 2, 3, 4},
                    {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7},
                ),
            )
        assert (
            str(excinfo.value)
            == "prob_dist must have total probability 1."
        )

    def test_invalid_zero_probability(self):
        with raises(ValueError) as excinfo:
            fuzzy_cond_prob_dist(
                {1: 0.0, 2: 0.2, 3: 0.4, 4: 0.4},
                FuzzySet(
                    {1, 2, 3, 4},
                    {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7},
                ),
            )
        assert (
            str(excinfo.value)
            == "{1} must have non-zero total probability."
        )
