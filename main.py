"""Implementation for 'An Investigation into Fuzzy Systems' Q2."""

from collections.abc import Hashable, Set
from itertools import pairwise
from typing import Callable, Iterator

# The alpha-cuts of a fuzzy set, represented as a map from sets of
# elements to intervals of alpha-values.
type AlphaCuts[Element: Hashable] = dict[
    Set[Element], tuple[float, float]
]


class FuzzySet[Element: Hashable](Set[Element], Hashable):
    """
    A discrete fuzzy set with a finite number of elements.

    Parameters
    ----------
    elements
        A set of elements.
    membership_function
        A map from elements to their membership values.
    """

    def __init__(
        self,
        elements: Set[Element],
        membership_function: dict[Element, float],
    ):
        self.elements = elements
        self.membership_function = membership_function

    def __contains__(self, element: Element) -> bool:
        return element in self.elements

    def __hash__(self) -> int:
        return hash(self.elements)

    def __iter__(self) -> Iterator[Element]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def alpha_cut(self, alpha: float) -> Set[Element]:
        """
        Q2(a). Given an alpha value, returns the alpha-cut of the fuzzy
        set.

        Parameters
        ----------
        alpha
            An alpha value.

        Return
        ----------
        alpha_cut
            The alpha-cut of the fuzzy set.
        """
        return frozenset(
            {
                element
                for element in self.elements
                if self.membership_function[element] >= alpha
            }
        )

    def alpha_cuts(self) -> AlphaCuts[Element]:
        """
        Q2(a). Returns the alpha-cuts of the fuzzy set.

        Return
        ----------
        alpha_cuts
            The alpha-cuts of the fuzzy set.
        """
        values = [0.0] + sorted(set(self.membership_function.values()))
        return {
            self.alpha_cut(upper): (lower, upper)
            for [lower, upper] in pairwise(values)
        }

    @staticmethod
    def from_alpha_cuts(alpha_cuts: AlphaCuts[Element]):
        """
        Q2(b). Given a set of alpha-cuts, returns the fuzzy set.

        Parameters
        ----------
        alpha_cuts
            A set of alpha-cuts.

        Return
        ----------
        from_alpha_cuts
            The fuzzy set.
        """
        membership_function = dict[Element, float]()
        for elements, (_lower, upper) in alpha_cuts.items():
            for element in elements:
                membership_function[element] = upper
        return FuzzySet(set(membership_function), membership_function)

    def apply_elementwise[
        Result: Hashable
    ](self, function: Callable[[Element], Result]):
        """
        Q2(c). Given an element-wise function f : W -> W, returns the
        fuzzy set of results of applying the function.

        Parameters
        ----------
        function
            An element-wise function f : W -> W.

        Return
        ----------
        apply_elementwise
            The fuzzy set of results of applying the function.
        """
        return FuzzySet.from_alpha_cuts(
            merge_alpha_cuts(
                {
                    apply_elementwise(elements, function): (
                        lower,
                        upper,
                    )
                    for elements, (
                        lower,
                        upper,
                    ) in self.alpha_cuts().items()
                }
            )
        )

    def apply_numeric(
        self, function: Callable[[Set[Element]], float]
    ) -> float:
        """
        Given a function f : 2^W -> R, returns the result of applying the
        function to the fuzzy set.

        Parameters
        ----------
        function
            A function f : 2^W -> R.

        Return
        ----------
        apply_numeric
            The result of applying the function to the fuzzy set.
        """
        return sum(
            (upper - lower) * function(elements)
            for elements, (
                lower,
                upper,
            ) in self.alpha_cuts().items()
        )


def merge_alpha_cuts[
    Element: Hashable
](alpha_cuts: AlphaCuts[Element]) -> AlphaCuts[Element]:
    """
    Given alpha-cuts with duplicate elements, returns the merged
    alpha-cuts.

    Parameters
    ----------
    alpha_cuts
        The alpha-cuts.

    Return
    ----------
    merge_alpha_cuts
        The merged alpha-cuts.
    """
    merged: AlphaCuts[Element] = {}
    for elements, (lower, upper) in alpha_cuts.items():
        if elements in merged:
            merged[elements] = (
                min(lower, merged[elements][0]),
                max(upper, merged[elements][1]),
            )
        else:
            merged[elements] = (lower, upper)
    return merged


def apply_elementwise[
    Argument: Hashable, Result: Hashable
](
    elements: Set[Argument],
    function: Callable[[Argument], Result],
) -> Set[Result]:
    """
    Q2(c). Given a set of elements and an element-wise function f : W -> W,
    returns the set of results of applying the function to the elements.

    Parameters
    ----------
    elements
        A set of elements.
    function
        An element-wise function f : W -> W.

    Return
    ----------
    apply_elementwise
        The set of results of applying the function to the elements.
    """
    return frozenset({function(element) for element in elements})


def fuzzy_cond_prob_dist[
    PossibleWorld: Hashable
](
    prob_dist: dict[PossibleWorld, float],
    fuzzy_prop: FuzzySet[PossibleWorld],
) -> dict[PossibleWorld, float]:
    """
    Q2(d). Given a probability distribution over a finite set of possible
    worlds and a fuzzy proposition over the same set of possible worlds,
    returns the conditional probability distribution given the fuzzy
    proposition.

    Parameters
    ----------
    prob_dist
        A probability distribution over a finite set of possible worlds.
    fuzzy_prop
        A fuzzy proposition over the same set of possible worlds.

    Return
    ----------
    fuzzy_cond_prob_dist
        The conditional probability distribution given the fuzzy
        proposition.
    """

    if not prob_dist or not fuzzy_prop:
        raise ValueError("prob_dist and fuzzy_prop must be non-empty.")

    if set(prob_dist.keys()) != fuzzy_prop.elements:
        raise ValueError(
            (
                "prob_dist and fuzzy_prop must be defined for the same "
                "possible worlds."
            )
        )

    if 1.0 not in fuzzy_prop.membership_function.values():
        raise ValueError(
            (
                "fuzzy_prop must contain a possible world with membership"
                "value 1."
            )
        )

    if sum(prob_dist.values()) != 1.0:
        raise ValueError("prob_dist must have total probability 1.")

    def sum_prob(ws: Set[PossibleWorld]) -> float:
        """
        Given a set of possible worlds, returns the sum of their
        probabilities.
        """
        return sum(prob_dist[w] for w in ws)

    def make_cond_prob(
        w: PossibleWorld,
    ) -> Callable[[Set[PossibleWorld]], float]:
        """
        Given a possible world, returns a function that returns its
        conditional probability given a crisp proposition.
        """

        def cond_prob(ws: Set[PossibleWorld]) -> float:
            if sum_prob(ws) == 0.0:
                raise ValueError(
                    f"{set(ws)} must have non-zero total probability."
                )
            return (prob_dist[w] if w in ws else 0.0) / sum_prob(ws)

        return cond_prob

    return {
        w: fuzzy_prop.apply_numeric(make_cond_prob(w)) for w in prob_dist
    }
