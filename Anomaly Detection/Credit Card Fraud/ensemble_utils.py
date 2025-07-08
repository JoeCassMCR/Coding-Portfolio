"""
ensemble_utils.py

Build stacking or voting ensembles of multiple classifiers.
"""

import logging
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)


def build_stacking_ensemble(estimators: list, final_estimator=None) -> StackingClassifier:
    """
    Build a stacking ensemble given base estimators and a final estimator.

    Parameters
    ----------
    estimators : list of (str, estimator)
    final_estimator : estimator (defaults to LogisticRegression)

    Returns
    -------
    StackingClassifier
    """
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000, random_state=0)
    ensemble = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)
    return ensemble


def build_voting_ensemble(estimators: list, voting: str = 'soft') -> VotingClassifier:
    """
    Build a voting ensemble of classifiers.

    Parameters
    ----------
    estimators : list of (str, estimator)
    voting : 'hard' or 'soft'

    Returns
    -------
    VotingClassifier
    """
    ensemble = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
    return ensemble
