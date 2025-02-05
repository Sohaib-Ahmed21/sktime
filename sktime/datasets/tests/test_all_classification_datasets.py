"""Unit tests for all time series regressors."""

__author__ = ["felipeangelimvieira"]
import numpy as np
import pandas as pd

from sktime.tests.test_all_estimators import BaseFixtureGenerator, QuickTester


class ClassificationDatasetFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for classifier tests.

    Fixtures parameterized
    ----------------------
    estimator_class: estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
    estimator_instance: instance of estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
        instances are generated by create_test_instance class method
    scenario: instance of TestScenario
        ranges over all scenarios returned by retrieve_scenarios
    """

    # note: this should be separate from TestAllRegressors
    #   additional fixtures, parameters, etc should be added here
    #   TestAllRegressors should contain the tests only

    estimator_type_filter = "dataset_classification"


class TestAllClassificationDatasets(ClassificationDatasetFixtureGenerator, QuickTester):
    """Module level tests for all sktime regressors."""

    def test_tag_n_classes(self, estimator_instance):
        n_classes = estimator_instance.get_tag("n_classes")
        y = estimator_instance.load("y")

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        unique_y_values = np.unique(y)
        assert len(unique_y_values) == n_classes

    def test_tag_is_univariate(self, estimator_instance):
        is_univariate = estimator_instance.get_tag("is_univariate")
        X = estimator_instance.load("X")
        assert X.shape[1] == 1 if is_univariate else X.shape[1] > 1
