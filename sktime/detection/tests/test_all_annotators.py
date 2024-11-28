"""Tests for sktime annotators."""

__author__ = ["miraep8", "fkiraly", "klam-data", "pyyim", "mgorlin"]
__all__ = []

import numpy as np
import pandas as pd

from sktime.tests.test_all_estimators import BaseFixtureGenerator, QuickTester
from sktime.utils._testing.annotation import make_detection_problem
from sktime.utils.validation.annotation import check_learning_type, check_task


class AnnotatorsFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for time series annotator (outlier, change point, etc) tests.

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

    # note: this should be separate from TestAllAnnotators
    #   additional fixtures, parameters, etc should be added here
    #   TestAllAnnotators should contain the tests only

    estimator_type_filter = "detector"

    fixture_sequence = [
        "estimator_class",
        "estimator_instance",
        "fitted_estimator",
        "scenario",
        "method_nsc",
        "method_nsc_arraylike",
    ]


class TestAllAnnotators(AnnotatorsFixtureGenerator, QuickTester):
    """Module level tests for all sktime annotators."""

    def test_output_type(self, estimator_instance):
        """Test annotator output type."""
        estimator = estimator_instance

        X_train = make_detection_problem(
            n_timepoints=50, estimator_type=estimator.get_tag("distribution_type")
        )
        estimator.fit(X_train)
        X_test = make_detection_problem(
            n_timepoints=10, estimator_type=estimator.get_tag("distribution_type")
        )
        y_test = estimator.predict(X_test)
        assert isinstance(y_test, pd.Series)

    def test_transform_output_type(self, estimator_instance):
        """Test output type for the transform method."""
        X_train = make_detection_problem(
            n_timepoints=50,
            estimator_type=estimator_instance.get_tag("distribution_type"),
        )
        estimator_instance.fit(X_train)
        X_test = make_detection_problem(
            n_timepoints=10,
            estimator_type=estimator_instance.get_tag("distribution_type"),
        )
        y_test = estimator_instance.transform(X_test)
        assert isinstance(y_test, pd.DataFrame)
        assert len(y_test) == len(X_test)

    def test_predict_points(self, estimator_instance):
        X_train = make_detection_problem(
            n_timepoints=50,
            estimator_type=estimator_instance.get_tag("distribution_type"),
        )
        estimator_instance.fit(X_train)
        X_test = make_detection_problem(
            n_timepoints=10,
            estimator_type=estimator_instance.get_tag("distribution_type"),
        )
        y_pred = estimator_instance.predict_points(X_test)
        assert isinstance(y_pred, (pd.Series, np.ndarray))

    def test_predict_segments(self, estimator_instance):
        X_train = make_detection_problem(
            n_timepoints=50,
            estimator_type=estimator_instance.get_tag("distribution_type"),
        )
        estimator_instance.fit(X_train)

        X_test = make_detection_problem(
            n_timepoints=10,
            estimator_type=estimator_instance.get_tag("distribution_type"),
        )
        y_test = estimator_instance.predict_segments(X_test)
        assert isinstance(y_test, pd.Series)
        assert isinstance(y_test.index.dtype, pd.IntervalDtype)
        assert pd.api.types.is_integer_dtype(y_test)

    def test_annotator_tags(self, estimator_class):
        """Check the learning_type and task tags are valid."""
        check_task(estimator_class.get_class_tag("task"))
        check_learning_type(estimator_class.get_class_tag("learning_type"))
