# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of Timer for forecasting."""

from sktime.forecasting.hf_transformers_forecaster import HFTransformersForecaster


class TimerForecaster(HFTransformersForecaster):
    """
    Timer Forecaster for Zero-Shot Forecasting of Univariate Time Series.

    Wrapping implementation in [1]_ of method proposed in [2]_. See [3]_
    for hugging face tutorial.

    Timer: Generative Pre-trained Transformers Are Large Time Series Models
    It introduces a groundbreaking approach to leveraging transformers for
    accurate and scalable time series forecasting.
    """

    _tags = {
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.Series",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "authors": ["WenWeiTHU", "ZDandsomSP", "Sohaib-Ahmed21"],
        # WenWeiTHU, ZDandsomSP for thuml code
        "maintainers": ["Sohaib-Ahmed21"],
        "python_dependencies": ["transformers", "torch"],
        "capability:global_forecasting": True,
    }

    def __init__(
        self,
        model_path: str,
        fit_strategy="minimal",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        deterministic=False,
        callbacks=None,
        peft_config=None,
        trust_remote_code=False,
    ):
        super().__init__(
            model_path=model_path,
            fit_strategy=fit_strategy,
            validation_split=validation_split,
            config=config,
            training_args=training_args,
            compute_metrics=compute_metrics,
            deterministic=deterministic,
            callbacks=callbacks,
            peft_config=peft_config,
            trust_remote_code=trust_remote_code,
        )

    def update_config(self, config, X, y, fh):
        """Update config with user provided config."""
        _config = config.to_dict()
        _config.update(self._config)

        if fh is not None:
            _config["output_token_lens"][0] = max(
                *(fh.to_relative(self._cutoff)._values + 1),
                _config["output_token_lens"][0],
            )

        config = config.from_dict(_config)
        return config

    def load_model(self, config, model_path, **kwargs):
        """Load model from config."""
        from transformers import (
            AutoModelForCausalLM,
        )

        model, info = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
            trust_remote_code=self.trust_remote_code,
        )
        return model, info

    def get_seq_args(self, config):
        """Get context and pred length."""
        context_length = config.input_token_len
        prediction_length = config.output_token_lens[0]
        return context_length, prediction_length

    def pred_output(
        self,
        past_values,
        past_time_features,
        future_time_features,
        past_observed_mask,
        fh,
    ):
        """Predict output based on unique method of each model."""
        pred = self.model.generate(
            inputs=past_values,
            max_new_tokens=max(fh._values),
        )
        pred = pred.reshape((-1,))
        return pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        test_params = [
            {
                "model_path": "thuml/timer-base-84m",
                "trust_remote_code": True,
                "training_args": {
                    "max_steps": 4,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 4,
                    "report_to": "none",
                },
            },
            {
                "model_path": "thuml/timer-base-84m",
                "trust_remote_code": True,
                "config": {
                    "input_token_len": 16,
                    "output_token_lens": 8,
                },
                "validation_split": 0.2,
                "training_args": {
                    "max_steps": 5,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 4,
                    "report_to": "none",
                },
            },
            {
                "model_path": "thuml/timer-base-84m",
                "trust_remote_code": True,
                "config": {
                    "input_token_len": 20,
                    "output_token_lens": 12,
                },
                "validation_split": 0.2,
                "training_args": {
                    "max_steps": 5,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 4,
                    "report_to": "none",
                },
                "fit_strategy": "full",
            },
        ]
        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in test_params]
        test_params.extend(params_broadcasting)
        return test_params
