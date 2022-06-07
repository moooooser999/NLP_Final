import logging
import warnings
from collections import OrderedDict

from transformers import (
    AutoConfig,
    BertConfig,
)
from transformers import PretrainedConfig

from bert_for_task2 import BertForTask2Regression, BertForTask2MultiClass

logger = logging.getLogger(__name__)

MODEL_FOR_TASK2_REGRESSION_MAPPING = OrderedDict(
    [
        (BertConfig, BertForTask2Regression)
    ]
)

MODEL_FOR_TASK2_MULTI_CLASS_MAPPING = OrderedDict(
    [
        (BertConfig, BertForTask2MultiClass)
    ]
)


class AutoModelForTask2Regression:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTask2Regression is designed to be instantiated "
            "using the `AutoModelForTask2Regression.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTask2Regression.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_TASK2_REGRESSION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModelForTask2Regression: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_TASK2_REGRESSION_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_TASK2_REGRESSION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModelForTask2Regression: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_TASK2_REGRESSION_MAPPING.keys())
            )
        )

class AutoModelForTask2MultiClass:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTask2MultiClass is designed to be instantiated "
            "using the `AutoModelForTask2MultiClass.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTask2MultiClass.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_TASK2_MULTI_CLASS_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModelForTask2MultiClass: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_TASK2_MULTI_CLASS_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_TASK2_MULTI_CLASS_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModelForTask2MultiClass: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_TASK2_MULTI_CLASS_MAPPING.keys())
            )
        )
