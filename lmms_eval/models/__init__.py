"""Model registry for the compact FocusGUI release.

Only the four omni-modal backbones evaluated by FocusGUI are exposed here.  Model
modules are imported lazily so optional, backbone-specific dependencies remain
isolated until that backbone is selected.
"""

from __future__ import annotations

import importlib
import os


AVAILABLE_SIMPLE_MODELS = {
    "qwen2_5_omni": "lmms_eval.models.simple.qwen2_5_omni.Qwen2_5_Omni",
}

AVAILABLE_CHAT_TEMPLATE_MODELS = {
    "baichuan_omni": "lmms_eval.models.simple.baichuan_omni.BaichuanOmni",
    "minicpm_o": "lmms_eval.models.simple.minicpm_o.MiniCPMo",
    "vita": "lmms_eval.models.simple.vita.VITA",
}


def get_model(model_name, force_simple: bool = False):
    registries = (AVAILABLE_SIMPLE_MODELS, AVAILABLE_CHAT_TEMPLATE_MODELS)
    for registry in registries:
        if model_name in registry:
            module_name, class_name = registry[model_name].rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
    available = sorted(set().union(*registries))
    raise ValueError(f"Model {model_name!r} is unavailable; choose from {available}.")


if os.environ.get("LMMS_EVAL_PLUGINS"):
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        module = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(module, "AVAILABLE_MODELS").items():
            AVAILABLE_SIMPLE_MODELS[model_name] = (
                f"{plugin}.models.{model_name}.{model_class}"
            )
