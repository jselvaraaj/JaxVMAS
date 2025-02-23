#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import importlib
import os
import os.path as osp
import sys
from pathlib import Path


def load(name: str):
    pathname = None
    for dirpath, _, filenames in os.walk(osp.dirname(__file__)):
        if pathname is None:
            for filename in filenames:
                if name == filename or Path(name) == Path(dirpath) / Path(filename):
                    pathname = os.path.join(dirpath, filename)
                    break
    assert pathname is not None, f"{name} scenario not found."

    module_name = f"jaxvmas.scenario.{Path(pathname).stem}"
    spec = importlib.util.spec_from_file_location(module_name, pathname)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register the module
    spec.loader.exec_module(module)
    return module
