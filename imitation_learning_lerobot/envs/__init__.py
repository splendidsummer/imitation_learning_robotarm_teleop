from .env import Env
from .env_factory import EnvFactory

from .pick_and_place_env import PickAndPlaceEnv
from .dishwasher_env import DishwasherEnv
from .bartend_env import BartendEnv
from .pick_box_env import PickBoxEnv

EnvFactory.register_all()
