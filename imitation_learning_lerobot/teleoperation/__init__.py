from .handler import Handler
from .handler_factory import HandlerFactory

from .joycon import PickBoxJoyconHandler
from .keyboard import PickBoxKeyboardHandler

HandlerFactory.register_all()
