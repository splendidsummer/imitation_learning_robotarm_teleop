#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试：验证 pick_box_joycon_evdev Handler 是否正确注册
"""

from imitation_learning_lerobot.teleoperation import HandlerFactory

# HandlerFactory.register_all() 已在 __init__.py 中调用

print("测试 HandlerFactory.get_strategies('pick_box_joycon_evdev')...")
handler_cls = HandlerFactory.get_strategies("pick_box_joycon_evdev")

if handler_cls is None:
    print("❌ 失败: 找不到 'pick_box_joycon_evdev' Handler!")
    print("\n已注册的 Handler:")
    for name in HandlerFactory._strategies.keys():
        print(f"  - {name}")
    exit(1)
else:
    print(f"✅ 成功: 找到 Handler")
    print(f"   类名: {handler_cls.__name__}")
    print(f"   名称: {handler_cls.name}")
    print("\n✅ 测试通过！Handler 已正确注册。")

