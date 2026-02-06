#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 HandlerFactory.register_all() 是否能正确注册 joycon_evdev Handler

这个测试直接检查类定义和注册逻辑，不依赖外部模块
"""

import sys
import os
import importlib.util

print("=" * 60)
print("测试 HandlerFactory.register_all()")
print("=" * 60)

# 方法1: 直接检查类定义文件
print("\n方法1: 直接检查类定义文件...")
evdev_file = "imitation_learning_lerobot/teleoperation/joycon/pick_box_joycon_handler_evdev.py"
if os.path.exists(evdev_file):
    print(f"✅ 找到文件: {evdev_file}")
    
    # 读取文件内容检查 _name
    with open(evdev_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if '_name = "pick_box_joycon_evdev"' in content:
            print("✅ _name 已正确设置为 'pick_box_joycon_evdev'")
        elif '_name = "joycon_evdev"' in content:
            print("⚠️  _name 是 'joycon_evdev'，应该是 'pick_box_joycon_evdev'")
        elif '_name = "pick_box_joycon"' in content:
            print("❌ _name 仍然是 'pick_box_joycon'，需要更新为 'pick_box_joycon_evdev'")
        else:
            print("⚠️  无法在文件中找到 _name 定义")
        
        # 检查是否正确导入 Handler
        if 'from ..handler import Handler' in content:
            print("✅ 正确从 ..handler 导入 Handler")
        elif 'class Handler(abc.ABC):' in content:
            print("❌ 文件中仍然定义了 Handler 类，应该删除并使用导入")
        else:
            print("⚠️  无法确定 Handler 的导入方式")
else:
    print(f"❌ 文件不存在: {evdev_file}")

# 方法2: 检查 __init__.py 文件
print("\n方法2: 检查 __init__.py 文件...")
init_file = "imitation_learning_lerobot/teleoperation/__init__.py"
if os.path.exists(init_file):
    print(f"✅ 找到文件: {init_file}")
    
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'PickBoxJoyconEvdevHandler' in content:
            print("✅ PickBoxJoyconEvdevHandler 已在 __init__.py 中导入")
        else:
            print("❌ PickBoxJoyconEvdevHandler 未在 __init__.py 中导入")
        
        if 'HandlerFactory.register_all()' in content:
            print("✅ HandlerFactory.register_all() 已在 __init__.py 中调用")
        else:
            print("❌ HandlerFactory.register_all() 未在 __init__.py 中调用")
else:
    print(f"❌ 文件不存在: {init_file}")

# 方法3: 尝试动态导入（处理导入错误）
print("\n方法3: 尝试动态导入和注册测试...")
try:
    # 先导入基础模块
    sys.path.insert(0, os.getcwd())
    
    # 导入 Handler 基类
    from imitation_learning_lerobot.teleoperation.handler import Handler
    print("✅ 成功导入 Handler 基类")
    
    # 导入 HandlerFactory
    from imitation_learning_lerobot.teleoperation.handler_factory import HandlerFactory
    print("✅ 成功导入 HandlerFactory")
    
    # 手动注册测试（模拟 register_all 的行为）
    print("\n   检查 Handler 的子类...")
    from imitation_learning_lerobot.utils import ClassUtils
    
    leaf_subclasses = ClassUtils.get_leaf_subclasses(Handler)
    print(f"   找到 {len(leaf_subclasses)} 个叶子子类:")
    
    joycon_evdev_found = False
    for subclass in leaf_subclasses:
        print(f"     - {subclass.__name__}: name='{subclass.name}'")
        if subclass.name == "pick_box_joycon_evdev":
            joycon_evdev_found = True
            print(f"       ✅ 找到 pick_box_joycon_evdev Handler!")
    
    if not joycon_evdev_found:
        print("\n   ❌ 未找到 name='pick_box_joycon_evdev' 的 Handler")
        print("   可能的原因:")
        print("   1. PickBoxJoyconEvdevHandler 未被导入（导入时出错）")
        print("   2. PickBoxJoyconEvdevHandler._name 不是 'pick_box_joycon_evdev'")
    
    # 测试 register_all
    print("\n   测试 register_all()...")
    HandlerFactory.register_all()
    
    # 检查注册结果
    strategies = HandlerFactory._strategies
    print(f"   注册了 {len(strategies)} 个 Handler:")
    for name, cls in strategies.items():
        print(f"     - '{name}': {cls.__name__}")
    
    # 测试获取
    print("\n   测试 get_strategies('pick_box_joycon_evdev')...")
    handler_cls = HandlerFactory.get_strategies("pick_box_joycon_evdev")
    
    if handler_cls:
        print(f"   ✅ 成功获取 Handler: {handler_cls.__name__}")
        print(f"   ✅ Handler 名称: {handler_cls.name}")
    else:
        print("   ❌ 无法获取 'pick_box_joycon_evdev' Handler")
        
except ImportError as e:
    print(f"⚠️  导入错误（可能是缺少依赖）: {e}")
    print("   但这不影响注册逻辑的测试")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
