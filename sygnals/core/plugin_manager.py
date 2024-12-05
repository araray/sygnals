import importlib.util
import os

PLUGIN_DIR = './plugins/'

def discover_plugins(plugin_dir=PLUGIN_DIR):
    """Discover Python plugins in the plugins directory."""
    if not os.path.exists(plugin_dir):
        os.makedirs(plugin_dir)  # Create the plugins directory if it doesn't exist

    plugins = {}
    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py'):
            module_name = filename[:-3]
            module_path = os.path.join(plugin_dir, filename)
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and hasattr(attr, "__plugin__"):  # Only consider functions marked as plugins
                    plugins[attr_name] = attr
    return plugins

def register_plugin(func):
    """Mark a function as a plugin."""
    func.__plugin__ = True
    return func
