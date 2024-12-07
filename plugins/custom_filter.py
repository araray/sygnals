from sygnals.core.plugin_manager import register_plugin


@register_plugin
def amplify(data, factor=2):
    """Amplifies the signal."""
    return data * factor
