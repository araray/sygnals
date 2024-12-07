from sygnals.core.plugin_manager import register_plugin


@register_plugin
def amplify_signal(data, factor=2):
    """Amplify the signal values by a given factor."""
    return data * factor

