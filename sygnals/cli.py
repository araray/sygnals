import click
import json
import numpy as np
import pandas as pd
from sygnals.core import (
    data_handler,
    dsp,
    transforms,
    filters,
    audio_handler,
    plugin_manager,
    batch_processor,
    custom_exec,
    storage
)
from sygnals.utils import visualizations
from tabulate import tabulate

@click.group()
def cli():
    """Sygnals: A versatile CLI for signal and audio processing."""
    pass

@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", type=click.Choice(["json", "csv", "tabulated"]), default="json")
def analyze(file, output):
    """Analyze a data or audio file."""
    if file.endswith(('.wav', '.mp3')):
        data, sr = audio_handler.load_audio(file)  # Load both data and sample rate
        metrics = audio_handler.get_audio_metrics(data, sr)
    else:
        data = data_handler.read_data(file)
        metrics = {
            "rows": len(data),
            "mean": data.mean().to_dict(),
            "max": data.max().to_dict(),
            "min": data.min().to_dict()
        }

    if output == "json":
        # Ensure compatibility with JSON serialization
        click.echo(json.dumps(metrics, indent=2, default=lambda x: float(x) if isinstance(x, (int, float)) else x))
    elif output == "csv":
        pd.DataFrame([metrics]).to_csv("analysis.csv", index=False)
        click.echo("Analysis saved to analysis.csv")
    else:
        click.echo(tabulate(metrics.items(), headers=["Metric", "Value"]))

@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--fft", is_flag=True, help="Apply FFT.")
@click.option("--wavelet", type=str, help="Apply Wavelet Transform (e.g., db4).")
@click.option("--output", type=click.Path(), required=True)
def transform(file, fft, wavelet, output):
    """Apply transforms (FFT, Wavelet) to data or audio."""
    data = data_handler.read_data(file) if not file.endswith(('.wav', '.mp3')) else audio_handler.load_audio(file)

    if fft:
        freqs, magnitudes = dsp.compute_fft(data['value'], fs=1)  # fs=1 for time-series data
        result = pd.DataFrame({"Frequency (Hz)": freqs, "Magnitude": magnitudes})
    elif wavelet:
        coeffs = transforms.wavelet_transform(data['value'], wavelet)
        result = pd.DataFrame({f"Level {i+1}": coeff for i, coeff in enumerate(coeffs)})
    else:
        raise click.UsageError("Specify a transform (FFT or Wavelet).")

    data_handler.save_data(result, output)
    click.echo(f"Transform saved to {output}")

@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--low-pass", type=float, help="Low-pass filter cutoff frequency (Hz).")
@click.option("--high-pass", type=float, help="High-pass filter cutoff frequency (Hz).")
@click.option("--band-pass", nargs=2, type=float, help="Band-pass filter cutoff frequencies (low, high).")
@click.option("--output", type=click.Path(), required=True)
def filter(file, low_pass, high_pass, band_pass, output):
    """Apply filters to a signal or audio."""
    data = data_handler.read_data(file)
    fs = 100  # Default sampling rate for time-series data (adjust based on your data)

    if low_pass:
        result = filters.low_pass_filter(data["value"], low_pass, fs)
    elif high_pass:
        result = filters.high_pass_filter(data["value"], high_pass, fs)
    elif band_pass:
        result = filters.band_pass_filter(data["value"], band_pass[0], band_pass[1], fs)
    else:
        raise click.UsageError("Specify a filter type (low-pass, high-pass, or band-pass).")

    # Save filtered data
    filtered_df = pd.DataFrame({"time": data["time"], "value": result})
    data_handler.save_data(filtered_df, output)
    click.echo(f"Filtered signal saved to {output}")

@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--query", type=str, help="SQL query to run on the data.")
@click.option("--filter", type=str, help="Pandas filter expression (e.g., 'value > 10').")
@click.option("--output", type=click.Path(), required=True)
def manipulate(file, query, filter, output):
    """Manipulate data using SQL or Pandas expressions."""
    data = data_handler.read_data(file)

    if query:
        result = data_handler.run_sql_query(data, query)
    elif filter:
        result = data_handler.filter_data(data, filter)
    else:
        raise click.UsageError("Specify a query or filter expression.")

    data_handler.save_data(result, output)
    click.echo(f"Manipulated data saved to {output}")

@click.group()
def audio():
    """Audio processing commands."""
    pass

@audio.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), help="Export audio data to csv, json, or wav.")
@click.option("--format", type=click.Choice(["csv", "json", "wav"]), help="Format for exporting audio data.")
def show(file, output, format):
    """Show audio data and optionally export it."""
    data, sr = audio_handler.load_audio(file)

    # Display audio data
    time_values = np.arange(len(data)) / sr
    audio_df = pd.DataFrame({"time": time_values, "amplitude": data})

    if not output:
        # If no output specified, just display the data
        click.echo(tabulate(audio_df.head(10), headers="keys", tablefmt="grid"))
        click.echo(f"... ({len(audio_df)} total samples)")
    else:
        # Export the data to the chosen format
        if format == "csv":
            audio_handler.save_audio_as_csv(audio_df, output)
            click.echo(f"Audio data exported to {output}")
        elif format == "json":
            audio_handler.save_audio_as_json(audio_df, output)
            click.echo(f"Audio data exported to {output}")
        elif format == "wav":
            audio_handler.save_audio(data, sr, output)
            click.echo(f"Audio data exported to {output}")
        else:
            raise click.UsageError("Unsupported format. Choose csv, json, or wav.")

@audio.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--effect", type=click.Choice(["stretch", "pitch-shift", "compression"]), required=True)
@click.option("--factor", type=float, help="Stretch factor or semitone shift (e.g., 1.5 for stretch, 2 for pitch shift).")
@click.option("--output", type=click.Path(), required=True)
def effect(file, effect, factor, output):
    """Apply audio effects like time-stretching or pitch-shifting."""
    data, sr = audio_handler.load_audio(file)

    if effect == "stretch":
        if factor is None:
            raise click.UsageError("Stretch effect requires --factor.")
        result = audio_handler.time_stretch(data, rate=factor)
    elif effect == "pitch-shift":
        if factor is None:
            raise click.UsageError("Pitch-shift effect requires --factor.")
        result = audio_handler.pitch_shift(data, sr, n_steps=factor)
    elif effect == "compression":
        result = audio_handler.dynamic_range_compression(data)
    else:
        raise click.UsageError("Invalid effect specified.")

    audio_handler.save_audio(result, sr, output)
    click.echo(f"Effect applied and saved to {output}")

@audio.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--start", type=float, required=True, help="Start time in seconds.")
@click.option("--end", type=float, required=True, help="End time in seconds.")
@click.option("--output", type=click.Path(), required=True)
def slice(file, start, end, output):
    """Slice a segment from an audio file."""
    data, sr = audio_handler.load_audio(file)
    sliced_data = audio_handler.slice_audio(data, sr, start, end)
    audio_handler.save_audio(sliced_data, sr, output)
    click.echo(f"Audio sliced and saved to {output}")

# Register the audio group
cli.add_command(audio)

@cli.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True)
@click.option("--transform", type=str, required=True, help="Transform to apply (e.g., fft, wavelet).")
@click.option("--output-dir", type=click.Path(), required=True)
def batch(input_dir, transform, output_dir):
    """Process multiple files in a directory."""
    batch_processor.process_batch(input_dir, output_dir, transform)
    click.echo(f"Batch processing completed. Results saved in {output_dir}")

@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--type", type=click.Choice(["fft", "spectrogram", "waveform"]), required=True)
@click.option("--output", type=click.Path(), required=True)
@click.option("--min_freq", type=click.Path(), required=False)
@click.option("--max_freq", type=click.Path(), required=False)
def visualize(file, type, output, min_freq, max_freq):
    """Generate visualizations like spectrograms or FFT plots."""
    if file.endswith(('.wav', '.mp3')):
        data, sr = audio_handler.load_audio(file)
    else:
        data = data_handler.read_data(file).values.flatten()
        sr = 1

    if type == "fft":
        visualizations.plot_fft(data, sr, output)
    elif type == "spectrogram":
        min_freq = float(min_freq) if min_freq else 0
        max_freq = float(max_freq) if max_freq else None
        visualizations.plot_spectrogram(data, sr, output, min_freq, max_freq)
    elif type == "waveform":
        visualizations.plot_waveform(data, sr, output)

    click.echo(f"{type.capitalize()} visualization saved to {output}")

@cli.command()
@click.option("--list", "list_plugins", is_flag=True, help="List all available plugins.")
@click.argument("plugin", required=False)
@click.argument("file", type=click.Path(exists=True), required=False)
@click.option("--output", type=click.Path(), help="Output file for plugin result.")
def plugin(list_plugins, plugin, file, output):
    """List or run custom plugins."""
    if list_plugins:
        plugins = plugin_manager.discover_plugins()
        click.echo("Available plugins:")
        for name, func in plugins.items():
            click.echo(f"- {name} (from {func.__module__})")
    elif plugin:
        plugins = plugin_manager.discover_plugins()
        if plugin not in plugins:
            raise click.UsageError(f"Plugin '{plugin}' not found.")
        func = plugins[plugin]

        data = data_handler.read_data(file)
        result = func(data)
        data_handler.save_data(result, output)
        click.echo(f"Plugin '{plugin}' applied and result saved to {output}")
    else:
        raise click.UsageError("Specify --list to view plugins or a plugin to execute.")

@cli.command()
@click.argument("expression", type=str)
@click.option("--x-range", type=str, required=True, help="Range for x as start,end,step (e.g., 0,1,0.01).")
@click.option("--output", type=click.Path(), required=True)
def math(expression, x_range, output):
    """Evaluate a custom mathematical expression."""

    # Parse x-range
    start, end, step = map(float, x_range.split(","))
    x_values = np.arange(start, end, step)

    # Evaluate the expression
    results = [eval(expression, {"np": np, "x": x}) for x in x_values]
    result_df = pd.DataFrame({"x": x_values, "result": results})

    # Save the results
    data_handler.save_data(result_df, output)
    click.echo(f"Math results saved to {output}")


if __name__ == "__main__":
    cli()
