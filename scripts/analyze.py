"""Analysis script for post-training model evaluation."""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_fx.analysis.plotting import TrainingAnalyzer
from neural_fx.artifacts import load_model
from neural_fx.data.dataset import AudioDataset


def load_checkpoint(checkpoint_path: str, config_path: str | None = None):
    loaded = load_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    return loaded.model, loaded.config


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trained neural audio effects model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML for legacy checkpoints without embedded configuration",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input audio file (overrides config)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to target audio file (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=48000,
        help="Number of samples to analyze",
    )
    parser.add_argument(
        "--generate_html",
        action="store_true",
        help="Generate HTML summary report",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config = load_checkpoint(args.checkpoint, args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input/target files
    input_file = args.input or config.data.train.input
    target_file = args.target or config.data.train.target

    print(f"Creating dataset from {input_file} and {target_file}...")

    # Create dataset
    dataset = AudioDataset(
        input_path=input_file,
        target_path=target_file,
        segment_length=config.training.segment_length,
        sample_rate=config.sample_rate,
        random_segments=False,
    )

    print(f"Dataset created with {len(dataset)} segments")

    # Create analyzer
    analyzer = TrainingAnalyzer(model, config)

    # Generate report
    print("Generating analysis report...")
    report = analyzer.generate_report(dataset, output_dir, args.num_samples)

    # Print summary
    print("\n" + "=" * 50)
    print("Analysis Summary")
    print("=" * 50)
    print(f"ESR: {report['esr']:.6f} - {report['esr_comment']}")
    print(f"MSE: {report['mse']:.6f}")
    print(f"Correlation: {report['correlation']:.4f}")
    print(f"Model Parameters: {report['num_params']:,}")
    print("\nPlots saved to:")
    for name, path in report["plots"].items():
        print(f"  {name}: {path}")

    # Generate HTML report if requested
    if args.generate_html:
        html_path = output_dir / "report.html"
        generate_html_report(report, html_path)
        print(f"\nHTML report saved to: {html_path}")

    # Save JSON report
    json_path = output_dir / "report.json"
    with open(json_path, "w") as f:
        # Convert to serializable format
        json_report = {
            "esr": float(report["esr"]),
            "esr_comment": report["esr_comment"],
            "mse": float(report["mse"]),
            "correlation": float(report["correlation"]),
            "num_params": int(report["num_params"]),
            "plots": report["plots"],
        }
        json.dump(json_report, f, indent=2)
    print(f"JSON report saved to: {json_path}")

    print("\nAnalysis complete!")


def generate_html_report(report: dict, output_path: Path):
    """Generate HTML summary report."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Neural Audio Effects - Model Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        .metric {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            font-size: 24px;
            color: #2c3e50;
        }}
        .good {{
            color: #27ae60;
        }}
        .warning {{
            color: #f39c12;
        }}
        .bad {{
            color: #e74c3c;
        }}
        .plot {{
            margin: 20px 0;
        }}
        .plot img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Neural Audio Effects - Model Analysis Report</h1>

    <h2>Metrics</h2>
    <div class="metric">
        <div class="metric-label">Error-to-Signal Ratio (ESR)</div>
        <div class="metric-value">{report["esr"]:.6f}</div>
        <div class="{get_esr_class(report["esr"])}">{report["esr_comment"]}</div>
    </div>

    <div class="metric">
        <div class="metric-label">Mean Squared Error (MSE)</div>
        <div class="metric-value">{report["mse"]:.6f}</div>
    </div>

    <div class="metric">
        <div class="metric-label">Correlation Coefficient</div>
        <div class="metric-value">{report["correlation"]:.4f}</div>
    </div>

    <div class="metric">
        <div class="metric-label">Model Parameters</div>
        <div class="metric-value">{report["num_params"]:,}</div>
    </div>

    <h2>Visualizations</h2>
    <div class="plot">
        <h3>Waveform Comparison</h3>
        <img src="waveform_comparison.png" alt="Waveform Comparison">
    </div>
    <div class="plot">
        <h3>Spectrograms</h3>
        <img src="spectrograms.png" alt="Spectrograms">
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html_content)


def get_esr_class(esr: float) -> str:
    """Get CSS class based on ESR value."""
    if esr < 0.01:
        return "good"
    elif esr < 0.1:
        return "warning"
    else:
        return "bad"


if __name__ == "__main__":
    main()
