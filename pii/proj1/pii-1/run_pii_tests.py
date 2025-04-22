import unittest
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pii_protection.test_pii_detection import PIIDetectionTest

def generate_html_report(test_results_dir: Path) -> None:
    """Generate an HTML report from test results"""
    
    # Load test results
    with open(test_results_dir / "pii_detection_results.json") as f:
        results = json.load(f)
    with open(test_results_dir / "performance_metrics.json") as f:
        metrics = json.load(f)
        
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PII Detection Test Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric-box {{ 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 10px; 
                border-radius: 5px;
                background-color: #f9f9f9;
            }}
            .visualization {{ margin: 20px 0; }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin: 20px 0;
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left;
            }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>PII Detection Test Results</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Performance Metrics</h2>
        <div class="metric-box">
            <p>Total Texts Processed: {metrics['total_texts_processed']}</p>
            <p>Average Processing Time: {metrics['avg_processing_time']:.2f} seconds</p>
        </div>
        
        <h2>Entity Types Found</h2>
        <table>
            <tr><th>Entity Type</th><th>Count</th></tr>
            {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics['entity_types_found'].items())}
        </table>
        
        <h2>Detection Methods</h2>
        <table>
            <tr><th>Method</th><th>Count</th></tr>
            {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics['detection_methods'].items())}
        </table>
        
        <h2>Visualizations</h2>
        <div class="visualization">
            <h3>Processing Time Distribution</h3>
            <img src="processing_time_distribution.png" alt="Processing Time Distribution">
        </div>
        <div class="visualization">
            <h3>Entity Types by Language</h3>
            <img src="entities_by_language.png" alt="Entity Types by Language">
        </div>
        <div class="visualization">
            <h3>Detection Methods Distribution</h3>
            <img src="detection_methods_distribution.png" alt="Detection Methods Distribution">
        </div>
        
        <h2>Sample Results</h2>
        <table>
            <tr>
                <th>Text ID</th>
                <th>Language</th>
                <th>Processing Time</th>
                <th>Entities Found</th>
            </tr>
            {''.join(f"""
                <tr>
                    <td>{r['text_id']}</td>
                    <td>{r['language']}</td>
                    <td>{r['processing_time']:.2f}s</td>
                    <td>{len(r['detected_entities'])}</td>
                </tr>
            """ for r in results[:10])}
        </table>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(test_results_dir / "test_report.html", 'w') as f:
        f.write(html_content)

def main():
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(PIIDetectionTest)
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    
    # Generate report if tests passed
    if test_result.wasSuccessful():
        test_results_dir = Path("test_results")
        generate_html_report(test_results_dir)
        print(f"\nTest report generated at {test_results_dir / 'test_report.html'}")
    else:
        print("\nTests failed. Report generation skipped.")

if __name__ == "__main__":
    main() 