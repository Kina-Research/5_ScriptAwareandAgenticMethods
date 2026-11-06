import matplotlib.pyplot as plt
import pandas as pd
import json
import os

def create_comparison_plots(report_path='evaluation_report.json'):
    if not os.path.exists(report_path):
        print(f"Report {report_path} not found.")
        return
    
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    tasks = data['Task_Comparisons']
    model_names = ['Baseline', 'Script-Aware', 'Agentic']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle('Amharic NLP Experimental Results: Comparison Across Models', fontsize=16)
    
    for i, (task_name, results) in enumerate(tasks.items()):
        metric = results['Metric']
        values = [results[m] for m in model_names]
        
        axes[i].bar(model_names, values, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[i].set_title(f"{task_name}\n({metric})")
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1.0) # Normalizing to check relative gains
        
        # Adding labels on top of bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparison_results.png', dpi=300)
    print("Comparison results plot saved as comparison_results.png")

if __name__ == "__main__":
    create_comparison_plots()
