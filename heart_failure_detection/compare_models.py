import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create output directory
output_dir = "artifacts/model_comparison"
os.makedirs(output_dir, exist_ok=True)

# Define paths to results
reduced_results_path = "artifacts/reduced_model_evaluation/reduced_model_results.csv"

# Check if results files exist
if not os.path.exists(reduced_results_path):
    print(f"Error: Reduced model results file not found at {reduced_results_path}")
    exit(1)

# Load results
reduced_results = pd.read_csv(reduced_results_path)

# Print summary
print("Reduced Dataset Model Performance:")
print(reduced_results.sort_values(by='Accuracy', ascending=False).head(5))

# Create bar chart of top models
plt.figure(figsize=(14, 8))
top_models = reduced_results.sort_values(by='Accuracy', ascending=False).head(10)
sns.barplot(x='Model', y='Accuracy', data=top_models, palette='viridis')
plt.title('Top 10 Models by Accuracy (Reduced Dataset)', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_models_accuracy.png'))
plt.close()

# Create a summary of the best model
best_model = reduced_results.sort_values(by='Accuracy', ascending=False).iloc[0]
print("\nBest Model Summary:")
print(f"Model: {best_model['Model']}")
print(f"Accuracy: {best_model['Accuracy']:.4f}")

# Create a summary table of the best model metrics
best_model_metrics = pd.DataFrame({
    'Metric': best_model.index,
    'Value': best_model.values
})
best_model_metrics = best_model_metrics[best_model_metrics['Metric'] != 'Model']
best_model_metrics = best_model_metrics.sort_values(by='Value', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Value', y='Metric', data=best_model_metrics, palette='viridis')
plt.title(f'Metrics for Best Model: {best_model["Model"]}', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Metric', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'best_model_metrics.png'))
plt.close()

print(f"\nResults saved to {output_dir}")
print("Done!")
