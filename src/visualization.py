"""
Visualization module for salience-based text summarization experiments.
Generates comprehensive plots for performance analysis, cost trade-offs, and method comparisons.
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.cost_model import add_cost_columns, calculate_cost_savings
    COST_MODEL_AVAILABLE = True
except ImportError:
    warnings.warn("src/cost_model.py not found. Using default pricing.")
    COST_MODEL_AVAILABLE = False
    
    # Define fallback cost model functions
    def add_cost_columns(df):
        """
        Add cost-related columns to the dataframe based on token counts.
        Using OpenAI GPT-4 pricing as default:
        - Input: $10 per 1M tokens
        - Output: $30 per 1M tokens
        """
        # Check if required columns exist
        required_cols = ['avg_tokens_before', 'avg_tokens_after']
        if not all(col in df.columns for col in required_cols):
            # Create dummy columns if they don't exist
            if 'avg_tokens_before' not in df.columns:
                df['avg_tokens_before'] = 1000  # Default value
            if 'avg_tokens_after' not in df.columns:
                df['avg_tokens_after'] = 500   # Default value
        
        # Calculate costs (assuming all tokens are input tokens for simplicity)
        input_cost_per_token = 10 / 1_000_000  # $10 per 1M tokens
        output_cost_per_token = 30 / 1_000_000  # $30 per 1M tokens
        
        # Calculate costs
        df['cost_before_usd'] = df['avg_tokens_before'] * input_cost_per_token
        df['cost_after_usd'] = df['avg_tokens_after'] * input_cost_per_token
        df['real_cost_savings_usd'] = df['cost_before_usd'] - df['cost_after_usd']
        df['real_cost_usd'] = df['cost_after_usd']
        
        return df
    
    def calculate_cost_savings(original_tokens, reduced_tokens):
        """Calculate cost savings based on token reduction"""
        input_cost_per_token = 10 / 1_000_000
        original_cost = original_tokens * input_cost_per_token
        reduced_cost = reduced_tokens * input_cost_per_token
        return original_cost - reduced_cost


def load_metrics_data(filepath: str = "results/metrics.csv") -> pd.DataFrame:
    """
    Load metrics data from CSV file and preprocess it.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    print(f"   Loading metrics from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    
    # Load the CSV
    df = pd.read_csv(filepath)
    
    print(f"   Raw data shape: {df.shape}")
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Deduplicate based on key columns
    key_cols = ['file', 'dataset', 'salience_type', 'token_budget']
    key_cols = [col for col in key_cols if col in df.columns]
    df = df.drop_duplicates(subset=key_cols, keep='first')
    
    print(f"   After deduplication: {df.shape}")
    
    # Process method names
    def process_method_name(row):
        # Handle baseline/full context methods
        if pd.isna(row.get('salience_type')) or str(row.get('salience_type', '')).strip() == '':
            # Check if it's a full context/baseline
            file_str = str(row.get('file', '')).lower()
            dataset_str = str(row.get('dataset', '')).lower()
            
            if 'full' in file_str or 'full' in dataset_str or 'pairs_full' in file_str:
                return 'Full Context'
            elif 'baseline' in file_str or 'baseline' in dataset_str:
                return 'Baseline'
            else:
                return 'Baseline'
        else:
            # Use the salience_type as method name
            method = str(row['salience_type']).strip()
            # Clean up method names
            if method == 'first_k':
                return 'First-K'
            elif method == 'random_k':
                return 'Random-K'
            elif method == 'lead_n':
                return 'Lead-N'
            elif method == 'tfidf':
                return 'TF-IDF'
            elif method == 'cosine':
                return 'Cosine'
            elif method == 'hybrid':
                return 'Hybrid'
            else:
                return method.title()  # Capitalize first letter
    
    df['method'] = df.apply(process_method_name, axis=1)
    
    # Clean dataset names
    def clean_dataset_name(name):
        if pd.isna(name):
            return 'unknown'
        name_str = str(name).lower()
        if 'arxiv' in name_str:
            return 'arxiv'
        elif 'cnn' in name_str or 'dailymail' in name_str:
            return 'cnn_dailymail'
        elif 'gov' in name_str or 'report' in name_str:
            return 'govreport'
        else:
            return name_str
    
    df['dataset'] = df['dataset'].apply(clean_dataset_name)
    
    # Ensure numeric columns are numeric
    numeric_cols = ['rouge1', 'rouge2', 'rougeL', 'bert_score_f1', 
                    'avg_tokens_before', 'avg_tokens_after', 'percentage_reduction',
                    'token_budget']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add cost columns
    df = add_cost_columns(df)
    
    # Filter out rows with missing essential metrics
    essential_cols = ['rouge1', 'rouge2', 'rougeL', 'bert_score_f1']
    df = df.dropna(subset=essential_cols, how='all')
    
    print(f"   Final data shape: {df.shape}")
    print(f"   Methods found: {sorted(df['method'].unique())}")
    print(f"   Datasets found: {sorted(df['dataset'].unique())}")
    
    return df


def aggregate_by_budget(df: pd.DataFrame, include_baselines: bool = True) -> pd.DataFrame:
    """
    Aggregate metrics by dataset, token budget, and method.
    
    Args:
        df: Input DataFrame
        include_baselines: Whether to include baseline methods
        
    Returns:
        Aggregated DataFrame
    """
    # Filter methods if needed
    if not include_baselines:
        baseline_methods = ['Full Context', 'Baseline']
        df = df[~df['method'].isin(baseline_methods)]
    
    # Define aggregation dictionary
    agg_dict = {
        'rouge1': 'mean',
        'rouge2': 'mean', 
        'rougeL': 'mean',
        'bert_score_f1': 'mean',
        'percentage_reduction': 'mean',
        'avg_tokens_before': 'mean',
        'avg_tokens_after': 'mean',
        'cost_before_usd': 'mean',
        'cost_after_usd': 'mean',
        'real_cost_savings_usd': 'mean',
        'real_cost_usd': 'mean'
    }
    
    # Only include columns that exist in the dataframe
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    # Group and aggregate
    group_cols = ['dataset', 'token_budget', 'method']
    group_cols = [col for col in group_cols if col in df.columns]
    
    if not group_cols:
        return df
    
    all_data = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    return all_data


def plot_tradeoff_scatter(df: pd.DataFrame, output_path: str = "results/plots/tradeoff_scatter.png") -> str:
    """
    Create scatter plot of ROUGE-L vs cost savings for different methods.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating trade-off scatter plot...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter out methods without cost savings data
    plot_df = df.copy()
    
    # Ensure we have required columns
    required_cols = ['rougeL', 'real_cost_savings_usd', 'method', 'dataset']
    missing_cols = [col for col in required_cols if col not in plot_df.columns]
    
    if missing_cols:
        warnings.warn(f"Missing columns for trade-off plot: {missing_cols}")
        # Create dummy columns if needed
        if 'real_cost_savings_usd' not in plot_df.columns and 'percentage_reduction' in plot_df.columns:
            # Use percentage reduction as proxy for cost savings
            plot_df['real_cost_savings_usd'] = plot_df['percentage_reduction'] / 100
    
    # Filter out rows with missing values
    plot_df = plot_df.dropna(subset=['rougeL', 'real_cost_savings_usd'])
    
    if plot_df.empty:
        warnings.warn("No data available for trade-off plot")
        return output_path
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    datasets = sorted(plot_df['dataset'].unique())
    
    # Define color palette for methods
    methods = sorted(plot_df['method'].unique())
    colors = sns.color_palette("husl", len(methods))
    method_colors = dict(zip(methods, colors))
    
    for idx, dataset in enumerate(datasets):
        if idx >= 3:  # Safety check
            break
            
        ax = axes[idx]
        dataset_df = plot_df[plot_df['dataset'] == dataset]
        
        # Plot each method
        for method in methods:
            method_df = dataset_df[dataset_df['method'] == method]
            if not method_df.empty:
                ax.scatter(
                    method_df['real_cost_savings_usd'], 
                    method_df['rougeL'],
                    color=method_colors[method],
                    alpha=0.7,
                    s=100,
                    label=method,
                    edgecolors='black',
                    linewidth=0.5
                )
        
        ax.set_xlabel('Cost Savings (USD)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('ROUGE-L Score', fontsize=12)
        ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(dataset_df) > 1:
            z = np.polyfit(dataset_df['real_cost_savings_usd'], dataset_df['rougeL'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(dataset_df['real_cost_savings_usd'].min(), 
                                 dataset_df['real_cost_savings_usd'].max(), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.5, linewidth=2)
    
    # Create legend
    handles = [Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=method_colors[m], markersize=10,
                      label=m) for m in methods]
    fig.legend(handles=handles, title='Methods', 
               loc='lower center', ncol=min(len(methods), 6),
               bbox_to_anchor=(0.5, -0.05), fontsize=10)
    
    plt.suptitle('Performance-Cost Trade-off: ROUGE-L vs Cost Savings', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_strategy_comparison_bar(df: pd.DataFrame, output_path: str = "results/plots/strategy_comparison.png") -> str:
    """
    Create bar plot comparing different strategies across datasets.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating strategy comparison bar plot...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Aggregate data
    all_data = aggregate_by_budget(df, include_baselines=True)
    
    # Filter out baseline for cleaner comparison
    comparison_data = all_data[~all_data['method'].isin(['Full Context', 'Baseline'])]
    
    if comparison_data.empty:
        warnings.warn("No data available for strategy comparison")
        return output_path
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bert_score_f1']
    metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore F1']
    
    # Get datasets and methods
    datasets = sorted(comparison_data['dataset'].unique())
    methods = sorted(comparison_data['method'].unique())
    
    # Set color palette
    colors = sns.color_palette("Set2", len(methods))
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        if idx >= 4:  # Safety check
            break
            
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data for this metric
        metric_data = []
        for dataset in datasets:
            dataset_df = comparison_data[comparison_data['dataset'] == dataset]
            for method in methods:
                method_df = dataset_df[dataset_df['method'] == method]
                if not method_df.empty:
                    metric_data.append({
                        'dataset': dataset,
                        'method': method,
                        'value': method_df[metric].mean(),
                        'std': method_df[metric].std() if len(method_df) > 1 else 0
                    })
        
        if not metric_data:
            continue
            
        metric_df = pd.DataFrame(metric_data)
        
        # Create grouped bar plot
        x = np.arange(len(datasets))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            method_vals = metric_df[metric_df['method'] == method]['value'].values
            method_stds = metric_df[metric_df['method'] == method]['std'].values
            
            if len(method_vals) == len(datasets):
                bars = ax.bar(x + i * width - width * (len(methods) - 1) / 2, 
                             method_vals, width, label=method,
                             color=colors[i], alpha=0.8,
                             yerr=method_stds if any(method_stds > 0) else None,
                             capsize=3, error_kw={'elinewidth': 1})
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        if idx == 0:  # Only add legend to first subplot
            ax.legend(title='Method', fontsize=10, title_fontsize=11,
                     bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Strategy Comparison Across Datasets', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_dataset_analysis(df: pd.DataFrame, output_path: str = "results/plots/dataset_analysis.png") -> str:
    """
    Create comprehensive dataset analysis plots.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating dataset analysis plot...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Token statistics
    ax1 = plt.subplot(3, 3, 1)
    if 'avg_tokens_before' in df.columns and 'avg_tokens_after' in df.columns:
        token_data = df.groupby('dataset')[['avg_tokens_before', 'avg_tokens_after']].mean()
        token_data.plot(kind='bar', ax=ax1, alpha=0.8)
        ax1.set_title('Average Token Counts', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Tokens')
        ax1.set_xlabel('Dataset')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(['Before', 'After'])
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.text(0.5, 0.5, 'Token data not available', 
                ha='center', va='center', fontsize=12)
        ax1.set_title('Average Token Counts', fontsize=14, fontweight='bold')
    
    # 2. Percentage reduction
    ax2 = plt.subplot(3, 3, 2)
    if 'percentage_reduction' in df.columns:
        reduction_data = df.groupby('dataset')['percentage_reduction'].mean()
        reduction_data.plot(kind='bar', ax=ax2, color='coral', alpha=0.8)
        ax2.set_title('Average Token Reduction', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Reduction (%)')
        ax2.set_xlabel('Dataset')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(reduction_data):
            ax2.text(i, v + 0.5, f'{v:.1f}%', 
                    ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Reduction data not available', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Average Token Reduction', fontsize=14, fontweight='bold')
    
    # 3. Cost savings
    ax3 = plt.subplot(3, 3, 3)
    if 'real_cost_savings_usd' in df.columns:
        cost_data = df.groupby('dataset')['real_cost_savings_usd'].mean()
        cost_data.plot(kind='bar', ax=ax3, color='green', alpha=0.8)
        ax3.set_title('Average Cost Savings', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Savings (USD)')
        ax3.set_xlabel('Dataset')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(cost_data):
            ax3.text(i, v + max(cost_data) * 0.01, f'${v:.4f}', 
                    ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Cost data not available', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Average Cost Savings', fontsize=14, fontweight='bold')
    
    # 4. ROUGE scores by dataset
    ax4 = plt.subplot(3, 3, 4)
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
    if all(m in df.columns for m in rouge_metrics):
        rouge_data = df.groupby('dataset')[rouge_metrics].mean()
        rouge_data.plot(kind='bar', ax=ax4, alpha=0.8)
        ax4.set_title('ROUGE Scores by Dataset', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_xlabel('Dataset')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'ROUGE data not available', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('ROUGE Scores by Dataset', fontsize=14, fontweight='bold')
    
    # 5. BERTScore by dataset
    ax5 = plt.subplot(3, 3, 5)
    if 'bert_score_f1' in df.columns:
        bert_data = df.groupby('dataset')['bert_score_f1'].mean()
        bert_data.plot(kind='bar', ax=ax5, color='purple', alpha=0.8)
        ax5.set_title('BERTScore by Dataset', fontsize=14, fontweight='bold')
        ax5.set_ylabel('BERTScore F1')
        ax5.set_xlabel('Dataset')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(bert_data):
            ax5.text(i, v + 0.005, f'{v:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'BERTScore data not available', 
                ha='center', va='center', fontsize=12)
        ax5.set_title('BERTScore by Dataset', fontsize=14, fontweight='bold')
    
    # 6. Method performance heatmap
    ax6 = plt.subplot(3, 3, 6)
    if 'method' in df.columns and 'rougeL' in df.columns:
        heatmap_data = df.pivot_table(values='rougeL', 
                                     index='dataset', 
                                     columns='method', 
                                     aggfunc='mean')
        
        if not heatmap_data.empty:
            im = ax6.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
            ax6.set_title('ROUGE-L Heatmap', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Method')
            ax6.set_ylabel('Dataset')
            ax6.set_xticks(range(len(heatmap_data.columns)))
            ax6.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
            ax6.set_yticks(range(len(heatmap_data.index)))
            ax6.set_yticklabels(heatmap_data.index)
            
            # Add colorbar
            plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
            
            # Add text annotations
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    value = heatmap_data.iloc[i, j]
                    if not pd.isna(value):
                        ax6.text(j, i, f'{value:.3f}', 
                                ha='center', va='center', 
                                color='black', fontsize=8)
        else:
            ax6.text(0.5, 0.5, 'Heatmap data not available', 
                    ha='center', va='center', fontsize=12)
            ax6.set_title('ROUGE-L Heatmap', fontsize=14, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Heatmap data not available', 
                ha='center', va='center', fontsize=12)
        ax6.set_title('ROUGE-L Heatmap', fontsize=14, fontweight='bold')
    
    # 7. Budget impact on ROUGE-L
    ax7 = plt.subplot(3, 3, 7)
    if 'token_budget' in df.columns and 'rougeL' in df.columns and 'method' in df.columns:
        budget_data = df[df['method'] != 'Full Context']  # Exclude full context
        if not budget_data.empty:
            for method in budget_data['method'].unique():
                method_df = budget_data[budget_data['method'] == method]
                ax7.scatter(method_df['token_budget'], method_df['rougeL'], 
                           alpha=0.6, s=50, label=method)
            
            ax7.set_title('Token Budget vs ROUGE-L', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Token Budget')
            ax7.set_ylabel('ROUGE-L')
            ax7.grid(True, alpha=0.3)
            ax7.legend(fontsize=9, ncol=2)
    else:
        ax7.text(0.5, 0.5, 'Budget data not available', 
                ha='center', va='center', fontsize=12)
        ax7.set_title('Token Budget vs ROUGE-L', fontsize=14, fontweight='bold')
    
    # 8. Reduction vs Performance trade-off
    ax8 = plt.subplot(3, 3, 8)
    if 'percentage_reduction' in df.columns and 'rougeL' in df.columns and 'method' in df.columns:
        tradeoff_data = df[df['method'] != 'Full Context']  # Exclude full context
        if not tradeoff_data.empty:
            for method in tradeoff_data['method'].unique():
                method_df = tradeoff_data[tradeoff_data['method'] == method]
                ax8.scatter(method_df['percentage_reduction'], method_df['rougeL'], 
                           alpha=0.6, s=50, label=method)
            
            ax8.set_title('Reduction vs Performance', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Token Reduction (%)')
            ax8.set_ylabel('ROUGE-L')
            ax8.grid(True, alpha=0.3)
            ax8.legend(fontsize=9, ncol=2)
    else:
        ax8.text(0.5, 0.5, 'Trade-off data not available', 
                ha='center', va='center', fontsize=12)
        ax8.set_title('Reduction vs Performance', fontsize=14, fontweight='bold')
    
    # 9. Dataset summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary text
    summary_text = "Dataset Summary:\n\n"
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        summary_text += f"{dataset.upper()}:\n"
        summary_text += f"  Samples: {len(dataset_df)}\n"
        
        if 'avg_tokens_before' in df.columns:
            avg_tokens = dataset_df['avg_tokens_before'].mean()
            summary_text += f"  Avg tokens: {avg_tokens:.0f}\n"
        
        if 'rougeL' in df.columns:
            avg_rouge = dataset_df['rougeL'].mean()
            summary_text += f"  Avg ROUGE-L: {avg_rouge:.3f}\n"
        
        summary_text += "\n"
    
    ax9.text(0, 1, summary_text, fontsize=10, fontfamily='monospace',
            verticalalignment='top', transform=ax9.transAxes)
    
    plt.suptitle('Dataset Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_method_performance_radar(df: pd.DataFrame, output_path: str = "results/plots/method_radar.png") -> str:
    """
    Create radar chart comparing method performance across metrics.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating method performance radar chart...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Aggregate data
    all_data = aggregate_by_budget(df, include_baselines=False)
    
    if all_data.empty:
        warnings.warn("No data available for radar chart")
        return output_path
    
    # Select metrics for radar chart
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bert_score_f1', 'percentage_reduction']
    metrics = [m for m in metrics if m in all_data.columns]
    
    if len(metrics) < 3:
        warnings.warn(f"Insufficient metrics for radar chart: {metrics}")
        return output_path
    
    # Average across datasets
    method_stats = all_data.groupby('method')[metrics].mean()
    
    # Normalize metrics to 0-1 scale for radar chart
    method_stats_normalized = method_stats.copy()
    for metric in metrics:
        if metric == 'percentage_reduction':
            # Higher is better for reduction
            method_stats_normalized[metric] = (method_stats[metric] - method_stats[metric].min()) / \
                                            (method_stats[metric].max() - method_stats[metric].min())
        else:
            # Higher is better for scores
            method_stats_normalized[metric] = (method_stats[metric] - method_stats[metric].min()) / \
                                            (method_stats[metric].max() - method_stats[metric].min())
    
    # Create radar chart
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Plot each method
    methods = method_stats_normalized.index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for idx, method in enumerate(methods):
        values = method_stats_normalized.loc[method].values.tolist()
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    # Set the labels
    metric_labels = [m.upper().replace('_', ' ') for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    
    # Set y-ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_ylim(0, 1.1)
    
    # Add title and legend
    ax.set_title('Method Performance Radar Chart\n(Normalized Metrics)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_cost_analysis(df: pd.DataFrame, output_path: str = "results/plots/cost_analysis.png") -> str:
    """
    Create cost analysis plots showing cost savings vs performance.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating cost analysis plot...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ensure we have cost data
    if 'real_cost_savings_usd' not in df.columns:
        # Create from percentage reduction if available
        if 'percentage_reduction' in df.columns and 'avg_tokens_before' in df.columns:
            input_cost_per_token = 10 / 1_000_000
            df['real_cost_savings_usd'] = df['percentage_reduction'] / 100 * df['avg_tokens_before'] * input_cost_per_token
        else:
            warnings.warn("No cost data available for cost analysis")
            return output_path
    
    # 1. Cost savings by method
    ax1 = axes[0, 0]
    if 'method' in df.columns:
        cost_by_method = df.groupby('method')['real_cost_savings_usd'].mean().sort_values()
        cost_by_method.plot(kind='barh', ax=ax1, color='teal', alpha=0.7)
        ax1.set_title('Average Cost Savings by Method', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cost Savings (USD)')
        ax1.set_ylabel('Method')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(cost_by_method):
            ax1.text(v + max(cost_by_method) * 0.01, i, f'${v:.4f}', 
                    va='center', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'Method data not available', 
                ha='center', va='center', fontsize=12)
        ax1.set_title('Average Cost Savings by Method', fontsize=14, fontweight='bold')
    
    # 2. Cost savings by dataset
    ax2 = axes[0, 1]
    if 'dataset' in df.columns:
        cost_by_dataset = df.groupby('dataset')['real_cost_savings_usd'].mean().sort_values()
        cost_by_dataset.plot(kind='bar', ax=ax2, color='orange', alpha=0.7)
        ax2.set_title('Average Cost Savings by Dataset', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Cost Savings (USD)')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(cost_by_dataset):
            ax2.text(i, v + max(cost_by_dataset) * 0.01, f'${v:.4f}', 
                    ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Dataset data not available', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Average Cost Savings by Dataset', fontsize=14, fontweight='bold')
    
    # 3. Cost vs ROUGE-L scatter
    ax3 = axes[0, 2]
    if 'rougeL' in df.columns and 'method' in df.columns:
        methods = df['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for idx, method in enumerate(methods):
            method_df = df[df['method'] == method]
            if not method_df.empty:
                ax3.scatter(method_df['real_cost_savings_usd'], method_df['rougeL'],
                           color=colors[idx], alpha=0.6, s=60, label=method)
        
        ax3.set_title('Cost Savings vs ROUGE-L', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Cost Savings (USD)')
        ax3.set_ylabel('ROUGE-L Score')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'ROUGE-L data not available', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Cost Savings vs ROUGE-L', fontsize=14, fontweight='bold')
    
    # 4. Percentage reduction vs cost savings
    ax4 = axes[1, 0]
    if 'percentage_reduction' in df.columns and 'method' in df.columns:
        methods = df['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for idx, method in enumerate(methods):
            method_df = df[df['method'] == method]
            if not method_df.empty:
                ax4.scatter(method_df['percentage_reduction'], method_df['real_cost_savings_usd'],
                           color=colors[idx], alpha=0.6, s=60, label=method)
        
        ax4.set_title('Token Reduction vs Cost Savings', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Token Reduction (%)')
        ax4.set_ylabel('Cost Savings (USD)')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Reduction data not available', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Token Reduction vs Cost Savings', fontsize=14, fontweight='bold')
    
    # 5. Cost efficiency (ROUGE-L per dollar)
    ax5 = axes[1, 1]
    if 'rougeL' in df.columns and 'real_cost_usd' in df.columns and 'method' in df.columns:
        df['cost_efficiency'] = df['rougeL'] / df['real_cost_usd'].clip(lower=0.0001)  # Avoid division by zero
        efficiency_by_method = df.groupby('method')['cost_efficiency'].mean().sort_values()
        
        efficiency_by_method.plot(kind='barh', ax=ax5, color='purple', alpha=0.7)
        ax5.set_title('Cost Efficiency (ROUGE-L per USD)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('ROUGE-L per USD')
        ax5.set_ylabel('Method')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(efficiency_by_method):
            ax5.text(v + max(efficiency_by_method) * 0.01, i, f'{v:.1f}', 
                    va='center', fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'Cost efficiency data not available', 
                ha='center', va='center', fontsize=12)
        ax5.set_title('Cost Efficiency (ROUGE-L per USD)', fontsize=14, fontweight='bold')
    
    # 6. Budget impact on cost savings
    ax6 = axes[1, 2]
    if 'token_budget' in df.columns and 'real_cost_savings_usd' in df.columns and 'method' in df.columns:
        methods = df['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for idx, method in enumerate(methods):
            method_df = df[df['method'] == method]
            if not method_df.empty:
                ax6.scatter(method_df['token_budget'], method_df['real_cost_savings_usd'],
                           color=colors[idx], alpha=0.6, s=60, label=method)
        
        ax6.set_title('Token Budget vs Cost Savings', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Token Budget')
        ax6.set_ylabel('Cost Savings (USD)')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=9)
    else:
        ax6.text(0.5, 0.5, 'Budget data not available', 
                ha='center', va='center', fontsize=12)
        ax6.set_title('Token Budget vs Cost Savings', fontsize=14, fontweight='bold')
    
    plt.suptitle('Cost Analysis Dashboard', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_performance_by_budget(df: pd.DataFrame, output_path: str = "results/plots/performance_by_budget.png") -> str:
    """
    Create line plots showing performance metrics across different token budgets.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating performance by budget plot...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter out baseline methods
    plot_df = df[~df['method'].isin(['Full Context', 'Baseline'])]
    
    if plot_df.empty or 'token_budget' not in plot_df.columns:
        warnings.warn("No budget data available")
        return output_path
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bert_score_f1']
    metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore F1']
    
    # Get unique methods and datasets
    methods = sorted(plot_df['method'].unique())
    datasets = sorted(plot_df['dataset'].unique())
    
    # Define color palette
    method_colors = sns.color_palette("tab10", len(methods))
    dataset_styles = ['-', '--', '-.', ':']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        if idx >= 4:
            break
            
        ax = axes[idx // 2, idx % 2]
        
        # Plot each method for each dataset
        for method_idx, method in enumerate(methods):
            for dataset_idx, dataset in enumerate(datasets):
                if dataset_idx >= len(dataset_styles):
                    break
                    
                method_dataset_df = plot_df[(plot_df['method'] == method) & (plot_df['dataset'] == dataset)]
                if not method_dataset_df.empty:
                    # Sort by budget
                    method_dataset_df = method_dataset_df.sort_values('token_budget')
                    
                    ax.plot(method_dataset_df['token_budget'], method_dataset_df[metric],
                           color=method_colors[method_idx],
                           linestyle=dataset_styles[dataset_idx],
                           linewidth=2,
                           marker='o',
                           markersize=6,
                           label=f'{method} ({dataset})' if idx == 0 else '')
        
        ax.set_xlabel('Token Budget', fontsize=12)
        ax.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax.set_title(f'{metric_name} by Token Budget', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to log scale if budgets vary widely
        if plot_df['token_budget'].max() / plot_df['token_budget'].min() > 100:
            ax.set_xscale('log')
    
    # Create combined legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Method (Dataset)', 
               loc='lower center', ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, -0.05), fontsize=10)
    
    plt.suptitle('Performance Metrics Across Token Budgets', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_method_comparison_grid(df: pd.DataFrame, output_path: str = "results/plots/method_comparison_grid.png") -> str:
    """
    Create a grid of small multiples comparing methods across datasets and metrics.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating method comparison grid...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter out baseline methods
    plot_df = df[~df['method'].isin(['Full Context', 'Baseline'])]
    
    if plot_df.empty:
        warnings.warn("No comparison data available")
        return output_path
    
    # Aggregate data
    aggregated = plot_df.groupby(['dataset', 'method']).agg({
        'rouge1': 'mean',
        'rouge2': 'mean',
        'rougeL': 'mean',
        'bert_score_f1': 'mean',
        'percentage_reduction': 'mean',
        'real_cost_savings_usd': 'mean'
    }).reset_index()
    
    # Create figure
    datasets = sorted(aggregated['dataset'].unique())
    methods = sorted(aggregated['method'].unique())
    
    # Define metrics to compare
    comparison_metrics = [
        ('rouge1', 'ROUGE-1', 'viridis'),
        ('rouge2', 'ROUGE-2', 'plasma'),
        ('rougeL', 'ROUGE-L', 'magma'),
        ('bert_score_f1', 'BERTScore', 'cividis'),
        ('percentage_reduction', 'Token Reduction (%)', 'coolwarm'),
        ('real_cost_savings_usd', 'Cost Savings (USD)', 'RdYlGn')
    ]
    
    # Filter to available metrics
    available_metrics = []
    for metric, name, cmap in comparison_metrics:
        if metric in aggregated.columns:
            available_metrics.append((metric, name, cmap))
    
    if not available_metrics:
        warnings.warn("No comparison metrics available")
        return output_path
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, len(datasets), figsize=(5*len(datasets), 4*n_metrics))
    
    # Handle case with single dataset or single metric
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    for metric_idx, (metric, metric_name, cmap_name) in enumerate(available_metrics):
        for dataset_idx, dataset in enumerate(datasets):
            ax = axes[metric_idx, dataset_idx]
            
            dataset_data = aggregated[aggregated['dataset'] == dataset]
            
            if not dataset_data.empty:
                # Sort methods by metric value
                dataset_data = dataset_data.sort_values(metric, ascending=True)
                
                # Get color map
                cmap = plt.cm.get_cmap(cmap_name)
                norm = Normalize(vmin=dataset_data[metric].min(), 
                               vmax=dataset_data[metric].max())
                
                # Create horizontal bar chart
                bars = ax.barh(range(len(dataset_data)), dataset_data[metric],
                             color=cmap(norm(dataset_data[metric].values)),
                             alpha=0.8,
                             edgecolor='black',
                             linewidth=0.5)
                
                ax.set_yticks(range(len(dataset_data)))
                ax.set_yticklabels(dataset_data['method'], fontsize=9)
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, dataset_data[metric])):
                    if metric in ['rouge1', 'rouge2', 'rougeL', 'bert_score_f1']:
                        label = f'{value:.3f}'
                    elif metric == 'percentage_reduction':
                        label = f'{value:.1f}%'
                    elif metric == 'real_cost_savings_usd':
                        label = f'${value:.4f}'
                    else:
                        label = f'{value:.2f}'
                    
                    ax.text(bar.get_width() + max(dataset_data[metric]) * 0.01, 
                           bar.get_y() + bar.get_height()/2,
                           label, va='center', fontsize=8)
                
                # Set titles
                if metric_idx == 0:
                    ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
                if dataset_idx == 0:
                    ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
                
                ax.grid(True, alpha=0.3, axis='x')
                ax.set_xlim(0, dataset_data[metric].max() * 1.2)
    
    plt.suptitle('Method Comparison Grid', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_summary_table(df: pd.DataFrame, output_path: str = "results/plots/summary_table.png") -> str:
    """
    Create a summary table of key statistics.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating summary table...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create summary statistics
    summary_data = []
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        
        for method in sorted(dataset_df['method'].unique()):
            method_df = dataset_df[dataset_df['method'] == dataset_df['method']]
            
            if not method_df.empty:
                summary_data.append({
                    'Dataset': dataset.upper(),
                    'Method': method,
                    'ROUGE-1': f"{method_df['rouge1'].mean():.3f}",
                    'ROUGE-2': f"{method_df['rouge2'].mean():.3f}",
                    'ROUGE-L': f"{method_df['rougeL'].mean():.3f}",
                    'BERTScore': f"{method_df['bert_score_f1'].mean():.3f}",
                    'Token Reduction': f"{method_df['percentage_reduction'].mean():.1f}%" if 'percentage_reduction' in method_df.columns else 'N/A',
                    'Cost Savings': f"${method_df['real_cost_savings_usd'].mean():.4f}" if 'real_cost_savings_usd' in method_df.columns else 'N/A'
                })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(4, len(summary_data) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Create table
        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center',
                        colColours=['#f2f2f2'] * len(summary_df.columns))
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Highlight header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4c72b0')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_df) + 1):
            if i % 2 == 0:
                for j in range(len(summary_df.columns)):
                    table[(i, j)].set_facecolor('#f9f9f9')
        
        plt.title('Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    else:
        ax.text(0.5, 0.5, 'No summary data available', 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path

def plot_bootstrap_ci(df: pd.DataFrame, output_path: str = "results/plots/bootstrap_ci.png") -> str:
    """
    Create bootstrap confidence interval plots for method comparison.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating bootstrap confidence interval plot...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter out baseline methods for comparison
    comparison_df = df[~df['method'].isin(['Full Context', 'Baseline'])]
    
    if comparison_df.empty:
        warnings.warn("No data available for bootstrap CI")
        return output_path
    
    # Bootstrap parameters
    n_bootstraps = 1000
    alpha = 0.95  # 95% confidence interval
    
    # Prepare data for bootstrap
    datasets = sorted(comparison_df['dataset'].unique())
    methods = sorted(comparison_df['method'].unique())
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bert_score_f1']
    
    # Create figure
    fig, axes = plt.subplots(len(datasets), len(metrics), 
                            figsize=(5*len(metrics), 4*len(datasets)))
    
    # Handle single dataset case
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    
    for dataset_idx, dataset in enumerate(datasets):
        dataset_df = comparison_df[comparison_df['dataset'] == dataset]
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[dataset_idx, metric_idx]
            
            bootstrap_data = []
            
            # Bootstrap for each method
            for method in methods:
                method_data = dataset_df[dataset_df['method'] == method][metric].dropna()
                
                if len(method_data) > 0:
                    # Perform bootstrap
                    bootstrap_stats = []
                    for _ in range(n_bootstraps):
                        sample = np.random.choice(method_data, size=len(method_data), replace=True)
                        bootstrap_stats.append(np.mean(sample))
                    
                    # Calculate confidence interval
                    lower = np.percentile(bootstrap_stats, (1 - alpha) * 100 / 2)
                    upper = np.percentile(bootstrap_stats, 100 - (1 - alpha) * 100 / 2)
                    mean_val = np.mean(method_data)
                    
                    bootstrap_data.append({
                        'method': method,
                        'mean': mean_val,
                        'lower': lower,
                        'upper': upper,
                        'ci_width': upper - lower
                    })
            
            if bootstrap_data:
                bootstrap_df = pd.DataFrame(bootstrap_data)
                
                # Sort by mean value
                bootstrap_df = bootstrap_df.sort_values('mean')
                
                # Plot error bars
                y_pos = np.arange(len(bootstrap_df))
                ax.errorbar(bootstrap_df['mean'], y_pos,
                          xerr=[bootstrap_df['mean'] - bootstrap_df['lower'], 
                                bootstrap_df['upper'] - bootstrap_df['mean']],
                          fmt='o', capsize=5, capthick=2, 
                          alpha=0.8, markersize=8)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(bootstrap_df['method'], fontsize=9)
                ax.set_xlabel(f'{metric.upper()} Score', fontsize=10)
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add zero line and set title
                if dataset_idx == 0:
                    ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
                if metric_idx == 0:
                    ax.set_ylabel(f'{dataset.upper()}', fontsize=11, fontweight='bold')
    
    plt.suptitle('Bootstrap 95% Confidence Intervals for Method Performance', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_rouge_drop(df: pd.DataFrame, output_path: str = "results/plots/rouge_drop.png") -> str:
    """
    Plot ROUGE score drop relative to full context for each method.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating ROUGE drop plot...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get full context baseline for each dataset
    full_context_rows = []
    
    for dataset in df['dataset'].unique():
        # Try to find full context baseline
        dataset_full = df[(df['dataset'] == dataset) & 
                         (df['method'].isin(['Full Context', 'Baseline']))]
        
        if not dataset_full.empty:
            # Take the first full context row
            full_row = dataset_full.iloc[0]
            full_context_rows.append({
                'dataset': dataset,
                'rouge1_full': full_row['rouge1'],
                'rouge2_full': full_row['rouge2'],
                'rougeL_full': full_row['rougeL'],
                'bert_full': full_row['bert_score_f1']
            })
    
    if not full_context_rows:
        warnings.warn("No full context baseline found for ROUGE drop calculation")
        return output_path
    
    full_context_df = pd.DataFrame(full_context_rows)
    
    # Filter out baseline methods for comparison
    comparison_df = df[~df['method'].isin(['Full Context', 'Baseline'])]
    
    if comparison_df.empty:
        warnings.warn("No comparison data available")
        return output_path
    
    # Calculate ROUGE drop percentage
    drop_data = []
    
    for _, row in comparison_df.iterrows():
        dataset = row['dataset']
        full_data = full_context_df[full_context_df['dataset'] == dataset]
        
        if not full_data.empty:
            full_row = full_data.iloc[0]
            
            # Calculate percentage drop (negative means improvement)
            rouge1_drop = ((row['rouge1'] - full_row['rouge1_full']) / full_row['rouge1_full']) * 100
            rouge2_drop = ((row['rouge2'] - full_row['rouge2_full']) / full_row['rouge2_full']) * 100
            rougeL_drop = ((row['rougeL'] - full_row['rougeL_full']) / full_row['rougeL_full']) * 100
            bert_drop = ((row['bert_score_f1'] - full_row['bert_full']) / full_row['bert_full']) * 100
            
            drop_data.append({
                'dataset': dataset,
                'method': row['method'],
                'rouge1_drop': rouge1_drop,
                'rouge2_drop': rouge2_drop,
                'rougeL_drop': rougeL_drop,
                'bert_drop': bert_drop,
                'percentage_reduction': row.get('percentage_reduction', 0),
                'cost_savings': row.get('real_cost_savings_usd', 0)
            })
    
    if not drop_data:
        warnings.warn("Could not calculate ROUGE drop")
        return output_path
    
    drop_df = pd.DataFrame(drop_data)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot ROUGE-L drop vs token reduction
    ax1 = axes[0, 0]
    for method in drop_df['method'].unique():
        method_data = drop_df[drop_df['method'] == method]
        ax1.scatter(method_data['percentage_reduction'], method_data['rougeL_drop'],
                   alpha=0.7, s=80, label=method)
    
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No Drop (Full Context)')
    ax1.set_xlabel('Token Reduction (%)', fontsize=12)
    ax1.set_ylabel('ROUGE-L Drop (%)', fontsize=12)
    ax1.set_title('ROUGE-L Drop vs Token Reduction', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Plot ROUGE drop by method
    ax2 = axes[0, 1]
    methods = drop_df['method'].unique()
    x = np.arange(len(methods))
    width = 0.2
    
    for idx, metric in enumerate(['rouge1_drop', 'rouge2_drop', 'rougeL_drop']):
        metric_means = [drop_df[drop_df['method'] == m][metric].mean() for m in methods]
        metric_stds = [drop_df[drop_df['method'] == m][metric].std() for m in methods]
        
        ax2.bar(x + (idx - 1) * width, metric_means, width,
               label=metric.replace('_drop', '').upper(),
               alpha=0.7, yerr=metric_stds, capsize=3)
    
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('ROUGE Drop (%)', fontsize=12)
    ax2.set_title('Average ROUGE Drop by Method', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot ROUGE-L drop distribution
    ax3 = axes[1, 0]
    drop_metrics = ['rouge1_drop', 'rouge2_drop', 'rougeL_drop']
    
    for metric in drop_metrics:
        metric_data = [drop_df[metric].dropna().values]
        bp = ax3.boxplot(metric_data, positions=[drop_metrics.index(metric)],
                        widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('ROUGE Metric', fontsize=12)
    ax3.set_ylabel('Drop (%)', fontsize=12)
    ax3.set_title('ROUGE Drop Distribution', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(drop_metrics)))
    ax3.set_xticklabels(['R-1', 'R-2', 'R-L'])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot drop vs cost savings heatmap
    ax4 = axes[1, 1]
    
    # Create pivot table for heatmap
    heatmap_data = drop_df.pivot_table(values='rougeL_drop', 
                                      index='method', 
                                      columns='dataset', 
                                      aggfunc='mean')
    
    if not heatmap_data.empty:
        im = ax4.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto')
        ax4.set_title('ROUGE-L Drop Heatmap', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Dataset', fontsize=12)
        ax4.set_ylabel('Method', fontsize=12)
        ax4.set_xticks(range(len(heatmap_data.columns)))
        ax4.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
        ax4.set_yticks(range(len(heatmap_data.index)))
        ax4.set_yticklabels(heatmap_data.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('ROUGE-L Drop (%)', fontsize=10)
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if not pd.isna(value):
                    color = 'black' if abs(value) < 20 else 'white'
                    ax4.text(j, i, f'{value:.1f}%', 
                            ha='center', va='center', 
                            color=color, fontsize=9, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Heatmap data not available', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('ROUGE-L Drop Heatmap', fontsize=14, fontweight='bold')
    
    plt.suptitle('ROUGE Score Drop Analysis Relative to Full Context', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_cost_vs_quality_aggregated(df: pd.DataFrame, output_path: str = "results/plots/cost_vs_quality.png") -> str:
    """
    Plot cost vs quality trade-off curves.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating cost vs quality plot...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ensure we have cost and quality metrics
    required_cols = ['rougeL', 'real_cost_usd', 'method', 'dataset']
    if not all(col in df.columns for col in required_cols):
        warnings.warn(f"Missing required columns: {required_cols}")
        return output_path
    
    # Filter out baseline methods
    plot_df = df[~df['method'].isin(['Full Context', 'Baseline'])]
    
    if plot_df.empty:
        warnings.warn("No comparison data available")
        return output_path
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    datasets = sorted(plot_df['dataset'].unique())
    
    # Define color palette
    methods = sorted(plot_df['method'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for idx, dataset in enumerate(datasets):
        if idx >= 3:
            break
            
        ax = axes[idx]
        dataset_df = plot_df[plot_df['dataset'] == dataset]
        
        # Plot Pareto frontier
        for method_idx, method in enumerate(methods):
            method_df = dataset_df[dataset_df['method'] == method]
            
            if not method_df.empty:
                # Average across budgets if multiple exist
                avg_cost = method_df['real_cost_usd'].mean()
                avg_rouge = method_df['rougeL'].mean()
                
                ax.scatter(avg_cost, avg_rouge,
                          color=colors[method_idx],
                          s=200, alpha=0.8,
                          label=method,
                          edgecolors='black',
                          linewidth=1.5,
                          zorder=5)
                
                # Add method label
                ax.text(avg_cost, avg_rouge, method, 
                       fontsize=9, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Find Pareto optimal points
        points = []
        for method in methods:
            method_df = dataset_df[dataset_df['method'] == method]
            if not method_df.empty:
                points.append({
                    'method': method,
                    'cost': method_df['real_cost_usd'].mean(),
                    'rouge': method_df['rougeL'].mean()
                })
        
        if points:
            points_df = pd.DataFrame(points)
            points_df = points_df.sort_values('cost')
            
            # Simple Pareto frontier detection
            pareto_points = []
            best_rouge = -np.inf
            
            for _, point in points_df.iterrows():
                if point['rouge'] > best_rouge:
                    pareto_points.append((point['cost'], point['rouge']))
                    best_rouge = point['rouge']
            
            if len(pareto_points) > 1:
                pareto_costs, pareto_rouges = zip(*pareto_points)
                ax.plot(pareto_costs, pareto_rouges, 'r--', alpha=0.7, linewidth=2,
                       label='Pareto Frontier')
        
        ax.set_xlabel('Cost (USD)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('ROUGE-L Score', fontsize=12)
        ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add baseline reference if available
        baseline_df = df[(df['dataset'] == dataset) & (df['method'].isin(['Full Context', 'Baseline']))]
        if not baseline_df.empty:
            baseline_cost = baseline_df['real_cost_usd'].iloc[0]
            baseline_rouge = baseline_df['rougeL'].iloc[0]
            ax.axvline(x=baseline_cost, color='gray', linestyle=':', alpha=0.7, label='Full Context Cost')
            ax.axhline(y=baseline_rouge, color='gray', linestyle=':', alpha=0.7, label='Full Context ROUGE-L')
    
    # Create legend
    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    fig.legend(unique_handles, unique_labels, 
               loc='lower center', ncol=min(len(unique_labels), 4),
               bbox_to_anchor=(0.5, -0.05), fontsize=10)
    
    plt.suptitle('Cost vs Quality Trade-off: Pareto Frontier Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def plot_token_distribution(df: pd.DataFrame, output_path: str = "results/plots/token_distribution.png") -> str:
    """
    Plot token distribution before and after compression.
    
    Args:
        df: DataFrame with metrics
        output_path: Output file path
        
    Returns:
        Path to saved plot
    """
    print(f"   Creating token distribution plot...")
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check for required columns
    if 'avg_tokens_before' not in df.columns or 'avg_tokens_after' not in df.columns:
        warnings.warn("Token distribution data not available")
        return output_path
    
    # Filter out baseline methods
    plot_df = df[~df['method'].isin(['Full Context', 'Baseline'])]
    
    if plot_df.empty:
        warnings.warn("No token distribution data available")
        return output_path
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Token reduction by method
    ax1 = axes[0, 0]
    method_reduction = plot_df.groupby('method')['percentage_reduction'].mean().sort_values()
    method_reduction.plot(kind='barh', ax=ax1, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Token Reduction (%)', fontsize=12)
    ax1.set_ylabel('Method', fontsize=12)
    ax1.set_title('Average Token Reduction by Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(method_reduction):
        ax1.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
    
    # 2. Before/after token comparison
    ax2 = axes[0, 1]
    token_comparison = plot_df.groupby('method')[['avg_tokens_before', 'avg_tokens_after']].mean()
    token_comparison.plot(kind='bar', ax=ax2, alpha=0.7)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Token Count', fontsize=12)
    ax2.set_title('Token Count Before vs After Compression', fontsize=14, fontweight='bold')
    ax2.legend(['Before', 'After'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Token reduction vs ROUGE-L scatter
    ax3 = axes[1, 0]
    methods = plot_df['method'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for idx, method in enumerate(methods):
        method_df = plot_df[plot_df['method'] == method]
        if not method_df.empty:
            ax3.scatter(method_df['percentage_reduction'], method_df['rougeL'],
                       color=colors[idx], alpha=0.6, s=60, label=method)
    
    ax3.set_xlabel('Token Reduction (%)', fontsize=12)
    ax3.set_ylabel('ROUGE-L Score', fontsize=12)
    ax3.set_title('Token Reduction vs ROUGE-L', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # 4. Token reduction distribution
    ax4 = axes[1, 1]
    reduction_data = [plot_df[plot_df['method'] == m]['percentage_reduction'].dropna().values 
                     for m in methods if len(plot_df[plot_df['method'] == m]) > 0]
    
    if reduction_data:
        bp = ax4.boxplot(reduction_data, labels=methods, patch_artist=True)
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(reduction_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_xlabel('Method', fontsize=12)
        ax4.set_ylabel('Token Reduction (%)', fontsize=12)
        ax4.set_title('Token Reduction Distribution by Method', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No reduction data available', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Token Reduction Distribution', fontsize=14, fontweight='bold')
    
    plt.suptitle('Token Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")
    return output_path


def generate_all_visualizations(metrics_file: str = "results/metrics.csv", 
                               output_dir: str = "results/plots") -> List[str]:
    """
    Generate all visualizations from metrics data.
    
    Args:
        metrics_file: Path to metrics CSV file
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated plots
    """
    print("=" * 60)
    print("Generating all visualizations...")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("[1/4] Loading metrics data...")
    try:
        df = load_metrics_data(metrics_file)
        if df.empty:
            raise ValueError("Loaded dataframe is empty")
        print(f"   Successfully loaded {len(df)} records")
    except Exception as e:
        print(f"   Error loading metrics: {e}")
        return []
    
    # Generate plots
    print("[2/4] Generating core analysis plots...")
    plot_files = []
    
    # Core plots
    core_plots = [
        (plot_tradeoff_scatter, "tradeoff_scatter.png"),
        (plot_strategy_comparison_bar, "strategy_comparison.png"),
        (plot_dataset_analysis, "dataset_analysis.png"),
        (plot_cost_analysis, "cost_analysis.png"),
        (plot_performance_by_budget, "performance_by_budget.png"),
        (plot_bootstrap_ci, "bootstrap_ci.png"),
        (plot_rouge_drop, "rouge_drop.png"),
        (plot_cost_vs_quality_aggregated, "cost_vs_quality.png"),
        (plot_token_distribution, "token_distribution.png")
    ]
    
    for plot_func, filename in core_plots:
        try:
            plot_path = plot_func(df, os.path.join(output_dir, filename))
            plot_files.append(plot_path)
        except Exception as e:
            print(f"   Error generating {filename}: {e}")
    
    print("[3/4] Generating method comparison plots...")
    # Method comparison plots
    comparison_plots = [
        (plot_method_performance_radar, "method_radar.png"),
        (plot_method_comparison_grid, "method_comparison_grid.png"),
        (plot_summary_table, "summary_table.png")
    ]
    
    for plot_func, filename in comparison_plots:
        try:
            plot_path = plot_func(df, os.path.join(output_dir, filename))
            plot_files.append(plot_path)
        except Exception as e:
            print(f"   Error generating {filename}: {e}")
    
    print("[4/4] Generating summary statistics...")
    # Create a summary CSV with aggregated statistics
    try:
        summary_path = os.path.join(output_dir, "summary_statistics.csv")
        aggregated = aggregate_by_budget(df, include_baselines=True)
        aggregated.to_csv(summary_path, index=False)
        print(f"   Summary statistics saved to: {summary_path}")
        plot_files.append(summary_path)
    except Exception as e:
        print(f"   Error generating summary statistics: {e}")
    
    print("\n" + "=" * 60)
    print(f"Visualization complete!")
    print(f"Generated {len(plot_files)} plots/files:")
    for plot_file in plot_files:
        print(f"  - {plot_file}")
    print("=" * 60)
    
    return plot_files


def generate_single_plot(plot_type: str, df: pd.DataFrame = None, 
                        metrics_file: str = "results/metrics.csv",
                        output_dir: str = "results/plots") -> str:
    """
    Generate a single specific plot type.
    
    Args:
        plot_type: Type of plot to generate
        df: DataFrame with metrics (optional, loads from file if not provided)
        metrics_file: Path to metrics CSV file
        output_dir: Directory to save plot
        
    Returns:
        Path to saved plot
    """
    # Map plot types to functions
    plot_functions = {
        'tradeoff': (plot_tradeoff_scatter, 'tradeoff_scatter.png'),
        'strategy': (plot_strategy_comparison_bar, 'strategy_comparison.png'),
        'dataset': (plot_dataset_analysis, 'dataset_analysis.png'),
        'cost': (plot_cost_analysis, 'cost_analysis.png'),
        'budget': (plot_performance_by_budget, 'performance_by_budget.png'),
        'radar': (plot_method_performance_radar, 'method_radar.png'),
        'grid': (plot_method_comparison_grid, 'method_comparison_grid.png'),
        'table': (plot_summary_table, 'summary_table.png'),
        'all': (generate_all_visualizations, 'all_plots')
    }
    
    if plot_type not in plot_functions:
        available = ', '.join(plot_functions.keys())
        raise ValueError(f"Unknown plot type: {plot_type}. Available: {available}")
    
    # Load data if not provided
    if df is None:
        df = load_metrics_data(metrics_file)
    
    # Generate the plot
    if plot_type == 'all':
        return generate_all_visualizations(metrics_file, output_dir)
    else:
        plot_func, filename = plot_functions[plot_type]
        output_path = os.path.join(output_dir, filename)
        return plot_func(df, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations from metrics data')
    parser.add_argument('--metrics', type=str, default='results/metrics.csv',
                       help='Path to metrics CSV file')
    parser.add_argument('--output', type=str, default='results/plots',
                       help='Output directory for plots')
    parser.add_argument('--plot-type', type=str, default='all',
                       help='Type of plot to generate (tradeoff, strategy, dataset, cost, budget, radar, grid, table, all)')
    
    args = parser.parse_args()
    
    # Generate visualizations
    if args.plot_type == 'all':
        plot_files = generate_all_visualizations(args.metrics, args.output)
    else:
        plot_file = generate_single_plot(args.plot_type, 
                                        metrics_file=args.metrics,
                                        output_dir=args.output)
        print(f"Generated plot: {plot_file}")