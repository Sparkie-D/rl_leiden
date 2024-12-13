import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='SingleSample_CNV')
    parser.add_argument('--data_name', type=str, default='SOL1307')
    parser.add_argument('--method', type=str, default='leiden')
    parser.add_argument('--data_type', type=str, default='CNV')
    parser.add_argument('--meta_column', type=str, default='celltype', help='Column in meta_df to calculate proportions')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # Step 1: Load data
    meta_df = pd.read_csv(f'/home/ubuntu/duxinghao/clone/data/{args.task_type}/{args.data_name}_meta.csv', index_col=0)
    tree_path_df = pd.read_csv(f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{args.task_type}/{args.method}/{args.data_type}/leiden/{args.data_name}_{args.data_type}/tree_path.csv', index_col=0)
    cell2cluster_df = pd.read_csv(f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{args.task_type}/{args.method}/{args.data_type}/leiden/{args.data_name}_{args.data_type}/cell2cluster.csv', index_col=0)

    # Merge data to associate cells with clusters and the selected meta column
    meta_df.reset_index(inplace=True)
    meta_df.rename(columns={'index': 'cell'}, inplace=True)
    merged_df = meta_df.merge(cell2cluster_df, on='cell', how='inner')

    # Step 2: Compute counts of each category per cluster node
    meta_column = args.meta_column
    node_category_counts = merged_df.groupby(['cluster', meta_column]).size().unstack(fill_value=0)

    # Step 3: Calculate proportions for each category in each cluster
    node_ratios = node_category_counts.div(node_category_counts.sum(axis=1), axis=0)

    # Step 4: Calculate purity for each node (maximum category proportion)
    node_purity = node_ratios.max(axis=1)

    # Step 5: Calculate weighted average purity across all nodes
    total_cells_per_node = node_category_counts.sum(axis=1)  # Total number of cells in each node
    average_purity = (node_purity * total_cells_per_node).sum() / total_cells_per_node.sum()

    # Step 6: Visualize proportions
    node_ratios_sorted = node_ratios.sort_values(by=node_ratios.columns[0], ascending=False)
    ax = node_ratios_sorted.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')

    plt.title(f'{meta_column} Proportion in Each Cluster Node\n(Average Purity: {average_purity:.4f})', fontsize=14)
    plt.xlabel('Cluster Node')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.legend(title=meta_column, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add average purity to the plot
    plt.text(1.2, 0.7, f'Avg Purity: {average_purity:.4f}', transform=plt.gca().transAxes, fontsize=12, ha='right')

    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{args.task_type}/{args.method}/{args.data_type}/leiden/{args.data_name}_{args.data_type}/{meta_column}_proportions.pdf')