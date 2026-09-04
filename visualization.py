import os
import math
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class Visualization:
    def __init__(self):
        pass

    @staticmethod
    def visuals(df, folder_name="preprocessing"):
        # 1. Ensure the directory exists (creates it if it doesn't)
        os.makedirs(folder_name, exist_ok=True)

        # ── STAGE 3: MISSING VALUES ──────────────────────────────────────
        missing     = df.isnull().sum()
        missing_pct = missing / len(df) * 100
        missing_df  = pd.DataFrame({'count': missing, '%': missing_pct})
        missing_df  = missing_df[missing_df['count'] > 0].sort_values('%', ascending=False)
        print(missing_df)

        plt.figure(figsize=(14, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='gray')
        plt.title('Missing value map')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'missing_value_map.png'))
        plt.close()

        rows_any_missing = df.isnull().any(axis=1).sum()
        print(f"Rows with any missing: {rows_any_missing} ({rows_any_missing/len(df)*100:.1f}%)")

        # ── STAGE 5: TARGET COLUMN ───────────────────────────────────────
        target = 'chronic_disease'       
        if target in df.columns:
            counts = df[target].value_counts()
            pct    = df[target].value_counts(normalize=True) * 100

            plt.figure(figsize=(6, 6))   
            labels = {1: 'Have diabetes', 0: 'Diabetes free'}
            pars = plt.bar(pct.index, pct.values)
            pct.index = pct.index.map(labels).astype(str)
            for bar, value in zip(pars, pct.values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom"
                )
            plt.title('Target Class Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, 'target_class_distribution.png'))
            plt.close()

        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        # 6.1 — Histograms
        df[numeric_cols].hist(figsize=(30, 30), bins=30, edgecolor='black')
        plt.suptitle('Feature distributions', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'feature_distributions_histograms.png'))
        plt.close()

        # 6.2 — Boxplots
        elements = len(numeric_cols)
        elmint_row = 4
        rows = math.ceil(elements / elmint_row)
        fig, axes = plt.subplots(rows, elmint_row, figsize=(20, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, numeric_cols):
            sns.boxplot(y=df[col], ax=ax, color='steelblue')
            ax.set_title(col)
        for ax in axes[elements:]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'feature_distributions_boxplots.png'))
        plt.close()

        # 7.1 — Correlation Heatmap
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(16, 12))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5, square=True
        )
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'correlation_matrix_heatmap.png'))
        plt.close()

        if target in df.columns:
            target_corr = corr[target].drop(target).sort_values()
            colors = ['tomato' if x < 0 else 'steelblue' for x in target_corr]

            n = len(target_corr)
            plt.figure(figsize=(8, n * 0.4))
            target_corr.plot(kind='barh', color=colors)
            plt.axvline(0, color='black', linewidth=0.8)
            plt.title(f'Correlation with {target}')
            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, 'target_correlation_barh.png'))
            plt.close()

            top_n = 6
            top_features = target_corr.abs().sort_values(ascending=False).head(top_n).index.tolist()

            # 8.1 — Top features boxplots
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))
            for ax, feat in zip(axes.flatten(), top_features):
                sns.boxplot(data=df, x=target, y=feat, ax=ax, palette='Set2')
                ax.set_title(f'{feat}\n(r = {target_corr[feat]:.2f})')
            plt.suptitle(f'Top {top_n} features vs {target}')
            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, 'top_features_boxplots.png'))
            plt.close()

            # 8.2 — Top features violin plots
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))
            for ax, feat in zip(axes.flatten(), top_features):
                sns.violinplot(data=df, x=target, y=feat, ax=ax, palette='Set2', inner='quartile')
                ax.set_title(feat)
            plt.suptitle(f'Top {top_n} features — violin plots')
            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, 'top_features_violinplots.png'))
            plt.close()

            # 8.3 — Pairplot
            sample = df.sample(min(2000, len(df)), random_state=42)
            cols_to_plot = top_features[:4] + [target]
            g = sns.pairplot(sample[cols_to_plot], hue=target, plot_kws={'alpha': 0.3}, palette='Set2')
            g.fig.suptitle('Pairplot — top features', y=1.01)
            g.savefig(os.path.join(folder_name, 'top_features_pairplot.png'))
            plt.close('all')