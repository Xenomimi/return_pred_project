from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def ensure_output_dir(path: str = "outputs/eda") -> Path:
    """Tworzy katalog na wykresy jeśli nie istnieje."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def basic_info(df: pd.DataFrame) -> None:
    """Podstawowe info o zbiorze."""
    print("Wymiary:", df.shape)
    print("\nTypy kolumn:")
    print(df.dtypes)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Statystyki opisowe dla kolumn numerycznych."""
    return df.describe().T


def target_distribution(df: pd.DataFrame, target_col: str = "Returned", save_path: str | None = None) -> None:
    """Rozkład klasy docelowej + wykres słupkowy."""
    counts = df[target_col].value_counts().sort_index()
    perc = (counts / len(df) * 100).round(2)

    print("Rozkład targetu:")
    print(pd.DataFrame({"count": counts, "percent": perc}))

    ax = counts.plot(kind="bar")
    ax.set_title(f"Rozkład klasy {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Liczba transakcji")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def histograms(df: pd.DataFrame, cols: list[str], bins: int = 30, save_dir: str | None = None) -> None:
    """Histogramy dla wybranych kolumn."""
    out = ensure_output_dir(save_dir) if save_dir else None

    for col in cols:
        if col not in df.columns:
            continue

        plt.figure()
        df[col].hist(bins=bins)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Liczność")
        plt.tight_layout()

        if out:
            plt.savefig(out / f"hist_{col}.png", dpi=150)
        plt.show()


def boxplots_by_target(df: pd.DataFrame, cols: list[str], target_col: str = "Returned", save_dir: str | None = None) -> None:
    """Boxploty cech w podziale na Returned (0 vs 1)."""
    out = ensure_output_dir(save_dir) if save_dir else None

    for col in cols:
        if col not in df.columns:
            continue

        data0 = df[df[target_col] == 0][col]
        data1 = df[df[target_col] == 1][col]

        plt.figure()
        plt.boxplot([data0, data1], labels=["0", "1"], showfliers=False)
        plt.title(f"Boxplot: {col} vs {target_col}")
        plt.xlabel(target_col)
        plt.ylabel(col)
        plt.tight_layout()

        if out:
            plt.savefig(out / f"box_{col}_by_{target_col}.png", dpi=150)
        plt.show()


def scatter_by_target(df: pd.DataFrame, x: str, y: str, target_col: str = "Returned", save_path: str | None = None) -> None:
    """Scatter x vs y z rozróżnieniem klas."""
    if x not in df.columns or y not in df.columns:
        return

    d0 = df[df[target_col] == 0]
    d1 = df[df[target_col] == 1]

    plt.figure()
    plt.scatter(d0[x], d0[y], alpha=0.25, label=f"{target_col}=0")
    plt.scatter(d1[x], d1[y], alpha=0.25, label=f"{target_col}=1")
    plt.title(f"Scatter: {x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def correlation_with_target(df: pd.DataFrame, target_col: str = "Returned", top_n: int = 10) -> pd.Series:
    """
    Liczy korelacje (Pearson) cech numerycznych z targetem.
    Zwraca posortowaną serię po wartości bezwzględnej.
    """
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(numeric_only=True)

    if target_col not in corr.columns:
        print("Brak targetu w macierzy korelacji.")
        return pd.Series(dtype=float)

    corr_to_target = corr[target_col].drop(target_col).sort_values(key=lambda s: s.abs(), ascending=False)
    print(f"Top {top_n} korelacji z {target_col} (po |corr|):")
    print(corr_to_target.head(top_n))
    return corr_to_target


def correlation_heatmap(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Heatmapa korelacji dla kolumn numerycznych."""
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.title("Heatmapa korelacji (cechy numeryczne)")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
