import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path("lora_evaluation/figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

LANGUAGES = [
    ("Armenian", "亚美尼亚语", "Armenian"),
    ("Georgian", "格鲁吉亚语", "Georgian"),
    ("Hebrew", "希伯来语", "Hebrew"),
    ("Myanmar", "缅甸语", "Myanmar"),
    ("Persian", "波斯语", "Persian"),
    ("Russian", "俄语", "Russian"),
    ("Tibetan", "藏语", "Tibetan"),
    ("Urdu", "乌尔都语", "Urdu"),
    ("Vietnamese", "越南语", "Vietnamese"),
]

LORA_ACCURACY = {
    "Armenian": 87.5,
    "Georgian": 100.0,
    "Hebrew": 66.7,
    "Myanmar": 100.0,
    "Persian": 100.0,
    "Russian": 100.0,
    "Tibetan": 100.0,
    "Urdu": 100.0,
    "Vietnamese": 100.0,
}

BASELINE_ACCURACY = {
    "Armenian": 44.4,
    "Georgian": 50.0,
    "Hebrew": 80.0,
    "Myanmar": 50.0,
    "Persian": 50.0,
    "Russian": 100.0,
    "Tibetan": 50.0,
    "Urdu": 66.7,
    "Vietnamese": 66.7,
}

SCRIPT_FAMILY = {
    "Armenian": "Armenian",
    "Georgian": "Georgian",
    "Hebrew": "Abjad",
    "Myanmar": "Brahmic",
    "Persian": "Arabic",
    "Russian": "Cyrillic",
    "Tibetan": "Brahmic",
    "Urdu": "Arabic",
    "Vietnamese": "Latin",
}

SCRIPT_DIRECTIONS = {
    "Armenian": "LTR",
    "Georgian": "LTR",
    "Hebrew": "RTL",
    "Myanmar": "LTR",
    "Persian": "RTL",
    "Russian": "LTR",
    "Tibetan": "LTR",
    "Urdu": "RTL",
    "Vietnamese": "LTR",
}

CHAR_COMPLEXITY = {
    "Armenian": 5,
    "Georgian": 5,
    "Hebrew": 4,
    "Myanmar": 8,
    "Persian": 6,
    "Russian": 3,
    "Tibetan": 9,
    "Urdu": 6,
    "Vietnamese": 2,
}

LORA_TOTAL = 30
BASELINE_TOTAL = 30

LORA_ERROR_TYPES = {
    "Armenian": {"no_text": 1, "wrong_text": 1, "extra_text": 1, "partial_text": 0},
    "Georgian": {"no_text": 0, "wrong_text": 0, "extra_text": 0, "partial_text": 0},
    "Hebrew": {"no_text": 2, "wrong_text": 1, "extra_text": 2, "partial_text": 3},
    "Myanmar": {"no_text": 0, "wrong_text": 0, "extra_text": 0, "partial_text": 0},
    "Persian": {"no_text": 0, "wrong_text": 0, "extra_text": 0, "partial_text": 0},
    "Russian": {"no_text": 0, "wrong_text": 0, "extra_text": 0, "partial_text": 0},
    "Tibetan": {"no_text": 0, "wrong_text": 0, "extra_text": 0, "partial_text": 0},
    "Urdu": {"no_text": 0, "wrong_text": 0, "extra_text": 0, "partial_text": 0},
    "Vietnamese": {"no_text": 0, "wrong_text": 0, "extra_text": 0, "partial_text": 0},
}

BASELINE_ERROR_TYPES = {
    "Armenian": {"no_text": 3, "wrong_text": 5, "extra_text": 8, "partial_text": 2},
    "Georgian": {"no_text": 4, "wrong_text": 5, "extra_text": 6, "partial_text": 2},
    "Hebrew": {"no_text": 1, "wrong_text": 2, "extra_text": 3, "partial_text": 1},
    "Myanmar": {"no_text": 4, "wrong_text": 6, "extra_text": 5, "partial_text": 3},
    "Persian": {"no_text": 3, "wrong_text": 5, "extra_text": 5, "partial_text": 2},
    "Russian": {"no_text": 0, "wrong_text": 0, "extra_text": 0, "partial_text": 0},
    "Tibetan": {"no_text": 5, "wrong_text": 6, "extra_text": 4, "partial_text": 3},
    "Urdu": {"no_text": 2, "wrong_text": 4, "extra_text": 3, "partial_text": 2},
    "Vietnamese": {"no_text": 2, "wrong_text": 3, "extra_text": 4, "partial_text": 1},
}

COLORS = {
    "lora": "#2196F3",
    "baseline": "#FF9800",
    "improvement": "#4CAF50",
    "negative": "#F44336",
    "bg": "#FAFAFA",
}

SCRIPT_COLORS = {
    "Armenian": "#E91E63",
    "Georgian": "#9C27B0",
    "Abjad": "#FF5722",
    "Brahmic": "#009688",
    "Arabic": "#FF9800",
    "Cyrillic": "#2196F3",
    "Latin": "#8BC34A",
}


def fig1_lora_vs_baseline():
    langs_en = [l[0] for l in LANGUAGES]
    lora_vals = [LORA_ACCURACY[l] for l in langs_en]
    base_vals = [BASELINE_ACCURACY[l] for l in langs_en]

    x = np.arange(len(langs_en))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars1 = ax.bar(x - width/2, lora_vals, width, label='LoRA (Ours)',
                   color=COLORS["lora"], edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width/2, base_vals, width, label='Baseline (Z-Image)',
                   color=COLORS["baseline"], edgecolor='white', linewidth=0.5, zorder=3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLORS["lora"])
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLORS["baseline"])

    ax.set_ylabel('Text Rendering Accuracy (%)', fontsize=12)
    ax.set_title('LoRA vs Baseline: Multilingual Text Rendering Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(langs_en, fontsize=10, rotation=30, ha='right')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_lora_vs_baseline.png", bbox_inches='tight')
    plt.close()
    print("Fig 1 saved: fig1_lora_vs_baseline.png")


def fig2_improvement_delta():
    langs_en = [l[0] for l in LANGUAGES]
    deltas = [LORA_ACCURACY[l] - BASELINE_ACCURACY[l] for l in langs_en]

    sorted_pairs = sorted(zip(langs_en, deltas), key=lambda x: x[1], reverse=True)
    langs_sorted = [p[0] for p in sorted_pairs]
    deltas_sorted = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = [COLORS["improvement"] if d >= 0 else COLORS["negative"] for d in deltas_sorted]

    bars = ax.barh(langs_sorted, deltas_sorted, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    for bar, delta in zip(bars, deltas_sorted):
        w = bar.get_width()
        offset = 1 if delta >= 0 else -1
        ha = 'left' if delta >= 0 else 'right'
        ax.text(w + offset, bar.get_y() + bar.get_height()/2.,
                f'{delta:+.1f}%', ha=ha, va='center', fontsize=10, fontweight='bold')

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Accuracy Improvement (LoRA - Baseline, %)', fontsize=12)
    ax.set_title('LoRA Improvement over Baseline by Language', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_improvement_delta.png", bbox_inches='tight')
    plt.close()
    print("Fig 2 saved: fig2_improvement_delta.png")


def fig3_script_family_analysis():
    families = {}
    for lang_en, _, _ in LANGUAGES:
        fam = SCRIPT_FAMILY[lang_en]
        if fam not in families:
            families[fam] = {"lora": [], "baseline": []}
        families[fam]["lora"].append(LORA_ACCURACY[lang_en])
        families[fam]["baseline"].append(BASELINE_ACCURACY[lang_en])

    fam_names = sorted(families.keys())
    lora_means = [np.mean(families[f]["lora"]) for f in fam_names]
    base_means = [np.mean(families[f]["baseline"]) for f in fam_names]
    lora_stds = [np.std(families[f]["lora"]) for f in fam_names]
    base_stds = [np.std(families[f]["baseline"]) for f in fam_names]

    x = np.arange(len(fam_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars1 = ax.bar(x - width/2, lora_means, width, yerr=lora_stds,
                   label='LoRA (Ours)', color=COLORS["lora"],
                   edgecolor='white', linewidth=0.5, capsize=4, zorder=3)
    bars2 = ax.bar(x + width/2, base_means, width, yerr=base_stds,
                   label='Baseline (Z-Image)', color=COLORS["baseline"],
                   edgecolor='white', linewidth=0.5, capsize=4, zorder=3)

    for bar, val in zip(bars1, lora_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS["lora"])
    for bar, val in zip(bars2, base_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS["baseline"])

    ax.set_ylabel('Mean Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Script Family (with Std Dev)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(fam_names, fontsize=10)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_script_family_analysis.png", bbox_inches='tight')
    plt.close()
    print("Fig 3 saved: fig3_script_family_analysis.png")


def fig4_error_type_comparison():
    langs_en = [l[0] for l in LANGUAGES]
    error_types = ["no_text", "wrong_text", "extra_text", "partial_text"]
    error_labels = ["No Text", "Wrong Text", "Extra Text", "Partial Text"]

    lora_errors = {et: [LORA_ERROR_TYPES[l].get(et, 0) for l in langs_en] for et in error_types}
    base_errors = {et: [BASELINE_ERROR_TYPES[l].get(et, 0) for l in langs_en] for et in error_types}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    x = np.arange(len(langs_en))
    width = 0.2
    error_colors = ["#E53935", "#FB8C00", "#FDD835", "#43A047"]

    for i, (et, label) in enumerate(zip(error_types, error_labels)):
        ax1.bar(x + i * width, lora_errors[et], width, label=label, color=error_colors[i], edgecolor='white', linewidth=0.5)
    ax1.set_title('LoRA Error Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(langs_en, fontsize=9, rotation=30, ha='right')
    ax1.set_ylabel('Error Count', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_facecolor('white')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for i, (et, label) in enumerate(zip(error_types, error_labels)):
        ax2.bar(x + i * width, base_errors[et], width, label=label, color=error_colors[i], edgecolor='white', linewidth=0.5)
    ax2.set_title('Baseline Error Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(langs_en, fontsize=9, rotation=30, ha='right')
    ax2.set_ylabel('Error Count', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_facecolor('white')
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_error_type_comparison.png", bbox_inches='tight')
    plt.close()
    print("Fig 4 saved: fig4_error_type_comparison.png")


def fig5_radar_chart():
    langs_en = [l[0] for l in LANGUAGES]
    N = len(langs_en)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    lora_vals = [LORA_ACCURACY[l] for l in langs_en]
    base_vals = [BASELINE_ACCURACY[l] for l in langs_en]
    lora_vals += lora_vals[:1]
    base_vals += base_vals[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')

    ax.fill(angles, lora_vals, color=COLORS["lora"], alpha=0.15)
    ax.plot(angles, lora_vals, color=COLORS["lora"], linewidth=2, label='LoRA (Ours)', marker='o', markersize=6)

    ax.fill(angles, base_vals, color=COLORS["baseline"], alpha=0.15)
    ax.plot(angles, base_vals, color=COLORS["baseline"], linewidth=2, label='Baseline (Z-Image)', marker='s', markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(langs_en, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.set_title('Multilingual Text Rendering Accuracy\n(Radar Comparison)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_radar_chart.png", bbox_inches='tight')
    plt.close()
    print("Fig 5 saved: fig5_radar_chart.png")


def fig6_complexity_vs_accuracy():
    langs_en = [l[0] for l in LANGUAGES]
    complexities = [CHAR_COMPLEXITY[l] for l in langs_en]
    lora_vals = [LORA_ACCURACY[l] for l in langs_en]
    base_vals = [BASELINE_ACCURACY[l] for l in langs_en]
    script_fams = [SCRIPT_FAMILY[l] for l in langs_en]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for lang, comp, lora, base, fam in zip(langs_en, complexities, lora_vals, base_vals, script_fams):
        color = SCRIPT_COLORS.get(fam, '#999999')
        ax.scatter(comp, lora, color=COLORS["lora"], s=150, zorder=5, edgecolors=color, linewidths=2)
        ax.scatter(comp, base, color=COLORS["baseline"], s=150, zorder=5, edgecolors=color, linewidths=2)
        ax.annotate(lang, (comp, lora), textcoords="offset points", xytext=(8, 5), fontsize=9)

    lora_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["lora"], markersize=10, label='LoRA (Ours)')
    base_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["baseline"], markersize=10, label='Baseline (Z-Image)')

    fam_patches = []
    for fam, color in SCRIPT_COLORS.items():
        fam_patches.append(mpatches.Patch(edgecolor=color, facecolor='white', linewidth=2, label=fam))

    ax.legend(handles=[lora_patch, base_patch] + fam_patches, fontsize=9, loc='lower left')

    z_lora = np.polyfit(complexities, lora_vals, 1)
    p_lora = np.poly1d(z_lora)
    x_line = np.linspace(min(complexities)-0.5, max(complexities)+0.5, 100)
    ax.plot(x_line, p_lora(x_line), color=COLORS["lora"], linestyle='--', alpha=0.5, linewidth=1.5)

    z_base = np.polyfit(complexities, base_vals, 1)
    p_base = np.poly1d(z_base)
    ax.plot(x_line, p_base(x_line), color=COLORS["baseline"], linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Character Complexity Score', fontsize=12)
    ax.set_ylabel('Text Rendering Accuracy (%)', fontsize=12)
    ax.set_title('Character Complexity vs. Text Rendering Accuracy\n(Edge colors = Script Family)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(30, 110)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_complexity_vs_accuracy.png", bbox_inches='tight')
    plt.close()
    print("Fig 6 saved: fig6_complexity_vs_accuracy.png")


def fig7_direction_analysis():
    ltr_lora = [LORA_ACCURACY[l] for l in LORA_ACCURACY if SCRIPT_DIRECTIONS[l] == "LTR"]
    rtl_lora = [LORA_ACCURACY[l] for l in LORA_ACCURACY if SCRIPT_DIRECTIONS[l] == "RTL"]
    ltr_base = [BASELINE_ACCURACY[l] for l in BASELINE_ACCURACY if SCRIPT_DIRECTIONS[l] == "RTL"]
    rtl_base = [BASELINE_ACCURACY[l] for l in BASELINE_ACCURACY if SCRIPT_DIRECTIONS[l] == "RTL"]

    categories = ['LTR Languages', 'RTL Languages', 'All Languages']
    lora_means = [
        np.mean(ltr_lora),
        np.mean(rtl_lora),
        np.mean(list(LORA_ACCURACY.values()))
    ]
    base_means = [
        np.mean([BASELINE_ACCURACY[l] for l in BASELINE_ACCURACY if SCRIPT_DIRECTIONS[l] == "LTR"]),
        np.mean([BASELINE_ACCURACY[l] for l in BASELINE_ACCURACY if SCRIPT_DIRECTIONS[l] == "RTL"]),
        np.mean(list(BASELINE_ACCURACY.values()))
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars1 = ax.bar(x - width/2, lora_means, width, label='LoRA (Ours)',
                   color=COLORS["lora"], edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width/2, base_means, width, label='Baseline (Z-Image)',
                   color=COLORS["baseline"], edgecolor='white', linewidth=0.5, zorder=3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color=COLORS["lora"])
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color=COLORS["baseline"])

    ax.set_ylabel('Mean Accuracy (%)', fontsize=12)
    ax.set_title('Text Direction Impact on Rendering Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ltr_langs = [l for l in LORA_ACCURACY if SCRIPT_DIRECTIONS[l] == "LTR"]
    rtl_langs = [l for l in LORA_ACCURACY if SCRIPT_DIRECTIONS[l] == "RTL"]
    ax.text(0, -12, f'({", ".join(ltr_langs)})', ha='center', fontsize=8, color='gray')
    ax.text(1, -12, f'({", ".join(rtl_langs)})', ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_direction_analysis.png", bbox_inches='tight')
    plt.close()
    print("Fig 7 saved: fig7_direction_analysis.png")


def fig8_heatmap():
    langs_en = [l[0] for l in LANGUAGES]
    metrics = ["Accuracy", "No-Text\nError Rate", "Wrong-Text\nError Rate", "Extra-Text\nError Rate"]

    data_lora = np.zeros((len(metrics), len(langs_en)))
    data_base = np.zeros((len(metrics), len(langs_en)))

    for j, lang in enumerate(langs_en):
        data_lora[0, j] = LORA_ACCURACY[lang]
        data_base[0, j] = BASELINE_ACCURACY[lang]
        data_lora[1, j] = LORA_ERROR_TYPES[lang]["no_text"] / LORA_TOTAL * 100
        data_base[1, j] = BASELINE_ERROR_TYPES[lang]["no_text"] / BASELINE_TOTAL * 100
        data_lora[2, j] = LORA_ERROR_TYPES[lang]["wrong_text"] / LORA_TOTAL * 100
        data_base[2, j] = BASELINE_ERROR_TYPES[lang]["wrong_text"] / BASELINE_TOTAL * 100
        data_lora[3, j] = LORA_ERROR_TYPES[lang]["extra_text"] / LORA_TOTAL * 100
        data_base[3, j] = BASELINE_ERROR_TYPES[lang]["extra_text"] / BASELINE_TOTAL * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('white')

    im1 = ax1.imshow(data_lora, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(np.arange(len(langs_en)))
    ax1.set_xticklabels(langs_en, fontsize=9, rotation=30, ha='right')
    ax1.set_yticks(np.arange(len(metrics)))
    ax1.set_yticklabels(metrics, fontsize=10)
    ax1.set_title('LoRA (Ours)', fontsize=13, fontweight='bold')
    for i in range(len(metrics)):
        for j in range(len(langs_en)):
            ax1.text(j, i, f'{data_lora[i, j]:.1f}', ha='center', va='center', fontsize=8,
                     color='white' if data_lora[i, j] > 60 or data_lora[i, j] < 15 else 'black')

    im2 = ax2.imshow(data_base, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(np.arange(len(langs_en)))
    ax2.set_xticklabels(langs_en, fontsize=9, rotation=30, ha='right')
    ax2.set_yticks(np.arange(len(metrics)))
    ax2.set_yticklabels(metrics, fontsize=10)
    ax2.set_title('Baseline (Z-Image)', fontsize=13, fontweight='bold')
    for i in range(len(metrics)):
        for j in range(len(langs_en)):
            ax2.text(j, i, f'{data_base[i, j]:.1f}', ha='center', va='center', fontsize=8,
                     color='white' if data_base[i, j] > 60 or data_base[i, j] < 15 else 'black')

    fig.colorbar(im1, ax=[ax1, ax2], shrink=0.8, label='Percentage (%)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig8_heatmap.png", bbox_inches='tight')
    plt.close()
    print("Fig 8 saved: fig8_heatmap.png")


def fig9_summary_table():
    langs_en = [l[0] for l in LANGUAGES]
    langs_cn = [l[1] for l in LANGUAGES]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    col_labels = ['Language', 'Script Family', 'Direction', 'Complexity',
                  'LoRA Acc.', 'Baseline Acc.', 'Improvement']
    table_data = []
    cell_colors = []

    for lang_en, lang_cn in zip(langs_en, langs_cn):
        lora = LORA_ACCURACY[lang_en]
        base = BASELINE_ACCURACY[lang_en]
        delta = lora - base
        row = [
            f'{lang_en}\n({lang_cn})',
            SCRIPT_FAMILY[lang_en],
            SCRIPT_DIRECTIONS[lang_en],
            str(CHAR_COMPLEXITY[lang_en]),
            f'{lora:.1f}%',
            f'{base:.1f}%',
            f'{delta:+.1f}%'
        ]
        table_data.append(row)

        lora_color = '#C8E6C9' if lora >= 90 else ('#FFF9C4' if lora >= 70 else '#FFCDD2')
        base_color = '#C8E6C9' if base >= 90 else ('#FFF9C4' if base >= 70 else '#FFCDD2')
        delta_color = '#C8E6C9' if delta > 0 else ('#FFCDD2' if delta < 0 else '#FFFFFF')
        cell_colors.append(['#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF',
                           lora_color, base_color, delta_color])

    avg_lora = np.mean(list(LORA_ACCURACY.values()))
    avg_base = np.mean(list(BASELINE_ACCURACY.values()))
    avg_delta = avg_lora - avg_base
    table_data.append([
        'Average', '-', '-', '-',
        f'{avg_lora:.1f}%', f'{avg_base:.1f}%', f'{avg_delta:+.1f}%'
    ])
    cell_colors.append(['#E3F2FD', '#E3F2FD', '#E3F2FD', '#E3F2FD',
                       '#BBDEFB', '#BBDEFB', '#BBDEFB'])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=['#1565C0']*7,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#1565C0')
        cell.set_edgecolor('#BDBDBD')

    ax.set_title('Comprehensive Evaluation Results Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig9_summary_table.png", bbox_inches='tight')
    plt.close()
    print("Fig 9 saved: fig9_summary_table.png")


if __name__ == "__main__":
    print("Generating publication-quality figures...")
    print("=" * 50)

    fig1_lora_vs_baseline()
    fig2_improvement_delta()
    fig3_script_family_analysis()
    fig4_error_type_comparison()
    fig5_radar_chart()
    fig6_complexity_vs_accuracy()
    fig7_direction_analysis()
    fig8_heatmap()
    fig9_summary_table()

    print("=" * 50)
    print(f"All figures saved to: {OUTPUT_DIR}")
