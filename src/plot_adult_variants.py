import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from io import StringIO

OUTPUT_DIR = "results/final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_text = """variant_name,label_mode,label_type,feature_mode,test_acc,test_f1,num_train_samples,attack_accuracy,attack_precision,attack_recall,attack_f1,attack_auc
student_raw_soft_original,raw_soft,soft,original,0.8514689323369843,0.6605847953216374,31258,0.5076261643975842,0.5232159551262076,0.17187020165830688,0.25874556942518107,0.5217764729044805
student_raw_soft_anon,raw_soft,soft,anon,0.8448152318558706,0.6223218734429496,31258,0.5119254785546116,0.5068581856713957,0.8813594021906029,0.6435939602332187,0.5128882550830417
student_raw_soft_filtered,raw_soft,soft,filtered,0.8292558091923431,0.6309734513274337,7485,0.48323313293253173,0.49071676899178934,0.8863059452237809,0.631689202056751,0.35082186640035806
student_temp_soft_original,temp_soft,soft,original,0.8538233186610708,0.6567307692307692,31258,0.5170437096939298,0.510419926153076,0.8348858634455932,0.6335249339754544,0.5273798331382251
student_temp_soft_anon,temp_soft,soft,anon,0.8413348346811342,0.6055979643765903,31258,0.5097758214760979,0.5055186362323028,0.8954857201351213,0.6462288542513113,0.5142349487198716
student_temp_soft_filtered,temp_soft,soft,filtered,0.8184051591769884,0.6269974768713205,7485,0.49298597194388777,0.4962794982637659,0.9356045424181697,0.6485460270420448,0.40167131858908195
student_clip_soft_original,clip_soft,soft,original,0.8580202681953117,0.6671466282697385,31258,0.5128467601596888,0.507967748079487,0.8190193469137066,0.627037617554859,0.5179850471786276
student_clip_soft_anon,clip_soft,soft,anon,0.8458388780837343,0.6453132359868111,31258,0.5114136554406797,0.5074477322824127,0.7776640393080152,0.6141471301535975,0.5137011154830857
student_clip_soft_filtered,clip_soft,soft,filtered,0.8200429931415703,0.6207937877480587,7485,0.5480293921175685,0.6257432668765303,0.23901135604542417,0.34590100541376645,0.4992489722263498
student_hard_original,hard,hard,original,0.8557682464940116,0.6639637491056523,31258,0.5136144948305865,0.5082959081836327,0.8341693110860886,0.6316809425991241,0.5175753791171622
student_hard_anon,hard,hard,anon,0.8453270549698024,0.6438840443082724,31258,0.5087521752482342,0.5058477532316531,0.7570887501279557,0.6064780647806478,0.514598443200591
student_hard_filtered,hard,hard,filtered,0.8157436789845429,0.6221662468513854,7485,0.37294589178356713,0.3874822527212494,0.43754175016700064,0.41099328606387653,0.36861705597791
"""

df = pd.read_csv(StringIO(csv_text))

teacher_acc = 0.8395
teacher_auc = 0.5812

label_markers = {
    "raw_soft": "o",
    "temp_soft": "s",
    "clip_soft": "D",
    "hard": "^",
}

feature_colors = {
    "original": "tab:blue",
    "anon": "tab:green",
    "filtered": "tab:red",
}

plt.figure(figsize=(11, 7))
ax = plt.gca()

# Teacher point
ax.scatter(
    teacher_acc,
    teacher_auc,
    color="black",
    marker="X",
    s=180,
    zorder=5,
    label="Adult Teacher"
)

ax.annotate(
    "Adult Teacher",
    (teacher_acc, teacher_auc),
    textcoords="offset points",
    xytext=(8, 8),
    fontsize=9,
    fontweight="bold",
)

# Teacher -> student arrows
for _, row in df.iterrows():
    ax.arrow(
        teacher_acc,
        teacher_auc,
        row["test_acc"] - teacher_acc,
        row["attack_auc"] - teacher_auc,
        color=feature_colors[row["feature_mode"]],
        alpha=0.25,
        linewidth=0.9,
        head_width=0.0025,
        head_length=0.004,
        length_includes_head=True,
        zorder=1,
    )

# Student points
for _, row in df.iterrows():
    ax.scatter(
        row["test_acc"],
        row["attack_auc"],
        color=feature_colors[row["feature_mode"]],
        marker=label_markers[row["label_mode"]],
        s=95,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
        zorder=4,
    )

# Original -> anon / filtered arrows within each label mode
for label_mode in df["label_mode"].unique():
    original = df[(df["label_mode"] == label_mode) & (df["feature_mode"] == "original")]
    anon = df[(df["label_mode"] == label_mode) & (df["feature_mode"] == "anon")]
    filtered = df[(df["label_mode"] == label_mode) & (df["feature_mode"] == "filtered")]

    if len(original) == 1 and len(anon) == 1:
        o = original.iloc[0]
        a = anon.iloc[0]
        ax.annotate(
            "",
            xy=(a["test_acc"], a["attack_auc"]),
            xytext=(o["test_acc"], o["attack_auc"]),
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                linewidth=1.1,
                alpha=0.8,
                linestyle="dotted",
            ),
            zorder=3,
        )

    if len(original) == 1 and len(filtered) == 1:
        o = original.iloc[0]
        f = filtered.iloc[0]
        ax.annotate(
            "",
            xy=(f["test_acc"], f["attack_auc"]),
            xytext=(o["test_acc"], o["attack_auc"]),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                linewidth=1.2,
                alpha=0.85,
                linestyle="dashed",
            ),
            zorder=3,
        )

# Reference line
ax.axhline(0.5, linestyle="--", color="red", linewidth=1.2, label="Random Attack AUC")

# Grid
ax.minorticks_on()
ax.xaxis.set_major_locator(MultipleLocator(0.01))
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.xaxis.set_minor_locator(MultipleLocator(0.005))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.65)
ax.grid(which="minor", linestyle="--", linewidth=0.45, alpha=0.35)

ax.set_title("Adult Student Variants: Accuracy vs Membership Leakage")
ax.set_xlabel("Test Accuracy")
ax.set_ylabel("Attack ROC AUC")

ax.set_xlim(0.81, 0.865)
ax.set_ylim(0.34, 0.60)

# Manual legend
for feature, color in feature_colors.items():
    ax.scatter([], [], color=color, marker="o", s=80, edgecolor="black", label=f"feature: {feature}")

for label, marker in label_markers.items():
    ax.scatter([], [], color="gray", marker=marker, s=80, edgecolor="black", label=f"label: {label}")

ax.plot([], [], color="black", linestyle="dashed", linewidth=1.2, label="Original → Filtered")
ax.plot([], [], color="gray", linestyle="dotted", linewidth=1.1, label="Original → Anon")

ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "adult_variant_detail_plot.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved: {out_path}")