import subprocess
import csv
import os
import itertools

values = [0.0, 0.25, 0.5, 0.75, 1.0]
loss_combinations = []
tol = 1e-6
for alpha_ce, lam_ridge, lam_lig in itertools.product(values, repeat=3):
    if abs((alpha_ce + lam_ridge + lam_lig) - 1.0) < tol:
        loss_combinations.append((alpha_ce, lam_ridge, lam_lig))

print("Parameter combinations to test:")
for comb in loss_combinations:
    print(comb)

results_summary = []

for alpha_ce, lam_ridge, lam_lig in loss_combinations:
    print(f"\nRunning ablation for alpha_ce={alpha_ce}, lam_ridge={lam_ridge}, lam_lig={lam_lig}")

    subprocess.run([
        "python", "train.py",
        "--alpha_ce", str(alpha_ce),
        "--lam_ridge", str(lam_ridge),
        "--lam_lig", str(lam_lig),
        "--lr", "0.0005"
    ], check=True)

    eval_output = subprocess.run(
        ["python", "utils/evaluate.py",
                    "--eval_on_val"
        ],
        check=True,
        capture_output=True,
        text=True
    )
    eval_stdout = eval_output.stdout

    # default
    ridge_zero_count = None
    lig_zero_count = None

    for line in eval_stdout.split("\n"):
        line = line.strip()
        if "No predictions for ridge (class 1) in" in line:
            parts = line.split()
            ridge_zero_count = parts[6]  # e.g. "2"
        elif "No predictions for ligament (class 2) in" in line:
            parts = line.split()
            lig_zero_count = parts[6]    # e.g. "5"

    with open("evaluation.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        mean_row = next((row for row in rows if row["patientname"].strip().upper() == "MEAN"), None)
        if mean_row:
            ch_r = float(mean_row["ch_r"])
            ch_l = float(mean_row["ch_l"])
            hd_r = float(mean_row["hd_r"])
            hd_l = float(mean_row["hd_l"])
            nn_r = float(mean_row["nn_r"])
            nn_l = float(mean_row["nn_l"])

            results_summary.append({
                "alpha_ce": alpha_ce,
                "lam_ridge": lam_ridge,
                "lam_lig": lam_lig,
                "ch_r": ch_r,
                "ch_l": ch_l,
                "hd_r": hd_r,
                "hd_l": hd_l,
                "nn_r": nn_r,
                "nn_l": nn_l,
                "ridge_zero_pred": ridge_zero_count,
                "lig_zero_pred": lig_zero_count
            })

summary_file = "ablation_summary.csv"
file_exists = os.path.exists(summary_file)
fieldnames = [
    "alpha_ce", "lam_ridge", "lam_lig",
    "ch_r", "ch_l", "hd_r", "hd_l", "nn_r", "nn_l",
    "ridge_zero_pred", "lig_zero_pred"
]

with open(summary_file, "a", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    for res in results_summary:
        writer.writerow(res)

print("\nAblation Results Summary:")
print("alpha_ce, lam_ridge, lam_lig, ch_r, ch_l, hd_r, hd_l, nn_r, nn_l, ridge_zero_pred, lig_zero_pred")
for res in results_summary:
    print(f"{res['alpha_ce']}, {res['lam_ridge']}, {res['lam_lig']}, "
          f"{res['ch_r']:.4f}, {res['ch_l']:.4f}, {res['hd_r']:.4f}, {res['hd_l']:.4f}, "
          f"{res['nn_r']:.4f}, {res['nn_l']:.4f}, {res['ridge_zero_pred']}, {res['lig_zero_pred']}")

print(f"\nAblation summary appended to {summary_file}")
