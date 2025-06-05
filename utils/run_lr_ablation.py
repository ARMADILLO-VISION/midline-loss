import subprocess
import csv
import os

lr_values = [0.00025, 0.0005, 0.00075, 0.001, 0.005, 0.0075, 0.01]
results_summary = []

for lr in lr_values:
    print(f"\nRunning ablation for learning rate={lr}")
    
    subprocess.run([
        "python", "train.py",
        "--alpha_ce", "0.5",
        "--lam_ridge", "0.5",
        "--lam_lig", "0.0",
        "--lr", str(lr)
    ], check=True)
    
    subprocess.run([
        "python", "utils/evaluate.py",
        "--eval_on_val"
    ], check=True)
    
    with open("evaluation.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        mean_row = next((row for row in rows if row["patientname"].strip().upper() == "MEAN"), None)
        if mean_row:
            # convert string values to floats
            ch_r = float(mean_row["ch_r"])
            ch_l = float(mean_row["ch_l"])
            hd_r = float(mean_row["hd_r"])
            hd_l = float(mean_row["hd_l"])
            nn_r = float(mean_row["nn_r"])
            nn_l = float(mean_row["nn_l"])
            
            # store them in a results dictionary
            results_summary.append({
                "lr": lr,
                "ch_r": ch_r,
                "ch_l": ch_l,
                "hd_r": hd_r,
                "hd_l": hd_l,
                "nn_r": nn_r,
                "nn_l": nn_l
            })

summary_file = "lr_ablation_summary.csv"
file_exists = os.path.exists(summary_file)
with open(summary_file, "a", newline='') as f:
    fieldnames = ["lr", "ch_r", "ch_l", "hd_r", "hd_l", "nn_r", "nn_l"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    for res in results_summary:
        writer.writerow(res)

print("\nAblation Results Summary:")
print("lr, ch_r, ch_l, hd_r, hd_l, nn_r, nn_l")
for res in results_summary:
    print(f"{res['lr']}, {res['ch_r']:.4f}, {res['ch_l']:.4f}, "
          f"{res['hd_r']:.4f}, {res['hd_l']:.4f}, {res['nn_r']:.4f}, {res['nn_l']:.4f}")

print(f"\nAblation summary appended to {summary_file}")
