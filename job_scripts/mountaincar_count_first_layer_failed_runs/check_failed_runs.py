import os

def find_failed_configs(config_file, output_file="failed_configs.txt", base_dir="data"):
    failed_missing_folder = []
    failed_missing_metrics = []
    failed_lines = []

    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '--output-folder-name' in line:
                parts = line.strip().split('--output-folder-name ')

                if len(parts) > 1:
                    # Get the folder path exactly as written in configs.txt
                    folder_path = parts[1].split()[0]

                    # Prepend the base directory (e.g. data/mountaincar_count...)
                    full_path = os.path.join(base_dir, folder_path) if base_dir else folder_path

                    # Check if the folder generated
                    if not os.path.isdir(full_path):
                        failed_missing_folder.append(full_path)
                        failed_lines.append(line)  # Save the exact command line
                    else:
                        # Check if metrics.npz generated inside the folder
                        metrics_path = os.path.join(full_path, "metrics.npz")
                        if not os.path.isfile(metrics_path):
                            failed_missing_metrics.append(full_path)
                            failed_lines.append(line)  # Save the exact command line

    # Print results
    print(f"Configs missing their output directory: {len(failed_missing_folder)}")
    for folder in failed_missing_folder:
        print(f"  - {folder}")

    print(f"\nConfigs missing 'metrics.npz' in their directory: {len(failed_missing_metrics)}")
    for folder in failed_missing_metrics:
        print(f"  - {folder}")

    # Write the failed configuration lines to a new file
    if failed_lines:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for failed_line in failed_lines:
                out_f.write(failed_line)
        print(f"\nSaved {len(failed_lines)} failed configs to '{output_file}' so you can re-run them.")
    else:
        print(f"\nAll configs succeeded! '{output_file}' was not created.")

if __name__ == "__main__":
    # Change the config_file to point to your actual configs.txt path
    find_failed_configs(
        config_file="job_scripts/mountaincar_count_first_layer/configs.txt",
        output_file="failed_configs.txt",
        base_dir="data/"
    )
