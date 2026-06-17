# SPDX-License-Identifier: BSD-3-Clause

import argparse
from pathlib import Path


def main(fname, requirements_folder):
    """
    Get versions from requirements folder and copy them to the conda env file.
    This ensures that the same versions are used across both pip and conda environments.
    """
    print(f"Syncing versions from {requirements_folder} to {fname}\n")

    with open(fname, "r") as f:
        lines = f.readlines()

    with open(fname, "w") as f:
        for line in lines:
            if "==" not in line:
                f.write(line)
                continue

            # Grep in requirements folder for package name and copy version over
            parts = line.split("==")
            package_name = parts[0].strip()
            old_version = parts[1].strip()
            pattern = package_name.lstrip(" -") + "=="
            found = False
            for req_file in requirements_folder.glob("*.txt"):
                if "nightly" in req_file.name:
                    continue
                with open(req_file, "r") as req_f:
                    for req_line in req_f:
                        if pattern in req_line:
                            version = req_line.split("==")[1].strip()
                            if version != old_version:
                                print(f"{package_name}: {old_version} --> {version}")
                            f.write(f"  {package_name}=={version}\n")
                            found = True
                            break
                if found:
                    break

            if not found:
                f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get versions from requirements folder and copy them to the conda env file."
    )
    parser.add_argument("fname", type=str, help="Path to conda env file.")
    parser.add_argument(
        "requirements_folder", type=str, help="Path to requirements folder."
    )
    args = parser.parse_args()

    main(fname=Path(args.fname), requirements_folder=Path(args.requirements_folder))
