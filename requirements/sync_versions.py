# SPDX-License-Identifier: BSD-3-Clause

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(" ")
logging.basicConfig(level=logging.INFO)


def main(fname, requirements_folder):
    """
    Get versions from requirements folder and copy them to the conda env file.
    This ensures that the same versions are used across both pip and conda environments.
    """
    logger.info("Syncing versions from %s to %s\n", requirements_folder, fname)

    with open(fname) as f:
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
            clean_package_name = package_name.lstrip(" -")
            pattern = clean_package_name + "=="
            found = False
            for req_file in requirements_folder.glob("*.txt"):
                if "nightly" in req_file.name:
                    continue
                with open(req_file) as req_f:
                    for req_line in req_f:
                        if pattern in req_line:
                            # We need to guard against cases where a package with a long
                            # name contains the pattern of a shorter package. For
                            # example "autodoc-pydantic==2.2.0" contains
                            # "pydantic==2.2.0", which would be a false match.
                            split_req_line = req_line.split("==")
                            if split_req_line[0].strip() != clean_package_name:
                                continue
                            version = split_req_line[1].strip()
                            if version != old_version:
                                logger.info(
                                    "%s: %s --> %s", package_name, old_version, version
                                )
                            f.write(f"  {package_name}=={version}\n")
                            found = True
                            break
                if found:
                    break

            if not found:
                f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize versions from requirements folder to conda env file."
    )
    parser.add_argument("fname", type=str, help="Path to conda env file.")
    parser.add_argument(
        "requirements_folder", type=str, help="Path to requirements folder."
    )
    args = parser.parse_args()

    main(fname=Path(args.fname), requirements_folder=Path(args.requirements_folder))
