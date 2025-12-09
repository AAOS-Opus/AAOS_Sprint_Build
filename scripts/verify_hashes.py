#!/usr/bin/env python3
"""
AAOS Hash Verification Script
DevZen Enhancement #10: Archive continuity verification

Verifies SHA256 hashes of artifacts against manifest.
"""

import argparse
import hashlib
import sys
from pathlib import Path


def calculate_hash(filepath: str) -> str:
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_manifest(manifest_file: str = "sha_manifest.txt") -> bool:
    """Verify all hashes in manifest"""
    print(f"Verifying hashes from: {manifest_file}")
    print("-" * 60)

    all_valid = True

    try:
        with open(manifest_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse manifest line (format: hash  filename)
                parts = line.split()
                if len(parts) >= 2:
                    expected_hash = parts[0]
                    filepath = parts[1]

                    if not Path(filepath).exists():
                        print(f"MISSING: {filepath}")
                        all_valid = False
                        continue

                    actual_hash = calculate_hash(filepath)

                    if actual_hash == expected_hash:
                        print(f"VALID:   {filepath}")
                    else:
                        print(f"INVALID: {filepath}")
                        print(f"  Expected: {expected_hash}")
                        print(f"  Actual:   {actual_hash}")
                        all_valid = False

    except FileNotFoundError:
        print(f"ERROR: Manifest file not found: {manifest_file}")
        return False

    print("-" * 60)
    if all_valid:
        print("All hashes verified successfully!")
    else:
        print("Hash verification FAILED!")

    return all_valid


def generate_manifest(files: list, manifest_file: str = "sha_manifest.txt"):
    """Generate hash manifest for files"""
    print(f"Generating manifest: {manifest_file}")

    with open(manifest_file, 'w') as f:
        f.write("# AAOS Phase 4 Artifact Hash Manifest\n")
        f.write(f"# Generated: {__import__('datetime').datetime.utcnow().isoformat()}\n\n")

        for filepath in files:
            if Path(filepath).exists():
                file_hash = calculate_hash(filepath)
                f.write(f"{file_hash}  {filepath}\n")
                print(f"Added: {filepath}")
            else:
                print(f"Skipped (not found): {filepath}")

    print(f"\nManifest written to: {manifest_file}")


def main():
    parser = argparse.ArgumentParser(
        description="AAOS Hash Verification (DevZen Enhancement #10)"
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify hashes against manifest"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate hash manifest"
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default="sha_manifest.txt",
        help="Manifest file path. Default: sha_manifest.txt"
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        default=[
            "phase4_baseline_artifacts.tar.gz",
            "baseline_summary.json",
            "aaos_prod.log",
            "telemetry_baseline_24h.log"
        ],
        help="Files to include in manifest (for generate mode)"
    )

    args = parser.parse_args()

    if args.verify:
        success = verify_manifest(args.manifest)
        sys.exit(0 if success else 1)
    elif args.generate:
        generate_manifest(args.files, args.manifest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
