#!/usr/bin/env python3
"""
AAOS Artifact Archive Script
DevZen Enhancement #10: Archive continuity with hash manifest

Creates phase4_baseline_artifacts.tar.gz and sha_manifest.txt
"""

import argparse
import hashlib
import os
import sys
import tarfile
from datetime import datetime
from pathlib import Path


def calculate_hash(filepath: str) -> str:
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def archive_artifacts(
    output_file: str = "phase4_baseline_artifacts.tar.gz",
    manifest_file: str = "sha_manifest.txt",
    files: list = None
):
    """Create artifact archive and hash manifest"""

    if files is None:
        files = [
            "aaos_prod.log",
            "telemetry_baseline_24h.log",
            "baseline_summary.json",
            "phase4_validation_log_005.md"
        ]

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"AAOS Artifact Archive")
    print(f"Timestamp: {timestamp}")
    print("-" * 60)

    # Filter to existing files
    existing_files = [f for f in files if os.path.exists(f)]
    missing_files = [f for f in files if not os.path.exists(f)]

    if missing_files:
        print(f"Warning: Missing files: {', '.join(missing_files)}")

    if not existing_files:
        print("Error: No files to archive!")
        return False

    # Create archive
    print(f"\nCreating archive: {output_file}")
    with tarfile.open(output_file, "w:gz") as tar:
        for filepath in existing_files:
            tar.add(filepath)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  Added: {filepath} ({size_kb:.1f} KB)")

    archive_size = os.path.getsize(output_file) / 1024
    print(f"\nArchive created: {archive_size:.1f} KB")

    # Generate hash manifest
    print(f"\nGenerating hash manifest: {manifest_file}")

    archive_hash = calculate_hash(output_file)

    with open(manifest_file, 'w') as f:
        f.write(f"# AAOS Phase 4 Artifact Hash Manifest\n")
        f.write(f"# Generated: {timestamp}\n\n")

        # Archive hash
        f.write(f"{archive_hash}  {output_file}\n")
        print(f"  {output_file}: {archive_hash[:16]}...")

        # Individual file hashes
        for filepath in existing_files:
            file_hash = calculate_hash(filepath)
            f.write(f"{file_hash}  {filepath}\n")
            print(f"  {filepath}: {file_hash[:16]}...")

    print(f"\nManifest written: {manifest_file}")
    print("-" * 60)
    print("Archive complete!")
    print(f"\nVerify with: python scripts/verify_hashes.py --verify")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="AAOS Artifact Archive (DevZen Enhancement #10)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="phase4_baseline_artifacts.tar.gz",
        help="Output archive file. Default: phase4_baseline_artifacts.tar.gz"
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default="sha_manifest.txt",
        help="Hash manifest file. Default: sha_manifest.txt"
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        default=None,
        help="Files to archive (default: standard Phase 4 artifacts)"
    )

    args = parser.parse_args()

    success = archive_artifacts(
        output_file=args.output,
        manifest_file=args.manifest,
        files=args.files
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
