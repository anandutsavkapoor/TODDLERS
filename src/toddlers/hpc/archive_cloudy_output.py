#!/usr/bin/env python3
"""Archive non-essential Cloudy output to relieve inode pressure on large campaigns.

A finished Cloudy model leaves ~100 small files per parameter directory, most of
which (the diagnostic ``.ovr``, ``.heat``, ``.cool``, ``.grTemp`` ... dumps) are not
read by the downstream STAB build. A grid of thousands of models therefore runs into
the cluster's *file-count* (inode) quota long before its byte quota.

This utility walks each parameter directory, packs the non-essential files into a
single per-directory ``output_archive.tar``, and removes the originals. The files the
SED interpolant / STAB stage needs (``.in``, ``.out``, ``.cont``, ``.phy``, ``.rad``)
are kept loose, so archiving is safe to run *before* the build. It is fully reversible
with ``--untar``.

Cloudy run files are only archived once the model is confirmed successful
(:meth:`CloudyOutputHandler.check_cloudy_success`), so a failed model's files stay in
place for inspection / resume.

Run as a standalone pass over a whole ``cloudy_output`` tree::

    python -m toddlers.hpc.archive_cloudy_output /path/to/cloudy_output            # archive
    python -m toddlers.hpc.archive_cloudy_output /path/to/cloudy_output --dry-run  # preview
    python -m toddlers.hpc.archive_cloudy_output /path/to/cloudy_output --untar     # restore

The campaign orchestrator (:mod:`toddlers.hpc.campaign`) invokes this automatically
after the Cloudy grid is complete unless ``--no-archive-cloudy`` is passed.
"""

import os
import sys
import tarfile
import logging
from pathlib import Path
from typing import Set, Tuple, List, Optional

from ..cloudy_output_handler import CloudyOutputHandler
from ..constants import MYR_TO_SEC


def setup_logging():
    """Configure basic logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('archiver')


class CloudyArchiver:
    """Archives non-essential Cloudy output files while preserving essential ones."""

    def __init__(self, param_dir: Path):
        """
        Initialize archiver for a parameter directory.

        Args:
            param_dir: Parameter set directory path
        """
        self.param_dir = Path(param_dir)

        # Files needed for analysis / the downstream SED interpolant + STAB build.
        self.essential_extensions = {'.in', '.out', '.cont', '.phy', '.rad'}

        # One archive per parameter directory.
        self.archive_path = self.param_dir / 'output_archive.tar'

    def _get_archived_files(self) -> Set[str]:
        """Get set of files already in archive."""
        if not self.archive_path.exists():
            return set()

        with tarfile.open(self.archive_path, 'r:') as tar:
            return {member.name for member in tar.getmembers()}

    def _get_cloudy_info(self, filepath: Path) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Extract Cloudy run information from filepath.

        Args:
            filepath: Path to file

        Returns:
            Tuple of:
            - bool: Whether file is from Cloudy run
            - Optional[str]: Model prefix if Cloudy file (shell/unified/dig/dissolved)
            - Optional[float]: Time point if Cloudy file
        """
        # Common Cloudy model prefixes
        CLOUDY_PREFIXES = {'shell', 'unified', 'dig', 'dissolved'}

        try:
            # Parse filename to get prefix and time, e.g. shell_1.23.cont / unified_4.56.out
            parts = filepath.stem.split('_')
            if len(parts) == 2 and parts[0] in CLOUDY_PREFIXES:
                return True, parts[0], float(parts[1])
        except (ValueError, IndexError):
            pass

        return False, None, None

    def _should_archive_file(self, filepath: Path) -> bool:
        """
        Check if file should be archived based on:
        1. Not an essential file extension
        2. Not hidden/system file
        3. Regular file (not directory/symlink)
        4. Not already archived
        5. For Cloudy files: only if the run completed successfully

        Args:
            filepath: Path to file to check

        Returns:
            bool: Whether file should be archived
        """
        # Basic checks first (faster)
        if (filepath.suffix in self.essential_extensions or
                filepath.name.startswith('.') or
                not filepath.is_file() or
                'archive' in filepath.name):
            return False

        # Check if it's a Cloudy run file
        is_cloudy, prefix, time = self._get_cloudy_info(filepath)
        if is_cloudy:
            # CloudyOutputHandler(absolute_path=...) chdir's into the directory; guard the
            # cwd so this pass leaves the process where it started.
            cwd = os.getcwd()
            try:
                handler = CloudyOutputHandler(
                    model_prefix=prefix,
                    time=time * MYR_TO_SEC,  # Convert Myr to seconds
                    absolute_path=filepath.parent,
                    parse_data=False  # Don't need to parse data for the success check
                )
                return handler.check_cloudy_success()
            except Exception as e:
                # If we can't determine success, don't archive
                print(f"Warning: Could not check success for {filepath}: {str(e)}")
                return False
            finally:
                os.chdir(cwd)

        # For non-Cloudy files, use default logic
        return True

    def archive_files(self, dry_run: bool = False) -> Tuple[int, List[Path]]:
        """
        Archive new files in parameter directory and remove originals.

        Args:
            dry_run: If True, only show what would be done

        Returns:
            Tuple of (number of files archived, list of files that failed)
        """
        # Get previously archived files
        archived_files = self._get_archived_files()

        # Find new files to archive
        new_files = []
        for file_path in self.param_dir.rglob('*'):
            if self._should_archive_file(file_path):
                rel_path = str(file_path.relative_to(self.param_dir))
                if rel_path not in archived_files:
                    new_files.append((file_path, rel_path))

        if not new_files:
            return 0, []

        if dry_run:
            print(f"\nWould archive {len(new_files)} files in {self.param_dir.name}:")
            for file_path, _ in new_files:
                print(f"  {file_path.relative_to(self.param_dir)}")
            return len(new_files), []

        # Open archive in append mode if exists, else create new
        mode = 'a:' if self.archive_path.exists() else 'w:'
        with tarfile.open(self.archive_path, mode) as tar:
            for file_path, rel_path in new_files:
                tar.add(file_path, arcname=rel_path)

        # Remove files
        for file_path, _ in new_files:
            file_path.unlink()  # Remove original

        return len(new_files), []

    def untar(self):
        """Extract all files from the archive to their original locations and clean up."""
        if not self.archive_path.exists():
            print(f"No archive found at {self.archive_path}")
            return

        with tarfile.open(self.archive_path, 'r:') as tar:
            # Use None for the filter as all files are to be extracted
            tar.extractall(path=self.param_dir)
            print(f"Extracted all files to {self.param_dir}")

        # Clean up archive after extraction
        self.archive_path.unlink()
        print(f"Cleaned up archive at {self.archive_path}")


def find_param_dirs(cloudy_output: Path) -> list:
    """Find parameter set directories containing Cloudy outputs."""
    param_dirs = []
    for path in cloudy_output.rglob("*"):
        if (path.is_dir() and
                any(p.name.startswith(('Z', 'z')) and 'eta' in p.name for p in [path]) and
                any(f.suffix in ['.in', '.out'] for f in path.iterdir())):
            param_dirs.append(path)
    return param_dirs


def main(argv=None):
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Archive and remove non-essential Cloudy output files"
    )
    parser.add_argument("cloudy_output", help="Path to cloudy_output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")
    parser.add_argument("--untar", action="store_true",
                        help="Extract files from archive back to their original locations")

    args = parser.parse_args(argv)

    logger = setup_logging()
    cloudy_output = Path(args.cloudy_output)

    if not cloudy_output.exists():
        logger.error(f"Directory not found: {cloudy_output}")
        sys.exit(1)

    # Find all parameter directories
    param_dirs = find_param_dirs(cloudy_output)
    if not param_dirs:
        logger.warning("No parameter directories found!")
        sys.exit(0)

    logger.info(f"Found {len(param_dirs)} parameter directories")
    if args.dry_run:
        logger.info("DRY RUN - no files will be modified")

    # Process each directory
    total_archived = 0
    for param_dir in sorted(param_dirs):
        try:
            archiver = CloudyArchiver(param_dir)
            if args.untar:
                archiver.untar()
            else:
                n_files, _ = archiver.archive_files(dry_run=args.dry_run)
                if n_files:
                    logger.info(f"Archived {n_files} files in {param_dir.name}")
                    total_archived += n_files
        except Exception as e:
            logger.error(f"Error processing {param_dir.name}: {str(e)}")
            continue

    if not args.untar:
        logger.info(f"Archived {total_archived} files total")


if __name__ == "__main__":
    main()
