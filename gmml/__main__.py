"""Enables 'python -m gmml ...' (equivalent to the installed 'gmml' command)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
