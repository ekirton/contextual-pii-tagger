"""Entry point for `python -m contextual_pii_tagger.hooks.scan`.

Spec: specifications/hook-script.md §1
"""

from __future__ import annotations

import argparse
import sys

from contextual_pii_tagger.hooks.scan import scan


def main() -> None:
    parser = argparse.ArgumentParser(description="PII hook scanner")
    parser.add_argument(
        "--hook",
        required=True,
        choices=["user_prompt", "pre_tool_use", "post_tool_use"],
    )
    args = parser.parse_args()

    exit_code, stderr_content = scan(args.hook, sys.stdin)

    if stderr_content:
        print(stderr_content, file=sys.stderr, end="")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
