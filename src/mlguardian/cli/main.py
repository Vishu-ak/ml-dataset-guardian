"""CLI entrypoint for mlguardian."""

from __future__ import annotations

import argparse
import logging
import sys

from mlguardian import __version__
from mlguardian.auditor import DatasetAuditor, save_output, should_fail
from mlguardian.config import load_config
from mlguardian.exceptions import (
    AuditError,
    ConfigurationError,
    DatasetLoadError,
    SchemaMismatchError,
)
from mlguardian.reporting import render_json, render_terminal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlguardian")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit train/test datasets")
    audit.add_argument("--train", required=True)
    audit.add_argument("--test")
    audit.add_argument("--target", required=True)
    audit.add_argument("--format", choices=["terminal", "json"], default="terminal")
    audit.add_argument("--output")
    audit.add_argument("--config")
    audit.add_argument("--fail-on", choices=["info", "warning", "high", "critical"])
    audit.add_argument("--key")
    audit.add_argument("--verbose", action="store_true")

    subparsers.add_parser("version", help="Show version")
    return parser


def app() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "version":
        print(__version__)
        raise SystemExit(0)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    try:
        overrides = {}
        if args.fail_on:
            overrides["fail_on"] = args.fail_on
        if args.key:
            overrides["key"] = args.key
        config = load_config(args.config, overrides=overrides or None)
        auditor = DatasetAuditor(target=args.target, config=config)
        report = auditor.audit(train=args.train, test=args.test)
        rendered = render_json(report) if args.format == "json" else render_terminal(report)
        save_output(rendered, args.output)
        if args.output is None:
            print(rendered)

        code = 1 if should_fail(report, config.fail_on) else 0
        raise SystemExit(code)
    except (ConfigurationError, DatasetLoadError, SchemaMismatchError) as exc:
        print(f"Input/configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except AuditError as exc:
        print(f"Internal error: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
