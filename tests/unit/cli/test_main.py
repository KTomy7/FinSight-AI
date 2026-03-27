import pytest

from finsight.cli.main import _build_parser


def test_train_parser_accepts_train_without_tickers_arg() -> None:
    parser = _build_parser()

    args = parser.parse_args(["train", "--cutoff", "2025-06-01"])

    assert args.command == "train"
    assert args.cutoff == "2025-06-01"


def test_train_parser_rejects_tickers_arg() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--cutoff", "2025-06-01", "--tickers", "AAPL", "MSFT"])


def test_train_parser_rejects_interval_arg() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--cutoff", "2025-06-01", "--interval", "1d"])


