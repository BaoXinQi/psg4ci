#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_CHANNEL_TABLE = Path(__file__).resolve().with_name("channel_table.csv")

# SleepFM-aligned modality caps.
MAX_CHANNELS_BY_MODALITY: Dict[str, int] = {
    "BAS": 10,
    "RESP": 7,
    "EKG": 2,
    "EMG": 4,
}

MODALITY_ORDER = ["BAS", "RESP", "EKG", "EMG"]

# Canonical within-modality priority used for stable ordering and cap truncation.
CANONICAL_PRIORITY_BY_MODALITY: Dict[str, List[str]] = {
    "BAS": [
        "F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1",
        "LOC", "ROC", "E1", "E2", "F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2",
    ],
    "RESP": [
        "AIRFLOW", "NASAL_PRESSURE", "CHEST", "ABDOMINAL", "SaO2", "PULSE", "SNORE", "CFLOW", "CPAP_PRESSURE",
    ],
    "EKG": ["EKG"],
    "EMG": ["CHIN", "LAT", "RAT", "LLEG+", "LLEG-", "RLEG+", "RLEG-", "ARM_EMG"],
}


@dataclass(frozen=True)
class ChannelInfo:
    raw_name: str
    normalized_name: str
    canonical_name: Optional[str]
    family: Optional[str]
    modality: Optional[str]
    priority: int
    matched: bool
    source: str


@dataclass(frozen=True)
class GroupedChannel:
    raw_name: str
    normalized_name: str
    canonical_name: str
    family: str
    modality: str
    priority: int
    source: str
    original_index: int


class ChannelMapper:
    def __init__(self, table_path: Path | str = DEFAULT_CHANNEL_TABLE):
        self.table_path = Path(table_path)
        if not self.table_path.exists():
            raise FileNotFoundError(f"Channel table not found: {self.table_path}")

        self.lookup: Dict[str, Dict[str, object]] = {}
        self._load_table()

        self.canonical_priority: Dict[str, Dict[str, int]] = {}
        for modality, names in CANONICAL_PRIORITY_BY_MODALITY.items():
            self.canonical_priority[modality] = {name: i for i, name in enumerate(names)}

    def _load_table(self) -> None:
        with self.table_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {
                "lookup_key", "canonical_name", "family", "modality",
                "priority", "source", "is_active",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Channel table missing required columns: {sorted(missing)}")

            for row in reader:
                is_active = str(row.get("is_active", "1")).strip().lower() in {"1", "true", "yes"}
                if not is_active:
                    continue

                key = str(row["lookup_key"]).strip()
                if not key:
                    continue

                self.lookup[key] = {
                    "canonical_name": str(row["canonical_name"]).strip(),
                    "family": str(row["family"]).strip(),
                    "modality": str(row["modality"]).strip().upper(),
                    "priority": int(float(row["priority"])),
                    "source": str(row["source"]).strip(),
                }

    @staticmethod
    def normalize_channel_name(name: str) -> str:
        value = str(name).strip().lower()
        value = value.replace("_", " ")
        value = value.replace("/", " ")
        value = value.replace(":", "-")
        value = value.replace("+", " plus ")
        value = value.replace("&", " and ")
        value = re.sub(r"\s+", " ", value)
        value = value.replace(" - ", "-")
        return value.strip()

    def map_channel(self, name: str) -> ChannelInfo:
        normalized = self.normalize_channel_name(name)
        row = self.lookup.get(normalized)
        if row is None:
            return ChannelInfo(
                raw_name=str(name),
                normalized_name=normalized,
                canonical_name=None,
                family=None,
                modality=None,
                priority=10_000,
                matched=False,
                source="unmatched",
            )

        return ChannelInfo(
            raw_name=str(name),
            normalized_name=normalized,
            canonical_name=str(row["canonical_name"]),
            family=str(row["family"]),
            modality=str(row["modality"]),
            priority=int(row["priority"]),
            matched=True,
            source=str(row["source"]),
        )

    def group_channels_for_sleepfm(
        self,
        channel_names: Sequence[str],
        keep_unmatched: bool = False,
        preserve_original_order: bool = False,
    ) -> Dict[str, object]:
        grouped: Dict[str, List[GroupedChannel]] = {m: [] for m in MODALITY_ORDER}
        unmatched: List[ChannelInfo] = []

        for idx, raw_name in enumerate(channel_names):
            info = self.map_channel(raw_name)
            if not info.matched:
                unmatched.append(info)
                continue

            grouped[info.modality].append(
                GroupedChannel(
                    raw_name=info.raw_name,
                    normalized_name=info.normalized_name,
                    canonical_name=info.canonical_name or "",
                    family=info.family or "",
                    modality=info.modality or "",
                    priority=info.priority,
                    source=info.source,
                    original_index=idx,
                )
            )

        kept: Dict[str, List[GroupedChannel]] = {}
        dropped: Dict[str, List[GroupedChannel]] = {}

        for modality in MODALITY_ORDER:
            channels = grouped[modality]

            deduped: List[GroupedChannel] = []
            seen = set()
            for ch in channels:
                dedup_key = (ch.normalized_name, ch.canonical_name, ch.original_index)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                deduped.append(ch)

            if preserve_original_order:
                ordered = list(deduped)
            else:
                canonical_rank = self.canonical_priority.get(modality, {})
                ordered = sorted(
                    deduped,
                    key=lambda ch: (
                        canonical_rank.get(ch.canonical_name, 10_000),
                        ch.priority,
                        ch.original_index,
                    ),
                )

            cap = MAX_CHANNELS_BY_MODALITY[modality]
            kept[modality] = ordered[:cap]
            dropped[modality] = ordered[cap:]

        out = {
            "grouped": kept,
            "dropped": dropped,
            "unmatched": unmatched,
        }
        if not keep_unmatched:
            out.pop("unmatched", None)
        return out

    def canonical_names(self, modality: str) -> List[str]:
        modality = modality.upper()
        return list(CANONICAL_PRIORITY_BY_MODALITY.get(modality, []))


@lru_cache(maxsize=4)
def get_channel_mapper(table_path: str | Path = DEFAULT_CHANNEL_TABLE) -> ChannelMapper:
    return ChannelMapper(Path(table_path))


def normalize_channel_name(name: str) -> str:
    return ChannelMapper.normalize_channel_name(name)


def map_channel_to_canonical(name: str, table_path: str | Path = DEFAULT_CHANNEL_TABLE) -> ChannelInfo:
    mapper = get_channel_mapper(Path(table_path))
    return mapper.map_channel(name)


def map_channel_to_modality(name: str, table_path: str | Path = DEFAULT_CHANNEL_TABLE) -> Optional[str]:
    info = map_channel_to_canonical(name=name, table_path=table_path)
    return info.modality if info.matched else None


def group_channels_for_sleepfm(
    channel_names: Sequence[str],
    table_path: str | Path = DEFAULT_CHANNEL_TABLE,
    keep_unmatched: bool = False,
    preserve_original_order: bool = False,
) -> Dict[str, object]:
    mapper = get_channel_mapper(Path(table_path))
    return mapper.group_channels_for_sleepfm(
        channel_names=channel_names,
        keep_unmatched=keep_unmatched,
        preserve_original_order=preserve_original_order,
    )
