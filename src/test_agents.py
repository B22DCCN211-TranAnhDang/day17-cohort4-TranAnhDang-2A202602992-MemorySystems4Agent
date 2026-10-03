from __future__ import annotations

from pathlib import Path

import pytest

from config import LabConfig, ProviderConfig
from memory_store import (
    UserProfileStore,
    CompactMemoryManager,
    estimate_tokens,
)


def make_config(tmp_path: Path) -> LabConfig:
    """Create an isolated LabConfig for tests.

    The ``state_dir`` is placed inside the temporary directory so that all
    file‑system side effects are sandboxed. Thresholds are reduced to trigger
    compaction quickly.
    """
    base_dir = Path(__file__).resolve().parents[2]
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir = base_dir / "data"

    provider_cfg = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
        api_key=None,
        base_url=None,
    )
    judge_cfg = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
        api_key=None,
        base_url=None,
    )

    return LabConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=5,  # tiny threshold for fast compaction
        compact_keep_messages=2,
        model=provider_cfg,
        judge_model=judge_cfg,
    )


def test_user_profile_store(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    store = UserProfileStore(cfg.state_dir / "profiles")
    user_id = "alice"

    # Initially the file does not exist – read_text should return a default profile.
    default = store.read_text(user_id)
    assert "User Profile" in default

    # Write a profile and verify persistence.
    content = "# Alice\nName: Alice\nLocation: Hanoi\n"
    store.write_text(user_id, content)
    assert store.file_size(user_id) == len(content.encode())
    read_back = store.read_text(user_id)
    assert read_back == content

    # Edit a fact and ensure the change is reflected.
    changed = store.edit_text(user_id, "Alice", "Bob")
    assert changed is True
    edited = store.read_text(user_id)
    assert "Bob" in edited
    # Trying to replace a non‑existent token returns False.
    unchanged = store.edit_text(user_id, "Charlie", "Dave")
    assert unchanged is False


def test_compact_memory_manager_trigger(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    manager = CompactMemoryManager(
        threshold_tokens=cfg.compact_threshold_tokens,
        keep_messages=cfg.compact_keep_messages,
    )
    thread_id = "thread-1"

    # Append messages until we exceed the tiny threshold.
    for i in range(5):
        manager.append(thread_id, "user", f"Message {i}")

    # After compaction we should have at most ``keep_messages`` recent messages.
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= cfg.compact_keep_messages
    # Summary should be non‑empty because compaction happened.
    assert ctx["summary"] != ""
    # Compaction count must be at least one.
    assert manager.compaction_count(thread_id) >= 1


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("Hello") == 1  # 5 chars // 4 -> 1
    long_text = "a" * 100
    # 100 // 4 = 25 tokens
    assert estimate_tokens(long_text) == 25


def test_baseline_agent(tmp_path: Path) -> None:
    from agent_baseline import BaselineAgent

    cfg = make_config(tmp_path)
    agent = BaselineAgent(config=cfg, force_offline=True)

    # Thread 1: user introduces themselves
    res1 = agent.reply(user_id="user1", thread_id="t1", message="My name is Alice")
    assert "BaselineAgent" in res1["response"] or "Alice" in res1["response"]
    assert agent.token_usage("t1") > 0
    assert agent.prompt_token_usage("t1") > 0
    assert agent.compaction_count("t1") == 0

    # Thread 1: recall question in same thread
    res2 = agent.reply(user_id="user1", thread_id="t1", message="Mình tên gì?")
    assert "Alice" in res2["response"]

def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify AdvancedAgent remembers facts across threads via User.md while BaselineAgent does not."""
    from agent_advanced import AdvancedAgent
    from agent_baseline import BaselineAgent

    cfg = make_config(tmp_path)
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "user_test"

    # Thread 1: User gives facts
    baseline.reply(user_id=user_id, thread_id="t1", message="My name is Alice")
    advanced.reply(user_id=user_id, thread_id="t1", message="My name is Alice")

    # Thread 2 (New session): Ask recall question
    base_res = baseline.reply(user_id=user_id, thread_id="t2", message="Mình tên gì?")
    adv_res = advanced.reply(user_id=user_id, thread_id="t2", message="Mình tên gì?")

    # Baseline forgets across new thread
    assert "Alice" not in base_res["response"]
    # Advanced remembers across threads via persistent User.md
    assert "Alice" in adv_res["response"]
    assert advanced.memory_file_size(user_id) > 0


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Verify long threads trigger compaction in AdvancedAgent and track prompt load."""
    from agent_advanced import AdvancedAgent

    cfg = make_config(tmp_path)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    thread_id = "long_thread"

    # Send multiple messages to trigger compaction
    for i in range(10):
        advanced.reply(user_id="user_long", thread_id=thread_id, message=f"This is message number {i} with some content")

    # Compaction count should be > 0
    assert advanced.compaction_count(thread_id) > 0
    assert advanced.prompt_token_usage(thread_id) > 0


