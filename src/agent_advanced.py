from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                return self._reply_offline(user_id, thread_id, message)
            except Exception:
                return self._reply_offline(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return total response tokens generated for thread_id."""
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return total prompt tokens processed for thread_id."""
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        """Return size of user profile markdown file in bytes."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return number of memory compactions performed for thread_id."""
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Implement the deterministic advanced path.

        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """
        # 1. Extract facts and update profile
        updates = extract_profile_updates(message)
        if updates:
            self._update_user_profile(user_id, updates)

        # 2. Append message to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt tokens for this turn
        prompt_tokens_turn = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens_turn

        # 4. Generate response using profile + thread context
        response_text = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply to compact memory
        self.compact_memory.append(thread_id, "assistant", response_text)

        # 6. Track generated tokens
        res_tokens = estimate_tokens(response_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + res_tokens

        return {
            "response": response_text,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
            "compactions": self.compaction_count(thread_id),
            "memory_file_size": self.memory_file_size(user_id),
        }

    def _update_user_profile(self, user_id: str, updates: dict[str, str]) -> None:
        """Parse existing profile facts and merge new updates."""
        existing_text = self.profile_store.read_text(user_id)
        facts: dict[str, str] = {}

        # Parse existing key-value lines like "- name: Alice"
        for line in existing_text.splitlines():
            if line.strip().startswith("- "):
                parts = line.strip()[2:].split(":", 1)
                if len(parts) == 2:
                    facts[parts[0].strip()] = parts[1].strip()

        # Apply new updates
        facts.update(updates)

        # Reconstruct markdown profile
        lines = [f"# User Profile: {user_id}", ""]
        for k, v in facts.items():
            lines.append(f"- {k}: {v}")
        new_content = "\n".join(lines) + "\n"
        self.profile_store.write_text(user_id, new_content)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Includes:
        - `User.md` profile content
        - compact summary text
        - recent kept messages
        """
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        msgs = ctx.get("messages", [])
        msgs_text = " ".join(m.get("content", "") for m in msgs)

        full_prompt = f"{profile_text}\nSummary: {summary}\nMessages: {msgs_text}"
        return estimate_tokens(full_prompt)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory."""
        msg_lower = message.lower()
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        recent_msgs = " ".join(m.get("content", "") for m in ctx.get("messages", []))

        is_question = any(q in msg_lower for q in ["?", "gì", "nào", "đâu", "who", "what", "where", "nhắc lại", "là ai"])

        if is_question:
            # Check profile first for persistent long-term facts across sessions
            if any(kw in msg_lower for kw in ["tên", "name", "who am i"]):
                for line in profile_text.splitlines():
                    if "name:" in line.lower():
                        val = line.split(":", 1)[1].strip()
                        return f"Tên của bạn là {val}."
                if "name" in summary.lower() or "name" in recent_msgs.lower():
                    return f"Theo trí nhớ: {summary} {recent_msgs}"
                return "Tôi chưa ghi nhận tên của bạn."

            if any(kw in msg_lower for kw in ["nghề", "job", "profession", "làm gì"]):
                for line in profile_text.splitlines():
                    if "profession:" in line.lower() or "job:" in line.lower():
                        val = line.split(":", 1)[1].strip()
                        return f"Nghề nghiệp của bạn là {val}."
                return "Tôi chưa ghi nhận nghề nghiệp của bạn."

            if any(kw in msg_lower for kw in ["thích", "prefer", "favorite", "style"]):
                found = []
                for line in profile_text.splitlines():
                    if any(k in line.lower() for k in ["preferences:", "favorite:"]):
                        found.append(line.strip())
                if found:
                    return f"Sở thích / style của bạn: {'; '.join(found)}"
                return "Tôi chưa ghi nhận sở thích của bạn."

            if any(kw in msg_lower for kw in ["sống", "ở đâu", "live", "location"]):
                for line in profile_text.splitlines():
                    if "location:" in line.lower():
                        val = line.split(":", 1)[1].strip()
                        return f"Bạn sống tại {val}."
                return "Tôi chưa ghi nhận nơi ở của bạn."

        return f"AdvancedAgent đã nhận: {message}"

    def _maybe_build_langchain_agent(self):
        """Wire a live agent with tools and compact middleware if configured."""
        if self.force_offline or not self.config.model.api_key:
            self.langchain_agent = None
            return
        try:
            model = build_chat_model(self.config.model)
            self.langchain_agent = model
        except Exception:
            self.langchain_agent = None
