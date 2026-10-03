from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            # If live agent is configured, use live path (fall back to offline if needed)
            try:
                # Live agent invocation placeholder
                return self._reply_offline(thread_id, message)
            except Exception:
                return self._reply_offline(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate how much prompt context this baseline kept processing."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement a simple offline behavior.

        Within-session memory only:
        - Stores messages per thread_id
        - Can answer recall questions if mentioned earlier in the SAME thread_id
        - Does NOT remember across different thread_ids
        """
        session = self.sessions.setdefault(thread_id, SessionState())

        # 1. Estimate prompt tokens processed for this turn (all history + new message)
        history_text = " ".join(m["content"] for m in session.messages)
        prompt_tokens_this_turn = estimate_tokens(history_text) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens_this_turn

        # 2. Append user message to thread session
        session.messages.append({"role": "user", "content": message})

        # 3. Generate response based ONLY on current thread history
        response_text = self._generate_offline_response(session.messages, message)

        # 4. Append assistant response to session
        session.messages.append({"role": "assistant", "content": response_text})

        # 5. Update agent token usage
        res_tokens = estimate_tokens(response_text)
        session.token_usage += res_tokens

        return {
            "response": response_text,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
            "compactions": 0,
        }

    def _generate_offline_response(self, history: list[dict[str, str]], current_msg: str) -> str:
        """Helper to generate a deterministic answer from in-session history."""
        msg_lower = current_msg.lower()

        is_question = any(q in msg_lower for q in ["?", "gì", "nào", "đâu", "who", "what", "where", "nhắc lại", "là ai"])

        if is_question:
            # Check for recall questions against past messages in current thread (including all previous turns)
            if any(kw in msg_lower for kw in ["tên", "name", "who am i"]):
                for m in reversed(history[:-1]):
                    text = m["content"]
                    if any(k in text.lower() for k in ["name is", "tên là", "i'm"]):
                        return f"Bạn đã nói tên trong phiên này: {text}"
                return "Tôi không có thông tin về tên của bạn trong phiên làm việc này."

            if any(kw in msg_lower for kw in ["nghề", "job", "profession", "làm gì", "thích", "prefer", "favorite"]):
                for m in reversed(history[:-1]):
                    text = m["content"]
                    if any(k in text.lower() for k in ["làm", "job", "prefer", "thích", "favorite", "is a", "am a"]):
                        return f"Theo thông tin trong phiên này: {text}"
                return "Tôi không tìm thấy thông tin này trong phiên làm việc hiện tại."

        return f"BaselineAgent đã nhận: {current_msg}"

    def _maybe_build_langchain_agent(self):
        """Optionally wire `create_agent` + `InMemorySaver` here."""
        if self.force_offline or not self.config.model.api_key:
            self.langchain_agent = None
            return
        try:
            model = build_chat_model(self.config.model)
            self.langchain_agent = model
        except Exception:
            self.langchain_agent = None
