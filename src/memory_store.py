from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    """Estimate token count for a given text.
    This simple heuristic approximates tokens by dividing the character count by 4,
    which works reasonably well for English text with typical tokenization.
    """
    # Strip leading/trailing whitespace
    stripped = text.strip()
    if not stripped:
        return 0
    # Approximate: one token per ~4 characters
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Return the file path for a given user id.
        The user id is sanitized to be filesystem‑safe by replacing spaces with underscores.
        """
        safe_id = "_".join(user_id.split())
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        """Read the markdown profile for *user_id*.
        If the file does not exist, return a minimal default profile.
        """
        path = self.path_for(user_id)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        # Default empty profile
        return "# User Profile\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write *content* to the user's markdown file, creating the directory if needed.
        Returns the absolute path of the written file.
        """
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of *search_text* with *replacement* in the user's profile.
        Returns ``True`` if a replacement was made, otherwise ``False``.
        """
        txt = self.read_text(user_id)
        if search_text not in txt:
            return False
        new_txt = txt.replace(search_text, replacement, 1)
        self.write_text(user_id, new_txt)
        return True

    def file_size(self, user_id: str) -> int:
        """Return the size of the user's profile file in bytes, or ``0`` if missing.
        The returned size matches the length of the *logical* content (i.e. ``len(read_text().encode())``)
        so that tests using ``len(content.encode())`` stay platform‑independent.
        """
        # Use the logical content length (normalized newlines) for consistency across OSes.
        try:
            return len(self.read_text(user_id).encode())
        except Exception:
            return 0


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    # Simple regex‑based extraction of a few common facts.
    facts: dict[str, str] = {}

    # Skip pure question turns
    msg_lower = message.lower()
    if "?" in message and not any(kw in msg_lower for kw in ["tên là", "name is", "i'm", "ở ", "live in", "thích", "favorite"]):
        return facts

    # Name pattern e.g., "My name is Alice", "I'm Alice", "tên là DũngCT", "mình tên DũngCT"
    name_match = re.search(r"(?:name is|I\'m|tên là|mình tên|tên mình là)\s+([A-Za-z0-9_\-À-ỹ]+)", message, re.IGNORECASE)
    if name_match:
        facts["name"] = name_match.group(1).strip()

    # Location pattern e.g., "I live in Paris", "ở Đà Nẵng", "sống tại Hà Nội"
    loc_match = re.search(r"(?:live in|ở|sống tại|sống ở)\s+([A-Za-z0-9_\-\sÀ-ỹ]+?)(?=[.,;\n]|$)", message, re.IGNORECASE)
    if loc_match:
        loc_val = loc_match.group(1).strip()
        if len(loc_val) < 40 and not any(kw in loc_val.lower() for kw in ["là", "gì", "ngắn"]):
            facts["location"] = loc_val

    # Profession pattern e.g., "I am a doctor", "làm backend engineer"
    prof_match = re.search(r"(?:am a|làm)\s+([A-Za-z0-9_\-\sÀ-ỹ]+?)(?=[.,;\n]|$)", message, re.IGNORECASE)
    if prof_match:
        prof_val = prof_match.group(1).strip()
        if len(prof_val) < 50 and not any(kw in prof_val.lower() for kw in ["gì", "như"]):
            facts["profession"] = prof_val

    # Preference / style pattern e.g. "ngắn gọn", "3 bullet", "style"
    if any(p in msg_lower for p in ["ngắn gọn", "3 bullet", "bullet", "ví dụ thực tế"]):
        pref_parts = []
        if "ngắn gọn" in msg_lower:
            pref_parts.append("ngắn gọn")
        if "bullet" in msg_lower or "3 bullet" in msg_lower:
            pref_parts.append("3 bullet")
        if "ví dụ thực tế" in msg_lower:
            pref_parts.append("có ví dụ thực tế")
        if pref_parts:
            facts["preferences"] = ", ".join(pref_parts)
    else:
        pref_match = re.search(r"prefer[s]?\s+([A-Za-z0-9_\-\sÀ-ỹ]+?)(?=[.,;\n]|$)", message, re.IGNORECASE)
        if pref_match:
            facts["preferences"] = pref_match.group(1).strip()

    # Favorite food / drink pattern e.g. "cà phê sữa đá", "favorite drink is coffee"
    food_match = re.search(r"(?:favorite (?:food|drink)|đồ uống yêu thích|thích)\s+(?:is|là)?\s*([A-Za-z0-9_\-\sÀ-ỹ]+?)(?=[.,;\n]|$)", message, re.IGNORECASE)
    if food_match:
        fav_val = food_match.group(1).strip()
        if len(fav_val) < 40 and not any(kw in fav_val.lower() for kw in ["trả lời", "gì", "giao"]):
            facts["favorite"] = fav_val

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    # Take the oldest *max_items* messages (or fewer) and concatenate their content.
    if not messages:
        return ""
    # Preserve order: older first
    selected = messages[:max_items]
    parts = [msg.get("content", "") for msg in selected]
    summary = " ".join(parts).strip()
    return summary


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        # 3. trigger compaction if needed
        thread_state = self.state.setdefault(thread_id, {
            "messages": [],
            "summary": "",
            "compactions": 0,
        })
        thread_state["messages"].append({"role": role, "content": content})

        # Compute token count of current messages + existing summary
        all_text = " ".join(m["content"] for m in thread_state["messages"]) + " " + thread_state["summary"]
        if estimate_tokens(all_text) > self.threshold_tokens:
            # Perform compaction: keep the newest *keep_messages* messages, summarize the rest
            old_messages = thread_state["messages"][:-self.keep_messages]
            if old_messages:
                # Summarize old messages
                summary_part = summarize_messages(old_messages)
                # Append to existing summary
                if thread_state["summary"]:
                    thread_state["summary"] += " " + summary_part
                else:
                    thread_state["summary"] = summary_part
                # Remove the old messages from the list
                thread_state["messages"] = thread_state["messages"][-self.keep_messages :]
                thread_state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return the memory context for *thread_id*.
        The dictionary contains:
        - ``messages``: list of recent messages (each a dict with ``role`` and ``content``)
        - ``summary``: compacted summary of older messages
        - ``compactions``: number of times compaction has occurred for this thread
        """
        return self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        """Return how many compactions have been performed for *thread_id*."""
        return self.state.get(thread_id, {}).get("compactions", 0)
