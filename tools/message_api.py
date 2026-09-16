"""Auto-generated MessageAPI implementation."""

import json
import math
import re
import copy
from datetime import datetime
from typing import List, Dict, Any, Optional


class MessageAPI:
    """API to manage user interactions and messaging in a workspace."""

    def __init__(self, initial_config: dict) -> None:
        """Initialize the MessageAPI with the given configuration."""
        self.workspace_id: str = initial_config.get("workspace_id", "")
        self.user_count: int = initial_config.get("user_count", 0)
        self.user_map: Dict[str, str] = copy.deepcopy(initial_config.get("user_map", {}))
        self.messages_sent_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = copy.deepcopy(initial_config.get("messages_sent_map", {}))
        self.messages_inbox_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = copy.deepcopy(initial_config.get("messages_inbox_map", {}))

        configured_message_count = int(initial_config.get("message_count", 0) or 0)
        existing_message_ids = []
        for outer_map in (self.messages_sent_map, self.messages_inbox_map):
            for peer_map in outer_map.values():
                if not isinstance(peer_map, dict):
                    continue
                for messages in peer_map.values():
                    if not isinstance(messages, list):
                        continue
                    for message in messages:
                        try:
                            existing_message_ids.append(int(message.get("message_id")))
                        except (AttributeError, TypeError, ValueError):
                            pass
        self.message_count = max(
            configured_message_count,
            max(existing_message_ids, default=0),
        )
        self.current_user: str = initial_config.get("current_user", "")

    def add_contact(self, user_name: str) -> dict:
        """Add a contact to the workspace."""
        if not user_name:
            return {
                "added_status": False,
                "user_id": "",
                "message": "User name cannot be empty."
            }

        # Case-insensitive check
        existing_name = None
        for name in self.user_map:
            if name.lower() == user_name.lower():
                existing_name = name
                break

        if existing_name:
            return {
                "added_status": False,
                "user_id": self.user_map[existing_name],
                "message": f"User '{user_name}' already exists in the workspace."
            }

        existing_ids = {
            uid for uid in self.user_map.values()
        } | set(self.messages_sent_map.keys()) | set(self.messages_inbox_map.keys())
        max_num = 0
        for uid in existing_ids:
            try:
                num = int(uid.replace("USR", ""))
                if num > max_num:
                    max_num = num
            except (ValueError, AttributeError):
                pass

        new_num = max_num + 1
        new_user_id = f"USR{new_num:03d}"
        self.user_count += 1
        self.user_map[user_name] = new_user_id

        # Initialize message maps for the new user
        self.messages_sent_map[new_user_id] = {}
        self.messages_inbox_map[new_user_id] = {}

        return {
            "added_status": True,
            "user_id": new_user_id,
            "message": f"Contact '{user_name}' added successfully."
        }

    def delete_message(self, receiver_id: str, message_id: Optional[int] = None) -> dict:
        """Delete a message sent to a receiver by its message_id."""
        if not self.current_user:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": "No user currently logged in."
            }

        if not receiver_id:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": "Receiver ID cannot be empty."
            }

        sender_id = self.current_user

        # Check if sender has sent any messages to the receiver
        if sender_id not in self.messages_sent_map or receiver_id not in self.messages_sent_map[sender_id]:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": "No messages found to delete."
            }

        sent_messages = self.messages_sent_map[sender_id][receiver_id]

        if not sent_messages:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": "No messages found to delete."
            }

        # ``message_id`` is optional in the BFCL contract: omitting it means
        # "delete the latest message to this receiver".  The previous
        # implementation searched for ID 0, making every such valid call fail.
        # Legacy configs may also contain bare message strings rather than the
        # newer structured records, so keep deletion deterministic for both.
        found_idx = None
        if message_id is None:
            found_idx = len(sent_messages) - 1
            latest = sent_messages[found_idx]
            msg_id_int = (
                int(latest.get("message_id", found_idx + 1))
                if isinstance(latest, dict)
                else found_idx + 1
            )
        else:
            try:
                msg_id_int = int(message_id)
            except (TypeError, ValueError):
                return {
                    "deleted_status": False,
                    "message_id": "0",
                    "message": f"Invalid message ID {message_id}.",
                }
            for idx, message_record in enumerate(sent_messages):
                if (
                    isinstance(message_record, dict)
                    and message_record.get("message_id") == msg_id_int
                ):
                    found_idx = idx
                    break

        if found_idx is None:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": f"Message with ID {message_id} not found."
            }

        removed_message = sent_messages.pop(found_idx)

        # Remove the corresponding message from the receiver's inbox
        if receiver_id in self.messages_inbox_map and sender_id in self.messages_inbox_map[receiver_id]:
            inbox_messages = self.messages_inbox_map[receiver_id][sender_id]
            for idx, message_record in enumerate(inbox_messages):
                same_id = (
                    isinstance(message_record, dict)
                    and message_record.get("message_id") == msg_id_int
                )
                same_legacy_content = (
                    isinstance(removed_message, str)
                    and message_record == removed_message
                )
                if same_id or same_legacy_content:
                    inbox_messages.pop(idx)
                    break

        return {
            "deleted_status": True,
            "message_id": str(msg_id_int),
            "message": "Message deleted successfully."
        }

    def get_user_id(self, user: str) -> dict:
        """Get user ID from user name."""
        if not user:
            return {
                "user_id": ""
            }

        # Case-insensitive lookup
        for name, uid in self.user_map.items():
            if name.lower() == user.lower():
                return {"user_id": uid}
        return {
            "user_id": ""
        }

    def message_login(self, user_id: str) -> dict:
        """Log in a user with the given user ID to message application."""
        if not user_id:
            return {
                "login_status": False,
                "message": "User ID cannot be empty."
            }

        user_exists = user_id in self.user_map.values() or user_id in self.messages_sent_map or user_id in self.messages_inbox_map

        if not user_exists:
            return {
                "login_status": False,
                "message": f"User ID '{user_id}' not found in the workspace."
            }

        self.current_user = user_id
        return {
            "login_status": True,
            "message": f"User '{user_id}' logged in successfully."
        }

    def get_message_stats(self) -> dict:
        """Return deterministic message statistics for the current user.

        The BFCL contract describes this as a current-user operation.  A
        logged-out call therefore returns zero counts plus ``logged_in=False``
        instead of leaking another user's mailbox.
        """
        if not self.current_user:
            return {
                "logged_in": False,
                "current_user": "",
                "sent_count": 0,
                "received_count": 0,
                "total_count": 0,
                "conversation_count": 0,
            }

        sent_map = self.messages_sent_map.get(self.current_user, {})
        inbox_map = self.messages_inbox_map.get(self.current_user, {})
        sent_count = sum(len(messages) for messages in sent_map.values())
        received_count = sum(len(messages) for messages in inbox_map.values())
        conversation_count = len(set(sent_map) | set(inbox_map))
        return {
            "logged_in": True,
            "current_user": self.current_user,
            "sent_count": sent_count,
            "received_count": received_count,
            "total_count": sent_count + received_count,
            "conversation_count": conversation_count,
        }

    def list_users(self) -> dict:
        """List all users in the workspace in a stable order."""
        users = [
            {"name": name, "user_id": user_id}
            for name, user_id in sorted(
                self.user_map.items(),
                key=lambda item: (item[0].casefold(), item[1]),
            )
        ]
        return {
            "workspace_id": self.workspace_id,
            "user_count": len(users),
            "users": users,
        }

    def message_get_login_status(self) -> dict:
        """Return the current Message API authentication state."""
        return {
            "logged_in": bool(self.current_user),
            "current_user": self.current_user,
        }

    def view_messages_sent(self) -> dict:
        """Return all messages sent by the current user.

        Messages are flattened and sorted by ID so that simulator replay is
        deterministic and downstream tools can reference individual results.
        """
        if not self.current_user:
            return {
                "logged_in": False,
                "current_user": "",
                "message_count": 0,
                "messages": [],
            }

        messages: List[Dict[str, Any]] = []
        for receiver_id, sent_messages in self.messages_sent_map.get(
            self.current_user, {}
        ).items():
            for message in sent_messages:
                if isinstance(message, dict):
                    messages.append({
                        "receiver_id": receiver_id,
                        "message_id": int(message.get("message_id", 0) or 0),
                        "message": str(message.get("message", "")),
                    })
                else:
                    messages.append({
                        "receiver_id": receiver_id,
                        "message_id": 0,
                        "message": str(message),
                    })
        messages.sort(key=lambda item: (item["message_id"], item["receiver_id"]))
        return {
            "logged_in": True,
            "current_user": self.current_user,
            "message_count": len(messages),
            "messages": messages,
        }

    def search_messages(self, keyword: str) -> dict:
        """Search for messages containing a specific keyword."""
        results: List[Dict[str, Any]] = []

        if not keyword:
            return {
                "results": []
            }

        if not self.current_user:
            return {
                "results": []
            }

        sender_id = self.current_user

        # Search in sent messages
        if sender_id in self.messages_sent_map:
            for receiver_id, messages in self.messages_sent_map[sender_id].items():
                for msg in messages:
                    msg_text = msg.get("message", "") if isinstance(msg, dict) else str(msg)
                    if keyword.lower() in msg_text.lower():
                        results.append({
                            "receiver_id": receiver_id,
                            "message": msg_text,
                            "direction": "sent"
                        })

        # Search in received messages (inbox)
        if sender_id in self.messages_inbox_map:
            for sender, messages in self.messages_inbox_map[sender_id].items():
                for msg in messages:
                    msg_text = msg.get("message", "") if isinstance(msg, dict) else str(msg)
                    if keyword.lower() in msg_text.lower():
                        results.append({
                            "sender_id": sender,
                            "message": msg_text,
                            "direction": "received"
                        })

        return {
            "results": results
        }

    def send_message(self, receiver_id: str, message: str) -> dict:
        """Send a message to a user."""
        if not self.current_user:
            return {
                "sent_status": False,
                "message_id": "0",
                "message": "No user currently logged in."
            }

        if not receiver_id:
            return {
                "sent_status": False,
                "message_id": "0",
                "message": "Receiver ID cannot be empty."
            }

        if not message:
            return {
                "sent_status": False,
                "message_id": "0",
                "message": "Message cannot be empty."
            }

        sender_id = self.current_user

        self.message_count += 1
        msg_id = self.message_count

        # Create message dict with ID (both stored as int for lookup, returned as string per schema)
        msg_dict = {"message_id": msg_id, "message": message}

        # Initialize sender's sent map if needed
        if sender_id not in self.messages_sent_map:
            self.messages_sent_map[sender_id] = {}

        # Initialize receiver's entry in sender's sent map if needed
        if receiver_id not in self.messages_sent_map[sender_id]:
            self.messages_sent_map[sender_id][receiver_id] = []

        # Add message dict to sender's sent map
        self.messages_sent_map[sender_id][receiver_id].append(msg_dict)

        # Initialize receiver's inbox map if needed
        if receiver_id not in self.messages_inbox_map:
            self.messages_inbox_map[receiver_id] = {}

        # Initialize sender's entry in receiver's inbox map if needed
        if sender_id not in self.messages_inbox_map[receiver_id]:
            self.messages_inbox_map[receiver_id][sender_id] = []

        # Add message dict to receiver's inbox map
        self.messages_inbox_map[receiver_id][sender_id].append(msg_dict)

        return {
            "sent_status": True,
            "message_id": str(msg_id),
            "message": "Message sent successfully."
        }
