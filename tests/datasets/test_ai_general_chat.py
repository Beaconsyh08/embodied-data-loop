"""Account-only conversations do not acquire dataset access or execution tools."""

import json
import uuid

import pytest

from lerobot.data_platform.ai_tools.personalization import conversations
from lerobot.data_platform.control_plane import ControlPlaneUser
from lerobot.data_platform.job_management import JobConflictError
from tests.datasets.test_ai_assistant import assistant, call, draft  # noqa: F401
from tests.datasets.test_ai_assistant_context import text_response
from tests.datasets.test_ai_assistant_runtime import processing_setup  # noqa: F401


@pytest.mark.parametrize("location", [None, "", "omitted"])
def test_general_chat_without_dataset_is_private_and_restorable(assistant, location, monkeypatch):  # noqa: F811
    service, fake, owner, value = assistant
    value = {**value, "message": "介绍数据处理流程", "location_id": location}
    if location == "omitted":
        value.pop("location_id")

    def no_dataset(*args, **kwargs):
        pytest.fail("General chat must not access any dataset")

    monkeypatch.setattr(service.manager, "_location", no_dataset)
    monkeypatch.setattr("lerobot.data_platform.ai_tools.strategies.Strategies.select", no_dataset)
    fake.responses = [
        call("read_context", {}),
        call("describe_operator", {"operator_id": "preprocess.split"}),
        text_response("可以先讨论流程，执行时再选择数据集。"),
    ]
    result = service.submit(owner, value)
    assert result["status"] == "done", result
    assert not result.get("proposal") and not result.get("action")
    restored = service.conversation(owner, value["conversation_id"])
    assert restored["location_id"] == ""
    assert restored["context"] == {"type": "general", "target_id": "account"}
    assert len(restored["history"]) == 2
    assert conversations(service, owner)["conversations"][0]["dataset"] == "General conversation"
    assert service.submit(owner, value) == result
    assert len(fake.requests) == 3
    names = {tool["function"]["name"] for tool in fake.requests[0]["tools"]}
    assert names == {"read_context", "describe_operator"}
    with pytest.raises((KeyError, PermissionError)):
        service.conversation(str(uuid.uuid4()), value["conversation_id"])
    assert service.store.list_jobs() == []
    with service.store.sessions.begin() as session:
        session.get(ControlPlaneUser, owner).active = False
    with pytest.raises(PermissionError):
        service.conversation(owner, value["conversation_id"])


@pytest.mark.parametrize("tool", ["preview_pipeline", "prepare_action"])
def test_general_chat_rejects_forged_execution_tools(assistant, tool):  # noqa: F811
    service, fake, owner, value = assistant
    fake.responses = [call(tool, {"definition": draft(), "summary": "执行"}), text_response("请先选择数据集")]
    result = service.submit(owner, {**value, "location_id": ""})
    assert result["status"] == "done", result
    assert "Choose a dataset" in json.dumps(fake.requests[-1]["messages"])
    assert not result.get("proposal") and not result.get("action")
    assert service.store.list_jobs() == [] and service.manager.list(owner) == []


def test_general_chat_cannot_rebind_to_dataset_or_accept_draft(assistant):  # noqa: F811
    service, fake, owner, value = assistant
    fake.responses = [text_response("Hello")]
    with pytest.raises(ValueError, match="Choose a dataset"):
        service.submit(owner, {**value, "location_id": "", "draft": draft()})
    service.submit(owner, {**value, "location_id": ""})
    revision = service.conversation(owner, value["conversation_id"])["revision"]
    with pytest.raises(JobConflictError):
        service.submit(owner, {**value, "request_id": str(uuid.uuid4()), "expected_revision": revision})
    with pytest.raises(ValueError, match="General conversations"):
        service.submit(owner, {**value, "context": {"type": "general", "target_id": "account"}})


def test_general_chat_uses_only_account_memory(assistant):  # noqa: F811
    from tests.datasets.test_ai_memory_followups import save

    service, fake, owner, value = assistant
    save(service, owner, "Account terminology")
    save(
        service, owner, "Dataset confidential terminology", scope="dataset", location_id=value["location_id"]
    )
    fake.responses = [text_response("Hello")]
    assert service.submit(owner, {**value, "location_id": ""})["status"] == "done"
    sent = json.dumps(fake.requests)
    assert "Account terminology" in sent
    assert "Dataset confidential terminology" not in sent
    assert value["location_id"] not in sent
