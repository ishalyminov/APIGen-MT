from types import SimpleNamespace

from tool_manager import ToolManager
from tools.gorilla_file_system import GorillaFileSystem


def _manager_with(**instances):
    manager = ToolManager.__new__(ToolManager)
    manager.python_tool_instances = instances
    return manager


def test_snapshot_restore_preserves_integer_mapping_keys():
    trading = SimpleNamespace(orders={7: {"id": 7}})
    manager = _manager_with(trading_bot=trading)

    snapshot = manager.get_api_state()
    assert list(snapshot["trading_bot"]["orders"]) == [7]

    trading.orders[8] = {"id": 8}
    manager.restore_api_state(snapshot)
    assert trading.orders == {7: {"id": 7}}


def test_filesystem_restore_rebuilds_physical_workspace():
    filesystem = GorillaFileSystem(
        {"note.txt": {"type": "file", "content": "hello"}}
    )
    try:
        manager = _manager_with(gorilla_file_system=filesystem)
        snapshot = manager.get_api_state()

        assert filesystem.mkdir("logs")["success"] is True
        manager.restore_api_state(snapshot)

        assert filesystem.ls()["files"] == ["note.txt"]
        assert filesystem.mkdir("logs")["success"] is True
    finally:
        filesystem.cleanup()
