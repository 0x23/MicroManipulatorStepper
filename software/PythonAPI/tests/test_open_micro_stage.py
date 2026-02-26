"""Tests for the OpenMicroStageInterface class."""

import unittest
from unittest.mock import patch

import numpy as np

from open_micro_stage_api.api import OpenMicroStageInterface, SerialInterface

DESIRED_FIRMWARE_VERSION = "v1.0.1"


class MockSerialInterface:
    """Generic mock implementation of SerialInterface for testing."""

    ReplyStatus = SerialInterface.ReplyStatus
    LogLevel = SerialInterface.LogLevel

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud_rate: int = 115200,
        command_msg_callback=None,
        log_msg_callback=None,
        unsolicited_msg_callback=None,
        reconnect_timeout: int = 5,
    ):
        """Initialize mock serial interface."""
        self.port = port
        self.baud_rate = baud_rate
        self.reconnect_timeout = reconnect_timeout
        self.command_msg_callback = command_msg_callback
        self.log_message_callback = log_msg_callback
        self.unsolicited_msg_callback = unsolicited_msg_callback

        # Command response queue
        self.responses = []
        self.response_index = 0
        self.call_history = []

    def set_response(self, status: SerialInterface.ReplyStatus, response: str):
        """Set a single response for the next send_command call."""
        self.responses = [(status, response)]
        self.response_index = 0

    def set_responses(self, responses: list):
        """Set multiple responses for sequential send_command calls.

        Args:
            responses: List of (status, response) tuples
        """
        self.responses = responses
        self.response_index = 0

    def send_command(self, cmd: str, timeout=2):
        """Send a command and return a mocked response."""
        self.call_history.append((cmd, timeout))

        if self.response_index >= len(self.responses):
            # If we run out of responses, return the last one or OK
            if self.responses:
                status, response = self.responses[-1]
            else:
                status, response = SerialInterface.ReplyStatus.OK, ""
        else:
            status, response = self.responses[self.response_index]
            self.response_index += 1

        # Call the command callback if set
        if self.command_msg_callback:
            self.command_msg_callback(cmd, None, "")

        return status, response

    def close(self):
        """Close the mock connection."""
        pass

    def reset(self):
        """Reset mock state for a new test."""
        self.responses = []
        self.response_index = 0
        self.call_history = []

    def get_last_command(self):
        """Get the last command that was sent."""
        if self.call_history:
            return self.call_history[-1][0]
        return None

    def get_all_commands(self):
        """Get all commands that were sent."""
        return [cmd for cmd, _ in self.call_history]

    def assert_command_called(self, cmd: str):
        """Assert that a specific command was called."""
        if cmd not in self.get_all_commands():
            raise AssertionError(f"Command '{cmd}' was not called. Commands: {self.get_all_commands()}")

    def assert_command_called_with_args(self, cmd: str, timeout=None):
        """Assert that a specific command was called with specific arguments."""
        for called_cmd, called_timeout in self.call_history:
            if called_cmd == cmd and (timeout is None or called_timeout == timeout):
                return
        raise AssertionError(
            f"Command '{cmd}' with timeout={timeout} was not called. Call history: {self.call_history}"
        )


class TestOpenMicroStageInterface(unittest.TestCase):
    """End-to-end tests for OpenMicroStageInterface."""

    def setUp(self):
        """Set up test fixtures with mocked SerialInterface."""
        # Create a persistent mock instance
        self.mock_serial_instance = MockSerialInterface()

        # Patch SerialInterface to return the same mock instance every time
        self.patcher = patch("open_micro_stage.api.SerialInterface.__new__", return_value=self.mock_serial_instance)
        self.patcher.start()

        # Create the interface
        self.interface = OpenMicroStageInterface(show_communication=False, show_log_messages=False)

        # default connect for the majority of commands
        self.mock_serial_instance.set_response(
            SerialInterface.ReplyStatus.OK,
            DESIRED_FIRMWARE_VERSION,
        )
        self.interface.connect("/dev/ttyACM0")
        self.calls_for_initailization = len(self.mock_serial_instance.call_history)

    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()

    def test_initialization(self):
        """Test that OpenMicroStageInterface initializes with correct defaults."""
        interface = OpenMicroStageInterface()
        self.assertIsNone(interface.serial)
        self.assertTrue(np.array_equal(interface.workspace_transform, np.eye(4)))
        self.assertTrue(interface.show_communication)
        self.assertTrue(interface.show_log_messages)
        self.assertFalse(interface.disable_message_callbacks)

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        interface = OpenMicroStageInterface(show_communication=False, show_log_messages=False)
        self.assertFalse(interface.show_communication)
        self.assertFalse(interface.show_log_messages)

    def test_connect_success(self):
        """Test successful connection to device."""
        self.mock_serial_instance.set_response(
            SerialInterface.ReplyStatus.OK,
            DESIRED_FIRMWARE_VERSION,
        )
        self.interface.connect("/dev/ttyACM0")

        self.assertIsNotNone(self.interface.serial)
        self.assertEqual(self.interface.serial.port, "/dev/ttyACM0")

    def test_connect_with_custom_baud_rate(self):
        """Test connection with custom baud rate."""
        self.mock_serial_instance.set_response(
            SerialInterface.ReplyStatus.OK,
            DESIRED_FIRMWARE_VERSION,
        )

        self.interface.connect("/dev/ttyACM0", baud_rate=115200)

        # Verify the mock was called with correct parameters
        self.assertIsNotNone(self.interface.serial)
        self.assertEqual(self.interface.serial.baud_rate, 115200)

    def test_connect_incompatible_firmware(self):
        """Test connection fails with incompatible firmware version."""
        self.mock_serial_instance.set_response(
            SerialInterface.ReplyStatus.OK,
            "v0.9.0",
        )

        self.interface.connect("/dev/ttyACM0")

        # Serial should be set to None on incompatible version
        self.assertIsNone(self.interface.serial)

    def test_disconnect(self):
        """Test disconnection from device."""
        self.mock_serial_instance.set_response(
            SerialInterface.ReplyStatus.OK,
            DESIRED_FIRMWARE_VERSION,
        )

        self.interface.connect("/dev/ttyACM0")
        self.interface.disconnect()

        self.assertIsNone(self.interface.serial)

    def test_disconnect_when_not_connected(self):
        """Test disconnect gracefully handles when not connected."""
        # Should not raise exception
        self.interface.disconnect()
        self.assertIsNone(self.interface.serial)

    def test_set_and_get_workspace_transform(self):
        """Test setting and getting workspace transform."""
        transform = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])

        self.interface.set_workspace_transform(transform)
        result = self.interface.get_workspace_transform()

        self.assertTrue(np.array_equal(result, transform))

    def test_read_firmware_version(self):
        """Test reading firmware version."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "v1.2.3",
        )

        major, minor, patch = self.interface.read_firmware_version()

        self.assertEqual(major, 1)
        self.assertEqual(minor, 2)
        self.assertEqual(patch, 3)
        self.interface.serial.assert_command_called("M58")

    def test_read_firmware_version_error(self):
        """Test firmware version returns 0,0,0 on error."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.ERROR,
            "",
        )

        major, minor, patch = self.interface.read_firmware_version()

        self.assertEqual((major, minor, patch), (0, 0, 0))

    def test_home_all_axes(self):
        """Test homing all axes."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.home()

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.interface.serial.assert_command_called("G28 A B C D E F\n")

    def test_home_specific_axes(self):
        """Test homing specific axes."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.home(axis_list=[0, 2])

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.interface.serial.assert_command_called("G28 A C\n")

    def test_home_invalid_axis(self):
        """Test homing with invalid axis index raises error."""
        with self.assertRaises(ValueError):
            self.interface.home(axis_list=[10])

    def test_calibrate_joint_no_save(self):
        """Test calibrating a joint without saving results."""
        calibration_response = "0.5,1.0,100\n1.0,2.0,200\n1.5,3.0,300\n"
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            calibration_response,
        )

        result, data = self.interface.calibrate_joint(0, save_result=False)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0], [0.5, 1.0, 1.5])  # motor angles
        self.assertEqual(data[1], [1.0, 2.0, 3.0])  # field angles
        self.assertEqual(data[2], [100, 200, 300])  # encoder counts
        self.interface.serial.assert_command_called("M56 J0 P")

    def test_calibrate_joint_with_save(self):
        """Test calibrating a joint with saving results."""
        calibration_response = "0.5,1.0,100\n"
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            calibration_response,
        )

        result, data = self.interface.calibrate_joint(1, save_result=True)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.interface.serial.assert_command_called("M56 J1 P S")

    def test_read_current_position(self):
        """Test reading current position."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "X10.5 Y20.3 Z15.8",
        )

        x, y, z = self.interface.read_current_position()

        self.assertAlmostEqual(x, 10.5)
        self.assertAlmostEqual(y, 20.3)
        self.assertAlmostEqual(z, 15.8)
        self.interface.serial.assert_command_called("M50")

    def test_read_current_position_error(self):
        """Test reading current position returns None on error."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.ERROR,
            "",
        )

        x, y, z = self.interface.read_current_position()

        self.assertIsNone(x)
        self.assertIsNone(y)
        self.assertIsNone(z)

    def test_read_current_position_invalid_format(self):
        """Test reading current position with invalid format raises error."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "invalid format",
        )

        with self.assertRaises(ValueError):
            self.interface.read_current_position()

    def test_move_to_immediate(self):
        """Test moving to position with immediate execution."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.move_to(5.0, 10.0, 15.0, f=20.0, move_immediately=True)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        cmd = self.interface.serial.get_last_command()
        self.assertIn("G0 X5.000000 Y10.000000 Z15.000000 F20.000", cmd)
        self.assertIn("I", cmd)

    def test_move_to_with_workspace_transform(self):
        """Test move_to applies workspace transform correctly."""
        # Set a simple translation transform
        transform = np.array([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])
        self.interface.set_workspace_transform(transform)

        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.move_to(0, 0, 0, f=10.0)

        # Expected transformed position is (2, 3, 4)
        cmd = self.interface.serial.get_last_command()
        self.assertIn("X2.000000", cmd)
        self.assertIn("Y3.000000", cmd)
        self.assertIn("Z4.000000", cmd)

    def test_move_to_blocking_busy_retry(self):
        """Test move_to retries on BUSY when blocking is True."""
        self.interface.serial.set_responses(
            [
                (SerialInterface.ReplyStatus.BUSY, ""),
                (SerialInterface.ReplyStatus.OK, ""),
            ]
        )

        result = self.interface.move_to(5.0, 10.0, 15.0, f=20.0, blocking=True, timeout=0.01)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(len(self.interface.serial.call_history), 2 + self.calls_for_initailization)

    def test_move_to_non_blocking_returns_busy(self):
        """Test move_to returns BUSY immediately when blocking is False."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.BUSY,
            "",
        )

        result = self.interface.move_to(5.0, 10.0, 15.0, f=20.0, blocking=False)

        self.assertEqual(result, SerialInterface.ReplyStatus.BUSY)
        self.assertEqual(len(self.interface.serial.call_history), 1 + self.calls_for_initailization)

    def test_dwell(self):
        """Test dwell command."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.dwell(time_s=2.5, blocking=True)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        cmd = self.interface.serial.get_last_command()
        self.assertIn("G4 S2.500000", cmd)

    def test_set_max_acceleration(self):
        """Test setting max acceleration."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.set_max_acceleration(linear_accel=100.0, angular_accel=50.0)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        cmd = self.interface.serial.get_last_command()
        self.assertIn("M204 L100.000000 A50.000000", cmd)

    def test_set_max_acceleration_minimum_values(self):
        """Test max acceleration enforces minimum values."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        self.interface.set_max_acceleration(linear_accel=0.001, angular_accel=0.001)

        cmd = self.interface.serial.get_last_command()
        # Should be clamped to 0.01
        self.assertIn("M204 L0.010000 A0.010000", cmd)

    def test_wait_for_stop_ready(self):
        """Test wait_for_stop returns when device is ready."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "1",
        )

        result = self.interface.wait_for_stop()

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.interface.serial.assert_command_called("M53\n")

    def test_wait_for_stop_polls_until_ready(self):
        """Test wait_for_stop polls until device is ready."""
        self.interface.serial.set_responses(
            [
                (SerialInterface.ReplyStatus.OK, "0"),
                (SerialInterface.ReplyStatus.OK, "0"),
                (SerialInterface.ReplyStatus.OK, "1"),
            ]
        )

        result = self.interface.wait_for_stop()

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(len(self.interface.serial.call_history), 3 + self.calls_for_initailization)

    def test_wait_for_stop_error(self):
        """Test wait_for_stop returns error status."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.ERROR,
            "",
        )

        result = self.interface.wait_for_stop()

        self.assertEqual(result, SerialInterface.ReplyStatus.ERROR)

    def test_enable_motors(self):
        """Test enabling motors."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.enable_motors(enable=True)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.interface.serial.assert_command_called_with_args("M17", timeout=5)

    def test_disable_motors(self):
        """Test disabling motors."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.enable_motors(enable=False)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.interface.serial.assert_command_called_with_args("M18", timeout=5)

    def test_set_pose(self):
        """Test setting pose."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.set_pose(x=5.0, y=10.0, z=15.0)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        cmd = self.interface.serial.get_last_command()
        self.assertIn("G24 X5.000000 Y10.000000 Z15.000000", cmd)

    def test_send_custom_command(self):
        """Test sending custom command."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "response data",
        )

        result, response = self.interface.send_command("M57", timeout_s=3.0)

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(response, "response data")

    def test_read_device_state_info(self):
        """Test reading device state info."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "state info",
        )

        result = self.interface.read_device_state_info()

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.interface.serial.assert_command_called("M57")

    def test_set_servo_parameter_defaults(self):
        """Test setting servo parameters with defaults."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.set_servo_parameter()

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        cmd = self.interface.serial.get_last_command()
        self.assertIn("M55", cmd)
        self.assertIn("A150.000000", cmd)  # pos_kp
        self.assertIn("B50000.000000", cmd)  # pos_ki
        self.assertIn("C0.200000", cmd)  # vel_kp
        self.assertIn("D100.000000", cmd)  # vel_ki
        self.assertIn("F0.002500", cmd)  # vel_filter_tc

    def test_set_servo_parameter_custom(self):
        """Test setting servo parameters with custom values."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.set_servo_parameter(
            pos_kp=200, pos_ki=60000, vel_kp=0.3, vel_ki=120, vel_filter_tc=0.003
        )

        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        cmd = self.interface.serial.get_last_command()
        self.assertIn("A200.000000", cmd)
        self.assertIn("B60000.000000", cmd)
        self.assertIn("C0.300000", cmd)
        self.assertIn("D120.000000", cmd)
        self.assertIn("F0.003000", cmd)

    def test_read_encoder_angles(self):
        """Test reading encoder angles returns empty list."""
        self.interface.serial.set_response(
            SerialInterface.ReplyStatus.OK,
            "",
        )

        result = self.interface.read_encoder_angles()

        self.assertEqual(result, [])
        self.interface.serial.assert_command_called("M51")

    def test_parse_table_data(self):
        """Test parsing table data."""
        data_string = "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"
        result = OpenMicroStageInterface._parse_table_data(data_string, 3)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [1.0, 4.0, 7.0])
        self.assertEqual(result[1], [2.0, 5.0, 8.0])
        self.assertEqual(result[2], [3.0, 6.0, 9.0])

    def test_parse_table_data_with_malformed_lines(self):
        """Test parsing table data skips malformed lines."""
        data_string = "1.0,2.0,3.0\ninvalid\n4.0,5.0,6.0"
        result = OpenMicroStageInterface._parse_table_data(data_string, 3)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [1.0, 4.0])
        self.assertEqual(result[1], [2.0, 5.0])
        self.assertEqual(result[2], [3.0, 6.0])

    def test_parse_table_data_single_row(self):
        """Test parsing single row of data."""
        data_string = "10.5,20.3,15.8"
        result = OpenMicroStageInterface._parse_table_data(data_string, 3)

        self.assertEqual(result[0], [10.5])
        self.assertEqual(result[1], [20.3])
        self.assertEqual(result[2], [15.8])

    def test_workflow_connect_home_move_stop(self):
        """Test end-to-end workflow: connect, home, move, wait for stop."""
        # Set responses for each command in sequence
        self.mock_serial_instance.set_responses(
            [
                (SerialInterface.ReplyStatus.OK, DESIRED_FIRMWARE_VERSION),  # firmware version
                (SerialInterface.ReplyStatus.OK, ""),  # home
                (SerialInterface.ReplyStatus.OK, ""),  # move_to
                (SerialInterface.ReplyStatus.OK, "1"),  # wait_for_stop
            ]
        )

        self.interface.connect("/dev/ttyACM0")
        self.assertIsNotNone(self.interface.serial)

        home_result = self.interface.home()
        self.assertEqual(home_result, SerialInterface.ReplyStatus.OK)

        move_result = self.interface.move_to(5.0, 10.0, 15.0, f=20.0)
        self.assertEqual(move_result, SerialInterface.ReplyStatus.OK)

        stop_result = self.interface.wait_for_stop()
        self.assertEqual(stop_result, SerialInterface.ReplyStatus.OK)

    def test_workflow_calibrate_and_move(self):
        """Test end-to-end workflow: calibrate joint and then move."""
        calibration_data = "0.5,1.0,100\n1.0,2.0,200\n"

        self.mock_serial_instance.set_responses(
            [
                (SerialInterface.ReplyStatus.OK, calibration_data),  # calibrate
                (SerialInterface.ReplyStatus.OK, "X5.0 Y10.0 Z15.0"),  # read position
            ]
        )

        result, data = self.interface.calibrate_joint(0, save_result=True)
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(len(data), 3)

        x, y, z = self.interface.read_current_position()
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 10.0)
        self.assertAlmostEqual(z, 15.0)


# Manually run unittest
if __name__ == "__main__":
    unittest.main()
