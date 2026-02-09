"""Tests for the OpenMicroStageInterface class."""

import re
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from open_micro_stage.api import OpenMicroStageInterface, SerialInterface


class TestOpenMicroStageInterface(unittest.TestCase):
    """End-to-end tests for OpenMicroStageInterface."""

    def setUp(self):
        """Set up test fixtures with mocked SerialInterface."""
        # Patch SerialInterface.__init__ to prevent actual connection attempts
        self.patcher = patch("open_micro_stage.api.SerialInterface.__init__", return_value=None)
        self.mock_init = self.patcher.start()
        
        # Create a mock serial instance that we can control
        self.mock_serial_instance = MagicMock(spec=SerialInterface)
        
        # Patch the SerialInterface class to return our mock instance
        self.class_patcher = patch(
            "open_micro_stage.api.SerialInterface",
            return_value=self.mock_serial_instance
        )
        self.mock_serial_class = self.class_patcher.start()
        
        # Create the interface
        self.interface = OpenMicroStageInterface(
            show_communication=False, show_log_messages=False
        )

    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()
        self.class_patcher.stop()

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
        interface = OpenMicroStageInterface(
            show_communication=False, show_log_messages=False
        )
        self.assertFalse(interface.show_communication)
        self.assertFalse(interface.show_log_messages)

    def test_connect_success(self):
        """Test successful connection to device."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "v1.0.1",
        )
        
        self.interface.connect("/dev/ttyACM0")
        
        self.assertIsNotNone(self.interface.serial)
        self.mock_serial_class.assert_called_once_with(
            "/dev/ttyACM0",
            921600,
            log_msg_callback=self.interface.log_msg_callback,
            command_msg_callback=self.interface.command_msg_callback,
            unsolicited_msg_callback=self.interface.unsolicited_msg_callback,
        )

    def test_connect_with_custom_baud_rate(self):
        """Test connection with custom baud rate."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "v1.0.1",
        )
        
        self.interface.connect("/dev/ttyACM0", baud_rate=115200)
        
        self.mock_serial_class.assert_called_once_with(
            "/dev/ttyACM0",
            115200,
            log_msg_callback=self.interface.log_msg_callback,
            command_msg_callback=self.interface.command_msg_callback,
            unsolicited_msg_callback=self.interface.unsolicited_msg_callback,
        )

    def test_connect_incompatible_firmware(self):
        """Test connection fails with incompatible firmware version."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "v0.9.0",
        )
        
        self.interface.connect("/dev/ttyACM0")
        
        # Serial should be set to None on incompatible version
        self.assertIsNone(self.interface.serial)

    def test_disconnect(self):
        """Test disconnection from device."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "v1.0.1",
        )
        
        self.interface.connect("/dev/ttyACM0")
        self.interface.disconnect()
        
        self.mock_serial_instance.close.assert_called_once()
        self.assertIsNone(self.interface.serial)

    def test_disconnect_when_not_connected(self):
        """Test disconnect gracefully handles when not connected."""
        # Should not raise exception
        self.interface.disconnect()
        self.assertIsNone(self.interface.serial)

    def test_set_and_get_workspace_transform(self):
        """Test setting and getting workspace transform."""
        transform = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        
        self.interface.set_workspace_transform(transform)
        result = self.interface.get_workspace_transform()
        
        self.assertTrue(np.array_equal(result, transform))

    def test_read_firmware_version(self):
        """Test reading firmware version."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "v1.2.3",
        )
        self.interface.serial = self.mock_serial_instance
        
        major, minor, patch = self.interface.read_firmware_version()
        
        self.assertEqual(major, 1)
        self.assertEqual(minor, 2)
        self.assertEqual(patch, 3)
        self.mock_serial_instance.send_command.assert_called_with("M58")

    def test_read_firmware_version_error(self):
        """Test firmware version returns 0,0,0 on error."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.ERROR,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        major, minor, patch = self.interface.read_firmware_version()
        
        self.assertEqual((major, minor, patch), (0, 0, 0))

    def test_home_all_axes(self):
        """Test homing all axes."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.home()
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.mock_serial_instance.send_command.assert_called_with("G28 A B C D E F\n", 10)

    def test_home_specific_axes(self):
        """Test homing specific axes."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.home(axis_list=[0, 2])
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.mock_serial_instance.send_command.assert_called_with("G28 A C\n", 10)

    def test_home_invalid_axis(self):
        """Test homing with invalid axis index raises error."""
        self.interface.serial = self.mock_serial_instance
        
        with self.assertRaises(ValueError):
            self.interface.home(axis_list=[10])

    def test_calibrate_joint_no_save(self):
        """Test calibrating a joint without saving results."""
        calibration_response = "0.5,1.0,100\n1.0,2.0,200\n1.5,3.0,300\n"
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            calibration_response,
        )
        self.interface.serial = self.mock_serial_instance
        
        result, data = self.interface.calibrate_joint(0, save_result=False)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0], [0.5, 1.0, 1.5])  # motor angles
        self.assertEqual(data[1], [1.0, 2.0, 3.0])  # field angles
        self.assertEqual(data[2], [100, 200, 300])  # encoder counts
        self.mock_serial_instance.send_command.assert_called_with("M56 J0 P", 30)

    def test_calibrate_joint_with_save(self):
        """Test calibrating a joint with saving results."""
        calibration_response = "0.5,1.0,100\n"
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            calibration_response,
        )
        self.interface.serial = self.mock_serial_instance
        
        result, data = self.interface.calibrate_joint(1, save_result=True)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.mock_serial_instance.send_command.assert_called_with("M56 J1 P S", 30)

    def test_read_current_position(self):
        """Test reading current position."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "X10.5 Y20.3 Z15.8",
        )
        self.interface.serial = self.mock_serial_instance
        
        x, y, z = self.interface.read_current_position()
        
        self.assertAlmostEqual(x, 10.5)
        self.assertAlmostEqual(y, 20.3)
        self.assertAlmostEqual(z, 15.8)
        self.mock_serial_instance.send_command.assert_called_with("M50")

    def test_read_current_position_error(self):
        """Test reading current position returns None on error."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.ERROR,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        x, y, z = self.interface.read_current_position()
        
        self.assertIsNone(x)
        self.assertIsNone(y)
        self.assertIsNone(z)

    def test_read_current_position_invalid_format(self):
        """Test reading current position with invalid format raises error."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "invalid format",
        )
        self.interface.serial = self.mock_serial_instance
        
        with self.assertRaises(ValueError):
            self.interface.read_current_position()

    def test_move_to_immediate(self):
        """Test moving to position with immediate execution."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.move_to(5.0, 10.0, 15.0, f=20.0, move_immediately=True)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        call_args = self.mock_serial_instance.send_command.call_args
        self.assertIn("G0 X5.000000 Y10.000000 Z15.000000 F20.000", call_args[0][0])
        self.assertIn("I", call_args[0][0])

    def test_move_to_with_workspace_transform(self):
        """Test move_to applies workspace transform correctly."""
        # Set a simple translation transform
        transform = np.array([
            [1, 0, 0, 2],
            [0, 1, 0, 3],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ])
        self.interface.set_workspace_transform(transform)
        
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.move_to(0, 0, 0, f=10.0)
        
        # Expected transformed position is (2, 3, 4)
        call_args = self.mock_serial_instance.send_command.call_args
        cmd = call_args[0][0]
        self.assertIn("X2.000000", cmd)
        self.assertIn("Y3.000000", cmd)
        self.assertIn("Z4.000000", cmd)

    def test_move_to_blocking_busy_retry(self):
        """Test move_to retries on BUSY when blocking is True."""
        self.mock_serial_instance.send_command.side_effect = [
            (SerialInterface.ReplyStatus.BUSY, ""),
            (SerialInterface.ReplyStatus.OK, ""),
        ]
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.move_to(5.0, 10.0, 15.0, f=20.0, blocking=True)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(self.mock_serial_instance.send_command.call_count, 2)

    def test_move_to_non_blocking_returns_busy(self):
        """Test move_to returns BUSY immediately when blocking is False."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.BUSY,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.move_to(5.0, 10.0, 15.0, f=20.0, blocking=False)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.BUSY)
        self.assertEqual(self.mock_serial_instance.send_command.call_count, 1)

    def test_dwell(self):
        """Test dwell command."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.dwell(time_s=2.5, blocking=True)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        call_args = self.mock_serial_instance.send_command.call_args
        self.assertIn("G4 S2.500000", call_args[0][0])

    def test_set_max_acceleration(self):
        """Test setting max acceleration."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.set_max_acceleration(
            linear_accel=100.0, angular_accel=50.0
        )
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        call_args = self.mock_serial_instance.send_command.call_args
        self.assertIn("M204 L100.000000 A50.000000", call_args[0][0])

    def test_set_max_acceleration_minimum_values(self):
        """Test max acceleration enforces minimum values."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.set_max_acceleration(
            linear_accel=0.001, angular_accel=0.001
        )
        
        call_args = self.mock_serial_instance.send_command.call_args
        # Should be clamped to 0.01
        self.assertIn("M204 L0.010000 A0.010000", call_args[0][0])

    def test_wait_for_stop_ready(self):
        """Test wait_for_stop returns when device is ready."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "1",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.wait_for_stop()
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.mock_serial_instance.send_command.assert_called_with("M53\n")

    def test_wait_for_stop_polls_until_ready(self):
        """Test wait_for_stop polls until device is ready."""
        self.mock_serial_instance.send_command.side_effect = [
            (SerialInterface.ReplyStatus.OK, "0"),
            (SerialInterface.ReplyStatus.OK, "0"),
            (SerialInterface.ReplyStatus.OK, "1"),
        ]
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.wait_for_stop()
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(self.mock_serial_instance.send_command.call_count, 3)

    def test_wait_for_stop_error(self):
        """Test wait_for_stop returns error status."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.ERROR,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.wait_for_stop()
        
        self.assertEqual(result, SerialInterface.ReplyStatus.ERROR)

    def test_enable_motors(self):
        """Test enabling motors."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.enable_motors(enable=True)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.mock_serial_instance.send_command.assert_called_with("M17", timeout=5)

    def test_disable_motors(self):
        """Test disabling motors."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.enable_motors(enable=False)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.mock_serial_instance.send_command.assert_called_with("M18", timeout=5)

    def test_set_pose(self):
        """Test setting pose."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.set_pose(x=5.0, y=10.0, z=15.0)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        call_args = self.mock_serial_instance.send_command.call_args
        self.assertIn("G24 X5.000000 Y10.000000 Z15.000000", call_args[0][0])

    def test_send_custom_command(self):
        """Test sending custom command."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "response data",
        )
        self.interface.serial = self.mock_serial_instance
        
        result, response = self.interface.send_command("M57", timeout_s=3.0)
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(response, "response data")
        self.mock_serial_instance.send_command.assert_called_with("M57", 3.0)

    def test_read_device_state_info(self):
        """Test reading device state info."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "state info",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.read_device_state_info()
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.mock_serial_instance.send_command.assert_called_with("M57")

    def test_set_servo_parameter_defaults(self):
        """Test setting servo parameters with defaults."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.set_servo_parameter()
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        call_args = self.mock_serial_instance.send_command.call_args
        cmd = call_args[0][0]
        self.assertIn("M55", cmd)
        self.assertIn("A150.000000", cmd)  # pos_kp
        self.assertIn("B50000.000000", cmd)  # pos_ki
        self.assertIn("C0.200000", cmd)  # vel_kp
        self.assertIn("D100.000000", cmd)  # vel_ki
        self.assertIn("F0.002500", cmd)  # vel_filter_tc

    def test_set_servo_parameter_custom(self):
        """Test setting servo parameters with custom values."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.set_servo_parameter(
            pos_kp=200, pos_ki=60000, vel_kp=0.3, vel_ki=120, vel_filter_tc=0.003
        )
        
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        call_args = self.mock_serial_instance.send_command.call_args
        cmd = call_args[0][0]
        self.assertIn("A200.000000", cmd)
        self.assertIn("B60000.000000", cmd)
        self.assertIn("C0.300000", cmd)
        self.assertIn("D120.000000", cmd)
        self.assertIn("F0.003000", cmd)

    def test_read_encoder_angles(self):
        """Test reading encoder angles returns empty list."""
        self.mock_serial_instance.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "",
        )
        self.interface.serial = self.mock_serial_instance
        
        result = self.interface.read_encoder_angles()
        
        self.assertEqual(result, [])
        self.mock_serial_instance.send_command.assert_called_with("M51")

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
        # Mock return values for each command
        self.mock_serial_instance.send_command.side_effect = [
            (SerialInterface.ReplyStatus.OK, "v1.0.1"),  # firmware version
            (SerialInterface.ReplyStatus.OK, ""),  # home
            (SerialInterface.ReplyStatus.OK, ""),  # move_to
            (SerialInterface.ReplyStatus.OK, "1"),  # wait_for_stop
        ]
        
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
        
        self.mock_serial_instance.send_command.side_effect = [
            (SerialInterface.ReplyStatus.OK, calibration_data),  # calibrate
            (SerialInterface.ReplyStatus.OK, "X5.0 Y10.0 Z15.0"),  # read position
        ]
        
        self.interface.serial = self.mock_serial_instance
        
        result, data = self.interface.calibrate_joint(0, save_result=True)
        self.assertEqual(result, SerialInterface.ReplyStatus.OK)
        self.assertEqual(len(data), 3)
        
        x, y, z = self.interface.read_current_position()
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 10.0)
        self.assertAlmostEqual(z, 15.0)


if __name__ == "__main__":
    unittest.main()
