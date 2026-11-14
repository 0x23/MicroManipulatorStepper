# --------------------------------------------------------------------------------------
# Project: OpenMicroManipulator
# License: MIT (see LICENSE file for full description)
#          All text in here must be included in any redistribution.
# Author:  M. S. (diffraction limited)
# --------------------------------------------------------------------------------------

import os
import threading
import time
import re
from enum import Enum

import serial
import numpy as np
from colorama import Fore, Style, init
from serial.tools import list_ports

# --- SerialInterface --------------------------------------------------------------------------------------------------

class SerialInterface:

    class ReplyStatus(Enum):
        OK = 'ok'
        ERROR = 'error'
        TIMEOUT = 'timeout'
        BUSY = 'busy'

    class LogLevel(Enum):
        DEBUG = 'debug'
        INFO = 'info'
        WARNING = 'warning'
        ERROR = 'error'

    # Static mapping from prefix to LogLevel
    log_level_prefix_map = {
        "D)": LogLevel.DEBUG,
        "I)": LogLevel.INFO,
        "W)": LogLevel.WARNING,
        "E)": LogLevel.ERROR,
    }

    def __init__(self, port: str, baud_rate: int = 115200,
                 command_msg_callback=None,
                 log_msg_callback=None,
                 unsolicited_msg_callback=None,
                 reconnect_timeout: int = 5):
        """
        Initializes the serial connection and starts background reader.
        :param port: Serial port name (e.g., 'COM3' or '/dev/ttyUSB0').
        :param baud_rate: Serial baud rate.
        :param log_msg_callback: called when a log message is received
        :param unsolicited_msg_callback: Optional function to call with unsolicited messages.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.reconnect_timeout = reconnect_timeout
        self.serial = None  # initialized on connect

        self.command_msg_callback = command_msg_callback
        self.log_message_callback = log_msg_callback
        self.unsolicited_msg_callback = unsolicited_msg_callback

        # Synchronization for blocking send/receive
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._waiting_for_response = False
        self._response_string = ""
        self._response_status = None
        self._response_error_msg = None
        self._stop_event = threading.Event()
        self._reader_thread = None

        if self.connect(self.reconnect_timeout):
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            while not self._reader_thread.is_alive():
                time.sleep(0.001)


    def connect(self, timeout):
        """
        Try to open the serial port. Retry until timeout expires.
        """
        if self._stop_event.is_set():
            return False

        deadline = time.time() + timeout
        print(Fore.MAGENTA, end='')
        print(f"[SerialInterface] Connecting to port '{self.port}'...", end='')
        while time.time() < deadline:
            try:
                self.serial = serial.Serial(self.port, self.baud_rate, timeout=2)
                print(f" [OK]")
                print(Style.RESET_ALL, end='')
                return True
            except (serial.SerialException, OSError) as e:
                if getattr(e, 'errno', None) == 13:  # Permission denied
                    import os, sys
                    print(f"\n[SerialInterface] Permission denied on '{self.port}'.")
                    if sys.platform.startswith('linux'):
                        username = os.getenv('USER') or os.getenv('LOGNAME') or 'your_user'
                        print(f"[SerialInterface] Add your user to the dialout group and re-login: sudo usermod -aG dialout {username}")
                    print(Style.RESET_ALL, end='')
                    self.serial = None
                    return False
                print('.', end='')
                time.sleep(0.2)

        print(f" [FAILED] Timeout after {timeout} seconds.")
        print(f"[SerialInterface] Connection is permanently closed")
        print(Style.RESET_ALL, end='')
        self.serial = None
        return False

    def _reader_loop(self):
        """
        Asynchronous reader loop, collecting serial data into a buffer
        """
        buffer = ""
        while not self._stop_event.is_set():
            try:
                waiting = self.serial.in_waiting if self.serial is not None else 0
                if waiting:
                    # Drain the whole receive buffer in one read. Reading one byte at a
                    # time cannot keep up with the device and lets its USB TX buffer back
                    # up, which blocks the firmware's (core0) serial writes and stalls
                    # motion. Read everything available and split into lines here.
                    data = self.serial.read(waiting).decode('ascii', errors='ignore')
                    for char in data:
                        if char in ['\n', '\r']:
                            if len(buffer) > 0:
                                self._handle_line(buffer)
                                buffer = ""
                        else:
                            buffer += char
                else:
                    time.sleep(0.001)
            except (serial.SerialException, OSError) as e:
                if self._stop_event.is_set():
                    break

                print(Fore.MAGENTA+f"[SerialInterface] Lost connection: {e}"+Style.RESET_ALL)
                try:
                    if self.serial is not None and self.serial.is_open:
                        self.serial.close()
                except Exception:
                    pass

                self.serial = None
                self.connect(self.reconnect_timeout)

    def _handle_line(self, line: str):
        """
        Handles a single serial line sent by the device
        :param line: string containing a single line
        """
        with self._lock:
            log_level, log_msg = self._check_log_msg(line)
            # print(line)
            # log message
            if log_level is not None:
                if self.log_message_callback: self.log_message_callback(log_level, log_msg)
            # response
            elif self._waiting_for_response:
                line_lower = line.lower()
                if line_lower.startswith("ok"):
                    self._response_status = SerialInterface.ReplyStatus.OK
                elif line_lower.startswith("busy"):
                    self._response_status = SerialInterface.ReplyStatus.BUSY
                elif line_lower.startswith("error"):
                    self._response_status = SerialInterface.ReplyStatus.ERROR
                    parts = line.split(":", 1)
                    self._response_error_msg = parts[1].strip() if len(parts) > 1 else ""

                if self._response_status is not None:
                    self._condition.notify()
                else:
                    self._response_string += line + '\n'

            # unsolicited message
            else:
                if self.unsolicited_msg_callback: self.unsolicited_msg_callback(line)

    def _check_log_msg(self, msg: str):
        if len(msg) < 2:
            return None, ''
        return self.log_level_prefix_map.get(msg[:2]), msg[2:]

    def send_command(self, cmd: str, timeout=2) -> tuple[ReplyStatus, str]:
        """
        Sends a command and blocks until 'ok' or 'error' is received.
        :param cmd: The command to send.
        :param timeout: Maximum time to wait for response.
        :return: Tuple containing Status enum (OK | ERROR | TIMEOUT), and response lines.
        """
        with self._lock:
            if not self.serial or not self.serial.is_open:
                return SerialInterface.ReplyStatus.ERROR, 'Serial not open'

            # Reset state
            self._waiting_for_response = True
            self._response_string = ""
            self._response_error_msg = ""
            self._response_status = None

            cmd = (cmd.strip() + "\n")
            self.command_msg_callback(cmd, None, '')

            # Send command
            self.serial.write(cmd.encode('ascii'))
            self.serial.flush()

            # Wait for completion
            end_time = time.time() + timeout
            while self._response_status is None:
                remaining = end_time - time.time()
                if remaining <= 0:
                    self._waiting_for_response = False
                    print(Fore.MAGENTA + f"[SerialInterface] Command timeout, device didn't reply in time" + Style.RESET_ALL)
                    return SerialInterface.ReplyStatus.TIMEOUT, self._response_string
                self._condition.wait(timeout=remaining)

            self._waiting_for_response = False
            self.command_msg_callback(self._response_string, self._response_status, self._response_error_msg)
            return self._response_status, self._response_string

    def close(self):
        """Closes the serial port."""
        self._stop_event.set()
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.serial = None

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

# --- OpenMicroStageInterface ------------------------------------------------------------------------------------------

class OpenMicroStageInterface:
    # Mapping log levels to colors
    LOG_COLORS = {
        SerialInterface.LogLevel.DEBUG: Fore.WHITE+Style.DIM,
        SerialInterface.LogLevel.INFO: Style.RESET_ALL,
        SerialInterface.LogLevel.WARNING: Fore.YELLOW,
        SerialInterface.LogLevel.ERROR: Fore.RED,
    }

    def __init__(self, show_communication=True, show_log_messages=True):
        self.serial = None
        self.workspace_transform = np.eye(4)
        self.workspace_transform_inv = np.linalg.inv(self.workspace_transform)
        self.show_communication = show_communication
        self.show_log_messages = show_log_messages
        self.disable_message_callbacks = False

    @staticmethod
    def enumerate_devices():
        devices = []

        for port in sorted(list_ports.comports(), key=lambda item: item.device):
            description = (port.description or "").strip()
            device_name = os.path.basename(port.device)
            label = device_name
            if description and description.lower() != "n/a" and description != port.device:
                label = f"{device_name} - {description}"

            devices.append({
                "id": port.device,
                "label": label,
                "port": port.device,
            })

        return devices

    def connect(self, port: str, baud_rate: int = 921600):
        def version_to_str(v):
            return f"v{v[0]}.{v[1]}.{v[2]}"

        if self.serial is not None:
            self.disconnect()

        serial_interface = SerialInterface(port, baud_rate,
                                           log_msg_callback=self.log_msg_callback,
                                           command_msg_callback=self.command_msg_callback,
                                           unsolicited_msg_callback=self.unsolicited_msg_callback)
        if serial_interface.serial is None:
            serial_interface.close()
            return False

        self.serial = serial_interface

        self.disable_message_callbacks = True
        fw_version = self.read_firmware_version()
        min_fw_version = (1, 0, 1)
        print(Fore.MAGENTA + f"Firmware version: {version_to_str(fw_version)}" + Style.RESET_ALL)
        if fw_version < min_fw_version:
            print(Fore.MAGENTA + f"Firmware version {version_to_str(fw_version)} incompatible. "
                                 f"At least {version_to_str(min_fw_version)} required" + Style.RESET_ALL)
            self.serial.close()
            self.serial = None
            print('')
            self.disable_message_callbacks = False
            return False
        print('')
        self.disable_message_callbacks = False
        return True

    def disconnect(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None

    def is_connected(self):
        return self.serial is not None

    def log_msg_callback(self, log_level, msg):
        if not self.show_log_messages or self.disable_message_callbacks:
            return

        color = OpenMicroStageInterface.LOG_COLORS.get(log_level, Fore.WHITE)
        if log_level not in [SerialInterface.LogLevel.INFO, SerialInterface.LogLevel.DEBUG]:
            print(f"{color}[{log_level.name}] {msg}{Style.RESET_ALL}")
        else:
            print(f"{color}{msg}{Style.RESET_ALL}")

    def command_msg_callback(self, msg, reply_status: SerialInterface.ReplyStatus, error_msg: str):
        if not self.show_communication or self.disable_message_callbacks:
            return

        if reply_status is not None:
            if msg:
                msg = '\n'.join('> ' + line for line in msg.splitlines())
                print(f"{msg.rstrip()}")
            if error_msg:
                print(f"{Style.BRIGHT}{str(reply_status.name)}:{Style.RESET_ALL} {error_msg}\n")
            else:
                print(f"{Style.BRIGHT}{str(reply_status.name)} {Style.RESET_ALL}\n")
        else:
            print(f"{Fore.GREEN+Style.BRIGHT}{msg.rstrip()}{Style.RESET_ALL}")

    def unsolicited_msg_callback(self, msg):
        print(Fore.CYAN+msg+Style.RESET_ALL)
        pass

    def set_workspace_transform(self, transform):
        self.workspace_transform = transform
        self.workspace_transform_inv = np.linalg.inv(self.workspace_transform)

    def get_workspace_transform(self):
        return self.workspace_transform

    def read_firmware_version(self):
        ok, response = self.serial.send_command("M58")
        if ok != SerialInterface.ReplyStatus.OK or len(response) == 0:
            return 0, 0, 0

        match = re.match(r'v(\d+)\.(\d+)\.(\d+)', response)
        if match is None:
            return 0, 0, 0

        major, minor, patch = map(int, match.groups())
        return major,minor,patch

    def home(self, axis_list=None):
        """
        Homes one or more axes on the device
        :param axis_list: Optional list of axis indices to home. If None, all axes are homed.
        :return: The status of the command (e.g. OK, ERROR, TIMEOUT).
        """
        cmd = 'G28'
        axis_chars = ['A', 'B', 'C', 'D', 'E', 'F']
        if axis_list is None:
            axis_list = [i for i in range(len(axis_chars))]

        for axis_idx in axis_list:
            if 0 > axis_idx >= len(axis_chars):
                raise ValueError('Axis index out of range')
            cmd += ' '+axis_chars[axis_idx]

        res, msg = self.serial.send_command(cmd + "\n", 10)
        return res

    def calibrate_joint(self, joint_index: int, save_result: bool):
        """
        Calibrates the given joint and returns the measured data as three lists containing:
            data[0]: list of motor angles
            data[1]: list of electric field angles
            data[2]: list of raw encoder counts
        :param joint_index:
        :param save_result:
        :return:
        """
        cmd = f"M56 J{joint_index} P"
        if save_result: cmd += ' S'
        res, msg = self.serial.send_command(cmd, 30)

        calibration_data = self._parse_table_data(msg, 3)
        return res, calibration_data

    def move_to(self, x, y, z, f, move_immediately=False, blocking=True, timeout=1):
        """
        Moves the stage to an absolute position with a specified feed rate.
        The workspace transform is applied.
        :param x: Target X position (in workspace coordinates).
        :param y: Target Y position (in workspace coordinates).
        :param z: Target Z position (in workspace coordinates).
        :param f: Feed rate in mm/s.
        :param move_immediately: If True, execution starts without buffering delay.
        :param blocking: If True, waits and retries if the device is busy. If False, returns immediately on 'BUSY'.
        :param timeout: Timeout in seconds for each command attempt.
        :return: Status of the move command (e.g. OK, ERROR, BUSY, TIMEOUT).
        """
        # Convert to homogeneous vector
        transformed = self.workspace_transform @ np.array([x, y, z, 1.0])
        x_t, y_t, z_t = transformed[:3] / transformed[3]

        cmd = f"G0 X{x_t:.6f} Y{y_t:.6f} Z{z_t:.6f} F{f:.3f}"
        if move_immediately:
            cmd += " I"

        # resend messages if queue is full
        while True:
            res, msg = self.serial.send_command(cmd + "\n", timeout=timeout)
            if res != SerialInterface.ReplyStatus.BUSY or not blocking:
                return res

    def dwell(self, time_s, blocking, timeout=1):
        cmd = f"G4 S{time_s:.6f}\n"
        # resend messages if queue is full
        while True:
            res, msg = self.serial.send_command(cmd + "\n", timeout=timeout)
            if res != SerialInterface.ReplyStatus.BUSY or not blocking:
                return res

    def set_max_acceleration(self, linear_accel, angular_accel):
        linear_accel = max(linear_accel, 0.01)
        angular_accel = max(angular_accel, 0.01)
        cmd = f"M204 L{linear_accel:.6f} A{angular_accel:.6f}\n"
        res, msg = self.serial.send_command(cmd)
        return res

    def wait_for_stop(self, polling_interval_ms=50, disable_callbacks=True):
        disable_message_callbacks_prev = self.disable_message_callbacks
        if disable_callbacks: self.disable_message_callbacks = True

        while True:
            res, msg = self.serial.send_command("M53\n")
            if res != SerialInterface.ReplyStatus.OK: return res
            elif msg.strip() == "1":
                self.disable_message_callbacks = disable_message_callbacks_prev
                return SerialInterface.ReplyStatus.OK
            time.sleep(polling_interval_ms*0.001)

    def read_current_position(self, apply_inv_workspace_transform):
        """
        Reads the current position of the dives EXCLUDING the workspace transform.
        If you want to use the result with a
        """
        ok, response = self.serial.send_command("M50")
        if ok != SerialInterface.ReplyStatus.OK or len(response) == 0:
            return None, None, None

        # Match values with NO space between axis letter and number
        match = re.search(
            r"X([-+]?\d*\.?\d+)\s*Y([-+]?\d*\.?\d+)\s*Z([-+]?\d*\.?\d+)",
            response
        )
        if not match:
            raise ValueError(f"Invalid format: {response}")

        x, y, z = match.groups()
        if apply_inv_workspace_transform:
            transformed = self.workspace_transform_inv @ np.array([float(x), float(y), float(z), 1.0])
            x, y, z = transformed[:3] / transformed[3]

        return float(x), float(y), float(z)

    def read_encoder_angles(self):
        ok, response = self.serial.send_command("M51")
        if ok != SerialInterface.ReplyStatus.OK or len(response) == 0:
            return []
        return []

    def read_device_state_info(self):
        res, msg = self.serial.send_command("M57")
        return res, msg

    def set_servo_parameter(self, pos_kp=150, pos_ki=50000, vel_kp=0.2, vel_ki=100, vel_filter_tc=0.0025):
        cmd = f"M55 A{pos_kp:.6f} B{pos_ki:.6f} C{vel_kp:.6f} D{vel_ki:.6f} F{vel_filter_tc:.6f}"
        res, msg = self.serial.send_command(cmd)
        return res

    def enable_motors(self, enable: bool):
        cmd = f"M17" if enable else "M18"
        res, msg = self.serial.send_command(cmd, timeout=5)
        return res

    def set_pose(self, x: float, y: float, z: float):
        # Convert to homogeneous vector
        transformed = self.workspace_transform @ np.array([x, y, z, 1.0])
        x_t, y_t, z_t = transformed[:3] / transformed[3]

        cmd = f"G24 X{x_t:.6f} Y{y_t:.6f} Z{z_t:.6f}" # TODO: A, B ,C
        res, msg = self.serial.send_command(cmd)
        return res

    def set_tool_output(self, tool_idx: int, output_value: float, immediate: bool = True):
        # sets the output value for the specified tool
        cmd = f"M3 T{tool_idx} S{output_value}"
        res, msg = self.serial.send_command(cmd)

        # send dwell command to update tool value immediately
        if immediate:
            self.serial.send_command("G4 S0.001")

        return res

    def send_command(self, cmd: str, timeout_s: float=5):
        res, msg = self.serial.send_command(cmd, timeout_s)
        return res, msg

    @staticmethod
    def _parse_table_data(data_string, cols):
        # Parse the data
        data = [[] for _ in range(cols)]

        for line in data_string.strip().splitlines():
            parts = line.strip().split(',')
            if len(parts) != cols:
                continue  # skip malformed lines
            numbers = map(float, parts)
            for i, n in enumerate(numbers):
                data[i].append(n)

        return data
