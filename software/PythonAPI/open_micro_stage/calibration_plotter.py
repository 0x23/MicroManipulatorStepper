from open_micro_stage import OpenMicroStageInterface
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

def plot_calibration_data(ax_encoder_counts, ax_field_angel, label, data):
    # Plot on the provided Axes object
    if ax_encoder_counts is not None:
        ax_encoder_counts.plot(data[0], data[2], label=label)
        ax_encoder_counts.set_xlabel('Motor Angle [rad]')
        ax_encoder_counts.set_ylabel('Encoder Counts Raw')
        ax_encoder_counts.set_title('Encoder Count Plot')
        ax_encoder_counts.legend()
        ax_encoder_counts.grid(True)

    # Plot on the provided Axes object
    if ax_field_angel is not None:
        ax_field_angel.plot(data[0], data[1], label=label)
        ax_field_angel.set_xlabel('Motor Angle [rad]')
        ax_field_angel.set_ylabel('Motor Field Angle [rad]')
        ax_field_angel.set_title('Field Angle Plot')
        ax_field_angel.legend()
        ax_field_angel.grid(True)

def main():
    # create interface and connect
    oms = OpenMicroStageInterface(show_communication=True, show_log_messages=True)
    oms.connect('/dev/ttyACM0')

    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), sharex='all')

    for i in range(3):
        res, data = oms.calibrate_joint(i, save_result=False)
        plot_calibration_data(ax, None, f'Actuator {i}', data)

    # Adjust layout and show
    plt.tight_layout()
    plt.show()