// --------------------------------------------------------------------------------------
// This code is a modified version of the MT6835 Driver from the SimpleFOC project
// being distributed under the MIT liscence as well. Thank you SimpleFOC !
// --------------------------------------------------------------------------------------

#pragma once

#include <stdint.h>
#include "hardware/spi.h"
#include "hardware/gpio.h"

#define MT6835_CPR (1<<21)

#define MT6835_OP_READ  0b0011
#define MT6835_OP_WRITE 0b0110
#define MT6835_OP_PROG  0b1100
#define MT6835_OP_ZERO  0b0101
#define MT6835_OP_ANGLE 0b1010

#define MT6835_CMD_MASK  0b111100000000000000000000
#define MT6835_ADDR_MASK 0b000011111111111100000000
#define MT6835_DATA_MASK 0b000000000000000011111111

#define MT6835_STATUS_OVERSPEED 0x01
#define MT6835_STATUS_WEAKFIELD 0x02
#define MT6835_STATUS_UNDERVOLT 0x04
#define MT6835_CRC_ERROR 0x08

#define MT6835_WRITE_ACK 0x55

#define MT6835_REG_USERID 0x001

#define MT6835_REG_ANGLE1 0x003
#define MT6835_REG_ANGLE2 0x004
#define MT6835_REG_ANGLE3 0x005
#define MT6835_REG_ANGLE4 0x006

#define MT6835_REG_ABZ_RES1 0x007
#define MT6835_REG_ABZ_RES2 0x008

#define MT6835_REG_ZERO1 0x009
#define MT6835_REG_ZERO2 0x00A

#define MT6835_REG_OPTS0 0x00A
#define MT6835_REG_OPTS1 0x00B
#define MT6835_REG_OPTS2 0x00C
#define MT6835_REG_OPTS3 0x00D
#define MT6835_REG_OPTS4 0x00E
#define MT6835_REG_OPTS5 0x011

// NLC table, 192 bytes
#define MT6835_REG_NLC_BASE 0x013
#define MT6835_REG_CAL_STATUS 0x113

//*** DATATYPES **********************************************************************************/

union MT6835ABZRes {
	struct {
		uint8_t ab_swap:1;
		uint8_t abz_off:1;
		uint8_t abz_res_low:6;
	};
	uint8_t reg;
};

union MT6835Options0 {
	struct {
		uint8_t z_pul_wid:3;
		uint8_t z_edge:1;
		uint8_t zero_pos_low:4;
	};
	uint8_t reg;
};

union MT6835Options1 {
	struct {
		uint8_t uvw_res:4;
		uint8_t uvw_off:1;
		uint8_t uvw_mux:1;
		uint8_t z_phase:2;
	};
	uint8_t reg;
};

union MT6835Options2 {
	struct {
		uint8_t pwm_sel:3;
		uint8_t pwm_pol:1;
		uint8_t pwm_fq:1;
		uint8_t nlc_en:1;
		uint8_t reserved:2;
	};
	uint8_t reg;
};

union MT6835Options3 {
	struct {
		uint8_t hyst:3;
		uint8_t rot_dir:1;
		uint8_t reserved:4;
	};
	uint8_t reg;
};

union MT6835Options4 {
	struct {
		uint8_t reserved:4;
		uint8_t autocal_freq:3;
		uint8_t gpio_ds:1;
	};
	uint8_t reg;
};

union MT6835Options5 {
	struct {
		uint8_t bw:3;
		uint8_t reserved:5;
	};
	uint8_t reg;
};

union MT6835Command {
	struct {
		uint32_t unused:8;
		uint32_t data:8;
		uint32_t addr:12;
		uint32_t cmd:4;
	};
	uint32_t val;
};

//*** CLASS ************************************************************************************/

class MT6835Encoder {
  public:
    typedef int32_t AbsRawAngleType;

  public:
      static constexpr float RAW_TO_ANGLE = (2.0f*3.14159265358979323846f)/float(MT6835_CPR);

      // use this to setup a HW spi. It can then be used by multiple instances of MT6835
      static void setup_spi(spi_inst_t* spi, uint pin_sck, uint pin_mosi, uint pin_miso, int32_t baudrate_hz);

      // Constructor: pass SPI instance (spi0 or spi1), CS pin
      MT6835Encoder(spi_inst_t *spi, int32_t cs_pin);
      virtual ~MT6835Encoder();

      bool init(uint8_t bandwidth=0x5, uint8_t hysteresis=0x4);
      bool is_connected();
      bool is_initialized();

      void reset_abs_angle(int32_t abs_raw_angle=0);  // resets the total revolutions of abs angle
      void reset_abs_angle_period();                  // Brings abs angle into [0..2pi)
      float read_abs_angle();                         // returns the absolute angle in radians
      float get_last_abs_angle() const;               // returns the last read abs angle
      AbsRawAngleType read_abs_angle_raw();           // returns the absolute angle in raw counts
      AbsRawAngleType get_last_abs_raw_angle() const; // returns the last read abs raw angle

      int32_t get_rawcounts_per_rev();                // returns the number of raw counts per revolution

      uint8_t get_bandwidth();
      void set_bandwidth(uint8_t bw);

      uint8_t get_hysteresis();
      void set_hysteresis(uint8_t hyst);

      uint8_t get_rotation_direction();
      void set_rotation_direction(uint8_t dir);

      uint16_t get_abz_resolution();
      void set_abz_resolution(uint16_t res);

      bool is_abz_enabled();
      void set_abz_enabled(bool enabled);

      bool is_ab_swapped();
      void set_ab_swapped(bool swapped);

      uint16_t get_zero_position();
      void set_zero_position(uint16_t pos);

      MT6835Options1 get_options1();
      void set_options1(MT6835Options1 opts);

      MT6835Options2 get_options2();
      void set_options2(MT6835Options2 opts);

      MT6835Options3 get_options3();
      void set_options3(MT6835Options3 opts);

      MT6835Options4 get_options4();
      void set_options4(MT6835Options4 opts);

      uint8_t get_status();
      void set_crc_enabled(bool enable);
      bool is_crc_enabled();
      uint32_t get_crc_error_count(bool reset=false);

      uint8_t get_calibration_status();

      bool set_zero_from_current_position();
      bool write_eeprom();  // takes ~6s to complete after calling

      bool check_crc = false;

  private:
      bool initialized=false;
      spi_inst_t *spi;
      int32_t cs_pin;
      uint8_t last_status = 0;
      uint8_t last_crc = 0;
      uint32_t crc_error_count = 0;

      int32_t last_raw_angle = 0;
      AbsRawAngleType abs_raw_angle = 0;

      AbsRawAngleType update_abs_raw_angle(int32_t raw_angle);

      void spi_begin_transaction();
      void spi_transfer(uint8_t* data, size_t length);
      void spi_end_transaction();

      void transfer_24(MT6835Command *out_value);
      uint8_t read_register(uint16_t reg);
      bool write_register(uint16_t reg, uint8_t value);
      uint8_t calc_crc(uint32_t angle, uint8_t status);
};