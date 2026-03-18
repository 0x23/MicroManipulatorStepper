// --------------------------------------------------------------------------------------
// This code is a modified version of the MT6835 Driver from the SimpleFOC project
// being distributed under the MIT liscence as well. Thank you SimpleFOC !
// --------------------------------------------------------------------------------------

#include "MT6835_encoder.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include "pico/stdlib.h"
#include "utilities/logging.h"

void MT6835Encoder::setup_spi(spi_inst_t* spi, uint pin_sck, uint pin_mosi, uint pin_miso, int32_t baudrate_hz) {
  // Set GPIO functions to SPI
  gpio_set_function(pin_sck, GPIO_FUNC_SPI);
  gpio_set_function(pin_mosi, GPIO_FUNC_SPI);
  gpio_set_function(pin_miso, GPIO_FUNC_SPI);
  
  // SPI format: 8 bits, mode 3 (CPOL=1, CPHA=1)
  spi_init(spi, baudrate_hz);
  spi_set_format(spi, 8, SPI_CPOL_1, SPI_CPHA_1, SPI_MSB_FIRST);

  sleep_ms(10);
}

MT6835Encoder::MT6835Encoder(spi_inst_t* spi, int32_t cs_pin) : spi(spi), cs_pin(cs_pin) {
  if (cs_pin >= 0) {
      gpio_init(cs_pin);
      gpio_set_dir(cs_pin, GPIO_OUT);
      gpio_put(cs_pin, 1);  // CS high
  }
}

MT6835Encoder::~MT6835Encoder() {
  // nop
}

bool MT6835Encoder::init(uint8_t bandwidth, uint8_t hysteresis) {
  sleep_ms(100);

  if(is_connected() == false)
    return false;

  set_rotation_direction(0);    // needs to be set, otherwise might be random
  set_bandwidth(bandwidth);
  set_hysteresis(hysteresis);

  last_raw_angle = 0;
  abs_raw_angle = 0;
  crc_error_count = 0;
  initialized = true;

  return true;
}

bool MT6835Encoder::is_connected() {
  uint8_t check_bytes[] = {37, 109, 179, 251, 1};
  for(int i=0; i<sizeof(check_bytes); i++) {
    write_register(MT6835_REG_USERID, check_bytes[i]);
    uint8_t user_id = read_register(MT6835_REG_USERID);
    if(user_id != check_bytes[i])
      return false;
  }

  return true;
}

bool MT6835Encoder::is_initialized() {
  return initialized;
}

void MT6835Encoder::reset_abs_angle(int32_t abs_raw_angle) {
  MT6835Encoder::abs_raw_angle = abs_raw_angle;
}

void MT6835Encoder::reset_abs_angle_period() {
  abs_raw_angle %= MT6835_CPR;
  if (abs_raw_angle < 0)
    abs_raw_angle += MT6835_CPR;
}

float MT6835Encoder::read_abs_angle() {
  int32_t raw_angle = read_abs_angle_raw();
  return raw_angle * RAW_TO_ANGLE;
}

MT6835Encoder::AbsRawAngleType MT6835Encoder::read_abs_angle_raw() {
  uint8_t data[6] = {0};
  data[0] = MT6835_OP_ANGLE << 4;
  data[1] = MT6835_REG_ANGLE1;
  // rest zero

  spi_begin_transaction();
  spi_transfer(data, 6);
  spi_end_transaction();

  last_status = data[4] & 0x07;
  last_crc = data[5];
  int32_t raw_angle = ((int32_t)data[2] << 13) | ((int32_t)data[3] << 5) | (data[4] >> 3);
  
  if (check_crc) {
      if (last_crc != calc_crc(raw_angle, last_status)) {
          last_status |= MT6835_CRC_ERROR;
          crc_error_count++;
          // LOG_ERROR("chip_crc: %i - calc_crc: %i", last_crc, calc_crc(raw_angle, last_status));
     //     return -1.0f; // CRC error indicator
      }
  }

  return update_abs_raw_angle(raw_angle);
}

MT6835Encoder::AbsRawAngleType MT6835Encoder::get_last_abs_raw_angle() const {
  return abs_raw_angle;
}

float MT6835Encoder::get_last_abs_angle() const {
  return abs_raw_angle * RAW_TO_ANGLE;
}

int32_t MT6835Encoder::get_rawcounts_per_rev() {
  return MT6835_CPR;
}

uint8_t MT6835Encoder::get_status() {
    return last_status;
}

void MT6835Encoder::set_crc_enabled(bool enable) {
  check_crc = enable;
}

bool MT6835Encoder::is_crc_enabled() {
  return check_crc;
}

uint32_t MT6835Encoder::get_crc_error_count(bool reset) {
  uint32_t result = crc_error_count;
  if(reset)
    crc_error_count = 0;
  return result;
}

uint8_t MT6835Encoder::get_calibration_status() {
    uint8_t data[3] = {0};
    data[0] = (MT6835_OP_READ << 4) | (MT6835_REG_CAL_STATUS >> 8);
    data[1] = MT6835_REG_CAL_STATUS & 0xFF;

    spi_begin_transaction();
    spi_transfer(data, 3);
    spi_end_transaction();

    return data[2] >> 6;
}

bool MT6835Encoder::set_zero_from_current_position() {
    MT6835Command cmd{};
    cmd.cmd = MT6835_OP_ZERO;
    cmd.addr = 0x000;
    transfer_24(&cmd);
    abs_raw_angle = 0;
    last_raw_angle = 0;
    return cmd.data == MT6835_WRITE_ACK;
}

bool MT6835Encoder::write_eeprom() {
    sleep_ms(1); // wait at least 1 ms
    MT6835Command cmd{};
    cmd.cmd = MT6835_OP_PROG;
    cmd.addr = 0x000;
    transfer_24(&cmd);
    return cmd.data == MT6835_WRITE_ACK;
}

uint8_t MT6835Encoder::get_bandwidth() {
    MT6835Options5 opts{ .reg = read_register(MT6835_REG_OPTS5) };
    return opts.bw;
}
void MT6835Encoder::set_bandwidth(uint8_t bw) {
    MT6835Options5 opts{ .reg = read_register(MT6835_REG_OPTS5) };
    opts.bw = bw;
    write_register(MT6835_REG_OPTS5, opts.reg);
}

uint8_t MT6835Encoder::get_hysteresis() {
    MT6835Options3 opts{ .reg = get_options3().reg };
    return opts.hyst;
}
void MT6835Encoder::set_hysteresis(uint8_t hyst) {
    MT6835Options3 opts{ .reg = get_options3().reg };
    opts.hyst = hyst;
    set_options3(opts);
}

uint8_t MT6835Encoder::get_rotation_direction() {
    MT6835Options3 opts{ .reg = get_options3().reg };
    return opts.rot_dir;
}
void MT6835Encoder::set_rotation_direction(uint8_t dir) {
    MT6835Options3 opts{ .reg = get_options3().reg };
    opts.rot_dir = dir;
    set_options3(opts);
}

uint16_t MT6835Encoder::get_abz_resolution() {
    uint8_t hi = read_register(MT6835_REG_ABZ_RES1);
    MT6835ABZRes lo{ .reg = read_register(MT6835_REG_ABZ_RES2) };
    return (hi << 6) | lo.abz_res_low;
}
void MT6835Encoder::set_abz_resolution(uint16_t res) {
    uint8_t hi = (res >> 6);
    MT6835ABZRes lo{ .reg = read_register(MT6835_REG_ABZ_RES2) };
    lo.abz_res_low = (res & 0x3F);
    write_register(MT6835_REG_ABZ_RES1, hi);
    write_register(MT6835_REG_ABZ_RES2, lo.reg);
}

bool MT6835Encoder::is_abz_enabled() {
    MT6835ABZRes lo{ .reg = read_register(MT6835_REG_ABZ_RES2) };
    return lo.abz_off == 0;
}
void MT6835Encoder::set_abz_enabled(bool enabled) {
    MT6835ABZRes lo{ .reg = read_register(MT6835_REG_ABZ_RES2) };
    lo.abz_off = enabled ? 0 : 1;
    write_register(MT6835_REG_ABZ_RES2, lo.reg);
}

bool MT6835Encoder::is_ab_swapped() {
    MT6835ABZRes lo{ .reg = read_register(MT6835_REG_ABZ_RES2) };
    return lo.ab_swap == 1;
}
void MT6835Encoder::set_ab_swapped(bool swapped) {
    MT6835ABZRes lo{ .reg = read_register(MT6835_REG_ABZ_RES2) };
    lo.ab_swap = swapped ? 1 : 0;
    write_register(MT6835_REG_ABZ_RES2, lo.reg);
}

uint16_t MT6835Encoder::get_zero_position() {
    uint8_t hi = read_register(MT6835_REG_ZERO1);
    MT6835Options0 lo{ .reg = read_register(MT6835_REG_ZERO2) };
    return (hi << 4) | lo.zero_pos_low;
}
void MT6835Encoder::set_zero_position(uint16_t pos) {
    uint8_t hi = (pos >> 4);
    MT6835Options0 lo{ .reg = read_register(MT6835_REG_ZERO2) };
    lo.zero_pos_low = pos & 0x0F;
    write_register(MT6835_REG_ZERO1, hi);
    write_register(MT6835_REG_ZERO2, lo.reg);
}

MT6835Options1 MT6835Encoder::get_options1() {
    MT6835Options1 result{ .reg = read_register(MT6835_REG_OPTS1) };
    return result;
}
void MT6835Encoder::set_options1(MT6835Options1 opts) {
    write_register(MT6835_REG_OPTS1, opts.reg);
}

MT6835Options2 MT6835Encoder::get_options2() {
    MT6835Options2 result{ .reg = read_register(MT6835_REG_OPTS2) };
    return result;
}
void MT6835Encoder::set_options2(MT6835Options2 opts) {
    MT6835Options2 val = get_options2();
    val.nlc_en = opts.nlc_en;
    val.pwm_fq = opts.pwm_fq;
    val.pwm_pol = opts.pwm_pol;
    val.pwm_sel = opts.pwm_sel;
    write_register(MT6835_REG_OPTS2, val.reg);
}

MT6835Options3 MT6835Encoder::get_options3() {
    MT6835Options3 result{ .reg = read_register(MT6835_REG_OPTS3) };
    return result;
}
void MT6835Encoder::set_options3(MT6835Options3 opts) {
    MT6835Options3 val = get_options3();
    val.rot_dir = opts.rot_dir;
    val.hyst = opts.hyst;
    write_register(MT6835_REG_OPTS3, val.reg);
}

MT6835Options4 MT6835Encoder::get_options4() {
    MT6835Options4 result{ .reg = read_register(MT6835_REG_OPTS4) };
    return result;
}
void MT6835Encoder::set_options4(MT6835Options4 opts) {
    MT6835Options4 val = get_options4();
    val.gpio_ds = opts.gpio_ds;
    val.autocal_freq = opts.autocal_freq;
    write_register(MT6835_REG_OPTS4, val.reg);
}

static inline uint32_t swap_bytes(uint32_t val) {
    return __builtin_bswap32(val);
}

void MT6835Encoder::transfer_24(MT6835Command* cmd) {
    uint32_t buff = swap_bytes(cmd->val);
    spi_begin_transaction();
    spi_transfer((uint8_t*)&buff, 3);
    spi_end_transaction();
    cmd->val = swap_bytes(buff);
}

uint8_t MT6835Encoder::read_register(uint16_t reg) {
    MT6835Command cmd{};
    cmd.cmd = MT6835_OP_READ;
    cmd.addr = reg;
    transfer_24(&cmd);
    return cmd.data;
}

bool MT6835Encoder::write_register(uint16_t reg, uint8_t value) {
    MT6835Command cmd{};
    cmd.cmd = MT6835_OP_WRITE;
    cmd.addr = reg;
    cmd.data = value;
    transfer_24(&cmd);
    return cmd.data == MT6835_WRITE_ACK;
}

MT6835Encoder::AbsRawAngleType MT6835Encoder::update_abs_raw_angle(int32_t raw_angle) {
    if(raw_angle >= 0) { 
      int32_t half_max = MT6835_CPR>>1;

      int32_t d = raw_angle - last_raw_angle;
      if (d > half_max) d -= MT6835_CPR;
      else if (d < -half_max) d += MT6835_CPR;

      abs_raw_angle += d;
      last_raw_angle = raw_angle;
    }

    return abs_raw_angle;
}

// Helper SPI transaction helpers for CS handling and SPI transfer
void MT6835Encoder::spi_begin_transaction() {
    // No real beginTransaction in Pico SDK; just pull CS low if used
    if (cs_pin >= 0)
        gpio_put(cs_pin, 0);
}

void MT6835Encoder::spi_transfer(uint8_t* data, size_t length) {
    // Full-duplex transfer, sending and receiving on SPI
    spi_write_read_blocking(spi, data, data, length);
}

void MT6835Encoder::spi_end_transaction() {
    if (cs_pin >= 0)
        gpio_put(cs_pin, 1);
}

uint8_t MT6835Encoder::calc_crc(uint32_t angle, uint8_t status) {
    uint8_t crc = 0x00;
    uint8_t input;

    input = (angle >> 13) & 0xFF;
    crc ^= input;
    for (int k = 8; k > 0; k--)
        crc = (crc & 0x80) ? (crc << 1) ^ 0x07 : crc << 1;

    input = (angle >> 5) & 0xFF;
    crc ^= input;
    for (int k = 8; k > 0; k--)
        crc = (crc & 0x80) ? (crc << 1) ^ 0x07 : crc << 1;

    input = ((angle & 0x1F) << 3) | (status & 0x07);
    crc ^= input;
    for (int k = 8; k > 0; k--)
        crc = (crc & 0x80) ? (crc << 1) ^ 0x07 : crc << 1;

    return crc;
}