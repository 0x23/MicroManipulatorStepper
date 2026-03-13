#include "pwm_tool.h"
#include "hardware/clocks.h"
#include "hardware/pio.h"
#include "utilities/logging.h"

//--- PIO -----------------------------------------------------------------------------------------

/*
.program pwm
.side_set 1 opt

    pull noblock           ; Pull from FIFO to OSR if available, else copy X to OSR.
    mov x, osr             ; Copy most-recently-pulled value back to scratch X
    mov y, isr             ; ISR contains PWM period. Y used as counter.
    
    jmp x!=y set_output_off
    jmp compare_end
set_output_off:
    nop side 0
compare_end:

countloop:
    jmp x!=y noset         ; Set pin high if X == Y, keep the two paths length matched
    jmp skip        side 1
noset:
    nop                    ; Single dummy cycle to keep the two paths the same length
skip:
    jmp y-- countloop      ; Loop until Y hits 0, then pull a fresh PWM value from FIFO

% c-sdk {
static inline void pwm_program_init(PIO pio, uint sm, uint offset, uint pin) {
   pio_gpio_init(pio, pin);
   pio_sm_set_consecutive_pindirs(pio, sm, pin, 1, true);
   pio_sm_config c = pwm_program_get_default_config(offset);
   sm_config_set_sideset_pins(&c, pin);
   pio_sm_init(pio, sm, offset, &c);
}
%}
*/

#define pwm_wrap_target 0
#define pwm_wrap 9

static const uint16_t pwm_program_instructions[] = {
            //     .wrap_target
    0x8080, //  0: pull   noblock                    
    0xa027, //  1: mov    x, osr                     
    0xa046, //  2: mov    y, isr                     
    0x00a5, //  3: jmp    x != y, 5                  
    0x0006, //  4: jmp    6                          
    0xb042, //  5: nop                    side 0     
    0x00a8, //  6: jmp    x != y, 8                  
    0x1809, //  7: jmp    9               side 1     
    0xa042, //  8: nop                               
    0x0086, //  9: jmp    y--, 6                     
            //     .wrap
};

static const struct pio_program pwm_program = {
    .instructions = pwm_program_instructions,
    .length = 10,
    .origin = -1,
};

// Helper to set the PWM period (the 'resolution') via the ISR
void pio_pwm_set_period(PIO pio, uint sm, uint32_t period) {
    pio_sm_set_enabled(pio, sm, false);
    pio_sm_put_blocking(pio, sm, period);
    // Pull the period from FIFO to OSR, then move OSR to ISR
    pio_sm_exec(pio, sm, pio_encode_pull(false, false));
    pio_sm_exec(pio, sm, pio_encode_out(pio_isr, 32));
    pio_sm_set_enabled(pio, sm, true);
}

static inline pio_sm_config pwm_program_get_default_config(uint offset) {
    pio_sm_config c = pio_get_default_sm_config();
    sm_config_set_wrap(&c, offset + pwm_wrap_target, offset + pwm_wrap);
    sm_config_set_sideset(&c, 2, true, false);
    return c;
}

static inline void pwm_program_init(PIO pio, uint sm, uint offset, uint pin, float clk_div) {
  pio_gpio_init(pio, pin);
  pio_sm_set_consecutive_pindirs(pio, sm, pin, 1, true);
  pio_sm_config c = pwm_program_get_default_config(offset);
  sm_config_set_sideset_pins(&c, pin);
  sm_config_set_clkdiv(&c, clk_div); // added
  pio_sm_init(pio, sm, offset, &c);
}

//--- PwmTool -------------------------------------------------------------------------------------

void PwmTool::init(int output_pin, int frequency, int resolution_bits) {
  PwmTool::pio = pio0;
    
  // calculate max value
  max_pwm = (1u << resolution_bits) - 1;
  this->output_pin = output_pin;

  // load program and claim state machine
  uint offset = pio_add_program(pio, &pwm_program);
  state_machine = pio_claim_unused_sm(pio, true);
  if(state_machine < 0)
    LOG_ERROR("PwmTool::init: No free PIO statemachine available");

  // compute clock divisor
  float sysclk = (float)clock_get_hz(clk_sys);
  float clkdiv = sysclk / (frequency * (max_pwm + 1) * 2);

  // initialize program
  pwm_program_init(pio, state_machine, offset, output_pin, clkdiv);
  
  // subtract -1 here since the PIO loop includes start value AND 0
  pio_pwm_set_period(pio, state_machine, max_pwm-1);
  set_value(0.0f);
}

void PwmTool::set_value(float v) {
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;

    // Scale 0.0-1.0 to 0-max_pwm
    uint32_t duty = (uint32_t)(v * max_pwm);

    // subtract -1 to match decresed period - the underflow results in correct
    // contnuous off state for 0 duty cycle 
    duty--;

    // The PIO program pulls from the TX FIFO
    pio_sm_put_blocking(pio, state_machine, duty);
}