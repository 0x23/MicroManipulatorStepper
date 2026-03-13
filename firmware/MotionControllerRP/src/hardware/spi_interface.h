// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

#include "hardware/spi.h"
#include "hardware/dma.h"
#include "hardware/gpio.h"

//*** CLASS *****************************************************************************

class SpiDma {
  public:
      SpiDma(spi_inst_t* spi, uint sck, uint mosi, uint miso, uint cs);

      void begin(uint32_t baudrate_hz);
      bool is_busy() const;
      void transfer(const void* tx_buf, void* rx_buf, size_t bytes);
      void wait_for_finish();
      void abort();

  private:
      spi_inst_t* spi;
      uint sck, mosi, miso, cs;

      int dma_tx;
      int dma_rx;
      dma_channel_config cfg_tx;
      dma_channel_config cfg_rx;
};



