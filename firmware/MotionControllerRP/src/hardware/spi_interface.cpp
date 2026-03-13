#include "spi_interface.h"

//*** CLASS *****************************************************************************

SpiDma::SpiDma(spi_inst_t* spi, uint sck, uint mosi, uint miso, uint cs)
        : spi(spi), sck(sck), mosi(mosi), miso(miso), cs(cs) 
{
}

void SpiDma::begin(uint32_t baudrate_hz) {
  // SPI init
  spi_init(spi, baudrate_hz);
  spi_set_format(spi, 8,
                  SPI_CPOL_0,
                  SPI_CPHA_0,
                  SPI_MSB_FIRST);

  gpio_set_function(sck, GPIO_FUNC_SPI);
  gpio_set_function(mosi, GPIO_FUNC_SPI);
  gpio_set_function(miso, GPIO_FUNC_SPI);

  // Manual CS
  gpio_init(cs);
  gpio_set_dir(cs, GPIO_OUT);
  gpio_put(cs, 1);

  // DMA TX channel
  dma_tx = dma_claim_unused_channel(true);
  cfg_tx = dma_channel_get_default_config(dma_tx);
  channel_config_set_transfer_data_size(&cfg_tx, DMA_SIZE_8);
  channel_config_set_dreq(&cfg_tx, spi_get_dreq(spi, true));
  channel_config_set_read_increment(&cfg_tx, true);
  channel_config_set_write_increment(&cfg_tx, false);

  // --- DMA RX channel ---
  dma_rx = dma_claim_unused_channel(true);
  cfg_rx = dma_channel_get_default_config(dma_rx);
  channel_config_set_transfer_data_size(&cfg_rx, DMA_SIZE_8);
  channel_config_set_dreq(&cfg_rx, spi_get_dreq(spi, false));
  channel_config_set_read_increment(&cfg_rx, false);
  channel_config_set_write_increment(&cfg_rx, true);

  // Configure but don't start yet
  dma_channel_configure(
      dma_tx, &cfg_tx,
      &spi_get_hw(spi)->dr,
      nullptr, 0, false);

  dma_channel_configure(
      dma_rx, &cfg_rx,
      nullptr,
      &spi_get_hw(spi)->dr,
      0, false);
}

bool SpiDma::is_busy() const {
    return dma_channel_is_busy(dma_tx) ||
           dma_channel_is_busy(dma_rx);
}

void SpiDma::transfer(const void* tx_buf, void* rx_buf, size_t bytes) {
    while (is_busy()) {}

    gpio_put(cs, 0);

    // Ensure minimum CS low time (50 ns) - Approx 14 cycles at 250 MHz
    for (int i = 0; i < 14; i++) __asm volatile("nop");

    // Setup RX DMA - must be set first!
    if (rx_buf != nullptr) {
        dma_channel_set_write_addr(dma_rx, rx_buf, false);
        dma_channel_set_trans_count(dma_rx, bytes, false);
    }

    // Setup TX DMA
    if (tx_buf != nullptr) {
        dma_channel_set_read_addr(dma_tx, tx_buf, false);
        dma_channel_set_trans_count(dma_tx, bytes, true);
    }
}

void SpiDma::wait_for_finish() {
  while (is_busy()) {}
  gpio_put(cs, 1);

  // Ensure minimum CS low time (50 ns) - Approx 14 cycles at 250 MHz
  for (int i = 0; i < 14; i++) __asm volatile("nop");
}

void SpiDma::abort() {
    // Stop TX DMA if running
    if (dma_channel_is_busy(dma_tx)) {
        dma_channel_abort(dma_tx);
    }

    // Stop RX DMA if running
    if (dma_channel_is_busy(dma_rx)) {
        dma_channel_abort(dma_rx);
    }

    // Flush the SPI FIFOs to prevent leftover data
    spi_get_hw(spi)->dr = 0;       // clear DR (TX FIFO)
    spi_get_hw(spi)->sr;           // read SR to clear flags

    // Deassert CS pin to leave SPI slave in idle state
    gpio_put(cs, 1);
}




