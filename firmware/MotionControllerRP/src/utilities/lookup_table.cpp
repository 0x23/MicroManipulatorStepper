// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#include <pico/stdlib.h>
#include <LittleFS.h> 
#include <algorithm> 

#include "lookup_table.h"
#include "utilities/logging.h"
#include "utilities/math_constants.h" 

//*** FUNCTIONS *************************************************************************

bool almost_equal(float a, float b, float rel_tol = 1e-6f, float abs_tol = 1e-6f) {
    return std::fabs(a - b) <= std::max(rel_tol * std::max(std::fabs(a), std::fabs(b)), abs_tol);
}

//*** CLASS *****************************************************************************

bool LookupTable::init(int32_t table_size, float input_min, float input_max) {
  lookup_table.clear();
  lookup_table.resize(table_size, 0.0f);
  if(input_min>=input_max) {
    LOG_ERROR("LookupTable: input_max must be larger than input_min");
    return false;
  }

  LookupTable::input_min = input_min;
  LookupTable::input_max = input_max;
  LookupTable::one_over_input_range = 1.0f/(input_max-input_min);

  return true;
}

/*
bool LookupTable::init_approximating(std::vector<std::pair<float, float>> in_out_pairs, int table_size, float sigma) {
    const int N = int(in_out_pairs.size());
    if (N < 2 || table_size < 2)
        return false;

    // Sort by x ascending
    std::sort(in_out_pairs.begin(), in_out_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    input_min = in_out_pairs.front().first;
    input_max = in_out_pairs.back().first;
    if (input_min >= input_max)
        return false;

    one_over_input_range = 1.0f / (input_max - input_min);
    lookup_table.resize(table_size, 0.0f);

    int start = 0;
    int end = 0;
    const float hw_size = (sigma*3.0f)*0.5f;
    const float two_sigma_sq = 2.0f * powf(sigma, 2.0f);

    for (int i = 0; i < table_size; ++i) {
      // compute current evaluation position
      float t = static_cast<float>(i) / (table_size - 1);
      float x = input_min + t * (input_max - input_min);

      // Advance window start and end index
      float x_min = x - hw_size;
      while (start < N - 1 && in_out_pairs[start].first < x_min) start++;
      float x_max = x + hw_size;
      while (end < N - 1 && in_out_pairs[end].first < x_max) end++;

      // compute weighted average in window range
      float sum = 0.0f;
      float wcount = 0;
      for (int idx = start; idx <= end; idx++) {
        float dx = in_out_pairs[idx].first - x;
        float w = std::exp(-(dx * dx) / two_sigma_sq);
        sum += in_out_pairs[idx].second * w;
        wcount += w;
      }

      // store weighted average value
      lookup_table[i] = sum / wcount;
    }

    return true;
}
*/

bool LookupTable::init_interpolating(std::vector<std::pair<float, float>> in_out_pairs, 
                                     int table_size, bool sort_input)
{
  if (in_out_pairs.size() < 2)
    return false;

  // If input is in descending order, reverse it
  if(sort_input) {
    std::sort(in_out_pairs.begin(), in_out_pairs.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }
  else if (in_out_pairs.front().first > in_out_pairs.back().first) {
    std::reverse(in_out_pairs.begin(), in_out_pairs.end());
  }

  // check if values are now ascending (i.e the original input values where monotonic)
  bool is_ascending = std::is_sorted(in_out_pairs.begin(), in_out_pairs.end(),
                                     [](const auto& a, const auto& b) { return a.first < b.first; });
  if(is_ascending == false) {
    LOG_ERROR("LookupTable could not be initialized, input values must be monotonic");
    return false;
  }

  // get input range
  input_min = in_out_pairs.front().first;
  input_max = in_out_pairs.back().first;

  // check for invalid input range
  if (input_min >= input_max)
    return false; 
  one_over_input_range = 1.0f / (input_max - input_min);

  // define lookup table size
  lookup_table.resize(table_size, 0.0f);

  int j = 0; // index in in_out_pairs
  for (int i = 0; i < table_size; ++i) {
    float t = static_cast<float>(i) / (table_size - 1); // normalized [0,1]
    float x = input_min + t * (input_max - input_min);

    // advance j until x is between in_out_pairs[j] and in_out_pairs[j + 1]
    while (j + 1 < in_out_pairs.size() && x > in_out_pairs[j + 1].first) j++;

    float x0 = in_out_pairs[j].first;
    float y0 = in_out_pairs[j].second;
    float x1 = in_out_pairs[j + 1].first;
    float y1 = in_out_pairs[j + 1].second;

    if (std::fabs(x1 - x0) < std::numeric_limits<float>::epsilon()) {
      lookup_table[i] = y0;
    } else {
      float alpha = (x - x0) / (x1 - x0);
      lookup_table[i] = y0 + alpha * (y1 - y0);
    }
  }

  return true;
}

bool LookupTable::optimize_lut(std::vector<std::pair<float, float>> in_out_pairs, float& rms_error) {
  if (lookup_table.empty() || in_out_pairs.empty())
    return false;

  LOG_DEBUG("Optimizing lookup table...");

  const float learning_rate = 0.02f;
  const int max_iterations = 1000;
  const int N = static_cast<int>(lookup_table.size());

  // Gradient descent loop
  std::vector<float> gradients(N, 0.0f);
  float total_loss = 0.0f;
  for (int iter = 0; iter < max_iterations; iter++) {
    // Accumulate gradients for each pair
    total_loss = 0.0f;
    for (const auto &pair : in_out_pairs) {
      float x = pair.first;
      float y_target = pair.second;

      // Compute interpolation indices and weights
      int idx_a, idx_b;
      float w_a, w_b;
      linear_interpolate(x, idx_a, idx_b, w_a, w_b);

      // Current output from LUT
      float y_pred = w_a * lookup_table[idx_a] + w_b * lookup_table[idx_b];
      float error = y_pred - y_target;

      total_loss += error * error;

      // Gradient of loss wrt y_pred = 2 * error
      float grad_loss = 2.0f * error;

      // Distribute gradient to LUT entries weighted by interpolation weights
      gradients[idx_a] += grad_loss * w_a;
      gradients[idx_b] += grad_loss * w_b;

      //if(iter == max_iterations-1)
      //  LOG_DEBUG("error=%f", iter, error);
    }

    // Update LUT entries
    for (int i = 0; i < N; i++) {
      lookup_table[i] -= learning_rate * gradients[i];
      gradients[i] = 0.0f;
    }

    //if(iter%100 == 0)
    //  LOG_DEBUG("iteration %04i: rms=%f", iter, sqrtf(total_loss));
  }

  rms_error = sqrt(total_loss);
  LOG_DEBUG("Optimizing lookup table finished: rms_error=%f", rms_error);

  return true;
}

void LookupTable::clear() {
  lookup_table.clear();
}

// returns the size of the lookup table
uint32_t LookupTable::size() const {
  return (uint32_t)lookup_table.size();
}

void LookupTable::set_entry(int32_t idx, float v) {
  lookup_table[idx] = v;
}

// set an entry of the lookup table
float LookupTable::get_entry(int32_t idx) const {
  return lookup_table[idx];
}

float LookupTable::evaluate(float x) const {
  if (lookup_table.size() < 2)
    return 0.0f;

  int idx_a, idx_b; 
  float weight_a, weight_b;
  linear_interpolate(x, idx_a, idx_b, weight_a, weight_b);

  float a = lookup_table[idx_a];
  float b = lookup_table[idx_b];
  return a + weight_b * (b - a);
}

void LookupTable::linear_interpolate(float x, int& idx_a, int& idx_b, 
                                     float& weight_a, float& weight_b) const 
{
  int32_t table_size = lookup_table.size();
  float t = (x - input_min) * one_over_input_range;
  float pos = t * (table_size - 1);
  float frac;
  size_t index;

  if (t < 0.0f) {
    idx_a = idx_b = 0;
    weight_a = 1.0f;
    weight_b = 0.0f;
  } else if (t >= 1.0f) {
    idx_a = idx_b = table_size-1;
    weight_a = 0.0f;
    weight_b = 1.0f;
  } else {
    index = static_cast<size_t>(std::floor(pos));  
    idx_a = index;
    idx_b = index + 1;
    weight_b = pos - index;
    weight_a = (1.0f-weight_b);
  }
}

bool LookupTable::is_monotonic() const {
  if (lookup_table.size() < 2)
    return true;

  bool increasing = true, decreasing = true;
  for (size_t i = 1; i < lookup_table.size(); ++i) {
    float b = lookup_table[i - 1];
    float a = lookup_table[i];

    if (a < b) increasing = false;
    if (a > b) decreasing = false;
  }

  return increasing || decreasing;
}

bool LookupTable::in_input_range(float x) const {
  return x >= input_min && x <= input_max;
}

bool LookupTable::in_output_range(float x) const {
  if(lookup_table.size() < 2) return false;
  float a = lookup_table.front();
  float b = lookup_table.back();
  return x >= std::min(a,b) && x <= std::max(a,b);
}

void LookupTable::get_intput_range(float& input_min, float& input_max) const {
  input_min = LookupTable::input_min;
  input_max = LookupTable::input_max;
}

float LookupTable::evaluate_inverse(float y) const {
  int size = static_cast<int>(lookup_table.size());
  if (size < 2) return input_min;

  int low = 0;
  int high = size - 1;
  bool increasing = lookup_table.front() < lookup_table.back();

  // Clamp y outside the range
  // Clamp y outside the range
  if ((increasing && y <= lookup_table.front()) || 
      (!increasing && y >= lookup_table.front()))
    return input_min;
  if ((increasing && y >= lookup_table.back()) || 
      (!increasing && y <= lookup_table.back()))
    return input_max;

  // Binary search to find the interval
  while (high - low > 1) {
    int mid = (low + high) / 2;
    float val = lookup_table[mid];

    if ((increasing && val < y) || (!increasing && val > y))
      low = mid;
    else
      high = mid;
  }

  // Interpolate between low and high
  float y0 = lookup_table[low];
  float y1 = lookup_table[high];

  if (std::fabs(y1 - y0) < std::numeric_limits<float>::epsilon()) {
    // Avoid division by zero if both entries are equal
    float t = float(low) / (size - 1);
    return input_min + t * (input_max - input_min);
  }

  float t = (y - y0) / (y1 - y0);
  float pos = (float(low) + t) / (size - 1);

  return input_min + pos * (input_max - input_min);
}

// inverts the lookup table so it represents the funcion x = fi(y) given y = f(x) 
bool LookupTable::invert(int new_size) {
  if (lookup_table.empty() || new_size <= 0) {
    LOG_ERROR("invert_lut(): lut size is zero");
    return false;
  }

  if (!is_monotonic()) {
    LOG_ERROR("invert_lut(): lut is not monotonic");
    return false;
  }

  // Find the output (y) range of the current LUT
  float output_min = lookup_table.front();
  float output_max = lookup_table.back();
  if (output_max < output_min) {
      std::swap(output_min, output_max);
  }

  // Prepare new LUT data
  std::vector<float> new_lut(new_size);
  float delta_y = (output_max - output_min) / (new_size - 1);

  for (int i = 0; i < new_size; ++i) {
      float y = output_min + i * delta_y;
      new_lut[i] = evaluate_inverse(y);  // find x for given y
  }

  // Replace old LUT with the inverted LUT
  lookup_table = std::move(new_lut);
  input_min = output_min;
  input_max = output_max;
  one_over_input_range = 1.0f/(input_max-input_min);

  return true;
}

void LookupTable::print_to_log() const {
  int size = lookup_table.size();
  if (size == 0) return;

  float step = (input_max - input_min) / (size - 1);
  for (int i = 0; i < size; ++i) {
    float x = input_min + i * step;
    float y = lookup_table[i];
    LOG_INFO("%.6f;%.6f", x, y);
  }
}

//*** FUNCTION ***********************************************************************************/

void build_linear_lut(LookupTable& lut, float in_min, float in_max, float out_min, float out_max) {
  lut.init(2, in_min, in_max);
  lut.set_entry(0, out_min);
  lut.set_entry(1, out_max);
}

bool save_lut_to_file(const LookupTable& lut, const char* filename) {
  LOG_DEBUG("Saving Lookup table to file '%s'...", filename);

  File file = LittleFS.open(filename, "w");
  if (!file) {
    LOG_ERROR("Failed to open file '%s' for writing", filename);
    return false;
  }

  // Save metadata
  uint32_t size = lut.size();
  float input_min, input_max;
  lut.get_intput_range(input_min, input_max);

  // Write metadata (size, input_min, input_max)
  if (file.write((uint8_t*)&size, sizeof(size)) != sizeof(size)) return false;
  if (file.write((uint8_t*)&input_min, sizeof(input_min)) != sizeof(input_min)) return false;
  if (file.write((uint8_t*)&input_max, sizeof(input_max)) != sizeof(input_max)) return false;

  // Write all LUT entries
  for (uint32_t i = 0; i < size; i++) {
    float v = lut.get_entry(i);
    if (file.write((uint8_t*)&v, sizeof(v)) != sizeof(v)) return false;
  }

  file.close();

  // LOG_DEBUG("Saving Lookup table to file successful");
  return true;
}

bool load_lut_from_file(LookupTable& lut, const char* filename) {
  LOG_DEBUG("Loading Lookup table from file '%s'...", filename);
  File file = LittleFS.open(filename, "r");
  if (!file) {
    LOG_ERROR("Failed to open file '%s' for reading", filename);
    return false;
  }

  // Read metadata
  uint32_t size = 0;
  float input_min = 0.0f, input_max = 0.0f;

  if (file.read((uint8_t*)&size, sizeof(size)) != sizeof(size)) return false;
  if (file.read((uint8_t*)&input_min, sizeof(input_min)) != sizeof(input_min)) return false;
  if (file.read((uint8_t*)&input_max, sizeof(input_max)) != sizeof(input_max)) return false;

  if (!lut.init(size, input_min, input_max)) {
    LOG_ERROR("Failed to create LUT from file");
    file.close();
    return false;
  }

  // Read LUT entries
  for (uint32_t i = 0; i < size; i++) {
    float v = 0.0f;
    if (file.read((uint8_t*)&v, sizeof(v)) != sizeof(v)) {
      LOG_ERROR("Reading LUT data failed");
      file.close();
      return false;
    }
    lut.set_entry(i, v);
  }

  file.close();
  // LOG_DEBUG("Loading Lookup table from file successful");

  return true;
}
