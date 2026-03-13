// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>

//*** CLASS *****************************************************************************

class LookupTable {
  public:
    LookupTable() {}

    // initializes the lookup table to a given size and input range
    bool init(int32_t table_size, float input_min, float input_max);

    // initializes the lookup table from a list of input output value pairs
    // approximation/interpolation will be used to sample the input range in
    // equidistant steps. 
    bool init_interpolating(std::vector<std::pair<float, float>> in_out_pairs, 
                            int table_size, bool sort_input);
    
    // optimizes the lookup table using gradient descen to minimize error to provided data
    bool optimize_lut(std::vector<std::pair<float, float>> in_out_pairs, float& rms_error);

    // clear the lookup table, use init to use it again
    void clear();

    // returns the size of the lookup table
    uint32_t size() const;

    // set an entry of the lookup table
    void set_entry(int32_t idx, float v);

    // set an entry of the lookup table
    float get_entry(int32_t idx) const;

    // evaluate the lookup table at a given position with linear interpolation
    float evaluate(float x) const;

    // evaluate the inverse of the lookup table function (very slow), the LUT must be monotonic
    float evaluate_inverse(float y) const;

    // inverts the lookup table so it represents the funcion x = fi(y) given y = f(x) 
    bool invert(int new_size);

    // check if the lookup table is monotonic
    bool is_monotonic() const;

    // check if the given value is inside the input range
    bool in_input_range(float x) const;

    // check if the given value is inside the output range
    bool in_output_range(float x) const;

    // get input range
    void get_intput_range(float& input_min, float& input_max) const;

    // prints the lookup table using the logger
    void print_to_log() const;

  private:
    void linear_interpolate(float x, int& idx_a, int& idx_b, float& weight_a, float& weight_b) const;

  private:
    float input_min = 0.0f;
    float input_max = 0.0f;
    float one_over_input_range = 1.0f;
    std::vector<float> lookup_table;
};

//*** FUNCTION ***********************************************************************************/

void build_linear_lut(LookupTable& lut, float in_min, float in_max, float out_min, float out_max);
bool save_lut_to_file(const LookupTable& lut, const char* filename);
bool load_lut_from_file(LookupTable& lut, const char* filename);

