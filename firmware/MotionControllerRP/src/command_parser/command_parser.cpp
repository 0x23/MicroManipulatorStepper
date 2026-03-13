
//*** INCLUDE ***************************************************************************

#include "command_parser.h"
#include <cstring>

//*** CLASS *****************************************************************************

//--- GCodeCommand ----------------------------------------------------------------------

GCodeCommand::GCodeCommand() {
  reset();
}

void GCodeCommand::set_command(const char* cmd) {
  if (cmd) {
    command = cmd;
  } else {
    command = "";
  }
}

const std::string& GCodeCommand::get_command() const {
  return command;
}

GCodeCommand::EParseStatus GCodeCommand::from_command_str(const char* gcode_command) {
  // Copy line into a mutable buffer
  char line_copy[256];
  strncpy(line_copy, gcode_command, sizeof(line_copy));
  line_copy[sizeof(line_copy) - 1] = '\0';

  // Tokenize the first word (e.g., G0, M3, T1)
  char* saveptr = nullptr;
  char* token = strtok_r(line_copy, " ", &saveptr);
  if (!token || token[0] < 'A' || token[0] > 'Z') {
    return EParseStatus::MALFORMED_COMMAND;
  }

  reset();
  set_command(token);

  // Parse remaining words (e.g., X1.0, Y2.5, F200)
  while ((token = strtok_r(nullptr, " ", &saveptr))) {
    if (token[0] >= 'A' && token[0] <= 'Z') {
      if(token[1] == '\0')
        set_value(token[0], 0.0f);
      else
        set_value(token[0], strtof(token + 1, nullptr));
    } else {
      return EParseStatus::INVALID_PARAMETER;
    }
  }

  return EParseStatus::OK;
}


void GCodeCommand::reset() {
  // Initialize all word values to NaN to represent "not set"
  for (float& value : word_values) {
    value = std::numeric_limits<float>::quiet_NaN();
  }
  command[0] = 0;
}

void GCodeCommand::set_value(char word, float value) {
  if (word >= 'A' && word <= 'Z') {
    word_values[word-'A'] = value;
  }
}

float GCodeCommand::get_value(char word) const {
  return get_value(word, std::numeric_limits<float>::quiet_NaN());
}

float GCodeCommand::get_value(char word, float default_value) const {
  if (word >= 'A' && word <= 'Z') {
    float value = word_values[word-'A'];
    return std::isnan(value) ? default_value : value;
  }
  return default_value;
}

bool GCodeCommand::has_word(char word) const {
  if (word >= 'A' && word <= 'Z') {
    return !std::isnan(word_values[word-'A']);
  }
  return false;
}

int GCodeCommand::get_word_count() const {
  int c = 0;
  for(int i=0; i<LETTER_COUNT; i++) {
    if(std::isnan(word_values[i]))
      c++;
  }
  return c;
}

bool GCodeCommand::contains_unsupported_words(const std::string& supported_words_str) const {
    // Parse supported words from the string into a fixed-size lookup table
    bool supported[LETTER_COUNT] = {false};

    // Parse the supported letters from the comma-separated string
    for (size_t i = 0; i < supported_words_str.length(); ++i) {
        char c = supported_words_str[i];
        if (c >= 'A' && c <= 'Z')
            supported[c - 'A'] = true;
    }

    // Now check which words are used in the command and not supported
    for (int i = 0; i < LETTER_COUNT; ++i) {
        char word = 'A' + i;
        if (has_word(word) && !supported[i]) {
            return true;  // Found an unsupported word
        }
    }

    return false;  // All words used in command are supported
}

//--- CommandParser ---------------------------------------------------------------------

CommandParser::CommandParser() : buffer_index(0), command_processor(nullptr) {

}

void CommandParser::set_command_processor(ICommandProcessor* cp) {
  command_processor = cp;
}

void CommandParser::update() {
  if(command_processor == nullptr)
    return;

  if(command_ready == false)
    return;

  // Call user callback with parsed command
  if(command_processor->can_process_command(command)) {
    std::string reply;
    command_processor->process_command(command, reply);
    command_processor->send_reply(reply.c_str());

    // mark command as processed
    command_ready = false;
  }
}

bool CommandParser::is_command_ready() {
  return command_ready;
}

// Feed input chars one by one
void CommandParser::add_input_character(char c) {
  if (c == '\n' || c == '\r') {
    if (buffer_index > 0) {
      buffer[buffer_index] = '\0';
      parse_line(buffer);
      buffer_index = 0;
    }
  } else if (buffer_index < sizeof(buffer) - 1) {
    buffer[buffer_index++] = c;
  }
}

bool CommandParser::parse_line(const char* line) {
  command_ready = false;

  // reset and parse gcode command from string
  GCodeCommand::EParseStatus ret = command.from_command_str(line);

  if(ret == GCodeCommand::EParseStatus::MALFORMED_COMMAND) {
    command_processor->send_reply("error: malformed command\n");
    return false;
  }

  if(ret == GCodeCommand::EParseStatus::INVALID_PARAMETER) {
    command_processor->send_reply("error: invalid parameter\n");
    return false;
  }

  command_ready = true;
  return true;
}