#pragma once

//*** INCLUDE ***************************************************************************

#include <functional>
#include <string>
#include <stdlib.h>
#include <stdint.h>
#include "utilities/math3d.h"

//*** CLASS *****************************************************************************

static constexpr int LETTER_COUNT = 26;

//--- GCodeCommand ----------------------------------------------------------------------

class GCodeCommand {
  public:
    enum class EParseStatus {
      OK=1,
      MALFORMED_COMMAND=-1,
      INVALID_PARAMETER=-2
    };

    GCodeCommand();

    EParseStatus        from_command_str(const char* cmd);

    void                reset();
    void                set_command(const char* cmd);
    const std::string&  get_command() const;
    void                set_value(char word, float value);
    float               get_value(char word) const;
    float               get_value(char word, float default_value) const;
    bool                has_word(char word) const;
    int                 get_word_count() const;
    bool                contains_unsupported_words(const std::string& supported_words_str) const;

  private:
    std::string command;
    float word_values[LETTER_COUNT];
};

//--- ICommandProcessor -----------------------------------------------------------------

class ICommandProcessor {
  public:
    virtual ~ICommandProcessor() {};

    virtual void send_reply(const char* str) = 0;
    virtual bool can_process_command(const GCodeCommand& cmd) = 0;
    virtual void process_command(const GCodeCommand& cmd, std::string& reply) = 0;
};

//--- CommandParser ---------------------------------------------------------------------

class CommandParser {
  public:
    CommandParser();

    void set_command_processor(ICommandProcessor* cp);
    void add_input_character(char c);  // Feed input chars one by one
    void update();                     
    bool is_command_ready();  // true if command was parsed and is waiting to be executed
    bool parse_line(const char* line);

  private:
    char buffer[255];
    int buffer_index;
    ICommandProcessor* command_processor;

    GCodeCommand command;
    bool command_ready;
};

