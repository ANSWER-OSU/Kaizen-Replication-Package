#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <utility>
#include <vector>

namespace {
template<typename T>
struct CmdArgReaderImpl {
  std::string value;
  T* value_ptr{nullptr};

  CmdArgReaderImpl() = default;
  CmdArgReaderImpl(const T& value) : value(value), value_ptr(&value) {}
  CmdArgReaderImpl(const std::string& value) : value(value) {}
  T getValue(const std::string& value) {
    if (value_ptr) {
      return *value_ptr;
    } else {
      std::stringstream ss;
      ss << value;
      T t{};
      ss >> t;
      return t;
    }
  }
};

struct CmdArgReader {
  std::map<std::string, CmdArgReaderImpl<bool>>& args;
  std::map<std::string, std::string>& unprocessed;
  std::map<std::string, CmdArgReaderImpl<std::string>>::iterator iter;
  std::map<std::string, CmdArgReaderImpl<std::string>>::iterator iter_unprocessed;

  static CmdArgReader* self;
  static char** rargv;
  static int rargc;

  void createArgsMaps(int argc, const char** argv) {
    std::string token;
    std::string val_str;
    std::map<std::string, std::string> args;

    std::string::size_type pos;
    std::string arg;
    for (int i = 1; i < argc; ++i) {
      arg = argv[i];

      if (arg[0] != '-') {
        throw std::runtime_error("Invalid command line argument.");
      }

      int numDashes = (arg[1] == '-' ? 2 : 1);

      // check if only flag or if a value is given
      if ((pos = arg.find('=')) == std::string::npos) {
        unprocessed[std::string(arg, numDashes, arg.length() - numDashes)] = "FLAG";
      } else {
        unprocessed[std::string(arg, numDashes, pos - numDashes)] =
            std::string(arg, pos + 1, arg.length() - 1);
      }
    }
  }

  template<typename T>
  static T getValue(const std::string& name) {
    if (self) {
      return self->args.at(name).getValue(value);
    }
  }
};

CmdArgReader* CmdArgReader::self{nullptr};
char** CmdArgReader::rargv{nullptr};
int CmdArgReader::rargc{0};
} // namespace

using namespace std;

int main(int argc, char* argv[]) {
  try {
    CmdArgReader::init(argc, argv);
  } catch (const exception& e) {
    cout << "Runtime error: " << e.what() << endl;
    return 1;
  }

  while (true) {
    cout << "Enter an integer value: ";
    int value;
    cin >> value;
    cout << "You entered: " << CmdArgReader::getValue<int>(value) << endl;
  }

  return 0;
}