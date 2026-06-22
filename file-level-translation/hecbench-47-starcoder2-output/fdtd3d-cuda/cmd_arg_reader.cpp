// Include OpenMP header file
#include <omp.h>

// Create OpenMP version of CmdArgReader class
class CmdArgReader
{
public:
  // Public construction interface
  // @return a handle to the class instance
  // @param argc number of command line arguments (as given to main())
  // @param argv command line argument string (as given to main())
  static void init(const int argc, const char** argv);

  // Public functions
  int getInt(const std::string& param, const int default_value);
  bool getFlag(const std::string& param);
  std::string getString(const std::string& param, const std::string& default_value);
  std::vector<int> getInts(const std::string& param);
  std::vector<std::string> getStrings(const std::string& param);
  bool hasValue(const std::string& param);

private:
  CmdArgReader();
  ~CmdArgReader();

  void createArgsMaps(const int argc, const char** argv);

  // Data members
  std::map<std::string, std::pair<std::type_info*, void*>> args;
  std::map<std::string, std::string> unprocessed;
  std::map<std::string, std::pair<std::type_info*, void*>>::iterator iter;
  std::map<std::string, std::string>::iterator iter_unprocessed;

  // Static data members
  static CmdArgReader* self;
  static char** rargv;
  static int rargc;
};