#include "Args.h"
#include <algorithm>
#include <sstream>
#include <string>
using namespace std;
using namespace std::string_literals;

vector<KeyVal> Args::splitUses(string ss) {
 vector<KeyVal> ret;
 std::replace(ss.begin(), ss.end(), ',', ' ');
 std::istringstream iss{ss};
 string s;
 while (iss >> s) {
 auto pos = s.find('=');
 string key = (pos == string::npos) ? s : s.substr(0, pos);
 string val = (pos == string::npos) ? "1"s : s.substr(pos+1);
 ret.push_back({key, val});
 }
 return ret;
}
