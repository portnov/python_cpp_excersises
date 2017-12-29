#include <iostream>
#include <map>
#include <list>

using namespace std;

int main(int argc, char* argv[]) {
  map<int,list<int> > my_map;

  my_map[10].push_back(100);
  my_map[10].push_back(200);
  my_map[10].push_back(300);
  my_map[5].push_back(400);

  for (auto const & entry : my_map) {
    auto const & key = entry.first;
    auto const & values = entry.second;
    for (auto const & value : values) {
      cout << "Key: " << key << " => " << value;
    }
  }
}
