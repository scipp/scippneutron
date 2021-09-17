#include <scipp/dataset/dataset.h>
#include <scipp/neutron/convert.h>

int main() {
  scipp::Dataset d;
  d + d;
}
