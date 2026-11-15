// Lightweight Gazebo Fortress transport-side probe. No image conversion or disk I/O.
// Build with ignition-transport11 and ignition-msgs8; see --help.
#include <ignition/msgs/image.pb.h>
#include <ignition/transport/Node.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Sample = std::pair<int64_t, int64_t>;  // source stamp, steady arrival time

void printGaps(const std::vector<int64_t> &gaps) {
  if (gaps.empty()) {
    std::cout << "{\"count\":0,\"mean_ms\":null,\"p95_ms\":null,\"max_ms\":null,"
                 "\"over_100ms\":0,\"over_250ms\":0}";
    return;
  }
  auto sorted = gaps;
  std::sort(sorted.begin(), sorted.end());
  double total = 0;
  size_t over100 = 0, over250 = 0;
  for (auto gap : gaps) {
    total += gap;
    over100 += gap > 100000000;
    over250 += gap > 250000000;
  }
  const auto p95 = static_cast<size_t>(std::ceil(.95 * sorted.size())) - 1;
  std::cout << "{\"count\":" << gaps.size() << ",\"mean_ms\":" << total / gaps.size() / 1e6
            << ",\"p95_ms\":" << sorted[p95] / 1e6 << ",\"max_ms\":" << sorted.back() / 1e6
            << ",\"over_100ms\":" << over100 << ",\"over_250ms\":" << over250 << "}";
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "Usage: camera_native_timing [wall_seconds=30]\n"
                 "Read-only Ignition /camera/image timing. Does not publish or decode images.\n"
                 "Build: g++ -O2 -std=c++17 tools/camera_native_timing.cpp "
                 "-o /tmp/wheelarm_camera_native_timing "
                 "$(pkg-config --cflags --libs ignition-transport11 ignition-msgs8)\n";
    return 0;
  }
  double seconds = 30.;
  try {
    if (argc > 2) throw std::invalid_argument("too many arguments");
    if (argc == 2) {
      size_t used = 0;
      seconds = std::stod(argv[1], &used);
      if (used != std::string(argv[1]).size()) throw std::invalid_argument("invalid duration");
    }
    if (!std::isfinite(seconds) || seconds <= 0 || seconds > 600)
      throw std::invalid_argument("duration must be in (0, 600] seconds");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::mutex mutex;
  std::vector<Sample> samples;
  ignition::transport::Node node;
  std::function<void(const ignition::msgs::Image &)> receive = [&](const ignition::msgs::Image &image) {
    const auto arrival = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const auto &stamp = image.header().stamp();
    const int64_t source = stamp.sec() * int64_t{1000000000} + stamp.nsec();
    std::lock_guard<std::mutex> lock(mutex);
    samples.emplace_back(source, arrival);
  };
  if (!node.Subscribe("/camera/image", receive)) {
    std::cerr << "Could not subscribe to Ignition /camera/image\n";
    return 1;
  }
  std::cerr << "[native timing] /camera/image; " << seconds << " wall seconds\n";
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  node.Unsubscribe("/camera/image");
  std::vector<Sample> snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex);
    snapshot = samples;
  }
  std::vector<int64_t> sourceGaps, arrivalGaps;
  size_t invalid = 0, duplicates = 0, backwards = 0, valid = 0;
  int64_t previousSource = 0, firstSource = 0;
  for (size_t i = 0; i < snapshot.size(); ++i) {
    const auto [source, arrival] = snapshot[i];
    if (i) arrivalGaps.push_back(arrival - snapshot[i - 1].second);
    if (source <= 0) { ++invalid; continue; }
    if (!valid++) firstSource = source;
    if (previousSource) {
      const auto gap = source - previousSource;
      if (gap > 0) sourceGaps.push_back(gap);
      else if (gap == 0) ++duplicates;
      else ++backwards;
    }
    previousSource = source;
  }
  std::cout << std::setprecision(10)
            << "{\n  \"kind\":\"ignition_camera_timing\",\n  \"topic\":\"/camera/image\",\n"
            << "  \"messages\":" << snapshot.size() << ",\n  \"arrival_wall_hz\":";
  if (snapshot.size() >= 2 && snapshot.back().second > snapshot.front().second)
    std::cout << (snapshot.size() - 1) * 1e9 / (snapshot.back().second - snapshot.front().second);
  else std::cout << "null";
  std::cout << ",\n  \"source_stamp_hz\":";
  if (valid >= 2 && !backwards && previousSource > firstSource)
    std::cout << (valid - 1) * 1e9 / (previousSource - firstSource);
  else std::cout << "null";
  std::cout << ",\n  \"arrival_gaps\":";
  printGaps(arrivalGaps);
  std::cout << ",\n  \"positive_source_gaps\":";
  printGaps(sourceGaps);
  std::cout << ",\n  \"nonpositive_source_stamps\":" << invalid
            << ",\n  \"duplicate_source_stamps\":" << duplicates
            << ",\n  \"backward_source_stamps\":" << backwards << "\n}\n";
  return snapshot.size() >= 2 ? 0 : 1;
}
