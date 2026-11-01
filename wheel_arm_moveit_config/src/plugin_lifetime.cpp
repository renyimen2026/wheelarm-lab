#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstring>
#include <dlfcn.h>

// Humble plugin clients can leave weak control blocks in longer-lived ROS nodes.
// Keep MoveIt DSOs mapped until process exit, so their control blocks can be
// destroyed normally. Objects, dlclose calls and shutdown signals are untouched.
extern "C" void * dlopen(const char * filename, int flags) noexcept
{
  using Open = void * (*)(const char *, int);
  static const auto next_open = reinterpret_cast<Open>(dlsym(RTLD_NEXT, "dlopen"));
  if (!next_open) {
    return nullptr;
  }
  if (filename) {
    const char * slash = std::strrchr(filename, '/');
    const char * name = slash ? slash + 1 : filename;
    constexpr char prefix[] = "libmoveit_";
    if (std::strncmp(name, prefix, sizeof(prefix) - 1) == 0) {
      flags |= RTLD_NODELETE;
    }
  }
  return next_open(filename, flags);
}
