// Exact immutable FP8 row retrieval; bounded set-associative RAM cache.
// No CUDA calls in the callback: suitable for cudaLaunchHostFunc graph nodes.
//
// The callback blocks the compute stream, so every microsecond spent here is a
// microsecond the GPU is idle. Two things keep it short:
//   * misses are serviced by a shared thread pool (NVMe at QD=1 delivers ~3.5k
//     IOPS on GB10, at QD=64 ~112k -- the drive, not the latency, is the limit);
//   * a miss costs one pread, not two into unrelated 4 KiB pages -- either from
//     a repacked local shard (weight+scale adjacent, and no NFS hop on a worker)
//     or, failing that, from the owned scale shard pinned in RAM (8 B/row).
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint64_t kRowBytes = 264;   // 256 B weight + 8 B scale
constexpr uint64_t kWeightBytes = 256;
constexpr uint64_t kScaleBytes = 8;
constexpr uint64_t kPage = 4096;
constexpr uint64_t kPackedHeader = 4096;   // keeps row 0 page-aligned
constexpr uint64_t kPackedMagic = 0x31344e4531565344ULL;  // "DSV41EN1"
// One id per ticket. Misses inside a ticket are serviced serially by the thread
// that claimed it, so a larger chunk would cap in-flight reads at count/kChunk --
// exactly the queue depth this pool exists to raise. A decode step at batch 1
// only offers ~144 ids per layer, and fetch_add is ~20 ns against a ~200 us read.
constexpr uint64_t kChunk = 1;
constexpr uint64_t kSerialBelow = 8;  // dispatch is not worth it under this

uint64_t env_u64(const char *name, uint64_t fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw) return fallback;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw) return fallback;
  return uint64_t(value);
}

bool env_flag(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw) return fallback;
  return !(std::strcmp(raw, "0") == 0 || std::strcmp(raw, "off") == 0 ||
           std::strcmp(raw, "false") == 0);
}

}  // namespace

struct Store {
  int fd;
  uint64_t rows, weight_offset, scale_offset, slots, row_lo, row_hi;
  uint64_t sets = 0, ways = 1;
  uint8_t *cache = nullptr;
  uint64_t *keys = nullptr;
  uint8_t *victim = nullptr;          // per-set round-robin replacement cursor
  uint8_t *resident = nullptr;        // OFFLOAD_MODE=ram: whole shard mapped
  size_t resident_size = 0;
  uint8_t *scales = nullptr;          // owned scale rows, [row_lo, row_hi)
  size_t scales_size = 0;
  int packed_fd = -1;                 // repacked owned rows, weight+scale adjacent
  std::mutex locks[256];
  std::atomic<uint64_t> hits{0}, misses{0}, reads{0};
};

struct Work {
  Store *store;
  const int64_t *ids;
  uint8_t *weights, *scales;
  uint64_t count;
};

static void fail(const char *reason) {
  std::fprintf(stderr, "Engram retrieval failed: %s (errno=%d)\n", reason, errno);
  std::abort(); // Never allow a generation to continue with missing/stale rows.
}

static void read_from(Store *s, int fd, uint64_t offset, uint8_t *out, size_t length) {
  if (s->resident && fd == s->fd) {
    std::memcpy(out, s->resident + offset, length);
    return;
  }
  alignas(4096) uint8_t page[8192];
  const uint64_t base = offset & ~(kPage - 1);
  const size_t delta = offset - base;
  const size_t requested = ((delta + length + kPage - 1) / kPage) * kPage;
  ssize_t got;
  do { got = pread(fd, page, requested, base); } while (got < 0 && errno == EINTR);
  if (got < 0 || size_t(got) < delta + length) fail("short or failed direct read");
  std::memcpy(out, page + delta, length);
  s->reads.fetch_add(1, std::memory_order_relaxed);
}

static void read_bytes(Store *s, uint64_t offset, uint8_t *out, size_t length) {
  read_from(s, s->fd, offset, out, length);
}

// Fetch one owned row into `row` (264 B), consulting and filling the cache.
static void fetch_row(Store *s, uint64_t id, uint8_t *row) {
  const uint64_t set = s->sets ? id % s->sets : 0;
  std::mutex &lock = s->locks[set % 256];
  if (s->sets) {
    std::lock_guard<std::mutex> guard(lock);
    const uint64_t base = set * s->ways;
    for (uint64_t w = 0; w < s->ways; ++w) {
      if (s->keys[base + w] == id + 1) {
        std::memcpy(row, s->cache + (base + w) * kRowBytes, kRowBytes);
        s->hits.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
  }
  // Miss: read outside the lock so same-set misses overlap on the drive.
  if (s->packed_fd >= 0) {
    // Weight and scale are adjacent here, so the whole row is one read -- and on
    // a worker it is a local read instead of an NFS round trip to the head.
    read_from(s, s->packed_fd, kPackedHeader + (id - s->row_lo) * kRowBytes,
              row, kRowBytes);
  } else {
    read_bytes(s, s->weight_offset + id * kWeightBytes, row, kWeightBytes);
    if (s->scales) {
      std::memcpy(row + kWeightBytes, s->scales + (id - s->row_lo) * kScaleBytes,
                  kScaleBytes);
    } else {
      read_bytes(s, s->scale_offset + id * kScaleBytes, row + kWeightBytes, kScaleBytes);
    }
  }
  s->misses.fetch_add(1, std::memory_order_relaxed);
  if (s->sets) {
    std::lock_guard<std::mutex> guard(lock);
    const uint64_t base = set * s->ways;
    const uint64_t w = s->victim[set] % s->ways;
    s->victim[set] = uint8_t((s->victim[set] + 1) % s->ways);
    std::memcpy(s->cache + (base + w) * kRowBytes, row, kRowBytes);
    s->keys[base + w] = id + 1;
  }
}

static void serve(Work *work, uint64_t begin, uint64_t end) {
  Store *s = work->store;
  for (uint64_t i = begin; i < end; ++i) {
    const int64_t id = work->ids[i];
    if (id < 0 || uint64_t(id) >= s->rows) fail("row ID out of bounds");
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(work->weights + i * kWeightBytes, 0, kWeightBytes);
      std::memset(work->scales + i * kScaleBytes, 0, kScaleBytes);
      continue;
    }
    uint8_t row[kRowBytes];
    fetch_row(s, uint64_t(id), row);
    std::memcpy(work->weights + i * kWeightBytes, row, kWeightBytes);
    std::memcpy(work->scales + i * kScaleBytes, row + kWeightBytes, kScaleBytes);
  }
}

namespace {

// One process-wide pool. The two Engram layers never look up concurrently, so
// they share it rather than each holding its own idle threads.
class Pool {
 public:
  explicit Pool(uint64_t threads) {
    workers_.reserve(threads);
    for (uint64_t i = 0; i < threads; ++i) workers_.emplace_back([this] { loop(); });
  }

  uint64_t size() const { return workers_.size(); }

  void run(Work *work) {
    // One job at a time: the two Engram layers issue their callbacks on the same
    // stream, but a shared pool must not depend on that.
    std::lock_guard<std::mutex> serialize(caller_);
    const uint64_t chunks = (work->count + kChunk - 1) / kChunk;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      job_ = work;
      next_.store(0, std::memory_order_relaxed);
      chunks_ = chunks;
      remaining_.store(chunks, std::memory_order_relaxed);
      ++generation_;
    }
    ready_.notify_all();
    drain(work);  // the caller works too, so progress never depends on a wakeup
    // Wait for the chunks *and* for every worker to leave drain(): only then is
    // it safe for the next job to reset the ticket counter under them.
    std::unique_lock<std::mutex> guard(mutex_);
    done_.wait(guard, [this] {
      return remaining_.load(std::memory_order_acquire) == 0 &&
             active_.load(std::memory_order_acquire) == 0;
    });
    job_ = nullptr;
  }

  ~Pool() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      stop_ = true;
      ++generation_;
    }
    ready_.notify_all();
    for (auto &worker : workers_) worker.join();
  }

 private:
  void drain(Work *work) {
    if (!work) return;
    for (;;) {
      const uint64_t ticket = next_.fetch_add(1, std::memory_order_relaxed);
      if (ticket >= chunks_) return;
      const uint64_t begin = ticket * kChunk;
      serve(work, begin, std::min(begin + kChunk, work->count));
      if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> guard(mutex_);
        done_.notify_one();
      }
    }
  }

  void loop() {
    uint64_t seen = 0;
    for (;;) {
      Work *work = nullptr;
      {
        std::unique_lock<std::mutex> guard(mutex_);
        ready_.wait(guard, [this, seen] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        work = job_;
        // Counted under the lock so run()'s predicate can never miss a worker
        // that is about to enter drain().
        if (work) active_.fetch_add(1, std::memory_order_release);
      }
      if (!work) continue;
      drain(work);
      std::lock_guard<std::mutex> guard(mutex_);
      active_.fetch_sub(1, std::memory_order_release);
      done_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_, caller_;
  std::condition_variable ready_, done_;
  Work *job_ = nullptr;
  std::atomic<uint64_t> next_{0}, remaining_{0}, active_{0};
  uint64_t chunks_ = 0, generation_ = 0;
  bool stop_ = false;
};

Pool *pool() {
  static Pool *instance = [] {
    const uint64_t threads = env_u64("DSV41_IO_THREADS", 32);
    return threads > 1 ? new Pool(threads - 1) : nullptr;  // caller is a worker
  }();
  return instance;
}

// Pin the owned scale rows so a miss costs one pread, not two.
bool pin_scales(Store *s) {
  const uint64_t count = s->row_hi - s->row_lo;
  if (!count) return false;
  const uint64_t bytes = count * kScaleBytes;
  void *map = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (map == MAP_FAILED) return false;
  auto *out = static_cast<uint8_t *>(map);
  const uint64_t start = s->scale_offset + s->row_lo * kScaleBytes;
  const uint64_t end = start + bytes;
  if (s->resident) {
    std::memcpy(out, s->resident + start, bytes);
    s->scales = out;
    s->scales_size = bytes;
    return true;
  }
  const size_t span = size_t(8) << 20;
  void *raw = nullptr;
  if (posix_memalign(&raw, kPage, span)) { munmap(map, bytes); return false; }
  auto *buffer = static_cast<uint8_t *>(raw);
  uint64_t offset = start & ~(kPage - 1);
  while (offset < end) {
    const uint64_t want = std::min<uint64_t>(span, ((end - offset + kPage - 1) / kPage) * kPage);
    ssize_t got;
    do { got = pread(s->fd, buffer, size_t(want), off_t(offset)); }
    while (got < 0 && errno == EINTR);
    if (got <= 0) { std::free(raw); munmap(map, bytes); return false; }
    const uint64_t lo = std::max(offset, start);
    const uint64_t hi = std::min<uint64_t>(offset + uint64_t(got), end);
    if (hi > lo) std::memcpy(out + (lo - start), buffer + (lo - offset), hi - lo);
    offset += uint64_t(got);
  }
  std::free(raw);
  s->scales = out;
  s->scales_size = bytes;
  return true;
}

}  // namespace

extern "C" Store *row_store_open(const char *path, uint64_t rows,
                                 uint64_t woff, uint64_t soff, uint64_t budget) {
  auto *s = new Store;
  const char *mode = std::getenv("OFFLOAD_MODE");
  const bool ram = mode && std::strcmp(mode, "ram") == 0;
  s->fd = open(path, O_RDONLY | O_CLOEXEC | (ram ? 0 : O_DIRECT));
  if (s->fd < 0 && !ram) {
    // SSHFS / NFS often reject O_DIRECT; buffered reads still return exact bytes.
    s->fd = open(path, O_RDONLY | O_CLOEXEC);
    if (s->fd >= 0) {
      std::fprintf(stderr,
          "Engram: O_DIRECT unavailable for %s, using buffered reads (errno was %d)\n",
          path, errno);
    }
  }
  if (s->fd < 0) { delete s; return nullptr; }
  struct stat statbuf;
  if (fstat(s->fd, &statbuf) || woff > uint64_t(statbuf.st_size) ||
      soff > uint64_t(statbuf.st_size) || rows > (uint64_t(statbuf.st_size)-woff)/256 ||
      rows > (uint64_t(statbuf.st_size)-soff)/8) fail("invalid table extent");
  if (ram) {
    s->resident_size = statbuf.st_size;
    s->resident = static_cast<uint8_t *>(mmap(nullptr,s->resident_size,
        PROT_READ,MAP_SHARED | MAP_POPULATE,s->fd,0));
    if (s->resident == MAP_FAILED) fail("resident mapping");
    if (mlock(s->resident,s->resident_size)) fail("RAM mode requires memlock capability and sufficient RAM");
    budget = 0;
  }
  s->rows = rows; s->weight_offset = woff; s->scale_offset = soff;
  s->row_lo = 0; s->row_hi = rows;
  s->ways = std::max<uint64_t>(1, std::min<uint64_t>(16, env_u64("DSV41_CACHE_WAYS", 4)));
  s->sets = budget / ((kRowBytes + sizeof(uint64_t)) * s->ways + 1);
  s->slots = s->sets * s->ways;
  if (s->slots) {
    s->cache = static_cast<uint8_t *>(mmap(nullptr, s->slots * kRowBytes,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    s->keys = static_cast<uint64_t *>(mmap(nullptr, s->slots * sizeof(uint64_t),
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    s->victim = static_cast<uint8_t *>(mmap(nullptr, s->sets,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (s->cache == MAP_FAILED || s->keys == MAP_FAILED || s->victim == MAP_FAILED)
      fail("cache allocation");
  }
  return s;
}

extern "C" void row_store_lookup(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Pool *workers = pool();
  if (!workers || work->count < kSerialBelow) {
    serve(work, 0, work->count);
    return;
  }
  workers->run(work);
}

extern "C" void row_store_stats(Store *s, uint64_t *out) {
  out[0] = s->hits.load(); out[1] = s->misses.load(); out[2] = s->reads.load();
  out[3] = s->slots * (kRowBytes + sizeof(uint64_t)) + s->sets;
  out[4] = s->slots;
  out[5] = s->scales_size;
  out[6] = s->ways;
  out[8] = s->packed_fd >= 0 ? 1 : 0;
  Pool *workers = pool();
  out[7] = workers ? workers->size() + 1 : 1;
}

// Attach a shard produced by scripts/pack_engram.py: this rank's owned rows,
// weight and scale adjacent, on local disk. Missing is fine (we fall back to the
// checkpoint); present but wrong is a repack bug and must never be served.
extern "C" int row_store_attach_packed(Store *s, const char *path, uint64_t layer_id) {
  const int fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
  if (fd < 0) return 0;
  alignas(4096) uint8_t header[kPage];
  ssize_t got;
  do { got = pread(fd, header, kPage, 0); } while (got < 0 && errno == EINTR);
  if (got < ssize_t(kPage)) { close(fd); fail("packed Engram shard has no header"); }
  uint64_t magic, lo, hi, rows, layer, row_bytes;
  std::memcpy(&magic, header + 0, 8);
  std::memcpy(&layer, header + 8, 8);
  std::memcpy(&lo, header + 16, 8);
  std::memcpy(&hi, header + 24, 8);
  std::memcpy(&rows, header + 32, 8);
  std::memcpy(&row_bytes, header + 40, 8);
  if (magic != kPackedMagic || layer != layer_id || lo != s->row_lo ||
      hi != s->row_hi || rows != s->rows || row_bytes != kRowBytes)
    fail("packed Engram shard does not match this rank's range; re-run the pack step");
  struct stat statbuf;
  if (fstat(fd, &statbuf) ||
      uint64_t(statbuf.st_size) < kPackedHeader + (hi - lo) * kRowBytes)
    fail("packed Engram shard is truncated");
  s->packed_fd = fd;
  return 1;
}

extern "C" void row_store_range(Store *s, uint64_t lo, uint64_t hi) {
  if (lo > hi || hi > s->rows) fail("invalid row ownership range");
  // The pinned shard is indexed off row_lo, so moving the range under it would
  // silently return another row's scale. Rows are immutable and the range is set
  // once at load; anything else is a bug, not a case to paper over.
  if (s->scales && (lo != s->row_lo || hi != s->row_hi))
    fail("ownership range changed after the scale shard was pinned");
  s->row_lo = lo; s->row_hi = hi;
  if (!s->scales && hi > lo && env_flag("DSV41_RESIDENT_SCALES", true) && !pin_scales(s)) {
    std::fprintf(stderr, "Engram: could not pin the owned scale shard (%llu rows); "
                 "misses will cost a second read\n",
                 static_cast<unsigned long long>(hi - lo));
  }
}

extern "C" void row_store_close(Store *s) {
  if (s->packed_fd >= 0) close(s->packed_fd);
  if (s->resident) munmap(s->resident,s->resident_size);
  if (s->scales) munmap(s->scales, s->scales_size);
  if (s->slots) {
    munmap(s->cache, s->slots * kRowBytes);
    munmap(s->keys, s->slots * sizeof(uint64_t));
    munmap(s->victim, s->sets);
  }
  close(s->fd); delete s;
}
