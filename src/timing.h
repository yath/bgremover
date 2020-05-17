#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <ostream>

struct Timing {
    using duration = std::chrono::steady_clock::duration;

    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes): Intentionally.
    unsigned int nframes;

    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes): Intentionally.
    duration prepare_input, inference, mask, total;

    static std::chrono::time_point<std::chrono::steady_clock> now() {
        return std::chrono::steady_clock::now();
    }

    static uint64_t ms(duration d) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    }

    void reset() {
        nframes = 0;
        prepare_input = inference = mask = total = duration(0);
    }

    Timing() { reset(); }
};

static std::ostream& operator<<(std::ostream& os, const Timing& t) {
    if (!t.nframes) {
        os << "Timing(n=0)";
    } else {
        os << "Timing(n=" << t.nframes
           << ", avg(prepare_input)=" << Timing::ms(t.prepare_input) / t.nframes << "ms"
           << ", avg(inference)=" << Timing::ms(t.inference) / t.nframes << "ms"
           << ", avg(mask)=" << Timing::ms(t.mask) / t.nframes << "ms"
           << ", avg(total)=" << Timing::ms(t.total) / t.nframes << "ms";
    }
    return os;
};

#endif  // TIMING_H
