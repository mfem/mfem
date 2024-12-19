// SPDX-ArtifactOfProjectName:     noisy
// SPDX-ArtifactOfProjectHomePage: https://github.com/VincentZalzal/noisy
// SPDX-FileCopyrightText:         Copyright 2024 Vincent Zalzal
// SPDX-License-Identifier:        MIT

#pragma once

#include <iomanip>
#include <iostream>

namespace vz {

struct Counters {
    unsigned m_def_ctor    = 0;
    unsigned m_copy_ctor   = 0;
    unsigned m_move_ctor   = 0;
    unsigned m_copy_assign = 0;
    unsigned m_move_assign = 0;
    unsigned m_dtor        = 0;

    void reset() {
        *this = {};
    }

    bool leaks() const {
        return m_def_ctor + m_copy_ctor + m_move_ctor != m_dtor;
    }

    friend std::ostream& operator<<(std::ostream& os, const Counters& c) {
        stream_counter(os, "Default constructor count: ", c.m_def_ctor   );
        stream_counter(os, "Copy constructor count:    ", c.m_copy_ctor  );
        stream_counter(os, "Move constructor count:    ", c.m_move_ctor  );
        stream_counter(os, "Copy assignment count:     ", c.m_copy_assign);
        stream_counter(os, "Move assignment count:     ", c.m_move_assign);
        stream_counter(os, "Destructor count:          ", c.m_dtor       );
        return os;
    }

    friend bool operator==(const Counters& lhs, const Counters& rhs) {
        return
            lhs.m_def_ctor    == rhs.m_def_ctor    &&
            lhs.m_copy_ctor   == rhs.m_copy_ctor   &&
            lhs.m_move_ctor   == rhs.m_move_ctor   &&
            lhs.m_copy_assign == rhs.m_copy_assign &&
            lhs.m_move_assign == rhs.m_move_assign &&
            lhs.m_dtor        == rhs.m_dtor        ;
    }

    friend bool operator!=(const Counters& lhs, const Counters& rhs) { return !(lhs == rhs); }

private:
    static void stream_counter(std::ostream& os, const char* msg, unsigned value) {
        if (value != 0)
            os << msg << std::setw(2) << value << '\n';
    }
};

namespace detail {

struct Globals {
    ~Globals() {
        if (m_verbose)
            std::cout << "\n===== Noisy counters =====\n" << m_counters;
    }

    Counters m_counters;
    unsigned m_next_id = 0;
    bool     m_verbose = true;
};

}

class Noisy {
private:
    static detail::Globals& globals() {
        static detail::Globals s_globals;
        return s_globals;
    }

public:
    static Counters& counters() { return globals().m_counters; }
    static void set_verbose(bool verbose) { globals().m_verbose = verbose; }

    Noisy() {
        if (globals().m_verbose)
            std::cout << *this << ": default constructor\n";
        globals().m_counters.m_def_ctor++;
    }

    Noisy(const Noisy& other) {
        if (globals().m_verbose)
            std::cout << *this << ": copy constructor from " << other << '\n';
        globals().m_counters.m_copy_ctor++;
    }

    Noisy(Noisy&& other) noexcept {
        if (globals().m_verbose)
            std::cout << *this << ": move constructor from " << other << '\n';
        globals().m_counters.m_move_ctor++;
    }

    ~Noisy() {
        if (globals().m_verbose)
            std::cout << *this << ": destructor\n";
        globals().m_counters.m_dtor++;
    }

    Noisy& operator=(const Noisy& other) {
        if (globals().m_verbose)
            std::cout << *this << ": copy assignment from " << other << '\n';
        globals().m_counters.m_copy_assign++;
        return *this;
    }

    Noisy& operator=(Noisy&& other) noexcept {
        if (globals().m_verbose)
            std::cout << *this << ": move assignment from " << other << '\n';
        globals().m_counters.m_move_assign++;
        return *this;
    }

    unsigned id() const { return m_id; }

    friend std::ostream& operator<<(std::ostream& os, const Noisy& noisy) { return os << "Noisy(" << std::setw(2) << noisy.m_id << ')'; }

private:
    unsigned m_id = globals().m_next_id++;
};

}