#pragma once

// Known-factor bookkeeping for the ECM drivers.
//
// The engine always works modulo the full Mersenne number, so a later curve that
// kills a factor already found (or supplied as known) in the same step as a new
// one reports the product.  Reporting that product as a new factor is wrong: it
// is composite and shares a prime with a factor that was already reported.

#include <gmpxx.h>

#include <cctype>
#include <string>
#include <vector>

namespace ecm_known {

// Parses a known-factor string (decimal, or hexadecimal with a 0x prefix; leading
// and trailing white space and a sign are accepted).  Returns false when the string
// is not a number or its magnitude is not above 1.
inline bool parse(const std::string& text, mpz_class& out) {
    std::size_t b = 0, e = text.size();
    while (b < e && std::isspace(static_cast<unsigned char>(text[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(text[e - 1]))) --e;
    if (b == e) return false;
    std::string s = text.substr(b, e - b);
    bool negative = false;
    if (s[0] == '+' || s[0] == '-') {
        negative = (s[0] == '-');
        s.erase(0, 1);
    }
    if (s.empty()) return false;
    int base = 10;
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        base = 16;
        s.erase(0, 2);
    }
    if (s.empty()) return false;
    // mpz_set_str skips white space anywhere in the string; only digits are acceptable here.
    for (char ch : s) {
        const unsigned char u = static_cast<unsigned char>(ch);
        if (base == 10 ? !std::isdigit(u) : !std::isxdigit(u)) return false;
    }
    mpz_class v;
    if (mpz_set_str(v.get_mpz_t(), s.c_str(), base) != 0) return false;
    if (negative) v = -v;
    if (v < 0) v = -v;
    if (v <= 1) return false;
    out = v;
    return true;
}

// Divides every known factor out of `g`.  Returns true when nothing new remains
// (g is 1 or a product of known factors); `g` is then left unchanged so that it can
// be shown as the known factor.  Otherwise `g` is replaced by the part that the known
// factors do not explain and false is returned.
inline bool strip(mpz_class& g, const std::vector<std::string>& known) {
    if (g <= 1) return true;
    std::vector<mpz_class> factors;
    for (const std::string& s : known) {
        mpz_class f;
        if (parse(s, f)) factors.push_back(f);
    }
    mpz_class rest = g;
    bool changed = true;
    while (changed && rest > 1) {
        changed = false;
        for (const mpz_class& f : factors) {
            while (rest > 1 && mpz_divisible_p(rest.get_mpz_t(), f.get_mpz_t())) {
                mpz_divexact(rest.get_mpz_t(), rest.get_mpz_t(), f.get_mpz_t());
                changed = true;
            }
        }
    }
    if (rest <= 1) return true;
    g = rest;
    return false;
}

} // namespace ecm_known
