/* SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright:
 *   2020      Evan Nemerson <evan@nemerson.com>
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(SIMDE_X86_AVX512_EXPAND_H)
#define SIMDE_X86_AVX512_EXPAND_H

#include "types.h"
#include "mov.h"
#include "mov_mask.h"

HEDLEY_DIAGNOSTIC_PUSH
SIMDE_DISABLE_UNWANTED_DIAGNOSTICS
SIMDE_BEGIN_DECLS_

inline simde__m256i simde_mm256_maskz_expand_epi32(simde__mmask8 k, simde__m256i a) {
  simde__m256i_private
    a_ = simde__m256i_to_private(a);
  simde__m256i_private r_;

  size_t src_idx = 0;
  for (size_t i = 0; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
    if (k & (UINT64_C(1) << i)) {
      r_.i32[i] = a_.i32[src_idx++];
    } else {
      r_.i32[i] = 0;
    }
  }

  return simde__m256i_from_private(r_);
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
#undef _mm256_maskz_expand_epi32
#define _mm256_maskz_expand_epi32(k, a) simde_mm256_maskz_expand_epi32(k, a)
#endif

SIMDE_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(SIMDE_X86_AVX512_EXPAND_H) */
