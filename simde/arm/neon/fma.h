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
*   2021      Atharva Nimbalkar <atharvakn@gmail.com>
*/

#if !defined(SIMDE_ARM_NEON_FMA_H)
#define SIMDE_ARM_NEON_FMA_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
SIMDE_DISABLE_UNWANTED_DIAGNOSTICS
SIMDE_BEGIN_DECLS_

SIMDE_FUNCTION_ATTRIBUTES
simde_float32x2_t
simde_vfma_f32(simde_float32x2_t a, simde_float32x2_t b, simde_float32x2_t c) {
  #if defined(SIMDE_ARM_NEON_A32V7_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
    return vfma_f32(a, b, c);
  #else
    simde_float32x2_private
      a_ = simde_float32x2_to_private(a),
      b_ = simde_float32x2_to_private(b),
      c_ = simde_float32x2_to_private(c);

    SIMDE_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      a_.values[i] += b_.values[i] * c_.values[i];
    }

    return simde_float32x2_from_private(a_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vfma_f32
  #define vfma_f32(a, b, c) simde_vfma_f32(a, b, c)
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde_float32x4_t
simde_vfmaq_f32(simde_float32x4_t a, simde_float32x4_t b, simde_float32x4_t c) {
  #if defined(SIMDE_ARM_NEON_A32V7_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
    return vfmaq_f32(a, b, c);
  #elif defined(SIMDE_WASM_SIMD128_NATIVE)
    return wasm_f32x4_add(a, wasm_f32x4_mul(b, c));
  #else
    simde_float32x4_private
      a_ = simde_float32x4_to_private(a),
      b_ = simde_float32x4_to_private(b),
      c_ = simde_float32x4_to_private(c);

    SIMDE_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
      a_.values[i] += b_.values[i] * c_.values[i];
    }

    return simde_float32x4_from_private(a_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vfmaq_f32
  #define vfmaq_f32(a, b, c) simde_vfmaq_f32(a, b, c)
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde_float64x1_t
simde_vfma_f64(simde_float64x1_t a, simde_float64x1_t b, simde_float64x1_t c) {
  #if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
    return vfma_f64(a, b, c);
  #else
    simde_float64x1_private
      a_ = simde_float64x1_to_private(a),
      b_ = simde_float64x1_to_private(b),
      c_ = simde_float64x1_to_private(c);

    a_.values[0] += b_.values[0] * c_.values[0];

    return simde_float64x1_from_private(a_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vfma_f64
  #define vfma_f64(a, b, c) simde_vfma_f64(a, b, c)
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde_float64x2_t
simde_vfmaq_f64(simde_float64x2_t a, simde_float64x2_t b, simde_float64x2_t c) {
  #if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
    return vfmaq_f64(a, b, c);
  #elif defined(SIMDE_X86_FMA_NATIVE)
    return _mm_fmadd_pd(b, c, a);
  #elif defined(SIMDE_X86_SSE2_NATIVE)
    return _mm_add_pd(a, _mm_mul_pd(b, c));
  #elif defined(SIMDE_WASM_SIMD128_NATIVE)
    return wasm_f64x2_add(a, wasm_f64x2_mul(b, c));
  #else
    simde_float64x2_private
      a_ = simde_float64x2_to_private(a),
      b_ = simde_float64x2_to_private(b),
      c_ = simde_float64x2_to_private(c);

    #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
      a_.values += b_.values * c_.values;
    #else
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.values) / sizeof(a_.values[0])) ; i++) {
        a_.values[i] += b_.values[i] * c_.values[i];
      }
    #endif

    return simde_float64x2_from_private(a_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vfmaq_f64
  #define vfmaq_f64(a, b, c) simde_vfmaq_f64(a, b, c)
#endif

SIMDE_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(SIMDE_ARM_NEON_CMLA_H) */
