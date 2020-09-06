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
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 */

#if !defined(SIMDE_X86_AVX512_CMP_H)
#define SIMDE_X86_AVX512_CMP_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"
#include "mov_mask.h"
#include "setzero.h"
#include "setone.h"

HEDLEY_DIAGNOSTIC_PUSH
SIMDE_DISABLE_UNWANTED_DIAGNOSTICS
SIMDE_BEGIN_DECLS_

typedef enum {
  SIMDE_MM_CMPINT_EQ    = 0,
  SIMDE_MM_CMPINT_LT    = 1,
  SIMDE_MM_CMPINT_LE    = 2,
  SIMDE_MM_CMPINT_FALSE = 3,
  SIMDE_MM_CMPINT_NE    = 4,
  SIMDE_MM_CMPINT_NLT   = 5,
  #define SIMDE_MM_CMPINT_GE SIMDE_MM_CMPINT_NLT
  SIMDE_MM_CMPINT_NLE   = 6,
  #define SIMDE_MM_CMPINT_GT SIMDE_MM_CMPINT_NLE
  SIMDE_MM_CMPINT_TRUE  = 7,
} SIMDE_MM_CMPINT_ENUM;

#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES) && !(defined(_MM_CMPINT_GE) || defined(_MM_CMPINT_NLT))
  typedef SIMDE_MM_CMPINT_ENUM _MM_CMPINT_ENUM;
  #define _MM_CMPINT_EQ    SIMDE_MM_CMPINT_EQ
  #define _MM_CMPINT_LT    SIMDE_MM_CMPINT_LT
  #define _MM_CMPINT_LE    SIMDE_MM_CMPINT_LE
  #define _MM_CMPINT_FALSE SIMDE_MM_CMPINT_FALSE
  #define _MM_CMPINT_NE    SIMDE_MM_CMPINT_NE
  #define _MM_CMPINT_NLT   SIMDE_MM_CMPINT_NLT
  #define _MM_CMPINT_GE    SIMDE_MM_CMPINT_GE
  #define _MM_CMPINT_NLE   SIMDE_MM_CMPINT_NLE
  #define _MM_CMPINT_GT    SIMDE_MM_CMPINT_GT
  #define _MM_CMPINT_TRUE  SIMDE_MM_CMPINT_TRUE
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask16
simde_mm_cmp_epi8_mask (simde__m128i a, simde__m128i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask16 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm_cmp_epi8_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #elif defined(SIMDE_X86_SSE2_NATIVE)
    switch (imm8) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm_movepi8_mask(simde_mm_cmpeq_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = simde_mm_movepi8_mask(simde_mm_cmplt_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = simde_mm_movepi8_mask(simde_mm_or_si128(simde_mm_cmplt_epi8(a, b), simde_mm_cmpeq_epi8(a, b)));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm_movepi8_mask(simde_mm_cmpeq_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = ~simde_mm_movepi8_mask(simde_mm_cmplt_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = ~simde_mm_movepi8_mask(simde_mm_or_si128(simde_mm_cmplt_epi8(a, b), simde_mm_cmpeq_epi8(a, b)));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
      default:
        {
          simde__m128i_private
            r_,
            a_ = simde__m128i_to_private(a),
            b_ = simde__m128i_to_private(b);

          switch(HEDLEY_STATIC_CAST(int, imm8)) {
            case SIMDE_MM_CMPINT_EQ:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vceqq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 == b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] == b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vcltq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 < b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] < b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vcleq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 <= b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] <= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vmvnq_u8(vceqq_s8(a_.neon_i8, b_.neon_i8));
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 != b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] != b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vcgeq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 >= b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] >= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u8 = vcgtq_s8(a_.neon_i8, b_.neon_i8);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 > b_.i8);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                  r_.i8[i] = (a_.i8[i] > b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                }
              #endif
              break;
          }

          r = simde_mm_movepi8_mask(simde__m128i_from_private(r_));
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_epi8_mask
  #define _mm_cmp_epi8_mask(a, b, imm8) simde_mm_cmp_epi8_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm_cmp_epi16_mask (simde__m128i a, simde__m128i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm_cmp_epi16_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #elif defined(SIMDE_X86_SSE2_NATIVE)
    switch (imm8) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm_movepi16_mask(simde_mm_cmpeq_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = simde_mm_movepi16_mask(simde_mm_cmplt_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = simde_mm_movepi16_mask(simde_mm_or_si128(simde_mm_cmplt_epi16(a, b), simde_mm_cmpeq_epi16(a, b)));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm_movepi16_mask(simde_mm_cmpeq_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = ~simde_mm_movepi16_mask(simde_mm_cmplt_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = ~simde_mm_movepi16_mask(simde_mm_or_si128(simde_mm_cmplt_epi16(a, b), simde_mm_cmpeq_epi16(a, b)));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
      default:
        {
          simde__m128i_private
            r_,
            a_ = simde__m128i_to_private(a),
            b_ = simde__m128i_to_private(b);

          switch(HEDLEY_STATIC_CAST(int, imm8)) {
            case SIMDE_MM_CMPINT_EQ:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vceqq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 == b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] == b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vcltq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 < b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vcleq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 <= b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] <= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vmvnq_u16(vceqq_s16(a_.neon_i16, b_.neon_i16));
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 != b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] != b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vcgeq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 >= b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] >= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u16 = vcgtq_s16(a_.neon_i16, b_.neon_i16);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 > b_.i16);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                  r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                }
              #endif
              break;
          }

          r = simde_mm_movepi16_mask(simde__m128i_from_private(r_));
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_epi16_mask
  #define _mm_cmp_epi16_mask(a, b, imm8) simde_mm_cmp_epi16_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm_cmp_epi32_mask (simde__m128i a, simde__m128i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm_cmp_epi32_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #elif defined(SIMDE_X86_SSE2_NATIVE)
    switch (imm8) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm_movepi32_mask(simde_mm_cmpeq_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = simde_mm_movepi32_mask(simde_mm_cmplt_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = simde_mm_movepi32_mask(simde_mm_or_si128(simde_mm_cmplt_epi32(a, b), simde_mm_cmpeq_epi32(a, b)));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = simde_mm_movepi32_mask(simde_mm_cmpeq_epi32(a, b)) ^ UINT8_C(0x0f);
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = simde_mm_movepi32_mask(simde_mm_cmplt_epi32(a, b)) ^ UINT8_C(0x0f);
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = simde_mm_movepi32_mask(simde_mm_or_si128(simde_mm_cmplt_epi32(a, b), simde_mm_cmpeq_epi32(a, b))) ^ UINT8_C(0x0f);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0x0f);
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0x0f);
        break;
      default:
        {
          simde__m128i_private
            r_,
            a_ = simde__m128i_to_private(a),
            b_ = simde__m128i_to_private(b);

          switch(HEDLEY_STATIC_CAST(int, imm8)) {
            case SIMDE_MM_CMPINT_EQ:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vceqq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 == b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] == b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vcltq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 < b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] < b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vcleq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 <= b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] <= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vmvnq_u32(vceqq_s32(a_.neon_i32, b_.neon_i32));
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 != b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] != b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLT:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vcgeq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 >= b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] >= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLE:
              #if defined(SIMDE_ARM_NEON_A32V7_NATIVE)
                r_.neon_u32 = vcgtq_s32(a_.neon_i32, b_.neon_i32);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 > b_.i32);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                  r_.i32[i] = (a_.i32[i] > b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                }
              #endif
              break;
          }

          r = simde_mm_movepi32_mask(simde__m128i_from_private(r_));
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_epi32_mask
  #define _mm_cmp_epi32_mask(a, b, imm8) simde_mm_cmp_epi32_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm_cmp_epi64_mask (simde__m128i a, simde__m128i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm_cmp_epi64_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = HEDLEY_STATIC_CAST(simde__mmask8, 0x03);
        break;
      default:
        {
          simde__m128i_private
            r_,
            a_ = simde__m128i_to_private(a),
            b_ = simde__m128i_to_private(b);

          switch(HEDLEY_STATIC_CAST(int, imm8)) {
            case SIMDE_MM_CMPINT_EQ:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vceqq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 == b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] == b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LT:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vcltq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 < b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] < b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_LE:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vcleq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 <= b_.i64);
              #elif defined(HEDLEY_GCC_VERSION) && !HEDLEY_GCC_VERSION_CHECK(4,8,0)
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #else
                SIMDE_VECTORIZE
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NE:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u32 = vmvnq_u32(vreinterpretq_u32_u64(vceqq_s64(a_.neon_i64, b_.neon_i64)));
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 != b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] != b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLT:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vcgeq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 >= b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] >= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
            case SIMDE_MM_CMPINT_NLE:
              #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
                r_.neon_u64 = vcgtq_s64(a_.neon_i64, b_.neon_i64);
              #elif defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 > b_.i64);
              #else
                for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                  r_.i64[i] = (a_.i64[i] > b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                }
              #endif
              break;
          }

          r = simde_mm_movepi64_mask(simde__m128i_from_private(r_));
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_epi64_mask
  #define _mm_cmp_epi64_mask(a, b, imm8) simde_mm_cmp_epi64_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask32
simde_mm256_cmp_epi8_mask (simde__m256i a, simde__m256i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask32 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm256_cmp_epi8_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #elif defined(SIMDE_X86_AVX2_NATIVE)
    switch (imm8) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm256_movepi8_mask(simde_mm256_cmpeq_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = ~simde_mm256_movepi8_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi8(a, b), simde_mm256_cmpeq_epi8(a, b)));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = ~simde_mm256_movepi8_mask(simde_mm256_cmpgt_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT32_C(0x00000000);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm256_movepi8_mask(simde_mm256_cmpeq_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = simde_mm256_movepi8_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi8(a, b), simde_mm256_cmpeq_epi8(a, b)));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = simde_mm256_movepi8_mask(simde_mm256_cmpgt_epi8(a, b));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT32_C(0xffffffff);
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT32_C(0x00000000);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT32_C(0xffffffff);
        break;
      default:
        {
          simde__m256i_private
            a_ = simde__m256i_to_private(a),
            b_ = simde__m256i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm_cmp_epi8_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
            }
          #else
            simde__m256i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 == b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] == b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 < b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] < b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 <= b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] <= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 != b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] != b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 >= b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] >= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 > b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] > b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
            }

            r = simde_mm256_movepi8_mask(simde__m256i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(128) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_epi8_mask
  #define _mm256_cmp_epi8_mask(a, b, imm8) simde_mm256_cmp_epi8_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask16
simde_mm256_cmp_epi16_mask (simde__m256i a, simde__m256i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask16 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm256_cmp_epi16_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #elif defined(SIMDE_X86_AVX2_NATIVE)
    switch (imm8) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm256_movepi16_mask(simde_mm256_cmpeq_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = ~simde_mm256_movepi16_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi16(a, b), simde_mm256_cmpeq_epi16(a, b)));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = ~simde_mm256_movepi16_mask(simde_mm256_cmpgt_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x0000);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm256_movepi16_mask(simde_mm256_cmpeq_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = simde_mm256_movepi16_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi16(a, b), simde_mm256_cmpeq_epi16(a, b)));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = simde_mm256_movepi16_mask(simde_mm256_cmpgt_epi16(a, b));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
      default:
        {
          simde__m256i_private
            a_ = simde__m256i_to_private(a),
            b_ = simde__m256i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm_cmp_epi16_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
            }
          #else
            simde__m256i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 == b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] == b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 < b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 <= b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] <= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 != b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] != b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 >= b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] >= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 > b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
            }

            r = simde_mm256_movepi16_mask(simde__m256i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(128) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_epi16_mask
  #define _mm256_cmp_epi16_mask(a, b, imm8) simde_mm256_cmp_epi16_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm256_cmp_epi32_mask (simde__m256i a, simde__m256i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm256_cmp_epi32_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #elif defined(SIMDE_X86_AVX2_NATIVE)
    switch (imm8) {
      case SIMDE_MM_CMPINT_EQ:
        r = simde_mm256_movepi32_mask(simde_mm256_cmpeq_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_LT:
        r = ~simde_mm256_movepi32_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi32(a, b), simde_mm256_cmpeq_epi32(a, b)));
        break;
      case SIMDE_MM_CMPINT_LE:
        r = ~simde_mm256_movepi32_mask(simde_mm256_cmpgt_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_NE:
        r = ~simde_mm256_movepi32_mask(simde_mm256_cmpeq_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_NLT:
        r = simde_mm256_movepi32_mask(simde_mm256_or_si256(simde_mm256_cmpgt_epi32(a, b), simde_mm256_cmpeq_epi32(a, b)));
        break;
      case SIMDE_MM_CMPINT_NLE:
        r = simde_mm256_movepi32_mask(simde_mm256_cmpgt_epi32(a, b));
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
    }
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
      default:
        {
          simde__m256i_private
            a_ = simde__m256i_to_private(a),
            b_ = simde__m256i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi32_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE) << (4 * i);
                }
                break;
            }
          #else
            simde__m256i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 == b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] == b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 < b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] < b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 <= b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] <= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 != b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] != b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 >= b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] >= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 > b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] > b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
            }

            r = simde_mm256_movepi32_mask(simde__m256i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(128) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_epi32_mask
  #define _mm256_cmp_epi32_mask(a, b, imm8) simde_mm256_cmp_epi32_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm256_cmp_epi64_mask (simde__m256i a, simde__m256i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm256_cmp_epi64_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x00);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0x0f);
        break;
      default:
        {
          simde__m256i_private
            a_ = simde__m256i_to_private(a),
            b_ = simde__m256i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(128)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_EQ) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LT) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_LE) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NE) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLT) << (2 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
                  r |= simde_mm_cmp_epi64_mask(a_.m128i[i], b_.m128i[i], SIMDE_MM_CMPINT_NLE) << (2 * i);
                }
                break;
            }
          #else
            simde__m256i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 == b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] == b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 < b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] < b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 <= b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 != b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] != b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 >= b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] >= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 > b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] > b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
            }

            r = simde_mm256_movepi64_mask(simde__m256i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(128) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_epi64_mask
  #define _mm256_cmp_epi64_mask(a, b, imm8) simde_mm256_cmp_epi64_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask64
simde_mm512_cmp_epi8_mask (simde__m512i a, simde__m512i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask64 r;

  #if defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm512_cmp_epi8_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT64_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT64_C(0xffffffffffffffff);
        break;
      default:
        {
          simde__m512i_private
            a_ = simde__m512i_to_private(a),
            b_ = simde__m512i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(256)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask32 t = simde_mm256_cmp_epi8_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask64, t) << (32 * i);
                }
                break;
            }
          #else
            simde__m512i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 == b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] == b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 < b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] < b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 <= b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] <= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 != b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] != b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 >= b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] >= b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 > b_.i8);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
                    r_.i8[i] = (a_.i8[i] > b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
                  }
                #endif
                break;
            }

            r = simde_mm512_movepi8_mask(simde__m512i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(256) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_epi8_mask
  #define _mm512_cmp_epi8_mask(a, b, imm8) simde_mm512_cmp_epi8_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask32
simde_mm512_cmp_epi16_mask (simde__m512i a, simde__m512i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask32 r;

  #if defined(SIMDE_X86_AVX512BW_NATIVE)
    SIMDE_CONSTIFY_8_(_mm512_cmp_epi16_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT32_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT32_C(0xffffffff);
        break;
      default:
        {
          simde__m512i_private
            a_ = simde__m512i_to_private(a),
            b_ = simde__m512i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(256)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask16 t = simde_mm256_cmp_epi16_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask32, t) << (16 * i);
                }
                break;
            }
          #else
            simde__m512i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 == b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] == b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 < b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 <= b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] <= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 != b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] != b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 >= b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] >= b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), a_.i16 > b_.i16);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
                    r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
                  }
                #endif
                break;
            }

            r = simde_mm512_movepi16_mask(simde__m512i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(256) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_epi16_mask
  #define _mm512_cmp_epi16_mask(a, b, imm8) simde_mm512_cmp_epi16_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask16
simde_mm512_cmp_epi32_mask (simde__m512i a, simde__m512i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask16 r;

  #if defined(SIMDE_X86_AVX512F_NATIVE)
    SIMDE_CONSTIFY_8_(_mm512_cmp_epi32_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT16_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT16_C(0xffff);
        break;
      default:
        {
          simde__m512i_private
            a_ = simde__m512i_to_private(a),
            b_ = simde__m512i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(256)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_EQ);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLT);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  const simde__mmask8 t = simde_mm256_cmp_epi32_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLE);
                  r |= HEDLEY_STATIC_CAST(simde__mmask16, t) << (8 * i);
                }
                break;
            }
          #else
            simde__m512i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 == b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] == b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 < b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] < b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 <= b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] <= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 != b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] != b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 >= b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] >= b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 > b_.i32);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
                    r_.i32[i] = (a_.i32[i] > b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
                  }
                #endif
                break;
            }

            r = simde_mm512_movepi32_mask(simde__m512i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(256) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_epi32_mask
  #define _mm512_cmp_epi32_mask(a, b, imm8) simde_mm512_cmp_epi32_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm512_cmp_epi64_mask (simde__m512i a, simde__m512i b, const SIMDE_MM_CMPINT_ENUM imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 7) {
  simde__mmask8 r;

  #if defined(SIMDE_X86_AVX512F_NATIVE)
    SIMDE_CONSTIFY_8_(_mm512_cmp_epi64_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
  #else
    switch (HEDLEY_STATIC_CAST(int, imm8)) {
      case SIMDE_MM_CMPINT_FALSE:
        r = UINT8_C(0x0);
        break;
      case SIMDE_MM_CMPINT_TRUE:
        r = UINT8_C(0xff);
        break;
      default:
        {
          simde__m512i_private
            a_ = simde__m512i_to_private(a),
            b_ = simde__m512i_to_private(b);

          #if SIMDE_NATURAL_VECTOR_SIZE_LE(256)
            r = 0;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_EQ) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LT) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_LE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_LE) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NE) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLT:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLT) << (4 * i);
                }
                break;
              case SIMDE_MM_CMPINT_NLE:
                for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
                  r |= simde_mm256_cmp_epi64_mask(a_.m256i[i], b_.m256i[i], SIMDE_MM_CMPINT_NLE) << (4 * i);
                }
                break;
            }
          #else
            simde__m512i_private r_;

            switch(HEDLEY_STATIC_CAST(int, imm8)) {
              case SIMDE_MM_CMPINT_EQ:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 == b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] == b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 < b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] < b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_LE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 <= b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] <= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 != b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] != b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLT:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 >= b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] >= b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
              case SIMDE_MM_CMPINT_NLE:
                #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
                  r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 > b_.i64);
                #else
                  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
                    r_.i64[i] = (a_.i64[i] > b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
                  }
                #endif
                break;
            }

            r = simde_mm512_movepi64_mask(simde__m512i_from_private(r_));
          #endif /* SIMDE_NATURAL_VECTOR_SIZE_LE(256) */
        }
      break;
    }
  #endif

  return r;
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_epi64_mask
  #define _mm512_cmp_epi64_mask(a, b, imm8) simde_mm512_cmp_epi64_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask16
simde_mm512_cmp_ps_mask (simde__m512 a, simde__m512 b, const int imm8)
    SIMDE_REQUIRE_CONSTANT(imm8)
    HEDLEY_REQUIRE_MSG(((imm8 >= 0) && (imm8 <= 31)), "imm8 must be one of the SIMDE_CMP_* macros (values: [0, 31])") {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    simde__mmask16 r;
    SIMDE_CONSTIFY_32_(_mm512_cmp_ps_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
    return r;
  #else
    simde__m512_private
      r_,
      a_ = simde__m512_to_private(a),
      b_ = simde__m512_to_private(b);

    #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
      switch (imm8) {
        case SIMDE_CMP_EQ_OQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 == b_.f32));
          break;
        case SIMDE_CMP_LT_OS:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 < b_.f32));
          break;
        case SIMDE_CMP_LE_OS:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 <= b_.f32));
          break;
        case SIMDE_CMP_UNORD_Q:
          #if defined(simde_math_isnanf)
            for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.u32[i] = (simde_math_isnanf(a_.f32[i]) || simde_math_isnanf(b_.f32[i])) ? ~UINT32_C(0) : UINT32_C(0);
            }
          #else
            HEDLEY_UNREACHABLE();
          #endif
          break;
        case SIMDE_CMP_NEQ_UQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 != b_.f32));
          break;
        case SIMDE_CMP_NLT_US:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 >= b_.f32));
          break;
        case SIMDE_CMP_NLE_US:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 > b_.f32));
          break;
        case SIMDE_CMP_ORD_Q:
          #if defined(simde_math_isnanf)
            for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.u32[i] = (!simde_math_isnanf(a_.f32[i]) && !simde_math_isnanf(b_.f32[i])) ? ~UINT32_C(0) : UINT32_C(0);
              }
          #else
            HEDLEY_UNREACHABLE();
          #endif
          break;
        case SIMDE_CMP_EQ_UQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 == b_.f32));
          break;
        case SIMDE_CMP_NGE_US:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 < b_.f32));
          break;
        case SIMDE_CMP_NGT_US:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 <= b_.f32));
          break;
        case SIMDE_CMP_FALSE_OQ:
          r_ = simde__m512_to_private(simde_mm512_setzero_ps());
          break;
        case SIMDE_CMP_NEQ_OQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 != b_.f32));
          break;
        case SIMDE_CMP_GE_OS:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 >= b_.f32));
          break;
        case SIMDE_CMP_GT_OS:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 > b_.f32));
          break;
        case SIMDE_CMP_TRUE_UQ:
          r_ = simde__m512_to_private(simde_x_mm512_setone_ps());
          break;
        case SIMDE_CMP_EQ_OS:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 == b_.f32));
          break;
        case SIMDE_CMP_LT_OQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 < b_.f32));
          break;
        case SIMDE_CMP_LE_OQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 <= b_.f32));
          break;
        case SIMDE_CMP_UNORD_S:
          #if defined(simde_math_isnanf)
            for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
                r_.u32[i] = (simde_math_isnanf(a_.f32[i]) || simde_math_isnanf(b_.f32[i])) ? ~UINT32_C(0) : UINT32_C(0);
            }
          #else
            HEDLEY_UNREACHABLE();
          #endif
          break;
        case SIMDE_CMP_NEQ_US:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 != b_.f32));
          break;
        case SIMDE_CMP_NLT_UQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 >= b_.f32));
          break;
        case SIMDE_CMP_NLE_UQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 > b_.f32));
          break;
        case SIMDE_CMP_ORD_S:
          #if defined(simde_math_isnanf)
            for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
              r_.u32[i] = (simde_math_isnanf(a_.f32[i]) || simde_math_isnanf(b_.f32[i])) ? UINT32_C(0) : ~UINT32_C(0);
            }
          #else
            HEDLEY_UNREACHABLE();
          #endif
          break;
        case SIMDE_CMP_EQ_US:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 == b_.f32));
          break;
        case SIMDE_CMP_NGE_UQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 < b_.f32));
          break;
        case SIMDE_CMP_NGT_UQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 <= b_.f32));
          break;
        case SIMDE_CMP_FALSE_OS:
          r_ = simde__m512_to_private(simde_mm512_setzero_ps());
          break;
        case SIMDE_CMP_NEQ_OS:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 != b_.f32));
          break;
        case SIMDE_CMP_GE_OQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 >= b_.f32));
          break;
        case SIMDE_CMP_GT_OQ:
          r_.i32 = HEDLEY_STATIC_CAST(__typeof__(r_.i32), (a_.f32 > b_.f32));
          break;
        case SIMDE_CMP_TRUE_US:
          r_ = simde__m512_to_private(simde_x_mm512_setone_ps());
          break;
        default:
          HEDLEY_UNREACHABLE();
          break;
      }
    #else /* defined(SIMDE_VECTOR_SUBSCRIPT_OPS) */
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        switch (imm8) {
          case SIMDE_CMP_EQ_OQ:
            r_.u32[i] = (a_.f32[i] == b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_LT_OS:
            r_.u32[i] = (a_.f32[i] < b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_LE_OS:
            r_.u32[i] = (a_.f32[i] <= b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_UNORD_Q:
            #if defined(simde_math_isnanf)
              r_.u32[i] = (simde_math_isnanf(a_.f32[i]) || simde_math_isnanf(b_.f32[i])) ? ~UINT32_C(0) : UINT32_C(0);
            #else
              HEDLEY_UNREACHABLE();
            #endif
            break;
          case SIMDE_CMP_NEQ_UQ:
            r_.u32[i] = (a_.f32[i] != b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_NLT_US:
            r_.u32[i] = (a_.f32[i] >= b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_NLE_US:
            r_.u32[i] = (a_.f32[i] > b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_ORD_Q:
            #if defined(simde_math_isnanf)
              r_.u32[i] = (!simde_math_isnanf(a_.f32[i]) && !simde_math_isnanf(b_.f32[i])) ? ~UINT32_C(0) : UINT32_C(0);
            #else
              HEDLEY_UNREACHABLE();
            #endif
            break;
          case SIMDE_CMP_EQ_UQ:
            r_.u32[i] = (a_.f32[i] == b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_NGE_US:
            r_.u32[i] = (a_.f32[i] < b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_NGT_US:
            r_.u32[i] = (a_.f32[i] <= b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_FALSE_OQ:
            r_.u32[i] = UINT32_C(0);
            break;
          case SIMDE_CMP_NEQ_OQ:
            r_.u32[i] = (a_.f32[i] != b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_GE_OS:
            r_.u32[i] = (a_.f32[i] >= b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_GT_OS:
            r_.u32[i] = (a_.f32[i] > b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_TRUE_UQ:
            r_.u32[i] = ~UINT32_C(0);
            break;
          case SIMDE_CMP_EQ_OS:
            r_.u32[i] = (a_.f32[i] == b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_LT_OQ:
            r_.u32[i] = (a_.f32[i] < b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_LE_OQ:
            r_.u32[i] = (a_.f32[i] <= b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_UNORD_S:
            #if defined(simde_math_isnanf)
              r_.u32[i] = (simde_math_isnanf(a_.f32[i]) || simde_math_isnanf(b_.f32[i])) ? ~UINT32_C(0) : UINT32_C(0);
            #else
              HEDLEY_UNREACHABLE();
            #endif
            break;
          case SIMDE_CMP_NEQ_US:
            r_.u32[i] = (a_.f32[i] != b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_NLT_UQ:
            r_.u32[i] = (a_.f32[i] >= b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_NLE_UQ:
            r_.u32[i] = (a_.f32[i] > b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_ORD_S:
            #if defined(simde_math_isnanf)
              r_.u32[i] = (simde_math_isnanf(a_.f32[i]) || simde_math_isnanf(b_.f32[i])) ? UINT32_C(0) : ~UINT32_C(0);
            #else
              HEDLEY_UNREACHABLE();
            #endif
            break;
          case SIMDE_CMP_EQ_US:
            r_.u32[i] = (a_.f32[i] == b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_NGE_UQ:
            r_.u32[i] = (a_.f32[i] < b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_NGT_UQ:
            r_.u32[i] = (a_.f32[i] <= b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_FALSE_OS:
            r_.u32[i] = UINT32_C(0);
            break;
          case SIMDE_CMP_NEQ_OS:
            r_.u32[i] = (a_.f32[i] != b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_GE_OQ:
            r_.u32[i] = (a_.f32[i] >= b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_GT_OQ:
            r_.u32[i] = (a_.f32[i] > b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
            break;
          case SIMDE_CMP_TRUE_US:
            r_.u32[i] = ~UINT32_C(0);
            break;
          default:
            HEDLEY_UNREACHABLE();
            break;
        }
      }
    #endif

    return simde_mm512_movepi32_mask(simde_mm512_castps_si512(simde__m512_from_private(r_)));
  #endif
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_ps_mask
  #define _mm512_cmp_ps_mask(a, b, imm8) simde_mm512_cmp_ps_mask((a), (b), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__mmask8
simde_mm512_cmp_pd_mask (simde__m512d a, simde__m512d b, const int imm8)
    SIMDE_REQUIRE_CONSTANT(imm8)
    HEDLEY_REQUIRE_MSG(((imm8 >= 0) && (imm8 <= 31)), "imm8 must be one of the SIMDE_CMP_* macros (values: [0, 31])") {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    simde__mmask8 r;
    SIMDE_CONSTIFY_32_(_mm512_cmp_pd_mask, r, (HEDLEY_UNREACHABLE(), 0), HEDLEY_STATIC_CAST(int, imm8), a, b);
    return r;
  #else
    simde__m512d_private
      r_,
      a_ = simde__m512d_to_private(a),
      b_ = simde__m512d_to_private(b);

    #if defined(SIMDE_VECTOR_SUBSCRIPT_OPS)
      switch (imm8) {
        case SIMDE_CMP_EQ_OQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 == b_.f64));
          break;
        case SIMDE_CMP_LT_OS:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 < b_.f64));
          break;
        case SIMDE_CMP_LE_OS:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 <= b_.f64));
          break;
        case SIMDE_CMP_UNORD_Q:
          #if defined(simde_math_isnanf)
            for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.u64[i] = (simde_math_isnanf(a_.f64[i]) || simde_math_isnanf(b_.f64[i])) ? ~UINT64_C(0) : UINT64_C(0);
            }
          #else
            HEDLEY_UNREACHABLE();
          #endif
          break;
        case SIMDE_CMP_NEQ_UQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 != b_.f64));
          break;
        case SIMDE_CMP_NLT_US:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 >= b_.f64));
          break;
        case SIMDE_CMP_NLE_US:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 > b_.f64));
          break;
        case SIMDE_CMP_ORD_Q:
          #if defined(simde_math_isnanf)
            for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.u64[i] = (!simde_math_isnanf(a_.f64[i]) && !simde_math_isnanf(b_.f64[i])) ? ~UINT64_C(0) : UINT64_C(0);
              }
          #else
            HEDLEY_UNREACHABLE();
          #endif
          break;
        case SIMDE_CMP_EQ_UQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 == b_.f64));
          break;
        case SIMDE_CMP_NGE_US:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 < b_.f64));
          break;
        case SIMDE_CMP_NGT_US:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 <= b_.f64));
          break;
        case SIMDE_CMP_FALSE_OQ:
          r_ = simde__m512d_to_private(simde_mm512_setzero_pd());
          break;
        case SIMDE_CMP_NEQ_OQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 != b_.f64));
          break;
        case SIMDE_CMP_GE_OS:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 >= b_.f64));
          break;
        case SIMDE_CMP_GT_OS:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 > b_.f64));
          break;
        case SIMDE_CMP_TRUE_UQ:
          r_ = simde__m512d_to_private(simde_x_mm512_setone_pd());
          break;
        case SIMDE_CMP_EQ_OS:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 == b_.f64));
          break;
        case SIMDE_CMP_LT_OQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 < b_.f64));
          break;
        case SIMDE_CMP_LE_OQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 <= b_.f64));
          break;
        case SIMDE_CMP_UNORD_S:
          #if defined(simde_math_isnanf)
            for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
                r_.u64[i] = (simde_math_isnanf(a_.f64[i]) || simde_math_isnanf(b_.f64[i])) ? ~UINT64_C(0) : UINT64_C(0);
            }
          #else
            HEDLEY_UNREACHABLE();
          #endif
          break;
        case SIMDE_CMP_NEQ_US:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 != b_.f64));
          break;
        case SIMDE_CMP_NLT_UQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 >= b_.f64));
          break;
        case SIMDE_CMP_NLE_UQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 > b_.f64));
          break;
        case SIMDE_CMP_ORD_S:
          #if defined(simde_math_isnanf)
            for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
              r_.u64[i] = (simde_math_isnanf(a_.f64[i]) || simde_math_isnanf(b_.f64[i])) ? UINT64_C(0) : ~UINT64_C(0);
            }
          #else
            HEDLEY_UNREACHABLE();
          #endif
          break;
        case SIMDE_CMP_EQ_US:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 == b_.f64));
          break;
        case SIMDE_CMP_NGE_UQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 < b_.f64));
          break;
        case SIMDE_CMP_NGT_UQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 <= b_.f64));
          break;
        case SIMDE_CMP_FALSE_OS:
          r_ = simde__m512d_to_private(simde_mm512_setzero_pd());
          break;
        case SIMDE_CMP_NEQ_OS:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 != b_.f64));
          break;
        case SIMDE_CMP_GE_OQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 >= b_.f64));
          break;
        case SIMDE_CMP_GT_OQ:
          r_.i64 = HEDLEY_STATIC_CAST(__typeof__(r_.i64), (a_.f64 > b_.f64));
          break;
        case SIMDE_CMP_TRUE_US:
          r_ = simde__m512d_to_private(simde_x_mm512_setone_pd());
          break;
        default:
          HEDLEY_UNREACHABLE();
          break;
      }
    #else /* defined(SIMDE_VECTOR_SUBSCRIPT_OPS) */
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        switch (imm8) {
          case SIMDE_CMP_EQ_OQ:
            r_.u64[i] = (a_.f64[i] == b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_LT_OS:
            r_.u64[i] = (a_.f64[i] < b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_LE_OS:
            r_.u64[i] = (a_.f64[i] <= b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_UNORD_Q:
            #if defined(simde_math_isnanf)
              r_.u64[i] = (simde_math_isnanf(a_.f64[i]) || simde_math_isnanf(b_.f64[i])) ? ~UINT64_C(0) : UINT64_C(0);
            #else
              HEDLEY_UNREACHABLE();
            #endif
            break;
          case SIMDE_CMP_NEQ_UQ:
            r_.u64[i] = (a_.f64[i] != b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_NLT_US:
            r_.u64[i] = (a_.f64[i] >= b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_NLE_US:
            r_.u64[i] = (a_.f64[i] > b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_ORD_Q:
            #if defined(simde_math_isnanf)
              r_.u64[i] = (!simde_math_isnanf(a_.f64[i]) && !simde_math_isnanf(b_.f64[i])) ? ~UINT64_C(0) : UINT64_C(0);
            #else
              HEDLEY_UNREACHABLE();
            #endif
            break;
          case SIMDE_CMP_EQ_UQ:
            r_.u64[i] = (a_.f64[i] == b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_NGE_US:
            r_.u64[i] = (a_.f64[i] < b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_NGT_US:
            r_.u64[i] = (a_.f64[i] <= b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_FALSE_OQ:
            r_.u64[i] = UINT64_C(0);
            break;
          case SIMDE_CMP_NEQ_OQ:
            r_.u64[i] = (a_.f64[i] != b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_GE_OS:
            r_.u64[i] = (a_.f64[i] >= b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_GT_OS:
            r_.u64[i] = (a_.f64[i] > b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_TRUE_UQ:
            r_.u64[i] = ~UINT64_C(0);
            break;
          case SIMDE_CMP_EQ_OS:
            r_.u64[i] = (a_.f64[i] == b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_LT_OQ:
            r_.u64[i] = (a_.f64[i] < b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_LE_OQ:
            r_.u64[i] = (a_.f64[i] <= b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_UNORD_S:
            #if defined(simde_math_isnanf)
              r_.u64[i] = (simde_math_isnanf(a_.f64[i]) || simde_math_isnanf(b_.f64[i])) ? ~UINT64_C(0) : UINT64_C(0);
            #else
              HEDLEY_UNREACHABLE();
            #endif
            break;
          case SIMDE_CMP_NEQ_US:
            r_.u64[i] = (a_.f64[i] != b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_NLT_UQ:
            r_.u64[i] = (a_.f64[i] >= b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_NLE_UQ:
            r_.u64[i] = (a_.f64[i] > b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_ORD_S:
            #if defined(simde_math_isnanf)
              r_.u64[i] = (simde_math_isnanf(a_.f64[i]) || simde_math_isnanf(b_.f64[i])) ? UINT64_C(0) : ~UINT64_C(0);
            #else
              HEDLEY_UNREACHABLE();
            #endif
            break;
          case SIMDE_CMP_EQ_US:
            r_.u64[i] = (a_.f64[i] == b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_NGE_UQ:
            r_.u64[i] = (a_.f64[i] < b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_NGT_UQ:
            r_.u64[i] = (a_.f64[i] <= b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_FALSE_OS:
            r_.u64[i] = UINT64_C(0);
            break;
          case SIMDE_CMP_NEQ_OS:
            r_.u64[i] = (a_.f64[i] != b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_GE_OQ:
            r_.u64[i] = (a_.f64[i] >= b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_GT_OQ:
            r_.u64[i] = (a_.f64[i] > b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
            break;
          case SIMDE_CMP_TRUE_US:
            r_.u64[i] = ~UINT64_C(0);
            break;
          default:
            HEDLEY_UNREACHABLE();
            break;
        }
      }
    #endif

    return simde_mm512_movepi64_mask(simde_mm512_castpd_si512(simde__m512d_from_private(r_)));
  #endif
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmp_pd_mask
  #define _mm512_cmp_pd_mask(a, b, imm8) simde_mm512_cmp_pd_mask((a), (b), (imm8))
#endif

SIMDE_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(SIMDE_X86_AVX512_CMP_H) */
