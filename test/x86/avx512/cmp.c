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

#define SIMDE_TEST_X86_AVX512_INSN cmp

#include <test/x86/avx512/test-avx512.h>
#include <simde/x86/avx512/set.h>
#include <simde/x86/avx512/cmp.h>

#if !defined(SIMDE_NATIVE_ALIASES_TESTING)

static int
test_simde_mm_cmp_epi8_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[16];
    const int8_t b[16];
    const simde__mmask16 r;
  } test_vec[] = {
    { { -INT8_C(  24), -INT8_C(  41), -INT8_C(  37), -INT8_C(  97),  INT8_C( 107),  INT8_C(  15),  INT8_C( 121),  INT8_C( 120),
        -INT8_C(   1), -INT8_C(  79), -INT8_C(  17),  INT8_C(  41),  INT8_C(   3), -INT8_C(  45),  INT8_C(  37),  INT8_C( 110) },
      { -INT8_C(  24),  INT8_C(  26), -INT8_C(  20),  INT8_C(  87),  INT8_C(   7), -INT8_C(  15),  INT8_C( 121), -INT8_C(  17),
        -INT8_C(   1),  INT8_C(  40), -INT8_C(  17),  INT8_C( 124),  INT8_C(   4),  INT8_C(  93), -INT8_C(  25),  INT8_C(  95) },
      UINT16_C( 1345) },
    { {  INT8_C(  56), -INT8_C( 107), -INT8_C( 115),      INT8_MAX, -INT8_C( 121),  INT8_C( 103),  INT8_C(  47), -INT8_C( 122),
         INT8_C(  46),  INT8_C(  30), -INT8_C( 118),  INT8_C(  50),  INT8_C( 123), -INT8_C(  22), -INT8_C( 111), -INT8_C(  99) },
      { -INT8_C(  83), -INT8_C( 113),  INT8_C(  79),      INT8_MAX,  INT8_C(  22),  INT8_C( 103),  INT8_C(  79),  INT8_C(  79),
        -INT8_C(  31),  INT8_C(  71), -INT8_C( 118),  INT8_C(  46),  INT8_C( 101),  INT8_C(  52),  INT8_C( 100), -INT8_C(  99) },
      UINT16_C(25300) },
    { {  INT8_C( 121), -INT8_C(  92), -INT8_C(  73),  INT8_C( 104),  INT8_C(  11),  INT8_C(  63), -INT8_C(  34), -INT8_C(  20),
        -INT8_C( 122), -INT8_C(  26),  INT8_C(  27),  INT8_C(  43), -INT8_C(  99),      INT8_MAX, -INT8_C( 119),  INT8_C(  72) },
      {  INT8_C( 113),  INT8_C( 102), -INT8_C(  73),  INT8_C( 104),  INT8_C( 114), -INT8_C( 114), -INT8_C( 114), -INT8_C(  99),
         INT8_C( 103), -INT8_C(  26),  INT8_C(  67),  INT8_C(  43), -INT8_C(  49), -INT8_C( 104), -INT8_C( 101),  INT8_C(  72) },
      UINT16_C(57118) },
    { {  INT8_C(  33), -INT8_C(  38), -INT8_C(  87), -INT8_C(  20),  INT8_C( 104),  INT8_C(  55),  INT8_C(  61), -INT8_C(  49),
         INT8_C(  29),      INT8_MAX, -INT8_C(  97), -INT8_C(  20),  INT8_C(  64), -INT8_C( 106),  INT8_C(  53),  INT8_C(  85) },
      {  INT8_C(  33),  INT8_C(  13), -INT8_C(  99), -INT8_C(  20), -INT8_C(  61), -INT8_C(  46),  INT8_C(  61), -INT8_C(  29),
         INT8_C(  34),  INT8_C( 122), -INT8_C(  97), -INT8_C(  14),  INT8_C(  64), -INT8_C(  62),  INT8_C(  50),  INT8_C( 109) },
      UINT16_C(    0) },
    { {  INT8_C(  94), -INT8_C(  45),  INT8_C( 114),  INT8_C(  33), -INT8_C(  90), -INT8_C(  81),  INT8_C(   4), -INT8_C(  56),
         INT8_C(  41), -INT8_C(  92), -INT8_C(  41),  INT8_C( 105),  INT8_C( 102), -INT8_C(  19), -INT8_C(  41),  INT8_C(   3) },
      { -INT8_C(  56), -INT8_C(  28),  INT8_C(   7), -INT8_C(  37),  INT8_C(  45), -INT8_C(  37),  INT8_C(  10), -INT8_C(  10),
        -INT8_C(  86),  INT8_C(  38), -INT8_C(  41),  INT8_C(  14),  INT8_C( 119),  INT8_C(  13), -INT8_C( 108), -INT8_C(  43) },
      UINT16_C(64511) },
    { {  INT8_C(   3),  INT8_C(  16), -INT8_C( 102),  INT8_C(  48), -INT8_C(  20), -INT8_C(  91), -INT8_C(  45), -INT8_C( 106),
        -INT8_C(  53),  INT8_C(  27), -INT8_C(  92),  INT8_C(  67),  INT8_C(  12),  INT8_C(  57), -INT8_C( 124), -INT8_C(  19) },
      {  INT8_C(  63),  INT8_C(  15),  INT8_C( 116), -INT8_C(  11),  INT8_C(  11), -INT8_C(  61), -INT8_C(  45), -INT8_C(  86),
        -INT8_C(  51),  INT8_C(  27), -INT8_C(  80), -INT8_C(  60),  INT8_C(  58), -INT8_C(  71), -INT8_C( 124),  INT8_C(  61) },
      UINT16_C(27210) },
    { { -INT8_C(   8),  INT8_C(   0),  INT8_C(  94), -INT8_C(  68), -INT8_C(  56),  INT8_C(  49), -INT8_C(  81), -INT8_C( 111),
         INT8_C(  77),  INT8_C(  96), -INT8_C(   5), -INT8_C( 121),  INT8_C(  25), -INT8_C(  38), -INT8_C(  59), -INT8_C(  29) },
      { -INT8_C(   8),  INT8_C(  51), -INT8_C( 103), -INT8_C(  68), -INT8_C(  56), -INT8_C(  27),  INT8_C(  75),  INT8_C(  91),
        -INT8_C(  42),  INT8_C(  29), -INT8_C(   5), -INT8_C(   1),  INT8_C(   7), -INT8_C( 121),  INT8_C( 104),  INT8_C(   1) },
      UINT16_C(13092) },
    { { -INT8_C( 106), -INT8_C(  84), -INT8_C(  62), -INT8_C( 116), -INT8_C( 110),  INT8_C(  13), -INT8_C(  24),  INT8_C(   5),
         INT8_C(  42), -INT8_C(  29),  INT8_C( 103),  INT8_C(  93),  INT8_C( 106),  INT8_C(  72),  INT8_C(  51), -INT8_C(  10) },
      { -INT8_C( 106),  INT8_C(  57),  INT8_C(  62), -INT8_C( 114), -INT8_C(  17),  INT8_C(  28), -INT8_C(  45),  INT8_C(   5),
         INT8_C(  79), -INT8_C(  96),  INT8_C(  53),  INT8_C(  93),  INT8_C(  49),  INT8_C(  72),  INT8_C(  99), -INT8_C(  10) },
           UINT16_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128i a = simde_mm_loadu_epi8(test_vec[i].a);
    simde__m128i b = simde_mm_loadu_epi8(test_vec[i].b);
    simde__mmask16 r = simde_mm_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a_[16];
    int8_t b_[16];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m128i a = simde_mm_loadu_epi8(a_);
    simde__m128i b = simde_mm_loadu_epi8(b_);
    simde__mmask16 r = simde_mm_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i8x16(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i8x16(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask16(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm_cmp_epi16_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[8];
    const int16_t b[8];
    const simde__mmask8 r;
  } test_vec[] = {
    { { -INT16_C(  4039),  INT16_C( 12767),  INT16_C( 15592),  INT16_C( 15098),  INT16_C(  6901),  INT16_C(  3425),  INT16_C( 11846),  INT16_C(  3579) },
      { -INT16_C(  4039),  INT16_C( 31584),  INT16_C(  3681),  INT16_C( 15098),  INT16_C(  6901),  INT16_C(  6239),  INT16_C( 11846), -INT16_C( 10122) },
      UINT8_C( 89) },
    { { -INT16_C(  9544),  INT16_C(  6413),  INT16_C( 29956), -INT16_C(  2798), -INT16_C(  8221),  INT16_C( 15202),  INT16_C( 27617),  INT16_C( 12607) },
      {  INT16_C( 18624),  INT16_C( 21097),  INT16_C( 29956), -INT16_C(  2798), -INT16_C(  8221),  INT16_C( 15202), -INT16_C(  9049),  INT16_C( 24780) },
      UINT8_C(131) },
    { { -INT16_C(  1040), -INT16_C(  2832),  INT16_C(   625), -INT16_C( 20457),  INT16_C( 32619), -INT16_C( 30577),  INT16_C( 23335), -INT16_C(  8472) },
      {  INT16_C( 25141),  INT16_C(  5501), -INT16_C(  1745), -INT16_C( 20457),  INT16_C( 32619),  INT16_C( 18705), -INT16_C( 23513),  INT16_C(  6119) },
      UINT8_C(187) },
    { {  INT16_C( 31023), -INT16_C( 17255), -INT16_C( 20292), -INT16_C( 20542),  INT16_C( 32304),  INT16_C( 22385),  INT16_C( 22562), -INT16_C( 16018) },
      {  INT16_C( 31023),  INT16_C(  2258),  INT16_C( 13934), -INT16_C( 20542),  INT16_C(  1066),  INT16_C( 30999),  INT16_C( 23855), -INT16_C( 17155) },
      UINT8_C(  0) },
    { {  INT16_C(  6581),  INT16_C(  9220), -INT16_C( 14769),  INT16_C( 31187),  INT16_C(  4797), -INT16_C( 29698), -INT16_C(  4281),  INT16_C( 26551) },
      {  INT16_C( 12422), -INT16_C( 12989),  INT16_C( 18453),  INT16_C( 31044),  INT16_C(  4797), -INT16_C( 29698),  INT16_C( 17742),  INT16_C(   903) },
      UINT8_C(207) },
    { { -INT16_C( 10497), -INT16_C( 13090), -INT16_C( 24546),  INT16_C(  2794), -INT16_C( 29518),  INT16_C( 10550), -INT16_C( 14127),  INT16_C( 12292) },
      {  INT16_C( 11130), -INT16_C( 13090),  INT16_C(  1318),  INT16_C(  2794),  INT16_C(   543),  INT16_C( 10550), -INT16_C( 14127), -INT16_C( 12104) },
      UINT8_C(234) },
    { { -INT16_C(  8808), -INT16_C( 16845),  INT16_C(  7651),  INT16_C(   712),  INT16_C( 32696), -INT16_C(  4053), -INT16_C(  6969),  INT16_C( 26048) },
      { -INT16_C( 23303), -INT16_C( 20958), -INT16_C( 17898),  INT16_C(  5142),  INT16_C( 32696), -INT16_C( 23068), -INT16_C(  7189), -INT16_C( 31989) },
      UINT8_C(231) },
    { {  INT16_C( 22479),  INT16_C(  5158),  INT16_C( 29713), -INT16_C( 18802), -INT16_C(  1633), -INT16_C(  8594), -INT16_C( 17885), -INT16_C(  3580) },
      { -INT16_C( 23624),  INT16_C(  5158), -INT16_C( 12883), -INT16_C( 18802), -INT16_C(  1633),  INT16_C( 21893), -INT16_C( 17885), -INT16_C(  3580) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128i a = simde_mm_loadu_epi16(test_vec[i].a);
    simde__m128i b = simde_mm_loadu_epi16(test_vec[i].b);
    simde__mmask8 r = simde_mm_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[8];
    int16_t b_[8];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i16() & 3))
        a_[j] = b_[j];

    simde__m128i a = simde_mm_loadu_epi16(a_);
    simde__m128i b = simde_mm_loadu_epi16(b_);
    simde__mmask8 r = simde_mm_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i16x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i16x8(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm_cmp_epi32_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[4];
    const int32_t b[4];
    const simde__mmask8 r;
  } test_vec[] = {
    { { -INT32_C(   145519450),  INT32_C(   296776414), -INT32_C(  1519279458), -INT32_C(   952076455) },
      { -INT32_C(   145519450),  INT32_C(  1257134046), -INT32_C(   998466168), -INT32_C(  1248046432) },
      UINT8_C(  1) },
    { { -INT32_C(   111893733), -INT32_C(   835616865), -INT32_C(  2142781472),  INT32_C(   718579215) },
      { -INT32_C(  2018504068), -INT32_C(   835616865),  INT32_C(  2002157759),  INT32_C(   718579215) },
      UINT8_C(  4) },
    { { -INT32_C(  1292576493),  INT32_C(   411051352), -INT32_C(  1265641819), -INT32_C(   975281284) },
      { -INT32_C(   133627508), -INT32_C(  1924870297), -INT32_C(   859565860),  INT32_C(   607776273) },
      UINT8_C( 13) },
    { { -INT32_C(   173049715), -INT32_C(  1801235353),  INT32_C(  1161711899), -INT32_C(    63297228) },
      { -INT32_C(  1441001129), -INT32_C(  1801235353),  INT32_C(  1161711899),  INT32_C(  1794944477) },
      UINT8_C(  0) },
    { {  INT32_C(  1839821830), -INT32_C(   486388536),  INT32_C(  2049523869),  INT32_C(   317007341) },
      {  INT32_C(  1904100561),  INT32_C(  1310068006),  INT32_C(   866832523), -INT32_C(   885980219) },
      UINT8_C( 15) },
    { { -INT32_C(  1900677016), -INT32_C(  1864578299),  INT32_C(   595851953),  INT32_C(    19723658) },
      { -INT32_C(   163486257),  INT32_C(   465266080),  INT32_C(   595851953), -INT32_C(  2040005346) },
      UINT8_C( 12) },
    { {  INT32_C(   369706357), -INT32_C(   496226967), -INT32_C(   595610178), -INT32_C(  1469847886) },
      { -INT32_C(   375228414), -INT32_C(   496226967), -INT32_C(  1075556484), -INT32_C(   707598241) },
      UINT8_C(  5) },
    { { -INT32_C(  2138649066),  INT32_C(  2019750652), -INT32_C(  1590213054),  INT32_C(  1568016943) },
      { -INT32_C(  2043321882),  INT32_C(  1340435070), -INT32_C(  1194122978), -INT32_C(   699007041) },
      UINT8_C( 15) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128i a = simde_mm_loadu_epi32(test_vec[i].a);
    simde__m128i b = simde_mm_loadu_epi32(test_vec[i].b);
    simde__mmask8 r = simde_mm_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[4];
    int32_t b_[4];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i32() & 3))
        a_[j] = b_[j];

    simde__m128i a = simde_mm_loadu_epi32(a_);
    simde__m128i b = simde_mm_loadu_epi32(b_);
    simde__mmask8 r = simde_mm_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i32x4(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i32x4(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm_cmp_epi64_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const int64_t b[2];
    const simde__mmask8 r;
  } test_vec[] = {
    { {  INT64_C( 1933546251694643553), -INT64_C( 6674426040188389969) },
      {  INT64_C( 3316585011335538905), -INT64_C( 1397919465970533867) },
      UINT8_C(  0) },
    { {  INT64_C( 5277093564638603313), -INT64_C( 1315097322887719273) },
      { -INT64_C( 6872449015084218189), -INT64_C( 4405021074379975170) },
      UINT8_C(  0) },
    { {  INT64_C( 6616989965536034123), -INT64_C( 2178074625616115189) },
      {  INT64_C( 6616989965536034123),  INT64_C( 8492415186196017232) },
      UINT8_C(  3) },
    { {  INT64_C( 2989634292083645491),  INT64_C( 8976485222817187412) },
      { -INT64_C( 4528093087150505358),  INT64_C( 8976485222817187412) },
      UINT8_C(  0) },
    { { -INT64_C( 8348038954527816028),  INT64_C( 9188914379591628719) },
      { -INT64_C( 2868702623850210345), -INT64_C(  974227990175860470) },
      UINT8_C(  3) },
    { { -INT64_C( 3067000787965063319), -INT64_C( 8674135663512655445) },
      {  INT64_C( 4593578800997777889),  INT64_C( 5885977995198498453) },
      UINT8_C(  0) },
    { { -INT64_C( 3151990947610216115),  INT64_C( 6097220498204270054) },
      { -INT64_C( 3151990947610216115), -INT64_C( 6408752416989317890) },
      UINT8_C(  2) },
    { { -INT64_C( 1073687052990391200), -INT64_C( 6099308486802790392) },
      { -INT64_C( 5069770386641178014),  INT64_C(  189502087319862710) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m128i a = simde_mm_loadu_epi64(test_vec[i].a);
    simde__m128i b = simde_mm_loadu_epi64(test_vec[i].b);
    simde__mmask8 r = simde_mm_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[4];
    int64_t b_[4];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m128i a = simde_mm_loadu_epi64(a_);
    simde__m128i b = simde_mm_loadu_epi64(b_);
    simde__mmask8 r = simde_mm_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i64x2(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i64x2(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_epi8_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[32];
    const int8_t b[32];
    const simde__mmask32 r;
  } test_vec[] = {
    { {  INT8_C(  43),  INT8_C( 113),  INT8_C(  56),  INT8_C(  27),  INT8_C(  80),  INT8_C(  65), -INT8_C(  20), -INT8_C(  77),
         INT8_C(   3),  INT8_C(  93),  INT8_C( 124), -INT8_C(  58), -INT8_C(  98), -INT8_C(  78), -INT8_C( 108), -INT8_C(  78),
        -INT8_C(  85), -INT8_C(  87),  INT8_C(  33), -INT8_C(  42),  INT8_C( 115),  INT8_C(  64),  INT8_C(  72), -INT8_C(  41),
         INT8_C(  93), -INT8_C(  68),  INT8_C(  27),  INT8_C(  15), -INT8_C(  77), -INT8_C(  43),  INT8_C(  56), -INT8_C(  33) },
      { -INT8_C( 113),  INT8_C( 113), -INT8_C(   6), -INT8_C(  33), -INT8_C(  78), -INT8_C(  26), -INT8_C( 109), -INT8_C(  75),
         INT8_C(  67),  INT8_C(  44),  INT8_C( 124), -INT8_C(  30), -INT8_C(  42), -INT8_C(  78), -INT8_C( 108), -INT8_C( 127),
         INT8_C(  92), -INT8_C(  75),  INT8_C(  87), -INT8_C(  49), -INT8_C(  11), -INT8_C(  97), -INT8_C(  90),  INT8_C(  82),
         INT8_C(  91), -INT8_C(  63), -INT8_C(  82),  INT8_C(  15), -INT8_C( 105), -INT8_C(  26), -INT8_C(  18),  INT8_C(  38) },
      UINT32_C( 134243330) },
    { {  INT8_C(  84),  INT8_C(  68),  INT8_C( 123),  INT8_C(  35), -INT8_C(  35), -INT8_C(  27),  INT8_C(  53), -INT8_C(  93),
         INT8_C( 117),  INT8_C(  42),  INT8_C(  62),  INT8_C(  98), -INT8_C(  78),  INT8_C(  91), -INT8_C(  84), -INT8_C( 119),
         INT8_C( 101), -INT8_C(  59),  INT8_C(  35), -INT8_C(  31),  INT8_C( 125), -INT8_C(  88),  INT8_C(  80),  INT8_C(  89),
        -INT8_C(  36), -INT8_C(  51),  INT8_C(  29), -INT8_C(  10),  INT8_C(  57),  INT8_C(  92),  INT8_C( 103), -INT8_C( 115) },
      { -INT8_C(  96), -INT8_C(  30), -INT8_C(  80),  INT8_C( 126),  INT8_C(  28), -INT8_C(  27),  INT8_C(  33), -INT8_C( 111),
         INT8_C(  15),  INT8_C(  95), -INT8_C(  12), -INT8_C(  62), -INT8_C(  70), -INT8_C(  96), -INT8_C(  78), -INT8_C( 119),
         INT8_C( 101), -INT8_C(  43),  INT8_C( 106), -INT8_C(  23),  INT8_C( 125), -INT8_C(  70), -INT8_C(  17),  INT8_C(  89),
        -INT8_C( 120),  INT8_C(  12),  INT8_C(  79), -INT8_C(  63),  INT8_C( 104), -INT8_C(  73),  INT8_C(  78),  INT8_C(   9) },
      UINT32_C(2519618072) },
    { {  INT8_C(   5), -INT8_C( 104),  INT8_C( 105), -INT8_C(  24), -INT8_C(  58), -INT8_C(  80),  INT8_C(  71), -INT8_C(  51),
         INT8_C(  17), -INT8_C(  42), -INT8_C(   4), -INT8_C(  57), -INT8_C(  79),  INT8_C(   4),  INT8_C(   8),  INT8_C(  51),
        -INT8_C( 112),  INT8_C(  50), -INT8_C(  19),  INT8_C(   0),  INT8_C(  41),  INT8_C(  57), -INT8_C(   8),  INT8_C( 112),
        -INT8_C(  22), -INT8_C(  79),  INT8_C(   8),  INT8_C( 124), -INT8_C(  72), -INT8_C( 107),  INT8_C(  47), -INT8_C(  67) },
      { -INT8_C(  77), -INT8_C( 104), -INT8_C(  91),  INT8_C( 121),  INT8_C(  72), -INT8_C( 127),  INT8_C(  71),  INT8_C(  52),
         INT8_C(  17), -INT8_C(  42), -INT8_C(   4),  INT8_C(   9), -INT8_C(  79),  INT8_C(   4),  INT8_C(  60),  INT8_C(  65),
         INT8_C(  54),  INT8_C(  41),  INT8_C(  65), -INT8_C( 127),  INT8_C(  41),  INT8_C(  57), -INT8_C(  15), -INT8_C( 116),
        -INT8_C(  22), -INT8_C(  16),  INT8_C(   8), -INT8_C(  94), -INT8_C( 123),  INT8_C(  55),  INT8_C(  95),  INT8_C(  57) },
      UINT32_C(3879075802) },
    { { -INT8_C( 127),  INT8_C(  85), -INT8_C(  31), -INT8_C(  79),  INT8_C(  79),  INT8_C(  45),  INT8_C(  72), -INT8_C(  64),
         INT8_C( 117), -INT8_C( 113), -INT8_C(  96), -INT8_C(  63), -INT8_C(  15),  INT8_C(  98),  INT8_C(  67), -INT8_C(  54),
         INT8_C(  10),  INT8_C(  70), -INT8_C(   6), -INT8_C(  95),  INT8_C(  58), -INT8_C(  75), -INT8_C(  57),  INT8_C(  31),
         INT8_C( 121), -INT8_C( 113), -INT8_C(  19),  INT8_C( 115), -INT8_C(  74),  INT8_C(  44),  INT8_C(  61), -INT8_C( 102) },
      { -INT8_C( 127),  INT8_C(  30), -INT8_C( 110), -INT8_C(  48),  INT8_C(  75), -INT8_C(  38), -INT8_C(  17), -INT8_C(  64),
        -INT8_C(  62), -INT8_C( 113), -INT8_C( 127), -INT8_C( 121), -INT8_C(  15), -INT8_C(  60),  INT8_C(  81),  INT8_C(  86),
         INT8_C(  10),  INT8_C(  75), -INT8_C(   9),  INT8_C(  68),  INT8_C(   1), -INT8_C(  66),  INT8_C(  99),  INT8_C( 122),
         INT8_C(  77), -INT8_C(  20), -INT8_C(  19),  INT8_C(   3),  INT8_C(  24),  INT8_C(  42), -INT8_C(  27), -INT8_C( 102) },
      UINT32_C(         0) },
    { { -INT8_C( 107), -INT8_C(  47), -INT8_C(  89), -INT8_C(  25), -INT8_C(  82), -INT8_C(   5), -INT8_C(   5), -INT8_C( 105),
        -INT8_C(  49), -INT8_C( 105),  INT8_C( 114),  INT8_C( 104), -INT8_C( 124), -INT8_C(  92),  INT8_C(  10), -INT8_C(  68),
        -INT8_C(  51), -INT8_C(  15), -INT8_C(  10),  INT8_C(  60),  INT8_C(  60), -INT8_C(  57), -INT8_C(  23),  INT8_C( 115),
         INT8_C(  74),  INT8_C(  34), -INT8_C( 114),  INT8_C(  22),  INT8_C(  29), -INT8_C(  84), -INT8_C(   3), -INT8_C(  44) },
      {  INT8_C(   0), -INT8_C(  47), -INT8_C(  69), -INT8_C(  82), -INT8_C(  52), -INT8_C(  73),  INT8_C(  69), -INT8_C( 100),
         INT8_C(  78), -INT8_C(  72),  INT8_C(   4), -INT8_C(  46),  INT8_C(  92),  INT8_C(  14), -INT8_C( 114),  INT8_C(  41),
         INT8_C(   0), -INT8_C( 124), -INT8_C(  35),  INT8_C(  60),  INT8_C(  46), -INT8_C(  57), -INT8_C(  81),  INT8_C( 120),
        -INT8_C(  23),  INT8_C( 113), -INT8_C( 114),  INT8_C(  40),  INT8_C(  29), -INT8_C(  72), -INT8_C(   3),  INT8_C(  29) },
      UINT32_C(2883059709) },
    { { -INT8_C(  71),  INT8_C( 104),  INT8_C( 112),  INT8_C(  41), -INT8_C(  91),  INT8_C(  99), -INT8_C(  26),  INT8_C(  66),
         INT8_C(  89),  INT8_C( 118),  INT8_C( 103),  INT8_C(  94), -INT8_C( 108), -INT8_C(  75),  INT8_C(  99),  INT8_C(  49),
        -INT8_C(  32), -INT8_C(  92),  INT8_C(   7), -INT8_C(  45), -INT8_C( 108),  INT8_C(  80), -INT8_C(  82), -INT8_C(  10),
         INT8_C(  39),  INT8_C( 100),  INT8_C( 117),  INT8_C(  24), -INT8_C(  77),  INT8_C(  17), -INT8_C(  47),  INT8_C( 109) },
      {  INT8_C( 121),  INT8_C(  66), -INT8_C( 106), -INT8_C(  14), -INT8_C(  91),  INT8_C( 124),  INT8_C(  52), -INT8_C(   2),
        -INT8_C(  14), -INT8_C( 101),  INT8_C(  93), -INT8_C( 122),  INT8_C(  80), -INT8_C(  64), -INT8_C(  67),  INT8_C(  49),
         INT8_C( 101), -INT8_C(  60),  INT8_C(   4), -INT8_C(   7),  INT8_C(  20), -INT8_C(  78), -INT8_C(  17),  INT8_C(  59),
         INT8_C( 101),  INT8_C( 100), -INT8_C(  13),  INT8_C(  24),  INT8_C( 118), -INT8_C(  60), -INT8_C( 123), -INT8_C(  17) },
      UINT32_C(3995389854) },
    { {  INT8_C(   8), -INT8_C(  63),  INT8_C(  75), -INT8_C(  96),  INT8_C( 112), -INT8_C(  11),  INT8_C(  43), -INT8_C( 118),
        -INT8_C( 107),  INT8_C(  60),  INT8_C(  48), -INT8_C(  61),  INT8_C(  10), -INT8_C(  64), -INT8_C(  16), -INT8_C(  36),
         INT8_C(  54),  INT8_C(  22),  INT8_C(  66),  INT8_C(  97),  INT8_C(  43),  INT8_C(  35),  INT8_C(  48), -INT8_C( 104),
        -INT8_C(   8), -INT8_C( 104),  INT8_C(  41), -INT8_C( 111),  INT8_C(  17),  INT8_C( 117),  INT8_C(  48), -INT8_C( 115) },
      {  INT8_C(  54),  INT8_C( 123),  INT8_C(  46),  INT8_C(  14),  INT8_C( 112),  INT8_C(  89), -INT8_C( 104),  INT8_C( 108),
        -INT8_C( 107),  INT8_C(  37),  INT8_C(  48), -INT8_C(  97), -INT8_C(  27),  INT8_C(  32),  INT8_C(  59), -INT8_C(  36),
         INT8_C(  54),  INT8_C( 125), -INT8_C(  66),  INT8_C(  97), -INT8_C(  96), -INT8_C(  18),  INT8_C(   7), -INT8_C( 104),
        -INT8_C( 122), -INT8_C( 100),  INT8_C(  41),  INT8_C(  11),  INT8_C(  17),  INT8_C(  90), -INT8_C( 103),  INT8_C(  72) },
      UINT32_C(1634998852) },
    { {  INT8_C( 116), -INT8_C(  30), -INT8_C(  28), -INT8_C(  36), -INT8_C(  36), -INT8_C( 106),  INT8_C(  73), -INT8_C(  16),
         INT8_C( 121), -INT8_C(  74), -INT8_C(  23),  INT8_C( 123),  INT8_C(  44), -INT8_C(  65), -INT8_C(  76),  INT8_C(  56),
         INT8_C( 111),  INT8_C(  78), -INT8_C(  28), -INT8_C(  44), -INT8_C(   2),  INT8_C(  41), -INT8_C( 117),  INT8_C(  44),
        -INT8_C( 104), -INT8_C(  15),  INT8_C( 123),  INT8_C(  96), -INT8_C(  98),  INT8_C(  18), -INT8_C(   2),  INT8_C(  18) },
      {  INT8_C(   0), -INT8_C(  30), -INT8_C(  90), -INT8_C(  36),  INT8_C( 121), -INT8_C(  17), -INT8_C(  51), -INT8_C(  14),
        -INT8_C( 116), -INT8_C(  74),  INT8_C( 109), -INT8_C(  72),  INT8_C( 117),  INT8_C(  33), -INT8_C(  16), -INT8_C(  56),
         INT8_C( 111), -INT8_C(  44), -INT8_C( 100),  INT8_C(  94), -INT8_C(   2),  INT8_C( 121), -INT8_C( 117), -INT8_C( 106),
         INT8_C( 106),  INT8_C(   6), -INT8_C(  10),  INT8_C(   8),  INT8_C(  25), -INT8_C(  11),  INT8_C(  26),  INT8_C(  25) },
                UINT32_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi8(test_vec[i].a);
    simde__m256i b = simde_mm256_loadu_epi8(test_vec[i].b);
    simde__mmask32 r = simde_mm256_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a_[32];
    int8_t b_[32];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m256i a = simde_mm256_loadu_epi8(a_);
    simde__m256i b = simde_mm256_loadu_epi8(b_);
    simde__mmask32 r = simde_mm256_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i8x32(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i8x32(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask32(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_epi16_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[16];
    const int16_t b[16];
    const simde__mmask16 r;
  } test_vec[] = {
    { {  INT16_C(  9101), -INT16_C( 20347),  INT16_C( 28106), -INT16_C( 22715), -INT16_C( 11474),  INT16_C( 18560), -INT16_C(  8264), -INT16_C( 13503),
        -INT16_C( 16546),  INT16_C( 30132),  INT16_C( 13426), -INT16_C( 26683),  INT16_C( 22930),  INT16_C(  2747), -INT16_C( 17032),  INT16_C(  1379) },
      { -INT16_C(  5664), -INT16_C( 21578), -INT16_C(  1194),  INT16_C( 21842), -INT16_C( 11474),  INT16_C( 16285), -INT16_C(  8264), -INT16_C( 22262),
        -INT16_C( 16546), -INT16_C( 12258), -INT16_C(  6925), -INT16_C( 13720),  INT16_C( 22930),  INT16_C(  2747),  INT16_C(  7703), -INT16_C(  2288) },
      UINT16_C(12624) },
    { { -INT16_C( 11874),  INT16_C( 24565), -INT16_C( 22330), -INT16_C( 29106),  INT16_C( 32249), -INT16_C( 31065),  INT16_C( 14262), -INT16_C( 20808),
         INT16_C(  6015), -INT16_C( 21863),  INT16_C( 20446), -INT16_C(   408),  INT16_C( 12480),  INT16_C( 20534),  INT16_C( 25864),  INT16_C(  9281) },
      { -INT16_C(  9162), -INT16_C(   763),  INT16_C( 21380),  INT16_C( 32395), -INT16_C( 25904), -INT16_C( 31065),  INT16_C( 24529),  INT16_C( 20532),
        -INT16_C( 12682),  INT16_C( 21755),  INT16_C( 25373), -INT16_C(  8621), -INT16_C( 30317), -INT16_C( 25810),  INT16_C(  5614),  INT16_C(  9281) },
      UINT16_C( 1741) },
    { { -INT16_C( 28886),  INT16_C(  5528),  INT16_C(  3099),  INT16_C( 16758),  INT16_C( 28710),  INT16_C(  8674),  INT16_C( 19349),  INT16_C( 21145),
         INT16_C( 22447),  INT16_C( 30360), -INT16_C( 30864),  INT16_C( 23757), -INT16_C( 14941), -INT16_C( 10847),  INT16_C( 22011),  INT16_C( 30711) },
      { -INT16_C( 28886), -INT16_C( 21108),  INT16_C(  3099),  INT16_C( 16758),  INT16_C( 22652),  INT16_C(  4450), -INT16_C(  2909),  INT16_C( 21145),
         INT16_C( 12620), -INT16_C( 17208), -INT16_C( 26440),  INT16_C( 23757),  INT16_C( 28253),  INT16_C( 22577),  INT16_C( 10435), -INT16_C(  4401) },
      UINT16_C(15501) },
    { { -INT16_C(   862),  INT16_C(  2744),  INT16_C( 17946),  INT16_C( 30702), -INT16_C(  7356), -INT16_C( 25842), -INT16_C(  6799),  INT16_C( 12647),
        -INT16_C(  9715),  INT16_C( 27494),  INT16_C( 32027), -INT16_C( 13331),  INT16_C( 25730), -INT16_C( 20674), -INT16_C( 24662),  INT16_C( 19861) },
      {  INT16_C( 19867), -INT16_C( 22441),  INT16_C( 17946),  INT16_C( 24096), -INT16_C( 23255), -INT16_C( 25842),  INT16_C( 30090), -INT16_C( 26676),
         INT16_C( 30031),  INT16_C( 27494),  INT16_C( 21490),  INT16_C( 29750),  INT16_C( 29879),  INT16_C( 24867), -INT16_C( 18413), -INT16_C( 20818) },
      UINT16_C(    0) },
    { { -INT16_C( 32690),  INT16_C(  4026),  INT16_C( 14836),  INT16_C( 32594), -INT16_C( 27194), -INT16_C( 14155), -INT16_C(  3471), -INT16_C( 28636),
        -INT16_C( 10603), -INT16_C( 29946), -INT16_C( 16696),  INT16_C( 12088),  INT16_C( 18329),  INT16_C( 11432), -INT16_C( 12284), -INT16_C(  2970) },
      { -INT16_C( 32690),  INT16_C( 17110),  INT16_C( 23225),  INT16_C( 32594),  INT16_C(  2287),  INT16_C( 24903),  INT16_C( 24826), -INT16_C( 28636),
         INT16_C( 10806), -INT16_C(   229),  INT16_C( 21736), -INT16_C( 32466), -INT16_C( 10597), -INT16_C( 24658),  INT16_C( 29862), -INT16_C(  2970) },
      UINT16_C(32630) },
    { { -INT16_C(  8345), -INT16_C(   414),  INT16_C( 26142),  INT16_C(  8396),  INT16_C( 27393), -INT16_C( 10874), -INT16_C( 23946), -INT16_C( 21537),
        -INT16_C(  5926), -INT16_C(  9420),  INT16_C( 26911),  INT16_C( 11404),  INT16_C( 20918),  INT16_C( 30944), -INT16_C( 32399), -INT16_C(  7123) },
      { -INT16_C( 28568), -INT16_C( 11806),  INT16_C( 26142),  INT16_C(  8396),  INT16_C( 21201),  INT16_C( 18421), -INT16_C( 11019), -INT16_C( 12301),
        -INT16_C( 17220), -INT16_C(  9420), -INT16_C( 16347), -INT16_C(  9209), -INT16_C(  6126), -INT16_C( 28844), -INT16_C( 32399), -INT16_C(  9869) },
      UINT16_C(65311) },
    { { -INT16_C( 27155), -INT16_C( 22115), -INT16_C(  4852), -INT16_C(  4712),  INT16_C(  3378),  INT16_C( 19348),  INT16_C(  8661),  INT16_C( 23072),
        -INT16_C( 15063),  INT16_C( 26117), -INT16_C( 29816),  INT16_C( 10234),  INT16_C(  7782),  INT16_C( 25976),  INT16_C(  8885), -INT16_C(  5625) },
      { -INT16_C( 12873), -INT16_C( 15541), -INT16_C( 31814), -INT16_C(  4712),  INT16_C( 11408),  INT16_C( 25912),  INT16_C( 22862),  INT16_C( 12736),
        -INT16_C( 15063), -INT16_C( 20073), -INT16_C( 28080), -INT16_C( 18727),  INT16_C(  4528),  INT16_C( 25976), -INT16_C( 22477), -INT16_C(  5625) },
      UINT16_C(23172) },
    { { -INT16_C( 19996),  INT16_C( 24348),  INT16_C( 14466), -INT16_C(  2876),  INT16_C(  4024),  INT16_C( 15284), -INT16_C( 23014),  INT16_C( 27155),
        -INT16_C( 25553),  INT16_C( 24622),  INT16_C( 17322),  INT16_C( 28949),  INT16_C( 17713), -INT16_C( 22249), -INT16_C( 22660),  INT16_C(  1429) },
      { -INT16_C( 19996),  INT16_C( 26212),  INT16_C( 10730),  INT16_C( 30555),  INT16_C(  4024), -INT16_C( 11341), -INT16_C( 14667), -INT16_C(  7107),
         INT16_C( 18530),  INT16_C( 24622),  INT16_C( 17322), -INT16_C(  9007), -INT16_C(  6008),  INT16_C(  1157),  INT16_C(  6799),  INT16_C( 29450) },
           UINT16_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi16(test_vec[i].a);
    simde__m256i b = simde_mm256_loadu_epi16(test_vec[i].b);
    simde__mmask16 r = simde_mm256_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[16];
    int16_t b_[16];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i16() & 3))
        a_[j] = b_[j];

    simde__m256i a = simde_mm256_loadu_epi16(a_);
    simde__m256i b = simde_mm256_loadu_epi16(b_);
    simde__mmask16 r = simde_mm256_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i16x16(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i16x16(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask16(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_epi32_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[8];
    const int32_t b[8];
    const simde__mmask8 r;
  } test_vec[] = {
    { { -INT32_C(  1520606249), -INT32_C(  1370141982), -INT32_C(   676213503), -INT32_C(  1831730832),  INT32_C(    73771260),  INT32_C(  2021947825), -INT32_C(   182757216),  INT32_C(   835755084) },
      { -INT32_C(   948442395),  INT32_C(   896912692),  INT32_C(   168568474),  INT32_C(   597483047),  INT32_C(   925303174),  INT32_C(   447720314), -INT32_C(    32519677),  INT32_C(   835755084) },
      UINT8_C(128) },
    { {  INT32_C(  1166098769), -INT32_C(  1986934796), -INT32_C(  1689957438), -INT32_C(   117824095), -INT32_C(   523595731), -INT32_C(   459884587), -INT32_C(   109648088), -INT32_C(  1637233711) },
      {  INT32_C(  1166098769), -INT32_C(   892464632), -INT32_C(    26930339),  INT32_C(  1727487801), -INT32_C(  1001995793),  INT32_C(   732486776), -INT32_C(   109648088),  INT32_C(    60284850) },
      UINT8_C(174) },
    { { -INT32_C(   410847300),  INT32_C(    20002357), -INT32_C(  2086996020), -INT32_C(  1272421515), -INT32_C(   503266497), -INT32_C(  1670295389), -INT32_C(   481853782), -INT32_C(   566364894) },
      { -INT32_C(    87637780),  INT32_C(    20002357), -INT32_C(  1568355283), -INT32_C(  1571377873), -INT32_C(   503266497), -INT32_C(  1115786989),  INT32_C(   295749103), -INT32_C(  1477386565) },
      UINT8_C(119) },
    { {  INT32_C(  1972662153), -INT32_C(   510760894),  INT32_C(   711556559),  INT32_C(   789323866),  INT32_C(  1930835089),  INT32_C(  1162132442),  INT32_C(   139277582), -INT32_C(  2082194694) },
      { -INT32_C(   319260503), -INT32_C(  1563588896),  INT32_C(   711556559),  INT32_C(   794392222), -INT32_C(  2069729366),  INT32_C(   717874730),  INT32_C(   139277582),  INT32_C(   428618095) },
      UINT8_C(  0) },
    { { -INT32_C(  1816974597), -INT32_C(  1065589462), -INT32_C(  1579764074), -INT32_C(   514329836),  INT32_C(  1490587826), -INT32_C(   871630189),  INT32_C(   192202602),  INT32_C(  1292873298) },
      { -INT32_C(  1816974597), -INT32_C(  1065589462),  INT32_C(   811749916), -INT32_C(   133056186), -INT32_C(   581899959),  INT32_C(  1319722212),  INT32_C(   861478179),  INT32_C(  1292873298) },
      UINT8_C(124) },
    { { -INT32_C(   360494398),  INT32_C(  1445873575),  INT32_C(  1263247980), -INT32_C(  1787611097),  INT32_C(   283670646), -INT32_C(   238122970),  INT32_C(   705609346), -INT32_C(  2136311618) },
      {  INT32_C(  1768566466), -INT32_C(   507537291),  INT32_C(  1630276154), -INT32_C(  1175019709),  INT32_C(  1036705303), -INT32_C(   382822297),  INT32_C(   336804950),  INT32_C(  2006236596) },
      UINT8_C( 98) },
    { { -INT32_C(  1653351929),  INT32_C(   611235194), -INT32_C(  2001592127), -INT32_C(   833111609),  INT32_C(   859552298),  INT32_C(   429036107),  INT32_C(   580650747),  INT32_C(   372610319) },
      { -INT32_C(   374101905), -INT32_C(  2146557603), -INT32_C(  2001592127),  INT32_C(    55970265),  INT32_C(   238457539),  INT32_C(   522766372), -INT32_C(   566049585), -INT32_C(   906725543) },
      UINT8_C(210) },
    { { -INT32_C(   862828029),  INT32_C(   415531994), -INT32_C(   173774531), -INT32_C(   848338478),  INT32_C(   506165109),  INT32_C(  1279116449), -INT32_C(  1561526101),  INT32_C(  1622770780) },
      { -INT32_C(   651408434),  INT32_C(   415531994), -INT32_C(  1106417428),  INT32_C(   495680713),  INT32_C(   506165109),  INT32_C(  1279116449),  INT32_C(   569322180),  INT32_C(  2105649326) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi32(test_vec[i].a);
    simde__m256i b = simde_mm256_loadu_epi32(test_vec[i].b);
    simde__mmask8 r = simde_mm256_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[8];
    int32_t b_[8];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i32() & 3))
        a_[j] = b_[j];

    simde__m256i a = simde_mm256_loadu_epi32(a_);
    simde__m256i b = simde_mm256_loadu_epi32(b_);
    simde__mmask8 r = simde_mm256_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i32x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i32x8(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_cmp_epi64_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const simde__mmask8 r;
  } test_vec[] = {
    { { -INT64_C( 3061706303324098721), -INT64_C( 7075376414461670625), -INT64_C( 8248903588976154960),  INT64_C(  754087591936956394) },
      {  INT64_C(  227787899817509374), -INT64_C( 1923559078301860082), -INT64_C( 2649140287693721396),  INT64_C( 4010940600286325747) },
      UINT8_C(  0) },
    { { -INT64_C( 1626301166014909010), -INT64_C( 7000663457614807716),  INT64_C( 3786838713518589141),  INT64_C( 6290469473638956404) },
      { -INT64_C( 4571573207251538703),  INT64_C(  327285041734901055),  INT64_C( 3786838713518589141),  INT64_C( 7767282717047275210) },
      UINT8_C( 10) },
    { { -INT64_C(  104664725362663840), -INT64_C( 2679084206771791897),  INT64_C( 7360512122289395314),  INT64_C( 7536310226373420634) },
      { -INT64_C( 8778927866836519120), -INT64_C( 5690558540880401134),  INT64_C( 7360512122289395314),  INT64_C( 7536310226373420634) },
      UINT8_C( 12) },
    { {  INT64_C( 1458947386577960448), -INT64_C( 6095539528045473816), -INT64_C( 3407628869727227501),  INT64_C( 3691192106094222744) },
      { -INT64_C( 4500607351733994358), -INT64_C( 6095539528045473816), -INT64_C( 6726885128119193027),  INT64_C( 3691192106094222744) },
      UINT8_C(  0) },
    { { -INT64_C( 7319844171321101507),  INT64_C( 6993771106197164796),  INT64_C( 4155832417833431850), -INT64_C( 4636994022111648454) },
      { -INT64_C( 8906344863618508727), -INT64_C( 5456731580380894176), -INT64_C(  124091139766138754),  INT64_C( 1389883622969095176) },
      UINT8_C( 15) },
    { {  INT64_C( 6215649242152406996), -INT64_C( 2227888630144339662), -INT64_C( 7251913801718489655),  INT64_C( 5678914690226300385) },
      {  INT64_C( 6215649242152406996), -INT64_C( 2227888630144339662), -INT64_C( 3639453938778642927), -INT64_C( 8010442780220248652) },
      UINT8_C( 11) },
    { {  INT64_C( 2495064606853273644), -INT64_C( 8707789476991879884), -INT64_C( 8213802854753458746), -INT64_C( 4291239295976548379) },
      {  INT64_C(  736836437842040077), -INT64_C( 5000272174874936199), -INT64_C( 8539889611890626788),  INT64_C( 2047907544154741262) },
      UINT8_C(  5) },
    { {  INT64_C(  363786789876318920),  INT64_C( 1001922729367468764),  INT64_C( 2617708196441159240), -INT64_C( 4347279376536444313) },
      {  INT64_C(  363786789876318920), -INT64_C( 5809673286333930776),  INT64_C( 2617708196441159240), -INT64_C( 6914029452691079975) },
      UINT8_C( 15) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi64(test_vec[i].a);
    simde__m256i b = simde_mm256_loadu_epi64(test_vec[i].b);
    simde__mmask8 r = simde_mm256_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[4];
    int64_t b_[4];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m256i a = simde_mm256_loadu_epi64(a_);
    simde__m256i b = simde_mm256_loadu_epi64(b_);
    simde__mmask8 r = simde_mm256_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i64x4(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i64x4(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_epi8_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[64];
    const int8_t b[64];
    const simde__mmask64 r;
  } test_vec[] = {
    { {  INT8_C(  75), -INT8_C(  47), -INT8_C(  35), -INT8_C(  54), -INT8_C(  26), -INT8_C( 123), -INT8_C(  57),  INT8_C(  53),
         INT8_C(  88), -INT8_C(  81),  INT8_C(   5),  INT8_C( 110),  INT8_C(  66),  INT8_C( 102), -INT8_C(  78), -INT8_C(  36),
         INT8_C(  43), -INT8_C( 105), -INT8_C(  97),  INT8_C(  48),  INT8_C(  21),  INT8_C(  36), -INT8_C(  27),  INT8_C( 125),
        -INT8_C(  62), -INT8_C(  12),  INT8_C(  19), -INT8_C(  67),  INT8_C(   6),  INT8_C( 116), -INT8_C(  26), -INT8_C(  81),
         INT8_C(  46), -INT8_C( 116),  INT8_C(  91),  INT8_C( 122), -INT8_C(  35),  INT8_C(  34), -INT8_C(  81),  INT8_C( 106),
        -INT8_C(  54),  INT8_C(  34), -INT8_C(  40),  INT8_C(  85), -INT8_C( 119),  INT8_C(   3),  INT8_C(  49), -INT8_C(  76),
        -INT8_C( 102), -INT8_C(  48),  INT8_C(   4),  INT8_C(  54), -INT8_C(  31), -INT8_C(  23), -INT8_C( 117), -INT8_C(   4),
        -INT8_C(  34), -INT8_C(  98),  INT8_C(  15), -INT8_C(  92), -INT8_C(   5), -INT8_C(  66),  INT8_C(  83),  INT8_C(  41) },
      {  INT8_C(  75), -INT8_C(  52), -INT8_C(  93),  INT8_C(  93), -INT8_C(  26),  INT8_C(  83), -INT8_C(  57), -INT8_C(  80),
         INT8_C( 117), -INT8_C(  97),  INT8_C(   5), -INT8_C(   2),  INT8_C(  66),  INT8_C(  55), -INT8_C(  78),  INT8_C(  99),
         INT8_C(   7), -INT8_C( 105), -INT8_C( 103), -INT8_C(  23), -INT8_C(  33),  INT8_C(  36), -INT8_C(  27),  INT8_C( 125),
        -INT8_C(  62), -INT8_C(  12),  INT8_C(  33), -INT8_C(  67), -INT8_C(  77),  INT8_C( 116), -INT8_C(  26), -INT8_C(   2),
         INT8_C(  64), -INT8_C( 118),  INT8_C(  91),  INT8_C(  38), -INT8_C(  35),  INT8_C(  34), -INT8_C(  41),  INT8_C(  82),
        -INT8_C(  63), -INT8_C(  36),  INT8_C(  81),  INT8_C(   3),  INT8_C(  19),  INT8_C(   3),  INT8_C( 102),  INT8_C(  27),
        -INT8_C( 102), -INT8_C(   1),  INT8_C(   4),  INT8_C( 122),  INT8_C(  36), -INT8_C(  23), -INT8_C(   9), -INT8_C(  26),
        -INT8_C(  34),  INT8_C(  24), -INT8_C(  92), -INT8_C( 111), -INT8_C( 116), -INT8_C( 118), -INT8_C( 113), -INT8_C(  52) },
      UINT64_C(   82507577696605265) },
    { { -INT8_C( 120), -INT8_C(  94), -INT8_C( 113),  INT8_C(  88), -INT8_C( 125),  INT8_C(  18),  INT8_C(  66), -INT8_C(  88),
             INT8_MIN, -INT8_C(   2), -INT8_C(  70), -INT8_C( 118),  INT8_C(   2), -INT8_C(  89), -INT8_C(  83), -INT8_C(  34),
         INT8_C( 109), -INT8_C(  59),  INT8_C( 125), -INT8_C(  31), -INT8_C( 100), -INT8_C(  69), -INT8_C(   3),  INT8_C(  89),
        -INT8_C(  65),  INT8_C( 107),  INT8_C(   3),  INT8_C(  90), -INT8_C(  19),  INT8_C(  49),  INT8_C(  52),  INT8_C( 117),
        -INT8_C(  45),  INT8_C( 102), -INT8_C( 100),  INT8_C(  45), -INT8_C(  42), -INT8_C(  34), -INT8_C(  43),  INT8_C(  59),
        -INT8_C( 102),  INT8_C(  99),  INT8_C(  42), -INT8_C( 100),  INT8_C(  10), -INT8_C( 113),  INT8_C( 108), -INT8_C(  75),
        -INT8_C( 100), -INT8_C(   8), -INT8_C( 106), -INT8_C(  71), -INT8_C(  82),  INT8_C(  75),  INT8_C(  18),  INT8_C( 110),
         INT8_C(  78), -INT8_C(  60),  INT8_C( 116), -INT8_C(  93),  INT8_C(  70),  INT8_C( 109),  INT8_C(  25),  INT8_C(  25) },
      {  INT8_C(  48), -INT8_C(  25),  INT8_C(  71),  INT8_C(   6), -INT8_C( 125),  INT8_C(  28),  INT8_C(  66),  INT8_C(  30),
             INT8_MIN,  INT8_C( 108), -INT8_C(  70), -INT8_C( 118),  INT8_C(  67),  INT8_C(  53),  INT8_C(   2), -INT8_C(  33),
         INT8_C(  45),  INT8_C(  80), -INT8_C( 104), -INT8_C(  31), -INT8_C( 100), -INT8_C(  86), -INT8_C(  65),  INT8_C(  82),
        -INT8_C(  65), -INT8_C(   8), -INT8_C(  10),  INT8_C(   6),  INT8_C( 101),  INT8_C(  15),  INT8_C(  31), -INT8_C( 106),
        -INT8_C(  10),  INT8_C( 102), -INT8_C( 100),  INT8_C( 121), -INT8_C( 125), -INT8_C(  34), -INT8_C( 105),  INT8_C(   3),
         INT8_C(  74),  INT8_C(  82), -INT8_C( 115), -INT8_C( 115), -INT8_C( 121), -INT8_C( 113),  INT8_C( 108), -INT8_C(  75),
        -INT8_C(  32),  INT8_C(   4), -INT8_C( 106),  INT8_C( 124), -INT8_C(  82),  INT8_C(  85), -INT8_C(  50),  INT8_C( 110),
         INT8_C(  78), -INT8_C(  60),  INT8_C( 116), -INT8_C(  77), -INT8_C(  45), -INT8_C( 109),  INT8_C(  73), -INT8_C(  55) },
      UINT64_C( 5200251333164331687) },
    { { -INT8_C( 111),  INT8_C(  44),  INT8_C(  75), -INT8_C(  41), -INT8_C(  66), -INT8_C(  92),  INT8_C(  74), -INT8_C(  12),
        -INT8_C(  52),  INT8_C( 113), -INT8_C(  57), -INT8_C( 104),  INT8_C(   1),  INT8_C(  23), -INT8_C(  92), -INT8_C( 109),
         INT8_C(  96),  INT8_C( 108),  INT8_C( 116),  INT8_C( 112),  INT8_C(  94), -INT8_C(   6),  INT8_C(  13), -INT8_C( 118),
        -INT8_C(  42),  INT8_C(   9), -INT8_C(  81),  INT8_C(  81),  INT8_C(  46), -INT8_C( 100),  INT8_C(  82), -INT8_C(  65),
         INT8_C(  50), -INT8_C(  99), -INT8_C(  57), -INT8_C(  16),  INT8_C(  65),  INT8_C(  78), -INT8_C(  28), -INT8_C(  37),
         INT8_C(  82), -INT8_C(  85),  INT8_C( 115),  INT8_C(  84), -INT8_C(  62),  INT8_C(  23), -INT8_C(  28), -INT8_C(  53),
        -INT8_C(  44),  INT8_C(  88), -INT8_C( 110), -INT8_C(  31),  INT8_C(  82), -INT8_C(  97), -INT8_C(  49),  INT8_C( 101),
         INT8_C(  95),  INT8_C( 126),  INT8_C( 122), -INT8_C(  42), -INT8_C( 104), -INT8_C(  52), -INT8_C( 107),  INT8_C(  77) },
      {  INT8_C( 106),  INT8_C(  44),  INT8_C(  61), -INT8_C(  85),  INT8_C(  13),  INT8_C(  33), -INT8_C( 122),  INT8_C(  95),
        -INT8_C(  52), -INT8_C(   7), -INT8_C(  77), -INT8_C( 114),  INT8_C(  16), -INT8_C( 104), -INT8_C(  80), -INT8_C( 109),
        -INT8_C(  16),  INT8_C(  66),  INT8_C( 116),  INT8_C(  67), -INT8_C(  30),  INT8_C(  67),  INT8_C( 107), -INT8_C( 118),
        -INT8_C(  62), -INT8_C(  27),  INT8_C(  97), -INT8_C(  35), -INT8_C(  78), -INT8_C(  10),  INT8_C(  42),  INT8_C(  28),
         INT8_C(  34),  INT8_C( 103), -INT8_C(  57),  INT8_C(  47), -INT8_C( 120),  INT8_C(  78), -INT8_C( 113),  INT8_C(  84),
         INT8_C(  71),  INT8_C(  66), -INT8_C(  30),  INT8_C(  88), -INT8_C(  38), -INT8_C( 110), -INT8_C(  21), -INT8_C(  53),
        -INT8_C(  44),  INT8_C(  96),  INT8_C(  14), -INT8_C(  74), -INT8_C(  93),  INT8_C( 121),  INT8_C(  65),  INT8_C( 101),
         INT8_C(  95), -INT8_C(  94),  INT8_C(  66),  INT8_C(  17), -INT8_C( 104),  INT8_C( 108),  INT8_C(  45), -INT8_C(  69) },
      UINT64_C( 8784230041835065779) },
    { {      INT8_MAX,  INT8_C(  87), -INT8_C(  76), -INT8_C(  56),  INT8_C(  27), -INT8_C(   5), -INT8_C(  51), -INT8_C(  90),
         INT8_C(  36),  INT8_C(  70), -INT8_C(  22),  INT8_C(  26), -INT8_C(  95),  INT8_C(  91),  INT8_C( 123), -INT8_C(  46),
         INT8_C(   4),  INT8_C(  51),  INT8_C(  39), -INT8_C( 119),  INT8_C(  95), -INT8_C(  59),  INT8_C(  72),  INT8_C(   3),
        -INT8_C(  69),      INT8_MIN, -INT8_C( 115),  INT8_C(  56), -INT8_C(  25), -INT8_C(  48),  INT8_C(  47),  INT8_C(  35),
         INT8_C(  40), -INT8_C(  29), -INT8_C(  21),  INT8_C(  67), -INT8_C(  34), -INT8_C(  72), -INT8_C(  76),  INT8_C(   2),
         INT8_C(  32), -INT8_C(  98),  INT8_C(  28), -INT8_C( 122), -INT8_C(   7),  INT8_C(  91), -INT8_C( 101),  INT8_C( 116),
        -INT8_C( 114),      INT8_MAX, -INT8_C(   3), -INT8_C(  19),  INT8_C(  68),  INT8_C(  39), -INT8_C(  15),  INT8_C(   0),
         INT8_C(  28),  INT8_C( 126),  INT8_C(  48), -INT8_C(  64), -INT8_C(  69),  INT8_C( 103), -INT8_C(  29),  INT8_C( 119) },
      {  INT8_C(  74), -INT8_C(  50), -INT8_C(  70),  INT8_C(  40), -INT8_C( 122),  INT8_C( 111),  INT8_C(  42), -INT8_C(  90),
         INT8_C(  13),  INT8_C(  70),  INT8_C(  44),  INT8_C(   7), -INT8_C(  95), -INT8_C( 124),  INT8_C( 123),  INT8_C(  48),
         INT8_C(   4),  INT8_C( 120),  INT8_C(  29),  INT8_C(  72), -INT8_C(  96),  INT8_C(  14),  INT8_C(  72), -INT8_C(  68),
        -INT8_C( 115),      INT8_MIN,  INT8_C( 124), -INT8_C(  36), -INT8_C(  25),  INT8_C(  95),  INT8_C(  83),  INT8_C(  49),
         INT8_C(  45),  INT8_C(  13),  INT8_C(  89), -INT8_C(  77),  INT8_C( 124), -INT8_C( 125),  INT8_C(  89), -INT8_C( 118),
        -INT8_C(  54), -INT8_C( 122), -INT8_C( 111),  INT8_C( 107),  INT8_C(  10),  INT8_C(  12), -INT8_C( 101),  INT8_C(  14),
        -INT8_C( 124), -INT8_C(  71),  INT8_C(  87),  INT8_C(  36), -INT8_C(  57), -INT8_C(  97), -INT8_C(  32),  INT8_C(  84),
         INT8_C(  32),  INT8_C(  92),  INT8_C(  48),  INT8_C(   7), -INT8_C(  69), -INT8_C( 125),  INT8_C(  57), -INT8_C(  24) },
      UINT64_C(                   0) },
    { {  INT8_C(  96), -INT8_C(  43), -INT8_C(  79),  INT8_C(   5),  INT8_C( 123),  INT8_C(  37),  INT8_C( 121),  INT8_C(  74),
        -INT8_C(  28), -INT8_C( 101),  INT8_C(  77), -INT8_C( 118), -INT8_C(  84),  INT8_C(  86), -INT8_C(  48), -INT8_C(  40),
        -INT8_C(  94),  INT8_C( 125), -INT8_C(  44), -INT8_C(   6), -INT8_C(  25), -INT8_C(  41),  INT8_C( 108),  INT8_C(  75),
        -INT8_C(  93), -INT8_C(  76),  INT8_C(   3),  INT8_C(  45), -INT8_C(  70), -INT8_C(  54), -INT8_C(  33),  INT8_C(  27),
        -INT8_C(  97), -INT8_C(   3), -INT8_C(  54),  INT8_C(  27),  INT8_C(  34),  INT8_C(  68), -INT8_C(  91),  INT8_C(   6),
         INT8_C(  13), -INT8_C(  78), -INT8_C( 112), -INT8_C(  71), -INT8_C(  69),  INT8_C(  10),  INT8_C(  52),  INT8_C(  93),
        -INT8_C(  35),  INT8_C(  43), -INT8_C(  68), -INT8_C(  19), -INT8_C(  32), -INT8_C(  61),  INT8_C(  56),  INT8_C( 109),
         INT8_C( 119), -INT8_C(  75), -INT8_C( 102),  INT8_C(  50), -INT8_C(  50), -INT8_C(  26),  INT8_C(  77),  INT8_C( 110) },
      { -INT8_C(  29),  INT8_C(  23), -INT8_C( 119),  INT8_C(   5),  INT8_C(  91), -INT8_C(  18),  INT8_C(  11),  INT8_C( 104),
        -INT8_C(  96), -INT8_C( 101),  INT8_C(  34),  INT8_C(  91), -INT8_C(   5),  INT8_C(  86), -INT8_C(  72), -INT8_C(  40),
         INT8_C(  95),  INT8_C(  16), -INT8_C(  99),  INT8_C(  63), -INT8_C(  45), -INT8_C(  98), -INT8_C(  84),  INT8_C(  75),
        -INT8_C(  93),  INT8_C(  70),  INT8_C( 125),  INT8_C( 113),  INT8_C(  44), -INT8_C(  54), -INT8_C(  33),  INT8_C(  15),
        -INT8_C(  31),  INT8_C( 104),  INT8_C(  20),  INT8_C(  61),  INT8_C(  86),  INT8_C(  31), -INT8_C(  91), -INT8_C(  10),
        -INT8_C(  70), -INT8_C(  57),  INT8_C(  81), -INT8_C(  75),  INT8_C(  30),  INT8_C(  10), -INT8_C( 114),  INT8_C( 125),
         INT8_C(  26),  INT8_C(  43), -INT8_C(  68), -INT8_C(  19), -INT8_C(  55),  INT8_C( 104),  INT8_C(  56),  INT8_C( 108),
        -INT8_C(  82), -INT8_C(  75), -INT8_C(  34), -INT8_C(  37),      INT8_MAX, -INT8_C(  67), -INT8_C(  22),  INT8_C(  97) },
      UINT64_C(18280638376564448759) },
    { {  INT8_C( 125),  INT8_C(  36),  INT8_C(  46), -INT8_C(  85),  INT8_C(  83),  INT8_C(  99), -INT8_C( 108),  INT8_C(  70),
         INT8_C(  38),  INT8_C(  78),  INT8_C(   7), -INT8_C(  78), -INT8_C(  82),  INT8_C(  25),  INT8_C(  44),  INT8_C( 112),
         INT8_C(  49),  INT8_C( 100), -INT8_C( 122), -INT8_C(  41),  INT8_C(  26),  INT8_C(   0), -INT8_C( 113),  INT8_C(  43),
         INT8_C(  88),  INT8_C(  54),  INT8_C(  85), -INT8_C(  18), -INT8_C(  29), -INT8_C(  66), -INT8_C(  63),  INT8_C(  96),
        -INT8_C(  30), -INT8_C(  16),  INT8_C(  99),  INT8_C(  69),  INT8_C(  83), -INT8_C(  59),  INT8_C( 123),  INT8_C( 121),
         INT8_C( 103), -INT8_C(  66),  INT8_C( 126), -INT8_C(  13),  INT8_C(  52), -INT8_C(  85),  INT8_C(  99),  INT8_C( 102),
         INT8_C(  15),  INT8_C(  95),  INT8_C(  38),  INT8_C(  41),  INT8_C(  95),  INT8_C(  56),  INT8_C(  84), -INT8_C(  72),
        -INT8_C(  31), -INT8_C(  87), -INT8_C(  28), -INT8_C(  60),  INT8_C(  99),  INT8_C( 104),      INT8_MIN,  INT8_C(  74) },
      {  INT8_C(  88), -INT8_C( 121),      INT8_MAX, -INT8_C(  85),  INT8_C( 126), -INT8_C(   5),  INT8_C(  36), -INT8_C(  61),
         INT8_C( 126), -INT8_C(  94), -INT8_C(  74), -INT8_C(  78),  INT8_C(  77),  INT8_C(  25),  INT8_C(  24),  INT8_C(  93),
         INT8_C( 120),  INT8_C(  62), -INT8_C( 122), -INT8_C(  41),  INT8_C( 119), -INT8_C(  37), -INT8_C( 113),  INT8_C(  88),
        -INT8_C( 124),  INT8_C(  54),  INT8_C(  28), -INT8_C(  20), -INT8_C(  98),  INT8_C(  64),  INT8_C(  54), -INT8_C(  10),
        -INT8_C(  57), -INT8_C(  75), -INT8_C(  95),  INT8_C(  69), -INT8_C(  80), -INT8_C(  59),  INT8_C(   8),  INT8_C(  46),
         INT8_C( 103), -INT8_C(  66), -INT8_C(  31), -INT8_C(  75), -INT8_C(  41), -INT8_C(   7),  INT8_C(  18),  INT8_C(  79),
         INT8_C(  56), -INT8_C( 104),  INT8_C(  38), -INT8_C(  81),  INT8_C( 115), -INT8_C(  74),  INT8_C(   7), -INT8_C(   8),
        -INT8_C(  20),  INT8_C(  35), -INT8_C(  28), -INT8_C( 118),  INT8_C(  99),  INT8_C(  26),      INT8_MIN,  INT8_C(  42) },
      UINT64_C(18189722233980513963) },
    { {  INT8_C(  41), -INT8_C(  34), -INT8_C(  29),  INT8_C(  54), -INT8_C(  14),  INT8_C( 101),  INT8_C( 120), -INT8_C( 106),
        -INT8_C(  22),  INT8_C(  36), -INT8_C(  61), -INT8_C( 125),  INT8_C( 111), -INT8_C(  79),  INT8_C( 122), -INT8_C(  40),
         INT8_C(  15),  INT8_C(  47), -INT8_C(   5), -INT8_C(  28), -INT8_C(  43), -INT8_C( 127),  INT8_C(  83),  INT8_C(  42),
        -INT8_C(  76), -INT8_C(  65), -INT8_C(  68),  INT8_C(  20),  INT8_C(  82),  INT8_C(  52), -INT8_C(  61),  INT8_C( 123),
        -INT8_C(  47), -INT8_C(  89), -INT8_C(  79), -INT8_C( 111),  INT8_C(  12), -INT8_C(  38), -INT8_C( 101), -INT8_C(  10),
        -INT8_C(  17),  INT8_C(  94), -INT8_C(  77),  INT8_C(  94),  INT8_C(  16),  INT8_C(  66), -INT8_C(   8),  INT8_C(  31),
         INT8_C( 114),  INT8_C(  54),  INT8_C(   4),  INT8_C(  32),  INT8_C( 116),  INT8_C(  83),  INT8_C( 109),  INT8_C(  40),
         INT8_C(  23),  INT8_C(   6),  INT8_C(  61),  INT8_C( 105), -INT8_C(   2),  INT8_C(  25), -INT8_C(  27),  INT8_C(  76) },
      { -INT8_C(  89), -INT8_C( 106),  INT8_C(  81), -INT8_C(  76), -INT8_C(  64), -INT8_C(  20), -INT8_C(  86), -INT8_C(  81),
         INT8_C(  74),  INT8_C(  36),  INT8_C(  14),  INT8_C(  90),  INT8_C( 102),  INT8_C(   6),  INT8_C( 122), -INT8_C(  40),
        -INT8_C(   7),  INT8_C( 126), -INT8_C(   7),  INT8_C( 110), -INT8_C(  43),  INT8_C(  67), -INT8_C( 106), -INT8_C(  20),
         INT8_C(  74), -INT8_C(  45),  INT8_C(  86), -INT8_C( 124), -INT8_C(  44),  INT8_C(  59), -INT8_C(  47),  INT8_C( 123),
        -INT8_C(  47),  INT8_C(  34),  INT8_C(  47), -INT8_C( 111),  INT8_C(  14), -INT8_C(  38),  INT8_C(  65),  INT8_C(  88),
        -INT8_C(   2),  INT8_C(  79), -INT8_C(  77),  INT8_C( 100),  INT8_C(  85),  INT8_C(  45),  INT8_C(  61),  INT8_C(  78),
        -INT8_C(  85),  INT8_C(  54), -INT8_C(  68),      INT8_MIN,  INT8_C( 121),  INT8_C(  83),  INT8_C( 109), -INT8_C(  61),
         INT8_C(  38), -INT8_C(  61),  INT8_C(  72), -INT8_C(   6), -INT8_C(   2),  INT8_C(  25),  INT8_C( 118), -INT8_C(  49) },
      UINT64_C( 9983673332761170043) },
    { {  INT8_C(  80),  INT8_C(   6),  INT8_C( 121), -INT8_C(   3),  INT8_C(  30),  INT8_C(  86), -INT8_C(  72), -INT8_C( 117),
        -INT8_C(  20),  INT8_C(  75),  INT8_C( 122), -INT8_C(  12),  INT8_C(   6), -INT8_C( 107),  INT8_C(  40), -INT8_C(  61),
         INT8_C(  93),  INT8_C(  42),  INT8_C(  49),  INT8_C(  63), -INT8_C(  66),  INT8_C( 106), -INT8_C(   2),  INT8_C(  44),
        -INT8_C(  71), -INT8_C( 104), -INT8_C( 115),  INT8_C(  49), -INT8_C(  36),  INT8_C(  28), -INT8_C(  71),  INT8_C(  44),
         INT8_C(  34),  INT8_C(  50),  INT8_C(  28),  INT8_C(  64),  INT8_C(  76), -INT8_C(  44), -INT8_C(  52),  INT8_C(  57),
         INT8_C(  32),  INT8_C(  70), -INT8_C( 109),  INT8_C(  64), -INT8_C(  37), -INT8_C(  69),  INT8_C(  18),  INT8_C(  56),
        -INT8_C(  27),  INT8_C(  53),  INT8_C( 119), -INT8_C(  93), -INT8_C(   2),  INT8_C(  58),  INT8_C(  90), -INT8_C(  73),
         INT8_C(  13),  INT8_C(  92), -INT8_C(  90), -INT8_C(  68),  INT8_C(  52),  INT8_C(  95),  INT8_C(  22), -INT8_C( 102) },
      { -INT8_C( 111),  INT8_C(  64), -INT8_C(  38),  INT8_C(  26),  INT8_C(   6), -INT8_C(  90), -INT8_C(  72),  INT8_C(  76),
        -INT8_C(  20),  INT8_C(  75), -INT8_C( 116), -INT8_C(  57),  INT8_C(   6), -INT8_C( 112), -INT8_C(   1), -INT8_C(  21),
        -INT8_C(  59),  INT8_C( 118), -INT8_C( 114),  INT8_C( 100), -INT8_C(  21),  INT8_C(  93),  INT8_C( 106), -INT8_C(   8),
        -INT8_C(  71),  INT8_C(  17), -INT8_C(  30),  INT8_C(  49),  INT8_C( 112), -INT8_C(   8), -INT8_C(  53),  INT8_C(   2),
         INT8_C(  56), -INT8_C(  90),  INT8_C(  28),  INT8_C(  62),  INT8_C(  76), -INT8_C(  44), -INT8_C( 118),  INT8_C(  57),
         INT8_C(  32),  INT8_C(  23),  INT8_C(   0),  INT8_C(  38), -INT8_C(  89),  INT8_C(   0),  INT8_C(  18),  INT8_C( 108),
         INT8_C( 118), -INT8_C(  96), -INT8_C(  48),  INT8_C(  98), -INT8_C(   2),  INT8_C(  58),  INT8_C(  90), -INT8_C(  73),
         INT8_C(  75),  INT8_C(  60), -INT8_C(  23), -INT8_C(  68),  INT8_C(  52), -INT8_C(  76), -INT8_C(  66),  INT8_C( 108) },
                         UINT64_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi8(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi8(test_vec[i].b);
    simde__mmask64 r = simde_mm512_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a_[64];
    int8_t b_[64];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m512i a = simde_mm512_loadu_epi8(a_);
    simde__m512i b = simde_mm512_loadu_epi8(b_);
    simde__mmask64 r = simde_mm512_cmp_epi8_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i8x64(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i8x64(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask64(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_epi16_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const simde__mmask32 r;
  } test_vec[] = {
    { { -INT16_C( 26663),  INT16_C( 31647), -INT16_C(  3155), -INT16_C(  4891),  INT16_C( 22831), -INT16_C( 24110), -INT16_C( 21561), -INT16_C(  9836),
         INT16_C( 21550),  INT16_C( 13739), -INT16_C( 24088),  INT16_C( 17127),  INT16_C( 13677), -INT16_C(  1207), -INT16_C(  3618), -INT16_C( 18423),
        -INT16_C( 17743), -INT16_C(  2608),  INT16_C(  6300), -INT16_C( 13534), -INT16_C(  2959),  INT16_C( 14700),  INT16_C(   160), -INT16_C( 12782),
        -INT16_C( 32428), -INT16_C( 28335),  INT16_C( 11699),  INT16_C(  8438),  INT16_C( 16226),  INT16_C( 16668),  INT16_C(  9521), -INT16_C( 17927) },
      {  INT16_C( 11470),  INT16_C( 27375),  INT16_C(  4676), -INT16_C( 19146), -INT16_C( 24058), -INT16_C( 22802),  INT16_C(   419), -INT16_C(  2188),
        -INT16_C( 17534),  INT16_C( 13739), -INT16_C( 24088),  INT16_C( 19285),  INT16_C( 29152),  INT16_C(  4492), -INT16_C( 31337),  INT16_C( 26059),
        -INT16_C( 17743), -INT16_C(  2608),  INT16_C(  1740), -INT16_C( 11350), -INT16_C( 26200),  INT16_C( 19321), -INT16_C(  4454),  INT16_C(  7235),
        -INT16_C(  4439), -INT16_C( 28335), -INT16_C( 22897),  INT16_C( 28636),  INT16_C( 26648), -INT16_C( 20607),  INT16_C( 19693), -INT16_C( 25068) },
      UINT32_C(  33752576) },
    { {  INT16_C( 32375), -INT16_C( 24025), -INT16_C( 24616),  INT16_C( 17685),  INT16_C( 22080),  INT16_C(   730),  INT16_C(  6165), -INT16_C( 19501),
         INT16_C( 17310),  INT16_C( 12972), -INT16_C( 27711),  INT16_C( 23888),  INT16_C( 30363), -INT16_C(  4850), -INT16_C( 32515),  INT16_C( 29878),
        -INT16_C( 27650), -INT16_C( 10538), -INT16_C(  1485),  INT16_C(  2153),  INT16_C(   848),  INT16_C( 25973),  INT16_C( 18459), -INT16_C( 18152),
        -INT16_C(  7227),  INT16_C( 19713),  INT16_C(  2987),  INT16_C(  7708), -INT16_C( 11597), -INT16_C( 20442), -INT16_C(  9134),  INT16_C( 20517) },
      { -INT16_C(  1169), -INT16_C( 24025),  INT16_C( 20725),  INT16_C( 17685), -INT16_C( 30125),  INT16_C( 28331), -INT16_C( 15405),  INT16_C( 24359),
         INT16_C( 10417),  INT16_C( 12972), -INT16_C( 14166),  INT16_C( 23888),  INT16_C( 30363), -INT16_C(  4850),  INT16_C( 13138), -INT16_C( 15810),
         INT16_C( 25902),  INT16_C(  9316),  INT16_C( 31413),  INT16_C(  2153),  INT16_C(  5124), -INT16_C( 10378), -INT16_C( 25128), -INT16_C( 30410),
        -INT16_C(  7227),  INT16_C( 28603),  INT16_C(  2987),  INT16_C( 18125), -INT16_C(  9342), -INT16_C( 11212),  INT16_C( 29198),  INT16_C( 15510) },
      UINT32_C(2048345252) },
    { {  INT16_C( 25775), -INT16_C( 22282),  INT16_C(  9022),  INT16_C( 10385),  INT16_C(  3584), -INT16_C( 19189),  INT16_C(  8923), -INT16_C( 23595),
         INT16_C(  2581), -INT16_C( 20111), -INT16_C( 14436), -INT16_C( 16431), -INT16_C( 31878),  INT16_C(  8164),  INT16_C(  7449), -INT16_C( 19533),
        -INT16_C( 22202), -INT16_C( 23460), -INT16_C(  9092),  INT16_C( 32005),  INT16_C(  4330), -INT16_C(  6186),  INT16_C(  1843),  INT16_C( 18537),
        -INT16_C(  9710), -INT16_C( 23503), -INT16_C( 13406),  INT16_C(  7277),  INT16_C( 20815),  INT16_C( 21307),  INT16_C(  3485),  INT16_C( 31238) },
      {  INT16_C( 25240),  INT16_C(  5150),  INT16_C(  9022),  INT16_C( 10385), -INT16_C( 15308),  INT16_C( 26606),  INT16_C( 22475), -INT16_C(  8785),
        -INT16_C( 22223), -INT16_C( 11380), -INT16_C(  1675), -INT16_C( 15121),  INT16_C( 11083),  INT16_C( 32535),  INT16_C(  7449), -INT16_C( 19975),
         INT16_C(  6272), -INT16_C( 16698),  INT16_C( 22331),  INT16_C( 28647), -INT16_C( 10981), -INT16_C(  6186), -INT16_C( 31188),  INT16_C( 24004),
         INT16_C( 20527), -INT16_C( 23503),  INT16_C(  8266), -INT16_C( 27288),  INT16_C( 32587),  INT16_C( 25876),  INT16_C(  3485),  INT16_C(  7446) },
      UINT32_C(2007465710) },
    { { -INT16_C( 10798),  INT16_C(  4092),  INT16_C( 25518), -INT16_C( 27928), -INT16_C( 31643),  INT16_C( 29865), -INT16_C( 26633),  INT16_C( 20551),
        -INT16_C(  3151), -INT16_C(  4143), -INT16_C( 11810), -INT16_C( 12693), -INT16_C( 16676),  INT16_C(  3903), -INT16_C( 19245), -INT16_C(  7743),
        -INT16_C( 24951), -INT16_C(    15), -INT16_C( 22270), -INT16_C(   190), -INT16_C( 26019), -INT16_C( 24855),  INT16_C( 12406),  INT16_C( 10222),
        -INT16_C( 30527), -INT16_C(  5098),  INT16_C( 27481),  INT16_C( 17611),  INT16_C(  2799), -INT16_C(   722),  INT16_C(  5404), -INT16_C( 11292) },
      { -INT16_C( 10798),  INT16_C(  9682),  INT16_C( 25518),  INT16_C(  3038), -INT16_C( 14339),  INT16_C( 29865), -INT16_C( 26633),  INT16_C(  6811),
        -INT16_C( 19882),  INT16_C(  9990), -INT16_C( 11810), -INT16_C( 12693), -INT16_C( 16676), -INT16_C(  1840), -INT16_C( 19245), -INT16_C( 22836),
        -INT16_C( 24951),  INT16_C( 14283), -INT16_C( 22270), -INT16_C(   190), -INT16_C(  5264),  INT16_C( 26483),  INT16_C(  3970), -INT16_C(  9855),
        -INT16_C( 30527), -INT16_C( 24832),  INT16_C( 27481),  INT16_C( 13677),  INT16_C( 15913), -INT16_C(   722), -INT16_C(  1294),  INT16_C( 31907) },
      UINT32_C(         0) },
    { {  INT16_C( 18961),  INT16_C( 21137),  INT16_C(  4808),  INT16_C( 21364), -INT16_C(  1406),  INT16_C( 25850),  INT16_C( 32701), -INT16_C( 17088),
         INT16_C( 28289), -INT16_C( 25053), -INT16_C( 29064), -INT16_C( 27773), -INT16_C( 19674), -INT16_C(  8015),  INT16_C( 20070), -INT16_C( 29831),
         INT16_C( 13646),  INT16_C( 10973),  INT16_C( 20765),  INT16_C( 10813),  INT16_C( 30795),  INT16_C(  2052),  INT16_C( 17655),  INT16_C( 31173),
         INT16_C( 10930), -INT16_C( 10009),  INT16_C( 27145), -INT16_C( 28902),  INT16_C( 17965), -INT16_C( 26865), -INT16_C(  7787), -INT16_C(  7282) },
      {  INT16_C(   148), -INT16_C( 20031),  INT16_C( 15953), -INT16_C( 25264),  INT16_C( 21686), -INT16_C( 20827),  INT16_C( 27544),  INT16_C( 19239),
         INT16_C(  3733), -INT16_C( 25053), -INT16_C( 29064), -INT16_C( 27186), -INT16_C(  8790), -INT16_C(  8660),  INT16_C( 20070), -INT16_C(  1420),
         INT16_C( 13646), -INT16_C( 24405), -INT16_C(   908),  INT16_C( 10813), -INT16_C(  7600), -INT16_C(  5672), -INT16_C(   179), -INT16_C(  7372),
         INT16_C( 22285), -INT16_C( 31359),  INT16_C( 20453), -INT16_C( 28902),  INT16_C( 17965), -INT16_C( 27795), -INT16_C(  7787), -INT16_C(  7282) },
      UINT32_C( 670480895) },
    { {  INT16_C(  2613),  INT16_C( 29526), -INT16_C( 12959), -INT16_C( 16300),  INT16_C(  9877), -INT16_C( 28688), -INT16_C(  7848), -INT16_C( 14973),
         INT16_C( 23009), -INT16_C( 13469),  INT16_C(  9481), -INT16_C( 19597),  INT16_C( 12712),  INT16_C( 28812),  INT16_C( 14569),  INT16_C( 31816),
        -INT16_C(  3305),  INT16_C( 24978), -INT16_C(  6406), -INT16_C( 24603),  INT16_C(  4364), -INT16_C( 10465), -INT16_C( 23860),  INT16_C( 29084),
         INT16_C(   249), -INT16_C( 14430), -INT16_C( 20269),  INT16_C( 31812), -INT16_C( 12063), -INT16_C( 13588),  INT16_C( 20744),  INT16_C( 19176) },
      {  INT16_C( 31245),  INT16_C(  1963), -INT16_C( 12959),  INT16_C( 28054), -INT16_C( 18978), -INT16_C( 21947), -INT16_C(  7848),  INT16_C( 20764),
         INT16_C( 23009), -INT16_C( 18975),  INT16_C(  9481), -INT16_C(  5583),  INT16_C(  7669), -INT16_C(   588), -INT16_C( 25489),  INT16_C( 31816),
        -INT16_C(  3305),  INT16_C( 30851),  INT16_C(  6592), -INT16_C( 24603),  INT16_C( 10959),  INT16_C( 10057),  INT16_C( 25868), -INT16_C(  4744),
         INT16_C( 22974), -INT16_C( 14430), -INT16_C( 11393),  INT16_C( 29873),  INT16_C( 26097),  INT16_C( 24690), -INT16_C( 17918),  INT16_C(  6620) },
      UINT32_C(3398039382) },
    { {  INT16_C( 29047), -INT16_C(  3325),  INT16_C( 24825), -INT16_C( 10481), -INT16_C(  9941), -INT16_C(  9102), -INT16_C( 30915), -INT16_C(  8498),
         INT16_C( 11829), -INT16_C( 19804),  INT16_C( 16994), -INT16_C(  4916), -INT16_C( 13737), -INT16_C( 22059),  INT16_C(  7191), -INT16_C( 14485),
        -INT16_C( 25796),  INT16_C( 13735), -INT16_C( 18437),  INT16_C(  3789),  INT16_C( 32400), -INT16_C( 13054),  INT16_C( 28439), -INT16_C(  7253),
         INT16_C( 20382), -INT16_C( 20212), -INT16_C( 10115), -INT16_C( 11107),  INT16_C( 29346), -INT16_C(  8323),  INT16_C(  5181),  INT16_C( 31124) },
      {  INT16_C( 15279), -INT16_C( 21842), -INT16_C( 17422), -INT16_C( 32048), -INT16_C( 11463),  INT16_C( 16207), -INT16_C(  1373), -INT16_C(  3550),
         INT16_C( 11829), -INT16_C( 19804),  INT16_C( 16646), -INT16_C( 22138),  INT16_C(   948), -INT16_C(  3704),  INT16_C(  7191), -INT16_C( 14485),
         INT16_C(  6488),  INT16_C( 19057),  INT16_C( 17108),  INT16_C(  3789),  INT16_C(  7189), -INT16_C( 18355),  INT16_C( 28439),  INT16_C( 19627),
         INT16_C( 20382), -INT16_C( 23298), -INT16_C( 31600),  INT16_C( 17485), -INT16_C( 10617), -INT16_C( 25034), -INT16_C( 24078),  INT16_C( 19045) },
      UINT32_C(4130343967) },
    { {  INT16_C(  8243),  INT16_C(  9778),  INT16_C( 16019), -INT16_C(  3944), -INT16_C( 24527), -INT16_C( 12553),  INT16_C( 28428), -INT16_C( 31409),
        -INT16_C(  3681), -INT16_C(  4229), -INT16_C( 27524), -INT16_C(  3131),  INT16_C( 27187), -INT16_C( 15881), -INT16_C( 25152),  INT16_C( 16149),
         INT16_C( 18274), -INT16_C(  2715), -INT16_C(   379), -INT16_C( 18458), -INT16_C( 13673),  INT16_C( 10093), -INT16_C( 30494), -INT16_C( 32392),
        -INT16_C(  2439), -INT16_C(  2733),  INT16_C(  6538), -INT16_C( 16919),  INT16_C( 13671), -INT16_C( 28801), -INT16_C( 27615), -INT16_C( 31794) },
      {  INT16_C( 13531),  INT16_C( 24697),  INT16_C( 24370), -INT16_C( 12265), -INT16_C( 31508), -INT16_C( 12553),  INT16_C( 28428), -INT16_C( 31409),
        -INT16_C( 23963), -INT16_C(  4229),  INT16_C( 25787),  INT16_C( 16301),  INT16_C( 11332),  INT16_C( 26062), -INT16_C( 25152), -INT16_C( 25623),
         INT16_C( 25297),  INT16_C(  1019),  INT16_C(  5057), -INT16_C( 21037), -INT16_C( 13673), -INT16_C( 23429), -INT16_C( 13767), -INT16_C( 24791),
        -INT16_C( 23444),  INT16_C( 10382),  INT16_C( 15112),  INT16_C( 19559),  INT16_C( 13671),  INT16_C( 10162), -INT16_C( 25646), -INT16_C( 23614) },
                UINT32_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi16(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi16(test_vec[i].b);
    simde__mmask32 r = simde_mm512_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[32];
    int16_t b_[32];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i16() & 3))
        a_[j] = b_[j];

    simde__m512i a = simde_mm512_loadu_epi16(a_);
    simde__m512i b = simde_mm512_loadu_epi16(b_);
    simde__mmask32 r = simde_mm512_cmp_epi16_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i16x32(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i16x32(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask32(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_epi32_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const simde__mmask16 r;
  } test_vec[] = {
    { { -INT32_C(  1234438653),  INT32_C(  1498911642), -INT32_C(  1552866568),  INT32_C(  1529431642),  INT32_C(   839041684), -INT32_C(  1606076101), -INT32_C(  1385333212), -INT32_C(  1959327209),
         INT32_C(  1668099529),  INT32_C(  1973279853), -INT32_C(    44290330),  INT32_C(  1875525395),  INT32_C(  1936815799), -INT32_C(  2030508190), -INT32_C(   298621595),  INT32_C(  1081843288) },
      { -INT32_C(   487145868),  INT32_C(  1498911642), -INT32_C(  1552866568), -INT32_C(   840702698), -INT32_C(  1103080356), -INT32_C(  1606076101), -INT32_C(   158369634), -INT32_C(  1959327209),
         INT32_C(  1668099529), -INT32_C(  1178811200), -INT32_C(    44290330),  INT32_C(  1875525395), -INT32_C(   819065964), -INT32_C(   495946940),  INT32_C(    47841259), -INT32_C(   762441719) },
      UINT16_C( 3494) },
    { {  INT32_C(  1452446509), -INT32_C(  1717521140), -INT32_C(  1351082373),  INT32_C(   861683008),  INT32_C(   837484339),  INT32_C(  1361171151), -INT32_C(   967043476),  INT32_C(  1448856793),
         INT32_C(  1861811973), -INT32_C(  1620824028), -INT32_C(  1907434418), -INT32_C(   742282726),  INT32_C(  1076702320),  INT32_C(  2109633555),  INT32_C(  1882254692), -INT32_C(   660009703) },
      { -INT32_C(  1583072899), -INT32_C(  1220474775), -INT32_C(    12152859), -INT32_C(  1429076166),  INT32_C(   837484339), -INT32_C(  1298432946), -INT32_C(  2094805911),  INT32_C(  1448856793),
         INT32_C(  1861811973), -INT32_C(   316262136),  INT32_C(    49048776), -INT32_C(  1481785741),  INT32_C(   198743997),  INT32_C(  2109633555),  INT32_C(   419488064), -INT32_C(  1284482131) },
      UINT16_C( 1542) },
    { {  INT32_C(   528531700),  INT32_C(  1375655787),  INT32_C(  1966212249),  INT32_C(  1131518837),  INT32_C(    32002341), -INT32_C(   220637185), -INT32_C(  2111665093), -INT32_C(  1877571428),
        -INT32_C(  1263495607),  INT32_C(   117812845), -INT32_C(  1954793549),  INT32_C(   652115201), -INT32_C(  1339570482), -INT32_C(   647731554), -INT32_C(  1889045783), -INT32_C(  2059555821) },
      {  INT32_C(  1963773960),  INT32_C(   830214526),  INT32_C(  1253963849),  INT32_C(  1131518837),  INT32_C(    15964258), -INT32_C(   220637185), -INT32_C(   310299426), -INT32_C(  1371168346),
        -INT32_C(   266120847), -INT32_C(  1423859614),  INT32_C(   251059864), -INT32_C(   598644870), -INT32_C(    19118593),  INT32_C(  2062595484), -INT32_C(  1889045783), -INT32_C(  2059555821) },
      UINT16_C(62953) },
    { { -INT32_C(  1397461441),  INT32_C(   344900160),  INT32_C(   642801527),  INT32_C(  1041429755), -INT32_C(     1738154), -INT32_C(   829767179),  INT32_C(  1192202619),  INT32_C(   965130234),
         INT32_C(  1726298662),  INT32_C(  1954182140), -INT32_C(   845493550), -INT32_C(  1123308354),  INT32_C(  2065557638),  INT32_C(   608806825),  INT32_C(   862673209),  INT32_C(   913109520) },
      {  INT32_C(   698175788), -INT32_C(  1768089660), -INT32_C(  1604110366),  INT32_C(  1767730915), -INT32_C(     1738154),  INT32_C(  1529097762), -INT32_C(  1735487609), -INT32_C(  1362167167),
         INT32_C(   265775947),  INT32_C(  1705342083), -INT32_C(  1912272725), -INT32_C(   856136842), -INT32_C(     3416611), -INT32_C(  1822757109),  INT32_C(     2877567),  INT32_C(   766441954) },
      UINT16_C(    0) },
    { {  INT32_C(   409156628),  INT32_C(  1493539745), -INT32_C(  1664724730),  INT32_C(   943376073), -INT32_C(  1481140450),  INT32_C(  1831684419), -INT32_C(  1621018006),  INT32_C(  1097827884),
         INT32_C(  1683608258),  INT32_C(  1941790317),  INT32_C(  2031059887),  INT32_C(    83865318),  INT32_C(  2021211956), -INT32_C(   219830648),  INT32_C(   227690208), -INT32_C(  1890713140) },
      {  INT32_C(  1089710035), -INT32_C(  1263292411),  INT32_C(  2083373619),  INT32_C(  1058922251), -INT32_C(  1481140450),  INT32_C(   513383485), -INT32_C(  1356125213), -INT32_C(    12682964),
         INT32_C(   624898336),  INT32_C(   349827809), -INT32_C(  1081079884),  INT32_C(    83865318), -INT32_C(  2102675899),  INT32_C(   916473171), -INT32_C(  1645884560),  INT32_C(  1687954500) },
      UINT16_C(63471) },
    { {  INT32_C(  1141001457), -INT32_C(   318693713),  INT32_C(   515247741), -INT32_C(   776581777),  INT32_C(   293050934), -INT32_C(   911551490),  INT32_C(   522142850),  INT32_C(   914904133),
         INT32_C(  1350208161),  INT32_C(   641563560), -INT32_C(   112921719),  INT32_C(  1912841723), -INT32_C(  1819721324), -INT32_C(  1755565291), -INT32_C(  1112114313),  INT32_C(  1911766736) },
      {  INT32_C(  1908502216),  INT32_C(  1939341033),  INT32_C(   862772210),  INT32_C(  1789540053), -INT32_C(  1929563273), -INT32_C(   568108698), -INT32_C(  1516512555), -INT32_C(   518615528),
        -INT32_C(   430778372), -INT32_C(   950408747), -INT32_C(  1711618620),  INT32_C(  1912841723),  INT32_C(  1073676504),  INT32_C(   790438490),  INT32_C(   366262524),  INT32_C(  1140255302) },
      UINT16_C(36816) },
    { {  INT32_C(   848044500),  INT32_C(   296764458), -INT32_C(  1963567074),  INT32_C(  1196844100), -INT32_C(  2086046591),  INT32_C(  1247589409),  INT32_C(  2101382144),  INT32_C(  1210100252),
        -INT32_C(  1564842691),  INT32_C(  2088780893),  INT32_C(  1342601116), -INT32_C(  1441558744),  INT32_C(  2116926467),  INT32_C(   518954269), -INT32_C(  1061193971),  INT32_C(   495827928) },
      {  INT32_C(   848044500),  INT32_C(    28183909), -INT32_C(  1403931516),  INT32_C(  1196844100),  INT32_C(  1053131809),  INT32_C(  1247589409), -INT32_C(  1148192542), -INT32_C(  1999102797),
        -INT32_C(  1564842691), -INT32_C(   157063054),  INT32_C(  1621292060),  INT32_C(  2057894233), -INT32_C(  1632080515),  INT32_C(    82318625), -INT32_C(  1061193971), -INT32_C(   783771757) },
      UINT16_C(45762) },
    { {  INT32_C(   199368610), -INT32_C(  1216309145),  INT32_C(  1661360438),  INT32_C(  1603130128), -INT32_C(  1520839238), -INT32_C(  1928786624), -INT32_C(   179247884),  INT32_C(  1859947980),
         INT32_C(   343588780), -INT32_C(  1916011945),  INT32_C(  1676726611),  INT32_C(  1606581668), -INT32_C(  1828447149),  INT32_C(  1663044905),  INT32_C(  1435267977), -INT32_C(  1379702784) },
      {  INT32_C(  2143370535), -INT32_C(  1978889161),  INT32_C(    49216861), -INT32_C(   849235846), -INT32_C(   178166324), -INT32_C(    78085774), -INT32_C(   179247884),  INT32_C(  1956779085),
        -INT32_C(  1980538031),  INT32_C(  1276313839),  INT32_C(  2001601021),  INT32_C(  2118496178), -INT32_C(  2022398444),  INT32_C(   478333991), -INT32_C(    32386127),  INT32_C(   947041255) },
           UINT16_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi32(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi32(test_vec[i].b);
    simde__mmask16 r = simde_mm512_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[16];
    int32_t b_[16];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i32() & 3))
        a_[j] = b_[j];

    simde__m512i a = simde_mm512_loadu_epi32(a_);
    simde__m512i b = simde_mm512_loadu_epi32(b_);
    simde__mmask16 r = simde_mm512_cmp_epi32_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i32x16(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i32x16(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask16(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_epi64_mask (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const simde__mmask8 r;
  } test_vec[] = {
    { {  INT64_C(  815426814384182538), -INT64_C( 3171329385968114620), -INT64_C( 8230427869076701561),  INT64_C( 2084909417815204586),
         INT64_C( 4992461192020513197), -INT64_C( 6997972495571653353),  INT64_C( 1544794234400857551), -INT64_C( 1133014558105260776) },
      {  INT64_C(  815426814384182538),  INT64_C( 5599850644529834206), -INT64_C( 8230427869076701561),  INT64_C( 6054806343208374738),
        -INT64_C( 7999325655028221805), -INT64_C( 6997972495571653353), -INT64_C( 8109267974603401356), -INT64_C( 4663851055046119768) },
      UINT8_C( 37) },
    { {  INT64_C(  314943195812753306), -INT64_C( 2501697685225398533), -INT64_C(  123825240566403747), -INT64_C( 4792185540628318021),
        -INT64_C( 6267670406683684139),  INT64_C( 6191148726960572633), -INT64_C( 8366236954574013875),  INT64_C( 7659196439573376538) },
      {  INT64_C(  314943195812753306),  INT64_C( 2593227683338024623),  INT64_C( 2383792892642253496),  INT64_C( 7293974096267030761),
        -INT64_C( 6267670406683684139), -INT64_C( 3661061373052316104),  INT64_C( 1533872782177549350),  INT64_C( 7659196439573376538) },
      UINT8_C( 78) },
    { {  INT64_C( 7306782805314696068),  INT64_C( 4568011377797348132),  INT64_C(  577426160441628932), -INT64_C( 9189351345856662814),
        -INT64_C(  559186510474580840), -INT64_C( 6521585401834452454), -INT64_C(  937168664740034858),  INT64_C( 1723173858224888238) },
      {  INT64_C( 6533081766334226909),  INT64_C( 7294731470644048630),  INT64_C( 7779697587739917241), -INT64_C( 9189351345856662814),
        -INT64_C( 8621616388207647041), -INT64_C( 4408079514205162801), -INT64_C(  937168664740034858), -INT64_C( 8117514230509080012) },
      UINT8_C(110) },
    { { -INT64_C( 7087406018461340711),  INT64_C( 1578307511195951783), -INT64_C( 3206819524879919840), -INT64_C( 3656953735145910880),
        -INT64_C( 2472103061205556146),  INT64_C( 3466626087279976835), -INT64_C( 8054510983497852476),  INT64_C( 4195864689923673698) },
      { -INT64_C(  723172496797065285),  INT64_C( 1578307511195951783),  INT64_C( 6169846445109806879), -INT64_C( 3948700237086378111),
        -INT64_C( 5337372428842416251),  INT64_C( 9052608391539333179),  INT64_C( 5645028817443232602),  INT64_C( 4195864689923673698) },
      UINT8_C(  0) },
    { { -INT64_C( 1344772335003037575),  INT64_C( 6912793192118502986), -INT64_C( 4364594109481085826), -INT64_C(  687703303177053746),
        -INT64_C( 6194245874054605388), -INT64_C( 7983050271184920002), -INT64_C( 6835846668000897898),  INT64_C(  640229329692288366) },
      { -INT64_C( 1344772335003037575),  INT64_C( 6912793192118502986), -INT64_C( 5382118327105768703), -INT64_C(  687703303177053746),
        -INT64_C( 8757538391267814438), -INT64_C( 7392990187872883649), -INT64_C( 3377535878610606970),  INT64_C(  640229329692288366) },
      UINT8_C(116) },
    { {  INT64_C( 6819904318137959795), -INT64_C( 1142813134477988786), -INT64_C( 3851044372280088878), -INT64_C( 8900216187008922246),
        -INT64_C( 5106163422351965971),  INT64_C( 2639724926454197296),  INT64_C( 2668071249071461296), -INT64_C(  581321648712506563) },
      { -INT64_C(   62804312891552516), -INT64_C( 4123797131256320821), -INT64_C( 3851044372280088878), -INT64_C( 8900216187008922246),
        -INT64_C( 1986355611550664288),  INT64_C( 5141222314171768888),  INT64_C( 2668071249071461296), -INT64_C( 2252320706351453007) },
      UINT8_C(207) },
    { { -INT64_C( 2622263577888628257),  INT64_C( 8339754642517624376),  INT64_C( 3200790277648262730),  INT64_C( 6542041936396860199),
         INT64_C( 5829708661834795745),  INT64_C( 7463910560856800826), -INT64_C(  449693853726649143),  INT64_C( 5182974009089025561) },
      {  INT64_C( 2188251220115954666),  INT64_C( 8339754642517624376),  INT64_C( 2657909982116818455),  INT64_C( 4746412852921284401),
        -INT64_C( 6682025271411920575),  INT64_C( 5427035690136793395),  INT64_C( 6166015185547188578),  INT64_C( 3559474213266672333) },
      UINT8_C(188) },
    { {  INT64_C( 5977711872274790846), -INT64_C( 6731727139327459557),  INT64_C( 5009458858281107556), -INT64_C( 9018258734263313439),
         INT64_C( 2992163722505636791), -INT64_C( 4112083580961091908),  INT64_C( 6344837915668859517),  INT64_C( 2681739022452898111) },
      {  INT64_C( 5977711872274790846), -INT64_C( 2821303568077464282),  INT64_C( 6615978186106779393), -INT64_C( 7063534414381159578),
        -INT64_C( 3757586874036024255), -INT64_C( 4112083580961091908), -INT64_C( 6764671361622241623), -INT64_C( 5046441195229204357) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi64(test_vec[i].a);
    simde__m512i b = simde_mm512_loadu_epi64(test_vec[i].b);
    simde__mmask8 r = simde_mm512_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[8];
    int64_t b_[8];

    simde_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    simde_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if (!(simde_test_codegen_random_i8() & 3))
        a_[j] = b_[j];

    simde__m512i a = simde_mm512_loadu_epi64(a_);
    simde__m512i b = simde_mm512_loadu_epi64(b_);
    simde__mmask8 r = simde_mm512_cmp_epi64_mask(a, b, HEDLEY_STATIC_CAST(SIMDE_MM_CMPINT_ENUM, i));

    simde_test_x86_write_i64x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i64x8(2, b, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_mmask8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_cmp_ps_mask (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde_float32 a[16];
    const simde_float32 b[16];
    const int imm8;
    const simde__mmask16 r;
  } test_vec[] = {
    { { SIMDE_FLOAT32_C(   662.74), SIMDE_FLOAT32_C(  -296.50), SIMDE_FLOAT32_C(    51.89), SIMDE_FLOAT32_C(  -877.47),
        SIMDE_FLOAT32_C(   872.85), SIMDE_FLOAT32_C(  -965.85), SIMDE_FLOAT32_C(  -688.43), SIMDE_FLOAT32_C(   580.42),
        SIMDE_FLOAT32_C(   433.23), SIMDE_FLOAT32_C(   579.01), SIMDE_FLOAT32_C(   251.51), SIMDE_FLOAT32_C(  -341.82),
        SIMDE_FLOAT32_C(  -712.66), SIMDE_FLOAT32_C(  -533.80), SIMDE_FLOAT32_C(   415.61), SIMDE_FLOAT32_C(   350.44) },
      { SIMDE_FLOAT32_C(  -243.73), SIMDE_FLOAT32_C(  -899.82), SIMDE_FLOAT32_C(  -876.74), SIMDE_FLOAT32_C(   548.84),
        SIMDE_FLOAT32_C(   112.62), SIMDE_FLOAT32_C(   228.46), SIMDE_FLOAT32_C(   592.01), SIMDE_FLOAT32_C(  -633.78),
        SIMDE_FLOAT32_C(   405.27), SIMDE_FLOAT32_C(   447.83), SIMDE_FLOAT32_C(  -706.03), SIMDE_FLOAT32_C(   628.46),
        SIMDE_FLOAT32_C(  -379.69), SIMDE_FLOAT32_C(   592.13), SIMDE_FLOAT32_C(   228.45), SIMDE_FLOAT32_C(  -716.95) },
       INT32_C(           0),
      UINT16_C(    0) },
    { { SIMDE_FLOAT32_C(  -704.37), SIMDE_FLOAT32_C(  -719.66), SIMDE_FLOAT32_C(  -594.42), SIMDE_FLOAT32_C(  -831.52),
        SIMDE_FLOAT32_C(  -685.51), SIMDE_FLOAT32_C(  -282.86), SIMDE_FLOAT32_C(   748.90), SIMDE_FLOAT32_C(   747.73),
        SIMDE_FLOAT32_C(  -703.85), SIMDE_FLOAT32_C(     0.41), SIMDE_FLOAT32_C(  -594.10), SIMDE_FLOAT32_C(  -416.51),
        SIMDE_FLOAT32_C(   466.61), SIMDE_FLOAT32_C(   821.52), SIMDE_FLOAT32_C(   933.94), SIMDE_FLOAT32_C(  -777.12) },
      { SIMDE_FLOAT32_C(   921.70), SIMDE_FLOAT32_C(  -942.80), SIMDE_FLOAT32_C(   771.72), SIMDE_FLOAT32_C(    34.32),
        SIMDE_FLOAT32_C(   285.66), SIMDE_FLOAT32_C(   363.73), SIMDE_FLOAT32_C(   400.54), SIMDE_FLOAT32_C(  -309.07),
        SIMDE_FLOAT32_C(  -188.44), SIMDE_FLOAT32_C(   694.50), SIMDE_FLOAT32_C(  -680.61), SIMDE_FLOAT32_C(   431.87),
        SIMDE_FLOAT32_C(   286.63), SIMDE_FLOAT32_C(   547.84), SIMDE_FLOAT32_C(   714.92), SIMDE_FLOAT32_C(   582.26) },
       INT32_C(           1),
      UINT16_C(35645) },
    { { SIMDE_FLOAT32_C(   828.18), SIMDE_FLOAT32_C(  -879.50), SIMDE_FLOAT32_C(   750.74), SIMDE_FLOAT32_C(  -857.33),
        SIMDE_FLOAT32_C(  -162.36), SIMDE_FLOAT32_C(   499.63), SIMDE_FLOAT32_C(   890.40), SIMDE_FLOAT32_C(   133.79),
        SIMDE_FLOAT32_C(  -499.96), SIMDE_FLOAT32_C(  -703.70), SIMDE_FLOAT32_C(   717.28), SIMDE_FLOAT32_C(   966.65),
        SIMDE_FLOAT32_C(  -882.18), SIMDE_FLOAT32_C(   651.22), SIMDE_FLOAT32_C(  -810.47), SIMDE_FLOAT32_C(  -960.48) },
      { SIMDE_FLOAT32_C(   708.42), SIMDE_FLOAT32_C(   961.25), SIMDE_FLOAT32_C(    73.84), SIMDE_FLOAT32_C(    -5.91),
        SIMDE_FLOAT32_C(   324.99), SIMDE_FLOAT32_C(  -525.63), SIMDE_FLOAT32_C(   685.02), SIMDE_FLOAT32_C(  -863.45),
        SIMDE_FLOAT32_C(  -831.12), SIMDE_FLOAT32_C(  -995.59), SIMDE_FLOAT32_C(   568.42), SIMDE_FLOAT32_C(   455.51),
        SIMDE_FLOAT32_C(   552.25), SIMDE_FLOAT32_C(   283.34), SIMDE_FLOAT32_C(    37.76), SIMDE_FLOAT32_C(   380.43) },
       INT32_C(           2),
      UINT16_C(53274) },
    { { SIMDE_FLOAT32_C(   403.83), SIMDE_FLOAT32_C(  -211.50), SIMDE_FLOAT32_C(   523.10), SIMDE_FLOAT32_C(  -758.53),
        SIMDE_FLOAT32_C(  -711.87), SIMDE_FLOAT32_C(   413.50), SIMDE_FLOAT32_C(   375.26), SIMDE_FLOAT32_C(  -211.83),
        SIMDE_FLOAT32_C(   709.80), SIMDE_FLOAT32_C(    92.55), SIMDE_FLOAT32_C(  -245.18), SIMDE_FLOAT32_C(   827.62),
        SIMDE_FLOAT32_C(  -256.23), SIMDE_FLOAT32_C(   -55.64), SIMDE_FLOAT32_C(   867.14), SIMDE_FLOAT32_C(  -547.81) },
      { SIMDE_FLOAT32_C(   -94.39), SIMDE_FLOAT32_C(   -59.02), SIMDE_FLOAT32_C(   446.28), SIMDE_FLOAT32_C(  -769.41),
        SIMDE_FLOAT32_C(   415.35), SIMDE_FLOAT32_C(   131.30), SIMDE_FLOAT32_C(  -632.86), SIMDE_FLOAT32_C(   584.23),
        SIMDE_FLOAT32_C(   135.71), SIMDE_FLOAT32_C(   935.56), SIMDE_FLOAT32_C(    39.74), SIMDE_FLOAT32_C(  -312.04),
        SIMDE_FLOAT32_C(   218.89), SIMDE_FLOAT32_C(  -922.50), SIMDE_FLOAT32_C(  -931.62), SIMDE_FLOAT32_C(  -377.28) },
       INT32_C(           3),
      UINT16_C(    0) },
    { { SIMDE_FLOAT32_C(  -134.00), SIMDE_FLOAT32_C(   591.48), SIMDE_FLOAT32_C(  -135.81), SIMDE_FLOAT32_C(   154.13),
        SIMDE_FLOAT32_C(     4.98), SIMDE_FLOAT32_C(  -760.55), SIMDE_FLOAT32_C(   942.30), SIMDE_FLOAT32_C(  -285.22),
        SIMDE_FLOAT32_C(   332.00), SIMDE_FLOAT32_C(  -302.88), SIMDE_FLOAT32_C(  -457.60), SIMDE_FLOAT32_C(  -924.23),
        SIMDE_FLOAT32_C(   641.48), SIMDE_FLOAT32_C(  -590.45), SIMDE_FLOAT32_C(  -472.04), SIMDE_FLOAT32_C(  -452.91) },
      { SIMDE_FLOAT32_C(   350.53), SIMDE_FLOAT32_C(   974.23), SIMDE_FLOAT32_C(  -222.32), SIMDE_FLOAT32_C(  -234.12),
        SIMDE_FLOAT32_C(   105.53), SIMDE_FLOAT32_C(   144.82), SIMDE_FLOAT32_C(  -649.88), SIMDE_FLOAT32_C(  -758.76),
        SIMDE_FLOAT32_C(    80.38), SIMDE_FLOAT32_C(   389.85), SIMDE_FLOAT32_C(   -70.81), SIMDE_FLOAT32_C(  -700.73),
        SIMDE_FLOAT32_C(   467.35), SIMDE_FLOAT32_C(    -2.42), SIMDE_FLOAT32_C(   -78.01), SIMDE_FLOAT32_C(  -666.65) },
       INT32_C(           4),
           UINT16_MAX },
    { { SIMDE_FLOAT32_C(  -410.94), SIMDE_FLOAT32_C(   786.18), SIMDE_FLOAT32_C(   487.47), SIMDE_FLOAT32_C(   594.03),
        SIMDE_FLOAT32_C(  -974.37), SIMDE_FLOAT32_C(   429.77), SIMDE_FLOAT32_C(  -691.18), SIMDE_FLOAT32_C(   357.63),
        SIMDE_FLOAT32_C(  -873.11), SIMDE_FLOAT32_C(  -148.78), SIMDE_FLOAT32_C(   433.40), SIMDE_FLOAT32_C(   768.37),
        SIMDE_FLOAT32_C(   260.77), SIMDE_FLOAT32_C(   961.36), SIMDE_FLOAT32_C(  -684.54), SIMDE_FLOAT32_C(  -388.70) },
      { SIMDE_FLOAT32_C(   935.59), SIMDE_FLOAT32_C(    93.14), SIMDE_FLOAT32_C(   377.18), SIMDE_FLOAT32_C(    41.12),
        SIMDE_FLOAT32_C(  -762.04), SIMDE_FLOAT32_C(   727.30), SIMDE_FLOAT32_C(   282.36), SIMDE_FLOAT32_C(   318.33),
        SIMDE_FLOAT32_C(   117.15), SIMDE_FLOAT32_C(  -788.45), SIMDE_FLOAT32_C(   617.60), SIMDE_FLOAT32_C(  -415.50),
        SIMDE_FLOAT32_C(   209.13), SIMDE_FLOAT32_C(  -460.41), SIMDE_FLOAT32_C(   -82.16), SIMDE_FLOAT32_C(   798.18) },
       INT32_C(           5),
      UINT16_C(14990) },
    { { SIMDE_FLOAT32_C(  -674.23), SIMDE_FLOAT32_C(  -594.68), SIMDE_FLOAT32_C(   392.22), SIMDE_FLOAT32_C(  -648.60),
        SIMDE_FLOAT32_C(   835.09), SIMDE_FLOAT32_C(   701.04), SIMDE_FLOAT32_C(   709.03), SIMDE_FLOAT32_C(   961.98),
        SIMDE_FLOAT32_C(  -447.74), SIMDE_FLOAT32_C(   142.43), SIMDE_FLOAT32_C(   730.35), SIMDE_FLOAT32_C(   813.03),
        SIMDE_FLOAT32_C(   103.79), SIMDE_FLOAT32_C(  -954.19), SIMDE_FLOAT32_C(  -575.67), SIMDE_FLOAT32_C(    39.38) },
      { SIMDE_FLOAT32_C(   138.95), SIMDE_FLOAT32_C(   801.51), SIMDE_FLOAT32_C(  -919.50), SIMDE_FLOAT32_C(   376.91),
        SIMDE_FLOAT32_C(   528.81), SIMDE_FLOAT32_C(   362.86), SIMDE_FLOAT32_C(  -304.76), SIMDE_FLOAT32_C(  -354.04),
        SIMDE_FLOAT32_C(   574.42), SIMDE_FLOAT32_C(  -687.16), SIMDE_FLOAT32_C(   230.46), SIMDE_FLOAT32_C(  -216.45),
        SIMDE_FLOAT32_C(  -147.57), SIMDE_FLOAT32_C(  -851.70), SIMDE_FLOAT32_C(  -418.27), SIMDE_FLOAT32_C(   178.19) },
       INT32_C(           6),
      UINT16_C( 7924) },
    { { SIMDE_FLOAT32_C(  -446.38), SIMDE_FLOAT32_C(   973.95), SIMDE_FLOAT32_C(   529.59), SIMDE_FLOAT32_C(  -611.29),
        SIMDE_FLOAT32_C(   674.99), SIMDE_FLOAT32_C(   238.63), SIMDE_FLOAT32_C(  -649.31), SIMDE_FLOAT32_C(  -772.76),
        SIMDE_FLOAT32_C(  -618.94), SIMDE_FLOAT32_C(  -918.96), SIMDE_FLOAT32_C(  -959.73), SIMDE_FLOAT32_C(   484.85),
        SIMDE_FLOAT32_C(  -873.15), SIMDE_FLOAT32_C(  -535.40), SIMDE_FLOAT32_C(  -475.77), SIMDE_FLOAT32_C(   265.80) },
      { SIMDE_FLOAT32_C(  -733.89), SIMDE_FLOAT32_C(  -395.27), SIMDE_FLOAT32_C(  -357.30), SIMDE_FLOAT32_C(   794.92),
        SIMDE_FLOAT32_C(   967.60), SIMDE_FLOAT32_C(   337.94), SIMDE_FLOAT32_C(  -559.13), SIMDE_FLOAT32_C(   542.02),
        SIMDE_FLOAT32_C(   650.78), SIMDE_FLOAT32_C(   671.33), SIMDE_FLOAT32_C(  -674.44), SIMDE_FLOAT32_C(  -496.79),
        SIMDE_FLOAT32_C(   819.63), SIMDE_FLOAT32_C(   -92.71), SIMDE_FLOAT32_C(   681.40), SIMDE_FLOAT32_C(  -626.75) },
       INT32_C(           7),
           UINT16_MAX },
    { { SIMDE_FLOAT32_C(  -118.76), SIMDE_FLOAT32_C(   210.99), SIMDE_FLOAT32_C(  -238.04), SIMDE_FLOAT32_C(  -443.77),
        SIMDE_FLOAT32_C(  -550.38), SIMDE_FLOAT32_C(   112.65), SIMDE_FLOAT32_C(  -216.52), SIMDE_FLOAT32_C(  -169.32),
        SIMDE_FLOAT32_C(   193.68), SIMDE_FLOAT32_C(  -176.25), SIMDE_FLOAT32_C(  -684.48), SIMDE_FLOAT32_C(   320.53),
        SIMDE_FLOAT32_C(   288.35), SIMDE_FLOAT32_C(  -160.25), SIMDE_FLOAT32_C(  -413.67), SIMDE_FLOAT32_C(   554.45) },
      { SIMDE_FLOAT32_C(   444.49), SIMDE_FLOAT32_C(   229.03), SIMDE_FLOAT32_C(   349.37), SIMDE_FLOAT32_C(   412.09),
        SIMDE_FLOAT32_C(  -433.02), SIMDE_FLOAT32_C(   790.25), SIMDE_FLOAT32_C(   -45.90), SIMDE_FLOAT32_C(  -782.24),
        SIMDE_FLOAT32_C(   461.58), SIMDE_FLOAT32_C(   279.66), SIMDE_FLOAT32_C(  -279.03), SIMDE_FLOAT32_C(   281.21),
        SIMDE_FLOAT32_C(  -813.04), SIMDE_FLOAT32_C(  -597.63), SIMDE_FLOAT32_C(   654.46), SIMDE_FLOAT32_C(    68.20) },
       INT32_C(           8),
      UINT16_C(    0) },
    { { SIMDE_FLOAT32_C(   613.36), SIMDE_FLOAT32_C(  -583.58), SIMDE_FLOAT32_C(   624.43), SIMDE_FLOAT32_C(  -937.02),
        SIMDE_FLOAT32_C(   529.07), SIMDE_FLOAT32_C(  -592.09), SIMDE_FLOAT32_C(  -106.35), SIMDE_FLOAT32_C(  -277.25),
        SIMDE_FLOAT32_C(   231.66), SIMDE_FLOAT32_C(   209.18), SIMDE_FLOAT32_C(  -956.71), SIMDE_FLOAT32_C(  -480.00),
        SIMDE_FLOAT32_C(  -951.07), SIMDE_FLOAT32_C(  -370.38), SIMDE_FLOAT32_C(  -925.54), SIMDE_FLOAT32_C(   493.42) },
      { SIMDE_FLOAT32_C(   858.65), SIMDE_FLOAT32_C(   423.83), SIMDE_FLOAT32_C(   -94.50), SIMDE_FLOAT32_C(  -574.37),
        SIMDE_FLOAT32_C(   214.07), SIMDE_FLOAT32_C(   859.61), SIMDE_FLOAT32_C(  -356.61), SIMDE_FLOAT32_C(  -324.35),
        SIMDE_FLOAT32_C(   139.27), SIMDE_FLOAT32_C(   364.36), SIMDE_FLOAT32_C(   956.86), SIMDE_FLOAT32_C(   326.23),
        SIMDE_FLOAT32_C(   766.72), SIMDE_FLOAT32_C(   611.33), SIMDE_FLOAT32_C(  -605.57), SIMDE_FLOAT32_C(   380.08) },
       INT32_C(           9),
      UINT16_C(32299) },
    { { SIMDE_FLOAT32_C(  -972.25), SIMDE_FLOAT32_C(  -981.14), SIMDE_FLOAT32_C(   443.06), SIMDE_FLOAT32_C(   556.82),
        SIMDE_FLOAT32_C(  -573.23), SIMDE_FLOAT32_C(  -663.28), SIMDE_FLOAT32_C(  -720.43), SIMDE_FLOAT32_C(   658.42),
        SIMDE_FLOAT32_C(   545.89), SIMDE_FLOAT32_C(  -677.14), SIMDE_FLOAT32_C(  -821.57), SIMDE_FLOAT32_C(   594.83),
        SIMDE_FLOAT32_C(   -47.52), SIMDE_FLOAT32_C(  -747.12), SIMDE_FLOAT32_C(    88.25), SIMDE_FLOAT32_C(  -188.87) },
      { SIMDE_FLOAT32_C(   676.71), SIMDE_FLOAT32_C(   993.75), SIMDE_FLOAT32_C(   236.76), SIMDE_FLOAT32_C(  -109.21),
        SIMDE_FLOAT32_C(   853.36), SIMDE_FLOAT32_C(   880.15), SIMDE_FLOAT32_C(   566.44), SIMDE_FLOAT32_C(    -7.37),
        SIMDE_FLOAT32_C(   244.51), SIMDE_FLOAT32_C(   523.30), SIMDE_FLOAT32_C(  -681.15), SIMDE_FLOAT32_C(    11.24),
        SIMDE_FLOAT32_C(   134.63), SIMDE_FLOAT32_C(  -286.72), SIMDE_FLOAT32_C(  -608.68), SIMDE_FLOAT32_C(   162.38) },
       INT32_C(          10),
      UINT16_C(46707) },
    { { SIMDE_FLOAT32_C(  -267.86), SIMDE_FLOAT32_C(   834.38), SIMDE_FLOAT32_C(  -280.80), SIMDE_FLOAT32_C(   158.91),
        SIMDE_FLOAT32_C(  -828.90), SIMDE_FLOAT32_C(    -1.23), SIMDE_FLOAT32_C(  -182.67), SIMDE_FLOAT32_C(   716.99),
        SIMDE_FLOAT32_C(   321.63), SIMDE_FLOAT32_C(    -4.24), SIMDE_FLOAT32_C(   311.82), SIMDE_FLOAT32_C(  -725.89),
        SIMDE_FLOAT32_C(   248.64), SIMDE_FLOAT32_C(  -599.93), SIMDE_FLOAT32_C(    85.24), SIMDE_FLOAT32_C(   -74.64) },
      { SIMDE_FLOAT32_C(  -606.18), SIMDE_FLOAT32_C(  -677.99), SIMDE_FLOAT32_C(   816.14), SIMDE_FLOAT32_C(  -752.82),
        SIMDE_FLOAT32_C(  -797.84), SIMDE_FLOAT32_C(   382.58), SIMDE_FLOAT32_C(   239.80), SIMDE_FLOAT32_C(   446.68),
        SIMDE_FLOAT32_C(   -94.12), SIMDE_FLOAT32_C(   558.66), SIMDE_FLOAT32_C(  -542.09), SIMDE_FLOAT32_C(  -959.49),
        SIMDE_FLOAT32_C(  -728.06), SIMDE_FLOAT32_C(  -150.77), SIMDE_FLOAT32_C(   202.89), SIMDE_FLOAT32_C(     4.08) },
       INT32_C(          11),
      UINT16_C(    0) },
    { { SIMDE_FLOAT32_C(  -316.38), SIMDE_FLOAT32_C(   922.09), SIMDE_FLOAT32_C(  -837.02), SIMDE_FLOAT32_C(  -145.29),
        SIMDE_FLOAT32_C(   -79.14), SIMDE_FLOAT32_C(   -19.68), SIMDE_FLOAT32_C(  -428.29), SIMDE_FLOAT32_C(  -757.51),
        SIMDE_FLOAT32_C(   976.07), SIMDE_FLOAT32_C(   883.53), SIMDE_FLOAT32_C(  -483.40), SIMDE_FLOAT32_C(   224.72),
        SIMDE_FLOAT32_C(  -716.41), SIMDE_FLOAT32_C(   601.84), SIMDE_FLOAT32_C(  -849.93), SIMDE_FLOAT32_C(  -322.59) },
      { SIMDE_FLOAT32_C(   923.85), SIMDE_FLOAT32_C(   966.22), SIMDE_FLOAT32_C(   -75.41), SIMDE_FLOAT32_C(  -873.98),
        SIMDE_FLOAT32_C(   348.80), SIMDE_FLOAT32_C(  -835.61), SIMDE_FLOAT32_C(   572.69), SIMDE_FLOAT32_C(  -745.32),
        SIMDE_FLOAT32_C(   723.05), SIMDE_FLOAT32_C(  -969.40), SIMDE_FLOAT32_C(  -704.81), SIMDE_FLOAT32_C(   994.98),
        SIMDE_FLOAT32_C(  -120.16), SIMDE_FLOAT32_C(   498.08), SIMDE_FLOAT32_C(    -0.94), SIMDE_FLOAT32_C(   563.45) },
       INT32_C(          12),
           UINT16_MAX },
    { { SIMDE_FLOAT32_C(   420.16), SIMDE_FLOAT32_C(   162.04), SIMDE_FLOAT32_C(  -581.83), SIMDE_FLOAT32_C(  -658.98),
        SIMDE_FLOAT32_C(  -857.64), SIMDE_FLOAT32_C(   -10.13), SIMDE_FLOAT32_C(  -416.49), SIMDE_FLOAT32_C(  -881.56),
        SIMDE_FLOAT32_C(  -126.60), SIMDE_FLOAT32_C(   100.10), SIMDE_FLOAT32_C(   343.15), SIMDE_FLOAT32_C(   156.99),
        SIMDE_FLOAT32_C(  -298.05), SIMDE_FLOAT32_C(   493.23), SIMDE_FLOAT32_C(   834.41), SIMDE_FLOAT32_C(  -374.20) },
      { SIMDE_FLOAT32_C(   459.44), SIMDE_FLOAT32_C(  -241.01), SIMDE_FLOAT32_C(  -248.18), SIMDE_FLOAT32_C(  -191.76),
        SIMDE_FLOAT32_C(   -76.61), SIMDE_FLOAT32_C(  -675.49), SIMDE_FLOAT32_C(    62.92), SIMDE_FLOAT32_C(  -353.57),
        SIMDE_FLOAT32_C(  -644.89), SIMDE_FLOAT32_C(   358.11), SIMDE_FLOAT32_C(  -358.58), SIMDE_FLOAT32_C(   234.95),
        SIMDE_FLOAT32_C(  -143.82), SIMDE_FLOAT32_C(   640.48), SIMDE_FLOAT32_C(  -201.60), SIMDE_FLOAT32_C(  -723.66) },
       INT32_C(          13),
      UINT16_C(50466) },
    { { SIMDE_FLOAT32_C(  -197.48), SIMDE_FLOAT32_C(   216.57), SIMDE_FLOAT32_C(  -382.64), SIMDE_FLOAT32_C(   -55.12),
        SIMDE_FLOAT32_C(  -793.56), SIMDE_FLOAT32_C(   200.87), SIMDE_FLOAT32_C(    63.32), SIMDE_FLOAT32_C(    79.84),
        SIMDE_FLOAT32_C(  -699.03), SIMDE_FLOAT32_C(  -593.53), SIMDE_FLOAT32_C(  -763.16), SIMDE_FLOAT32_C(     2.92),
        SIMDE_FLOAT32_C(   899.70), SIMDE_FLOAT32_C(  -928.76), SIMDE_FLOAT32_C(   628.72), SIMDE_FLOAT32_C(   359.14) },
      { SIMDE_FLOAT32_C(  -169.76), SIMDE_FLOAT32_C(  -619.46), SIMDE_FLOAT32_C(  -832.62), SIMDE_FLOAT32_C(   753.63),
        SIMDE_FLOAT32_C(  -294.96), SIMDE_FLOAT32_C(   230.30), SIMDE_FLOAT32_C(  -599.94), SIMDE_FLOAT32_C(    60.15),
        SIMDE_FLOAT32_C(  -411.60), SIMDE_FLOAT32_C(    41.48), SIMDE_FLOAT32_C(  -704.90), SIMDE_FLOAT32_C(   444.59),
        SIMDE_FLOAT32_C(  -318.04), SIMDE_FLOAT32_C(    93.51), SIMDE_FLOAT32_C(   720.93), SIMDE_FLOAT32_C(   484.48) },
       INT32_C(          14),
      UINT16_C( 4294) },
    { { SIMDE_FLOAT32_C(  -689.92), SIMDE_FLOAT32_C(  -661.71), SIMDE_FLOAT32_C(  -570.64), SIMDE_FLOAT32_C(  -483.48),
        SIMDE_FLOAT32_C(   539.16), SIMDE_FLOAT32_C(   492.68), SIMDE_FLOAT32_C(   596.36), SIMDE_FLOAT32_C(   840.13),
        SIMDE_FLOAT32_C(   899.15), SIMDE_FLOAT32_C(   833.20), SIMDE_FLOAT32_C(  -156.95), SIMDE_FLOAT32_C(   798.85),
        SIMDE_FLOAT32_C(   904.44), SIMDE_FLOAT32_C(  -528.23), SIMDE_FLOAT32_C(   157.99), SIMDE_FLOAT32_C(  -265.32) },
      { SIMDE_FLOAT32_C(  -147.69), SIMDE_FLOAT32_C(   325.37), SIMDE_FLOAT32_C(  -511.69), SIMDE_FLOAT32_C(   557.35),
        SIMDE_FLOAT32_C(  -444.33), SIMDE_FLOAT32_C(  -111.63), SIMDE_FLOAT32_C(  -382.50), SIMDE_FLOAT32_C(   144.07),
        SIMDE_FLOAT32_C(   929.85), SIMDE_FLOAT32_C(   -87.39), SIMDE_FLOAT32_C(  -411.34), SIMDE_FLOAT32_C(  -388.19),
        SIMDE_FLOAT32_C(  -993.88), SIMDE_FLOAT32_C(  -690.41), SIMDE_FLOAT32_C(  -903.71), SIMDE_FLOAT32_C(  -683.81) },
       INT32_C(          15),
           UINT16_MAX },
    { { SIMDE_FLOAT32_C(  -352.12), SIMDE_FLOAT32_C(  -474.35), SIMDE_FLOAT32_C(  -167.29), SIMDE_FLOAT32_C(  -812.96),
        SIMDE_FLOAT32_C(  -981.67), SIMDE_FLOAT32_C(  -570.92), SIMDE_FLOAT32_C(  -972.83), SIMDE_FLOAT32_C(   917.49),
        SIMDE_FLOAT32_C(  -737.72), SIMDE_FLOAT32_C(  -129.78), SIMDE_FLOAT32_C(   716.34), SIMDE_FLOAT32_C(  -833.28),
        SIMDE_FLOAT32_C(   341.99), SIMDE_FLOAT32_C(  -125.67), SIMDE_FLOAT32_C(   -98.59), SIMDE_FLOAT32_C(  -805.70) },
      { SIMDE_FLOAT32_C(  -800.30), SIMDE_FLOAT32_C(   389.72), SIMDE_FLOAT32_C(   751.65), SIMDE_FLOAT32_C(  -244.63),
        SIMDE_FLOAT32_C(  -721.91), SIMDE_FLOAT32_C(  -630.84), SIMDE_FLOAT32_C(   899.44), SIMDE_FLOAT32_C(  -792.06),
        SIMDE_FLOAT32_C(   281.76), SIMDE_FLOAT32_C(  -511.90), SIMDE_FLOAT32_C(  -180.26), SIMDE_FLOAT32_C(   287.88),
        SIMDE_FLOAT32_C(  -202.31), SIMDE_FLOAT32_C(   -83.97), SIMDE_FLOAT32_C(   604.08), SIMDE_FLOAT32_C(   445.56) },
       INT32_C(          16),
      UINT16_C(    0) },
    { { SIMDE_FLOAT32_C(   441.68), SIMDE_FLOAT32_C(  -563.21), SIMDE_FLOAT32_C(   632.60), SIMDE_FLOAT32_C(   460.01),
        SIMDE_FLOAT32_C(  -134.13), SIMDE_FLOAT32_C(   659.78), SIMDE_FLOAT32_C(   377.50), SIMDE_FLOAT32_C(   128.15),
        SIMDE_FLOAT32_C(  -470.00), SIMDE_FLOAT32_C(    93.84), SIMDE_FLOAT32_C(   294.87), SIMDE_FLOAT32_C(   871.99),
        SIMDE_FLOAT32_C(   968.16), SIMDE_FLOAT32_C(  -803.72), SIMDE_FLOAT32_C(  -933.71), SIMDE_FLOAT32_C(  -832.14) },
      { SIMDE_FLOAT32_C(   586.00), SIMDE_FLOAT32_C(   817.94), SIMDE_FLOAT32_C(   -76.77), SIMDE_FLOAT32_C(   864.08),
        SIMDE_FLOAT32_C(  -812.90), SIMDE_FLOAT32_C(  -177.33), SIMDE_FLOAT32_C(  -927.98), SIMDE_FLOAT32_C(   468.86),
        SIMDE_FLOAT32_C(   310.77), SIMDE_FLOAT32_C(  -108.24), SIMDE_FLOAT32_C(  -243.26), SIMDE_FLOAT32_C(  -891.55),
        SIMDE_FLOAT32_C(   807.79), SIMDE_FLOAT32_C(  -639.18), SIMDE_FLOAT32_C(   554.02), SIMDE_FLOAT32_C(   249.47) },
       INT32_C(          17),
      UINT16_C(57739) },
    { { SIMDE_FLOAT32_C(  -202.39), SIMDE_FLOAT32_C(   186.62), SIMDE_FLOAT32_C(  -290.52), SIMDE_FLOAT32_C(   663.47),
        SIMDE_FLOAT32_C(  -153.60), SIMDE_FLOAT32_C(  -913.02), SIMDE_FLOAT32_C(  -208.38), SIMDE_FLOAT32_C(   376.40),
        SIMDE_FLOAT32_C(   180.82), SIMDE_FLOAT32_C(  -913.51), SIMDE_FLOAT32_C(   248.39), SIMDE_FLOAT32_C(   148.98),
        SIMDE_FLOAT32_C(  -717.23), SIMDE_FLOAT32_C(   314.68), SIMDE_FLOAT32_C(   316.85), SIMDE_FLOAT32_C(   868.77) },
      { SIMDE_FLOAT32_C(   132.62), SIMDE_FLOAT32_C(  -759.92), SIMDE_FLOAT32_C(   732.85), SIMDE_FLOAT32_C(   319.72),
        SIMDE_FLOAT32_C(    62.75), SIMDE_FLOAT32_C(   804.87), SIMDE_FLOAT32_C(  -211.42), SIMDE_FLOAT32_C(  -626.48),
        SIMDE_FLOAT32_C(  -303.37), SIMDE_FLOAT32_C(   545.32), SIMDE_FLOAT32_C(  -518.02), SIMDE_FLOAT32_C(  -495.58),
        SIMDE_FLOAT32_C(   906.14), SIMDE_FLOAT32_C(  -964.00), SIMDE_FLOAT32_C(   753.90), SIMDE_FLOAT32_C(  -296.25) },
       INT32_C(          18),
      UINT16_C(21045) },
    { { SIMDE_FLOAT32_C(   222.62), SIMDE_FLOAT32_C(  -536.62), SIMDE_FLOAT32_C(  -632.78), SIMDE_FLOAT32_C(  -930.98),
        SIMDE_FLOAT32_C(  -449.64), SIMDE_FLOAT32_C(   158.84), SIMDE_FLOAT32_C(   445.42), SIMDE_FLOAT32_C(   731.18),
        SIMDE_FLOAT32_C(   245.34), SIMDE_FLOAT32_C(  -306.20), SIMDE_FLOAT32_C(  -119.84), SIMDE_FLOAT32_C(   528.11),
        SIMDE_FLOAT32_C(  -991.52), SIMDE_FLOAT32_C(  -802.99), SIMDE_FLOAT32_C(   396.88), SIMDE_FLOAT32_C(   141.10) },
      { SIMDE_FLOAT32_C(  -562.91), SIMDE_FLOAT32_C(   129.73), SIMDE_FLOAT32_C(  -539.18), SIMDE_FLOAT32_C(   499.84),
        SIMDE_FLOAT32_C(   -65.40), SIMDE_FLOAT32_C(   249.40), SIMDE_FLOAT32_C(   873.36), SIMDE_FLOAT32_C(   631.23),
        SIMDE_FLOAT32_C(  -205.28), SIMDE_FLOAT32_C(  -644.66), SIMDE_FLOAT32_C(  -864.35), SIMDE_FLOAT32_C(  -299.14),
        SIMDE_FLOAT32_C(  -608.67), SIMDE_FLOAT32_C(   889.55), SIMDE_FLOAT32_C(   404.61), SIMDE_FLOAT32_C(   613.95) },
       INT32_C(          19),
      UINT16_C(    0) },
    { { SIMDE_FLOAT32_C(  -647.07), SIMDE_FLOAT32_C(   771.83), SIMDE_FLOAT32_C(   682.97), SIMDE_FLOAT32_C(   -96.71),
        SIMDE_FLOAT32_C(   -69.32), SIMDE_FLOAT32_C(   128.39), SIMDE_FLOAT32_C(  -365.53), SIMDE_FLOAT32_C(  -823.99),
        SIMDE_FLOAT32_C(   822.19), SIMDE_FLOAT32_C(   514.63), SIMDE_FLOAT32_C(   704.12), SIMDE_FLOAT32_C(   830.67),
        SIMDE_FLOAT32_C(   711.64), SIMDE_FLOAT32_C(   101.00), SIMDE_FLOAT32_C(   -28.22), SIMDE_FLOAT32_C(  -851.27) },
      { SIMDE_FLOAT32_C(  -769.27), SIMDE_FLOAT32_C(   432.60), SIMDE_FLOAT32_C(   648.57), SIMDE_FLOAT32_C(   165.32),
        SIMDE_FLOAT32_C(  -318.00), SIMDE_FLOAT32_C(   521.93), SIMDE_FLOAT32_C(  -203.45), SIMDE_FLOAT32_C(   476.73),
        SIMDE_FLOAT32_C(   877.27), SIMDE_FLOAT32_C(   -67.79), SIMDE_FLOAT32_C(  -822.41), SIMDE_FLOAT32_C(  -731.40),
        SIMDE_FLOAT32_C(  -178.24), SIMDE_FLOAT32_C(   582.20), SIMDE_FLOAT32_C(   882.55), SIMDE_FLOAT32_C(   174.68) },
       INT32_C(          20),
           UINT16_MAX },
    { { SIMDE_FLOAT32_C(   354.04), SIMDE_FLOAT32_C(   565.52), SIMDE_FLOAT32_C(  -922.02), SIMDE_FLOAT32_C(  -715.29),
        SIMDE_FLOAT32_C(  -306.09), SIMDE_FLOAT32_C(  -287.55), SIMDE_FLOAT32_C(  -539.27), SIMDE_FLOAT32_C(  -483.90),
        SIMDE_FLOAT32_C(  -772.92), SIMDE_FLOAT32_C(  -835.15), SIMDE_FLOAT32_C(  -653.23), SIMDE_FLOAT32_C(   938.73),
        SIMDE_FLOAT32_C(   265.85), SIMDE_FLOAT32_C(   318.55), SIMDE_FLOAT32_C(  -912.54), SIMDE_FLOAT32_C(   496.58) },
      { SIMDE_FLOAT32_C(  -248.85), SIMDE_FLOAT32_C(   736.03), SIMDE_FLOAT32_C(  -338.10), SIMDE_FLOAT32_C(   433.16),
        SIMDE_FLOAT32_C(   257.97), SIMDE_FLOAT32_C(   458.45), SIMDE_FLOAT32_C(   -90.12), SIMDE_FLOAT32_C(   135.24),
        SIMDE_FLOAT32_C(  -609.34), SIMDE_FLOAT32_C(    87.47), SIMDE_FLOAT32_C(   403.84), SIMDE_FLOAT32_C(   212.42),
        SIMDE_FLOAT32_C(  -330.33), SIMDE_FLOAT32_C(   286.39), SIMDE_FLOAT32_C(  -612.90), SIMDE_FLOAT32_C(  -976.29) },
       INT32_C(          21),
      UINT16_C(47105) },
    { { SIMDE_FLOAT32_C(  -148.09), SIMDE_FLOAT32_C(  -534.92), SIMDE_FLOAT32_C(  -691.57), SIMDE_FLOAT32_C(   545.82),
        SIMDE_FLOAT32_C(   177.53), SIMDE_FLOAT32_C(  -230.85), SIMDE_FLOAT32_C(  -938.08), SIMDE_FLOAT32_C(   404.61),
        SIMDE_FLOAT32_C(   -65.99), SIMDE_FLOAT32_C(  -591.31), SIMDE_FLOAT32_C(   343.34), SIMDE_FLOAT32_C(  -800.14),
        SIMDE_FLOAT32_C(   727.24), SIMDE_FLOAT32_C(   430.80), SIMDE_FLOAT32_C(   696.43), SIMDE_FLOAT32_C(  -521.61) },
      { SIMDE_FLOAT32_C(   166.83), SIMDE_FLOAT32_C(  -641.66), SIMDE_FLOAT32_C(   911.55), SIMDE_FLOAT32_C(  -575.20),
        SIMDE_FLOAT32_C(   816.79), SIMDE_FLOAT32_C(  -178.57), SIMDE_FLOAT32_C(   560.03), SIMDE_FLOAT32_C(  -792.55),
        SIMDE_FLOAT32_C(   908.90), SIMDE_FLOAT32_C(   -36.13), SIMDE_FLOAT32_C(   419.87), SIMDE_FLOAT32_C(  -421.42),
        SIMDE_FLOAT32_C(  -749.74), SIMDE_FLOAT32_C(   806.97), SIMDE_FLOAT32_C(  -397.71), SIMDE_FLOAT32_C(   102.17) },
       INT32_C(          22),
      UINT16_C(20618) },
    { { SIMDE_FLOAT32_C(  -727.95), SIMDE_FLOAT32_C(   -89.29), SIMDE_FLOAT32_C(  -352.01), SIMDE_FLOAT32_C(   449.58),
        SIMDE_FLOAT32_C(   679.87), SIMDE_FLOAT32_C(  -290.09), SIMDE_FLOAT32_C(  -145.82), SIMDE_FLOAT32_C(  -386.13),
        SIMDE_FLOAT32_C(   118.59), SIMDE_FLOAT32_C(  -802.48), SIMDE_FLOAT32_C(  -186.27), SIMDE_FLOAT32_C(  -154.16),
        SIMDE_FLOAT32_C(   628.32), SIMDE_FLOAT32_C(  -489.83), SIMDE_FLOAT32_C(   324.23), SIMDE_FLOAT32_C(  -204.85) },
      { SIMDE_FLOAT32_C(  -131.50), SIMDE_FLOAT32_C(   235.78), SIMDE_FLOAT32_C(   219.94), SIMDE_FLOAT32_C(  -314.71),
        SIMDE_FLOAT32_C(  -942.79), SIMDE_FLOAT32_C(  -220.02), SIMDE_FLOAT32_C(  -107.26), SIMDE_FLOAT32_C(   966.11),
        SIMDE_FLOAT32_C(   743.85), SIMDE_FLOAT32_C(  -687.39), SIMDE_FLOAT32_C(  -455.31), SIMDE_FLOAT32_C(   994.11),
        SIMDE_FLOAT32_C(  -880.42), SIMDE_FLOAT32_C(   146.98), SIMDE_FLOAT32_C(    96.28), SIMDE_FLOAT32_C(  -608.37) },
       INT32_C(          23),
           UINT16_MAX },
    { { SIMDE_FLOAT32_C(  -942.31), SIMDE_FLOAT32_C(   744.27), SIMDE_FLOAT32_C(   841.21), SIMDE_FLOAT32_C(   737.56),
        SIMDE_FLOAT32_C(  -545.82), SIMDE_FLOAT32_C(  -304.61), SIMDE_FLOAT32_C(  -648.57), SIMDE_FLOAT32_C(   572.77),
        SIMDE_FLOAT32_C(  -107.09), SIMDE_FLOAT32_C(   165.16), SIMDE_FLOAT32_C(  -581.39), SIMDE_FLOAT32_C(  -478.77),
        SIMDE_FLOAT32_C(   675.33), SIMDE_FLOAT32_C(   742.84), SIMDE_FLOAT32_C(   316.38), SIMDE_FLOAT32_C(  -456.17) },
      { SIMDE_FLOAT32_C(   -21.39), SIMDE_FLOAT32_C(  -463.68), SIMDE_FLOAT32_C(   229.13), SIMDE_FLOAT32_C(    35.82),
        SIMDE_FLOAT32_C(   316.30), SIMDE_FLOAT32_C(  -878.13), SIMDE_FLOAT32_C(     1.93), SIMDE_FLOAT32_C(    60.15),
        SIMDE_FLOAT32_C(  -565.52), SIMDE_FLOAT32_C(   546.62), SIMDE_FLOAT32_C(    54.26), SIMDE_FLOAT32_C(  -445.94),
        SIMDE_FLOAT32_C(  -306.41), SIMDE_FLOAT32_C(  -849.45), SIMDE_FLOAT32_C(   -54.31), SIMDE_FLOAT32_C(  -248.72) },
       INT32_C(          24),
      UINT16_C(    0) },
    { { SIMDE_FLOAT32_C(   894.82), SIMDE_FLOAT32_C(  -213.10), SIMDE_FLOAT32_C(  -511.16), SIMDE_FLOAT32_C(  -651.01),
        SIMDE_FLOAT32_C(   482.29), SIMDE_FLOAT32_C(  -159.73), SIMDE_FLOAT32_C(   921.76), SIMDE_FLOAT32_C(  -624.79),
        SIMDE_FLOAT32_C(  -994.57), SIMDE_FLOAT32_C(  -659.63), SIMDE_FLOAT32_C(  -103.56), SIMDE_FLOAT32_C(   680.76),
        SIMDE_FLOAT32_C(  -916.80), SIMDE_FLOAT32_C(  -787.19), SIMDE_FLOAT32_C(  -775.40), SIMDE_FLOAT32_C(    61.82) },
      { SIMDE_FLOAT32_C(  -250.87), SIMDE_FLOAT32_C(   453.72), SIMDE_FLOAT32_C(  -902.36), SIMDE_FLOAT32_C(  -934.56),
        SIMDE_FLOAT32_C(   575.59), SIMDE_FLOAT32_C(    99.57), SIMDE_FLOAT32_C(   125.59), SIMDE_FLOAT32_C(  -989.93),
        SIMDE_FLOAT32_C(  -353.81), SIMDE_FLOAT32_C(  -820.15), SIMDE_FLOAT32_C(  -435.87), SIMDE_FLOAT32_C(   339.78),
        SIMDE_FLOAT32_C(  -669.60), SIMDE_FLOAT32_C(   509.82), SIMDE_FLOAT32_C(  -908.94), SIMDE_FLOAT32_C(  -774.79) },
       INT32_C(          25),
      UINT16_C(12594) },
    { { SIMDE_FLOAT32_C(  -703.28), SIMDE_FLOAT32_C(  -420.10), SIMDE_FLOAT32_C(  -425.79), SIMDE_FLOAT32_C(   779.02),
        SIMDE_FLOAT32_C(   420.16), SIMDE_FLOAT32_C(  -504.03), SIMDE_FLOAT32_C(  -845.78), SIMDE_FLOAT32_C(   425.59),
        SIMDE_FLOAT32_C(  -163.66), SIMDE_FLOAT32_C(    50.66), SIMDE_FLOAT32_C(   106.36), SIMDE_FLOAT32_C(   -80.46),
        SIMDE_FLOAT32_C(   263.47), SIMDE_FLOAT32_C(   330.95), SIMDE_FLOAT32_C(   981.36), SIMDE_FLOAT32_C(  -987.39) },
      { SIMDE_FLOAT32_C(  -215.33), SIMDE_FLOAT32_C(  -921.00), SIMDE_FLOAT32_C(  -921.96), SIMDE_FLOAT32_C(  -639.74),
        SIMDE_FLOAT32_C(   178.56), SIMDE_FLOAT32_C(   203.63), SIMDE_FLOAT32_C(  -629.67), SIMDE_FLOAT32_C(   824.75),
        SIMDE_FLOAT32_C(   383.48), SIMDE_FLOAT32_C(   -65.54), SIMDE_FLOAT32_C(   164.53), SIMDE_FLOAT32_C(   713.88),
        SIMDE_FLOAT32_C(  -555.72), SIMDE_FLOAT32_C(   255.59), SIMDE_FLOAT32_C(   939.10), SIMDE_FLOAT32_C(  -258.99) },
       INT32_C(          26),
      UINT16_C(36321) },
    { { SIMDE_FLOAT32_C(   835.48), SIMDE_FLOAT32_C(  -486.70), SIMDE_FLOAT32_C(  -479.98), SIMDE_FLOAT32_C(   255.65),
        SIMDE_FLOAT32_C(     9.27), SIMDE_FLOAT32_C(  -325.76), SIMDE_FLOAT32_C(  -318.76), SIMDE_FLOAT32_C(   845.61),
        SIMDE_FLOAT32_C(   724.90), SIMDE_FLOAT32_C(   787.60), SIMDE_FLOAT32_C(  -234.85), SIMDE_FLOAT32_C(   -11.62),
        SIMDE_FLOAT32_C(   118.55), SIMDE_FLOAT32_C(  -253.49), SIMDE_FLOAT32_C(     0.98), SIMDE_FLOAT32_C(   903.23) },
      { SIMDE_FLOAT32_C(  -174.49), SIMDE_FLOAT32_C(    79.03), SIMDE_FLOAT32_C(  -736.51), SIMDE_FLOAT32_C(  -995.93),
        SIMDE_FLOAT32_C(  -717.34), SIMDE_FLOAT32_C(  -366.18), SIMDE_FLOAT32_C(   828.82), SIMDE_FLOAT32_C(   666.15),
        SIMDE_FLOAT32_C(   568.29), SIMDE_FLOAT32_C(    -6.65), SIMDE_FLOAT32_C(   380.03), SIMDE_FLOAT32_C(  -987.43),
        SIMDE_FLOAT32_C(  -751.06), SIMDE_FLOAT32_C(   319.13), SIMDE_FLOAT32_C(  -246.43), SIMDE_FLOAT32_C(  -915.58) },
       INT32_C(          27),
      UINT16_C(    0) },
    { { SIMDE_FLOAT32_C(   832.43), SIMDE_FLOAT32_C(   273.60), SIMDE_FLOAT32_C(   340.07), SIMDE_FLOAT32_C(  -158.30),
        SIMDE_FLOAT32_C(   947.84), SIMDE_FLOAT32_C(  -978.69), SIMDE_FLOAT32_C(  -312.69), SIMDE_FLOAT32_C(   672.74),
        SIMDE_FLOAT32_C(   808.91), SIMDE_FLOAT32_C(   452.46), SIMDE_FLOAT32_C(  -338.88), SIMDE_FLOAT32_C(   -72.53),
        SIMDE_FLOAT32_C(  -801.03), SIMDE_FLOAT32_C(   662.10), SIMDE_FLOAT32_C(  -169.31), SIMDE_FLOAT32_C(    24.48) },
      { SIMDE_FLOAT32_C(  -258.87), SIMDE_FLOAT32_C(    94.18), SIMDE_FLOAT32_C(    28.55), SIMDE_FLOAT32_C(    23.79),
        SIMDE_FLOAT32_C(   728.01), SIMDE_FLOAT32_C(  -142.63), SIMDE_FLOAT32_C(  -310.06), SIMDE_FLOAT32_C(   296.29),
        SIMDE_FLOAT32_C(   850.72), SIMDE_FLOAT32_C(  -930.03), SIMDE_FLOAT32_C(   308.86), SIMDE_FLOAT32_C(  -900.34),
        SIMDE_FLOAT32_C(   389.10), SIMDE_FLOAT32_C(  -937.56), SIMDE_FLOAT32_C(  -815.92), SIMDE_FLOAT32_C(   221.53) },
       INT32_C(          28),
           UINT16_MAX },
    { { SIMDE_FLOAT32_C(   336.03), SIMDE_FLOAT32_C(   524.16), SIMDE_FLOAT32_C(  -936.77), SIMDE_FLOAT32_C(   283.87),
        SIMDE_FLOAT32_C(   545.47), SIMDE_FLOAT32_C(  -249.46), SIMDE_FLOAT32_C(   -43.38), SIMDE_FLOAT32_C(   354.38),
        SIMDE_FLOAT32_C(  -797.00), SIMDE_FLOAT32_C(   617.74), SIMDE_FLOAT32_C(  -718.15), SIMDE_FLOAT32_C(  -598.03),
        SIMDE_FLOAT32_C(   279.84), SIMDE_FLOAT32_C(   112.54), SIMDE_FLOAT32_C(   426.45), SIMDE_FLOAT32_C(  -979.03) },
      { SIMDE_FLOAT32_C(  -793.27), SIMDE_FLOAT32_C(  -545.00), SIMDE_FLOAT32_C(    44.77), SIMDE_FLOAT32_C(   934.73),
        SIMDE_FLOAT32_C(   312.37), SIMDE_FLOAT32_C(   734.71), SIMDE_FLOAT32_C(   231.03), SIMDE_FLOAT32_C(   163.09),
        SIMDE_FLOAT32_C(   804.68), SIMDE_FLOAT32_C(  -460.11), SIMDE_FLOAT32_C(   262.76), SIMDE_FLOAT32_C(   193.77),
        SIMDE_FLOAT32_C(  -397.67), SIMDE_FLOAT32_C(   446.84), SIMDE_FLOAT32_C(  -584.70), SIMDE_FLOAT32_C(   938.36) },
       INT32_C(          29),
      UINT16_C(21139) },
    { { SIMDE_FLOAT32_C(   -29.00), SIMDE_FLOAT32_C(  -521.48), SIMDE_FLOAT32_C(   222.23), SIMDE_FLOAT32_C(  -483.53),
        SIMDE_FLOAT32_C(   229.06), SIMDE_FLOAT32_C(  -821.16), SIMDE_FLOAT32_C(   870.85), SIMDE_FLOAT32_C(   432.06),
        SIMDE_FLOAT32_C(   796.58), SIMDE_FLOAT32_C(  -847.30), SIMDE_FLOAT32_C(   834.03), SIMDE_FLOAT32_C(    76.42),
        SIMDE_FLOAT32_C(   265.24), SIMDE_FLOAT32_C(   260.47), SIMDE_FLOAT32_C(    97.39), SIMDE_FLOAT32_C(   471.97) },
      { SIMDE_FLOAT32_C(   715.47), SIMDE_FLOAT32_C(  -857.84), SIMDE_FLOAT32_C(   406.70), SIMDE_FLOAT32_C(    27.84),
        SIMDE_FLOAT32_C(   876.87), SIMDE_FLOAT32_C(  -362.27), SIMDE_FLOAT32_C(  -809.06), SIMDE_FLOAT32_C(   681.54),
        SIMDE_FLOAT32_C(   177.62), SIMDE_FLOAT32_C(   453.69), SIMDE_FLOAT32_C(  -124.69), SIMDE_FLOAT32_C(   779.94),
        SIMDE_FLOAT32_C(   -99.47), SIMDE_FLOAT32_C(   290.61), SIMDE_FLOAT32_C(   718.30), SIMDE_FLOAT32_C(   871.53) },
       INT32_C(          30),
      UINT16_C( 5442) },
    { { SIMDE_FLOAT32_C(   769.14), SIMDE_FLOAT32_C(   -59.47), SIMDE_FLOAT32_C(  -612.01), SIMDE_FLOAT32_C(    -1.80),
        SIMDE_FLOAT32_C(   119.37), SIMDE_FLOAT32_C(  -741.16), SIMDE_FLOAT32_C(  -569.75), SIMDE_FLOAT32_C(   -84.05),
        SIMDE_FLOAT32_C(  -588.46), SIMDE_FLOAT32_C(  -735.72), SIMDE_FLOAT32_C(   992.37), SIMDE_FLOAT32_C(   676.78),
        SIMDE_FLOAT32_C(   524.75), SIMDE_FLOAT32_C(    89.76), SIMDE_FLOAT32_C(   148.75), SIMDE_FLOAT32_C(   240.22) },
      { SIMDE_FLOAT32_C(   231.92), SIMDE_FLOAT32_C(  -444.55), SIMDE_FLOAT32_C(  -731.94), SIMDE_FLOAT32_C(   108.79),
        SIMDE_FLOAT32_C(   193.18), SIMDE_FLOAT32_C(  -541.00), SIMDE_FLOAT32_C(  -209.67), SIMDE_FLOAT32_C(  -629.20),
        SIMDE_FLOAT32_C(   912.69), SIMDE_FLOAT32_C(   665.64), SIMDE_FLOAT32_C(  -849.26), SIMDE_FLOAT32_C(  -186.78),
        SIMDE_FLOAT32_C(   -43.74), SIMDE_FLOAT32_C(   869.04), SIMDE_FLOAT32_C(  -315.25), SIMDE_FLOAT32_C(  -274.61) },
       INT32_C(          31),
           UINT16_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512 a = simde_mm512_loadu_ps(test_vec[i].a);
    simde__m512 b = simde_mm512_loadu_ps(test_vec[i].b);
    simde__mmask16 r = simde_mm512_cmp_ps_mask(a, b, test_vec[i].imm8);
    simde_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
}

static int
test_simde_mm512_cmp_pd_mask (SIMDE_MUNIT_TEST_ARGS) {
  static const struct {
    const simde_float64 a[8];
    const simde_float64 b[8];
    const int imm8;
    const simde__mmask8 r;
  } test_vec[] = {
    { { SIMDE_FLOAT64_C(  -889.13), SIMDE_FLOAT64_C(   346.35), SIMDE_FLOAT64_C(  -842.69), SIMDE_FLOAT64_C(   879.16),
        SIMDE_FLOAT64_C(    37.28), SIMDE_FLOAT64_C(   607.79), SIMDE_FLOAT64_C(  -858.34), SIMDE_FLOAT64_C(  -122.77) },
      { SIMDE_FLOAT64_C(   597.22), SIMDE_FLOAT64_C(  -446.39), SIMDE_FLOAT64_C(  -495.07), SIMDE_FLOAT64_C(  -701.44),
        SIMDE_FLOAT64_C(   913.94), SIMDE_FLOAT64_C(   514.01), SIMDE_FLOAT64_C(  -970.90), SIMDE_FLOAT64_C(    91.95) },
       INT32_C(          18),
      UINT8_C(149) },
    { { SIMDE_FLOAT64_C(  -229.78), SIMDE_FLOAT64_C(   109.42), SIMDE_FLOAT64_C(   986.52), SIMDE_FLOAT64_C(   450.97),
        SIMDE_FLOAT64_C(  -621.15), SIMDE_FLOAT64_C(   366.22), SIMDE_FLOAT64_C(  -999.97), SIMDE_FLOAT64_C(  -551.44) },
      { SIMDE_FLOAT64_C(   -43.21), SIMDE_FLOAT64_C(  -236.56), SIMDE_FLOAT64_C(    73.66), SIMDE_FLOAT64_C(    21.45),
        SIMDE_FLOAT64_C(   426.81), SIMDE_FLOAT64_C(  -684.87), SIMDE_FLOAT64_C(  -547.62), SIMDE_FLOAT64_C(   194.20) },
       INT32_C(          30),
      UINT8_C( 46) },
    { { SIMDE_FLOAT64_C(   465.94), SIMDE_FLOAT64_C(  -899.85), SIMDE_FLOAT64_C(   236.88), SIMDE_FLOAT64_C(  -744.20),
        SIMDE_FLOAT64_C(   213.84), SIMDE_FLOAT64_C(    84.61), SIMDE_FLOAT64_C(    -4.00), SIMDE_FLOAT64_C(   791.14) },
      { SIMDE_FLOAT64_C(   691.24), SIMDE_FLOAT64_C(  -392.69), SIMDE_FLOAT64_C(    37.26), SIMDE_FLOAT64_C(   209.16),
        SIMDE_FLOAT64_C(  -604.04), SIMDE_FLOAT64_C(  -124.25), SIMDE_FLOAT64_C(  -288.59), SIMDE_FLOAT64_C(  -412.86) },
       INT32_C(          21),
      UINT8_C(244) },
    { { SIMDE_FLOAT64_C(  -618.86), SIMDE_FLOAT64_C(   797.13), SIMDE_FLOAT64_C(  -583.56), SIMDE_FLOAT64_C(    46.88),
        SIMDE_FLOAT64_C(   -89.41), SIMDE_FLOAT64_C(  -683.29), SIMDE_FLOAT64_C(    20.57), SIMDE_FLOAT64_C(  -213.31) },
      { SIMDE_FLOAT64_C(   887.10), SIMDE_FLOAT64_C(  -441.79), SIMDE_FLOAT64_C(   836.33), SIMDE_FLOAT64_C(   135.59),
        SIMDE_FLOAT64_C(   918.70), SIMDE_FLOAT64_C(   512.23), SIMDE_FLOAT64_C(  -895.63), SIMDE_FLOAT64_C(  -900.96) },
       INT32_C(          31),
         UINT8_MAX },
    { { SIMDE_FLOAT64_C(  -989.35), SIMDE_FLOAT64_C(   -86.98), SIMDE_FLOAT64_C(   193.68), SIMDE_FLOAT64_C(  -742.71),
        SIMDE_FLOAT64_C(  -727.59), SIMDE_FLOAT64_C(  -646.86), SIMDE_FLOAT64_C(   183.87), SIMDE_FLOAT64_C(   287.33) },
      { SIMDE_FLOAT64_C(  -774.81), SIMDE_FLOAT64_C(  -242.40), SIMDE_FLOAT64_C(    53.99), SIMDE_FLOAT64_C(  -593.99),
        SIMDE_FLOAT64_C(   779.72), SIMDE_FLOAT64_C(   806.29), SIMDE_FLOAT64_C(  -734.86), SIMDE_FLOAT64_C(  -839.78) },
       INT32_C(           6),
      UINT8_C(198) },
    { { SIMDE_FLOAT64_C(  -467.13), SIMDE_FLOAT64_C(   942.12), SIMDE_FLOAT64_C(   248.01), SIMDE_FLOAT64_C(   325.07),
        SIMDE_FLOAT64_C(  -486.56), SIMDE_FLOAT64_C(   428.42), SIMDE_FLOAT64_C(   503.39), SIMDE_FLOAT64_C(   520.75) },
      { SIMDE_FLOAT64_C(   191.14), SIMDE_FLOAT64_C(   441.43), SIMDE_FLOAT64_C(  -872.87), SIMDE_FLOAT64_C(  -283.89),
        SIMDE_FLOAT64_C(   651.45), SIMDE_FLOAT64_C(   971.81), SIMDE_FLOAT64_C(  -736.72), SIMDE_FLOAT64_C(   -71.12) },
       INT32_C(          17),
      UINT8_C( 49) },
    { { SIMDE_FLOAT64_C(   768.97), SIMDE_FLOAT64_C(    83.87), SIMDE_FLOAT64_C(  -412.86), SIMDE_FLOAT64_C(   997.28),
        SIMDE_FLOAT64_C(  -659.21), SIMDE_FLOAT64_C(   650.14), SIMDE_FLOAT64_C(   927.00), SIMDE_FLOAT64_C(    40.06) },
      { SIMDE_FLOAT64_C(  -783.94), SIMDE_FLOAT64_C(   289.86), SIMDE_FLOAT64_C(  -325.98), SIMDE_FLOAT64_C(  -693.23),
        SIMDE_FLOAT64_C(   823.83), SIMDE_FLOAT64_C(    81.84), SIMDE_FLOAT64_C(  -557.12), SIMDE_FLOAT64_C(   458.20) },
       INT32_C(          28),
         UINT8_MAX },
    { { SIMDE_FLOAT64_C(   728.01), SIMDE_FLOAT64_C(  -387.72), SIMDE_FLOAT64_C(  -341.65), SIMDE_FLOAT64_C(   -84.21),
        SIMDE_FLOAT64_C(   640.32), SIMDE_FLOAT64_C(  -112.91), SIMDE_FLOAT64_C(   308.09), SIMDE_FLOAT64_C(    20.16) },
      { SIMDE_FLOAT64_C(   745.10), SIMDE_FLOAT64_C(   919.13), SIMDE_FLOAT64_C(  -195.91), SIMDE_FLOAT64_C(  -612.27),
        SIMDE_FLOAT64_C(  -399.85), SIMDE_FLOAT64_C(  -354.18), SIMDE_FLOAT64_C(  -752.13), SIMDE_FLOAT64_C(   868.12) },
       INT32_C(          29),
      UINT8_C(120) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512d a = simde_mm512_loadu_pd(test_vec[i].a);
    simde__m512d b = simde_mm512_loadu_pd(test_vec[i].b);
    simde__mmask8 r = simde_mm512_cmp_pd_mask(a, b, test_vec[i].imm8);
    simde_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
}

#else

/* To avoid a warning about expr < 0 always evaluating to false
 * (-Wtype-limits) because there are no functions to test. */

static int
test_simde_dummy (SIMDE_MUNIT_TEST_ARGS) {
  return 0;
}

#endif /* !defined(SIMDE_NATIVE_ALIASES_TESTING */

SIMDE_TEST_FUNC_LIST_BEGIN
  #if !defined(SIMDE_NATIVE_ALIASES_TESTING)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_epi8_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_epi16_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_epi32_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm_cmp_epi64_mask)

    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_epi8_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_epi16_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_epi32_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm256_cmp_epi64_mask)

    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_epi8_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_epi16_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_epi32_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_epi64_mask)

    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_ps_mask)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_cmp_pd_mask)
  #else
    SIMDE_TEST_FUNC_LIST_ENTRY(dummy)
  #endif
SIMDE_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
