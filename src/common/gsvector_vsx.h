// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

// GSVector for POWER8+ VSX. GSVector4i and GSVector4 are implemented with VSX, matching the NEON implementation's
// semantics. The 64-bit GSVector2i and GSVector2 classes use the scalar implementation.

#pragma once

#include "common/intrin.h"
#include "common/types.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#define GSVECTOR_HAS_FAST_INT_SHUFFLE8 1
#define GSVECTOR_HAS_SRLV 1

using gsv_s8 = __vector signed char;
using gsv_u8 = __vector unsigned char;
using gsv_s16 = __vector signed short;
using gsv_u16 = __vector unsigned short;
using gsv_s32 = __vector signed int;
using gsv_u32 = __vector unsigned int;
using gsv_s64 = __vector signed long long;
using gsv_u64 = __vector unsigned long long;
using gsv_f32 = __vector float;
using gsv_f64 = __vector double;
using gsv_bool8 = __vector __bool char;

class GSVector2;
class GSVector2i;
class GSVector4;
class GSVector4i;

#define SSATURATE8(expr) static_cast<s8>(std::clamp<decltype(expr)>(expr, -128, 127))
#define USATURATE8(expr) static_cast<u8>(std::clamp<decltype(expr)>(expr, 0, 255))
#define SSATURATE16(expr) static_cast<s16>(std::clamp<decltype(expr)>(expr, -32768, 32767))
#define USATURATE16(expr) static_cast<u16>(std::clamp<decltype(expr)>(expr, 0, 65535))

#define ALL_LANES_8(expr)                                                                                              \
  GSVector2i ret;                                                                                                      \
  for (size_t i = 0; i < 8; i++)                                                                                       \
    expr;                                                                                                              \
  return ret;
#define ALL_LANES_16(expr)                                                                                             \
  GSVector2i ret;                                                                                                      \
  for (size_t i = 0; i < 4; i++)                                                                                       \
    expr;                                                                                                              \
  return ret;
#define ALL_LANES_32(expr)                                                                                             \
  GSVector2i ret;                                                                                                      \
  for (size_t i = 0; i < 2; i++)                                                                                       \
    expr;                                                                                                              \
  return ret;

class alignas(16) GSVector2i
{
  struct cxpr_init_tag
  {
  };
  static constexpr cxpr_init_tag cxpr_init{};

  constexpr GSVector2i(cxpr_init_tag, s32 x, s32 y) : S32{x, y} {}

  constexpr GSVector2i(cxpr_init_tag, s16 s0, s16 s1, s16 s2, s16 s3) : S16{s0, s1, s2, s3} {}

  constexpr GSVector2i(cxpr_init_tag, s8 b0, s8 b1, s8 b2, s8 b3, s8 b4, s8 b5, s8 b6, s8 b7)
    : S8{b0, b1, b2, b3, b4, b5, b6, b7}
  {
  }

public:
  union
  {
    struct
    {
      s32 x, y;
    };
    struct
    {
      s32 r, g;
    };
    float F32[2];
    s8 S8[8];
    s16 S16[4];
    s32 S32[2];
    s64 S64[1];
    u8 U8[8];
    u16 U16[4];
    u32 U32[2];
    u64 U64[1];
  };

  GSVector2i() = default;

  ALWAYS_INLINE constexpr static GSVector2i cxpr(s32 x, s32 y) { return GSVector2i(cxpr_init, x, y); }

  ALWAYS_INLINE constexpr static GSVector2i cxpr(s32 x) { return GSVector2i(cxpr_init, x, x); }

  ALWAYS_INLINE constexpr static GSVector2i cxpr16(s16 x) { return GSVector2i(cxpr_init, x, x, x, x); }

  ALWAYS_INLINE constexpr static GSVector2i cxpr16(s16 s0, s16 s1, s16 s2, s16 s3)
  {
    return GSVector2i(cxpr_init, s0, s1, s2, s3);
  }

  ALWAYS_INLINE constexpr static GSVector2i cxpr8(s8 b0, s8 b1, s8 b2, s8 b3, s8 b4, s8 b5, s8 b6, s8 b7)
  {
    return GSVector2i(cxpr_init, b0, b1, b2, b3, b4, b5, b6, b7);
  }

  ALWAYS_INLINE GSVector2i(s32 x, s32 y)
  {
    this->x = x;
    this->y = y;
  }

  ALWAYS_INLINE GSVector2i(s16 s0, s16 s1, s16 s2, s16 s3)
  {
    S16[0] = s0;
    S16[1] = s1;
    S16[2] = s2;
    S16[3] = s3;
  }

  ALWAYS_INLINE constexpr GSVector2i(s8 b0, s8 b1, s8 b2, s8 b3, s8 b4, s8 b5, s8 b6, s8 b7)
    : S8{b0, b1, b2, b3, b4, b5, b6, b7}
  {
  }

  ALWAYS_INLINE GSVector2i(const GSVector2i& v) { std::memcpy(S32, v.S32, sizeof(S32)); }

  ALWAYS_INLINE explicit GSVector2i(s32 i) { *this = i; }

  ALWAYS_INLINE explicit GSVector2i(const GSVector2& v);

  ALWAYS_INLINE static GSVector2i cast(const GSVector2& v);

  ALWAYS_INLINE GSVector2i& operator=(const GSVector2i& v)
  {
    std::memcpy(S32, v.S32, sizeof(S32));
    return *this;
  }
  ALWAYS_INLINE GSVector2i& operator=(s32 i)
  {
    x = i;
    y = i;
    return *this;
  }

  ALWAYS_INLINE GSVector2i sat_s8(const GSVector2i& min, const GSVector2i& max) const
  {
    return max_s8(min).min_s8(max);
  }
  ALWAYS_INLINE GSVector2i sat_s16(const GSVector2i& min, const GSVector2i& max) const
  {
    return max_s16(min).min_s16(max);
  }
  ALWAYS_INLINE GSVector2i sat_s32(const GSVector2i& min, const GSVector2i& max) const
  {
    return max_s32(min).min_s32(max);
  }

  ALWAYS_INLINE GSVector2i sat_u8(const GSVector2i& min, const GSVector2i& max) const
  {
    return max_u8(min).min_u8(max);
  }
  ALWAYS_INLINE GSVector2i sat_u16(const GSVector2i& min, const GSVector2i& max) const
  {
    return max_u16(min).min_u16(max);
  }
  ALWAYS_INLINE GSVector2i sat_u32(const GSVector2i& min, const GSVector2i& max) const
  {
    return max_u32(min).min_u32(max);
  }

  GSVector2i min_s8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = std::min(S8[i], v.S8[i])); }
  GSVector2i max_s8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = std::max(S8[i], v.S8[i])); }
  GSVector2i min_s16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = std::min(S16[i], v.S16[i])); }
  GSVector2i max_s16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = std::max(S16[i], v.S16[i])); }
  GSVector2i min_s32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = std::min(S32[i], v.S32[i])); }
  GSVector2i max_s32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = std::max(S32[i], v.S32[i])); }

  GSVector2i min_u8(const GSVector2i& v) const { ALL_LANES_8(ret.U8[i] = std::min(U8[i], v.U8[i])); }
  GSVector2i max_u8(const GSVector2i& v) const { ALL_LANES_8(ret.U8[i] = std::max(U8[i], v.U8[i])); }
  GSVector2i min_u16(const GSVector2i& v) const { ALL_LANES_16(ret.U16[i] = std::min(U16[i], v.U16[i])); }
  GSVector2i max_u16(const GSVector2i& v) const { ALL_LANES_16(ret.U16[i] = std::max(U16[i], v.U16[i])); }
  GSVector2i min_u32(const GSVector2i& v) const { ALL_LANES_32(ret.U32[i] = std::min(U32[i], v.U32[i])); }
  GSVector2i max_u32(const GSVector2i& v) const { ALL_LANES_32(ret.U32[i] = std::max(U32[i], v.U32[i])); }

  s32 addv_s32() const { return (S32[0] + S32[1]); }

  u8 minv_u8() const
  {
    return std::min(
      U8[0],
      std::min(U8[1], std::min(U8[2], std::min(U8[3], std::min(U8[4], std::min(U8[5], std::min(U8[6], U8[7])))))));
  }

  u16 maxv_u8() const
  {
    return std::max(
      U8[0],
      std::max(U8[1], std::max(U8[2], std::max(U8[3], std::max(U8[4], std::max(U8[5], std::max(U8[6], U8[7])))))));
  }

  u16 minv_u16() const { return std::min(U16[0], std::min(U16[1], std::min(U16[2], U16[3]))); }

  u16 maxv_u16() const { return std::max(U16[0], std::max(U16[1], std::max(U16[2], U16[3]))); }

  s32 minv_s32() const { return std::min(x, y); }

  u32 minv_u32() const { return std::min(U32[0], U32[1]); }

  s32 maxv_s32() const { return std::max(x, y); }

  u32 maxv_u32() const { return std::max(U32[0], U32[1]); }

  ALWAYS_INLINE GSVector2i clamp8() const { return pu16().upl8(); }

  GSVector2i blend8(const GSVector2i& v, const GSVector2i& mask) const
  {
    GSVector2i ret;
    for (size_t i = 0; i < 8; i++)
      ret.U8[i] = (mask.U8[i] & 0x80) ? v.U8[i] : U8[i];
    return ret;
  }

  template<s32 mask>
  GSVector2i blend16(const GSVector2i& v) const
  {
    GSVector2i ret;
    for (size_t i = 0; i < 4; i++)
      ret.U16[i] = ((mask & (1 << i)) != 0) ? v.U16[i] : U16[i];
    return ret;
  }

  template<s32 mask>
  GSVector2i blend32(const GSVector2i& v) const
  {
    GSVector2i ret;
    for (size_t i = 0; i < 2; i++)
      ret.U32[i] = ((mask & (1 << i)) != 0) ? v.U32[i] : U32[i];
    return ret;
  }

  GSVector2i blend(const GSVector2i& v, const GSVector2i& mask) const
  {
    GSVector2i ret;
    ret.U64[0] = (v.U64[0] & mask.U64[0]);
    return ret;
  }

  GSVector2i shuffle8(const GSVector2i& mask) const
  {
    ALL_LANES_8(ret.S8[i] = (mask.S8[i] & 0x80) ? 0 : (S8[mask.S8[i] & 0xf]));
  }

  GSVector2i ps16() const { ALL_LANES_8(ret.S8[i] = SSATURATE8(S16[(i < 4) ? i : (i - 4)])); }
  GSVector2i pu16() const { ALL_LANES_8(ret.U8[i] = USATURATE8(U16[(i < 4) ? i : (i - 4)])); }
  GSVector2i ps32() const { ALL_LANES_16(ret.S16[i] = SSATURATE16(S32[(i < 2) ? i : (i - 2)])); }
  GSVector2i pu32() const { ALL_LANES_16(ret.U16[i] = USATURATE16(U32[(i < 2) ? i : (i - 2)])); }

  GSVector2i upl8() const { return GSVector2i(S8[0], 0, S8[1], 0, S8[2], 0, S8[3], 0); }

  GSVector2i upl16() const { return GSVector2i(S16[0], 0, S16[1], 0); }

  GSVector2i upl32() const { return GSVector2i(S32[0], 0); }

  GSVector2i s8to16() const { ALL_LANES_16(ret.S16[i] = S8[i]); }
  GSVector2i s8to32() const { ALL_LANES_32(ret.S32[i] = S8[i]); }
  GSVector2i u8to16() const { ALL_LANES_16(ret.U16[i] = U8[i]); }
  GSVector2i u8to32() const { ALL_LANES_32(ret.U32[i] = U8[i]); }
  GSVector2i u16to32() const { ALL_LANES_32(ret.U32[i] = U16[i]); }
  GSVector2i s16to32() const { ALL_LANES_32(ret.S32[i] = S16[i]); }

  template<s32 v>
  GSVector2i srl() const
  {
    GSVector2i ret = {};
    if constexpr (v < 8)
    {
      for (s32 i = 0; i < (8 - v); i++)
        ret.U8[i] = U8[v + i];
    }
    return ret;
  }

  template<s32 v>
  GSVector2i sll() const
  {
    GSVector2i ret = {};
    if constexpr (v < 8)
    {
      for (s32 i = 0; i < (8 - v); i++)
        ret.U8[v + i] = U8[i];
    }
    return ret;
  }

  template<s32 v>
  GSVector2i sll16() const
  {
    ALL_LANES_16(ret.U16[i] = U16[i] << v);
  }

  GSVector2i sll16(s32 v) const { ALL_LANES_16(ret.U16[i] = U16[i] << v); }

  GSVector2i sllv16(const GSVector2i& v) const { ALL_LANES_16(ret.U16[i] = U16[i] << v.U16[i]); }

  template<s32 v>
  GSVector2i srl16() const
  {
    ALL_LANES_16(ret.U16[i] = U16[i] >> v);
  }

  GSVector2i srl16(s32 v) const { ALL_LANES_16(ret.U16[i] = U16[i] >> v); }

  GSVector2i srlv16(const GSVector2i& v) const { ALL_LANES_16(ret.U16[i] = U16[i] >> v.U16[i]); }

  template<s32 v>
  GSVector2i sra16() const
  {
    ALL_LANES_16(ret.S16[i] = S16[i] >> v);
  }

  GSVector2i sra16(s32 v) const { ALL_LANES_16(ret.S16[i] = S16[i] >> v); }

  GSVector2i srav16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = S16[i] >> v.S16[i]); }

  template<s32 v>
  GSVector2i sll32() const
  {
    ALL_LANES_32(ret.U32[i] = U32[i] << v);
  }

  GSVector2i sll32(s32 v) const { ALL_LANES_32(ret.U32[i] = U32[i] << v); }

  GSVector2i sllv32(const GSVector2i& v) const { ALL_LANES_32(ret.U32[i] = U32[i] << v.U32[i]); }

  template<s32 v>
  GSVector2i srl32() const
  {
    ALL_LANES_32(ret.U32[i] = U32[i] >> v);
  }

  GSVector2i srl32(s32 v) const { ALL_LANES_32(ret.U32[i] = U32[i] >> v); }

  GSVector2i srlv32(const GSVector2i& v) const { ALL_LANES_32(ret.U32[i] = U32[i] >> v.U32[i]); }

  template<s32 v>
  GSVector2i sra32() const
  {
    ALL_LANES_32(ret.S32[i] = S32[i] >> v);
  }

  GSVector2i sra32(s32 v) const { ALL_LANES_32(ret.S32[i] = S32[i] >> v); }

  GSVector2i srav32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = S32[i] >> v.S32[i]); }

  GSVector2i add8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = S8[i] + v.S8[i]); }

  GSVector2i add16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = S16[i] + v.S16[i]); }

  GSVector2i add32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = S32[i] + v.S32[i]); }

  GSVector2i adds8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = SSATURATE8(S8[i] + v.S8[i])); }

  GSVector2i adds16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = SSATURATE16(S16[i] + v.S16[i])); }

  GSVector2i addus8(const GSVector2i& v) const { ALL_LANES_8(ret.U8[i] = USATURATE8(U8[i] + v.U8[i])); }

  GSVector2i addus16(const GSVector2i& v) const { ALL_LANES_16(ret.U16[i] = USATURATE16(U16[i] + v.U16[i])); }

  GSVector2i sub8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = S8[i] - v.S8[i]); }

  GSVector2i sub16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = S16[i] - v.S16[i]); }

  GSVector2i sub32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = S32[i] - v.S32[i]); }

  GSVector2i subs8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = SSATURATE8(S8[i] - v.S8[i])); }

  GSVector2i subs16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = SSATURATE16(S16[i] - v.S16[i])); }

  GSVector2i subus8(const GSVector2i& v) const { ALL_LANES_8(ret.U8[i] = USATURATE8(U8[i] - v.U8[i])); }

  GSVector2i subus16(const GSVector2i& v) const { ALL_LANES_16(ret.U16[i] = USATURATE16(U16[i] - v.U16[i])); }

  GSVector2i avg8(const GSVector2i& v) const { ALL_LANES_8(ret.U8[i] = (U8[i] + v.U8[i]) >> 1); }

  GSVector2i avg16(const GSVector2i& v) const { ALL_LANES_16(ret.U16[i] = (U16[i] + v.U16[i]) >> 1); }

  GSVector2i mul16l(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = S16[i] * v.S16[i]); }

  GSVector2i mul32l(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = S32[i] * v.S32[i]); }

  ALWAYS_INLINE bool eq(const GSVector2i& v) const { return (std::memcmp(S32, v.S32, sizeof(S32))) == 0; }

  GSVector2i eq8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = (S8[i] == v.S8[i]) ? -1 : 0); }
  GSVector2i eq16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = (S16[i] == v.S16[i]) ? -1 : 0); }
  GSVector2i eq32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = (S32[i] == v.S32[i]) ? -1 : 0); }

  GSVector2i neq8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = (S8[i] != v.S8[i]) ? -1 : 0); }
  GSVector2i neq16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = (S16[i] != v.S16[i]) ? -1 : 0); }
  GSVector2i neq32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = (S32[i] != v.S32[i]) ? -1 : 0); }

  GSVector2i gt8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = (S8[i] > v.S8[i]) ? -1 : 0); }
  GSVector2i gt16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = (S16[i] > v.S16[i]) ? -1 : 0); }
  GSVector2i gt32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = (S32[i] > v.S32[i]) ? -1 : 0); }

  GSVector2i ge8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = (S8[i] >= v.S8[i]) ? -1 : 0); }
  GSVector2i ge16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = (S16[i] >= v.S16[i]) ? -1 : 0); }
  GSVector2i ge32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = (S32[i] >= v.S32[i]) ? -1 : 0); }

  GSVector2i lt8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = (S8[i] < v.S8[i]) ? -1 : 0); }
  GSVector2i lt16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = (S16[i] < v.S16[i]) ? -1 : 0); }
  GSVector2i lt32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = (S32[i] < v.S32[i]) ? -1 : 0); }

  GSVector2i le8(const GSVector2i& v) const { ALL_LANES_8(ret.S8[i] = (S8[i] <= v.S8[i]) ? -1 : 0); }
  GSVector2i le16(const GSVector2i& v) const { ALL_LANES_16(ret.S16[i] = (S16[i] <= v.S16[i]) ? -1 : 0); }
  GSVector2i le32(const GSVector2i& v) const { ALL_LANES_32(ret.S32[i] = (S32[i] <= v.S32[i]) ? -1 : 0); }

  ALWAYS_INLINE GSVector2i andnot(const GSVector2i& v) const
  {
    GSVector2i ret;
    ret.U64[0] = (~v.U64[0]) & U64[0];
    return ret;
  }

  s32 mask() const
  {
    return static_cast<s32>((static_cast<u32>(U8[0] >> 7) << 0) | (static_cast<u32>(U8[1] >> 7) << 1) |
                            (static_cast<u32>(U8[2] >> 7) << 2) | (static_cast<u32>(U8[3] >> 7) << 3) |
                            (static_cast<u32>(U8[4] >> 7) << 4) | (static_cast<u32>(U8[5] >> 7) << 5) |
                            (static_cast<u32>(U8[6] >> 7) << 6) | (static_cast<u32>(U8[7] >> 7) << 7));
  }

  ALWAYS_INLINE bool alltrue() const { return (U64[0] == 0xFFFFFFFFFFFFFFFFULL); }
  ALWAYS_INLINE bool anytrue() const { return (U64[0] != 0); }
  ALWAYS_INLINE bool allfalse() const { return (U64[0] == 0); }

  template<s32 i>
  ALWAYS_INLINE GSVector2i insert8(s32 a) const
  {
    GSVector2i ret = *this;
    ret.S8[i] = static_cast<s8>(a);
    return ret;
  }

  template<s32 i>
  ALWAYS_INLINE s32 extract8() const
  {
    return S8[i];
  }

  template<s32 i>
  ALWAYS_INLINE GSVector2i insert16(s32 a) const
  {
    GSVector2i ret = *this;
    ret.S16[i] = static_cast<s16>(a);
    return ret;
  }

  template<s32 i>
  ALWAYS_INLINE s32 extract16() const
  {
    return S16[i];
  }

  template<s32 i>
  ALWAYS_INLINE GSVector2i insert32(s32 a) const
  {
    GSVector2i ret = *this;
    ret.S32[i] = a;
    return ret;
  }

  template<s32 i>
  ALWAYS_INLINE s32 extract32() const
  {
    return S32[i];
  }

  ALWAYS_INLINE static GSVector2i load32(const void* p)
  {
    GSVector2i ret;
    std::memcpy(&ret.x, p, sizeof(s32));
    ret.y = 0;
    return ret;
  }

  ALWAYS_INLINE static GSVector2i set32(s32 v) { return GSVector2i(v, 0); }

  template<bool aligned>
  ALWAYS_INLINE static GSVector2i load(const void* p)
  {
    GSVector2i ret;
    std::memcpy(ret.S32, p, sizeof(ret.S32));
    return ret;
  }

  template<bool aligned>
  ALWAYS_INLINE static void store(void* p, const GSVector2i& v)
  {
    std::memcpy(p, v.S32, sizeof(S32));
  }

  ALWAYS_INLINE static void store32(void* p, const GSVector2i& v) { std::memcpy(p, &v.x, sizeof(s32)); }

  ALWAYS_INLINE void operator&=(const GSVector2i& v) { U64[0] &= v.U64[0]; }
  ALWAYS_INLINE void operator|=(const GSVector2i& v) { U64[0] |= v.U64[0]; }
  ALWAYS_INLINE void operator^=(const GSVector2i& v) { U64[0] ^= v.U64[0]; }

  ALWAYS_INLINE friend GSVector2i operator&(const GSVector2i& v1, const GSVector2i& v2)
  {
    GSVector2i ret;
    ret.U64[0] = v1.U64[0] & v2.U64[0];
    return ret;
  }

  ALWAYS_INLINE friend GSVector2i operator|(const GSVector2i& v1, const GSVector2i& v2)
  {
    GSVector2i ret;
    ret.U64[0] = v1.U64[0] | v2.U64[0];
    return ret;
  }

  ALWAYS_INLINE friend GSVector2i operator^(const GSVector2i& v1, const GSVector2i& v2)
  {
    GSVector2i ret;
    ret.U64[0] = v1.U64[0] ^ v2.U64[0];
    return ret;
  }

  ALWAYS_INLINE friend GSVector2i operator&(const GSVector2i& v, s32 i) { return v & GSVector2i(i); }

  ALWAYS_INLINE friend GSVector2i operator|(const GSVector2i& v, s32 i) { return v | GSVector2i(i); }

  ALWAYS_INLINE friend GSVector2i operator^(const GSVector2i& v, s32 i) { return v ^ GSVector2i(i); }

  ALWAYS_INLINE friend GSVector2i operator~(const GSVector2i& v) { return v ^ v.eq32(v); }

  ALWAYS_INLINE static constexpr GSVector2i zero() { return GSVector2i::cxpr(0, 0); }

  ALWAYS_INLINE GSVector2i xy() const { return *this; }
  ALWAYS_INLINE GSVector2i xx() const { return GSVector2i(x, x); }
  ALWAYS_INLINE GSVector2i yx() const { return GSVector2i(y, x); }
  ALWAYS_INLINE GSVector2i yy() const { return GSVector2i(y, y); }
};

class alignas(16) GSVector2
{
  struct cxpr_init_tag
  {
  };
  static constexpr cxpr_init_tag cxpr_init{};

  constexpr GSVector2(cxpr_init_tag, float x, float y) : F32{x, y} {}

  constexpr GSVector2(cxpr_init_tag, int x, int y) : I32{x, y} {}

public:
  union
  {
    struct
    {
      float x, y;
    };
    struct
    {
      float r, g;
    };
    float F32[2];
    double F64[2];
    s8 I8[16];
    s16 I16[8];
    s32 I32[4];
    s64 I64[2];
    u8 U8[16];
    u16 U16[8];
    u32 U32[4];
    u64 U64[2];
  };

  GSVector2() = default;

  constexpr static GSVector2 cxpr(float x, float y) { return GSVector2(cxpr_init, x, y); }

  constexpr static GSVector2 cxpr(float x) { return GSVector2(cxpr_init, x, x); }

  constexpr static GSVector2 cxpr(int x, int y) { return GSVector2(cxpr_init, x, y); }

  constexpr static GSVector2 cxpr(int x) { return GSVector2(cxpr_init, x, x); }

  ALWAYS_INLINE GSVector2(float x, float y)
  {
    this->x = x;
    this->y = y;
  }

  ALWAYS_INLINE GSVector2(int x, int y)
  {
    this->x = static_cast<float>(x);
    this->y = static_cast<float>(y);
  }

  ALWAYS_INLINE explicit GSVector2(float f) { x = y = f; }

  ALWAYS_INLINE explicit GSVector2(int i) { x = y = static_cast<float>(i); }

  ALWAYS_INLINE explicit GSVector2(const GSVector2i& v);

  ALWAYS_INLINE static GSVector2 cast(const GSVector2i& v);

  ALWAYS_INLINE GSVector2& operator=(float f)
  {
    x = y = f;
    return *this;
  }

  GSVector2 abs() const { return GSVector2(std::fabs(x), std::fabs(y)); }

  GSVector2 neg() const { return GSVector2(-x, -y); }

  GSVector2 floor() const { return GSVector2(std::floor(x), std::floor(y)); }

  GSVector2 ceil() const { return GSVector2(std::ceil(x), std::ceil(y)); }

  GSVector2 sat(const GSVector2& min, const GSVector2& max) const
  {
    return GSVector2(std::clamp(x, min.x, max.x), std::clamp(y, min.y, max.y));
  }

  GSVector2 sat(const float scale = 255) const { return sat(zero(), GSVector2(scale)); }

  GSVector2 clamp(const float scale = 255) const { return min(GSVector2(scale)); }

  GSVector2 min(const GSVector2& v) const { return GSVector2(std::min(x, v.x), std::min(y, v.y)); }

  GSVector2 max(const GSVector2& v) const { return GSVector2(std::max(x, v.x), std::max(y, v.y)); }

  template<int mask>
  GSVector2 blend32(const GSVector2& v) const
  {
    return GSVector2((mask & 1) ? v.x : x, (mask & 2) ? v.y : y);
  }

  ALWAYS_INLINE GSVector2 blend32(const GSVector2& v, const GSVector2& mask) const
  {
    return GSVector2((mask.U32[0] & 0x80000000u) ? v.x : x, (mask.U32[1] & 0x80000000u) ? v.y : y);
  }

  ALWAYS_INLINE GSVector2 andnot(const GSVector2& v) const
  {
    GSVector2 ret;
    ret.U32[0] = ((~v.U32[0]) & U32[0]);
    ret.U32[1] = ((~v.U32[1]) & U32[1]);
    return ret;
  }

  ALWAYS_INLINE int mask() const { return (U32[0] >> 31) | ((U32[1] >> 30) & 2); }

  ALWAYS_INLINE bool alltrue() const { return (U64[0] == 0xFFFFFFFFFFFFFFFFULL); }
  ALWAYS_INLINE bool anytrue() const { return (U64[0] != 0); }
  ALWAYS_INLINE bool allfalse() const { return (U64[0] == 0); }

  ALWAYS_INLINE GSVector2 replace_nan(const GSVector2& v) const { return v.blend32(*this, *this == *this); }

  template<int src, int dst>
  ALWAYS_INLINE GSVector2 insert32(const GSVector2& v) const
  {
    GSVector2 ret = *this;
    ret.F32[dst] = v.F32[src];
    return ret;
  }

  template<int i>
  ALWAYS_INLINE int extract32() const
  {
    return I32[i];
  }

  ALWAYS_INLINE float dot(const GSVector2& v) const { return (x * v.x + y * v.y); }

  ALWAYS_INLINE static constexpr GSVector2 zero() { return GSVector2::cxpr(0.0f, 0.0f); }

  ALWAYS_INLINE static constexpr GSVector2 xffffffff()
  {
    GSVector2 ret = zero();
    ret.U64[0] = ~ret.U64[0];
    return ret;
  }

  template<bool aligned>
  ALWAYS_INLINE static GSVector2 load(const void* p)
  {
    GSVector2 ret;
    std::memcpy(ret.F32, p, sizeof(F32));
    return ret;
  }

  template<bool aligned>
  ALWAYS_INLINE static void store(void* p, const GSVector2& v)
  {
    std::memcpy(p, &v.F32, sizeof(F32));
  }

  ALWAYS_INLINE GSVector2 operator-() const { return neg(); }

  void operator+=(const GSVector2& v_)
  {
    x = x + v_.x;
    y = y + v_.y;
  }
  void operator-=(const GSVector2& v_)
  {
    x = x - v_.x;
    y = y - v_.y;
  }
  void operator*=(const GSVector2& v_)
  {
    x = x * v_.x;
    y = y * v_.y;
  }
  void operator/=(const GSVector2& v_)
  {
    x = x / v_.x;
    y = y / v_.y;
  }

  void operator+=(const float v_)
  {
    x = x + v_;
    y = y + v_;
  }
  void operator-=(const float v_)
  {
    x = x - v_;
    y = y - v_;
  }
  void operator*=(const float v_)
  {
    x = x * v_;
    y = y * v_;
  }
  void operator/=(const float v_)
  {
    x = x / v_;
    y = y / v_;
  }

  void operator&=(const GSVector2& v_) { U64[0] &= v_.U64[0]; }
  void operator|=(const GSVector2& v_) { U64[0] |= v_.U64[0]; }
  void operator^=(const GSVector2& v_) { U64[0] ^= v_.U64[0]; }

  friend GSVector2 operator+(const GSVector2& v1, const GSVector2& v2) { return GSVector2(v1.x + v2.x, v1.y + v2.y); }

  friend GSVector2 operator-(const GSVector2& v1, const GSVector2& v2) { return GSVector2(v1.x - v2.x, v1.y - v2.y); }

  friend GSVector2 operator*(const GSVector2& v1, const GSVector2& v2) { return GSVector2(v1.x * v2.x, v1.y * v2.y); }

  friend GSVector2 operator/(const GSVector2& v1, const GSVector2& v2) { return GSVector2(v1.x / v2.x, v1.y / v2.y); }

  friend GSVector2 operator+(const GSVector2& v, float f) { return GSVector2(v.x + f, v.y + f); }

  friend GSVector2 operator-(const GSVector2& v, float f) { return GSVector2(v.x - f, v.y - f); }

  friend GSVector2 operator*(const GSVector2& v, float f) { return GSVector2(v.x * f, v.y * f); }

  friend GSVector2 operator/(const GSVector2& v, float f) { return GSVector2(v.x / f, v.y / f); }

  friend GSVector2 operator&(const GSVector2& v1, const GSVector2& v2)
  {
    GSVector2 ret;
    ret.U64[0] = v1.U64[0] & v2.U64[0];
    return ret;
  }

  ALWAYS_INLINE friend GSVector2 operator|(const GSVector2& v1, const GSVector2& v2)
  {
    GSVector2 ret;
    ret.U64[0] = v1.U64[0] | v2.U64[0];
    return ret;
  }

  ALWAYS_INLINE friend GSVector2 operator^(const GSVector2& v1, const GSVector2& v2)
  {
    GSVector2 ret;
    ret.U64[0] = v1.U64[0] ^ v2.U64[0];
    return ret;
  }

  ALWAYS_INLINE friend GSVector2 operator==(const GSVector2& v1, const GSVector2& v2)
  {
    GSVector2 ret;
    ret.I32[0] = (v1.x == v2.x) ? -1 : 0;
    ret.I32[1] = (v1.y == v2.y) ? -1 : 0;
    return ret;
  }

  ALWAYS_INLINE friend GSVector2 operator!=(const GSVector2& v1, const GSVector2& v2)
  {
    GSVector2 ret;
    ret.I32[0] = (v1.x != v2.x) ? -1 : 0;
    ret.I32[1] = (v1.y != v2.y) ? -1 : 0;
    return ret;
  }

  ALWAYS_INLINE friend GSVector2 operator>(const GSVector2& v1, const GSVector2& v2)
  {
    GSVector2 ret;
    ret.I32[0] = (v1.x > v2.x) ? -1 : 0;
    ret.I32[1] = (v1.y > v2.y) ? -1 : 0;
    return ret;
  }

  ALWAYS_INLINE friend GSVector2 operator<(const GSVector2& v1, const GSVector2& v2)
  {
    GSVector2 ret;
    ret.I32[0] = (v1.x < v2.x) ? -1 : 0;
    ret.I32[1] = (v1.y < v2.y) ? -1 : 0;
    return ret;
  }

  ALWAYS_INLINE friend GSVector2 operator>=(const GSVector2& v1, const GSVector2& v2)
  {
    GSVector2 ret;
    ret.I32[0] = (v1.x >= v2.x) ? -1 : 0;
    ret.I32[1] = (v1.y >= v2.y) ? -1 : 0;
    return ret;
  }

  ALWAYS_INLINE friend GSVector2 operator<=(const GSVector2& v1, const GSVector2& v2)
  {
    GSVector2 ret;
    ret.I32[0] = (v1.x <= v2.x) ? -1 : 0;
    ret.I32[1] = (v1.y <= v2.y) ? -1 : 0;
    return ret;
  }

  ALWAYS_INLINE GSVector2 xy() const { return *this; }
  ALWAYS_INLINE GSVector2 xx() const { return GSVector2(x, x); }
  ALWAYS_INLINE GSVector2 yx() const { return GSVector2(y, x); }
  ALWAYS_INLINE GSVector2 yy() const { return GSVector2(y, y); }
};

#undef ALL_LANES_8
#undef ALL_LANES_16
#undef ALL_LANES_32

#define ALL_LANES_8(expr)                                                                                              \
  GSVector4i ret;                                                                                                      \
  for (size_t i = 0; i < 16; i++)                                                                                      \
    expr;                                                                                                              \
  return ret;
#define ALL_LANES_16(expr)                                                                                             \
  GSVector4i ret;                                                                                                      \
  for (size_t i = 0; i < 8; i++)                                                                                       \
    expr;                                                                                                              \
  return ret;
#define ALL_LANES_32(expr)                                                                                             \
  GSVector4i ret;                                                                                                      \
  for (size_t i = 0; i < 4; i++)                                                                                       \
    expr;                                                                                                              \
  return ret;
#define ALL_LANES_64(expr)                                                                                             \
  GSVector4i ret;                                                                                                      \
  for (size_t i = 0; i < 2; i++)                                                                                       \
    expr;                                                                                                              \
  return ret;

class alignas(16) GSVector4i
{
  struct cxpr_init_tag
  {
  };
  static constexpr cxpr_init_tag cxpr_init{};

  constexpr GSVector4i(cxpr_init_tag, s32 x, s32 y, s32 z, s32 w) : S32{x, y, z, w} {}

  constexpr GSVector4i(cxpr_init_tag, s16 s0, s16 s1, s16 s2, s16 s3, s16 s4, s16 s5, s16 s6, s16 s7)
    : S16{s0, s1, s2, s3, s4, s5, s6, s7}
  {
  }

  constexpr GSVector4i(cxpr_init_tag, s8 b0, s8 b1, s8 b2, s8 b3, s8 b4, s8 b5, s8 b6, s8 b7, s8 b8, s8 b9, s8 b10,
                       s8 b11, s8 b12, s8 b13, s8 b14, s8 b15)
    : S8{b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
  {
  }

  // Byte indices {start, start + 1, ..., start + 15}, for vec_perm-based byte shifts.
  ALWAYS_INLINE static gsv_u8 seq_u8(u8 start)
  {
    const gsv_u8 seq = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    return vec_add(vec_splats(start), seq);
  }

  // Shifts with NEON semantics: counts at or beyond the lane width produce zero, or the sign for arithmetic shifts.
  template<typename V, typename C>
  ALWAYS_INLINE static V shl_lanes(V a, C count, C limit)
  {
    return vec_sel(vec_sl(a, count), V{}, vec_cmpgt(count, limit));
  }
  template<typename V, typename C>
  ALWAYS_INLINE static V shr_lanes(V a, C count, C limit)
  {
    return vec_sel(vec_sr(a, count), V{}, vec_cmpgt(count, limit));
  }
  template<typename V, typename C>
  ALWAYS_INLINE static V sra_lanes(V a, C count, C limit)
  {
    return vec_sra(a, vec_min(count, limit));
  }

public:
  union
  {
    struct
    {
      s32 x, y, z, w;
    };
    struct
    {
      s32 r, g, b, a;
    };
    struct
    {
      s32 left, top, right, bottom;
    };
    float F32[4];
    s8 S8[16];
    s16 S16[8];
    s32 S32[4];
    s64 S64[2];
    u8 U8[16];
    u16 U16[8];
    u32 U32[4];
    u64 U64[2];
    gsv_s32 v4s;
  };

  GSVector4i() = default;

  ALWAYS_INLINE constexpr static GSVector4i cxpr(s32 x, s32 y, s32 z, s32 w)
  {
    return GSVector4i(cxpr_init, x, y, z, w);
  }

  ALWAYS_INLINE constexpr static GSVector4i cxpr(s32 x) { return GSVector4i(cxpr_init, x, x, x, x); }

  ALWAYS_INLINE constexpr static GSVector4i cxpr16(s16 x) { return GSVector4i(cxpr_init, x, x, x, x, x, x, x, x); }

  ALWAYS_INLINE constexpr static GSVector4i cxpr16(s16 s0, s16 s1, s16 s2, s16 s3, s16 s4, s16 s5, s16 s6, s16 s7)
  {
    return GSVector4i(cxpr_init, s0, s1, s2, s3, s4, s5, s6, s7);
  }

  ALWAYS_INLINE constexpr static GSVector4i cxpr8(s8 b0, s8 b1, s8 b2, s8 b3, s8 b4, s8 b5, s8 b6, s8 b7, s8 b8, s8 b9,
                                                  s8 b10, s8 b11, s8 b12, s8 b13, s8 b14, s8 b15)
  {
    return GSVector4i(cxpr_init, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
  }

  ALWAYS_INLINE GSVector4i(s32 x, s32 y, s32 z, s32 w) : v4s(gsv_s32{x, y, z, w}) {}

  ALWAYS_INLINE GSVector4i(s16 s0, s16 s1, s16 s2, s16 s3, s16 s4, s16 s5, s16 s6, s16 s7)
    : v4s(reinterpret_cast<gsv_s32>(gsv_s16{s0, s1, s2, s3, s4, s5, s6, s7}))
  {
  }

  constexpr GSVector4i(s8 b0, s8 b1, s8 b2, s8 b3, s8 b4, s8 b5, s8 b6, s8 b7, s8 b8, s8 b9, s8 b10, s8 b11, s8 b12,
                       s8 b13, s8 b14, s8 b15)
    : S8{b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
  {
  }

  ALWAYS_INLINE explicit GSVector4i(const GSVector2i& v) : v4s(gsv_s32{v.x, v.y, 0, 0}) {}

  ALWAYS_INLINE explicit GSVector4i(s32 i) { *this = i; }

  ALWAYS_INLINE constexpr explicit GSVector4i(gsv_s32 m) : v4s(m) {}

  // Truncating, saturating conversion like vcvt_s32_f32.
  ALWAYS_INLINE explicit GSVector4i(const GSVector2& v)
  {
    const gsv_f32 f = {v.x, v.y, 0.0f, 0.0f};
    v4s = vec_cts(f, 0);
  }
  ALWAYS_INLINE explicit GSVector4i(const GSVector4& v);

  ALWAYS_INLINE static GSVector4i cast(const GSVector4& v);

  ALWAYS_INLINE void operator=(s32 i) { v4s = vec_splats(i); }

  ALWAYS_INLINE operator gsv_s32() const { return v4s; }

  // rect

  ALWAYS_INLINE s32 width() const { return right - left; }
  ALWAYS_INLINE s32 height() const { return bottom - top; }

  ALWAYS_INLINE GSVector2i rsize() const { return zwzw().sub32(xyxy()).xy(); }

  ALWAYS_INLINE bool rempty() const { return !(x < z && y < w); }

  ALWAYS_INLINE bool rvalid() const { return (x < z && y < w); }

  ALWAYS_INLINE GSVector4i runion(const GSVector4i& a) const { return min_s32(a).upl64(max_s32(a).srl<8>()); }

  ALWAYS_INLINE GSVector4i rintersect(const GSVector4i& a) const { return sat_s32(a); }
  ALWAYS_INLINE bool rintersects(const GSVector4i& v) const { return rintersect(v).rvalid(); }
  ALWAYS_INLINE bool rcontains(const GSVector4i& v) const { return rintersect(v).eq(v); }

  ALWAYS_INLINE u32 rgba32() const { return static_cast<u32>(ps32().pu16().extract32<0>()); }

  ALWAYS_INLINE GSVector4i sat_s8(const GSVector4i& min, const GSVector4i& max) const
  {
    return max_s8(min).min_s8(max);
  }
  ALWAYS_INLINE GSVector4i sat_s8(const GSVector4i& minmax) const
  {
    return max_s8(minmax.xyxy()).min_s8(minmax.zwzw());
  }
  ALWAYS_INLINE GSVector4i sat_s16(const GSVector4i& min, const GSVector4i& max) const
  {
    return max_s16(min).min_s16(max);
  }
  ALWAYS_INLINE GSVector4i sat_s16(const GSVector4i& minmax) const
  {
    return max_s16(minmax.xyxy()).min_s16(minmax.zwzw());
  }
  ALWAYS_INLINE GSVector4i sat_s32(const GSVector4i& min, const GSVector4i& max) const
  {
    return max_s32(min).min_s32(max);
  }
  ALWAYS_INLINE GSVector4i sat_s32(const GSVector4i& minmax) const
  {
    return max_s32(minmax.xyxy()).min_s32(minmax.zwzw());
  }

  ALWAYS_INLINE GSVector4i sat_u8(const GSVector4i& min, const GSVector4i& max) const
  {
    return max_u8(min).min_u8(max);
  }
  ALWAYS_INLINE GSVector4i sat_u8(const GSVector4i& minmax) const
  {
    return max_u8(minmax.xyxy()).min_u8(minmax.zwzw());
  }
  ALWAYS_INLINE GSVector4i sat_u16(const GSVector4i& min, const GSVector4i& max) const
  {
    return max_u16(min).min_u16(max);
  }
  ALWAYS_INLINE GSVector4i sat_u16(const GSVector4i& minmax) const
  {
    return max_u16(minmax.xyxy()).min_u16(minmax.zwzw());
  }
  ALWAYS_INLINE GSVector4i sat_u32(const GSVector4i& min, const GSVector4i& max) const
  {
    return max_u32(min).min_u32(max);
  }
  ALWAYS_INLINE GSVector4i sat_u32(const GSVector4i& minmax) const
  {
    return max_u32(minmax.xyxy()).min_u32(minmax.zwzw());
  }

#define GSV_BINOP(name, func, type)                                                                                    \
  ALWAYS_INLINE GSVector4i name(const GSVector4i& v) const                                                            \
  {                                                                                                                    \
    return GSVector4i(reinterpret_cast<gsv_s32>(func(reinterpret_cast<type>(v4s), reinterpret_cast<type>(v.v4s))));   \
  }

  GSV_BINOP(min_s8, vec_min, gsv_s8)
  GSV_BINOP(max_s8, vec_max, gsv_s8)
  GSV_BINOP(min_s16, vec_min, gsv_s16)
  GSV_BINOP(max_s16, vec_max, gsv_s16)
  GSV_BINOP(min_s32, vec_min, gsv_s32)
  GSV_BINOP(max_s32, vec_max, gsv_s32)
  GSV_BINOP(min_u8, vec_min, gsv_u8)
  GSV_BINOP(max_u8, vec_max, gsv_u8)
  GSV_BINOP(min_u16, vec_min, gsv_u16)
  GSV_BINOP(max_u16, vec_max, gsv_u16)
  GSV_BINOP(min_u32, vec_min, gsv_u32)
  GSV_BINOP(max_u32, vec_max, gsv_u32)

  ALWAYS_INLINE GSVector4i madd_s16(const GSVector4i& v) const
  {
    return GSVector4i(vec_msum(reinterpret_cast<gsv_s16>(v4s), reinterpret_cast<gsv_s16>(v.v4s), vec_splats(0)));
  }

  ALWAYS_INLINE GSVector4i addp_s32() const
  {
    return GSVector4i(
      vec_add(__builtin_shufflevector(v4s, v4s, 0, 2, 0, 2), __builtin_shufflevector(v4s, v4s, 1, 3, 1, 3)));
  }

  ALWAYS_INLINE s32 addv_s32() const { return (S32[0] + S32[1] + S32[2] + S32[3]); }

  ALWAYS_INLINE u8 minv_u8() const
  {
    u8 ret = U8[0];
    for (size_t i = 1; i < 16; i++)
      ret = std::min(ret, U8[i]);
    return ret;
  }

  ALWAYS_INLINE u16 maxv_u8() const
  {
    u8 ret = U8[0];
    for (size_t i = 1; i < 16; i++)
      ret = std::max(ret, U8[i]);
    return ret;
  }

  ALWAYS_INLINE u16 minv_u16() const
  {
    u16 ret = U16[0];
    for (size_t i = 1; i < 8; i++)
      ret = std::min(ret, U16[i]);
    return ret;
  }

  ALWAYS_INLINE u16 maxv_u16() const
  {
    u16 ret = U16[0];
    for (size_t i = 1; i < 8; i++)
      ret = std::max(ret, U16[i]);
    return ret;
  }

  ALWAYS_INLINE s32 minv_s32() const { return std::min(std::min(S32[0], S32[1]), std::min(S32[2], S32[3])); }
  ALWAYS_INLINE u32 minv_u32() const { return std::min(std::min(U32[0], U32[1]), std::min(U32[2], U32[3])); }
  ALWAYS_INLINE s32 maxv_s32() const { return std::max(std::max(S32[0], S32[1]), std::max(S32[2], S32[3])); }
  ALWAYS_INLINE u32 maxv_u32() const { return std::max(std::max(U32[0], U32[1]), std::max(U32[2], U32[3])); }

  ALWAYS_INLINE GSVector4i clamp8() const { return pu16().upl8(); }

  ALWAYS_INLINE GSVector4i blend8(const GSVector4i& a, const GSVector4i& mask) const
  {
    const gsv_bool8 sel = vec_cmplt(reinterpret_cast<gsv_s8>(mask.v4s), vec_splats(static_cast<s8>(0)));
    return GSVector4i(
      reinterpret_cast<gsv_s32>(vec_sel(reinterpret_cast<gsv_u8>(v4s), reinterpret_cast<gsv_u8>(a.v4s), sel)));
  }

  template<int mask>
  ALWAYS_INLINE GSVector4i blend16(const GSVector4i& a) const
  {
    return GSVector4i(reinterpret_cast<gsv_s32>(__builtin_shufflevector(
      reinterpret_cast<gsv_s16>(v4s), reinterpret_cast<gsv_s16>(a.v4s), ((mask & 0x01) == 0) ? 0 : 8,
      ((mask & 0x02) == 0) ? 1 : 9, ((mask & 0x04) == 0) ? 2 : 10, ((mask & 0x08) == 0) ? 3 : 11,
      ((mask & 0x10) == 0) ? 4 : 12, ((mask & 0x20) == 0) ? 5 : 13, ((mask & 0x40) == 0) ? 6 : 14,
      ((mask & 0x80) == 0) ? 7 : 15)));
  }

  template<int mask>
  ALWAYS_INLINE GSVector4i blend32(const GSVector4i& v) const
  {
    return GSVector4i(__builtin_shufflevector(v4s, v.v4s, ((mask & 1) == 0) ? 0 : 4, ((mask & 2) == 0) ? 1 : 5,
                                              ((mask & 4) == 0) ? 2 : 6, ((mask & 8) == 0) ? 3 : 7));
  }

  ALWAYS_INLINE GSVector4i blend(const GSVector4i& v, const GSVector4i& mask) const
  {
    return GSVector4i(vec_sel(v4s, v.v4s, reinterpret_cast<gsv_u32>(mask.v4s)));
  }

  // Like vqtbl1q_u8(): indices outside 0-15 produce zero.
  ALWAYS_INLINE GSVector4i shuffle8(const GSVector4i& mask) const
  {
    const gsv_u8 idx = reinterpret_cast<gsv_u8>(mask.v4s);
    const gsv_u8 zero = vec_splats(static_cast<u8>(0));
    const gsv_u8 res = vec_perm(reinterpret_cast<gsv_u8>(v4s), zero, idx);
    return GSVector4i(
      reinterpret_cast<gsv_s32>(vec_sel(res, zero, vec_cmpgt(idx, vec_splats(static_cast<u8>(15))))));
  }

  ALWAYS_INLINE GSVector4i ps16(const GSVector4i& v) const
  {
    return GSVector4i(
      reinterpret_cast<gsv_s32>(vec_packs(reinterpret_cast<gsv_s16>(v4s), reinterpret_cast<gsv_s16>(v.v4s))));
  }

  ALWAYS_INLINE GSVector4i ps16() const { return ps16(*this); }

  ALWAYS_INLINE GSVector4i pu16(const GSVector4i& v) const
  {
    return GSVector4i(
      reinterpret_cast<gsv_s32>(vec_packsu(reinterpret_cast<gsv_s16>(v4s), reinterpret_cast<gsv_s16>(v.v4s))));
  }

  ALWAYS_INLINE GSVector4i pu16() const { return pu16(*this); }

  ALWAYS_INLINE GSVector4i ps32(const GSVector4i& v) const
  {
    return GSVector4i(reinterpret_cast<gsv_s32>(vec_packs(v4s, v.v4s)));
  }

  ALWAYS_INLINE GSVector4i ps32() const { return ps32(*this); }

  ALWAYS_INLINE GSVector4i pu32(const GSVector4i& v) const
  {
    return GSVector4i(reinterpret_cast<gsv_s32>(vec_packsu(v4s, v.v4s)));
  }

  ALWAYS_INLINE GSVector4i pu32() const { return pu32(*this); }

  GSV_BINOP(upl8, vec_mergeh, gsv_s8)
  GSV_BINOP(uph8, vec_mergel, gsv_s8)
  GSV_BINOP(upl16, vec_mergeh, gsv_s16)
  GSV_BINOP(uph16, vec_mergel, gsv_s16)
  GSV_BINOP(upl32, vec_mergeh, gsv_s32)
  GSV_BINOP(uph32, vec_mergel, gsv_s32)
  GSV_BINOP(upl64, vec_mergeh, gsv_s64)
  GSV_BINOP(uph64, vec_mergel, gsv_s64)

  ALWAYS_INLINE GSVector4i upl8() const { return upl8(zero()); }
  ALWAYS_INLINE GSVector4i uph8() const { return uph8(zero()); }
  ALWAYS_INLINE GSVector4i upl16() const { return upl16(zero()); }
  ALWAYS_INLINE GSVector4i uph16() const { return uph16(zero()); }
  ALWAYS_INLINE GSVector4i upl32() const { return upl32(zero()); }
  ALWAYS_INLINE GSVector4i uph32() const { return uph32(zero()); }
  ALWAYS_INLINE GSVector4i upl64() const { return upl64(zero()); }
  ALWAYS_INLINE GSVector4i uph64() const { return uph64(zero()); }

  // Widening: interleave the low lanes with their sign (or zero).
  ALWAYS_INLINE GSVector4i s8to16() const { return upl8(GSVector4i(reinterpret_cast<gsv_s32>(vec_sra(reinterpret_cast<gsv_s8>(v4s), vec_splats(static_cast<u8>(7)))))); }
  ALWAYS_INLINE GSVector4i u8to16() const { return upl8(); }
  ALWAYS_INLINE GSVector4i s16to32() const { return upl16(GSVector4i(reinterpret_cast<gsv_s32>(vec_sra(reinterpret_cast<gsv_s16>(v4s), vec_splats(static_cast<u16>(15)))))); }
  ALWAYS_INLINE GSVector4i u16to32() const { return upl16(); }
  ALWAYS_INLINE GSVector4i s32to64() const { return upl32(GSVector4i(vec_sra(v4s, vec_splats(31u)))); }
  ALWAYS_INLINE GSVector4i u32to64() const { return upl32(); }
  ALWAYS_INLINE GSVector4i s8to32() const { return s8to16().s16to32(); }
  ALWAYS_INLINE GSVector4i u8to32() const { return u8to16().u16to32(); }
  ALWAYS_INLINE GSVector4i s8to64() const { return s8to32().s32to64(); }
  ALWAYS_INLINE GSVector4i u8to64() const { return u8to32().u32to64(); }
  ALWAYS_INLINE GSVector4i s16to64() const { return s16to32().s32to64(); }
  ALWAYS_INLINE GSVector4i u16to64() const { return u16to32().u32to64(); }

  // Byte shifts of the whole register, like vextq_u8().
  template<int i>
  ALWAYS_INLINE GSVector4i srl() const
  {
    if constexpr (i >= 16)
      return zero();
    else
      return GSVector4i(reinterpret_cast<gsv_s32>(
        vec_perm(reinterpret_cast<gsv_u8>(v4s), vec_splats(static_cast<u8>(0)), seq_u8(static_cast<u8>(i)))));
  }

  template<int i>
  ALWAYS_INLINE GSVector4i srl(const GSVector4i& v) const
  {
    if constexpr (i >= 16)
      return v.srl<i - 16>();
    else
      return GSVector4i(reinterpret_cast<gsv_s32>(
        vec_perm(reinterpret_cast<gsv_u8>(v4s), reinterpret_cast<gsv_u8>(v.v4s), seq_u8(static_cast<u8>(i)))));
  }

  template<int i>
  ALWAYS_INLINE GSVector4i sll() const
  {
    if constexpr (i >= 16)
      return zero();
    else
      return GSVector4i(reinterpret_cast<gsv_s32>(vec_perm(vec_splats(static_cast<u8>(0)),
                                                             reinterpret_cast<gsv_u8>(v4s),
                                                             seq_u8(static_cast<u8>(16 - i)))));
  }

#define GSV_SHIFTS(bits, utype, stype, ctype)                                                                          \
  template<int i>                                                                                                      \
  ALWAYS_INLINE GSVector4i sll##bits() const                                                                          \
  {                                                                                                                    \
    if constexpr (i >= bits)                                                                                           \
      return zero();                                                                                                   \
    else                                                                                                               \
      return GSVector4i(reinterpret_cast<gsv_s32>(                                                                     \
        vec_sl(reinterpret_cast<utype>(v4s), vec_splats(static_cast<ctype>(i)))));                                     \
  }                                                                                                                    \
  ALWAYS_INLINE GSVector4i sll##bits(s32 i) const                                                                     \
  {                                                                                                                    \
    return GSVector4i(reinterpret_cast<gsv_s32>(shl_lanes(reinterpret_cast<utype>(v4s),                             \
                                                          vec_splats(static_cast<ctype>(i)),                          \
                                                          vec_splats(static_cast<ctype>(bits - 1)))));                \
  }                                                                                                                    \
  ALWAYS_INLINE GSVector4i sllv##bits(const GSVector4i& v) const                                                      \
  {                                                                                                                    \
    return GSVector4i(reinterpret_cast<gsv_s32>(shl_lanes(reinterpret_cast<utype>(v4s),                             \
                                                          reinterpret_cast<utype>(v.v4s),                             \
                                                          vec_splats(static_cast<ctype>(bits - 1)))));                \
  }                                                                                                                    \
  template<int i>                                                                                                      \
  ALWAYS_INLINE GSVector4i srl##bits() const                                                                          \
  {                                                                                                                    \
    if constexpr (i >= bits)                                                                                           \
      return zero();                                                                                                   \
    else                                                                                                               \
      return GSVector4i(reinterpret_cast<gsv_s32>(                                                                     \
        vec_sr(reinterpret_cast<utype>(v4s), vec_splats(static_cast<ctype>(i)))));                                     \
  }                                                                                                                    \
  ALWAYS_INLINE GSVector4i srl##bits(s32 i) const                                                                     \
  {                                                                                                                    \
    return GSVector4i(reinterpret_cast<gsv_s32>(shr_lanes(reinterpret_cast<utype>(v4s),                             \
                                                          vec_splats(static_cast<ctype>(i)),                          \
                                                          vec_splats(static_cast<ctype>(bits - 1)))));                \
  }                                                                                                                    \
  ALWAYS_INLINE GSVector4i srlv##bits(const GSVector4i& v) const                                                      \
  {                                                                                                                    \
    return GSVector4i(reinterpret_cast<gsv_s32>(shr_lanes(reinterpret_cast<utype>(v4s),                             \
                                                          reinterpret_cast<utype>(v.v4s),                             \
                                                          vec_splats(static_cast<ctype>(bits - 1)))));                \
  }

  GSV_SHIFTS(16, gsv_u16, gsv_s16, u16)
  GSV_SHIFTS(32, gsv_u32, gsv_s32, u32)
  GSV_SHIFTS(64, gsv_u64, gsv_s64, unsigned long long)

#undef GSV_SHIFTS

  template<int i>
  ALWAYS_INLINE GSVector4i sra16() const
  {
    constexpr int count = (i & ~15) ? 15 : i;
    return GSVector4i(reinterpret_cast<gsv_s32>(
      vec_sra(reinterpret_cast<gsv_s16>(v4s), vec_splats(static_cast<u16>(count)))));
  }

  ALWAYS_INLINE GSVector4i sra16(s32 i) const
  {
    return GSVector4i(reinterpret_cast<gsv_s32>(sra_lanes(
      reinterpret_cast<gsv_s16>(v4s), vec_splats(static_cast<u16>(i)), vec_splats(static_cast<u16>(15)))));
  }

  // NOTE: Matches the NEON implementation, which shifts logically here.
  ALWAYS_INLINE GSVector4i srav16(const GSVector4i& v) const { return srlv16(v); }

  template<int i>
  ALWAYS_INLINE GSVector4i sra32() const
  {
    constexpr u32 count = (i >= 32) ? 31 : i;
    return GSVector4i(vec_sra(v4s, vec_splats(count)));
  }

  ALWAYS_INLINE GSVector4i sra32(s32 i) const
  {
    return GSVector4i(sra_lanes(v4s, vec_splats(static_cast<u32>(i)), vec_splats(31u)));
  }

  ALWAYS_INLINE GSVector4i srav32(const GSVector4i& v) const
  {
    return GSVector4i(sra_lanes(v4s, reinterpret_cast<gsv_u32>(v.v4s), vec_splats(31u)));
  }

  GSV_BINOP(add8, vec_add, gsv_s8)
  GSV_BINOP(add16, vec_add, gsv_s16)
  GSV_BINOP(add32, vec_add, gsv_s32)
  GSV_BINOP(adds8, vec_adds, gsv_s8)
  GSV_BINOP(adds16, vec_adds, gsv_s16)
  GSV_BINOP(addus8, vec_adds, gsv_u8)
  GSV_BINOP(addus16, vec_adds, gsv_u16)
  GSV_BINOP(sub8, vec_sub, gsv_s8)
  GSV_BINOP(sub16, vec_sub, gsv_s16)
  GSV_BINOP(sub32, vec_sub, gsv_s32)
  GSV_BINOP(subs8, vec_subs, gsv_s8)
  GSV_BINOP(subs16, vec_subs, gsv_s16)
  GSV_BINOP(subus8, vec_subs, gsv_u8)
  GSV_BINOP(subus16, vec_subs, gsv_u16)
  GSV_BINOP(avg8, vec_avg, gsv_u8)
  GSV_BINOP(avg16, vec_avg, gsv_u16)
  GSV_BINOP(mul32l, vec_mul, gsv_s32)

  ALWAYS_INLINE GSVector4i hadds16(const GSVector4i& v) const
  {
    const gsv_s16 a = reinterpret_cast<gsv_s16>(v4s);
    const gsv_s16 b = reinterpret_cast<gsv_s16>(v.v4s);
    return GSVector4i(reinterpret_cast<gsv_s32>(vec_adds(__builtin_shufflevector(a, b, 0, 2, 4, 6, 8, 10, 12, 14),
                                                          __builtin_shufflevector(a, b, 1, 3, 5, 7, 9, 11, 13, 15))));
  }

  ALWAYS_INLINE GSVector4i mul16hs(const GSVector4i& v) const
  {
    const gsv_s16 a = reinterpret_cast<gsv_s16>(v4s);
    const gsv_s16 b = reinterpret_cast<gsv_s16>(v.v4s);
    const gsv_s16 even = reinterpret_cast<gsv_s16>(vec_mule(a, b));
    const gsv_s16 odd = reinterpret_cast<gsv_s16>(vec_mulo(a, b));
    return GSVector4i(reinterpret_cast<gsv_s32>(__builtin_shufflevector(even, odd, 1, 9, 3, 11, 5, 13, 7, 15)));
  }

  ALWAYS_INLINE GSVector4i mul16l(const GSVector4i& v) const
  {
    return GSVector4i(reinterpret_cast<gsv_s32>(vec_mladd(reinterpret_cast<gsv_s16>(v4s),
                                                          reinterpret_cast<gsv_s16>(v.v4s),
                                                          vec_splats(static_cast<s16>(0)))));
  }

  // Rounding high multiply like vrshrn_n_s32(a * b, 15), without the saturation of vec_mradds().
  ALWAYS_INLINE GSVector4i mul16hrs(const GSVector4i& v) const
  {
    const gsv_s16 a = reinterpret_cast<gsv_s16>(v4s);
    const gsv_s16 b = reinterpret_cast<gsv_s16>(v.v4s);
    const gsv_s32 round = vec_splats(0x4000);
    const gsv_u32 shift = vec_splats(15u);
    const gsv_s16 even = reinterpret_cast<gsv_s16>(vec_sra(vec_add(vec_mule(a, b), round), shift));
    const gsv_s16 odd = reinterpret_cast<gsv_s16>(vec_sra(vec_add(vec_mulo(a, b), round), shift));
    return GSVector4i(reinterpret_cast<gsv_s32>(__builtin_shufflevector(even, odd, 0, 8, 2, 10, 4, 12, 6, 14)));
  }

  ALWAYS_INLINE bool eq(const GSVector4i& v) const { return vec_all_eq(v4s, v.v4s); }

#define GSV_CMP(name, expr, type)                                                                                      \
  ALWAYS_INLINE GSVector4i name(const GSVector4i& v) const                                                            \
  {                                                                                                                    \
    const type a = reinterpret_cast<type>(v4s);                                                                        \
    const type b = reinterpret_cast<type>(v.v4s);                                                                      \
    return GSVector4i(reinterpret_cast<gsv_s32>(expr));                                                                \
  }

  GSV_CMP(eq8, vec_cmpeq(a, b), gsv_s8)
  GSV_CMP(eq16, vec_cmpeq(a, b), gsv_s16)
  GSV_CMP(eq32, vec_cmpeq(a, b), gsv_s32)
  GSV_CMP(eq64, vec_cmpeq(a, b), gsv_s64)
  GSV_CMP(gt8, vec_cmpgt(a, b), gsv_s8)
  GSV_CMP(gt16, vec_cmpgt(a, b), gsv_s16)
  GSV_CMP(gt32, vec_cmpgt(a, b), gsv_s32)
  GSV_CMP(lt8, vec_cmpgt(b, a), gsv_s8)
  GSV_CMP(lt16, vec_cmpgt(b, a), gsv_s16)
  GSV_CMP(lt32, vec_cmpgt(b, a), gsv_s32)
  GSV_CMP(ge8, vec_nor(vec_cmpgt(b, a), vec_cmpgt(b, a)), gsv_s8)
  GSV_CMP(ge16, vec_nor(vec_cmpgt(b, a), vec_cmpgt(b, a)), gsv_s16)
  GSV_CMP(ge32, vec_nor(vec_cmpgt(b, a), vec_cmpgt(b, a)), gsv_s32)
  GSV_CMP(le8, vec_nor(vec_cmpgt(a, b), vec_cmpgt(a, b)), gsv_s8)
  GSV_CMP(le16, vec_nor(vec_cmpgt(a, b), vec_cmpgt(a, b)), gsv_s16)
  GSV_CMP(le32, vec_nor(vec_cmpgt(a, b), vec_cmpgt(a, b)), gsv_s32)

#undef GSV_CMP

  ALWAYS_INLINE GSVector4i neq8(const GSVector4i& v) const { return ~eq8(v); }
  ALWAYS_INLINE GSVector4i neq16(const GSVector4i& v) const { return ~eq16(v); }
  ALWAYS_INLINE GSVector4i neq32(const GSVector4i& v) const { return ~eq32(v); }

  ALWAYS_INLINE GSVector4i andnot(const GSVector4i& v) const { return GSVector4i(vec_andc(v4s, v.v4s)); }

  // Bit 7 of each byte, like _mm_movemask_epi8(). Same approach as GCC's emmintrin.h for POWER8.
  ALWAYS_INLINE int mask() const
  {
    const gsv_u8 perm = {0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
                         0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00};
    const gsv_u64 res = reinterpret_cast<gsv_u64>(vec_vbpermq(reinterpret_cast<gsv_u8>(v4s), perm));
    return static_cast<int>(res[1]);
  }

  ALWAYS_INLINE bool alltrue() const { return vec_all_eq(v4s, vec_splats(-1)); }
  ALWAYS_INLINE bool anytrue() const { return vec_any_ne(v4s, vec_splats(0)); }
  ALWAYS_INLINE bool allfalse() const { return vec_all_eq(v4s, vec_splats(0)); }

  template<int i>
  ALWAYS_INLINE GSVector4i insert8(int a) const
  {
    return GSVector4i(reinterpret_cast<gsv_s32>(vec_insert(static_cast<u8>(a), reinterpret_cast<gsv_u8>(v4s), i)));
  }

  template<int i>
  ALWAYS_INLINE int extract8() const
  {
    return vec_extract(reinterpret_cast<gsv_u8>(v4s), i);
  }

  template<int i>
  ALWAYS_INLINE GSVector4i insert16(int a) const
  {
    return GSVector4i(reinterpret_cast<gsv_s32>(vec_insert(static_cast<u16>(a), reinterpret_cast<gsv_u16>(v4s), i)));
  }

  template<int i>
  ALWAYS_INLINE int extract16() const
  {
    return vec_extract(reinterpret_cast<gsv_u16>(v4s), i);
  }

  template<int i>
  ALWAYS_INLINE GSVector4i insert32(int a) const
  {
    return GSVector4i(vec_insert(static_cast<s32>(a), v4s, i));
  }

  template<int i>
  ALWAYS_INLINE int extract32() const
  {
    return vec_extract(v4s, i);
  }

  template<int i>
  ALWAYS_INLINE GSVector4i insert64(s64 a) const
  {
    return GSVector4i(reinterpret_cast<gsv_s32>(vec_insert(static_cast<long long>(a), reinterpret_cast<gsv_s64>(v4s), i)));
  }

  template<int i>
  ALWAYS_INLINE s64 extract64() const
  {
    return vec_extract(reinterpret_cast<gsv_s64>(v4s), i);
  }

  ALWAYS_INLINE static GSVector4i loadnt(const void* p) { return load<true>(p); }

  ALWAYS_INLINE static GSVector4i load32(const void* p)
  {
    s32 val;
    std::memcpy(&val, p, sizeof(s32));
    return zext32(val);
  }

  ALWAYS_INLINE static GSVector4i zext32(s32 v) { return GSVector4i(gsv_s32{v, 0, 0, 0}); }

  template<bool aligned>
  ALWAYS_INLINE static GSVector4i loadl(const void* p)
  {
    s64 val;
    std::memcpy(&val, p, sizeof(val));
    return GSVector4i(reinterpret_cast<gsv_s32>(gsv_s64{val, 0}));
  }

  ALWAYS_INLINE static GSVector4i loadl(const GSVector2i& v) { return GSVector4i(gsv_s32{v.x, v.y, 0, 0}); }

  template<bool aligned>
  ALWAYS_INLINE static GSVector4i loadh(const void* p)
  {
    s64 val;
    std::memcpy(&val, p, sizeof(val));
    return GSVector4i(reinterpret_cast<gsv_s32>(gsv_s64{0, val}));
  }

  ALWAYS_INLINE static GSVector4i loadh(const GSVector2i& v) { return GSVector4i(gsv_s32{0, 0, v.x, v.y}); }

  template<bool aligned>
  ALWAYS_INLINE static GSVector4i load(const void* p)
  {
    gsv_s32 v;
    std::memcpy(&v, p, sizeof(v));
    return GSVector4i(v);
  }

  ALWAYS_INLINE static void storent(void* p, const GSVector4i& v) { store<true>(p, v); }

  ALWAYS_INLINE static void store32(void* p, const GSVector4i& v)
  {
    const s32 val = v.extract32<0>();
    std::memcpy(p, &val, sizeof(val));
  }

  template<bool aligned>
  ALWAYS_INLINE static void storel(void* p, const GSVector4i& v)
  {
    const s64 val = v.extract64<0>();
    std::memcpy(p, &val, sizeof(val));
  }

  template<bool aligned>
  ALWAYS_INLINE static void storeh(void* p, const GSVector4i& v)
  {
    const s64 val = v.extract64<1>();
    std::memcpy(p, &val, sizeof(val));
  }

  template<bool aligned>
  ALWAYS_INLINE static void store(void* p, const GSVector4i& v)
  {
    std::memcpy(p, &v.v4s, sizeof(v.v4s));
  }

  ALWAYS_INLINE static GSVector4i broadcast128(const GSVector4i& v) { return v; }

  template<bool aligned>
  ALWAYS_INLINE static GSVector4i broadcast128(const void* v)
  {
    return load<aligned>(v);
  }

  ALWAYS_INLINE void operator&=(const GSVector4i& v) { v4s = vec_and(v4s, v.v4s); }
  ALWAYS_INLINE void operator|=(const GSVector4i& v) { v4s = vec_or(v4s, v.v4s); }
  ALWAYS_INLINE void operator^=(const GSVector4i& v) { v4s = vec_xor(v4s, v.v4s); }

  ALWAYS_INLINE friend GSVector4i operator&(const GSVector4i& v1, const GSVector4i& v2)
  {
    return GSVector4i(vec_and(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4i operator|(const GSVector4i& v1, const GSVector4i& v2)
  {
    return GSVector4i(vec_or(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4i operator^(const GSVector4i& v1, const GSVector4i& v2)
  {
    return GSVector4i(vec_xor(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4i operator&(const GSVector4i& v, int i) { return v & GSVector4i(i); }
  ALWAYS_INLINE friend GSVector4i operator|(const GSVector4i& v, int i) { return v | GSVector4i(i); }
  ALWAYS_INLINE friend GSVector4i operator^(const GSVector4i& v, int i) { return v ^ GSVector4i(i); }

  ALWAYS_INLINE friend GSVector4i operator~(const GSVector4i& v) { return GSVector4i(vec_nor(v.v4s, v.v4s)); }

  ALWAYS_INLINE static constexpr GSVector4i zero() { return GSVector4i::cxpr(0, 0, 0, 0); }

  ALWAYS_INLINE static GSVector4i xffffffff() { return GSVector4i(vec_splats(-1)); }

  ALWAYS_INLINE GSVector4i xyxy(const GSVector4i& v) const { return upl64(v); }

  ALWAYS_INLINE static GSVector4i xyxy(const GSVector2i& xy, const GSVector2i& zw)
  {
    return GSVector4i(gsv_s32{xy.x, xy.y, zw.x, zw.y});
  }

  ALWAYS_INLINE static GSVector4i xyxy(const GSVector2i& xyzw)
  {
    return GSVector4i(gsv_s32{xyzw.x, xyzw.y, xyzw.x, xyzw.y});
  }

  static GSVector4i rfit(const GSVector4i& fit_rect, const GSVector2i& image_size);

  ALWAYS_INLINE GSVector2i xy() const { return GSVector2i(x, y); }

  ALWAYS_INLINE GSVector2i zw() const { return GSVector2i(z, w); }

#undef GSV_BINOP

#define VECTOR4i_SHUFFLE_4(xs, xn, ys, yn, zs, zn, ws, wn)                                                             \
  ALWAYS_INLINE GSVector4i xs##ys##zs##ws() const                                                                      \
  {                                                                                                                    \
    return GSVector4i(__builtin_shufflevector(v4s, v4s, xn, yn, zn, wn));                                              \
  }                                                                                                                    \
  ALWAYS_INLINE GSVector4i xs##ys##zs##ws##l() const                                                                   \
  {                                                                                                                    \
    return GSVector4i(reinterpret_cast<gsv_s32>(__builtin_shufflevector(                                               \
      reinterpret_cast<gsv_s16>(v4s), reinterpret_cast<gsv_s16>(v4s), xn, yn, zn, wn, 4, 5, 6, 7)));                   \
  }                                                                                                                    \
  ALWAYS_INLINE GSVector4i xs##ys##zs##ws##h() const                                                                   \
  {                                                                                                                    \
    return GSVector4i(reinterpret_cast<gsv_s32>(__builtin_shufflevector(                                               \
      reinterpret_cast<gsv_s16>(v4s), reinterpret_cast<gsv_s16>(v4s), 0, 1, 2, 3, 4 + xn, 4 + yn, 4 + zn, 4 + wn)));   \
  }

#define VECTOR4i_SHUFFLE_3(xs, xn, ys, yn, zs, zn)                                                                     \
  VECTOR4i_SHUFFLE_4(xs, xn, ys, yn, zs, zn, x, 0);                                                                    \
  VECTOR4i_SHUFFLE_4(xs, xn, ys, yn, zs, zn, y, 1);                                                                    \
  VECTOR4i_SHUFFLE_4(xs, xn, ys, yn, zs, zn, z, 2);                                                                    \
  VECTOR4i_SHUFFLE_4(xs, xn, ys, yn, zs, zn, w, 3);

#define VECTOR4i_SHUFFLE_2(xs, xn, ys, yn)                                                                             \
  VECTOR4i_SHUFFLE_3(xs, xn, ys, yn, x, 0);                                                                            \
  VECTOR4i_SHUFFLE_3(xs, xn, ys, yn, y, 1);                                                                            \
  VECTOR4i_SHUFFLE_3(xs, xn, ys, yn, z, 2);                                                                            \
  VECTOR4i_SHUFFLE_3(xs, xn, ys, yn, w, 3);

#define VECTOR4i_SHUFFLE_1(xs, xn)                                                                                     \
  VECTOR4i_SHUFFLE_2(xs, xn, x, 0);                                                                                    \
  VECTOR4i_SHUFFLE_2(xs, xn, y, 1);                                                                                    \
  VECTOR4i_SHUFFLE_2(xs, xn, z, 2);                                                                                    \
  VECTOR4i_SHUFFLE_2(xs, xn, w, 3);

  VECTOR4i_SHUFFLE_1(x, 0);
  VECTOR4i_SHUFFLE_1(y, 1);
  VECTOR4i_SHUFFLE_1(z, 2);
  VECTOR4i_SHUFFLE_1(w, 3);

#undef VECTOR4i_SHUFFLE_1
#undef VECTOR4i_SHUFFLE_2
#undef VECTOR4i_SHUFFLE_3
#undef VECTOR4i_SHUFFLE_4
};


class alignas(16) GSVector4
{
  struct cxpr_init_tag
  {
  };
  static constexpr cxpr_init_tag cxpr_init{};

  constexpr GSVector4(cxpr_init_tag, float x, float y, float z, float w) : F32{x, y, z, w} {}

  constexpr GSVector4(cxpr_init_tag, int x, int y, int z, int w) : I32{x, y, z, w} {}

  constexpr GSVector4(cxpr_init_tag, u64 x, u64 y) : U64{x, y} {}

  constexpr GSVector4(cxpr_init_tag, double x, double y) : F64{x, y} {}

  ALWAYS_INLINE static GSVector4 from_s32(gsv_s32 v) { return GSVector4(vec_ctf(v, 0)); }
  ALWAYS_INLINE static GSVector4 from_f64(gsv_f64 v) { return GSVector4(reinterpret_cast<gsv_f32>(v)); }
  ALWAYS_INLINE gsv_f64 as_f64() const { return reinterpret_cast<gsv_f64>(v4s); }
  ALWAYS_INLINE gsv_u32 as_u32() const { return reinterpret_cast<gsv_u32>(v4s); }
  ALWAYS_INLINE static GSVector4 from_u32(gsv_u32 v) { return GSVector4(reinterpret_cast<gsv_f32>(v)); }

public:
  union
  {
    struct
    {
      float x, y, z, w;
    };
    struct
    {
      float r, g, b, a;
    };
    struct
    {
      float left, top, right, bottom;
    };
    float F32[4];
    double F64[2];
    s8 I8[16];
    s16 I16[8];
    s32 I32[4];
    s64 I64[2];
    u8 U8[16];
    u16 U16[8];
    u32 U32[4];
    u64 U64[2];
    gsv_f32 v4s;
  };

  GSVector4() = default;

  constexpr static GSVector4 cxpr(float x, float y, float z, float w) { return GSVector4(cxpr_init, x, y, z, w); }
  constexpr static GSVector4 cxpr(float x) { return GSVector4(cxpr_init, x, x, x, x); }

  constexpr static GSVector4 cxpr(int x, int y, int z, int w) { return GSVector4(cxpr_init, x, y, z, w); }
  constexpr static GSVector4 cxpr(int x) { return GSVector4(cxpr_init, x, x, x, x); }

  constexpr static GSVector4 cxpr64(u64 x, u64 y) { return GSVector4(cxpr_init, x, y); }
  constexpr static GSVector4 cxpr64(u64 x) { return GSVector4(cxpr_init, x, x); }

  constexpr static GSVector4 cxpr64(double x, double y) { return GSVector4(cxpr_init, x, y); }
  constexpr static GSVector4 cxpr64(double x) { return GSVector4(cxpr_init, x, x); }

  constexpr static GSVector4 cxpr_rgba32(u32 rgba)
  {
    return GSVector4(cxpr_init, static_cast<float>(rgba & 0xff), static_cast<float>((rgba >> 8) & 0xff),
                     static_cast<float>((rgba >> 16) & 0xff), static_cast<float>((rgba >> 24) & 0xff));
  }

  constexpr static GSVector4 cxpr_unorm8(u32 rgba)
  {
    return GSVector4(cxpr_init, static_cast<float>(rgba & 0xff) / 255.0f,
                     static_cast<float>((rgba >> 8) & 0xff) / 255.0f, static_cast<float>((rgba >> 16) & 0xff) / 255.0f,
                     static_cast<float>((rgba >> 24) & 0xff) / 255.0f);
  }

  ALWAYS_INLINE GSVector4(float x, float y, float z, float w) : v4s(gsv_f32{x, y, z, w}) {}

  ALWAYS_INLINE GSVector4(float x, float y) : v4s(gsv_f32{x, y, 0.0f, 0.0f}) {}

  ALWAYS_INLINE GSVector4(int x, int y, int z, int w) : v4s(from_s32(gsv_s32{x, y, z, w}).v4s) {}

  ALWAYS_INLINE GSVector4(int x, int y) : v4s(from_s32(gsv_s32{x, y, 0, 0}).v4s) {}

  ALWAYS_INLINE explicit GSVector4(const GSVector2& v) : v4s(gsv_f32{v.x, v.y, 0.0f, 0.0f}) {}

  ALWAYS_INLINE explicit GSVector4(const GSVector2i& v) : v4s(from_s32(gsv_s32{v.x, v.y, 0, 0}).v4s) {}

  ALWAYS_INLINE constexpr explicit GSVector4(gsv_f32 m) : v4s(m) {}

  ALWAYS_INLINE explicit GSVector4(float f) : v4s(vec_splats(f)) {}

  ALWAYS_INLINE explicit GSVector4(int i) : v4s(from_s32(vec_splats(i)).v4s) {}

  ALWAYS_INLINE explicit GSVector4(const GSVector4i& v);

  ALWAYS_INLINE static GSVector4 cast(const GSVector4i& v);

  ALWAYS_INLINE static GSVector4 f64(double x, double y) { return from_f64(gsv_f64{x, y}); }

  ALWAYS_INLINE static GSVector4 f64(double x) { return from_f64(vec_splats(x)); }

  ALWAYS_INLINE void operator=(float f) { v4s = vec_splats(f); }

  ALWAYS_INLINE void operator=(gsv_f32 m) { v4s = m; }

  ALWAYS_INLINE operator gsv_f32() const { return v4s; }

  ALWAYS_INLINE u32 rgba32() const { return GSVector4i(*this).rgba32(); }

  ALWAYS_INLINE static GSVector4 rgba32(u32 rgba)
  {
    return GSVector4(GSVector4i::zext32(static_cast<s32>(rgba)).u8to32());
  }

  ALWAYS_INLINE static GSVector4 unorm8(u32 rgba) { return rgba32(rgba) * GSVector4::cxpr(1.0f / 255.0f); }

  ALWAYS_INLINE GSVector4 abs() const { return GSVector4(vec_abs(v4s)); }

  ALWAYS_INLINE GSVector4 neg() const { return from_u32(vec_xor(as_u32(), vec_splats(0x80000000u))); }

  ALWAYS_INLINE GSVector4 floor() const { return GSVector4(vec_floor(v4s)); }

  ALWAYS_INLINE GSVector4 ceil() const { return GSVector4(vec_ceil(v4s)); }

  // Pairwise ops, like vpaddq_f32() / vuzp.
  ALWAYS_INLINE GSVector4 hadd() const { return hadd(*this); }

  ALWAYS_INLINE GSVector4 hadd(const GSVector4& v) const
  {
    return GSVector4(vec_add(__builtin_shufflevector(v4s, v.v4s, 0, 2, 4, 6), __builtin_shufflevector(v4s, v.v4s, 1, 3, 5, 7)));
  }

  ALWAYS_INLINE GSVector4 hsub() const { return hsub(*this); }

  ALWAYS_INLINE GSVector4 hsub(const GSVector4& v) const
  {
    return GSVector4(vec_sub(__builtin_shufflevector(v4s, v.v4s, 0, 2, 4, 6), __builtin_shufflevector(v4s, v.v4s, 1, 3, 5, 7)));
  }

  // Same summation order as vaddvq_f32(): (x + y) + (z + w).
  ALWAYS_INLINE float dot(const GSVector4& v) const
  {
    const gsv_f32 p = vec_mul(v4s, v.v4s);
    return (p[0] + p[1]) + (p[2] + p[3]);
  }

  ALWAYS_INLINE float addv() const { return (F32[0] + F32[1]) + (F32[2] + F32[3]); }

  ALWAYS_INLINE float minv() const { return std::min(std::min(F32[0], F32[1]), std::min(F32[2], F32[3])); }

  ALWAYS_INLINE float maxv() const { return std::max(std::max(F32[0], F32[1]), std::max(F32[2], F32[3])); }

  ALWAYS_INLINE float width() const { return right - left; }
  ALWAYS_INLINE float height() const { return bottom - top; }

  ALWAYS_INLINE GSVector2 rsize() const { return (zw() - xy()); }

  ALWAYS_INLINE bool rempty() const { return !(x < z && y < w); }

  ALWAYS_INLINE bool rvalid() const { return !(x >= z) && !(y >= w); }

  ALWAYS_INLINE GSVector4 runion(const GSVector4 v) const { return min(v).blend32<0xc>(max(v)); }

  ALWAYS_INLINE GSVector4 rintersect(const GSVector4& a) const { return sat(a); }
  ALWAYS_INLINE bool rintersects(const GSVector4& v) const { return rintersect(v).rvalid(); }
  ALWAYS_INLINE bool rcontains(const GSVector4& v) const { return rintersect(v).eq(v); }

  ALWAYS_INLINE GSVector4 sat(const GSVector4& a, const GSVector4& b) const { return max(a).min(b); }

  ALWAYS_INLINE GSVector4 sat(const GSVector4& a) const { return sat(a.xyxy(), a.zwzw()); }

  ALWAYS_INLINE GSVector4 sat(const float scale = 255) const { return sat(zero(), GSVector4(scale)); }

  ALWAYS_INLINE GSVector4 clamp(const float scale = 255) const { return min(GSVector4(scale)); }

  ALWAYS_INLINE GSVector4 min(const GSVector4& a) const { return GSVector4(vec_min(v4s, a.v4s)); }

  ALWAYS_INLINE GSVector4 max(const GSVector4& a) const { return GSVector4(vec_max(v4s, a.v4s)); }

  template<int mask>
  ALWAYS_INLINE GSVector4 blend32(const GSVector4& a) const
  {
    return GSVector4(__builtin_shufflevector(v4s, a.v4s, (mask & 1) ? 4 : 0, (mask & 2) ? 5 : 1, (mask & 4) ? 6 : 2,
                                             (mask & 8) ? 7 : 3));
  }

  // Selects by the sign bit of each mask lane.
  ALWAYS_INLINE GSVector4 blend32(const GSVector4& a, const GSVector4& mask) const
  {
    const gsv_u32 bitmask =
      reinterpret_cast<gsv_u32>(vec_sra(reinterpret_cast<gsv_s32>(mask.v4s), vec_splats(31u)));
    return GSVector4(vec_sel(v4s, a.v4s, bitmask));
  }

  ALWAYS_INLINE GSVector4 upl(const GSVector4& a) const { return GSVector4(vec_mergeh(v4s, a.v4s)); }

  ALWAYS_INLINE GSVector4 uph(const GSVector4& a) const { return GSVector4(vec_mergel(v4s, a.v4s)); }

  ALWAYS_INLINE GSVector4 upld(const GSVector4& a) const { return from_f64(vec_mergeh(as_f64(), a.as_f64())); }

  ALWAYS_INLINE GSVector4 uphd(const GSVector4& a) const { return from_f64(vec_mergel(as_f64(), a.as_f64())); }

  ALWAYS_INLINE GSVector4 l2h(const GSVector4& a) const { return GSVector4(__builtin_shufflevector(v4s, a.v4s, 0, 1, 4, 5)); }

  ALWAYS_INLINE GSVector4 h2l(const GSVector4& a) const { return GSVector4(__builtin_shufflevector(v4s, a.v4s, 6, 7, 2, 3)); }

  ALWAYS_INLINE GSVector4 andnot(const GSVector4& v) const { return GSVector4(vec_andc(v4s, v.v4s)); }

  ALWAYS_INLINE int mask() const
  {
    return static_cast<int>((U32[0] >> 31) | ((U32[1] >> 31) << 1) | ((U32[2] >> 31) << 2) | ((U32[3] >> 31) << 3));
  }

  ALWAYS_INLINE bool alltrue() const { return vec_all_eq(as_u32(), vec_splats(0xFFFFFFFFu)); }

  ALWAYS_INLINE bool anytrue() const { return vec_any_ne(as_u32(), vec_splats(0u)); }

  ALWAYS_INLINE bool allfalse() const { return vec_all_eq(as_u32(), vec_splats(0u)); }

  ALWAYS_INLINE GSVector4 replace_nan(const GSVector4& v) const { return v.blend32(*this, *this == *this); }

  template<int src, int dst>
  ALWAYS_INLINE GSVector4 insert32(const GSVector4& v) const
  {
    return GSVector4(vec_insert(vec_extract(v.v4s, src), v4s, dst));
  }

  template<int i>
  ALWAYS_INLINE GSVector4 insert32(float v) const
  {
    return GSVector4(vec_insert(v, v4s, i));
  }

  template<int i>
  ALWAYS_INLINE float extract32() const
  {
    return vec_extract(v4s, i);
  }

  template<int dst>
  ALWAYS_INLINE GSVector4 insert64(double v) const
  {
    return from_f64(vec_insert(v, as_f64(), dst));
  }

  template<int src>
  ALWAYS_INLINE double extract64() const
  {
    return vec_extract(as_f64(), src);
  }

  ALWAYS_INLINE static constexpr GSVector4 zero() { return GSVector4::cxpr(0.0f, 0.0f, 0.0f, 0.0f); }

  ALWAYS_INLINE static GSVector4 xffffffff() { return from_u32(vec_splats(0xFFFFFFFFu)); }

  template<bool aligned>
  ALWAYS_INLINE static GSVector4 loadl(const void* p)
  {
    u64 val;
    std::memcpy(&val, p, sizeof(val));
    return GSVector4(reinterpret_cast<gsv_f32>(gsv_u64{val, 0}));
  }

  template<bool aligned>
  ALWAYS_INLINE static GSVector4 loadh(const void* p)
  {
    u64 val;
    std::memcpy(&val, p, sizeof(val));
    return GSVector4(reinterpret_cast<gsv_f32>(gsv_u64{0, val}));
  }

  ALWAYS_INLINE static GSVector4 loadh(const GSVector2& v) { return GSVector4(gsv_f32{0.0f, 0.0f, v.x, v.y}); }

  ALWAYS_INLINE static GSVector4 load(float f) { return GSVector4(gsv_f32{f, 0.0f, 0.0f, 0.0f}); }

  template<bool aligned>
  ALWAYS_INLINE static GSVector4 load(const void* p)
  {
    gsv_f32 v;
    std::memcpy(&v, p, sizeof(v));
    return GSVector4(v);
  }

  ALWAYS_INLINE static void storent(void* p, const GSVector4& v) { store<true>(p, v); }

  template<bool aligned>
  ALWAYS_INLINE static void storel(void* p, const GSVector4& v)
  {
    std::memcpy(p, &v.F32[0], sizeof(float) * 2);
  }

  template<bool aligned>
  ALWAYS_INLINE static void storeh(void* p, const GSVector4& v)
  {
    std::memcpy(p, &v.F32[2], sizeof(float) * 2);
  }

  template<bool aligned>
  ALWAYS_INLINE static void store(void* p, const GSVector4& v)
  {
    std::memcpy(p, &v.v4s, sizeof(v.v4s));
  }

  ALWAYS_INLINE static void store(float* p, const GSVector4& v) { *p = v.extract32<0>(); }

  ALWAYS_INLINE GSVector4 operator-() const { return neg(); }

  ALWAYS_INLINE void operator+=(const GSVector4& v) { v4s = vec_add(v4s, v.v4s); }
  ALWAYS_INLINE void operator-=(const GSVector4& v) { v4s = vec_sub(v4s, v.v4s); }
  ALWAYS_INLINE void operator*=(const GSVector4& v) { v4s = vec_mul(v4s, v.v4s); }
  ALWAYS_INLINE void operator/=(const GSVector4& v) { v4s = vec_div(v4s, v.v4s); }

  ALWAYS_INLINE void operator+=(float f) { *this += GSVector4(f); }
  ALWAYS_INLINE void operator-=(float f) { *this -= GSVector4(f); }
  ALWAYS_INLINE void operator*=(float f) { *this *= GSVector4(f); }
  ALWAYS_INLINE void operator/=(float f) { *this /= GSVector4(f); }

  ALWAYS_INLINE void operator&=(const GSVector4& v) { v4s = vec_and(v4s, v.v4s); }
  ALWAYS_INLINE void operator|=(const GSVector4& v) { v4s = vec_or(v4s, v.v4s); }
  ALWAYS_INLINE void operator^=(const GSVector4& v) { v4s = vec_xor(v4s, v.v4s); }

  ALWAYS_INLINE friend GSVector4 operator+(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(vec_add(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4 operator-(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(vec_sub(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4 operator*(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(vec_mul(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4 operator/(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(vec_div(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4 operator+(const GSVector4& v, float f) { return v + GSVector4(f); }
  ALWAYS_INLINE friend GSVector4 operator-(const GSVector4& v, float f) { return v - GSVector4(f); }
  ALWAYS_INLINE friend GSVector4 operator*(const GSVector4& v, float f) { return v * GSVector4(f); }
  ALWAYS_INLINE friend GSVector4 operator/(const GSVector4& v, float f) { return v / GSVector4(f); }

  ALWAYS_INLINE friend GSVector4 operator&(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(vec_and(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4 operator|(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(vec_or(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4 operator^(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(vec_xor(v1.v4s, v2.v4s));
  }

  ALWAYS_INLINE friend GSVector4 operator==(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(reinterpret_cast<gsv_f32>(vec_cmpeq(v1.v4s, v2.v4s)));
  }

  ALWAYS_INLINE friend GSVector4 operator!=(const GSVector4& v1, const GSVector4& v2)
  {
    const gsv_u32 eq = reinterpret_cast<gsv_u32>(vec_cmpeq(v1.v4s, v2.v4s));
    return from_u32(vec_nor(eq, eq));
  }

  ALWAYS_INLINE friend GSVector4 operator>(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(reinterpret_cast<gsv_f32>(vec_cmpgt(v1.v4s, v2.v4s)));
  }

  ALWAYS_INLINE friend GSVector4 operator<(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(reinterpret_cast<gsv_f32>(vec_cmplt(v1.v4s, v2.v4s)));
  }

  ALWAYS_INLINE friend GSVector4 operator>=(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(reinterpret_cast<gsv_f32>(vec_cmpge(v1.v4s, v2.v4s)));
  }

  ALWAYS_INLINE friend GSVector4 operator<=(const GSVector4& v1, const GSVector4& v2)
  {
    return GSVector4(reinterpret_cast<gsv_f32>(vec_cmple(v1.v4s, v2.v4s)));
  }

  ALWAYS_INLINE bool eq(const GSVector4& v) const { return vec_all_eq(v4s, v.v4s); }

  ALWAYS_INLINE GSVector4 mul64(const GSVector4& v) const { return from_f64(vec_mul(as_f64(), v.as_f64())); }

  ALWAYS_INLINE GSVector4 add64(const GSVector4& v) const { return from_f64(vec_add(as_f64(), v.as_f64())); }

  ALWAYS_INLINE GSVector4 sub64(const GSVector4& v) const { return from_f64(vec_sub(as_f64(), v.as_f64())); }

  ALWAYS_INLINE GSVector4 div64(const GSVector4& v) const { return from_f64(vec_div(as_f64(), v.as_f64())); }

  ALWAYS_INLINE GSVector4 gt64(const GSVector4& v) const
  {
    return from_f64(reinterpret_cast<gsv_f64>(vec_cmpgt(as_f64(), v.as_f64())));
  }

  ALWAYS_INLINE GSVector4 eq64(const GSVector4& v) const
  {
    return from_f64(reinterpret_cast<gsv_f64>(vec_cmpeq(as_f64(), v.as_f64())));
  }

  ALWAYS_INLINE GSVector4 lt64(const GSVector4& v) const
  {
    return from_f64(reinterpret_cast<gsv_f64>(vec_cmplt(as_f64(), v.as_f64())));
  }

  ALWAYS_INLINE GSVector4 ge64(const GSVector4& v) const
  {
    return from_f64(reinterpret_cast<gsv_f64>(vec_cmpge(as_f64(), v.as_f64())));
  }

  ALWAYS_INLINE GSVector4 le64(const GSVector4& v) const
  {
    return from_f64(reinterpret_cast<gsv_f64>(vec_cmple(as_f64(), v.as_f64())));
  }

  ALWAYS_INLINE GSVector4 min64(const GSVector4& v) const { return from_f64(vec_min(as_f64(), v.as_f64())); }

  ALWAYS_INLINE GSVector4 max64(const GSVector4& v) const { return from_f64(vec_max(as_f64(), v.as_f64())); }

  ALWAYS_INLINE GSVector4 abs64() const { return *this & GSVector4::cxpr64(static_cast<u64>(0x7FFFFFFFFFFFFFFFULL)); }

  ALWAYS_INLINE GSVector4 neg64() const { return *this ^ GSVector4::cxpr64(static_cast<u64>(0x8000000000000000ULL)); }

  ALWAYS_INLINE GSVector4 sqrt64() const { return from_f64(vec_sqrt(as_f64())); }

  ALWAYS_INLINE GSVector4 sqr64() const { return from_f64(vec_mul(as_f64(), as_f64())); }

  ALWAYS_INLINE GSVector4 floor64() const { return from_f64(vec_floor(as_f64())); }

  ALWAYS_INLINE static GSVector4 f32to64(const GSVector4& v)
  {
    return f64(static_cast<double>(v.F32[0]), static_cast<double>(v.F32[1]));
  }

  ALWAYS_INLINE static GSVector4 f32to64(const void* p)
  {
    float f[2];
    std::memcpy(f, p, sizeof(f));
    return f64(static_cast<double>(f[0]), static_cast<double>(f[1]));
  }

  ALWAYS_INLINE GSVector4i f64toi32() const
  {
    return GSVector4i(static_cast<s32>(F64[0]), static_cast<s32>(F64[1]), 0, 0);
  }

  ALWAYS_INLINE GSVector2 xy() const { return GSVector2(x, y); }

  ALWAYS_INLINE GSVector2 zw() const { return GSVector2(z, w); }

  ALWAYS_INLINE static GSVector4 xyxy(const GSVector2& l, const GSVector2& h) { return GSVector4(l.x, l.y, h.x, h.y); }

  ALWAYS_INLINE static GSVector4 xyxy(const GSVector2& l) { return GSVector4(l.x, l.y, l.x, l.y); }

#define VECTOR4_SHUFFLE_4(xs, xn, ys, yn, zs, zn, ws, wn)                                                              \
  ALWAYS_INLINE GSVector4 xs##ys##zs##ws() const                                                                       \
  {                                                                                                                    \
    return GSVector4(__builtin_shufflevector(v4s, v4s, xn, yn, zn, wn));                                               \
  }                                                                                                                    \
  ALWAYS_INLINE GSVector4 xs##ys##zs##ws(const GSVector4& v) const                                                     \
  {                                                                                                                    \
    return GSVector4(__builtin_shufflevector(v4s, v.v4s, xn, yn, 4 + zn, 4 + wn));                                     \
  }

#define VECTOR4_SHUFFLE_3(xs, xn, ys, yn, zs, zn)                                                                      \
  VECTOR4_SHUFFLE_4(xs, xn, ys, yn, zs, zn, x, 0);                                                                     \
  VECTOR4_SHUFFLE_4(xs, xn, ys, yn, zs, zn, y, 1);                                                                     \
  VECTOR4_SHUFFLE_4(xs, xn, ys, yn, zs, zn, z, 2);                                                                     \
  VECTOR4_SHUFFLE_4(xs, xn, ys, yn, zs, zn, w, 3);

#define VECTOR4_SHUFFLE_2(xs, xn, ys, yn)                                                                              \
  VECTOR4_SHUFFLE_3(xs, xn, ys, yn, x, 0);                                                                             \
  VECTOR4_SHUFFLE_3(xs, xn, ys, yn, y, 1);                                                                             \
  VECTOR4_SHUFFLE_3(xs, xn, ys, yn, z, 2);                                                                             \
  VECTOR4_SHUFFLE_3(xs, xn, ys, yn, w, 3);

#define VECTOR4_SHUFFLE_1(xs, xn)                                                                                      \
  VECTOR4_SHUFFLE_2(xs, xn, x, 0);                                                                                     \
  VECTOR4_SHUFFLE_2(xs, xn, y, 1);                                                                                     \
  VECTOR4_SHUFFLE_2(xs, xn, z, 2);                                                                                     \
  VECTOR4_SHUFFLE_2(xs, xn, w, 3);

  VECTOR4_SHUFFLE_1(x, 0);
  VECTOR4_SHUFFLE_1(y, 1);
  VECTOR4_SHUFFLE_1(z, 2);
  VECTOR4_SHUFFLE_1(w, 3);

#undef VECTOR4_SHUFFLE_1
#undef VECTOR4_SHUFFLE_2
#undef VECTOR4_SHUFFLE_3
#undef VECTOR4_SHUFFLE_4

  ALWAYS_INLINE GSVector4 broadcast32() const { return xxxx(); }

  ALWAYS_INLINE static GSVector4 broadcast32(const GSVector4& v) { return v.xxxx(); }

  ALWAYS_INLINE static GSVector4 broadcast32(const void* f)
  {
    float val;
    std::memcpy(&val, f, sizeof(val));
    return GSVector4(val);
  }

  ALWAYS_INLINE static GSVector4 broadcast64(const void* f)
  {
    u64 val;
    std::memcpy(&val, f, sizeof(val));
    return GSVector4(reinterpret_cast<gsv_f32>(gsv_u64{val, val}));
  }
};

ALWAYS_INLINE GSVector2i::GSVector2i(const GSVector2& v)
{
  x = static_cast<s32>(v.x);
  y = static_cast<s32>(v.y);
}

ALWAYS_INLINE GSVector2::GSVector2(const GSVector2i& v)
{
  x = static_cast<float>(v.x);
  y = static_cast<float>(v.y);
}

ALWAYS_INLINE GSVector2i GSVector2i::cast(const GSVector2& v)
{
  GSVector2i ret;
  std::memcpy(&ret.S32, &v.F32, sizeof(ret.S32));
  return ret;
}

ALWAYS_INLINE GSVector2 GSVector2::cast(const GSVector2i& v)
{
  GSVector2 ret;
  std::memcpy(&ret, &v, sizeof(ret));
  return ret;
}

ALWAYS_INLINE GSVector4i::GSVector4i(const GSVector4& v)
{
  // Truncating, saturating conversion like vcvtq_s32_f32().
  v4s = vec_cts(v.v4s, 0);
}

ALWAYS_INLINE GSVector4::GSVector4(const GSVector4i& v)
{
  v4s = vec_ctf(v.v4s, 0);
}

ALWAYS_INLINE GSVector4i GSVector4i::cast(const GSVector4& v)
{
  GSVector4i ret;
  std::memcpy(&ret.S32, &v.F32, sizeof(ret.S32));
  return ret;
}

ALWAYS_INLINE GSVector4 GSVector4::cast(const GSVector4i& v)
{
  GSVector4 ret;
  std::memcpy(&ret.F32, &v.S32, sizeof(ret.F32));
  return ret;
}

#undef SSATURATE8
#undef USATURATE8
#undef SSATURATE16
#undef USATURATE16
#undef ALL_LANES_8
#undef ALL_LANES_16
#undef ALL_LANES_32
#undef ALL_LANES_64
