// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

// Minimal Power ISA 2.07 (POWER8) little-endian instruction emitter for the ppc64le recompiler.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace ppcemit {

enum GPR : uint8_t
{
  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31,
};

/// Condition on a CR field, as tested by bc/isel.
enum class Cond : uint8_t
{
  LT, GE, GT, LE, EQ, NE, SO, NS,
};

enum class FixupKind : uint8_t
{
  Branch24, // I-form, +/- 32MB
  Branch14, // B-form, +/- 32KB
};

class Label
{
public:
  Label() = default;
  Label(const Label&) = delete;
  Label& operator=(const Label&) = delete;

  bool IsBound() const { return m_offset >= 0; }

private:
  friend class Assembler;
  std::ptrdiff_t m_offset = -1;
  std::vector<std::pair<std::ptrdiff_t, FixupKind>> m_fixups;
};

[[noreturn]] inline void Fail(const char* msg)
{
  std::fputs(msg, stderr);
  std::abort();
}

inline constexpr bool IsSImm16(int64_t v) { return v >= -32768 && v <= 32767; }
inline constexpr bool IsUImm16(uint64_t v) { return v <= 0xFFFF; }

/// High-adjusted half for addis/addi pairs, so that (ha << 16) + (s16)lo == value.
inline constexpr int32_t HA16(int32_t v) { return static_cast<int16_t>(static_cast<uint32_t>(v + 0x8000) >> 16); }
inline constexpr int32_t LO16(int32_t v) { return static_cast<int16_t>(static_cast<uint32_t>(v) & 0xFFFF); }

class Assembler
{
public:
  Assembler() = default;
  Assembler(uint8_t* buf, size_t size) { Reset(buf, size); }

  void Reset(uint8_t* buf, size_t size)
  {
    buffer = buf;
    cursor = buf;
    end = buf + size;
  }

  size_t GetSize() const { return static_cast<size_t>(cursor - buffer); }
  size_t GetRemaining() const { return static_cast<size_t>(end - cursor); }

  void Emit(uint32_t insn)
  {
    if (cursor + 4 > end) [[unlikely]]
      Fail("ppcemit: buffer overflow\n");
    std::memcpy(cursor, &insn, sizeof(insn));
    cursor += 4;
  }

  //////////////////////////////////////////////////////////////////////////
  // Encoding helpers
  //////////////////////////////////////////////////////////////////////////
  static constexpr uint32_t D(uint32_t op, uint32_t rt, uint32_t ra, uint32_t imm)
  {
    return (op << 26) | (rt << 21) | (ra << 16) | (imm & 0xFFFFu);
  }
  static constexpr uint32_t DS(uint32_t op, uint32_t rt, uint32_t ra, int32_t ds, uint32_t xo)
  {
    return (op << 26) | (rt << 21) | (ra << 16) | (static_cast<uint32_t>(ds) & 0xFFFCu) | xo;
  }
  static constexpr uint32_t X(uint32_t rt, uint32_t ra, uint32_t rb, uint32_t xo, uint32_t rc = 0)
  {
    return (31u << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc;
  }
  static constexpr uint32_t XO(uint32_t rt, uint32_t ra, uint32_t rb, uint32_t xo, uint32_t oe = 0, uint32_t rc = 0)
  {
    return (31u << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (oe << 10) | (xo << 1) | rc;
  }
  static constexpr uint32_t M(uint32_t op, uint32_t rs, uint32_t ra, uint32_t sh, uint32_t mb, uint32_t me, uint32_t rc)
  {
    return (op << 26) | (rs << 21) | (ra << 16) | ((sh & 31u) << 11) | ((mb & 31u) << 6) | ((me & 31u) << 1) | rc;
  }
  static constexpr uint32_t MD(uint32_t rs, uint32_t ra, uint32_t sh, uint32_t mb, uint32_t xo, uint32_t rc = 0)
  {
    const uint32_t mbf = ((mb & 31u) << 1) | ((mb >> 5) & 1u);
    return (30u << 26) | (rs << 21) | (ra << 16) | ((sh & 31u) << 11) | (mbf << 5) | (xo << 2) | (((sh >> 5) & 1u) << 1) |
           rc;
  }

  static constexpr std::pair<uint32_t, uint32_t> BOBI(Cond c, uint32_t cr)
  {
    // BO=12: branch if CR bit set, BO=4: branch if CR bit clear. BI = cr*4 + {LT=0,GT=1,EQ=2,SO=3}.
    switch (c)
    {
      case Cond::LT: return {12, cr * 4 + 0};
      case Cond::GE: return {4, cr * 4 + 0};
      case Cond::GT: return {12, cr * 4 + 1};
      case Cond::LE: return {4, cr * 4 + 1};
      case Cond::EQ: return {12, cr * 4 + 2};
      case Cond::NE: return {4, cr * 4 + 2};
      case Cond::SO: return {12, cr * 4 + 3};
      case Cond::NS: default: return {4, cr * 4 + 3};
    }
  }

  static constexpr Cond Invert(Cond c) { return static_cast<Cond>(static_cast<uint8_t>(c) ^ 1u); }

  //////////////////////////////////////////////////////////////////////////
  // Integer arithmetic
  //////////////////////////////////////////////////////////////////////////
  void addi(GPR rt, GPR ra, int32_t si) { CheckS16(si); Emit(D(14, rt, ra, static_cast<uint32_t>(si))); }
  void addis(GPR rt, GPR ra, int32_t si) { CheckS16(si); Emit(D(15, rt, ra, static_cast<uint32_t>(si))); }
  void li(GPR rt, int32_t si) { addi(rt, r0, si); }
  void lis(GPR rt, int32_t si) { addis(rt, r0, si); }
  void subfic(GPR rt, GPR ra, int32_t si) { CheckS16(si); Emit(D(8, rt, ra, static_cast<uint32_t>(si))); }
  void mulli(GPR rt, GPR ra, int32_t si) { CheckS16(si); Emit(D(7, rt, ra, static_cast<uint32_t>(si))); }

  void add(GPR rt, GPR ra, GPR rb) { Emit(XO(rt, ra, rb, 266)); }
  void subf(GPR rt, GPR ra, GPR rb) { Emit(XO(rt, ra, rb, 40)); } // rt = rb - ra
  void sub(GPR rt, GPR ra, GPR rb) { subf(rt, rb, ra); }           // rt = ra - rb
  void neg(GPR rt, GPR ra) { Emit(XO(rt, ra, r0, 104)); }
  void mullw(GPR rt, GPR ra, GPR rb) { Emit(XO(rt, ra, rb, 235)); }
  void mulhw(GPR rt, GPR ra, GPR rb) { Emit(XO(rt, ra, rb, 75)); }
  void mulhwu(GPR rt, GPR ra, GPR rb) { Emit(XO(rt, ra, rb, 11)); }
  void mulld(GPR rt, GPR ra, GPR rb) { Emit(XO(rt, ra, rb, 233)); }
  void divw(GPR rt, GPR ra, GPR rb) { Emit(XO(rt, ra, rb, 491)); }
  void divwu(GPR rt, GPR ra, GPR rb) { Emit(XO(rt, ra, rb, 459)); }

  //////////////////////////////////////////////////////////////////////////
  // Logical
  //////////////////////////////////////////////////////////////////////////
  void ori(GPR ra, GPR rs, uint32_t ui) { CheckU16(ui); Emit(D(24, rs, ra, ui)); }
  void oris(GPR ra, GPR rs, uint32_t ui) { CheckU16(ui); Emit(D(25, rs, ra, ui)); }
  void xori(GPR ra, GPR rs, uint32_t ui) { CheckU16(ui); Emit(D(26, rs, ra, ui)); }
  void xoris(GPR ra, GPR rs, uint32_t ui) { CheckU16(ui); Emit(D(27, rs, ra, ui)); }
  void andi_(GPR ra, GPR rs, uint32_t ui) { CheckU16(ui); Emit(D(28, rs, ra, ui)); }  // andi.
  void andis_(GPR ra, GPR rs, uint32_t ui) { CheckU16(ui); Emit(D(29, rs, ra, ui)); } // andis.

  void and_(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 28)); }
  void andc(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 60)); }
  void or_(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 444)); }
  void orc(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 412)); }
  void xor_(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 316)); }
  void nor(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 124)); }
  void not_(GPR ra, GPR rs) { nor(ra, rs, rs); }
  void mr(GPR ra, GPR rs) { or_(ra, rs, rs); }
  void nop() { Emit(0x60000000u); }

  void extsb(GPR ra, GPR rs) { Emit(X(rs, ra, r0, 954)); }
  void extsh(GPR ra, GPR rs) { Emit(X(rs, ra, r0, 922)); }
  void extsw(GPR ra, GPR rs) { Emit(X(rs, ra, r0, 986)); }
  void cntlzw(GPR ra, GPR rs) { Emit(X(rs, ra, r0, 26)); }

  //////////////////////////////////////////////////////////////////////////
  // Shifts and rotates
  //////////////////////////////////////////////////////////////////////////
  void slw(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 24)); }
  void srw(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 536)); }
  void sraw(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 792)); }
  void srawi(GPR ra, GPR rs, uint32_t sh) { Emit(X(rs, ra, static_cast<GPR>(sh & 31u), 824)); }
  void sld(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 27)); }
  void srd(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 539)); }
  void srad(GPR ra, GPR rs, GPR rb) { Emit(X(rs, ra, rb, 794)); }
  void sradi(GPR ra, GPR rs, uint32_t sh)
  {
    Emit((31u << 26) | (static_cast<uint32_t>(rs) << 21) | (static_cast<uint32_t>(ra) << 16) | ((sh & 31u) << 11) |
         (413u << 2) | (((sh >> 5) & 1u) << 1));
  }

  void rlwinm(GPR ra, GPR rs, uint32_t sh, uint32_t mb, uint32_t me) { Emit(M(21, rs, ra, sh, mb, me, 0)); }
  void rlwinm_(GPR ra, GPR rs, uint32_t sh, uint32_t mb, uint32_t me) { Emit(M(21, rs, ra, sh, mb, me, 1)); }
  void rlwimi(GPR ra, GPR rs, uint32_t sh, uint32_t mb, uint32_t me) { Emit(M(20, rs, ra, sh, mb, me, 0)); }
  void rlwnm(GPR ra, GPR rs, GPR rb, uint32_t mb, uint32_t me) { Emit(M(23, rs, ra, rb, mb, me, 0)); }
  void rldicl(GPR ra, GPR rs, uint32_t sh, uint32_t mb) { Emit(MD(rs, ra, sh, mb, 0)); }
  void rldicr(GPR ra, GPR rs, uint32_t sh, uint32_t me) { Emit(MD(rs, ra, sh, me, 1)); }
  void rldic(GPR ra, GPR rs, uint32_t sh, uint32_t mb) { Emit(MD(rs, ra, sh, mb, 2)); }
  void rldimi(GPR ra, GPR rs, uint32_t sh, uint32_t mb) { Emit(MD(rs, ra, sh, mb, 3)); }

  // Pseudo-ops. The 32-bit forms (rlwinm-based) zero the upper word of the result.
  void slwi(GPR ra, GPR rs, uint32_t n) { rlwinm(ra, rs, n, 0, 31 - n); }
  void srwi(GPR ra, GPR rs, uint32_t n) { rlwinm(ra, rs, (32 - n) & 31u, n, 31); }
  void clrlwi(GPR ra, GPR rs, uint32_t n) { rlwinm(ra, rs, 0, n, 31); }
  void clrrwi(GPR ra, GPR rs, uint32_t n) { rlwinm(ra, rs, 0, 0, 31 - n); }
  void sldi(GPR ra, GPR rs, uint32_t n) { rldicr(ra, rs, n, 63 - n); }
  void srdi(GPR ra, GPR rs, uint32_t n) { rldicl(ra, rs, (64 - n) & 63u, n); }
  void clrldi(GPR ra, GPR rs, uint32_t n) { rldicl(ra, rs, 0, n); }

  //////////////////////////////////////////////////////////////////////////
  // Compare and select
  //////////////////////////////////////////////////////////////////////////
  void cmpw(GPR ra, GPR rb, uint32_t cr = 0) { Emit(X(cr << 2, ra, rb, 0)); }
  void cmplw(GPR ra, GPR rb, uint32_t cr = 0) { Emit(X(cr << 2, ra, rb, 32)); }
  void cmpd(GPR ra, GPR rb, uint32_t cr = 0) { Emit(X((cr << 2) | 1u, ra, rb, 0)); }
  void cmpld(GPR ra, GPR rb, uint32_t cr = 0) { Emit(X((cr << 2) | 1u, ra, rb, 32)); }
  void cmpwi(GPR ra, int32_t si, uint32_t cr = 0) { CheckS16(si); Emit(D(11, cr << 2, ra, static_cast<uint32_t>(si))); }
  void cmpdi(GPR ra, int32_t si, uint32_t cr = 0) { CheckS16(si); Emit(D(11, (cr << 2) | 1u, ra, static_cast<uint32_t>(si))); }
  void cmplwi(GPR ra, uint32_t ui, uint32_t cr = 0) { CheckU16(ui); Emit(D(10, cr << 2, ra, ui)); }
  void cmpldi(GPR ra, uint32_t ui, uint32_t cr = 0) { CheckU16(ui); Emit(D(10, (cr << 2) | 1u, ra, ui)); }

  /// rt = cond ? ra : rb. NOTE: ra == r0 reads as zero.
  void isel(GPR rt, GPR ra, GPR rb, Cond c, uint32_t cr = 0)
  {
    const auto [bo, bi] = BOBI(c, cr);
    if (bo == 4)
      std::swap(ra, rb);
    Emit((31u << 26) | (static_cast<uint32_t>(rt) << 21) | (static_cast<uint32_t>(ra) << 16) |
         (static_cast<uint32_t>(rb) << 11) | (bi << 6) | (15u << 1));
  }

  //////////////////////////////////////////////////////////////////////////
  // Loads and stores. NOTE: ra == r0 means zero, not r0.
  //////////////////////////////////////////////////////////////////////////
  void lbz(GPR rt, int32_t d, GPR ra) { CheckS16(d); Emit(D(34, rt, ra, static_cast<uint32_t>(d))); }
  void lhz(GPR rt, int32_t d, GPR ra) { CheckS16(d); Emit(D(40, rt, ra, static_cast<uint32_t>(d))); }
  void lha(GPR rt, int32_t d, GPR ra) { CheckS16(d); Emit(D(42, rt, ra, static_cast<uint32_t>(d))); }
  void lwz(GPR rt, int32_t d, GPR ra) { CheckS16(d); Emit(D(32, rt, ra, static_cast<uint32_t>(d))); }
  void lwa(GPR rt, int32_t d, GPR ra) { CheckDS(d); Emit(DS(58, rt, ra, d, 2)); }
  void ld(GPR rt, int32_t d, GPR ra) { CheckDS(d); Emit(DS(58, rt, ra, d, 0)); }
  void stb(GPR rs, int32_t d, GPR ra) { CheckS16(d); Emit(D(38, rs, ra, static_cast<uint32_t>(d))); }
  void sth(GPR rs, int32_t d, GPR ra) { CheckS16(d); Emit(D(44, rs, ra, static_cast<uint32_t>(d))); }
  void stw(GPR rs, int32_t d, GPR ra) { CheckS16(d); Emit(D(36, rs, ra, static_cast<uint32_t>(d))); }
  void std_(GPR rs, int32_t d, GPR ra) { CheckDS(d); Emit(DS(62, rs, ra, d, 0)); }
  void stdu(GPR rs, int32_t d, GPR ra) { CheckDS(d); Emit(DS(62, rs, ra, d, 1)); }

  void lbzx(GPR rt, GPR ra, GPR rb) { Emit(X(rt, ra, rb, 87)); }
  void lhzx(GPR rt, GPR ra, GPR rb) { Emit(X(rt, ra, rb, 279)); }
  void lhax(GPR rt, GPR ra, GPR rb) { Emit(X(rt, ra, rb, 343)); }
  void lwzx(GPR rt, GPR ra, GPR rb) { Emit(X(rt, ra, rb, 23)); }
  void ldx(GPR rt, GPR ra, GPR rb) { Emit(X(rt, ra, rb, 21)); }
  void stbx(GPR rs, GPR ra, GPR rb) { Emit(X(rs, ra, rb, 215)); }
  void sthx(GPR rs, GPR ra, GPR rb) { Emit(X(rs, ra, rb, 407)); }
  void stwx(GPR rs, GPR ra, GPR rb) { Emit(X(rs, ra, rb, 151)); }
  void stdx(GPR rs, GPR ra, GPR rb) { Emit(X(rs, ra, rb, 149)); }

  //////////////////////////////////////////////////////////////////////////
  // Special registers and barriers
  //////////////////////////////////////////////////////////////////////////
  void mtctr(GPR rs) { Emit(0x7C0903A6u | (static_cast<uint32_t>(rs) << 21)); }
  void mfctr(GPR rt) { Emit(0x7C0902A6u | (static_cast<uint32_t>(rt) << 21)); }
  void mtlr(GPR rs) { Emit(0x7C0803A6u | (static_cast<uint32_t>(rs) << 21)); }
  void mflr(GPR rt) { Emit(0x7C0802A6u | (static_cast<uint32_t>(rt) << 21)); }
  void mfcr(GPR rt) { Emit(0x7C000026u | (static_cast<uint32_t>(rt) << 21)); }
  void mtcrf(uint32_t fxm, GPR rs) { Emit((31u << 26) | (static_cast<uint32_t>(rs) << 21) | ((fxm & 0xFFu) << 12) | (144u << 1)); }
  void sync() { Emit(0x7C0004ACu); }
  void isync() { Emit(0x4C00012Cu); }
  void trap() { Emit(0x7FE00008u); }

  //////////////////////////////////////////////////////////////////////////
  // Branches
  //////////////////////////////////////////////////////////////////////////
  void blr() { Emit(0x4E800020u); }
  void bctr() { Emit(0x4E800420u); }
  void bctrl() { Emit(0x4E800421u); }

  static bool IsInBranch24Range(const void* from, const void* to)
  {
    const std::ptrdiff_t disp = static_cast<const uint8_t*>(to) - static_cast<const uint8_t*>(from);
    return (disp >= -0x2000000 && disp < 0x2000000 && (disp & 3) == 0);
  }
  static bool IsInBranch14Range(const void* from, const void* to)
  {
    const std::ptrdiff_t disp = static_cast<const uint8_t*>(to) - static_cast<const uint8_t*>(from);
    return (disp >= -0x8000 && disp < 0x8000 && (disp & 3) == 0);
  }

  static uint32_t EncodeB(const void* from, const void* to, bool link)
  {
    if (!IsInBranch24Range(from, to)) [[unlikely]]
      Fail("ppcemit: b target out of range\n");
    const std::ptrdiff_t disp = static_cast<const uint8_t*>(to) - static_cast<const uint8_t*>(from);
    return (18u << 26) | (static_cast<uint32_t>(disp) & 0x03FFFFFCu) | (link ? 1u : 0u);
  }

  /// Unconditional branch to an absolute address within +/- 32MB.
  void b(const void* target) { Emit(EncodeB(cursor, target, false)); }
  void bl(const void* target) { Emit(EncodeB(cursor, target, true)); }

  /// Conditional branch to an absolute address within +/- 32KB.
  void bc(Cond c, const void* target, uint32_t cr = 0)
  {
    if (!IsInBranch14Range(cursor, target)) [[unlikely]]
      Fail("ppcemit: bc target out of range\n");
    const std::ptrdiff_t disp = static_cast<const uint8_t*>(target) - cursor;
    const auto [bo, bi] = BOBI(c, cr);
    Emit((16u << 26) | (bo << 21) | (bi << 16) | (static_cast<uint32_t>(disp) & 0xFFFCu));
  }

  void b(Label* label) { EmitLabelBranch(label, (18u << 26), FixupKind::Branch24); }
  void bc(Cond c, Label* label, uint32_t cr = 0)
  {
    const auto [bo, bi] = BOBI(c, cr);
    EmitLabelBranch(label, (16u << 26) | (bo << 21) | (bi << 16), FixupKind::Branch14);
  }

  void Bind(Label* label)
  {
    if (label->IsBound()) [[unlikely]]
      Fail("ppcemit: label bound twice\n");
    label->m_offset = cursor - buffer;
    for (const auto& [pos, kind] : label->m_fixups)
      PatchBranch(buffer + pos, cursor, kind);
    label->m_fixups.clear();
  }

  /// Rewrites the branch at `where` to point at `target`.
  static void PatchBranch(uint8_t* where, const void* target, FixupKind kind)
  {
    uint32_t insn;
    std::memcpy(&insn, where, sizeof(insn));
    const std::ptrdiff_t disp = static_cast<const uint8_t*>(target) - where;
    if (kind == FixupKind::Branch24)
    {
      if (!IsInBranch24Range(where, target)) [[unlikely]]
        Fail("ppcemit: label out of b range\n");
      insn = (insn & ~0x03FFFFFCu) | (static_cast<uint32_t>(disp) & 0x03FFFFFCu);
    }
    else
    {
      if (!IsInBranch14Range(where, target)) [[unlikely]]
        Fail("ppcemit: label out of bc range\n");
      insn = (insn & ~0xFFFCu) | (static_cast<uint32_t>(disp) & 0xFFFCu);
    }
    std::memcpy(where, &insn, sizeof(insn));
  }

  //////////////////////////////////////////////////////////////////////////
  // Constant materialization
  //////////////////////////////////////////////////////////////////////////

  /// Loads a 32-bit value. The upper word of the result is the sign extension of bit 31.
  void LoadImm32(GPR rt, uint32_t value)
  {
    const int32_t sv = static_cast<int32_t>(value);
    if (IsSImm16(sv))
    {
      li(rt, sv);
      return;
    }
    lis(rt, static_cast<int16_t>(value >> 16));
    if (value & 0xFFFFu)
      ori(rt, rt, value & 0xFFFFu);
  }

  /// Loads a 32-bit value zero-extended to 64 bits.
  void LoadImm32ZeroExt(GPR rt, uint32_t value)
  {
    LoadImm32(rt, value);
    if (value & 0x80000000u)
      clrldi(rt, rt, 32);
  }

  void LoadImm64(GPR rt, uint64_t value)
  {
    const int64_t sv = static_cast<int64_t>(value);
    if (sv >= INT32_MIN && sv <= INT32_MAX)
    {
      LoadImm32(rt, static_cast<uint32_t>(value));
      return;
    }
    if ((value >> 32) == 0)
    {
      LoadImm32ZeroExt(rt, static_cast<uint32_t>(value));
      return;
    }
    LoadImm32(rt, static_cast<uint32_t>(value >> 32));
    sldi(rt, rt, 32);
    if (const uint32_t hi = static_cast<uint32_t>(value >> 16) & 0xFFFFu; hi != 0)
      oris(rt, rt, hi);
    if (const uint32_t lo = static_cast<uint32_t>(value) & 0xFFFFu; lo != 0)
      ori(rt, rt, lo);
  }

private:
  static void CheckS16(int64_t v)
  {
    if (!IsSImm16(v)) [[unlikely]]
      Fail("ppcemit: signed 16-bit immediate out of range\n");
  }
  static void CheckU16(uint64_t v)
  {
    if (!IsUImm16(v)) [[unlikely]]
      Fail("ppcemit: unsigned 16-bit immediate out of range\n");
  }
  static void CheckDS(int64_t v)
  {
    if (!IsSImm16(v) || (v & 3) != 0) [[unlikely]]
      Fail("ppcemit: DS-form displacement out of range or misaligned\n");
  }

  void EmitLabelBranch(Label* label, uint32_t base, FixupKind kind)
  {
    uint8_t* const where = cursor;
    Emit(base);
    if (label->IsBound())
      PatchBranch(where, buffer + label->m_offset, kind);
    else
      label->m_fixups.emplace_back(where - buffer, kind);
  }

public:
  uint8_t* buffer = nullptr;
  uint8_t* cursor = nullptr;
  uint8_t* end = nullptr;
};

} // namespace ppcemit
