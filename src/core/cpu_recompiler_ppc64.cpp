// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#include "cpu_recompiler_ppc64.h"
#include "cpu_code_cache_private.h"
#include "cpu_core_private.h"
#include "cpu_pgxp.h"
#include "gte.h"
#include "settings.h"
#include "timing_event.h"

#include "common/align.h"
#include "common/assert.h"
#include "common/log.h"
#include "common/memmap.h"
#include "common/string_util.h"

#include <limits>

#ifdef CPU_ARCH_PPC64

LOG_CHANNEL(Recompiler);

#define OFFS(x) ((u32)(((u8*)(x)) - ((u8*)&CPU::g_state)))

using namespace ppcemit;

// The code buffer is small enough for every block link to be a single relative branch.
static constexpr u32 BLOCK_LINK_SIZE = 4; // b

// r3-r5 carry arguments, r11/r12 are scratch (r12 also holds the call target), r29 always holds zero.
#define RRET ppcemit::r3
#define RARG1 ppcemit::r3
#define RARG2 ppcemit::r4
#define RARG3 ppcemit::r5
#define RSCRATCH ppcemit::r11
#define RSCRATCH2 ppcemit::r12
#define RZERO ppcemit::r29
#define RMEMBASE ppcemit::r30
#define RSTATE ppcemit::r31

// Stack frame the JIT code runs in, so that callees have an ELFv2 linkage area to store LR/CR into.
static constexpr s32 JIT_STACK_FRAME_SIZE = 256;

// ELFv2 linkage area size, the register save area of thunks goes after it.
static constexpr u32 ELFV2_LINKAGE_AREA_SIZE = 32;

static bool ppcIsCallerSavedRegister(u32 id);
static void ppcMoveAddressToReg(Assembler* ppcAsm, GPR reg, const void* addr);
static void ppcEmitJmp(Assembler* ppcAsm, const void* ptr);
static void ppcEmitCall(Assembler* ppcAsm, const void* ptr);
static void ppcEmitFarLoad(Assembler* ppcAsm, GPR reg, const void* addr);
static void ppcEmitAddImm(Assembler* ppcAsm, GPR rd, GPR rs, s32 imm);
static void ppcEmitStateLoad32(Assembler* ppcAsm, GPR rt, u32 offset);
static void ppcEmitStateStore32(Assembler* ppcAsm, GPR rs, u32 offset);
static void ppcEmitStateLoad8(Assembler* ppcAsm, GPR rt, u32 offset);
static void ppcEmitStateStore8(Assembler* ppcAsm, GPR rs, u32 offset);
static void ppcEmitStateLoad64(Assembler* ppcAsm, GPR rt, u32 offset);
static void ppcEmitSExtB(Assembler* ppcAsm, GPR rd, GPR rs);
static void ppcEmitUExtB(Assembler* ppcAsm, GPR rd, GPR rs);
static void ppcEmitSExtH(Assembler* ppcAsm, GPR rd, GPR rs);
static void ppcEmitUExtH(Assembler* ppcAsm, GPR rd, GPR rs);
static void ppcEmitLUTIndex(Assembler* ppcAsm, GPR rd, GPR rs, u32 shift);

namespace CPU {

using namespace CPU;

PPC64Recompiler s_instance;
Recompiler* g_compiler = &s_instance;

} // namespace CPU

bool ppcIsCallerSavedRegister(u32 id)
{
  return id == 0 || (id >= 3 && id <= 12);
}

void ppcMoveAddressToReg(Assembler* ppcAsm, GPR reg, const void* addr)
{
  DebugAssert(reg != r0);

  // Everything in the binary is within +/- 2GB of the CPU state, which is always in RSTATE.
  const s64 disp = static_cast<s64>(reinterpret_cast<intptr_t>(addr) - reinterpret_cast<intptr_t>(&CPU::g_state));
  if (disp >= (static_cast<s64>(std::numeric_limits<s32>::min()) + 0x8000) &&
      disp <= (static_cast<s64>(std::numeric_limits<s32>::max()) - 0x8000))
  {
    const s32 d = static_cast<s32>(disp);
    ppcAsm->addis(reg, RSTATE, HA16(d));
    if (LO16(d) != 0)
      ppcAsm->addi(reg, reg, LO16(d));
  }
  else
  {
    ppcAsm->LoadImm64(reg, static_cast<u64>(reinterpret_cast<uintptr_t>(addr)));
  }
}

void ppcEmitJmp(Assembler* ppcAsm, const void* ptr)
{
  if (Assembler::IsInBranch24Range(ppcAsm->cursor, ptr))
  {
    ppcAsm->b(ptr);
    return;
  }

  ppcMoveAddressToReg(ppcAsm, RSCRATCH2, ptr);
  ppcAsm->mtctr(RSCRATCH2);
  ppcAsm->bctr();
}

void ppcEmitCall(Assembler* ppcAsm, const void* ptr)
{
  // ELFv2 requires the caller to extend 32-bit arguments. All of our helpers take unsigned 32-bit arguments.
  ppcAsm->clrldi(RARG1, RARG1, 32);
  ppcAsm->clrldi(RARG2, RARG2, 32);
  ppcAsm->clrldi(RARG3, RARG3, 32);

  // Global entry points expect their own address in r12, they derive the TOC from it.
  ppcMoveAddressToReg(ppcAsm, RSCRATCH2, ptr);
  ppcAsm->mtctr(RSCRATCH2);
  ppcAsm->bctrl();
}

void ppcEmitFarLoad(Assembler* ppcAsm, GPR reg, const void* addr)
{
  ppcMoveAddressToReg(ppcAsm, reg, addr);
  ppcAsm->lwz(reg, 0, reg);
}

void ppcEmitAddImm(Assembler* ppcAsm, GPR rd, GPR rs, s32 imm)
{
  DebugAssert(rs != r0);
  if (IsSImm16(imm))
  {
    ppcAsm->addi(rd, rs, imm);
    return;
  }

  ppcAsm->addis(rd, rs, HA16(imm));
  if (LO16(imm) != 0)
    ppcAsm->addi(rd, rd, LO16(imm));
}

void ppcEmitStateLoad32(Assembler* ppcAsm, GPR rt, u32 offset)
{
  if (IsSImm16(offset))
  {
    ppcAsm->lwz(rt, static_cast<s32>(offset), RSTATE);
    return;
  }

  ppcAsm->addis(RSCRATCH2, RSTATE, HA16(static_cast<s32>(offset)));
  ppcAsm->lwz(rt, LO16(static_cast<s32>(offset)), RSCRATCH2);
}

void ppcEmitStateStore32(Assembler* ppcAsm, GPR rs, u32 offset)
{
  if (IsSImm16(offset))
  {
    ppcAsm->stw(rs, static_cast<s32>(offset), RSTATE);
    return;
  }

  DebugAssert(rs != RSCRATCH2);
  ppcAsm->addis(RSCRATCH2, RSTATE, HA16(static_cast<s32>(offset)));
  ppcAsm->stw(rs, LO16(static_cast<s32>(offset)), RSCRATCH2);
}

void ppcEmitStateLoad8(Assembler* ppcAsm, GPR rt, u32 offset)
{
  if (IsSImm16(offset))
  {
    ppcAsm->lbz(rt, static_cast<s32>(offset), RSTATE);
    return;
  }

  ppcAsm->addis(RSCRATCH2, RSTATE, HA16(static_cast<s32>(offset)));
  ppcAsm->lbz(rt, LO16(static_cast<s32>(offset)), RSCRATCH2);
}

void ppcEmitStateStore8(Assembler* ppcAsm, GPR rs, u32 offset)
{
  if (IsSImm16(offset))
  {
    ppcAsm->stb(rs, static_cast<s32>(offset), RSTATE);
    return;
  }

  DebugAssert(rs != RSCRATCH2);
  ppcAsm->addis(RSCRATCH2, RSTATE, HA16(static_cast<s32>(offset)));
  ppcAsm->stb(rs, LO16(static_cast<s32>(offset)), RSCRATCH2);
}

void ppcEmitStateLoad64(Assembler* ppcAsm, GPR rt, u32 offset)
{
  DebugAssert((offset & 3) == 0);
  if (IsSImm16(offset))
  {
    ppcAsm->ld(rt, static_cast<s32>(offset), RSTATE);
    return;
  }

  ppcAsm->addis(RSCRATCH2, RSTATE, HA16(static_cast<s32>(offset)));
  ppcAsm->ld(rt, LO16(static_cast<s32>(offset)), RSCRATCH2);
}

void ppcEmitSExtB(Assembler* ppcAsm, GPR rd, GPR rs)
{
  ppcAsm->extsb(rd, rs);
}

void ppcEmitUExtB(Assembler* ppcAsm, GPR rd, GPR rs)
{
  ppcAsm->clrldi(rd, rs, 56);
}

void ppcEmitSExtH(Assembler* ppcAsm, GPR rd, GPR rs)
{
  ppcAsm->extsh(rd, rs);
}

void ppcEmitUExtH(Assembler* ppcAsm, GPR rd, GPR rs)
{
  ppcAsm->clrldi(rd, rs, 48);
}

void ppcEmitLUTIndex(Assembler* ppcAsm, GPR rd, GPR rs, u32 shift)
{
  // rd = (u32(rs) >> shift) * sizeof(void*), as a single rotate and mask.
  DebugAssert(shift >= 3 && shift < 32);
  ppcAsm->rlwinm(rd, rs, (32 - shift + 3) & 31u, shift - 3, 28);
}

void CPU::CodeCache::DisassembleAndLogHostCode(const void* start, u32 size)
{
  // No in-process disassembler, dump the words so they can be fed to objdump.
  const u32* code = static_cast<const u32*>(start);
  const u32 count = size / 4;
  for (u32 i = 0; i < count; i++)
    INFO_LOG("\t0x{:016X}\t.long 0x{:08X}", reinterpret_cast<uintptr_t>(code + i), code[i]);
}

u32 CPU::CodeCache::GetHostInstructionCount(const void* start, u32 size)
{
  return size / 4;
}

u32 CPU::CodeCache::EmitASMFunctions(void* code, u32 code_size)
{
  Assembler actual_asm(static_cast<u8*>(code), code_size);
  Assembler* ppcAsm = &actual_asm;

  Label dispatch;
  Label run_events_and_dispatch;

  g_recompiler_functions.enter_recompiler =
    reinterpret_cast<decltype(g_recompiler_functions.enter_recompiler)>(ppcAsm->cursor);
  {
    // Give callees a frame to store LR/CR into. We never return, execution leaves through fastjmp.
    ppcAsm->stdu(r1, -JIT_STACK_FRAME_SIZE, r1);

    // Need the CPU state for basically everything :-)
    ppcAsm->LoadImm64(RSTATE, static_cast<u64>(reinterpret_cast<uintptr_t>(&g_state)));
    ppcAsm->li(RZERO, 0);

    // Fastmem setup
    if (IsUsingFastmem())
      ppcEmitStateLoad64(ppcAsm, RMEMBASE, OFFS(&g_state.fastmem_base));

    // Fall through to event dispatcher
  }

  // check events then for frame done
  {
    Label skip_event_check;
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
    ppcEmitStateLoad32(ppcAsm, RARG2, OFFS(&g_state.downcount));
    ppcAsm->cmplw(RARG1, RARG2);
    ppcAsm->bc(Cond::LT, &skip_event_check);

    ppcAsm->Bind(&run_events_and_dispatch);
    g_recompiler_functions.run_events_and_dispatch = ppcAsm->cursor;
    ppcEmitCall(ppcAsm, reinterpret_cast<const void*>(&TimingEvents::RunEvents));

    ppcAsm->Bind(&skip_event_check);
  }

  g_recompiler_functions.dispatcher = ppcAsm->cursor;
  {
    ppcAsm->Bind(&dispatch);

    // RARG2 <- g_code_lut[pc >> 16]
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pc));
    ppcMoveAddressToReg(ppcAsm, RARG3, g_code_lut.data());
    ppcEmitLUTIndex(ppcAsm, RARG2, RARG1, 16);
    ppcAsm->ldx(RARG2, RARG2, RARG3);

    // RARG1 <- ((pc & 0xFFFF) >> 2) * sizeof(void*)
    ppcAsm->rlwinm(RARG1, RARG1, 1, 15, 28);

    // jump to RARG2[RARG1]
    ppcAsm->ldx(RARG1, RARG1, RARG2);
    ppcAsm->mtctr(RARG1);
    ppcAsm->bctr();
  }

  g_recompiler_functions.compile_or_revalidate_block = ppcAsm->cursor;
  {
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pc));
    ppcEmitCall(ppcAsm, reinterpret_cast<const void*>(&CompileOrRevalidateBlock));
    ppcAsm->b(&dispatch);
  }

  g_recompiler_functions.discard_and_recompile_block = ppcAsm->cursor;
  {
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pc));
    ppcEmitCall(ppcAsm, reinterpret_cast<const void*>(&DiscardAndRecompileBlock));
    ppcAsm->b(&dispatch);
  }

  g_recompiler_functions.interpret_block = ppcAsm->cursor;
  {
    ppcEmitCall(ppcAsm, CodeCache::GetInterpretUncachedBlockFunction());
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
    ppcEmitStateLoad32(ppcAsm, RARG2, OFFS(&g_state.downcount));
    ppcAsm->cmpw(RARG1, RARG2);
    ppcAsm->bc(Cond::GE, &run_events_and_dispatch);
    ppcAsm->b(&dispatch);
  }

  return static_cast<u32>(ppcAsm->GetSize());
}

void CPU::CodeCache::EmitAlignmentPadding(void* dst, size_t size)
{
  constexpr u8 padding_value = 0x00;
  std::memset(dst, padding_value, size);
}

u32 CPU::CodeCache::EmitJump(void* code, const void* dst, bool flush_icache)
{
  {
    Assembler assembler(static_cast<u8*>(code), BLOCK_LINK_SIZE);
    assembler.b(dst);
    DebugAssert(assembler.GetSize() == BLOCK_LINK_SIZE);
  }

  if (flush_icache)
    MemMap::FlushInstructionCache(code, BLOCK_LINK_SIZE);

  return BLOCK_LINK_SIZE;
}

CPU::PPC64Recompiler::PPC64Recompiler() = default;

CPU::PPC64Recompiler::~PPC64Recompiler() = default;

const void* CPU::PPC64Recompiler::GetCurrentCodePointer()
{
  return ppcAsm->cursor;
}

void CPU::PPC64Recompiler::Reset(CodeCache::Block* block, u8* code_buffer, u32 code_buffer_space,
                                 u8* far_code_buffer, u32 far_code_space)
{
  Recompiler::Reset(block, code_buffer, code_buffer_space, far_code_buffer, far_code_space);

  DebugAssert(!ppcAsm);
  m_emitter.Reset(code_buffer, code_buffer_space);
  m_far_emitter.Reset(far_code_buffer, far_code_space);
  ppcAsm = &m_emitter;

  // Need to wipe it out so it's correct when toggling fastmem.
  m_host_regs = {};

  const u32 membase_idx = CodeCache::IsUsingFastmem() ? static_cast<u32>(RMEMBASE) : NUM_HOST_REGS;
  for (u32 i = 0; i < NUM_HOST_REGS; i++)
  {
    HostRegAlloc& hra = m_host_regs[i];

    // Reserved: r0, sp(r1), toc(r2), args(r3-r5), scratch(r11/r12), thread pointer(r13), zero(r29), state(r31)
    if (i <= static_cast<u32>(RARG3) || i == static_cast<u32>(RSCRATCH) || i == static_cast<u32>(RSCRATCH2) ||
        i == 13 || i == static_cast<u32>(RZERO) || i == static_cast<u32>(RSTATE) || i == membase_idx)
    {
      continue;
    }

    hra.flags = HR_USABLE | (ppcIsCallerSavedRegister(i) ? 0 : HR_CALLEE_SAVED);
  }
}

void CPU::PPC64Recompiler::SwitchToFarCode(bool emit_jump, std::optional<Cond> cond)
{
  DebugAssert(ppcAsm == &m_emitter);
  if (emit_jump)
  {
    const void* target = m_far_emitter.cursor;
    if (cond.has_value())
    {
      if (Assembler::IsInBranch14Range(ppcAsm->cursor, target))
      {
        ppcAsm->bc(cond.value(), target);
      }
      else
      {
        Label skip;
        ppcAsm->bc(Assembler::Invert(cond.value()), &skip);
        ppcEmitJmp(ppcAsm, target);
        ppcAsm->Bind(&skip);
      }
    }
    else
    {
      ppcEmitJmp(ppcAsm, target);
    }
  }
  ppcAsm = &m_far_emitter;
}

void CPU::PPC64Recompiler::SwitchToNearCode(bool emit_jump)
{
  DebugAssert(ppcAsm == &m_far_emitter);
  if (emit_jump)
    ppcEmitJmp(ppcAsm, m_emitter.cursor);
  ppcAsm = &m_emitter;
}

void CPU::PPC64Recompiler::EmitMov(GPR dst, u32 val)
{
  ppcAsm->LoadImm32(dst, val);
}

void CPU::PPC64Recompiler::EmitCall(const void* ptr)
{
  ppcEmitCall(ppcAsm, ptr);
}

void CPU::PPC64Recompiler::SafeADDIW(GPR rd, GPR rs, u32 imm)
{
  ppcEmitAddImm(ppcAsm, rd, rs, static_cast<s32>(imm));
}

void CPU::PPC64Recompiler::SafeSUBIW(GPR rd, GPR rs, u32 imm)
{
  ppcEmitAddImm(ppcAsm, rd, rs, -static_cast<s32>(imm));
}

void CPU::PPC64Recompiler::SafeANDI(GPR rd, GPR rs, u32 imm)
{
  DebugAssert(rd != RSCRATCH && rs != RSCRATCH);
  if (IsUImm16(imm))
  {
    ppcAsm->andi_(rd, rs, imm);
  }
  else if ((imm & 0xFFFFu) == 0)
  {
    ppcAsm->andis_(rd, rs, imm >> 16);
  }
  else
  {
    EmitMov(RSCRATCH, imm);
    ppcAsm->and_(rd, rs, RSCRATCH);
  }
}

void CPU::PPC64Recompiler::SafeORI(GPR rd, GPR rs, u32 imm)
{
  if (IsUImm16(imm))
  {
    ppcAsm->ori(rd, rs, imm);
    return;
  }

  ppcAsm->oris(rd, rs, imm >> 16);
  if (imm & 0xFFFFu)
    ppcAsm->ori(rd, rd, imm & 0xFFFFu);
}

void CPU::PPC64Recompiler::SafeXORI(GPR rd, GPR rs, u32 imm)
{
  if (IsUImm16(imm))
  {
    ppcAsm->xori(rd, rs, imm);
    return;
  }

  ppcAsm->xoris(rd, rs, imm >> 16);
  if (imm & 0xFFFFu)
    ppcAsm->xori(rd, rd, imm & 0xFFFFu);
}

void CPU::PPC64Recompiler::SafeSLTI(GPR rd, GPR rs, u32 imm)
{
  DebugAssert(rs != RSCRATCH);
  if (IsSImm16(static_cast<s32>(imm)))
  {
    ppcAsm->cmpwi(rs, static_cast<s32>(imm));
  }
  else
  {
    EmitMov(RSCRATCH, imm);
    ppcAsm->cmpw(rs, RSCRATCH);
  }
  ppcAsm->li(RSCRATCH, 1);
  ppcAsm->isel(rd, RSCRATCH, RZERO, Cond::LT);
}

void CPU::PPC64Recompiler::SafeSLTIU(GPR rd, GPR rs, u32 imm)
{
  DebugAssert(rs != RSCRATCH);
  if (IsUImm16(imm))
  {
    ppcAsm->cmplwi(rs, imm);
  }
  else
  {
    EmitMov(RSCRATCH, imm);
    ppcAsm->cmplw(rs, RSCRATCH);
  }
  ppcAsm->li(RSCRATCH, 1);
  ppcAsm->isel(rd, RSCRATCH, RZERO, Cond::LT);
}

void CPU::PPC64Recompiler::EmitSLLI(GPR rd, GPR rs, u32 sh)
{
  ppcAsm->slwi(rd, rs, sh & 31u);
}

void CPU::PPC64Recompiler::EmitSRLI(GPR rd, GPR rs, u32 sh)
{
  ppcAsm->srwi(rd, rs, sh & 31u);
}

void CPU::PPC64Recompiler::EmitSRAI(GPR rd, GPR rs, u32 sh)
{
  ppcAsm->srawi(rd, rs, sh & 31u);
}

void CPU::PPC64Recompiler::EmitSExtB(GPR rd, GPR rs)
{
  ppcEmitSExtB(ppcAsm, rd, rs);
}

void CPU::PPC64Recompiler::EmitUExtB(GPR rd, GPR rs)
{
  ppcEmitUExtB(ppcAsm, rd, rs);
}

void CPU::PPC64Recompiler::EmitSExtH(GPR rd, GPR rs)
{
  ppcEmitSExtH(ppcAsm, rd, rs);
}

void CPU::PPC64Recompiler::EmitUExtH(GPR rd, GPR rs)
{
  ppcEmitUExtH(ppcAsm, rd, rs);
}

void CPU::PPC64Recompiler::GenerateBlockProtectCheck(const u8* ram_ptr, const u8* shadow_ptr, u32 size)
{
  // store it first to reduce code size, because we can offset
  ppcAsm->LoadImm64(RARG1, static_cast<u64>(reinterpret_cast<uintptr_t>(ram_ptr)));
  ppcAsm->LoadImm64(RARG2, static_cast<u64>(reinterpret_cast<uintptr_t>(shadow_ptr)));

  // Conditional branches only reach 32KB, so emit the exit path periodically for large blocks.
  static constexpr u32 COMPARES_PER_EXIT = 512;

  u32 offset = 0;
  u32 compares = 0;
  Label block_changed;
  Label block_unchanged;

  const auto rebase = [this, &offset]() {
    if (offset >= 32000)
    {
      ppcAsm->addi(RARG1, RARG1, static_cast<s32>(offset));
      ppcAsm->addi(RARG2, RARG2, static_cast<s32>(offset));
      offset = 0;
    }
  };
  const auto flush_exit = [this, &compares, &block_changed]() {
    if (++compares < COMPARES_PER_EXIT)
      return;

    Label cont;
    ppcAsm->b(&cont);
    ppcAsm->Bind(&block_changed);
    ppcEmitJmp(ppcAsm, CodeCache::g_recompiler_functions.discard_and_recompile_block);
    ppcAsm->Bind(&cont);
    block_changed.~Label();
    new (&block_changed) Label();
    compares = 0;
  };

  while (size >= 8)
  {
    rebase();
    ppcAsm->ld(RARG3, static_cast<s32>(offset), RARG1);
    ppcAsm->ld(RSCRATCH, static_cast<s32>(offset), RARG2);
    ppcAsm->cmpd(RARG3, RSCRATCH);
    ppcAsm->bc(Cond::NE, &block_changed);
    offset += 8;
    size -= 8;
    flush_exit();
  }

  while (size >= 4)
  {
    rebase();
    ppcAsm->lwz(RARG3, static_cast<s32>(offset), RARG1);
    ppcAsm->lwz(RSCRATCH, static_cast<s32>(offset), RARG2);
    ppcAsm->cmpw(RARG3, RSCRATCH);
    ppcAsm->bc(Cond::NE, &block_changed);
    offset += 4;
    size -= 4;
    flush_exit();
  }

  DebugAssert(size == 0);

  ppcAsm->b(&block_unchanged);
  ppcAsm->Bind(&block_changed);
  ppcEmitJmp(ppcAsm, CodeCache::g_recompiler_functions.discard_and_recompile_block);
  ppcAsm->Bind(&block_unchanged);
}

void CPU::PPC64Recompiler::GenerateICacheCheckAndUpdate()
{
  if (!m_block->HasFlag(CodeCache::BlockFlags::IsUsingICache))
  {
    if (m_block->HasFlag(CodeCache::BlockFlags::NeedsDynamicFetchTicks))
    {
      ppcEmitFarLoad(ppcAsm, RARG2, GetFetchMemoryAccessTimePtr());
      ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
      EmitMov(RARG3, m_block->size);
      ppcAsm->mullw(RARG2, RARG2, RARG3);
      ppcAsm->add(RARG1, RARG1, RARG2);
      ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
    }
    else
    {
      ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
      SafeADDIW(RARG1, RARG1, static_cast<u32>(m_block->uncached_fetch_ticks));
      ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
    }
  }
  else if (m_block->icache_line_count > 0)
  {
    const auto& ticks_reg = RARG1;
    const auto& current_tag_reg = RARG2;
    const auto& existing_tag_reg = RARG3;

    // start of block, nothing should be using this
    const auto& maddr_reg = r6;
    DebugAssert(!IsHostRegAllocated(maddr_reg));

    VirtualMemoryAddress current_pc = m_block->pc & ICACHE_TAG_ADDRESS_MASK;
    ppcEmitStateLoad32(ppcAsm, ticks_reg, OFFS(&g_state.pending_ticks));
    EmitMov(current_tag_reg, current_pc);

    for (u32 i = 0; i < m_block->icache_line_count; i++, current_pc += ICACHE_LINE_SIZE)
    {
      const TickCount fill_ticks = GetICacheFillTicks(current_pc);
      if (fill_ticks <= 0)
        continue;

      const u32 line = GetICacheLine(current_pc);
      const u32 offset = OFFSETOF(State, icache_tags) + (line * sizeof(u32));

      Label cache_hit;
      if (!IsSImm16(offset))
      {
        ppcEmitAddImm(ppcAsm, maddr_reg, RSTATE, static_cast<s32>(offset));
        ppcAsm->lwz(existing_tag_reg, 0, maddr_reg);
        ppcAsm->cmpw(existing_tag_reg, current_tag_reg);
        ppcAsm->bc(Cond::EQ, &cache_hit);
        ppcAsm->stw(current_tag_reg, 0, maddr_reg);
      }
      else
      {
        ppcAsm->lwz(existing_tag_reg, static_cast<s32>(offset), RSTATE);
        ppcAsm->cmpw(existing_tag_reg, current_tag_reg);
        ppcAsm->bc(Cond::EQ, &cache_hit);
        ppcAsm->stw(current_tag_reg, static_cast<s32>(offset), RSTATE);
      }

      SafeADDIW(ticks_reg, ticks_reg, static_cast<u32>(fill_ticks));
      ppcAsm->Bind(&cache_hit);

      if (i != (m_block->icache_line_count - 1))
        SafeADDIW(current_tag_reg, current_tag_reg, ICACHE_LINE_SIZE);
    }

    ppcEmitStateStore32(ppcAsm, ticks_reg, OFFS(&g_state.pending_ticks));
  }
}

void CPU::PPC64Recompiler::GenerateCall(const void* func, s32 arg1reg /*= -1*/, s32 arg2reg /*= -1*/,
                                        s32 arg3reg /*= -1*/)
{
  if (arg1reg >= 0 && arg1reg != static_cast<s32>(RARG1))
    ppcAsm->mr(RARG1, static_cast<GPR>(arg1reg));
  if (arg2reg >= 0 && arg2reg != static_cast<s32>(RARG2))
    ppcAsm->mr(RARG2, static_cast<GPR>(arg2reg));
  if (arg3reg >= 0 && arg3reg != static_cast<s32>(RARG3))
    ppcAsm->mr(RARG3, static_cast<GPR>(arg3reg));
  EmitCall(func);
}

void CPU::PPC64Recompiler::EndBlock(const std::optional<u32>& newpc, bool do_event_test)
{
  if (newpc.has_value())
  {
    if (m_dirty_pc || m_compiler_pc != newpc)
    {
      EmitMov(RSCRATCH, newpc.value());
      ppcEmitStateStore32(ppcAsm, RSCRATCH, OFFS(&g_state.pc));
    }
  }
  m_dirty_pc = false;

  // flush regs
  Flush(FLUSH_END_BLOCK);
  EndAndLinkBlock(newpc, do_event_test, false);
}

void CPU::PPC64Recompiler::EndBlockWithException(Exception excode)
{
  // flush regs, but not pc, it's going to get overwritten
  // flush cycles because of the GTE instruction stuff...
  Flush(FLUSH_END_BLOCK | FLUSH_FOR_EXCEPTION | FLUSH_FOR_C_CALL);

  // TODO: flush load delay

  EmitMov(RARG1, Cop0Registers::CAUSE::MakeValueForException(excode, m_current_instruction_branch_delay_slot, false,
                                                              inst->cop.cop_n));
  EmitMov(RARG2, m_current_instruction_pc);
  if (excode != Exception::BP)
  {
    EmitCall(reinterpret_cast<const void*>(static_cast<void (*)(u32, u32)>(&CPU::RaiseException)));
  }
  else
  {
    EmitMov(RARG3, inst->bits);
    EmitCall(reinterpret_cast<const void*>(&CPU::RaiseBreakException));
  }
  m_dirty_pc = false;

  EndAndLinkBlock(std::nullopt, true, false);
}

void CPU::PPC64Recompiler::EndAndLinkBlock(const std::optional<u32>& newpc, bool do_event_test, bool force_run_events)
{
  // event test
  // pc should've been flushed
  DebugAssert(!m_dirty_pc && !m_block_ended);
  m_block_ended = true;

  // TODO: try extracting this to a function
  // TODO: move the cycle flush in here..

  // save cycles for event test
  const TickCount cycles = std::exchange(m_cycles, 0);

  // pending_ticks += cycles
  // if (pending_ticks >= downcount) { dispatch_event(); }
  if (do_event_test || m_gte_done_cycle > cycles || cycles > 0)
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
  if (do_event_test)
    ppcEmitStateLoad32(ppcAsm, RARG2, OFFS(&g_state.downcount));
  if (cycles > 0)
  {
    SafeADDIW(RARG1, RARG1, cycles);
    ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
  }
  if (m_gte_done_cycle > cycles)
  {
    SafeADDIW(RARG3, RARG1, m_gte_done_cycle - cycles);
    ppcEmitStateStore32(ppcAsm, RARG3, OFFS(&g_state.gte_completion_tick));
  }

  if (do_event_test)
  {
    // TODO: see if we can do a far jump somehow with this..
    Label cont;
    ppcAsm->cmpw(RARG1, RARG2);
    ppcAsm->bc(Cond::LT, &cont);
    ppcEmitJmp(ppcAsm, CodeCache::g_recompiler_functions.run_events_and_dispatch);
    ppcAsm->Bind(&cont);
  }

  // jump to dispatcher or next block
  if (force_run_events)
  {
    ppcEmitJmp(ppcAsm, CodeCache::g_recompiler_functions.run_events_and_dispatch);
  }
  else if (!newpc.has_value())
  {
    ppcEmitJmp(ppcAsm, CodeCache::g_recompiler_functions.dispatcher);
  }
  else
  {
    // Block links are patched later, so they must be exactly BLOCK_LINK_SIZE.
    const void* target = (newpc.value() == m_block->pc) ?
                           CodeCache::CreateSelfBlockLink(m_block, ppcAsm->cursor, ppcAsm->buffer) :
                           CodeCache::CreateBlockLink(m_block, ppcAsm->cursor, newpc.value());
    ppcAsm->b(target);
  }
}

const void* CPU::PPC64Recompiler::EndCompile(u32* code_size, u32* far_code_size)
{
  u8* const code = m_emitter.buffer;
  *code_size = static_cast<u32>(m_emitter.GetSize());
  *far_code_size = static_cast<u32>(m_far_emitter.GetSize());
  ppcAsm = nullptr;
  return code;
}

const char* CPU::PPC64Recompiler::GetHostRegName(u32 reg) const
{
  static constexpr std::array<const char*, 32> reg64_names = {
    {"r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  "r8",  "r9",  "r10",
     "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21",
     "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"}};
  return (reg < reg64_names.size()) ? reg64_names[reg] : "UNKNOWN";
}

void CPU::PPC64Recompiler::LoadHostRegWithConstant(u32 reg, u32 val)
{
  EmitMov(static_cast<GPR>(reg), val);
}

void CPU::PPC64Recompiler::LoadHostRegFromCPUPointer(u32 reg, const void* ptr)
{
  ppcEmitStateLoad32(ppcAsm, static_cast<GPR>(reg), OFFS(ptr));
}

void CPU::PPC64Recompiler::StoreHostRegToCPUPointer(u32 reg, const void* ptr)
{
  ppcEmitStateStore32(ppcAsm, static_cast<GPR>(reg), OFFS(ptr));
}

void CPU::PPC64Recompiler::StoreConstantToCPUPointer(u32 val, const void* ptr)
{
  if (val == 0)
  {
    ppcEmitStateStore32(ppcAsm, RZERO, OFFS(ptr));
    return;
  }

  EmitMov(RSCRATCH, val);
  ppcEmitStateStore32(ppcAsm, RSCRATCH, OFFS(ptr));
}

void CPU::PPC64Recompiler::CopyHostReg(u32 dst, u32 src)
{
  if (src != dst)
    ppcAsm->mr(static_cast<GPR>(dst), static_cast<GPR>(src));
}

void CPU::PPC64Recompiler::AssertRegOrConstS(CompileFlags cf) const
{
  DebugAssert(cf.valid_host_s || cf.const_s);
}

void CPU::PPC64Recompiler::AssertRegOrConstT(CompileFlags cf) const
{
  DebugAssert(cf.valid_host_t || cf.const_t);
}

GPR CPU::PPC64Recompiler::CFGetSafeRegS(CompileFlags cf, GPR temp_reg)
{
  if (cf.valid_host_s)
  {
    return static_cast<GPR>(cf.host_s);
  }
  else if (cf.const_s)
  {
    if (HasConstantRegValue(cf.MipsS(), 0))
      return RZERO;

    EmitMov(temp_reg, GetConstantRegU32(cf.MipsS()));
    return temp_reg;
  }
  else
  {
    WARNING_LOG("Hit memory path in CFGetSafeRegS() for {}", GetRegName(cf.MipsS()));
    ppcEmitStateLoad32(ppcAsm, temp_reg, OFFS(&g_state.regs.r[cf.mips_s]));
    return temp_reg;
  }
}

GPR CPU::PPC64Recompiler::CFGetSafeRegT(CompileFlags cf, GPR temp_reg)
{
  if (cf.valid_host_t)
  {
    return static_cast<GPR>(cf.host_t);
  }
  else if (cf.const_t)
  {
    if (HasConstantRegValue(cf.MipsT(), 0))
      return RZERO;

    EmitMov(temp_reg, GetConstantRegU32(cf.MipsT()));
    return temp_reg;
  }
  else
  {
    WARNING_LOG("Hit memory path in CFGetSafeRegT() for {}", GetRegName(cf.MipsT()));
    ppcEmitStateLoad32(ppcAsm, temp_reg, OFFS(&g_state.regs.r[cf.mips_t]));
    return temp_reg;
  }
}

GPR CPU::PPC64Recompiler::CFGetRegD(CompileFlags cf) const
{
  DebugAssert(cf.valid_host_d);
  return static_cast<GPR>(cf.host_d);
}

GPR CPU::PPC64Recompiler::CFGetRegS(CompileFlags cf) const
{
  DebugAssert(cf.valid_host_s);
  return static_cast<GPR>(cf.host_s);
}

GPR CPU::PPC64Recompiler::CFGetRegT(CompileFlags cf) const
{
  DebugAssert(cf.valid_host_t);
  return static_cast<GPR>(cf.host_t);
}

GPR CPU::PPC64Recompiler::CFGetRegLO(CompileFlags cf) const
{
  DebugAssert(cf.valid_host_lo);
  return static_cast<GPR>(cf.host_lo);
}

GPR CPU::PPC64Recompiler::CFGetRegHI(CompileFlags cf) const
{
  DebugAssert(cf.valid_host_hi);
  return static_cast<GPR>(cf.host_hi);
}

void CPU::PPC64Recompiler::MoveSToReg(GPR dst, CompileFlags cf)
{
  if (cf.valid_host_s)
  {
    if (cf.host_s != dst)
      ppcAsm->mr(dst, static_cast<GPR>(cf.host_s));
  }
  else if (cf.const_s)
  {
    EmitMov(dst, GetConstantRegU32(cf.MipsS()));
  }
  else
  {
    WARNING_LOG("Hit memory path in MoveSToReg() for {}", GetRegName(cf.MipsS()));
    ppcEmitStateLoad32(ppcAsm, dst, OFFS(&g_state.regs.r[cf.mips_s]));
  }
}

void CPU::PPC64Recompiler::MoveTToReg(GPR dst, CompileFlags cf)
{
  if (cf.valid_host_t)
  {
    if (cf.host_t != dst)
      ppcAsm->mr(dst, static_cast<GPR>(cf.host_t));
  }
  else if (cf.const_t)
  {
    EmitMov(dst, GetConstantRegU32(cf.MipsT()));
  }
  else
  {
    WARNING_LOG("Hit memory path in MoveTToReg() for {}", GetRegName(cf.MipsT()));
    ppcEmitStateLoad32(ppcAsm, dst, OFFS(&g_state.regs.r[cf.mips_t]));
  }
}

void CPU::PPC64Recompiler::MoveMIPSRegToReg(GPR dst, Reg reg, bool ignore_load_delays)
{
  DebugAssert(reg < Reg::count);
  if (ignore_load_delays && m_load_delay_register == reg)
  {
    if (m_load_delay_value_register == NUM_HOST_REGS)
      ppcEmitStateLoad32(ppcAsm, dst, OFFS(&g_state.load_delay_value));
    else
      ppcAsm->mr(dst, static_cast<GPR>(m_load_delay_value_register));
  }
  else if (const std::optional<u32> hreg = CheckHostReg(0, Recompiler::HR_TYPE_CPU_REG, reg))
  {
    ppcAsm->mr(dst, static_cast<GPR>(hreg.value()));
  }
  else if (HasConstantReg(reg))
  {
    EmitMov(dst, GetConstantRegU32(reg));
  }
  else
  {
    ppcEmitStateLoad32(ppcAsm, dst, OFFS(&g_state.regs.r[static_cast<u8>(reg)]));
  }
}

void CPU::PPC64Recompiler::GeneratePGXPCallWithMIPSRegs(const void* func, u32 arg1val, Reg arg2reg /* = Reg::count */,
                                                        Reg arg3reg /* = Reg::count */)
{
  DebugAssert(g_settings.gpu_pgxp_enable);

  Flush(FLUSH_FOR_C_CALL);

  if (arg2reg != Reg::count)
    MoveMIPSRegToReg(RARG2, arg2reg);
  if (arg3reg != Reg::count)
    MoveMIPSRegToReg(RARG3, arg3reg);

  EmitMov(RARG1, arg1val);
  EmitCall(func);
}

void CPU::PPC64Recompiler::Flush(u32 flags)
{
  Recompiler::Flush(flags);

  if (flags & FLUSH_PC && m_dirty_pc)
  {
    StoreConstantToCPUPointer(m_compiler_pc, &g_state.pc);
    m_dirty_pc = false;
  }

  if (flags & FLUSH_INSTRUCTION_BITS)
  {
    // This sucks, but it's only used for fallbacks.
    EmitMov(RARG1, inst->bits);
    EmitMov(RARG2, m_current_instruction_pc);
    EmitMov(RARG3, m_current_instruction_branch_delay_slot);
    ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.current_instruction.bits));
    ppcEmitStateStore32(ppcAsm, RARG2, OFFS(&g_state.current_instruction_pc));
    ppcEmitStateStore8(ppcAsm, RARG3, OFFS(&g_state.current_instruction_in_branch_delay_slot));
  }

  if (flags & FLUSH_LOAD_DELAY_FROM_STATE && m_load_delay_dirty)
  {
    // This sucks :(
    // TODO: make it a function?
    ppcEmitStateLoad8(ppcAsm, RARG1, OFFS(&g_state.load_delay_reg));
    ppcEmitStateLoad32(ppcAsm, RARG2, OFFS(&g_state.load_delay_value));
    ppcAsm->slwi(RARG1, RARG1, 2); // *4
    ppcAsm->add(RARG1, RARG1, RSTATE);
    ppcAsm->stw(RARG2, OFFSETOF(CPU::State, regs.r[0]), RARG1);
    ppcAsm->li(RSCRATCH, static_cast<u8>(Reg::count));
    ppcEmitStateStore8(ppcAsm, RSCRATCH, OFFS(&g_state.load_delay_reg));
    m_load_delay_dirty = false;
  }

  if (flags & FLUSH_LOAD_DELAY && m_load_delay_register != Reg::count)
  {
    if (m_load_delay_value_register != NUM_HOST_REGS)
      FreeHostReg(m_load_delay_value_register);

    EmitMov(RSCRATCH, static_cast<u8>(m_load_delay_register));
    ppcEmitStateStore8(ppcAsm, RSCRATCH, OFFS(&g_state.load_delay_reg));
    m_load_delay_register = Reg::count;
    m_load_delay_dirty = true;
  }

  if (flags & FLUSH_GTE_STALL_FROM_STATE && m_dirty_gte_done_cycle)
  {
    // May as well flush cycles while we're here.
    // GTE spanning blocks is very rare, we _could_ disable this for speed.
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
    ppcEmitStateLoad32(ppcAsm, RARG2, OFFS(&g_state.gte_completion_tick));
    if (m_cycles > 0)
    {
      SafeADDIW(RARG1, RARG1, m_cycles);
      ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
      m_cycles = 0;
    }

    // pending_ticks = (gte_completion_tick > pending_ticks) ? (gte_completion_tick + 1) : pending_ticks
    Label no_stall;
    ppcAsm->cmpw(RARG1, RARG2);
    ppcAsm->bc(Cond::GE, &no_stall);
    ppcAsm->addi(RARG1, RARG2, 1);
    ppcAsm->Bind(&no_stall);
    ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
    m_dirty_gte_done_cycle = false;
  }

  if (flags & FLUSH_GTE_DONE_CYCLE && m_gte_done_cycle > m_cycles)
  {
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));

    // update cycles at the same time
    if (flags & FLUSH_CYCLES && m_cycles > 0)
    {
      SafeADDIW(RARG1, RARG1, m_cycles);
      ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
      m_gte_done_cycle -= m_cycles;
      m_cycles = 0;
    }

    SafeADDIW(RARG1, RARG1, m_gte_done_cycle);
    ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.gte_completion_tick));
    m_gte_done_cycle = 0;
    m_dirty_gte_done_cycle = true;
  }

  if (flags & FLUSH_CYCLES && m_cycles > 0)
  {
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
    SafeADDIW(RARG1, RARG1, m_cycles);
    ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.pending_ticks));
    m_gte_done_cycle = std::max<TickCount>(m_gte_done_cycle - m_cycles, 0);
    m_cycles = 0;
  }
}

void CPU::PPC64Recompiler::Compile_Fallback()
{
  WARNING_LOG("Compiling instruction fallback at PC=0x{:08X}, instruction=0x{:08X}", m_current_instruction_pc,
              inst->bits);

  Flush(FLUSH_FOR_INTERPRETER);

  EmitCall(reinterpret_cast<const void*>(&CPU::RecompilerThunks::InterpretInstruction));

  // TODO: make me less garbage
  // TODO: this is wrong, it flushes the load delay on the same cycle when we return.
  // but nothing should be going through here..
  ppcAsm->andi_(RSCRATCH, RRET, 0xFF);
  SwitchToFarCode(true, Cond::NE);
  BackupHostState();
  m_dirty_pc = false;
  EndBlock(std::nullopt, true);
  RestoreHostState();
  SwitchToNearCode(false);

  Label no_load_delay;
  ppcEmitStateLoad8(ppcAsm, RARG1, OFFS(&g_state.next_load_delay_reg));
  ppcAsm->li(RSCRATCH, static_cast<u8>(Reg::count));
  ppcAsm->cmpw(RARG1, RSCRATCH);
  ppcAsm->bc(Cond::EQ, &no_load_delay);
  ppcEmitStateLoad32(ppcAsm, RARG2, OFFS(&g_state.next_load_delay_value));
  ppcEmitStateStore8(ppcAsm, RARG1, OFFS(&g_state.load_delay_reg));
  ppcEmitStateStore32(ppcAsm, RARG2, OFFS(&g_state.load_delay_value));
  ppcEmitStateStore8(ppcAsm, RSCRATCH, OFFS(&g_state.next_load_delay_reg));
  ppcAsm->Bind(&no_load_delay);

  m_load_delay_dirty = EMULATE_LOAD_DELAYS;
}

void CPU::PPC64Recompiler::CheckBranchTarget(GPR pcreg)
{
  if (!g_settings.cpu_recompiler_memory_exceptions)
    return;

  DebugAssert(pcreg != RSCRATCH);
  ppcAsm->andi_(RSCRATCH, pcreg, 0x3);
  SwitchToFarCode(true, Cond::NE);

  BackupHostState();
  StoreHostRegToCPUPointer(pcreg, &g_state.cop0_regs.BadVaddr);
  EndBlockWithBranchTargetException();

  RestoreHostState();
  SwitchToNearCode(false);
}

void CPU::PPC64Recompiler::Compile_jr(CompileFlags cf)
{
  const GPR pcreg = CFGetRegS(cf);
  CheckBranchTarget(pcreg);

  ppcEmitStateStore32(ppcAsm, pcreg, OFFS(&g_state.pc));

  CompileBranchDelaySlot(false);
  EndBlock(std::nullopt, true);
}

void CPU::PPC64Recompiler::Compile_jalr(CompileFlags cf)
{
  const GPR pcreg = CFGetRegS(cf);
  if (MipsD() != Reg::zero)
    SetConstantReg(MipsD(), GetBranchReturnAddress(cf));

  CheckBranchTarget(pcreg);
  ppcEmitStateStore32(ppcAsm, pcreg, OFFS(&g_state.pc));

  CompileBranchDelaySlot(false);
  EndBlock(std::nullopt, true);
}

void CPU::PPC64Recompiler::Compile_bxx(CompileFlags cf, BranchCondition cond)
{
  AssertRegOrConstS(cf);

  const u32 taken_pc = GetConditionalBranchTarget(cf);

  Flush(FLUSH_FOR_BRANCH);

  DebugAssert(cf.valid_host_s);

  // MipsT() here should equal zero for zero branches.
  DebugAssert(cond == BranchCondition::Equal || cond == BranchCondition::NotEqual || cf.MipsT() == Reg::zero);

  Label taken;
  const GPR rs = CFGetRegS(cf);
  switch (cond)
  {
    case BranchCondition::Equal:
    case BranchCondition::NotEqual:
    {
      AssertRegOrConstT(cf);
      if (cf.const_t && HasConstantRegValue(cf.MipsT(), 0))
      {
        ppcAsm->cmpwi(rs, 0);
      }
      else
      {
        const GPR rt = cf.valid_host_t ? CFGetRegT(cf) : RARG1;
        if (!cf.valid_host_t)
          MoveTToReg(RARG1, cf);
        ppcAsm->cmpw(rs, rt);
      }
      ppcAsm->bc((cond == BranchCondition::Equal) ? Cond::EQ : Cond::NE, &taken);
    }
    break;

    case BranchCondition::GreaterThanZero:
    {
      ppcAsm->cmpwi(rs, 0);
      ppcAsm->bc(Cond::GT, &taken);
    }
    break;

    case BranchCondition::GreaterEqualZero:
    {
      ppcAsm->cmpwi(rs, 0);
      ppcAsm->bc(Cond::GE, &taken);
    }
    break;

    case BranchCondition::LessThanZero:
    {
      ppcAsm->cmpwi(rs, 0);
      ppcAsm->bc(Cond::LT, &taken);
    }
    break;

    case BranchCondition::LessEqualZero:
    {
      ppcAsm->cmpwi(rs, 0);
      ppcAsm->bc(Cond::LE, &taken);
    }
    break;
  }

  BackupHostState();
  if (!cf.delay_slot_swapped)
    CompileBranchDelaySlot();

  EndBlock(m_compiler_pc, true);

  ppcAsm->Bind(&taken);

  RestoreHostState();
  if (!cf.delay_slot_swapped)
    CompileBranchDelaySlot();

  EndBlock(taken_pc, true);
}

void CPU::PPC64Recompiler::Compile_addi(CompileFlags cf, bool overflow)
{
  const GPR rs = CFGetRegS(cf);
  const GPR rt = CFGetRegT(cf);
  if (const u32 imm = inst->i.imm_sext32(); imm != 0)
  {
    if (!overflow)
    {
      SafeADDIW(rt, rs, imm);
    }
    else
    {
      // 64-bit sum of the sign-extended operands, compared against the sign-extended 32-bit sum.
      ppcAsm->extsw(RARG1, rs);
      SafeADDIW(RARG1, RARG1, imm);
      SafeADDIW(rt, rs, imm);
      TestOverflow(RARG1, rt, rt);
    }
  }
  else if (rt != rs)
  {
    ppcAsm->mr(rt, rs);
  }
}

void CPU::PPC64Recompiler::Compile_addi(CompileFlags cf)
{
  Compile_addi(cf, g_settings.cpu_recompiler_memory_exceptions);
}

void CPU::PPC64Recompiler::Compile_addiu(CompileFlags cf)
{
  Compile_addi(cf, false);
}

void CPU::PPC64Recompiler::Compile_slti(CompileFlags cf)
{
  Compile_slti(cf, true);
}

void CPU::PPC64Recompiler::Compile_sltiu(CompileFlags cf)
{
  Compile_slti(cf, false);
}

void CPU::PPC64Recompiler::Compile_slti(CompileFlags cf, bool sign)
{
  if (sign)
    SafeSLTI(CFGetRegT(cf), CFGetRegS(cf), inst->i.imm_sext32());
  else
    SafeSLTIU(CFGetRegT(cf), CFGetRegS(cf), inst->i.imm_sext32());
}

void CPU::PPC64Recompiler::Compile_andi(CompileFlags cf)
{
  const GPR rt = CFGetRegT(cf);
  if (const u32 imm = inst->i.imm_zext32(); imm != 0)
    SafeANDI(rt, CFGetRegS(cf), imm);
  else
    EmitMov(rt, 0);
}

void CPU::PPC64Recompiler::Compile_ori(CompileFlags cf)
{
  const GPR rt = CFGetRegT(cf);
  const GPR rs = CFGetRegS(cf);
  if (const u32 imm = inst->i.imm_zext32(); imm != 0)
    SafeORI(rt, rs, imm);
  else if (rt != rs)
    ppcAsm->mr(rt, rs);
}

void CPU::PPC64Recompiler::Compile_xori(CompileFlags cf)
{
  const GPR rt = CFGetRegT(cf);
  const GPR rs = CFGetRegS(cf);
  if (const u32 imm = inst->i.imm_zext32(); imm != 0)
    SafeXORI(rt, rs, imm);
  else if (rt != rs)
    ppcAsm->mr(rt, rs);
}

void CPU::PPC64Recompiler::Compile_shift(CompileFlags cf, void (PPC64Recompiler::*op_const)(GPR, GPR, u32))
{
  const GPR rd = CFGetRegD(cf);
  const GPR rt = CFGetRegT(cf);
  if (inst->r.shamt > 0)
    (this->*op_const)(rd, rt, inst->r.shamt);
  else if (rd != rt)
    ppcAsm->mr(rd, rt);
}

void CPU::PPC64Recompiler::Compile_sll(CompileFlags cf)
{
  Compile_shift(cf, &PPC64Recompiler::EmitSLLI);
}

void CPU::PPC64Recompiler::Compile_srl(CompileFlags cf)
{
  Compile_shift(cf, &PPC64Recompiler::EmitSRLI);
}

void CPU::PPC64Recompiler::Compile_sra(CompileFlags cf)
{
  Compile_shift(cf, &PPC64Recompiler::EmitSRAI);
}

void CPU::PPC64Recompiler::Compile_variable_shift(CompileFlags cf, PpcRRROp op,
                                                  void (PPC64Recompiler::*op_const)(GPR, GPR, u32))
{
  const GPR rd = CFGetRegD(cf);

  AssertRegOrConstS(cf);
  AssertRegOrConstT(cf);

  const GPR rt = cf.valid_host_t ? CFGetRegT(cf) : RARG2;
  if (!cf.valid_host_t)
    MoveTToReg(rt, cf);

  if (cf.const_s)
  {
    if (const u32 shift = GetConstantRegU32(cf.MipsS()); shift != 0)
      (this->*op_const)(rd, rt, shift & 31u);
    else if (rd != rt)
      ppcAsm->mr(rd, rt);
  }
  else
  {
    // MIPS only uses the low 5 bits of the shift amount, PowerPC uses 6.
    ppcAsm->clrlwi(RSCRATCH, CFGetRegS(cf), 27);
    (ppcAsm->*op)(rd, rt, RSCRATCH);
  }
}

void CPU::PPC64Recompiler::Compile_sllv(CompileFlags cf)
{
  Compile_variable_shift(cf, &Assembler::slw, &PPC64Recompiler::EmitSLLI);
}

void CPU::PPC64Recompiler::Compile_srlv(CompileFlags cf)
{
  Compile_variable_shift(cf, &Assembler::srw, &PPC64Recompiler::EmitSRLI);
}

void CPU::PPC64Recompiler::Compile_srav(CompileFlags cf)
{
  Compile_variable_shift(cf, &Assembler::sraw, &PPC64Recompiler::EmitSRAI);
}

void CPU::PPC64Recompiler::Compile_mult(CompileFlags cf, bool sign)
{
  const GPR rs = cf.valid_host_s ? CFGetRegS(cf) : RARG1;
  if (!cf.valid_host_s)
    MoveSToReg(rs, cf);

  const GPR rt = cf.valid_host_t ? CFGetRegT(cf) : RARG2;
  if (!cf.valid_host_t)
    MoveTToReg(rt, cf);

  const GPR lo = CFGetRegLO(cf);
  const GPR hi = CFGetRegHI(cf);

  // High word into scratch first, in case lo/hi alias an input.
  if (sign)
    ppcAsm->mulhw(RSCRATCH, rs, rt);
  else
    ppcAsm->mulhwu(RSCRATCH, rs, rt);
  ppcAsm->mullw(lo, rs, rt);
  ppcAsm->mr(hi, RSCRATCH);
}

void CPU::PPC64Recompiler::Compile_mult(CompileFlags cf)
{
  Compile_mult(cf, true);
}

void CPU::PPC64Recompiler::Compile_multu(CompileFlags cf)
{
  Compile_mult(cf, false);
}

void CPU::PPC64Recompiler::Compile_div(CompileFlags cf)
{
  const GPR rs = cf.valid_host_s ? CFGetRegS(cf) : RARG1;
  if (!cf.valid_host_s)
    MoveSToReg(rs, cf);

  const GPR rt = cf.valid_host_t ? CFGetRegT(cf) : RARG2;
  if (!cf.valid_host_t)
    MoveTToReg(rt, cf);

  const GPR rlo = CFGetRegLO(cf);
  const GPR rhi = CFGetRegHI(cf);

  // PowerPC leaves the quotient undefined for these cases, so produce the MIPS results explicitly.
  Label done;
  Label not_divide_by_zero;
  ppcAsm->cmpwi(rt, 0);
  ppcAsm->bc(Cond::NE, &not_divide_by_zero);
  ppcAsm->mr(rhi, rs); // hi = num
  ppcAsm->srawi(rlo, rs, 31);
  ppcAsm->andi_(rlo, rlo, 2);
  ppcAsm->addi(rlo, rlo, -1); // lo = s >= 0 ? -1 : 1
  ppcAsm->b(&done);

  ppcAsm->Bind(&not_divide_by_zero);

  Label not_unrepresentable;
  ppcAsm->cmpwi(rt, -1);
  ppcAsm->bc(Cond::NE, &not_unrepresentable);
  ppcAsm->lis(rlo, -32768); // 0x80000000
  ppcAsm->cmpw(rs, rlo);
  ppcAsm->bc(Cond::NE, &not_unrepresentable);
  ppcAsm->li(rhi, 0);
  ppcAsm->b(&done);

  ppcAsm->Bind(&not_unrepresentable);

  // No modsw before POWER9, remainder = num - quotient * denom.
  ppcAsm->divw(RSCRATCH, rs, rt);
  ppcAsm->mullw(RSCRATCH2, RSCRATCH, rt);
  ppcAsm->sub(rhi, rs, RSCRATCH2);
  ppcAsm->mr(rlo, RSCRATCH);

  ppcAsm->Bind(&done);
}

void CPU::PPC64Recompiler::Compile_divu(CompileFlags cf)
{
  const GPR rs = cf.valid_host_s ? CFGetRegS(cf) : RARG1;
  if (!cf.valid_host_s)
    MoveSToReg(rs, cf);

  const GPR rt = cf.valid_host_t ? CFGetRegT(cf) : RARG2;
  if (!cf.valid_host_t)
    MoveTToReg(rt, cf);

  const GPR rlo = CFGetRegLO(cf);
  const GPR rhi = CFGetRegHI(cf);

  Label done;
  Label not_divide_by_zero;
  ppcAsm->cmplwi(rt, 0);
  ppcAsm->bc(Cond::NE, &not_divide_by_zero);
  ppcAsm->mr(rhi, rs);  // hi = num
  ppcAsm->li(rlo, -1); // lo = 0xFFFFFFFF
  ppcAsm->b(&done);

  ppcAsm->Bind(&not_divide_by_zero);
  ppcAsm->divwu(RSCRATCH, rs, rt);
  ppcAsm->mullw(RSCRATCH2, RSCRATCH, rt);
  ppcAsm->sub(rhi, rs, RSCRATCH2);
  ppcAsm->mr(rlo, RSCRATCH);

  ppcAsm->Bind(&done);
}

void CPU::PPC64Recompiler::TestOverflow(GPR long_res, GPR res, GPR reg_to_discard)
{
  ppcAsm->extsw(RSCRATCH, res);
  ppcAsm->cmpd(long_res, RSCRATCH);
  SwitchToFarCode(true, Cond::NE);

  BackupHostState();

  // toss the result
  ClearHostReg(reg_to_discard);

  EndBlockWithException(Exception::Ov);

  RestoreHostState();

  SwitchToNearCode(false);
}

void CPU::PPC64Recompiler::Compile_dst_op(CompileFlags cf, PpcRRROp op,
                                          void (PPC64Recompiler::*op_const)(GPR rd, GPR rs, u32 imm), bool is_sub,
                                          bool commutative, bool overflow)
{
  AssertRegOrConstS(cf);
  AssertRegOrConstT(cf);

  const GPR rd = CFGetRegD(cf);

  if (overflow)
  {
    const GPR rs = CFGetSafeRegS(cf, RARG1);
    const GPR rt = CFGetSafeRegT(cf, RARG2);

    // 64-bit result of the sign-extended operands, compared against the sign-extended 32-bit result.
    ppcAsm->extsw(RARG3, rs);
    ppcAsm->extsw(RSCRATCH, rt);
    if (is_sub)
      ppcAsm->sub(RARG3, RARG3, RSCRATCH);
    else
      ppcAsm->add(RARG3, RARG3, RSCRATCH);
    (ppcAsm->*op)(rd, rs, rt);
    TestOverflow(RARG3, rd, rd);
    return;
  }

  if (cf.valid_host_s && cf.valid_host_t)
  {
    (ppcAsm->*op)(rd, CFGetRegS(cf), CFGetRegT(cf));
  }
  else if (commutative && (cf.const_s || cf.const_t))
  {
    const GPR src = cf.const_s ? CFGetRegT(cf) : CFGetRegS(cf);
    if (const u32 cv = GetConstantRegU32(cf.const_s ? cf.MipsS() : cf.MipsT()); cv != 0)
    {
      (this->*op_const)(rd, src, cv);
    }
    else
    {
      if (rd != src)
        ppcAsm->mr(rd, src);
      overflow = false;
    }
  }
  else if (cf.const_s)
  {
    if (HasConstantRegValue(cf.MipsS(), 0))
    {
      (ppcAsm->*op)(rd, RZERO, CFGetRegT(cf));
    }
    else
    {
      EmitMov(RSCRATCH, GetConstantRegU32(cf.MipsS()));
      (ppcAsm->*op)(rd, RSCRATCH, CFGetRegT(cf));
    }
  }
  else if (cf.const_t)
  {
    const GPR rs = CFGetRegS(cf);
    if (const u32 cv = GetConstantRegU32(cf.const_s ? cf.MipsS() : cf.MipsT()); cv != 0)
    {
      (this->*op_const)(rd, rs, cv);
    }
    else
    {
      if (rd != rs)
        ppcAsm->mr(rd, rs);
      overflow = false;
    }
  }
}

void CPU::PPC64Recompiler::Compile_add(CompileFlags cf)
{
  Compile_dst_op(cf, &Assembler::add, &PPC64Recompiler::SafeADDIW, false, true,
                 g_settings.cpu_recompiler_memory_exceptions);
}

void CPU::PPC64Recompiler::Compile_addu(CompileFlags cf)
{
  Compile_dst_op(cf, &Assembler::add, &PPC64Recompiler::SafeADDIW, false, true, false);
}

void CPU::PPC64Recompiler::Compile_sub(CompileFlags cf)
{
  Compile_dst_op(cf, &Assembler::sub, &PPC64Recompiler::SafeSUBIW, true, false,
                 g_settings.cpu_recompiler_memory_exceptions);
}

void CPU::PPC64Recompiler::Compile_subu(CompileFlags cf)
{
  Compile_dst_op(cf, &Assembler::sub, &PPC64Recompiler::SafeSUBIW, true, false, false);
}

void CPU::PPC64Recompiler::Compile_and(CompileFlags cf)
{
  AssertRegOrConstS(cf);
  AssertRegOrConstT(cf);

  // special cases - and with self -> self, and with 0 -> 0
  const GPR regd = CFGetRegD(cf);
  if (cf.MipsS() == cf.MipsT())
  {
    ppcAsm->mr(regd, CFGetRegS(cf));
    return;
  }
  else if (HasConstantRegValue(cf.MipsS(), 0) || HasConstantRegValue(cf.MipsT(), 0))
  {
    EmitMov(regd, 0);
    return;
  }

  Compile_dst_op(cf, &Assembler::and_, &PPC64Recompiler::SafeANDI, false, true, false);
}

void CPU::PPC64Recompiler::Compile_or(CompileFlags cf)
{
  AssertRegOrConstS(cf);
  AssertRegOrConstT(cf);

  // or/nor with 0 -> no effect
  const GPR regd = CFGetRegD(cf);
  if (HasConstantRegValue(cf.MipsS(), 0) || HasConstantRegValue(cf.MipsT(), 0) || cf.MipsS() == cf.MipsT())
  {
    cf.const_s ? MoveTToReg(regd, cf) : MoveSToReg(regd, cf);
    return;
  }

  Compile_dst_op(cf, &Assembler::or_, &PPC64Recompiler::SafeORI, false, true, false);
}

void CPU::PPC64Recompiler::Compile_xor(CompileFlags cf)
{
  AssertRegOrConstS(cf);
  AssertRegOrConstT(cf);

  const GPR regd = CFGetRegD(cf);
  if (cf.MipsS() == cf.MipsT())
  {
    // xor with self -> zero
    EmitMov(regd, 0);
    return;
  }
  else if (HasConstantRegValue(cf.MipsS(), 0) || HasConstantRegValue(cf.MipsT(), 0))
  {
    // xor with zero -> no effect
    cf.const_s ? MoveTToReg(regd, cf) : MoveSToReg(regd, cf);
    return;
  }

  Compile_dst_op(cf, &Assembler::xor_, &PPC64Recompiler::SafeXORI, false, true, false);
}

void CPU::PPC64Recompiler::Compile_nor(CompileFlags cf)
{
  Compile_or(cf);
  ppcAsm->not_(CFGetRegD(cf), CFGetRegD(cf));
}

void CPU::PPC64Recompiler::Compile_slt(CompileFlags cf)
{
  Compile_slt(cf, true);
}

void CPU::PPC64Recompiler::Compile_sltu(CompileFlags cf)
{
  Compile_slt(cf, false);
}

void CPU::PPC64Recompiler::Compile_slt(CompileFlags cf, bool sign)
{
  AssertRegOrConstS(cf);
  AssertRegOrConstT(cf);

  const GPR rd = CFGetRegD(cf);
  const GPR rs = CFGetSafeRegS(cf, RARG1);

  if (cf.const_t)
  {
    if (sign)
      SafeSLTI(rd, rs, GetConstantRegU32(cf.MipsT()));
    else
      SafeSLTIU(rd, rs, GetConstantRegU32(cf.MipsT()));
    return;
  }

  const GPR rt = CFGetSafeRegT(cf, RARG2);
  if (sign)
    ppcAsm->cmpw(rs, rt);
  else
    ppcAsm->cmplw(rs, rt);
  ppcAsm->li(RSCRATCH, 1);
  ppcAsm->isel(rd, RSCRATCH, RZERO, Cond::LT);
}

GPR CPU::PPC64Recompiler::ComputeLoadStoreAddressArg(CompileFlags cf,
                                                     const std::optional<VirtualMemoryAddress>& address,
                                                     const std::optional<GPR>& reg)
{
  const u32 imm = inst->i.imm_sext32();
  if (cf.valid_host_s && imm == 0 && !reg.has_value())
    return CFGetRegS(cf);

  const GPR dst = reg.has_value() ? reg.value() : RARG1;
  if (address.has_value())
  {
    EmitMov(dst, address.value());
  }
  else if (imm == 0)
  {
    if (cf.valid_host_s)
    {
      if (const GPR src = CFGetRegS(cf); src != dst)
        ppcAsm->mr(dst, src);
    }
    else
    {
      ppcEmitStateLoad32(ppcAsm, dst, OFFS(&g_state.regs.r[cf.mips_s]));
    }
  }
  else
  {
    if (cf.valid_host_s)
    {
      SafeADDIW(dst, CFGetRegS(cf), inst->i.imm_sext32());
    }
    else
    {
      ppcEmitStateLoad32(ppcAsm, dst, OFFS(&g_state.regs.r[cf.mips_s]));
      SafeADDIW(dst, dst, inst->i.imm_sext32());
    }
  }

  return dst;
}

template<typename RegAllocFn>
GPR CPU::PPC64Recompiler::GenerateLoad(GPR addr_reg, MemoryAccessSize size, bool sign, bool use_fastmem,
                                       const RegAllocFn& dst_reg_alloc)
{
  if (use_fastmem)
  {
    m_cycles += Bus::RAM_READ_TICKS;

    const GPR dst = dst_reg_alloc();

    // Address is 32-bit, the upper word of the register is not guaranteed to be clear.
    GPR base_reg = RMEMBASE;
    if (g_settings.cpu_fastmem_mode == CPUFastmemMode::LUT)
    {
      DebugAssert(addr_reg != RARG3);
      ppcEmitLUTIndex(ppcAsm, RARG3, addr_reg, Bus::FASTMEM_LUT_PAGE_SHIFT);
      ppcAsm->ldx(RARG3, RARG3, RMEMBASE);
      base_reg = RARG3;
    }
    ppcAsm->clrldi(RSCRATCH, addr_reg, 32);

    // The backpatch region is two instructions, so it can be replaced with a branch to the slowmem thunk.
    u8* start = ppcAsm->cursor;
    switch (size)
    {
      case MemoryAccessSize::Byte:
        ppcAsm->lbzx(dst, base_reg, RSCRATCH);
        if (sign)
          ppcAsm->extsb(dst, dst);
        else
          ppcAsm->nop();
        break;

      case MemoryAccessSize::HalfWord:
        sign ? ppcAsm->lhax(dst, base_reg, RSCRATCH) : ppcAsm->lhzx(dst, base_reg, RSCRATCH);
        ppcAsm->nop();
        break;

      case MemoryAccessSize::Word:
        ppcAsm->lwzx(dst, base_reg, RSCRATCH);
        ppcAsm->nop();
        break;
    }

    AddLoadStoreInfo(start, 8, addr_reg, dst, size, sign, true);
    return dst;
  }

  if (addr_reg != RARG1)
    ppcAsm->mr(RARG1, addr_reg);

  const bool checked = g_settings.cpu_recompiler_memory_exceptions;
  switch (size)
  {
    case MemoryAccessSize::Byte:
    {
      EmitCall(checked ? reinterpret_cast<const void*>(&RecompilerThunks::ReadMemoryByte) :
                         reinterpret_cast<const void*>(&RecompilerThunks::UncheckedReadMemoryByte));
    }
    break;
    case MemoryAccessSize::HalfWord:
    {
      EmitCall(checked ? reinterpret_cast<const void*>(&RecompilerThunks::ReadMemoryHalfWord) :
                         reinterpret_cast<const void*>(&RecompilerThunks::UncheckedReadMemoryHalfWord));
    }
    break;
    case MemoryAccessSize::Word:
    {
      EmitCall(checked ? reinterpret_cast<const void*>(&RecompilerThunks::ReadMemoryWord) :
                         reinterpret_cast<const void*>(&RecompilerThunks::UncheckedReadMemoryWord));
    }
    break;
  }

  // TODO: turn this into an asm function instead
  if (checked)
  {
    // MSB is set on exception.
    ppcAsm->cmpdi(RRET, 0);
    SwitchToFarCode(true, Cond::LT);
    BackupHostState();

    // Need to stash this in a temp because of the flush.
    const GPR temp = static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED));
    ppcAsm->neg(temp, RRET);
    ppcAsm->slwi(temp, temp, 2);

    Flush(FLUSH_FOR_C_CALL | FLUSH_FLUSH_MIPS_REGISTERS | FLUSH_FOR_EXCEPTION);

    // cause_bits = (-result << 2) | BD | cop_n
    SafeORI(RARG1, temp,
            Cop0Registers::CAUSE::MakeValueForException(
              static_cast<Exception>(0), m_current_instruction_branch_delay_slot, false, inst->cop.cop_n));
    EmitMov(RARG2, m_current_instruction_pc);
    EmitCall(reinterpret_cast<const void*>(static_cast<void (*)(u32, u32)>(&CPU::RaiseException)));
    FreeHostReg(temp);
    EndBlock(std::nullopt, true);

    RestoreHostState();
    SwitchToNearCode(false);
  }

  const GPR dst_reg = dst_reg_alloc();
  switch (size)
  {
    case MemoryAccessSize::Byte:
    {
      sign ? EmitSExtB(dst_reg, RRET) : EmitUExtB(dst_reg, RRET);
    }
    break;
    case MemoryAccessSize::HalfWord:
    {
      sign ? EmitSExtH(dst_reg, RRET) : EmitUExtH(dst_reg, RRET);
    }
    break;
    case MemoryAccessSize::Word:
    {
      if (dst_reg != RRET)
        ppcAsm->mr(dst_reg, RRET);
    }
    break;
  }

  return dst_reg;
}

void CPU::PPC64Recompiler::GenerateStore(GPR addr_reg, GPR value_reg, MemoryAccessSize size, bool use_fastmem)
{
  if (use_fastmem)
  {
    DebugAssert(value_reg != RSCRATCH && value_reg != RARG3);

    GPR base_reg = RMEMBASE;
    if (g_settings.cpu_fastmem_mode == CPUFastmemMode::LUT)
    {
      DebugAssert(addr_reg != RARG3);
      ppcEmitLUTIndex(ppcAsm, RARG3, addr_reg, Bus::FASTMEM_LUT_PAGE_SHIFT);
      ppcAsm->ldx(RARG3, RARG3, RMEMBASE);
      base_reg = RARG3;
    }
    ppcAsm->clrldi(RSCRATCH, addr_reg, 32);

    u8* start = ppcAsm->cursor;
    switch (size)
    {
      case MemoryAccessSize::Byte:
        ppcAsm->stbx(value_reg, base_reg, RSCRATCH);
        break;

      case MemoryAccessSize::HalfWord:
        ppcAsm->sthx(value_reg, base_reg, RSCRATCH);
        break;

      case MemoryAccessSize::Word:
        ppcAsm->stwx(value_reg, base_reg, RSCRATCH);
        break;
    }

    // Pad the backpatch region to match loads.
    ppcAsm->nop();

    AddLoadStoreInfo(start, 8, addr_reg, value_reg, size, false, false);
    return;
  }

  if (addr_reg != RARG1)
    ppcAsm->mr(RARG1, addr_reg);
  if (value_reg != RARG2)
    ppcAsm->mr(RARG2, value_reg);

  const bool checked = g_settings.cpu_recompiler_memory_exceptions;
  switch (size)
  {
    case MemoryAccessSize::Byte:
    {
      EmitCall(checked ? reinterpret_cast<const void*>(&RecompilerThunks::WriteMemoryByte) :
                         reinterpret_cast<const void*>(&RecompilerThunks::UncheckedWriteMemoryByte));
    }
    break;
    case MemoryAccessSize::HalfWord:
    {
      EmitCall(checked ? reinterpret_cast<const void*>(&RecompilerThunks::WriteMemoryHalfWord) :
                         reinterpret_cast<const void*>(&RecompilerThunks::UncheckedWriteMemoryHalfWord));
    }
    break;
    case MemoryAccessSize::Word:
    {
      EmitCall(checked ? reinterpret_cast<const void*>(&RecompilerThunks::WriteMemoryWord) :
                         reinterpret_cast<const void*>(&RecompilerThunks::UncheckedWriteMemoryWord));
    }
    break;
  }

  // TODO: turn this into an asm function instead
  if (checked)
  {
    ppcAsm->cmpwi(RRET, 0);
    SwitchToFarCode(true, Cond::NE);
    BackupHostState();

    // Need to stash this in a temp because of the flush.
    const GPR temp = static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED));
    ppcAsm->slwi(temp, RRET, 2);

    Flush(FLUSH_FOR_C_CALL | FLUSH_FLUSH_MIPS_REGISTERS | FLUSH_FOR_EXCEPTION);

    // cause_bits = (result << 2) | BD | cop_n
    SafeORI(RARG1, temp,
            Cop0Registers::CAUSE::MakeValueForException(
              static_cast<Exception>(0), m_current_instruction_branch_delay_slot, false, inst->cop.cop_n));
    EmitMov(RARG2, m_current_instruction_pc);
    EmitCall(reinterpret_cast<const void*>(static_cast<void (*)(u32, u32)>(&CPU::RaiseException)));
    FreeHostReg(temp);
    EndBlock(std::nullopt, true);

    RestoreHostState();
    SwitchToNearCode(false);
  }
}

void CPU::PPC64Recompiler::Compile_lxx(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                                       const std::optional<VirtualMemoryAddress>& address)
{
  const std::optional<GPR> addr_reg = (g_settings.gpu_pgxp_enable && cf.MipsT() != Reg::zero) ?
                                        std::optional<GPR>(static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED))) :
                                        std::optional<GPR>();
  FlushForLoadStore(address, false, use_fastmem);
  const GPR addr = ComputeLoadStoreAddressArg(cf, address, addr_reg);
  const GPR data = GenerateLoad(addr, size, sign, use_fastmem, [this, cf]() {
    if (cf.MipsT() == Reg::zero)
      return RRET;

    return static_cast<GPR>(AllocateHostReg(GetFlagsForNewLoadDelayedReg(),
                                            EMULATE_LOAD_DELAYS ? HR_TYPE_NEXT_LOAD_DELAY_VALUE : HR_TYPE_CPU_REG,
                                            cf.MipsT()));
  });

  if (g_settings.gpu_pgxp_enable && cf.MipsT() != Reg::zero)
  {
    Flush(FLUSH_FOR_C_CALL);

    EmitMov(RARG1, inst->bits);
    ppcAsm->mr(RARG2, addr);
    ppcAsm->mr(RARG3, data);
    EmitCall(s_pgxp_mem_load_functions[static_cast<u32>(size)][static_cast<u32>(sign)]);
    FreeHostReg(addr_reg.value());
  }
}

void CPU::PPC64Recompiler::Compile_lwx(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                                       const std::optional<VirtualMemoryAddress>& address)
{
  DebugAssert(size == MemoryAccessSize::Word && !sign);

  const GPR addr = static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED));
  FlushForLoadStore(address, false, use_fastmem);

  // TODO: if address is constant, this can be simplified..

  // If we're coming from another block, just flush the load delay and hope for the best..
  if (m_load_delay_dirty)
    UpdateLoadDelay();

  // We'd need to be careful here if we weren't overwriting it..
  ComputeLoadStoreAddressArg(cf, address, addr);

  // Do PGXP first, it does its own load.
  if (g_settings.gpu_pgxp_enable && inst->r.rt != Reg::zero)
  {
    Flush(FLUSH_FOR_C_CALL);
    EmitMov(RARG1, inst->bits);
    ppcAsm->mr(RARG2, addr);
    MoveMIPSRegToReg(RARG3, inst->r.rt, true);
    EmitCall(reinterpret_cast<const void*>(&PGXP::CPU_LWx));
  }

  ppcAsm->clrrwi(RARG1, addr, 2); // addr & ~3
  GenerateLoad(RARG1, MemoryAccessSize::Word, false, use_fastmem, []() { return RRET; });

  if (inst->r.rt == Reg::zero)
  {
    FreeHostReg(addr);
    return;
  }

  // lwl/lwr from a load-delayed value takes the new value, but it itself, is load delayed, so the original value is
  // never written back. NOTE: can't trust T in cf because of the flush
  const Reg rt = inst->r.rt;
  GPR value;
  if (m_load_delay_register == rt)
  {
    const u32 existing_ld_rt = (m_load_delay_value_register == NUM_HOST_REGS) ?
                                 AllocateHostReg(HR_MODE_READ, HR_TYPE_LOAD_DELAY_VALUE, rt) :
                                 m_load_delay_value_register;
    RenameHostReg(existing_ld_rt, HR_MODE_WRITE, HR_TYPE_NEXT_LOAD_DELAY_VALUE, rt);
    value = static_cast<GPR>(existing_ld_rt);
  }
  else
  {
    if constexpr (EMULATE_LOAD_DELAYS)
    {
      value = static_cast<GPR>(AllocateHostReg(HR_MODE_WRITE, HR_TYPE_NEXT_LOAD_DELAY_VALUE, rt));
      if (const std::optional<u32> rtreg = CheckHostReg(HR_MODE_READ, HR_TYPE_CPU_REG, rt); rtreg.has_value())
        ppcAsm->mr(value, static_cast<GPR>(rtreg.value()));
      else if (HasConstantReg(rt))
        EmitMov(value, GetConstantRegU32(rt));
      else
        ppcEmitStateLoad32(ppcAsm, value, OFFS(&g_state.regs.r[static_cast<u8>(rt)]));
    }
    else
    {
      value = static_cast<GPR>(AllocateHostReg(HR_MODE_READ | HR_MODE_WRITE, HR_TYPE_CPU_REG, rt));
    }
  }

  DebugAssert(value != RARG2 && value != RARG3);
  ppcAsm->andi_(RARG2, addr, 3);
  ppcAsm->slwi(RARG2, RARG2, 3); // *8
  ppcAsm->subfic(RARG3, RARG2, 24); // 24 - shift

  if (inst->op == InstructionOp::lwl)
  {
    // const u32 mask = UINT32_C(0x00FFFFFF) >> shift;
    // new_value = (value & mask) | (RWRET << (24 - shift));
    EmitMov(RSCRATCH, 0xFFFFFFu);
    ppcAsm->srw(RSCRATCH, RSCRATCH, RARG2);
    ppcAsm->and_(value, value, RSCRATCH);
    ppcAsm->slw(RRET, RRET, RARG3);
    ppcAsm->or_(value, value, RRET);
  }
  else
  {
    // const u32 mask = UINT32_C(0xFFFFFF00) << (24 - shift);
    // new_value = (value & mask) | (RWRET >> shift);
    ppcAsm->srw(RRET, RRET, RARG2);
    EmitMov(RSCRATCH, 0xFFFFFF00u);
    ppcAsm->slw(RSCRATCH, RSCRATCH, RARG3);
    ppcAsm->and_(value, value, RSCRATCH);
    ppcAsm->or_(value, value, RRET);
  }

  FreeHostReg(addr);
}

void CPU::PPC64Recompiler::Compile_lwc2(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                                        const std::optional<VirtualMemoryAddress>& address)
{
  const u32 index = static_cast<u32>(inst->r.rt.GetValue());
  const auto [ptr, action] = GetGTERegisterPointer(index, true);
  const std::optional<GPR> addr_reg = g_settings.gpu_pgxp_enable ?
                                        std::optional<GPR>(static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED))) :
                                        std::optional<GPR>();
  FlushForLoadStore(address, false, use_fastmem);
  const GPR addr = ComputeLoadStoreAddressArg(cf, address, addr_reg);
  const GPR value = GenerateLoad(addr, MemoryAccessSize::Word, false, use_fastmem, [this, action = action]() {
    return (action == GTERegisterAccessAction::CallHandler && g_settings.gpu_pgxp_enable) ?
             static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED)) :
             RRET;
  });

  switch (action)
  {
    case GTERegisterAccessAction::Ignore:
    {
      break;
    }

    case GTERegisterAccessAction::Direct:
    {
      ppcEmitStateStore32(ppcAsm, value, OFFS(ptr));
      break;
    }

    case GTERegisterAccessAction::SignExtend16:
    {
      EmitSExtH(RARG3, value);
      ppcEmitStateStore32(ppcAsm, RARG3, OFFS(ptr));
      break;
    }

    case GTERegisterAccessAction::ZeroExtend16:
    {
      EmitUExtH(RARG3, value);
      ppcEmitStateStore32(ppcAsm, RARG3, OFFS(ptr));
      break;
    }

    case GTERegisterAccessAction::CallHandler:
    {
      Flush(FLUSH_FOR_C_CALL);
      ppcAsm->mr(RARG2, value);
      EmitMov(RARG1, index);
      EmitCall(reinterpret_cast<const void*>(&GTE::WriteRegister));
      break;
    }

    case GTERegisterAccessAction::PushFIFO:
    {
      // SXY0 <- SXY1
      // SXY1 <- SXY2
      // SXY2 <- SXYP
      DebugAssert(value != RARG2 && value != RARG3);
      ppcEmitStateLoad32(ppcAsm, RARG2, OFFS(&g_state.gte_regs.SXY1[0]));
      ppcEmitStateLoad32(ppcAsm, RARG3, OFFS(&g_state.gte_regs.SXY2[0]));
      ppcEmitStateStore32(ppcAsm, RARG2, OFFS(&g_state.gte_regs.SXY0[0]));
      ppcEmitStateStore32(ppcAsm, RARG3, OFFS(&g_state.gte_regs.SXY1[0]));
      ppcEmitStateStore32(ppcAsm, value, OFFS(&g_state.gte_regs.SXY2[0]));
      break;
    }

    default:
    {
      Panic("Unknown action");
      return;
    }
  }

  if (g_settings.gpu_pgxp_enable)
  {
    Flush(FLUSH_FOR_C_CALL);
    ppcAsm->mr(RARG3, value);
    if (value != RRET)
      FreeHostReg(value);
    ppcAsm->mr(RARG2, addr);
    FreeHostReg(addr_reg.value());
    EmitMov(RARG1, inst->bits);
    EmitCall(reinterpret_cast<const void*>(&PGXP::CPU_LWC2));
  }
}

void CPU::PPC64Recompiler::Compile_sxx(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                                       const std::optional<VirtualMemoryAddress>& address)
{
  AssertRegOrConstS(cf);
  AssertRegOrConstT(cf);

  const std::optional<GPR> addr_reg = g_settings.gpu_pgxp_enable ?
                                        std::optional<GPR>(static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED))) :
                                        std::optional<GPR>();
  FlushForLoadStore(address, true, use_fastmem);
  const GPR addr = ComputeLoadStoreAddressArg(cf, address, addr_reg);
  const GPR data = cf.valid_host_t ? CFGetRegT(cf) : RARG2;
  if (!cf.valid_host_t)
    MoveTToReg(RARG2, cf);

  GenerateStore(addr, data, size, use_fastmem);

  if (g_settings.gpu_pgxp_enable)
  {
    Flush(FLUSH_FOR_C_CALL);
    MoveMIPSRegToReg(RARG3, cf.MipsT());
    ppcAsm->mr(RARG2, addr);
    EmitMov(RARG1, inst->bits);
    EmitCall(s_pgxp_mem_store_functions[static_cast<u32>(size)]);
    FreeHostReg(addr_reg.value());
  }
}

void CPU::PPC64Recompiler::Compile_swx(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                                       const std::optional<VirtualMemoryAddress>& address)
{
  DebugAssert(size == MemoryAccessSize::Word && !sign);

  // TODO: this can take over rt's value if it's no longer needed
  // NOTE: can't trust T in cf because of the alloc
  const GPR addr = static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED));

  FlushForLoadStore(address, true, use_fastmem);

  // TODO: if address is constant, this can be simplified..
  // We'd need to be careful here if we weren't overwriting it..
  ComputeLoadStoreAddressArg(cf, address, addr);

  if (g_settings.gpu_pgxp_enable)
  {
    Flush(FLUSH_FOR_C_CALL);
    EmitMov(RARG1, inst->bits);
    ppcAsm->mr(RARG2, addr);
    MoveMIPSRegToReg(RARG3, inst->r.rt);
    EmitCall(reinterpret_cast<const void*>(&PGXP::CPU_SWx));
  }

  ppcAsm->clrrwi(RARG1, addr, 2); // addr & ~3
  GenerateLoad(RARG1, MemoryAccessSize::Word, false, use_fastmem, []() { return RRET; });

  ppcAsm->andi_(RSCRATCH, addr, 3);
  ppcAsm->slwi(RSCRATCH, RSCRATCH, 3); // *8
  ppcAsm->clrrwi(addr, addr, 2);       // addr & ~3

  // Load rt down here, the PGXP call and the load above can both clobber the argument registers.
  MoveMIPSRegToReg(RARG2, inst->r.rt);

  if (inst->op == InstructionOp::swl)
  {
    // const u32 mem_mask = UINT32_C(0xFFFFFF00) << shift;
    // new_value = (RWRET & mem_mask) | (value >> (24 - shift));
    EmitMov(RARG3, 0xFFFFFF00u);
    ppcAsm->slw(RARG3, RARG3, RSCRATCH);
    ppcAsm->and_(RRET, RRET, RARG3);

    ppcAsm->subfic(RARG3, RSCRATCH, 24);
    ppcAsm->srw(RARG2, RARG2, RARG3);
    ppcAsm->or_(RARG2, RARG2, RRET);
  }
  else
  {
    // const u32 mem_mask = UINT32_C(0x00FFFFFF) >> (24 - shift);
    // new_value = (RWRET & mem_mask) | (value << shift);
    ppcAsm->slw(RARG2, RARG2, RSCRATCH);

    ppcAsm->subfic(RARG3, RSCRATCH, 24);
    EmitMov(RSCRATCH, 0x00FFFFFFu);
    ppcAsm->srw(RSCRATCH, RSCRATCH, RARG3);
    ppcAsm->and_(RRET, RRET, RSCRATCH);
    ppcAsm->or_(RARG2, RARG2, RRET);
  }

  GenerateStore(addr, RARG2, MemoryAccessSize::Word, use_fastmem);
  FreeHostReg(addr);
}

void CPU::PPC64Recompiler::Compile_swc2(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                                        const std::optional<VirtualMemoryAddress>& address)
{
  const u32 index = static_cast<u32>(inst->r.rt.GetValue());
  const auto [ptr, action] = GetGTERegisterPointer(index, false);
  const GPR addr = (g_settings.gpu_pgxp_enable || action == GTERegisterAccessAction::CallHandler) ?
                     static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED)) :
                     RARG1;
  const GPR data = g_settings.gpu_pgxp_enable ? static_cast<GPR>(AllocateTempHostReg(HR_CALLEE_SAVED)) : RARG2;
  FlushForLoadStore(address, true, use_fastmem);
  ComputeLoadStoreAddressArg(cf, address, addr);

  switch (action)
  {
    case GTERegisterAccessAction::Direct:
    {
      ppcEmitStateLoad32(ppcAsm, data, OFFS(ptr));
    }
    break;

    case GTERegisterAccessAction::CallHandler:
    {
      // should already be flushed.. except in fastmem case
      Flush(FLUSH_FOR_C_CALL);
      EmitMov(RARG1, index);
      EmitCall(reinterpret_cast<const void*>(&GTE::ReadRegister));
      ppcAsm->mr(data, RRET);
    }
    break;

    default:
    {
      Panic("Unknown action");
    }
    break;
  }

  GenerateStore(addr, data, size, use_fastmem);

  if (!g_settings.gpu_pgxp_enable)
  {
    if (addr != RARG1)
      FreeHostReg(addr);
  }
  else
  {
    // TODO: This can be simplified because we don't need to validate in PGXP..
    Flush(FLUSH_FOR_C_CALL);
    ppcAsm->mr(RARG3, data);
    FreeHostReg(data);
    ppcAsm->mr(RARG2, addr);
    FreeHostReg(addr);
    EmitMov(RARG1, inst->bits);
    EmitCall(reinterpret_cast<const void*>(&PGXP::CPU_SWC2));
  }
}

void CPU::PPC64Recompiler::Compile_mtc0(CompileFlags cf)
{
  // TODO: we need better constant setting here.. which will need backprop
  AssertRegOrConstT(cf);

  const Cop0Reg reg = static_cast<Cop0Reg>(MipsD());
  const u32* ptr = GetCop0RegPtr(reg);
  const u32 mask = GetCop0RegWriteMask(reg);
  if (!ptr)
  {
    Compile_Fallback();
    return;
  }

  if (mask == 0)
  {
    // if it's a read-only register, ignore
    DEBUG_LOG("Ignoring write to read-only cop0 reg {}", static_cast<u32>(reg));
    return;
  }

  // for some registers, we need to test certain bits
  const bool needs_bit_test = (reg == Cop0Reg::SR);
  const GPR new_value = RARG1;
  const GPR old_value = RARG2;
  const GPR changed_bits = RARG3;
  const GPR mask_reg = RSCRATCH;

  // Load old value
  ppcEmitStateLoad32(ppcAsm, old_value, OFFS(ptr));

  // No way we fit this in an immediate..
  EmitMov(mask_reg, mask);

  // update value
  if (cf.valid_host_t)
    ppcAsm->and_(new_value, CFGetRegT(cf), mask_reg);
  else
    EmitMov(new_value, GetConstantRegU32(cf.MipsT()) & mask);

  if (needs_bit_test)
    ppcAsm->xor_(changed_bits, old_value, new_value);
  ppcAsm->andc(old_value, old_value, mask_reg);
  ppcAsm->or_(new_value, old_value, new_value);
  ppcEmitStateStore32(ppcAsm, new_value, OFFS(ptr));

  if (reg == Cop0Reg::SR)
  {
    // TODO: replace with register backup
    // We could just inline the whole thing..
    Flush(FLUSH_FOR_C_CALL);

    // Isolate cache bit (16) changed?
    Label caches_unchanged;
    ppcAsm->andis_(RSCRATCH, changed_bits, 1);
    ppcAsm->bc(Cond::EQ, &caches_unchanged);
    EmitCall(reinterpret_cast<const void*>(&CPU::UpdateMemoryPointers));
    ppcEmitStateLoad32(ppcAsm, new_value, OFFS(ptr));
    if (CodeCache::IsUsingFastmem())
      ppcEmitStateLoad64(ppcAsm, RMEMBASE, OFFS(&g_state.fastmem_base));
    ppcAsm->Bind(&caches_unchanged);

    TestInterrupts(RARG1);
  }
  else if (reg == Cop0Reg::CAUSE)
  {
    ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.cop0_regs.sr.bits));
    TestInterrupts(RARG1);
  }
  else if (reg == Cop0Reg::DCIC || reg == Cop0Reg::BPCM)
  {
    // need to check whether we're switching to debug mode
    Flush(FLUSH_FOR_C_CALL);
    EmitCall(reinterpret_cast<const void*>(&CPU::UpdateDebugDispatcherFlag));
    ppcAsm->andi_(RSCRATCH, RRET, 0xFF);
    SwitchToFarCode(true, Cond::NE);
    BackupHostState();
    Flush(FLUSH_FOR_EARLY_BLOCK_EXIT);
    EmitCall(reinterpret_cast<const void*>(&CPU::ExitExecution)); // does not return
    RestoreHostState();
    SwitchToNearCode(false);
  }
}

void CPU::PPC64Recompiler::Compile_rfe(CompileFlags cf)
{
  // shift mode bits right two, preserving upper bits
  ppcEmitStateLoad32(ppcAsm, RARG1, OFFS(&g_state.cop0_regs.sr.bits));
  ppcAsm->rlwimi(RARG1, RARG1, 30, 28, 31); // sr = (sr & ~0xF) | ((sr >> 2) & 0xF)
  ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.cop0_regs.sr.bits));

  TestInterrupts(RARG1);
}

void CPU::PPC64Recompiler::TestInterrupts(GPR sr)
{
  DebugAssert(sr != RSCRATCH);

  // if Iec == 0 then goto no_interrupt
  Label no_interrupt;
  ppcAsm->andi_(RSCRATCH, sr, 1);
  ppcAsm->bc(Cond::EQ, &no_interrupt);

  // sr & cause
  ppcEmitStateLoad32(ppcAsm, RSCRATCH, OFFS(&g_state.cop0_regs.cause.bits));
  ppcAsm->and_(sr, sr, RSCRATCH);

  // ((sr & cause) & 0xff00) == 0 goto no_interrupt
  ppcAsm->rlwinm_(sr, sr, 24, 24, 31);
  SwitchToFarCode(true, Cond::NE);

  BackupHostState();

  // Update load delay, this normally happens at the end of an instruction, but we're finishing it early.
  UpdateLoadDelay();

  Flush(FLUSH_END_BLOCK | FLUSH_FOR_EXCEPTION | FLUSH_FOR_C_CALL);

  // Can't use EndBlockWithException() here, because it'll use the wrong PC.
  // Can't use RaiseException() on the fast path if we're the last instruction, because the next PC is unknown.
  if (!iinfo->is_last_instruction)
  {
    EmitMov(RARG1, Cop0Registers::CAUSE::MakeValueForException(Exception::INT, iinfo->is_branch_instruction, false,
                                                                (inst + 1)->cop.cop_n));
    EmitMov(RARG2, m_compiler_pc);
    EmitCall(reinterpret_cast<const void*>(static_cast<void (*)(u32, u32)>(&CPU::RaiseException)));
    m_dirty_pc = false;
    EndAndLinkBlock(std::nullopt, true, false);
  }
  else
  {
    if (m_dirty_pc)
      EmitMov(RARG1, m_compiler_pc);
    ppcEmitStateStore32(ppcAsm, RZERO, OFFS(&g_state.downcount));
    if (m_dirty_pc)
      ppcEmitStateStore32(ppcAsm, RARG1, OFFS(&g_state.pc));
    m_dirty_pc = false;
    EndAndLinkBlock(std::nullopt, false, true);
  }

  RestoreHostState();
  SwitchToNearCode(false);

  ppcAsm->Bind(&no_interrupt);
}

void CPU::PPC64Recompiler::Compile_mfc2(CompileFlags cf)
{
  const u32 index = inst->cop.Cop2Index();
  const Reg rt = inst->r.rt;

  const auto [ptr, action] = GetGTERegisterPointer(index, false);
  if (action == GTERegisterAccessAction::Ignore)
    return;

  u32 hreg;
  if (action == GTERegisterAccessAction::Direct)
  {
    hreg = AllocateHostReg(GetFlagsForNewLoadDelayedReg(),
                           EMULATE_LOAD_DELAYS ? HR_TYPE_NEXT_LOAD_DELAY_VALUE : HR_TYPE_CPU_REG, rt);
    ppcEmitStateLoad32(ppcAsm, static_cast<GPR>(hreg), OFFS(ptr));
  }
  else if (action == GTERegisterAccessAction::CallHandler)
  {
    Flush(FLUSH_FOR_C_CALL);
    EmitMov(RARG1, index);
    EmitCall(reinterpret_cast<const void*>(&GTE::ReadRegister));

    hreg = AllocateHostReg(GetFlagsForNewLoadDelayedReg(),
                           EMULATE_LOAD_DELAYS ? HR_TYPE_NEXT_LOAD_DELAY_VALUE : HR_TYPE_CPU_REG, rt);
    ppcAsm->mr(static_cast<GPR>(hreg), RRET);
  }
  else
  {
    Panic("Unknown action");
  }

  if (g_settings.gpu_pgxp_enable)
  {
    Flush(FLUSH_FOR_C_CALL);
    EmitMov(RARG1, inst->bits);
    ppcAsm->mr(RARG2, static_cast<GPR>(hreg));
    EmitCall(reinterpret_cast<const void*>(&PGXP::CPU_MFC2));
  }
}

void CPU::PPC64Recompiler::Compile_mtc2(CompileFlags cf)
{
  const u32 index = inst->cop.Cop2Index();
  const auto [ptr, action] = GetGTERegisterPointer(index, true);
  if (action == GTERegisterAccessAction::Ignore)
    return;

  if (action == GTERegisterAccessAction::Direct)
  {
    if (cf.const_t)
      StoreConstantToCPUPointer(GetConstantRegU32(cf.MipsT()), ptr);
    else
      ppcEmitStateStore32(ppcAsm, CFGetRegT(cf), OFFS(ptr));
  }
  else if (action == GTERegisterAccessAction::SignExtend16 || action == GTERegisterAccessAction::ZeroExtend16)
  {
    const bool sign = (action == GTERegisterAccessAction::SignExtend16);
    if (cf.valid_host_t)
    {
      sign ? EmitSExtH(RARG1, CFGetRegT(cf)) : EmitUExtH(RARG1, CFGetRegT(cf));
      ppcEmitStateStore32(ppcAsm, RARG1, OFFS(ptr));
    }
    else if (cf.const_t)
    {
      const u16 cv = Truncate16(GetConstantRegU32(cf.MipsT()));
      StoreConstantToCPUPointer(sign ? ::SignExtend32(cv) : ::ZeroExtend32(cv), ptr);
    }
    else
    {
      Panic("Unsupported setup");
    }
  }
  else if (action == GTERegisterAccessAction::CallHandler)
  {
    Flush(FLUSH_FOR_C_CALL);
    EmitMov(RARG1, index);
    MoveTToReg(RARG2, cf);
    EmitCall(reinterpret_cast<const void*>(&GTE::WriteRegister));
  }
  else if (action == GTERegisterAccessAction::PushFIFO)
  {
    // SXY0 <- SXY1
    // SXY1 <- SXY2
    // SXY2 <- SXYP
    DebugAssert(RRET != RARG2 && RRET != RARG3);
    ppcEmitStateLoad32(ppcAsm, RARG2, OFFS(&g_state.gte_regs.SXY1[0]));
    ppcEmitStateLoad32(ppcAsm, RARG3, OFFS(&g_state.gte_regs.SXY2[0]));
    ppcEmitStateStore32(ppcAsm, RARG2, OFFS(&g_state.gte_regs.SXY0[0]));
    ppcEmitStateStore32(ppcAsm, RARG3, OFFS(&g_state.gte_regs.SXY1[0]));
    if (cf.valid_host_t)
      ppcEmitStateStore32(ppcAsm, CFGetRegT(cf), OFFS(&g_state.gte_regs.SXY2[0]));
    else if (cf.const_t)
      StoreConstantToCPUPointer(GetConstantRegU32(cf.MipsT()), &g_state.gte_regs.SXY2[0]);
    else
      Panic("Unsupported setup");
  }
  else
  {
    Panic("Unknown action");
  }
}

void CPU::PPC64Recompiler::Compile_cop2(CompileFlags cf)
{
  TickCount func_ticks;
  GTE::InstructionImpl func = GTE::GetInstructionImpl(inst->bits, &func_ticks);

  Flush(FLUSH_FOR_C_CALL);
  EmitMov(RARG1, inst->bits & GTE::Instruction::REQUIRED_BITS_MASK);
  EmitCall(reinterpret_cast<const void*>(func));

  AddGTETicks(func_ticks);
}

u32 CPU::Recompiler::CompileLoadStoreThunk(void* thunk_code, u32 thunk_space, void* code_address, u32 code_size,
                                           TickCount cycles_to_add, TickCount cycles_to_remove, u32 gpr_bitmask,
                                           u8 address_register, u8 data_register, MemoryAccessSize size, bool is_signed,
                                           bool is_load)
{
  Assembler actual_asm(static_cast<u8*>(thunk_code), thunk_space);
  Assembler* ppcAsm = &actual_asm;

  static constexpr u32 GPR_SIZE = 8;

  // save regs
  u32 num_gprs = 0;

  for (u32 i = 0; i < NUM_HOST_REGS; i++)
  {
    if ((gpr_bitmask & (1u << i)) && ppcIsCallerSavedRegister(i) && (!is_load || data_register != i))
      num_gprs++;
  }

  // New frame below the JIT's, the callee stores LR/CR into its linkage area.
  const u32 stack_size = Common::AlignUpPow2(ELFV2_LINKAGE_AREA_SIZE + num_gprs * GPR_SIZE, 16);
  ppcAsm->stdu(r1, -static_cast<s32>(stack_size), r1);

  if (num_gprs > 0)
  {
    u32 stack_offset = ELFV2_LINKAGE_AREA_SIZE;
    for (u32 i = 0; i < NUM_HOST_REGS; i++)
    {
      if ((gpr_bitmask & (1u << i)) && ppcIsCallerSavedRegister(i) && (!is_load || data_register != i))
      {
        ppcAsm->std_(static_cast<GPR>(i), static_cast<s32>(stack_offset), r1);
        stack_offset += GPR_SIZE;
      }
    }
  }

  if (cycles_to_add != 0)
  {
    // NOTE: we have to reload here, because memory writes can run DMA, which can screw with cycles
    ppcEmitStateLoad32(ppcAsm, RSCRATCH, OFFS(&g_state.pending_ticks));
    ppcEmitAddImm(ppcAsm, RSCRATCH, RSCRATCH, cycles_to_add);
    ppcEmitStateStore32(ppcAsm, RSCRATCH, OFFS(&g_state.pending_ticks));
  }

  if (address_register != static_cast<u8>(RARG1))
    ppcAsm->mr(RARG1, static_cast<GPR>(address_register));

  if (!is_load)
  {
    if (data_register != static_cast<u8>(RARG2))
      ppcAsm->mr(RARG2, static_cast<GPR>(data_register));
  }

  switch (size)
  {
    case MemoryAccessSize::Byte:
    {
      ppcEmitCall(ppcAsm, is_load ? reinterpret_cast<const void*>(&RecompilerThunks::UncheckedReadMemoryByte) :
                                    reinterpret_cast<const void*>(&RecompilerThunks::UncheckedWriteMemoryByte));
    }
    break;
    case MemoryAccessSize::HalfWord:
    {
      ppcEmitCall(ppcAsm, is_load ? reinterpret_cast<const void*>(&RecompilerThunks::UncheckedReadMemoryHalfWord) :
                                    reinterpret_cast<const void*>(&RecompilerThunks::UncheckedWriteMemoryHalfWord));
    }
    break;
    case MemoryAccessSize::Word:
    {
      ppcEmitCall(ppcAsm, is_load ? reinterpret_cast<const void*>(&RecompilerThunks::UncheckedReadMemoryWord) :
                                    reinterpret_cast<const void*>(&RecompilerThunks::UncheckedWriteMemoryWord));
    }
    break;
  }

  if (is_load)
  {
    const GPR dst = static_cast<GPR>(data_register);
    switch (size)
    {
      case MemoryAccessSize::Byte:
      {
        is_signed ? ppcEmitSExtB(ppcAsm, dst, RRET) : ppcEmitUExtB(ppcAsm, dst, RRET);
      }
      break;
      case MemoryAccessSize::HalfWord:
      {
        is_signed ? ppcEmitSExtH(ppcAsm, dst, RRET) : ppcEmitUExtH(ppcAsm, dst, RRET);
      }
      break;
      case MemoryAccessSize::Word:
      {
        if (dst != RRET)
          ppcAsm->mr(dst, RRET);
      }
      break;
    }
  }

  if (cycles_to_remove != 0)
  {
    ppcEmitStateLoad32(ppcAsm, RSCRATCH, OFFS(&g_state.pending_ticks));
    ppcEmitAddImm(ppcAsm, RSCRATCH, RSCRATCH, -cycles_to_remove);
    ppcEmitStateStore32(ppcAsm, RSCRATCH, OFFS(&g_state.pending_ticks));
  }

  // restore regs
  if (num_gprs > 0)
  {
    u32 stack_offset = ELFV2_LINKAGE_AREA_SIZE;
    for (u32 i = 0; i < NUM_HOST_REGS; i++)
    {
      if ((gpr_bitmask & (1u << i)) && ppcIsCallerSavedRegister(i) && (!is_load || data_register != i))
      {
        ppcAsm->ld(static_cast<GPR>(i), static_cast<s32>(stack_offset), r1);
        stack_offset += GPR_SIZE;
      }
    }
  }
  ppcAsm->addi(r1, r1, static_cast<s32>(stack_size));

  ppcEmitJmp(ppcAsm, static_cast<const u8*>(code_address) + code_size);

  return static_cast<u32>(ppcAsm->GetSize());
}

#endif // CPU_ARCH_PPC64
