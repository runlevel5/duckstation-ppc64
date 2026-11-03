// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: CC-BY-NC-ND-4.0 + Packaging Restriction

#include "cpu_test_harness.h"

#include "core/bus.h"
#include "core/cpu_code_cache.h"
#include "core/settings.h"
#include "core/system_private.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

using namespace CPUTestHarness;
using namespace Mips::Encoder;

namespace {

// Microbenchmarks for the recompiler backends. Skipped unless DUCKSTATION_BENCH is set, since they only produce
// numbers. DUCKSTATION_BENCH_ITERS overrides the loop count.
class CPUBench : public ExecutionModeTest
{
protected:
  static constexpr u32 BODY_REPEATS = 8;

  // Program with the instruction index tracked, for computing branch offsets and placement.
  struct Bench
  {
    Program prog;
    u32 index = 0;

    void Emit(u32 inst)
    {
      prog.Emit(inst);
      index++;
    }
    void LoadImmediate(Reg reg, u32 value)
    {
      prog.LoadImmediate(reg, value);
      index += 2;
    }
    // lui sign-extends in a 64-bit host register, ori/sll leaves the value zero-extended.
    void LoadZeroExtendedAddress(Reg reg, u32 address)
    {
      Emit(ori(reg, Reg::R0, static_cast<u16>(address >> 16)));
      Emit(sll(reg, reg, 16));
      if ((address & 0xFFFFu) != 0)
        Emit(ori(reg, reg, static_cast<u16>(address)));
    }
  };

  void SetUp() override
  {
    if (!std::getenv("DUCKSTATION_BENCH"))
      GTEST_SKIP() << "Set DUCKSTATION_BENCH=1 to run.";

    ExecutionModeTest::SetUp();
    m_iterations = 10000000;
    if (const char* env = std::getenv("DUCKSTATION_BENCH_ITERS"))
      m_iterations = static_cast<u32>(std::strtoul(env, nullptr, 10));
  }

  void TearDown() override
  {
    if (!g_settings.cpu_recompiler_block_linking)
    {
      g_settings.cpu_recompiler_block_linking = true;
      CPU::CodeCache::Reset();
    }
  }

  static std::vector<u32> Repeat(u32 inst, u32 count = BODY_REPEATS) { return std::vector<u32>(count, inst); }

  // Results have to be live and in distinct registers, or the recompiler discards the earlier writes.
  static Reg Dst(u32 i)
  {
    static constexpr Reg regs[] = {Reg::T1, Reg::T3, Reg::T4, Reg::T5, Reg::T6, Reg::T7, Reg::T8, Reg::T9};
    return regs[i % std::size(regs)];
  }

  /// Wraps body in a counted loop, returns the index of the loop branch.
  u32 EmitLoop(Bench& b, const std::vector<u32>& body)
  {
    b.LoadImmediate(Reg::T0, m_iterations);
    const u32 loop_start = b.index;
    for (const u32 inst : body)
      b.Emit(inst);
    b.Emit(addiu(Reg::T0, Reg::T0, -1));
    const u32 branch_index = b.index;
    b.Emit(bne(Reg::T0, Reg::R0, static_cast<s16>(static_cast<s32>(loop_start) - static_cast<s32>(branch_index + 1))));
    b.Emit(nop());
    return branch_index;
  }

  /// Runs the program three times and reports the best time per loop iteration and per body instruction.
  void Report(const char* name, Bench& b, u32 address = PROGRAM_BASE)
  {
    double best_ns = 1e300;
    for (u32 run = 0; run < 3; run++)
    {
      // Blocks recompiled too often within 15 frames get interpreted instead.
      for (u32 i = 0; i < 15; i++)
        System::IncrementFrameNumber();

      const auto start = std::chrono::steady_clock::now();
      b.prog.RunAt(address);
      const auto end = std::chrono::steady_clock::now();
      best_ns = std::min(best_ns, std::chrono::duration<double, std::nano>(end - start).count() / m_iterations);
    }

    std::printf("[BENCH] %-32s %9.2f ns/iter %8.2f ns/op\n", name, best_ns, best_ns / BODY_REPEATS);
    std::fflush(stdout);
  }

  u32 m_iterations = 0;
};

TEST_P(CPUBench, ALU)
{
  {
    Bench b;
    EmitLoop(b, Repeat(addiu(Reg::T1, Reg::T1, 1)));
    Report("alu (dependent)", b);
  }
  {
    Bench b;
    std::vector<u32> body;
    for (u32 i = 0; i < BODY_REPEATS; i++)
      body.push_back(addiu(Dst(i), Reg::T2, 1));
    EmitLoop(b, body);
    Report("alu (independent)", b);
  }
}

TEST_P(CPUBench, LoadFastmemLUT)
{
  ResetMemory(CPURAMSize::Default, CPUFastmemMode::LUT);
  Bench b;
  b.LoadZeroExtendedAddress(Reg::T2, RAM_TEST_ADDR);
  std::vector<u32> body;
  for (u32 i = 0; i < BODY_REPEATS; i++)
    body.push_back(lw(Dst(i), 0, Reg::T2));
  EmitLoop(b, body);
  Report("load (fastmem lut)", b);
}

TEST_P(CPUBench, LoadHandlerCall)
{
  ResetMemory(CPURAMSize::Default, CPUFastmemMode::Disabled);
  Bench b;
  b.LoadZeroExtendedAddress(Reg::T2, RAM_TEST_ADDR);
  std::vector<u32> body;
  for (u32 i = 0; i < BODY_REPEATS; i++)
    body.push_back(lw(Dst(i), 0, Reg::T2));
  EmitLoop(b, body);
  Report("load (handler call)", b);
}

TEST_P(CPUBench, DependentLoadFastmemLUT)
{
  ResetMemory(CPURAMSize::Default, CPUFastmemMode::LUT);

  // Chase a word that points at itself, so each load's address depends on the previous load. The nop keeps the
  // next load out of the load delay slot, which would otherwise use the old register value.
  const u32 self = RAM_TEST_ADDR;
  std::memcpy(&g_bus.ram[RAM_TEST_ADDR & g_bus.ram_mask], &self, sizeof(self));

  Bench b;
  b.LoadZeroExtendedAddress(Reg::T2, RAM_TEST_ADDR);
  std::vector<u32> body;
  for (u32 i = 0; i < BODY_REPEATS; i++)
  {
    body.push_back(lw(Reg::T2, 0, Reg::T2));
    body.push_back(nop());
  }
  EmitLoop(b, body);
  Report("dependent load (fastmem lut)", b);
}

TEST_P(CPUBench, Dispatch)
{
  // Blocks of 16 words that jump to the next one, so each j really leaves the block (a branch to the following
  // instruction is folded away). Unlinked, every transition goes through the dispatcher.
  static constexpr u32 BLOCK_WORDS = 16;
  for (const bool linked : {true, false})
  {
    g_settings.cpu_recompiler_block_linking = linked;
    CPU::CodeCache::Reset();
    Bench b;
    std::vector<u32> body;
    const u32 loop_start = 2; // after LoadImmediate(T0)
    const auto target = [&](u32 index) { return PROGRAM_BASE + (1 + index) * static_cast<u32>(sizeof(u32)); };
    for (u32 i = 0; i < BODY_REPEATS; i++)
    {
      body.push_back(addiu(Dst(i), Reg::T2, 1));
      body.push_back(j(target(loop_start + (i + 1) * BLOCK_WORDS)));
      body.push_back(nop());
      while ((body.size() % BLOCK_WORDS) != 0)
        body.push_back(nop());
    }
    EmitLoop(b, body);
    Report(linked ? "block exit (linked)" : "block exit (dispatcher)", b);
  }
}

TEST_P(CPUBench, Div)
{
  for (const bool is_signed : {true, false})
  {
    {
      // Each remainder feeds the next divide, so the whole div sequence is on the critical path. The divisor is
      // 2 or 3 from the loop counter (a constant divisor would be compiled to a multiply), and the remainder is
      // bumped back up to a 20-bit dividend so the divider can't take an early exit.
      Bench b;
      b.LoadImmediate(Reg::T1, 1000003u);
      b.LoadImmediate(Reg::T4, 1000003u);
      std::vector<u32> body;
      body.push_back(sll(Reg::T2, Reg::T0, 31));
      body.push_back(srl(Reg::T2, Reg::T2, 31));
      body.push_back(addiu(Reg::T2, Reg::T2, 2));
      for (u32 i = 0; i < BODY_REPEATS; i++)
      {
        body.push_back(is_signed ? div(Reg::T1, Reg::T2) : divu(Reg::T1, Reg::T2));
        body.push_back(mfhi(Reg::T3));
        body.push_back(addu(Reg::T1, Reg::T3, Reg::T4));
      }
      EmitLoop(b, body);
      Report(is_signed ? "div (dependent)" : "divu (dependent)", b);
    }
    {
      Bench b;
      std::vector<u32> body;
      body.push_back(sll(Reg::T2, Reg::T0, 31));
      body.push_back(srl(Reg::T2, Reg::T2, 31));
      body.push_back(addiu(Reg::T2, Reg::T2, 2));
      for (u32 i = 0; i < BODY_REPEATS; i++)
      {
        body.push_back(addiu(Dst(i), Reg::T0, static_cast<s16>(1000 + i)));
        body.push_back(is_signed ? div(Dst(i), Reg::T2) : divu(Dst(i), Reg::T2));
        body.push_back(mflo(Dst(i)));
      }
      EmitLoop(b, body);
      Report(is_signed ? "div (independent)" : "divu (independent)", b);
    }
  }
}

TEST_P(CPUBench, BlockProtectCheck)
{
  // The same 63-instruction loop block on a write-protected page, and with its delay slot crossing a page so that
  // it has to compare its code against RAM every time it runs. The body is independent instructions, a dependency
  // chain would hide the check.
  for (const bool manual : {false, true})
  {
    Bench b;
    std::vector<u32> body;
    for (u32 i = 0; i < 60; i++)
      body.push_back(addiu(Dst(i), Reg::T2, 1));
    const u32 branch_index = EmitLoop(b, body);
    // RunAt() prefixes one instruction, so index i is at address + (i + 1) * 4.
    const u32 page_end = PROGRAM_BASE + HOST_PAGE_SIZE * 2;
    const u32 address = manual ? (page_end - (branch_index + 2) * sizeof(u32)) : PROGRAM_BASE;
    Report(manual ? "63-insn block (manual check)" : "63-insn block (protected)", b, address);
  }
}

INSTANTIATE_TEST_SUITE_P(ExecutionModes, CPUBench, ::testing::Values(CPUExecutionMode::Recompiler),
                         GetExecutionModeTestName);

} // namespace
