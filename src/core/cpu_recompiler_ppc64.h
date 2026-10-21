// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#pragma once

#include "cpu_recompiler.h"

#ifdef CPU_ARCH_PPC64

#include "cpu_recompiler_ppc64_emitter.h"

namespace CPU {

using PpcRRROp = void (ppcemit::Assembler::*)(ppcemit::GPR, ppcemit::GPR, ppcemit::GPR);

class PPC64Recompiler final : public Recompiler
{
public:
  PPC64Recompiler();
  ~PPC64Recompiler() override;

protected:
  const char* GetHostRegName(u32 reg) const override;

  const void* GetCurrentCodePointer() override;

  void LoadHostRegWithConstant(u32 reg, u32 val) override;
  void LoadHostRegFromCPUPointer(u32 reg, const void* ptr) override;
  void StoreConstantToCPUPointer(u32 val, const void* ptr) override;
  void StoreHostRegToCPUPointer(u32 reg, const void* ptr) override;
  void CopyHostReg(u32 dst, u32 src) override;

  void Reset(CodeCache::Block* block, u8* code_buffer, u32 code_buffer_space, u8* far_code_buffer,
             u32 far_code_space) override;
  void GenerateBlockProtectCheck(const u8* ram_ptr, const u8* shadow_ptr, u32 size) override;
  void GenerateICacheCheckAndUpdate() override;
  void GenerateCall(const void* func, s32 arg1reg = -1, s32 arg2reg = -1, s32 arg3reg = -1) override;
  void EndBlock(const std::optional<u32>& newpc, bool do_event_test) override;
  void EndBlockWithException(Exception excode) override;
  void EndAndLinkBlock(const std::optional<u32>& newpc, bool do_event_test, bool force_run_events);
  const void* EndCompile(u32* code_size, u32* far_code_size) override;

  void Flush(u32 flags) override;

  void Compile_Fallback() override;

  void CheckBranchTarget(ppcemit::GPR pcreg);
  void Compile_jr(CompileFlags cf) override;
  void Compile_jalr(CompileFlags cf) override;
  void Compile_bxx(CompileFlags cf, BranchCondition cond) override;

  void Compile_addi(CompileFlags cf, bool overflow);
  void Compile_addi(CompileFlags cf) override;
  void Compile_addiu(CompileFlags cf) override;
  void Compile_slti(CompileFlags cf, bool sign);
  void Compile_slti(CompileFlags cf) override;
  void Compile_sltiu(CompileFlags cf) override;
  void Compile_andi(CompileFlags cf) override;
  void Compile_ori(CompileFlags cf) override;
  void Compile_xori(CompileFlags cf) override;

  void Compile_shift(CompileFlags cf, void (PPC64Recompiler::*op_const)(ppcemit::GPR, ppcemit::GPR, u32));
  void Compile_sll(CompileFlags cf) override;
  void Compile_srl(CompileFlags cf) override;
  void Compile_sra(CompileFlags cf) override;
  void Compile_variable_shift(CompileFlags cf, PpcRRROp op,
                              void (PPC64Recompiler::*op_const)(ppcemit::GPR, ppcemit::GPR, u32));
  void Compile_sllv(CompileFlags cf) override;
  void Compile_srlv(CompileFlags cf) override;
  void Compile_srav(CompileFlags cf) override;
  void Compile_mult(CompileFlags cf, bool sign);
  void Compile_mult(CompileFlags cf) override;
  void Compile_multu(CompileFlags cf) override;
  void Compile_div(CompileFlags cf) override;
  void Compile_divu(CompileFlags cf) override;
  void TestOverflow(ppcemit::GPR long_res, ppcemit::GPR res, ppcemit::GPR reg_to_discard);
  void Compile_dst_op(CompileFlags cf, PpcRRROp op, void (PPC64Recompiler::*op_const)(ppcemit::GPR rd, ppcemit::GPR rs, u32 imm),
                      bool is_sub, bool commutative, bool overflow);
  void Compile_add(CompileFlags cf) override;
  void Compile_addu(CompileFlags cf) override;
  void Compile_sub(CompileFlags cf) override;
  void Compile_subu(CompileFlags cf) override;
  void Compile_and(CompileFlags cf) override;
  void Compile_or(CompileFlags cf) override;
  void Compile_xor(CompileFlags cf) override;
  void Compile_nor(CompileFlags cf) override;
  void Compile_slt(CompileFlags cf, bool sign);
  void Compile_slt(CompileFlags cf) override;
  void Compile_sltu(CompileFlags cf) override;

  ppcemit::GPR ComputeLoadStoreAddressArg(CompileFlags cf, const std::optional<VirtualMemoryAddress>& address,
                                          const std::optional<ppcemit::GPR>& reg = std::nullopt);
  template<typename RegAllocFn>
  ppcemit::GPR GenerateLoad(ppcemit::GPR addr_reg, MemoryAccessSize size, bool sign, bool use_fastmem,
                            const RegAllocFn& dst_reg_alloc);
  void GenerateStore(ppcemit::GPR addr_reg, ppcemit::GPR value_reg, MemoryAccessSize size, bool use_fastmem);
  void Compile_lxx(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                   const std::optional<VirtualMemoryAddress>& address) override;
  void Compile_lwx(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                   const std::optional<VirtualMemoryAddress>& address) override;
  void Compile_lwc2(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                    const std::optional<VirtualMemoryAddress>& address) override;
  void Compile_sxx(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                   const std::optional<VirtualMemoryAddress>& address) override;
  void Compile_swx(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                   const std::optional<VirtualMemoryAddress>& address) override;
  void Compile_swc2(CompileFlags cf, MemoryAccessSize size, bool sign, bool use_fastmem,
                    const std::optional<VirtualMemoryAddress>& address) override;

  void TestInterrupts(ppcemit::GPR sr);
  void Compile_mtc0(CompileFlags cf) override;
  void Compile_rfe(CompileFlags cf) override;

  void Compile_mfc2(CompileFlags cf) override;
  void Compile_mtc2(CompileFlags cf) override;
  void Compile_cop2(CompileFlags cf) override;

  void GeneratePGXPCallWithMIPSRegs(const void* func, u32 arg1val, Reg arg2reg = Reg::count,
                                    Reg arg3reg = Reg::count) override;

private:
  void EmitMov(ppcemit::GPR dst, u32 val);
  void EmitCall(const void* ptr);

  /// Branches to far code if the condition (on CR0) holds. Without a condition, branches unconditionally.
  void SwitchToFarCode(bool emit_jump, std::optional<ppcemit::Cond> cond = std::nullopt);
  void SwitchToNearCode(bool emit_jump);

  void AssertRegOrConstS(CompileFlags cf) const;
  void AssertRegOrConstT(CompileFlags cf) const;

  // 32-bit operations with arbitrary immediates. Only the low word of the result is meaningful.
  void SafeADDIW(ppcemit::GPR rd, ppcemit::GPR rs, u32 imm);
  void SafeSUBIW(ppcemit::GPR rd, ppcemit::GPR rs, u32 imm);
  void SafeANDI(ppcemit::GPR rd, ppcemit::GPR rs, u32 imm);
  void SafeORI(ppcemit::GPR rd, ppcemit::GPR rs, u32 imm);
  void SafeXORI(ppcemit::GPR rd, ppcemit::GPR rs, u32 imm);
  void SafeSLTI(ppcemit::GPR rd, ppcemit::GPR rs, u32 imm);
  void SafeSLTIU(ppcemit::GPR rd, ppcemit::GPR rs, u32 imm);
  void EmitSLLI(ppcemit::GPR rd, ppcemit::GPR rs, u32 sh);
  void EmitSRLI(ppcemit::GPR rd, ppcemit::GPR rs, u32 sh);
  void EmitSRAI(ppcemit::GPR rd, ppcemit::GPR rs, u32 sh);

  void EmitSExtB(ppcemit::GPR rd, ppcemit::GPR rs);
  void EmitUExtB(ppcemit::GPR rd, ppcemit::GPR rs);
  void EmitSExtH(ppcemit::GPR rd, ppcemit::GPR rs);
  void EmitUExtH(ppcemit::GPR rd, ppcemit::GPR rs);

  ppcemit::GPR CFGetSafeRegS(CompileFlags cf, ppcemit::GPR temp_reg);
  ppcemit::GPR CFGetSafeRegT(CompileFlags cf, ppcemit::GPR temp_reg);

  ppcemit::GPR CFGetRegD(CompileFlags cf) const;
  ppcemit::GPR CFGetRegS(CompileFlags cf) const;
  ppcemit::GPR CFGetRegT(CompileFlags cf) const;
  ppcemit::GPR CFGetRegLO(CompileFlags cf) const;
  ppcemit::GPR CFGetRegHI(CompileFlags cf) const;

  void MoveSToReg(ppcemit::GPR dst, CompileFlags cf);
  void MoveTToReg(ppcemit::GPR dst, CompileFlags cf);
  void MoveMIPSRegToReg(ppcemit::GPR dst, Reg reg, bool ignore_load_delays = false);

  ppcemit::Assembler m_emitter;
  ppcemit::Assembler m_far_emitter;
  ppcemit::Assembler* ppcAsm = nullptr;
};

} // namespace CPU

#endif // CPU_ARCH_PPC64
