// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

// Win32 uses Fastjmp.asm, because MSVC doesn't support inline asm.
#if !defined(_WIN32) || defined(_M_ARM64)

#include "fastjmp.h"

#if defined(__APPLE__)
#define PREFIX "_"
#else
#define PREFIX ""
#endif

// clang-format off

#if defined(__x86_64__)

asm("\t.global " PREFIX "fastjmp_set\n"
	"\t.global " PREFIX "fastjmp_jmp\n"
	"\t.text\n"
	"\t" PREFIX "fastjmp_set:" R"(
	movq 0(%rsp), %rax
	movq %rsp, %rdx			# fixup stack pointer, so it doesn't include the call to fastjmp_set
	addq $8, %rdx
	movq %rax, 0(%rdi)	# actually rip
	movq %rbx, 8(%rdi)
	movq %rdx, 16(%rdi)	# actually rsp
	movq %rbp, 24(%rdi)
	movq %r12, 32(%rdi)
	movq %r13, 40(%rdi)
	movq %r14, 48(%rdi)
	movq %r15, 56(%rdi)
	xorl %eax, %eax
	ret
)"
	"\t" PREFIX "fastjmp_jmp:" R"(
	movl %esi, %eax
	movq 0(%rdi), %rdx	# actually rip
	movq 8(%rdi), %rbx
	movq 16(%rdi), %rsp	# actually rsp
	movq 24(%rdi), %rbp
	movq 32(%rdi), %r12
	movq 40(%rdi), %r13
	movq 48(%rdi), %r14
	movq 56(%rdi), %r15
	jmp *%rdx
)");

#elif defined(__aarch64__)

asm(
	"\t.global " PREFIX "fastjmp_set\n"
	"\t.global " PREFIX "fastjmp_jmp\n"
	"\t.text\n"
	"\t.align 16\n"
	"\t" PREFIX "fastjmp_set:" R"(
	mov x16, sp
	stp x16, x30, [x0]
	stp x19, x20, [x0, #16]
	stp x21, x22, [x0, #32]
	stp x23, x24, [x0, #48]
	stp x25, x26, [x0, #64]
	stp x27, x28, [x0, #80]
	str x29, [x0, #96]
	stp d8, d9, [x0, #112]
	stp d10, d11, [x0, #128]
	stp d12, d13, [x0, #144]
	stp d14, d15, [x0, #160]
	mov w0, wzr
	br x30
)"
".align 16\n"
"\t" PREFIX "fastjmp_jmp:" R"(
	ldp x16, x30, [x0]
	mov sp, x16
	ldp x19, x20, [x0, #16]
	ldp x21, x22, [x0, #32]
	ldp x23, x24, [x0, #48]
	ldp x25, x26, [x0, #64]
	ldp x27, x28, [x0, #80]
	ldr x29, [x0, #96]
	ldp d8, d9, [x0, #112]
	ldp d10, d11, [x0, #128]
	ldp d12, d13, [x0, #144]
	ldp d14, d15, [x0, #160]
	mov w0, w1
	br x30
)");

#elif defined(__arm__)

asm(
	"\t.global " PREFIX "fastjmp_set\n"
	"\t.global " PREFIX "fastjmp_jmp\n"
	"\t.text\n"
	"\t" PREFIX "fastjmp_set:" R"(
	vstmia r0!, {d8-d15}
	stmia r0!, {r4-r14}
	fmrx r1, fpscr
	str r1, [r0]
	mov r0, #0
	bx lr
)"

"\t" PREFIX "fastjmp_jmp:" R"(
	vldmia r0!, {d8-d15}
	ldmia r0!, {r4-r14}
	ldr r0, [r0]
	fmxr fpscr, r0
	mov r0, r1
	bx lr
)");

#elif defined(__riscv) && __riscv_xlen == 64

asm(
	"\t.global " PREFIX "fastjmp_set\n"
	"\t.global " PREFIX "fastjmp_jmp\n"
	"\t.attribute arch, \"rv64gc\"\n"
	"\t.text\n"
	"\t.align 16\n"
	"\t" PREFIX "fastjmp_set:" R"(
	sd sp, 0(a0)
	sd s0, 8(a0)
	sd s1, 16(a0)
	sd s2, 24(a0)
	sd s3, 32(a0)
	sd s4, 40(a0)
	sd s5, 48(a0)
	sd s6, 56(a0)
	sd s7, 64(a0)
	sd s8, 72(a0)
	sd s9, 80(a0)
	sd s10, 88(a0)
	sd s11, 96(a0)
	fsd fs0, 104(a0)
	fsd fs1, 112(a0)
	fsd fs2, 120(a0)
	fsd fs3, 128(a0)
	fsd fs4, 136(a0)
	fsd fs5, 144(a0)
	fsd fs6, 152(a0)
	fsd fs7, 160(a0)
	fsd fs8, 168(a0)
	fsd fs9, 176(a0)
	fsd fs10, 184(a0)
	fsd fs11, 192(a0)
	sd ra, 208(a0)
	li a0, 0
	jr ra
)"
".align 16\n"
"\t" PREFIX "fastjmp_jmp:" R"(
	ld ra, 208(a0)
	fld fs11, 192(a0)
	fld fs10, 184(a0)
	fld fs9, 176(a0)
	fld fs8, 168(a0)
	fld fs7, 160(a0)
	fld fs6, 152(a0)
	fld fs5, 144(a0)
	fld fs4, 136(a0)
	fld fs3, 128(a0)
	fld fs2, 120(a0)
	fld fs1, 112(a0)
	fld fs0, 104(a0)
	ld s11, 96(a0)
	ld s10, 88(a0)
	ld s9, 80(a0)
	ld s8, 72(a0)
	ld s7, 64(a0)
	ld s6, 56(a0)
	ld s5, 48(a0)
	ld s4, 40(a0)
	ld s3, 32(a0)
	ld s2, 24(a0)
	ld s1, 16(a0)
	ld s0, 8(a0)
	ld sp, 0(a0)
	mv a0, a1
	jr ra
)");

#elif defined(__loongarch64)

asm(
	"\t.global " PREFIX "fastjmp_set\n"
	"\t.global " PREFIX "fastjmp_jmp\n"
	"\t.text\n"
	"\t.align 16\n"
	"\t" PREFIX "fastjmp_set:" R"(
	st.d $sp, $a0, 0
	st.d $s0, $a0, 8
	st.d $s1, $a0, 16
	st.d $s2, $a0, 24
	st.d $s3, $a0, 32
	st.d $s4, $a0, 40
	st.d $s5, $a0, 48
	st.d $s6, $a0, 56
	st.d $s7, $a0, 64
	st.d $s8, $a0, 72
	st.d $s9, $a0, 80
	fst.d $fs0, $a0, 88
	fst.d $fs1, $a0, 96
	fst.d $fs2, $a0, 104
	fst.d $fs3, $a0, 112
	fst.d $fs4, $a0, 120
	fst.d $fs5, $a0, 128
	fst.d $fs6, $a0, 136
	fst.d $fs7, $a0, 144
	st.d $ra, $a0, 152
	addi.d $a0, $zero, 0
	jirl $zero, $ra, 0
)"
".align 16\n"
"\t" PREFIX "fastjmp_jmp:" R"(
	ld.d $ra, $a0, 152
	fld.d $fs7, $a0, 144
	fld.d $fs6, $a0, 136
	fld.d $fs5, $a0, 128
	fld.d $fs4, $a0, 120
	fld.d $fs3, $a0, 112
	fld.d $fs2, $a0, 104
	fld.d $fs1, $a0, 96
	fld.d $fs0, $a0, 88
	ld.d $s9, $a0, 80
	ld.d $s8, $a0, 72
	ld.d $s7, $a0, 64
	ld.d $s6, $a0, 56
	ld.d $s5, $a0, 48
	ld.d $s4, $a0, 40
	ld.d $s3, $a0, 32
	ld.d $s2, $a0, 24
	ld.d $s1, $a0, 16
	ld.d $s0, $a0, 8
	ld.d $sp, $a0, 0
	move $a0, $a1
	jirl $zero, $ra, 0
)");


#elif defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)

// ELFv2: r1, r2, r14-r31, f14-f31, v20-v31, CR2-CR4 and LR are preserved across calls.
// Layout: 0 r1, 8 r2, 16 r14-r31, 160 LR, 168 CR, 176 f14-f31, 320 v20-v31 (16-byte aligned).
asm(
  "\t.global " PREFIX "fastjmp_set\n"
  "\t.global " PREFIX "fastjmp_jmp\n"
  "\t.type " PREFIX "fastjmp_set, @function\n"
  "\t.type " PREFIX "fastjmp_jmp, @function\n"
  "\t.text\n"
  "\t.p2align 4\n"
  "\t" PREFIX "fastjmp_set:" R"(
  std 1, 0(3)
  std 2, 8(3)
  std 14, 16(3)
  std 15, 24(3)
  std 16, 32(3)
  std 17, 40(3)
  std 18, 48(3)
  std 19, 56(3)
  std 20, 64(3)
  std 21, 72(3)
  std 22, 80(3)
  std 23, 88(3)
  std 24, 96(3)
  std 25, 104(3)
  std 26, 112(3)
  std 27, 120(3)
  std 28, 128(3)
  std 29, 136(3)
  std 30, 144(3)
  std 31, 152(3)
  mflr 0
  std 0, 160(3)
  mfcr 0
  std 0, 168(3)
  stfd 14, 176(3)
  stfd 15, 184(3)
  stfd 16, 192(3)
  stfd 17, 200(3)
  stfd 18, 208(3)
  stfd 19, 216(3)
  stfd 20, 224(3)
  stfd 21, 232(3)
  stfd 22, 240(3)
  stfd 23, 248(3)
  stfd 24, 256(3)
  stfd 25, 264(3)
  stfd 26, 272(3)
  stfd 27, 280(3)
  stfd 28, 288(3)
  stfd 29, 296(3)
  stfd 30, 304(3)
  stfd 31, 312(3)
  li 5, 320
  stvx 20, 3, 5
  li 5, 336
  stvx 21, 3, 5
  li 5, 352
  stvx 22, 3, 5
  li 5, 368
  stvx 23, 3, 5
  li 5, 384
  stvx 24, 3, 5
  li 5, 400
  stvx 25, 3, 5
  li 5, 416
  stvx 26, 3, 5
  li 5, 432
  stvx 27, 3, 5
  li 5, 448
  stvx 28, 3, 5
  li 5, 464
  stvx 29, 3, 5
  li 5, 480
  stvx 30, 3, 5
  li 5, 496
  stvx 31, 3, 5
  li 3, 0
  blr
)"
  ".p2align 4\n"
  "\t" PREFIX "fastjmp_jmp:" R"(
  ld 0, 160(3)
  mtlr 0
  ld 0, 168(3)
  mtcrf 0x38, 0
  li 5, 320
  lvx 20, 3, 5
  li 5, 336
  lvx 21, 3, 5
  li 5, 352
  lvx 22, 3, 5
  li 5, 368
  lvx 23, 3, 5
  li 5, 384
  lvx 24, 3, 5
  li 5, 400
  lvx 25, 3, 5
  li 5, 416
  lvx 26, 3, 5
  li 5, 432
  lvx 27, 3, 5
  li 5, 448
  lvx 28, 3, 5
  li 5, 464
  lvx 29, 3, 5
  li 5, 480
  lvx 30, 3, 5
  li 5, 496
  lvx 31, 3, 5
  lfd 31, 312(3)
  lfd 30, 304(3)
  lfd 29, 296(3)
  lfd 28, 288(3)
  lfd 27, 280(3)
  lfd 26, 272(3)
  lfd 25, 264(3)
  lfd 24, 256(3)
  lfd 23, 248(3)
  lfd 22, 240(3)
  lfd 21, 232(3)
  lfd 20, 224(3)
  lfd 19, 216(3)
  lfd 18, 208(3)
  lfd 17, 200(3)
  lfd 16, 192(3)
  lfd 15, 184(3)
  lfd 14, 176(3)
  ld 31, 152(3)
  ld 30, 144(3)
  ld 29, 136(3)
  ld 28, 128(3)
  ld 27, 120(3)
  ld 26, 112(3)
  ld 25, 104(3)
  ld 24, 96(3)
  ld 23, 88(3)
  ld 22, 80(3)
  ld 21, 72(3)
  ld 20, 64(3)
  ld 19, 56(3)
  ld 18, 48(3)
  ld 17, 40(3)
  ld 16, 32(3)
  ld 15, 24(3)
  ld 14, 16(3)
  ld 2, 8(3)
  ld 1, 0(3)
  mr 3, 4
  blr
)");

#else

#error Unknown platform.

#endif

// clang-format on

#endif // __WIN32
