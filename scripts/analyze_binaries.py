#!/usr/bin/env python3
"""CUDA 工具链二进制分析脚本。

用于《Deep Dive Into CUDA》第 2 章：把 strings/nm 层面的"猜测"升级为
可复现的二进制证据。所有输出均来自真实二进制，可直接引用到文档。

用法（需先激活 venv，见 README/agents.md）::

    source $HOME/soft/venv/myenv/bin/activate

    # 一次性跑完 ELF 结构、符号、字符串交叉引用、调用图
    python scripts/analyze_binaries.py report --target cicc

    # 单跑某一项
    python scripts/analyze_binaries.py elf     --target libnvvm
    python scripts/analyze_binaries.py symbols --target cicc
    python scripts/analyze_binaries.py xref    --target cicc
    python scripts/analyze_binaries.py calls   --target cicc

    # angr 控制流图（较重，默认只分析 .text 前若干字节，可用 --full 全量）
    python scripts/analyze_binaries.py angr    --target ptxas --max-bytes 2000000

    # 调用图为流式解码；对超大 .text 可再加 --max-bytes 限制每节区解码量
    python scripts/analyze_binaries.py calls   --target /usr/local/cuda/bin/tileiras \
        --max-bytes 8000000

依赖：pyelftools、capstone；angr 仅 "angr" 子命令需要。
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

from elftools.elf.elffile import ELFFile

import capstone

# ---------------------------------------------------------------------------
# 常量与目标别名
# ---------------------------------------------------------------------------

CUDA_ROOT = Path("/usr/local/cuda")

TARGETS = {
    "cicc": CUDA_ROOT / "nvvm/bin/cicc",
    "libnvvm": CUDA_ROOT / "nvvm/lib64/libnvvm.so.4.0.0",
    "ptxas": CUDA_ROOT / "bin/ptxas",
    "nvlink": CUDA_ROOT / "bin/nvlink",
    "nvcc": CUDA_ROOT / "bin/nvcc",
}

SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4

# 默认交叉引用目标：NVVM C API + 动态加载线索
DEFAULT_XREF_PATTERNS = [
    "nvvmCompileProgram",
    "nvvmCreateProgram",
    "nvvmAddModuleToProgram",
    "nvvmGetCompiledResult",
    "__nvvm_reflect",
    "libnvvm.so",
]


def resolve_target(name: str) -> Path:
    """把别名或路径解析为实际文件路径。"""
    if name in TARGETS:
        return TARGETS[name]
    return Path(name)


def human(n: int) -> str:
    return f"{n:,}"


def sec_flags(sec) -> str:
    f = sec["sh_flags"]
    return "".join(
        c
        for c, bit in (("W", SHF_WRITE), ("A", SHF_ALLOC), ("X", SHF_EXECINSTR))
        if f & bit
    ) or "-"


# ---------------------------------------------------------------------------
# 1) ELF 结构
# ---------------------------------------------------------------------------


def cmd_elf(path: Path) -> None:
    size = path.stat().st_size
    with open(path, "rb") as f:
        elf = ELFFile(f)
        sec_names = [s.name for s in elf.iter_sections()]
        stripped = ".symtab" not in sec_names

        print(f"=== ELF 结构: {path.name} ===")
        print(f"路径        : {path}")
        print(f"文件大小    : {human(size)} 字节 ({size / 1024 / 1024:.1f} MiB)")
        print(f"类型        : {elf.header['e_type']}  (ET_EXEC=固定地址, ET_DYN=PIE/共享库)")
        print(f"机器        : {elf.header['e_machine']}")
        print(f"入口地址    : {hex(elf.header['e_entry'])}")
        print(f"节区数量    : {len(sec_names)}")
        print(f"符号表      : {'已剥离 (无 .symtab)' if stripped else '保留 .symtab'}")
        print()

        print("-- 可分配节区 (>64KB) --")
        print(f"{'节区':<18}{'大小':>14}  {'标志':<4} 地址")
        for sec in elf.iter_sections():
            if not (sec["sh_flags"] & SHF_ALLOC):
                continue
            if sec["sh_size"] < 65536:
                continue
            print(
                f"{sec.name:<18}{human(sec['sh_size']):>14}  "
                f"{sec_flags(sec):<4} {hex(sec['sh_addr'])}"
            )

        exec_total = sum(
            s["sh_size"] for s in elf.iter_sections() if s["sh_flags"] & SHF_EXECINSTR
        )
        print()
        print(f"可执行字节总数: {human(exec_total)} 字节 ({exec_total / size * 100:.1f}% of file)")


# ---------------------------------------------------------------------------
# 2) 符号：导出 / 导入
# ---------------------------------------------------------------------------


def _dynsym(elf):
    return elf.get_section_by_name(".dynsym")


def _dynsym_names(elf, predicate=None):
    dsym = _dynsym(elf)
    if dsym is None:
        return []
    out = []
    for sym in dsym.iter_symbols():
        if not sym.name:
            continue
        if predicate is None or predicate(sym):
            out.append(sym)
    return out


def cmd_symbols(path: Path) -> None:
    with open(path, "rb") as f:
        elf = ELFFile(f)
        dsym = _dynsym(elf)
        total = len(list(dsym.iter_symbols())) if dsym else 0
        undef = sorted(
            {s.name for s in _dynsym_names(elf, lambda s: s["st_shndx"] == "SHN_UNDEF")}
        )
        defined = _dynsym_names(
            elf, lambda s: s["st_shndx"] != "SHN_UNDEF" and s["st_value"] != 0
        )
        nvvm_exports = sorted({s.name for s in defined if "nvvm" in s.name.lower()})
        nvvm_undef = sorted({n for n in undef if "nvvm" in n.lower()})
        dl_calls = sorted({n for n in undef if n.startswith("dl")})

        print(f"=== 符号: {path.name} ===")
        print(f"动态符号总数 : {human(total)}")
        print(f"  未定义(导入): {human(len(undef))}")
        print(f"  已定义(导出): {human(len(defined))}")
        print()

        if nvvm_exports:
            print(f"-- 导出的 NVVM API 符号 ({len(nvvm_exports)} 个) --")
            for n in nvvm_exports:
                print(f"   {n}")
            print()

        print("-- 动态加载线索 (libdl) --")
        if dl_calls:
            print(f"   导入: {', '.join(dl_calls)}")
        else:
            print("   无")
        print()

        print("-- NVVM API 耦合方式判定 --")
        if nvvm_undef:
            print(f"   cicc 导入 nvvm* 符号 {len(nvvm_undef)} 个 -> 静态链接/PLT 直调")
            for n in nvvm_undef:
                print(f"     {n}")
        elif nvvm_exports:
            print("   该目标自身导出 NVVM API -> 它就是编译核心库")
        else:
            print("   dynsym 中不含 nvvm* 符号")
            if dl_calls:
                print("   + 存在 dlopen/dlsym 导入 -> 通过运行时动态加载调用 libnvvm (dlopen 路径)")
                print("   结论: cicc 与 libnvvm 之间是 dlopen/dlsym 弱耦合，而非静态链接")
            else:
                print("   结论: 需结合 xref 进一步判断")


# ---------------------------------------------------------------------------
# 3) 字符串 -> 代码交叉引用
# ---------------------------------------------------------------------------


def locate_strings(elf, patterns):
    """在非可执行的已分配节区中定位字符串，返回 {vaddr: [(pattern, section)]}。"""
    hits = defaultdict(list)
    for sec in elf.iter_sections():
        flags = sec["sh_flags"]
        if not (flags & SHF_ALLOC) or (flags & SHF_EXECINSTR):
            continue
        data = sec.data()
        base = sec["sh_addr"]
        for pat in patterns:
            needle = pat.encode()
            start = 0
            while True:
                i = data.find(needle, start)
                if i < 0:
                    break
                # 只接受 C 字符串起点（前一字节为 \0 或段首）
                if i == 0 or data[i - 1] == 0:
                    hits[base + i].append((pat, sec.name))
                start = i + 1
    return hits


def cmd_xref(path: Path, patterns) -> None:
    with open(path, "rb") as f:
        elf = ELFFile(f)
        strings = locate_strings(elf, patterns)

        print(f"=== 字符串交叉引用: {path.name} ===")
        print(f"目标字符串 {len(patterns)} 个, 在数据段命中 {len(strings)} 处地址")
        print()

        target_addrs = set(strings)
        found = defaultdict(list)

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        md.skipdata = True

        t0 = time.time()
        for sec in elf.iter_sections():
            if not (sec["sh_flags"] & SHF_EXECINSTR):
                continue
            for insn in iter_insns(f, sec, md):
                if not insn.operands:
                    continue
                for op in insn.operands:
                    tgt = None
                    # RIP 相对寻址: lea/mov reg, [rip+disp]
                    if op.type == capstone.x86.X86_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                        tgt = insn.address + insn.size + op.mem.disp
                    # 立即数寻址: ET_EXEC 非 PIE 下常见 mov reg, <abs32>
                    elif op.type == capstone.x86.X86_OP_IMM:
                        tgt = op.imm
                    if tgt in target_addrs:
                        found[tgt].append(
                            (sec.name, insn.address, f"{insn.mnemonic} {insn.op_str}")
                        )
        elapsed = time.time() - t0

        for addr in sorted(strings):
            pat, sect = strings[addr][0]
            refs = found.get(addr, [])
            print(f'"{pat}"  @ {hex(addr)} ({sect})  -> {len(refs)} 处代码引用')
            for sec_name, iaddr, text in refs[:6]:
                print(f"     {sec_name}:{hex(iaddr)}  {text}")
            if len(refs) > 6:
                print(f"     ... 其余 {len(refs) - 6} 处省略")
        print()
        print(f"扫描耗时: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# 4) 静态调用图（直接 call + PLT 命名）
# ---------------------------------------------------------------------------


def iter_insns(f, sec, md, max_bytes: int | None = None, chunk_size: int = 4 << 20):
    """按块从文件流式解码可执行节区，避免整段读入内存。

    以 ``chunk_size``（默认 4 MiB）为单位读取；若某块在指令中途结束，
    则回退到块内最后一条完整指令的末尾续读，保证与整段解码结果一致。
    对超大 ``.text``（如 tileiras 的 69 MB）可将峰值内存压到常数级。
    """
    remaining = sec["sh_size"] if max_bytes is None else min(sec["sh_size"], max_bytes)
    f.seek(sec["sh_offset"])
    base = sec["sh_addr"]
    while remaining > 0:
        data = f.read(min(chunk_size, remaining))
        if not data:
            break
        consumed = 0
        for insn in md.disasm(data, base):
            consumed = insn.address + insn.size - base
            yield insn
        step = consumed if consumed > 0 else len(data)
        f.seek(step - len(data), 1)   # 回退未用到的尾部字节
        base += step
        remaining -= step


def build_plt_map(elf):
    """返回 {plt_stub_addr: symbol_name}。兼容 .plt.sec 与 .plt。"""
    m = {}
    rela = elf.get_section_by_name(".rela.plt") or elf.get_section_by_name(".rel.plt")
    dynsym = elf.get_section_by_name(".dynsym")
    if rela is None or dynsym is None:
        return m
    plt_sec = elf.get_section_by_name(".plt.sec")
    plt = elf.get_section_by_name(".plt")
    entsize = 16
    for s in (plt_sec, plt):
        if s is not None and s["sh_entsize"]:
            entsize = s["sh_entsize"]
            break
    for k, reloc in enumerate(rela.iter_relocations()):
        sym = dynsym.get_symbol(reloc["r_info_sym"])
        if plt_sec is not None:
            addr = plt_sec["sh_addr"] + k * entsize      # .plt.sec 无 PLT0
        elif plt is not None:
            addr = plt["sh_addr"] + (k + 1) * entsize    # .plt 首项为 PLT0
        else:
            continue
        m[addr] = sym.name
    return m


def collect_direct_calls(f, elf, max_bytes: int | None = None):
    """返回 (callee_addr -> count)。流式解码，避免整段 .text 占用内存。"""
    counter = Counter()
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    md.skipdata = True
    for sec in elf.iter_sections():
        if not (sec["sh_flags"] & SHF_EXECINSTR):
            continue
        for insn in iter_insns(f, sec, md, max_bytes):
            if insn.mnemonic != "call" or not insn.operands:
                continue
            op = insn.operands[0]
            if op.type == capstone.x86.X86_OP_IMM:
                counter[op.imm] += 1
    return counter


def cmd_calls(path: Path, top: int, max_bytes: int | None = None) -> None:
    with open(path, "rb") as f:
        elf = ELFFile(f)
        plt = build_plt_map(elf)
        calls = collect_direct_calls(f, elf, max_bytes)

    total_sites = sum(calls.values())
    print(f"=== 静态调用图: {path.name} ===")
    if max_bytes:
        print(f"(仅统计每个可执行节区前 {human(max_bytes)} 字节)")
    print(f"直接 call 指令      : {human(total_sites)} 处")
    print(f"不同被调目标        : {human(len(calls))} 个")
    print(f"其中 PLT 桩(外部)   : {human(len(plt))} 个")
    print()

    ext = Counter()
    internal = Counter()
    for addr, cnt in calls.items():
        if addr in plt:
            ext[plt[addr]] += cnt
        else:
            internal[addr] += cnt

    print(f"-- 被调用最多的外部函数 (Top {top}) --")
    for name, cnt in ext.most_common(top):
        print(f"   {cnt:>7} 次  {name}")
    print()

    print(f"-- 被调用最多的内部函数 (Top {top}, 无符号名) --")
    for addr, cnt in internal.most_common(top):
        print(f"   {cnt:>7} 次  {hex(addr)}")
    print()
    print(f"外部调用占比: {sum(ext.values()) / total_sites * 100:.1f}%  "
          f"(外部 {human(sum(ext.values()))} / 内部 {human(sum(internal.values()))})")


# ---------------------------------------------------------------------------
# 5) angr 控制流图
# ---------------------------------------------------------------------------


def cmd_angr(path: Path, max_bytes: int | None, full: bool) -> None:
    import logging

    import angr

    logging.getLogger("angr").setLevel("ERROR")
    logging.getLogger("cle").setLevel("ERROR")

    t0 = time.time()
    proj = angr.Project(str(path), auto_load_libs=False)
    regs = None
    if not full and max_bytes:
        text = proj.loader.main_object.sections_map.get(".text")
        if text is not None:
            regs = [(text.vaddr, text.vaddr + min(max_bytes, text.memsize))]
    cfg = proj.analyses.CFGFast(
        regions=regs,
        resolve_indirect_jumps=False,
        data_references=False,
        normalize=True,
    )
    funcs = cfg.kb.functions
    called = Counter()
    for fn in funcs.values():
        for _, callee in fn.transition_graph.edges():
            if hasattr(callee, "addr"):
                called[callee.addr] += 1

    print(f"=== angr CFGFast: {path.name} ===")
    scope = "全量 .text" if regs is None else f"{human(max_bytes or 0)} 字节 (前部)"
    print(f"分析范围        : {scope}")
    print(f"识别函数数      : {human(len(funcs))}")
    print(f"耗时            : {time.time() - t0:.1f}s")
    print()
    print("-- 被最多函数调用的目标 (Top 10) --")
    for addr, cnt in called.most_common(10):
        name = funcs[addr].name if addr in funcs else "?"
        print(f"   {cnt:>5} 个调用者  {hex(addr)}  {name}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="CUDA 工具链二进制分析")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument(
            "--target",
            default="cicc",
            help="目标别名 (cicc/libnvvm/ptxas/nvlink/nvcc) 或绝对路径",
        )
        return p

    add("elf", "ELF 结构与节区概览")
    add("symbols", "导出/导入符号与耦合方式判定")

    p_xref = add("xref", "字符串到代码的交叉引用")
    p_xref.add_argument(
        "--patterns",
        default="",
        help="逗号分隔的自定义字符串（默认使用 NVVM API 一组）",
    )

    p_calls = add("calls", "静态直接调用图（流式解码，可用 --max-bytes 限流）")
    p_calls.add_argument("--top", type=int, default=15, help="每类展示条数")
    p_calls.add_argument(
        "--max-bytes", type=int, default=None,
        help="每个可执行节区最多解码的字节数（超大 .text 防 OOM）",
    )

    p_angr = add("angr", "angr CFGFast 控制流图")
    p_angr.add_argument("--max-bytes", type=int, default=2_000_000, help="默认分析字节数")
    p_angr.add_argument("--full", action="store_true", help="分析全部 .text（很慢）")

    p_rep = add("report", "依次执行 elf + symbols + xref + calls")
    p_rep.add_argument("--top", type=int, default=15)
    p_rep.add_argument(
        "--max-bytes", type=int, default=None,
        help="calls 阶段每个可执行节区最多解码的字节数（防 OOM）",
    )

    args = ap.parse_args()
    path = resolve_target(args.target)
    if not path.exists():
        raise SystemExit(f"目标不存在: {path}")

    if args.cmd == "elf":
        cmd_elf(path)
    elif args.cmd == "symbols":
        cmd_symbols(path)
    elif args.cmd == "xref":
        patterns = (
            [p.strip() for p in args.patterns.split(",") if p.strip()]
            if args.patterns
            else DEFAULT_XREF_PATTERNS
        )
        cmd_xref(path, patterns)
    elif args.cmd == "calls":
        cmd_calls(path, args.top, args.max_bytes)
    elif args.cmd == "angr":
        cmd_angr(path, args.max_bytes, args.full)
    elif args.cmd == "report":
        cmd_elf(path)
        print()
        cmd_symbols(path)
        print()
        cmd_xref(path, DEFAULT_XREF_PATTERNS)
        print()
        cmd_calls(path, args.top, args.max_bytes)


if __name__ == "__main__":
    main()
