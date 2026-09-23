#!/usr/bin/env python3
# Copyright (c) 2026 Hans-Kristian Arntzen
# SPDX-License-Identifier: MIT
#
# Regenerates the D3D12 (DXIL) shaders from the GLSL sources in ../../shaders.
#
#   GLSL --(shims)--> GLSL' --glslc--> SPIR-V --spirv-cross--> HLSL SM 6.4 --dxc--> DXIL
#
# The HLSL is committed next to this script (hlsl/) for readability and debugging,
# and the DXIL is embedded in generated/pyrowave_dxil.h, so building the library
# needs none of these tools. Run from anywhere; on WSL the Windows tools are used
# through /mnt/c. Tool locations can be overridden with PYROWAVE_GLSLC,
# PYROWAVE_SPIRV_CROSS and PYROWAVE_DXC.
#
# Why shims are needed (SPIRV-Cross HLSL cannot express these, or does it wrong):
#   - gl_SubgroupID / gl_NumSubgroups have no HLSL builtin. Derived from
#     gl_LocalInvocationIndex, which relies on the linear thread->lane mapping
#     every D3D12 driver we target uses (verified on Xbox Series and RDNA3).
#   - subgroupClustered* has no HLSL equivalent. Rewritten as shuffle-xor
#     butterflies (encoder only).
#   - The Xbox Series UWP driver treats the WaveReadLaneAt lane index as
#     wave-uniform (lane 0's index is used for every lane), so every
#     subgroupShuffle* that SPIRV-Cross lowers to WaveReadLaneAt silently computes
#     garbage there. Shaders in the portable profile must not contain any; this
#     script fails if one survives.
#   - image2D outputs become one-layer image2DArray, so the same UAV declaration
#     can target both a slice of the wavelet pyramid (a Texture2DArray) and a
#     plain Texture2D output plane.
#   - Likewise sampler2D inputs become one-layer sampler2DArray.
#   - HLSL has no 8-bit type at all, so uint8_t/uint16_t SSBO arrays (the encoder's
#     payload and bitstream buffers) become uint word arrays. Narrow stores do an
#     atomicAnd/atomicOr pair on the containing word: every byte has one writer, but
#     neighbouring threads' byte runs share words. Byte loads extract with shifts.
#
# Profiles:
#   portable  SM 6.4, no 16-bit types, no WaveReadLaneAt. Runs on Xbox Series UWP.
#             Used for everything the decoder needs.
#   desktop   SM 6.6 with native 16-bit types and [WaveSize(64)]. The encoder's rate
#             control stores FP16/u16 statistics, and resolve_rate_control needs its
#             workgroup to be exactly one wave. The encoder runs on the host PC.

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GLSL_DIR = ROOT / 'shaders'
HLSL_DIR = HERE / 'hlsl'
GENERATED = HERE / 'generated' / 'pyrowave_dxil.h'

ON_WSL = Path('/mnt/c').is_dir() and os.name != 'nt'
WIN = '/mnt/c' if ON_WSL else 'C:'

GLSLC = os.environ.get('PYROWAVE_GLSLC', f'{WIN}/VulkanSDK/1.3.268.0/Bin/glslc.exe')
SPIRV_CROSS = os.environ.get('PYROWAVE_SPIRV_CROSS', f'{WIN}/VulkanSDK/1.3.268.0/Bin/spirv-cross.exe')
DXC = os.environ.get('PYROWAVE_DXC', f'{WIN}/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe')

# (output name, GLSL source, glslc defines, dxc defines, profile)
VARIANTS = [
	# Decoder (portable).
	('wavelet_dequant', 'wavelet_dequant.comp', ['STORAGE_MODE=1'], [], 'portable'),
	('idwt_p2', 'idwt.comp', ['FP16=0', 'PRECISION=2'], [], 'portable'),
	('idwt_p2_dc', 'idwt.comp', ['FP16=0', 'PRECISION=2'], ['SPIRV_CROSS_CONSTANT_ID_0=true'], 'portable'),
	('idwt_p1', 'idwt.comp', ['FP16=0', 'PRECISION=1'], [], 'portable'),
	('idwt_p1_dc', 'idwt.comp', ['FP16=0', 'PRECISION=1'], ['SPIRV_CROSS_CONSTANT_ID_0=true'], 'portable'),
	# Encoder (desktop).
	('dwt_p2', 'dwt.comp', ['FP16=0', 'PRECISION=2'], [], 'desktop'),
	('dwt_p2_dc', 'dwt.comp', ['FP16=0', 'PRECISION=2'], ['SPIRV_CROSS_CONSTANT_ID_0=true'], 'desktop'),
	('dwt_p1', 'dwt.comp', ['FP16=0', 'PRECISION=1'], [], 'desktop'),
	('dwt_p1_dc', 'dwt.comp', ['FP16=0', 'PRECISION=1'], ['SPIRV_CROSS_CONSTANT_ID_0=true'], 'desktop'),
	('wavelet_quant', 'wavelet_quant.comp', [], [], 'desktop'),
	('analyze_rate_control', 'analyze_rate_control.comp', [], [], 'desktop'),
	('analyze_rate_control_finalize', 'analyze_rate_control_finalize.comp', [], [], 'desktop'),
	# Workgroup size is a specialization constant that must equal the wave size.
	('resolve_rate_control', 'resolve_rate_control.comp', [], ['SPIRV_CROSS_CONSTANT_ID_0=64'], 'desktop'),
	('block_packing', 'block_packing.comp', [], [], 'desktop'),
]


def win_path(p):
	if not ON_WSL:
		return str(p)
	return subprocess.check_output(['wslpath', '-w', str(p)], text=True).strip()


def run(cmd):
	r = subprocess.run(cmd, capture_output=True, text=True)
	if r.returncode != 0:
		sys.stderr.write(f"FAILED: {' '.join(cmd)}\n{r.stdout}{r.stderr}\n")
		sys.exit(1)


def matching_paren(s, open_index):
	depth = 0
	for i in range(open_index, len(s)):
		if s[i] == '(':
			depth += 1
		elif s[i] == ')':
			depth -= 1
			if depth == 0:
				return i
	raise ValueError('unbalanced parentheses')


def split_top_level_args(s):
	args, depth, start = [], 0, 0
	for i, c in enumerate(s):
		if c == '(':
			depth += 1
		elif c == ')':
			depth -= 1
		elif c == ',' and depth == 0:
			args.append(s[start:i])
			start = i + 1
	args.append(s[start:])
	return args


def rewrite_calls(s, name, fn):
	"""Replaces every call name(args...) with fn(list_of_arg_strings)."""
	out, pos = [], 0
	pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
	while True:
		m = pat.search(s, pos)
		if not m:
			out.append(s[pos:])
			return ''.join(out)
		close = matching_paren(s, m.end() - 1)
		out.append(s[pos:m.start()])
		out.append(fn(split_top_level_args(s[m.end():close])))
		pos = close + 1


def clustered_helpers(op, n):
	steps = [1 << i for i in range(n.bit_length() - 1)]
	out = []
	for t in ('int', 'uint', 'float'):
		body = ''
		for step in steps:
			other = f'subgroupShuffleXor(v, {step}u)'
			body += f'\tv = {"v + " + other if op == "add" else "max(v, " + other + ")"};\n'
		out.append(f'{t} pw_clustered_{op}_{n}({t} v)\n{{\n{body}\treturn v;\n}}\n')
	return ''.join(out)



def find_matching(s, open_index, open_char, close_char):
	depth = 0
	for i in range(open_index, len(s)):
		if s[i] == open_char:
			depth += 1
		elif s[i] == close_char:
			depth -= 1
			if depth == 0:
				return i
	raise ValueError('unbalanced ' + open_char)


def strip_cast(expr, cast):
	e = expr.strip()
	if e.startswith(cast + '(') and find_matching(e, len(cast), '(', ')') == len(e) - 1:
		return e[len(cast) + 1:-1].strip()
	return e


def rewrite_narrow_ssbos(s):
	"""uint8_t/uint16_t SSBO arrays -> uint words + helper calls (see header)."""
	helpers = ''
	# Narrow member arrays declared as '<type> data[];' inside a named buffer block.
	block_re = re.compile(r'layout\(([^)]*)\)\s*(writeonly\s+|readonly\s+)?buffer\s+(\w+)\s*\{(.*?)\}\s*(\w+)\s*;', re.S)
	blocks = {m.group(5): m for m in block_re.finditer(s)}
	narrow = {}   # instance name -> (word array expression, bits)
	# Aliases of a uint block at the same binding (block_packing's 16/8-bit views).
	by_binding = {}
	for name, m in blocks.items():
		by_binding.setdefault(m.group(1).replace(' ', ''), []).append(name)
	edits = []
	for name, m in blocks.items():
		body = m.group(4)
		nm = re.search(r'(uint8_t|uint16_t)\s+data\s*\[\s*\]\s*;', body)
		if not nm:
			continue
		bits = 8 if nm.group(1) == 'uint8_t' else 16
		siblings = [o for o in by_binding[m.group(1).replace(' ', '')] if o != name and
		            re.search(r'\buint\s+data\s*\[\s*\]', blocks[o].group(4))]
		if siblings:
			# A 32-bit alias exists: drop this block, write through the alias.
			narrow[name] = (siblings[0] + '.data', bits)
			edits.append((m.start(), m.end(), ''))
		else:
			narrow[name] = (name + '.pw_words', bits)
			new_body = body[:nm.start()] + 'uint pw_words[];' + body[nm.end():]
			edits.append((m.start(4), m.end(4), new_body))
	if not narrow:
		return s, ''
	# The atomics need read/write access on every block that is written narrowly.
	targets = {v[0].split('.')[0] for v in narrow.values()}
	for name in targets:
		m = blocks[name]
		if m.group(2) and m.group(2).strip() == 'writeonly':
			edits.append((m.start(2), m.end(2), ''))
	for start, end, text in sorted(edits, key=lambda e: -e[0]):
		s = s[:start] + text + s[end:]

	used = set()
	pat = re.compile(r'\b(' + '|'.join(map(re.escape, narrow)) + r')\.data\s*\[')
	out, pos = [], 0
	while True:
		m = pat.search(s, pos)
		if not m:
			out.append(s[pos:])
			break
		name = m.group(1)
		close = find_matching(s, m.end() - 1, '[', ']')
		index = s[m.end():close].strip()
		rest = s[close + 1:]
		store = re.match(r'\s*=(?!=)', rest)
		out.append(s[pos:m.start()])
		words, bits = narrow[name]
		key = words.replace('.', '_') + str(bits)
		if store:
			semi = close + 1 + store.end()
			depth = 0
			while not (s[semi] == ';' and depth == 0):
				depth += s[semi] in '([' ; depth -= s[semi] in ')]'
				semi += 1
			value = strip_cast(s[close + 1 + store.end():semi], 'uint8_t' if bits == 8 else 'uint16_t')
			out.append(f'pw_store{bits}_{key}({index}, uint({value}))')
			used.add(('store', words, bits, key))
			pos = semi
		else:
			out.append(f'pw_load{bits}_{key}({index})')
			used.add(('load', words, bits, key))
			pos = close + 1
	s = ''.join(out)

	for kind, words, bits, key in sorted(used):
		per = 32 // bits
		mask = (1 << bits) - 1
		if kind == 'store':
			helpers += (f'void pw_store{bits}_{key}(uint i, uint v)\n{{\n'
			            f'\tuint shift = (i % {per}u) * {bits}u;\n'
			            f'\tatomicAnd({words}[i / {per}u], ~(0x{mask:x}u << shift));\n'
			            f'\tatomicOr({words}[i / {per}u], (v & 0x{mask:x}u) << shift);\n}}\n')
		else:
			helpers += (f'uint pw_load{bits}_{key}(uint i)\n{{\n'
			            f'\treturn ({words}[i / {per}u] >> ((i % {per}u) * {bits}u)) & 0x{mask:x}u;\n}}\n')
	return s, helpers


# Only these write or read narrow SSBO arrays in the variants built here. The rewrite
# works on the text, preprocessor branches included, so it is not applied blindly
# (wavelet_dequant has 8-bit buffers too, in the STORAGE_MODE=0 branch we do not use).
NARROW_SSBO_SHADERS = {'wavelet_quant.comp', 'block_packing.comp'}


def shim(source, filename=''):
	s = source
	helpers = ''

	if 'gl_SubgroupID' in s or 'gl_NumSubgroups' in s:
		s = s.replace('gl_SubgroupID', 'pw_subgroup_id()').replace('gl_NumSubgroups', 'pw_num_subgroups()')
		helpers += ('uint pw_subgroup_id() { return gl_LocalInvocationIndex / gl_SubgroupSize; }\n'
		            'uint pw_num_subgroups() { return (gl_WorkGroupSize.x * gl_WorkGroupSize.y * '
		            'gl_WorkGroupSize.z + gl_SubgroupSize - 1u) / gl_SubgroupSize; }\n')

	used = set()
	for op in ('Add', 'Max'):
		def repl(args, op=op):
			n = int(args[1].strip())
			used.add((op.lower(), n))
			return f'pw_clustered_{op.lower()}_{n}({args[0].strip()})'
		s = rewrite_calls(s, f'subgroupClustered{op}', repl)
	for op, n in sorted(used):
		helpers += clustered_helpers(op, n)

	# wavelet_dequant: the cross-subgroup scan is a shuffle-up Hillis-Steele loop.
	# Over the contiguous active lanes it is exactly an inclusive prefix sum.
	if 'uint scan_subgroups(uint v)' in s:
		s = s.replace('uint scan_subgroups(uint v)', 'uint scan_subgroups_shuffle(uint v)')
		helpers += 'uint scan_subgroups(uint v) { return subgroupInclusiveAdd(v); }\n'

	# image2D outputs -> one-layer image2DArray.
	decl = re.search(r'uniform\s+image2D\s+(\w+)\s*;', s)
	if decl:
		image = decl.group(1)
		s = s[:decl.start()] + f'uniform image2DArray {image};' + s[decl.end():]
		s = rewrite_calls(s, 'imageStore',
		                  lambda a: f'imageStore({a[0].strip()}, ivec3({a[1].strip()}, 0), {a[2].strip()})'
		                  if a[0].strip() == image else f'imageStore({",".join(a)})')

	# sampler2D inputs -> one-layer sampler2DArray, for the same reason as image2D
	# outputs: one SRV type then covers both a pyramid LL slice and an input plane.
	sdecl = re.search(r'uniform\s+(?:mediump\s+|highp\s+)?sampler2D\s+(\w+)\s*;', s)
	if sdecl:
		sampler = sdecl.group(1)
		s = s[:sdecl.start()] + f'uniform sampler2DArray {sampler};' + s[sdecl.end():]
		s = rewrite_calls(s, 'textureGather',
		                  lambda a: f'textureGather({a[0].strip()}, vec3({a[1].strip()}, 0.0))'
		                  if a[0].strip() == sampler else f'textureGather({",".join(a)})')

	s, narrow_helpers = rewrite_narrow_ssbos(s) if filename in NARROW_SSBO_SHADERS else (s, '')
	if narrow_helpers:
		# After every buffer block they touch, i.e. right before the push constants.
		pc = s.index('layout(push_constant)')
		s = s[:pc] + '// --- narrow SSBO access (generated by d3d12/shaders/transpile.py) ---\n' + \
		    narrow_helpers + '// --- end narrow SSBO access ---\n\n' + s[pc:]

	if helpers:
		# After the last local_size declaration so gl_WorkGroupSize is defined.
		m = list(re.finditer(r'layout\s*\(\s*local_size[^;]*;[^\n]*\n', s))
		pos = m[-1].end()
		s = s[:pos] + '\n// --- D3D12 translation shims (generated by d3d12/shaders/transpile.py) ---\n' + \
		    helpers + '// --- end shims ---\n\n' + s[pos:]
	return s


def main():
	for tool in (GLSLC, SPIRV_CROSS, DXC):
		if not Path(tool).exists():
			sys.exit(f'Missing tool: {tool}')

	HLSL_DIR.mkdir(exist_ok=True)
	GENERATED.parent.mkdir(exist_ok=True)
	tmp = Path(tempfile.mkdtemp(prefix='pyrowave-d3d12-', dir=HERE))
	try:
		# Shim every source (includes too, harmlessly), preserving the include layout.
		for f in list(GLSL_DIR.glob('*.comp')) + list(GLSL_DIR.glob('*.h')):
			(tmp / f.name).write_text(shim(f.read_text(), f.name) if f.suffix == '.comp' else f.read_text())

		blobs = []
		for name, src, defines, dxc_defines, profile in VARIANTS:
			desktop = profile == 'desktop'
			spv, hlsl, dxil = tmp / f'{name}.spv', HLSL_DIR / f'{name}.hlsl', tmp / f'{name}.dxil'
			run([GLSLC, '-fshader-stage=comp', '--target-env=vulkan1.1'] + [f'-D{d}' for d in defines] +
			    ['-I', win_path(tmp), win_path(tmp / src), '-o', win_path(spv)])
			run([SPIRV_CROSS, win_path(spv), '--hlsl', '--shader-model', '66' if desktop else '64'] +
			    (['--hlsl-enable-16bit-types'] if desktop else []) +
			    # Encoder: every storage buffer as a UAV, so buffers never change state and
			    # stage boundaries are plain UAV barriers.
			    (['--hlsl-force-storage-buffer-as-uav'] if desktop else []) + ['--output', win_path(hlsl)])

			text = hlsl.read_text()
			if desktop:
				# Pin the wave size: the kernels assume it in places, and AMD otherwise
				# picks wave32 or wave64 per shader.
				text = text.replace('[numthreads(', '[WaveSize(64)]\n[numthreads(')
			elif 'WaveReadLaneAt' in text:
				sys.exit(f'{name}: WaveReadLaneAt survived translation; it is broken on Xbox Series '
				         f'with lane-varying indices. Add a shim for it.')
			header = (f'// Generated by d3d12/shaders/transpile.py from shaders/{src} -- do not edit.\n'
			          f'// Profile {profile}; glslc defines: {" ".join(defines) or "(none)"}; '
			          f'dxc defines: {" ".join(dxc_defines) or "(none)"}\n')
			hlsl.write_text(header + text)

			run([DXC, '-nologo', '-T', 'cs_6_6' if desktop else 'cs_6_4', '-E', 'main', '-O3', '-Qstrip_debug',
			     '-Qstrip_reflect'] + (['-enable-16bit-types'] if desktop else []) +
			    [f'-D{d}' for d in dxc_defines] + [win_path(hlsl), '-Fo', win_path(dxil)])
			blobs.append((name, dxil.read_bytes()))
			print(f'  {name} ({profile}): {len(blobs[-1][1])} bytes DXIL')

		with GENERATED.open('w', newline='\n') as f:
			f.write('// Generated by d3d12/shaders/transpile.py -- do not edit.\n')
			f.write('// DXIL for the PyroWave D3D12 backend: decoder cs_6_4 (portable), encoder cs_6_6 (desktop).\n')
			f.write('#pragma once\n\n#include <stddef.h>\n#include <stdint.h>\n\nnamespace PyroWave\n{\nnamespace DXIL\n{\n')
			for name, data in blobs:
				f.write(f'static const uint8_t {name}[] = {{\n')
				for i in range(0, len(data), 16):
					f.write('\t' + ', '.join(f'0x{b:02x}' for b in data[i:i + 16]) + ',\n')
				f.write('};\n\n')
			f.write('}\n}\n')
		print(f'Wrote {GENERATED.relative_to(ROOT)} and {HLSL_DIR.relative_to(ROOT)}/')
	finally:
		shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
	main()
