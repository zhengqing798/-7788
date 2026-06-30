#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清洗 persons_data.json，去重并输出到新文件"""

import json
import sys

# Windows 终端 UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

INPUT_FILE = 'persons_data.json'
OUTPUT_FILE = 'persons_data_clean.json'


def clean():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f'原始总数: {len(data)}')

    # ==================== 1. 去重 ====================
    # 按 douban_id 去重（保留第一条）
    seen_did = {}
    dupes_did = []
    for name in list(data.keys()):
        info = data[name]
        did = info.get('douban_id')
        if not did:
            continue
        if did in seen_did:
            dupes_did.append((name, seen_did[did]))
            del data[name]
        else:
            seen_did[did] = name
    print(f'douban_id 重复: {len(dupes_did)} 组')
    for dup_name, kept_name in dupes_did:
        print(f'  删除 "{dup_name}" (保留 "{kept_name}")')

    # 按 personage_id 去重
    seen_pid = {}
    dupes_pid = []
    for name in list(data.keys()):
        info = data[name]
        pid = info.get('personage_id')
        if not pid:
            continue
        if pid in seen_pid:
            dupes_pid.append((name, seen_pid[pid]))
            del data[name]
        else:
            seen_pid[pid] = name
    print(f'personage_id 重复: {len(dupes_pid)} 组')
    for dup_name, kept_name in dupes_pid:
        print(f'  删除 "{dup_name}" (保留 "{kept_name}")')

    # ==================== 2. 清理无效字段 ====================
    cleaned_fields = 0
    removed_entries = []
    for name in list(data.keys()):
        info = data[name]

        # 删除内部标记字段
        for key in ['_no_avatar', '_scrape_failed', '_scraped_at']:
            if key in info:
                del info[key]
                cleaned_fields += 1

        # 删除没有 douban_id 的条目（说明没爬到有效数据）
        if not info.get('douban_id'):
            removed_entries.append(name)
            del data[name]

    if removed_entries:
        print(f'无效条目(无douban_id): {len(removed_entries)} 条')
        for n in removed_entries:
            print(f'  删除 "{n}"')

    print(f'清理内部字段: {cleaned_fields} 个')

    # ==================== 3. 统计 ====================
    total = len(data)
    with_avatar = sum(1 for v in data.values() if v.get('avatar_local_path'))
    with_avatar_url = sum(1 for v in data.values() if v.get('avatar_url'))
    with_name_en = sum(1 for v in data.values() if v.get('name_en'))
    with_gender = sum(1 for v in data.values() if v.get('gender'))
    with_birth = sum(1 for v in data.values() if v.get('birth_date'))
    with_bio = sum(1 for v in data.values() if v.get('biography'))

    print(f'\n清洗后: {total} 条')
    print(f'  有本地头像:  {with_avatar} ({with_avatar/total*100:.1f}%)')
    print(f'  有头像URL:   {with_avatar_url} ({with_avatar_url/total*100:.1f}%)')
    print(f'  有英文名:    {with_name_en} ({with_name_en/total*100:.1f}%)')
    print(f'  有性别:      {with_gender} ({with_gender/total*100:.1f}%)')
    print(f'  有出生日期:  {with_birth} ({with_birth/total*100:.1f}%)')
    print(f'  有简介:      {with_bio} ({with_bio/total*100:.1f}%)')

    # ==================== 4. 写入 ====================
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'\n已写入: {OUTPUT_FILE}')


if __name__ == '__main__':
    clean()
