#!/usr/bin/env python3
"""fq fix: mount an fsid=0 NFSv4 export correctly.

files/nfs-share.sh writes "/export <client>(...,fsid=0)" into /etc/exports. With
fsid=0 that export *is* the NFSv4 pseudo-root, so clients must mount "<ip>:/" --
mounting "<ip>:/export" makes the server look for /export/export and fail with
"no such file or directory" (that is exactly how the worker volume broke after the
exporter container was recreated: the client spec still said :/dsv41-native).

Server side keeps NFS_EXPORT_NAME (the directory under HF_EXPORT_ROOT that the
exporter maps to /export)."""
import re

p = 'files/nfs-share.sh'
s = open(p, encoding='utf-8').read()
n0 = s.count('NFS_DEVICE=":/${NFS_EXPORT_NAME}"')
s = s.replace('NFS_DEVICE=":/${NFS_EXPORT_NAME}"',
              'NFS_DEVICE=":/"   # fsid=0 export -> mount the v4 pseudo-root')
s = s.replace('device="${NFS_DEVICE:-:/${NFS_EXPORT_NAME}}"', 'device="${NFS_DEVICE:-:/}"')
s = s.replace('NFS_EXPORT_NAME="${NFS_EXPORT_NAME:-dsv41-native}"',
              'NFS_EXPORT_NAME="${NFS_EXPORT_NAME:-export}"')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("NFS_DEVICE sites rewritten:", n0)
for i, line in enumerate(s.splitlines(), 1):
    if 'NFS_DEVICE' in line or line.startswith('NFS_EXPORT_NAME'):
        print(f"  {i}: {line.strip()}")
