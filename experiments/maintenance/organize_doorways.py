"""Flatten repeated uploaded experiment folders; preserve all distinct content."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'fourrooms_doorways'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=ROOT.resolve();files=list(root.rglob('*'));plan=[]
    for source in sorted((p for p in files if p.is_file()),key=lambda p:(len(p.parts),str(p))):
        relative=source.relative_to(root);parts=list(relative.parts)
        while parts and parts[0]=='fourrooms_doorways':parts.pop(0)
        if len(parts)>=2 and parts[0]=='runs' and 'smoke' in parts[1]:
            parts[0]='smoke_runs'
        if parts and parts[0] in ('advantage_stability','decline_analysis','stage_analysis'):
            parts.insert(0,'analyses')
        if parts and parts[0].startswith('goal_validity') and parts[0]!='goal_validity':
            parts=['analyses','goal_validity']+parts
        target=root.joinpath(*parts)
        if source==target:continue
        source.resolve().relative_to(root);target.resolve().relative_to(root)
        plan.append((source,target))
    if not args.apply:
        print(json.dumps(dict(files_to_organize=len(plan),examples=[dict(source=str(a),target=str(b)) for a,b in plan[:5]]),indent=2));return
    # Preserve a manifest of every move and every verified redundant copy.
    records=[];reclaimed=0
    for source,target in plan:
        source.resolve().relative_to(root);target.resolve().relative_to(root)
        sha=digest(source);size=source.stat().st_size
        if target.exists():
            if digest(target)==sha:
                source.unlink();action='removed_identical_duplicate';reclaimed+=size
            else:
                original=str(source.relative_to(root)).encode()
                target=root/'archive'/'import_conflicts'/hashlib.sha256(original).hexdigest()[:12]/target.relative_to(root)
                target.resolve().relative_to(root)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():raise FileExistsError(target)
                shutil.move(str(source),str(target));action='archived_differing_copy'
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source),str(target));action='moved'
        records.append(dict(source=str(source.relative_to(root)),target=str(target.relative_to(root)),action=action,sha256=sha,bytes=size))
    # Delete only empty old directories, never recursively delete directory trees.
    for directory in sorted((p for p in root.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        directory.resolve().relative_to(root)
        if not any(directory.iterdir()):directory.rmdir()
    manifest=root/'organization_manifest.json'
    manifest.write_text(json.dumps(dict(reclaimed_bytes=reclaimed,operations=records),indent=2))
    print(json.dumps(dict(operations=len(records),reclaimed_mb=reclaimed/1024**2,
        differing_copies_archived=sum(r['action']=='archived_differing_copy' for r in records),manifest=str(manifest)),indent=2))

if __name__=='__main__':main()
