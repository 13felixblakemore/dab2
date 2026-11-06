"""Isolated LR x Polyak sweep, reusing the existing double-doorway JAX runner.

Default: classic Cauchy continuing FB, seed 0, d {50,100},
LR {5e-4,1e-4,5e-5}, Polyak {0.90,0.95,0.99}; 18 sequential runs.
"""
from fb_navigation.paths import PROJECT_ROOT, source_path
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = PROJECT_ROOT
DEFAULT_OUTPUT=ROOT/'experiments/sweeps/fb_width2_lr_polyak_20261106'
TRAINING_SOURCES=['run_fourrooms_doorways.py','adjustable_four_rooms.py','run_fourrooms_reference.py',
    'fourrooms_reference_diagnostics.py','run_fb_jax.py','continuous_four_rooms_jax.py',
    'forward_backward_jax.py','control_bfs_bottlenecks.py','estimate_fb_bottlenecks_jax.py','paths.py']


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(base,lrs,polyaks,seeds,epochs=800,doorway_scale=2.,checkpoint_every=200,smoke=False,dimensions=None):
    dimensions=[50] if dimensions is None else dimensions
    if base['learner']['z_sampling']!='cauchy' or base['learner']['goal_terminated']:
        raise ValueError('This sweep requires classic Cauchy FB with goal termination disabled')
    if not dimensions or any(not isinstance(d,int) or d<1 for d in dimensions):
        raise ValueError('Dimensions must be positive integers')
    if any(not math.isfinite(v) or v<=0 for v in lrs):raise ValueError('Learning rates must be finite and positive')
    if any(not math.isfinite(v) or not 0<=v<1 for v in polyaks):raise ValueError('Polyak must be in [0,1)')
    if not 0<doorway_scale<3 or epochs<1 or checkpoint_every<1:raise ValueError('Invalid geometry or schedule')
    if not lrs or not polyaks or not seeds:raise ValueError('Empty sweep axis')
    if any(len(set(v))!=len(v) for v in (lrs,polyaks,seeds,dimensions)):raise ValueError('Duplicate sweep values')
    cells=[]
    for dim,lr,polyak,seed in itertools.product(dimensions,lrs,polyaks,seeds):
        cfg=copy.deepcopy(base)
        cfg['epochs']=epochs
        cfg['environment']['doorway_scale']=doorway_scale
        cfg['checkpoint_every']=checkpoint_every
        cfg['evaluation_epochs']=sorted({1,epochs,*range(checkpoint_every,epochs+1,checkpoint_every)})
        cfg['learner']['learning_rate']=lr;cfg['learner']['polyak_coef']=polyak
        cfg['learner']['embed_dim']=dim
        if smoke:
            cfg.update(epochs=2,cycles=1,evaluation_per_category=4,evaluation_epochs=[1,2])
            cfg['learner']['buffer_size']=512
        name=f'd{dim}_lr{lr:.12g}_polyak{polyak:.12g}_seed{seed}'
        cells.append(dict(index=len(cells),name=name,seed=seed,dimension=dim,learning_rate=lr,polyak=polyak,config=cfg))
    if len({c['name'] for c in cells})!=len(cells):raise ValueError('Sweep values produce colliding run names')
    return cells


def command(cell,output,device,recovery_every,log_every,resume=False):
    result=[sys.executable,'-u',str(ROOT/'run_fourrooms_doorways.py'),
        '--config',str(output/'configs'/f'{cell["name"]}.json'),
        '--doorway-scale',str(cell['config']['environment']['doorway_scale']),
        '--checkpoint-every',str(cell['config']['checkpoint_every']),
        '--recovery-every',str(recovery_every),'--log-every',str(log_every),
        '--seed',str(cell['seed']),'--device',device,'--output',str(output/'runs'/cell['name'])]
    if resume:result.append('--resume')
    return result


def aggregate(output,manifest):
    rows=[];status=[]
    for cell in manifest['cells']:
        folder=output/'runs'/cell['name'];complete=folder/'complete.json'
        training=folder/'training.json';completed=complete.exists()
        epoch=0
        if training.exists():
            history=json.loads(training.read_text());epoch=history[-1]['epoch'] if history else 0
        status.append(dict(run=cell['name'],completed=completed,epoch=epoch))
        for path in sorted((folder/'evaluation').glob('epoch_*.json')):
            for metric in json.loads(path.read_text()):
                rows.append(dict(run=cell['name'],dimension=cell['dimension'],learning_rate=cell['learning_rate'],polyak=cell['polyak'],
                    seed=cell['seed'],**metric))
    (output/'status.json').write_text(json.dumps(status,indent=2))
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (output/'goal_reaching.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        final=[r for r in rows if r['epoch']==next(c['config']['epochs'] for c in manifest['cells'] if c['name']==r['run'])]
        with (output/'final_goal_reaching.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(final)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for filename,category in [('diagonal_learning_curves.png','diagonal_rooms'),('adjacent_learning_curves.png','adjacent_rooms')]:
            fig,axes=plt.subplots(1,2,figsize=(13,4))
            for ax,horizon in zip(axes,(30,150)):
                for dim,lr,polyak in sorted(set((c['dimension'],c['learning_rate'],c['polyak']) for c in manifest['cells'])):
                    selected=[r for r in rows if r['dimension']==dim and r['learning_rate']==lr and r['polyak']==polyak and r['category']==category and r['horizon']==horizon]
                    epochs=sorted(set(r['epoch'] for r in selected))
                    values=[sum(r['success_rate'] for r in selected if r['epoch']==e)/sum(r['epoch']==e for r in selected) for e in epochs]
                    if epochs:ax.plot(epochs,values,marker='.',label=f'd={dim}, lr={lr:g}, p={polyak:g}')
                ax.set(xlabel='Epoch',ylabel='Success rate',ylim=(0,1),title=f'{category}, H={horizon}');ax.grid(alpha=.2)
            axes[-1].legend(fontsize=5,loc='center left',bbox_to_anchor=(1, .5));fig.tight_layout();fig.savefig(output/filename,dpi=160);plt.close(fig)
    print(json.dumps(dict(completed=sum(s['completed'] for s in status),total=len(status),summary=str(output/'goal_reaching.csv'))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config',type=Path,default=ROOT/'experiments/fourrooms_reference/config.json')
    parser.add_argument('--learning-rates',nargs='+',type=float,default=[5e-4,1e-4,5e-5])
    parser.add_argument('--dimensions',nargs='+',type=int,default=[50,100])
    parser.add_argument('--polyaks',nargs='+',type=float,default=[.90,.95,.99])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0])
    parser.add_argument('--epochs',type=int,default=800)
    parser.add_argument('--doorway-scale',type=float,default=2.)
    parser.add_argument('--checkpoint-every',type=int,default=200)
    parser.add_argument('--recovery-every',type=int,default=200)
    parser.add_argument('--log-every',type=int,default=20)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--run-index',type=int,help='Run one preprepared cell, e.g. in a GPU job array')
    parser.add_argument('--smoke',action='store_true',help='Explicit 2-epoch/80-update validation per cell')
    parser.add_argument('--restart-without-recovery',action='store_true',
                        help='With --resume, archive a partial cell with no recovery and restart just that cell')
    args=parser.parse_args();output=args.output.resolve()
    if args.smoke and output==DEFAULT_OUTPUT:output=Path(str(output)+'_smoke')
    manifest_path=output/'sweep.json'
    if args.summarize_only:
        aggregate(output,json.loads(manifest_path.read_text()));return
    if args.recovery_every<1 or args.log_every<0:parser.error('Invalid recovery/log interval')
    if args.resume or args.run_index is not None:
        if not manifest_path.exists():parser.error('Prepare this sweep first with --prepare-only')
        manifest=json.loads(manifest_path.read_text())
        if manifest['source_sha256']!={name:sha(source_path(name)) for name in TRAINING_SOURCES}:
            raise ValueError('Training source changed; restore the archived sources before continuing this sweep')
    else:
        if manifest_path.exists():
            raise FileExistsError('Sweep already exists; use --resume or --summarize-only')
        base=json.loads(args.base_config.read_text())
        cells=make_plan(base,args.learning_rates,args.polyaks,args.seeds,args.epochs,args.doorway_scale,args.checkpoint_every,args.smoke,args.dimensions)
        output.mkdir(parents=True,exist_ok=True);(output/'configs').mkdir();(output/'source').mkdir()
        for cell in cells:(output/'configs'/f'{cell["name"]}.json').write_text(json.dumps(cell['config'],indent=2))
        for name in TRAINING_SOURCES:(output/'source'/name).write_bytes((source_path(name)).read_bytes())
        (output/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        manifest=dict(created=now(),base_config=str(args.base_config.resolve()),base_sha256=sha(args.base_config),
            source_sha256={name:sha(source_path(name)) for name in TRAINING_SOURCES},
            source_paths={name:source_path(name).relative_to(ROOT).as_posix() for name in TRAINING_SOURCES},
            recovery_every=args.recovery_every,log_every=args.log_every,cells=cells)
        manifest_path.write_text(json.dumps(manifest,indent=2))
    print(f'Sweep: {len(manifest["cells"])} runs; output {output}',flush=True)
    if args.prepare_only:
        for cell in manifest['cells']:print(f'{cell["index"]}: {cell["name"]}')
        return
    cells=manifest['cells']
    if args.run_index is not None:
        if args.run_index<0 or args.run_index>=len(cells):parser.error('Run index outside sweep')
        cells=[cells[args.run_index]]
    for cell in cells:
        folder=output/'runs'/cell['name'];existing=folder/'config.json'
        if existing.exists() and json.loads(existing.read_text())!=cell['config']:
            raise ValueError(f'Run configuration mismatch: {folder}')
        if (folder/'complete.json').exists():
            complete=json.loads((folder/'complete.json').read_text())
            expected=cell['config']['epochs']*cell['config']['cycles']*cell['config']['updates']
            if complete['updates']!=expected:raise ValueError('Incorrect completed update count')
            print('Already complete: '+cell['name'],flush=True);continue
        resume=existing.exists()
        if resume and not args.resume:raise FileExistsError(f'Interrupted run {folder}; pass --resume')
        if resume and not (folder/'resume.pkl').exists():
            if not args.restart_without_recovery:
                raise FileNotFoundError(f'No recovery snapshot in {folder}; add --restart-without-recovery to archive and restart just this cell')
            archive=output/'archive'/f'{cell["name"]}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
            folder.resolve().relative_to(output);archive.resolve().relative_to(output)
            archive.parent.mkdir(parents=True,exist_ok=True);folder.rename(archive);resume=False
        folder.mkdir(parents=True,exist_ok=True)
        cmd=command(cell,output,args.device,manifest['recovery_every'],manifest['log_every'],resume)
        print(f'Running {cell["index"]+1}/{len(manifest["cells"])}: {cell["name"]}',flush=True)
        with (folder/'stdout.log').open('a') as stdout,(folder/'stderr.log').open('a') as stderr:
            process=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
        (folder/'launcher_status.json').write_text(json.dumps(dict(returncode=process.returncode,finished=now()),indent=2))
        if process.returncode:
            raise RuntimeError(f'Run failed (code {process.returncode}); see {folder / "stderr.log"}. Fix and use --resume.')
        if args.run_index is None:aggregate(output,manifest)
    if args.run_index is not None:
        print('Cell finished. After all jobs, run this script with --output <sweep> --summarize-only.',flush=True)


if __name__=='__main__':main()
