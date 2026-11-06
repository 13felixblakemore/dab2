"""Undiscounted first-hit control: cross-room pairs and a BFS-optimal policy.

No FB checkpoint/training. Deterministic four-neighbor discretised dynamics;
randomness comes from endpoints and uniform ties between shortest actions.
"""
import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from continuous_four_rooms_jax import ContinuousFourRooms
from estimate_fb_bottlenecks_jax import grid_regions, eligibility

ROOMS=['bottom_left','top_left','bottom_right','top_right']


def room_ids(points,size):
    x,y=(points/size*11).T
    result=np.full(len(points),-1,int)
    for i,mask in enumerate(((x<6)&(y<4),(x<6)&(y>5),(x>7)&(y<4),(x>7)&(y>5))):result[mask]=i
    return result


def build_graph(env,resolution):
    xy=(np.arange(resolution)+.5)/resolution*env.size
    x,y=np.meshgrid(xy,xy);points=np.column_stack((x.ravel(),y.ravel()))
    free=np.ones(len(points),bool)
    for w in env.wall_blocks:
        free &= ~((points[:,0]>=w.xmin)&(points[:,0]<=w.xmax)&(points[:,1]>=w.ymin)&(points[:,1]<=w.ymax))
    flat=np.flatnonzero(free);nodes=points[free]
    inverse=np.full(resolution**2,-1,int);inverse[flat]=np.arange(len(flat))
    row,col=flat//resolution,flat%resolution
    neighbours=np.full((len(nodes),4),-1,int)
    for action,(dr,dc) in enumerate(((0,-1),(0,1),(1,0),(-1,0))):
        nr,nc=row+dr,col+dc
        bounds=(nr>=0)&(nr<resolution)&(nc>=0)&(nc<resolution)
        ids=np.flatnonzero(bounds)
        dest=inverse[nr[ids]*resolution+nc[ids]]
        possible=dest>=0;ids=ids[possible];dest=dest[possible]
        a,b=nodes[ids],nodes[dest]
        blocked=np.zeros(len(ids),bool)
        # Closed rectangular segment intersection, vectorised slab method.
        for w in env.wall_blocks:
            tmin=np.zeros(len(ids));tmax=np.ones(len(ids));valid=np.ones(len(ids),bool)
            for axis,lower,upper in ((0,w.xmin,w.xmax),(1,w.ymin,w.ymax)):
                delta=b[:,axis]-a[:,axis];parallel=abs(delta)<1e-12
                safe=np.where(parallel,1.,delta)
                t1=(lower-a[:,axis])/safe;t2=(upper-a[:,axis])/safe
                tmin=np.where(parallel,tmin,np.maximum(tmin,np.minimum(t1,t2)))
                tmax=np.where(parallel,tmax,np.minimum(tmax,np.maximum(t1,t2)))
                valid &= ~parallel|((a[:,axis]>=lower)&(a[:,axis]<=upper))
            blocked |= valid & (tmax>1e-6)&((tmax-tmin)>1e-6)
        neighbours[ids[~blocked],action]=dest[~blocked]
    return nodes,neighbours


def bfs_distances(neighbours,terminal):
    """Multi-source BFS for the Euclidean goal disk on an undirected graph."""
    distance=np.full(len(neighbours),-1,int)
    queue=deque(np.flatnonzero(terminal).tolist());distance[terminal]=0
    while queue:
        node=queue.popleft()
        for nxt in neighbours[node]:
            if nxt>=0 and distance[nxt]<0:
                distance[nxt]=distance[node]+1;queue.append(int(nxt))
    return distance


def sample_cross_room_pairs(nodes,room,count,radius,seed):
    rng=np.random.default_rng(seed);valid_nodes=np.flatnonzero(room>=0)
    starts=[];goals=[];left=count
    while left:
        u=rng.choice(valid_nodes,max(64,left*2));g=rng.choice(valid_nodes,len(u))
        keep=(room[u]!=room[g])&(np.sum((nodes[u]-nodes[g])**2,axis=1)>radius**2)
        u=u[keep][:left];g=g[keep][:left]
        starts.extend(u.tolist());goals.extend(g.tolist());left-=len(u)
    return np.array(starts),np.array(goals)


def simulate(neighbours,distances,starts,goals,node_bins,rollouts,horizon,seed):
    """Each active transition reduces exact BFS distance by exactly one."""
    rng=np.random.default_rng(seed)
    goal_rows=np.repeat(goals,rollouts)
    current=np.repeat(starts,rollouts)
    n=len(current);first=np.full((n,int(node_bins.max())+1),-1,np.int32)
    length=np.zeros(n,int)
    for t in range(horizon):
        d=distances[goal_rows,current];active=d>0
        if not active.any():break
        idx=np.flatnonzero(active);cells=node_bins[current[idx]]
        unseen=first[idx,cells]<0;first[idx[unseen],cells[unseen]]=t
        nxt=neighbours[current[idx]]
        possible=(nxt>=0)&(distances[goal_rows[idx,None],np.maximum(nxt,0)]==d[idx,None]-1)
        counts=possible.sum(axis=1)
        if np.any(counts==0):raise AssertionError('BFS descent unavailable')
        choice=np.floor(rng.random(len(idx))*counts).astype(int)
        action=np.argmax(np.cumsum(possible,axis=1)>choice[:,None],axis=1)
        current[idx]=nxt[np.arange(len(idx)),action];length[idx]+=1
    finished=distances[goal_rows,current]==0
    return first,finished,length


def run(args):
    if min(args.samples,args.rollouts,args.graph_grid,args.grid)<1:raise ValueError('Counts must be positive')
    if args.goal_radius<0 or not np.isfinite(args.goal_radius):raise ValueError('Invalid goal radius')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    env=ContinuousFourRooms(include_stay_action=False)
    nodes,neighbours=build_graph(env,args.graph_grid)
    if args.goal_radius>=np.linalg.norm(np.ptp(nodes,axis=0)):
        raise ValueError('Goal radius leaves no valid start/goal pairs')
    rooms=room_ids(nodes,env.size)
    if len(np.unique(rooms[rooms>=0]))!=4:raise ValueError('Grid must resolve all four rooms')
    starts,goals=sample_cross_room_pairs(nodes,rooms,args.samples,args.goal_radius,args.seed)
    unique,inverse=np.unique(goals,return_inverse=True)
    distances=[]
    for i,g in enumerate(unique):
        terminal=np.sum((nodes-nodes[g])**2,axis=1)<=args.goal_radius**2
        d=bfs_distances(neighbours,terminal)
        if np.any(d<0):raise ValueError('Graph is disconnected; increase graph resolution')
        distances.append(d)
    distances=np.stack(distances)
    shortest=distances[inverse,starts]
    horizon=int(shortest.max()) if args.horizon is None else args.horizon
    if horizon<int(shortest.max()):raise ValueError('Horizon would censor shortest paths; increase it')
    points,lower,upper,keep=grid_regions(env,args.grid);candidate_ids=np.flatnonzero(keep)
    if not len(candidate_ids):raise ValueError('No whole wall-free candidate cells')
    xy=np.clip(np.floor(nodes/env.size*args.grid).astype(int),0,args.grid-1)
    node_bins=xy[:,1]*args.grid+xy[:,0]
    first,finished,length=simulate(neighbours,distances,starts,inverse,node_bins,args.rollouts,horizon,args.seed+1)
    np.testing.assert_array_equal(length,np.repeat(shortest,args.rollouts))
    if not np.all(finished):raise AssertionError('Censored optimal trajectories')
    pair_hits=(first[:,candidate_ids]>=0).reshape(args.samples,args.rollouts,-1).mean(axis=1)
    eligible=eligibility(nodes[goals],lower[keep],upper[keep],args.goal_radius)
    count=eligible.sum(axis=0)
    beta=np.divide(np.where(eligible,pair_hits,0).sum(axis=0),count,out=np.full(len(count),np.nan),where=count>0)
    se=np.sqrt(np.divide(np.where(eligible,(pair_hits-beta)**2,0).sum(axis=0),count,
                       out=np.zeros(len(count)),where=count>0))/np.sqrt(np.maximum(count-1,1))
    p=points[keep]/env.size*11;x,y=p.T
    doors=dict(bottom=(x>=6)&(x<=7)&(y>=1)&(y<=2),top=(x>=6)&(x<=7)&(y>=8)&(y<=9),
               left=(x>=2)&(x<=3)&(y>=4)&(y<=5),right=(x>=9)&(x<=10)&(y>=4)&(y<=5))
    # Door statistics refer to fully wall-free cells whose centers lie in doors.
    door_stats={}
    door_union=np.zeros(len(beta),bool)
    for name,selected in doors.items():
        door_union|=selected
        values=beta[selected & np.isfinite(beta)]
        door_stats[name]=dict(cells=int(selected.sum()),mean=float(values.mean()) if len(values) else None,
            mean_percentile=float(np.mean(beta[np.isfinite(beta)]<=values.mean())) if len(values) else None,
            maximum=float(values.max()) if len(values) else None)
    interior=~door_union & np.isfinite(beta)
    top=np.argsort(np.where(np.isfinite(beta),beta,-np.inf))[-20:][::-1]
    room_pair_counts={ROOMS[a]+' -> '+ROOMS[b]:int(np.sum((rooms[starts]==a)&(rooms[goals]==b))) for a in range(4) for b in range(4) if a!=b}
    metrics=dict(gamma=1.,samples=args.samples,rollouts_per_pair=args.rollouts,seed=args.seed,
        graph_grid=args.graph_grid,graph_nodes=len(nodes),candidate_grid=args.grid,candidate_cells=int(keep.sum()),
        goal_radius=args.goal_radius,horizon=horizon,all_trajectories_finished=bool(finished.all()),
        maximum_shortest_length=int(shortest.max()),median_shortest_length=float(np.median(shortest)),
        cross_room_pairs=bool(np.all(rooms[starts]!=rooms[goals])),room_pair_counts=room_pair_counts,
        policy='uniform over available neighbors whose BFS goal-disk distance decreases by one',
        dynamics='deterministic four-neighbor graph; no movement noise; edges cannot intersect walls',
        convention='count current state before goal entry; exclude candidate cells overlapping goal disk; include initial visits',
        doorway_scores=door_stats,mean_doorway_cell_score=float(np.nanmean(beta[door_union])),
        mean_other_cell_score=float(np.mean(beta[interior])),
        doorway_enrichment=float(np.nanmean(beta[door_union])/np.mean(beta[interior])),
        top20_doorway_cells=int(door_union[top].sum()),
        top20_cells=[dict(x=float(points[keep][j,0]),y=float(points[keep][j,1]),score=float(beta[j])) for j in top],
        caveats=['Optimality is exact on this graph, not on the original noisy continuous dynamics.',
                 'Averaging uniform shortest-action ties is not uniform sampling of all shortest paths.',
                 'Endpoint pairs are uniform over room grid nodes conditioned on different rooms and starts outside the goal.',
                 'Candidate resolution and geometric discretisation affect scores; whole doorway scores differ from individual-cell scores.',
                 'Pair-level SE reflects endpoint variability; nested rollouts sharing endpoints are not independent endpoint samples.'])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    np.savez_compressed(out/'intermediate.npz',nodes=nodes,neighbours=neighbours,starts=starts,goals=goals,
        pair_hits=pair_hits,eligible=eligible,beta=beta,standard_error=se,candidate_points=points[keep],candidate_mask=keep,
        first_hits=first[:,candidate_ids],lengths=length,shortest_lengths=shortest)
    with (out/'cells.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['x','y','first_hit_probability','pair_standard_error','eligible_pairs','doorway'])
        w.writerows(zip(points[keep][:,0],points[keep][:,1],beta,se,count,door_union))
    fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
    data=np.full(args.grid**2,np.nan);data[keep]=beta
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#bdbdbd')
    im=ax.imshow(data.reshape(args.grid,args.grid),origin='lower',extent=(0,env.size,0,env.size),vmin=0,cmap=cmap)
    for wall in env.wall_blocks:ax.add_patch(Rectangle((wall.xmin,wall.ymin),wall.xmax-wall.xmin,wall.ymax-wall.ymin,color='black'))
    ax.set_title('BFS-optimal cross-room first-hit probability, gamma=1')
    fig.colorbar(im,ax=ax,label='Probability of visiting cell before goal');fig.savefig(out/'bfs_bottlenecks.png',dpi=180);plt.close(fig)
    report=['# BFS cross-room undiscounted control','',f'{args.samples} cross-room pairs, {args.rollouts} shortest-action tie rollouts each, graph {args.graph_grid}x{args.graph_grid}, candidate grid {args.grid}x{args.grid}.',
            '',f'All trajectories finished: {finished.all()}. Maximum shortest-path length: {shortest.max()}. Gamma is exactly 1.',
            '', '| Door | Candidate cells | Mean first-hit probability | Mean-score percentile |','|---|---:|---:|---:|']
    for name,r in door_stats.items():report.append(f'| {name} | {r["cells"]} | {r["mean"]:.4f} | {r["mean_percentile"]:.1%} |')
    report+=['',f'Mean doorway-cell score / other-cell score: {metrics["doorway_enrichment"]:.2f}x. Doorway cells in top 20: {metrics["top20_doorway_cells"]}.',
             '', 'This is a deterministic discretised control with optimal BFS descent, not a model of optimal control under continuous movement noise. The goal disk and candidate bins are kept consistent with the preceding experiment. No FB parameters, training discount, or checkpoints were modified.',
             '', 'Reproduce: python control_bfs_bottlenecks.py. Configure --samples, --rollouts, --graph-grid, --grid, --goal-radius, --seed, --horizon and --output.']
    (out/'RESULTS.md').write_text('\n'.join(report));print(json.dumps(metrics,indent=2),flush=True)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=1000);p.add_argument('--rollouts',type=int,default=32)
    p.add_argument('--graph-grid',type=int,default=44);p.add_argument('--grid',type=int,default=20)
    p.add_argument('--goal-radius',type=float,default=.1);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--horizon',type=int,default=None);p.add_argument('--output',default='evaluation_bfs_control')
    run(p.parse_args())
