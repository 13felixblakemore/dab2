"""Geometry, native transition and free-space connectivity audit of saved goals."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from adjustable_four_rooms import AdjustableFourRooms
from continuous_four_rooms_jax import Point,_move
from control_bfs_bottlenecks import build_graph,bfs_distances

ROOT=Path('experiments/fourrooms_doorways')
RUN=ROOT/'runs/width2_seed0'


def point_status(points,env):
    points=np.asarray(points,dtype=np.float64)
    inside=np.zeros(len(points),bool);native_inside=np.zeros(len(points),bool)
    clearance=np.full(len(points),np.inf)
    p32=points.astype(np.float32)
    for wall in env.wall_blocks:
        lower=np.array([wall.xmin,wall.ymin]);upper=np.array([wall.xmax,wall.ymax])
        inside|=np.all((points>=lower)&(points<=upper),axis=1)
        native_inside|=np.all((p32>=lower.astype(np.float32))&(p32<=upper.astype(np.float32)),axis=1)
        nearest=np.clip(points,lower,upper)
        clearance=np.minimum(clearance,np.linalg.norm(points-nearest,axis=1))
    outside=np.any((points<0)|(points>env.size),axis=1)|~np.isfinite(points).all(axis=1)
    # The square success region may overlap walls even with a valid goal center.
    overlaps=np.zeros(len(points),bool)
    for wall in env.wall_blocks:
        overlaps|=((points[:,0]+env.threshold_distance>=wall.xmin)&(points[:,0]-env.threshold_distance<=wall.xmax)
                   &(points[:,1]+env.threshold_distance>=wall.ymin)&(points[:,1]-env.threshold_distance<=wall.ymax))
    return dict(count=len(points),inside_wall=int(inside.sum()),native_float32_inside_wall=int(native_inside.sum()),
        outside_world=int(outside.sum()),bad_indices=np.flatnonzero(inside|native_inside|outside).tolist(),
        minimum_wall_clearance=float(clearance.min()),success_square_overlaps_wall=int(overlaps.sum()))


def component_sizes(graph):
    unseen=set(range(len(graph)));sizes=[]
    while unseen:
        queue=[unseen.pop()];count=0
        while queue:
            node=queue.pop();count+=1
            for other in graph[node]:
                if int(other) in unseen:unseen.remove(int(other));queue.append(int(other))
        sizes.append(count)
    return sorted(sizes,reverse=True)


def native_graph_check(env,nodes,graph):
    source,action=np.where(graph>=0);dest=graph[source,action]
    vectors=jnp.array([(a.x,a.y) for a in env.actions]);walls=tuple(env.wall_blocks)
    @jax.jit
    def check(starts,selected):
        def one(p,a):
            nxt,_,blocked=_move(Point(*p),Point(*vectors[a]),jax.random.PRNGKey(0),0.,env.size,walls)
            return jnp.array([nxt.x,nxt.y]),blocked
        return jax.vmap(one)(starts,selected)
    moved,blocked=check(jnp.asarray(nodes[source]),jnp.asarray(action))
    error=np.max(abs(np.asarray(moved)-nodes[dest]),axis=1)
    return dict(directed_edges=len(source),native_wall_rejections=int(np.asarray(blocked).sum()),
        movement_mismatches_above_1e7=int((error>1e-7).sum()),maximum_coordinate_error=float(error.max()))


def reference_reachability(env,nodes,graph,states,goals,budgets):
    states=np.asarray(states,np.float32);goals=np.asarray(goals,np.float32)
    n=len(states);nearest=np.argsort(np.sum((states[:,None]-nodes[None])**2,axis=2),axis=1)[:,:8]
    walls=tuple(env.wall_blocks)
    @jax.jit
    def segment_blocked(starts,ends):
        return jax.vmap(lambda p,ns:jax.vmap(lambda q:jnp.any(jnp.array([
            w.intersects_segment(Point(*p),Point(*q)) for w in walls])))(ns))(starts,ends)
    blocked=np.asarray(segment_blocked(jnp.asarray(states),jnp.asarray(nodes[nearest])))
    valid=(~blocked).any(axis=1)
    attached=nearest[np.arange(n),np.argmax(~blocked,axis=1)]
    cache={};lengths=[];target_free=[]
    for start,goal in zip(attached,goals):
        key=goal.tobytes()
        if key not in cache:
            # Exact float32 coordinate-wise predicate used by the JAX environment.
            terminal=np.all(abs(nodes.astype(np.float32)-goal)<=np.float32(env.threshold_distance),axis=1)
            cache[key]=(bfs_distances(graph,terminal),int(terminal.sum()))
        distance,count=cache[key];lengths.append(distance[start]);target_free.append(count)
    lengths=np.asarray(lengths)
    return dict(no_visible_attachment=int((~valid).sum()),no_free_terminal_nodes=int((np.array(target_free)==0).sum()),
        disconnected_reference_pairs=int((lengths<0).sum()),minimum_reference_steps=int(lengths.min()),
        maximum_reference_steps=int(lengths.max()),reference_exceeds_30=int((lengths>30).sum()),
        reference_exceeds_150=int((lengths>150).sum()),reference_exceeds_assigned_budget=int((lengths>budgets).sum())),lengths


def main():
    config=json.loads((RUN/'config.json').read_text());meta=json.loads((RUN/'metadata.json').read_text())
    env=AdjustableFourRooms(**config['environment'])
    actual=np.array([[w.xmin,w.ymin,w.xmax,w.ymax] for w in env.wall_blocks])
    nodes,graph=build_graph(env,20)
    result=dict(walls_match_saved_metadata=bool(np.array_equal(actual,np.array(meta['wall_blocks']))),
        free_graph_nodes=len(nodes),connected_component_sizes=component_sizes(graph),native_graph=native_graph_check(env,nodes,graph),sets={},traces={})
    pairs=dict(np.load(RUN/'evaluation_pairs.npz'))
    fixed=dict(np.load(ROOT/'analyses/advantage_stability/fixed_probes.npz'))
    stage=json.loads((ROOT/'analyses/stage_analysis/probes.json').read_text())
    sets=dict(original_600=(pairs['starts'],pairs['goals'],np.full(600,30)),
        fixed_doorway_160=(fixed['states'],fixed['goals'],np.full(len(fixed['states']),30)),
        matched_stage_759=(np.array([r['state'] for r in stage]),np.array([r['goal'] for r in stage]),np.array([r['remaining_budget'] for r in stage])))
    lengths_by_set={}
    for label,(states,goals,budgets) in sets.items():
        reach,lengths=reference_reachability(env,nodes,graph,states,goals,budgets)
        result['sets'][label]=dict(states=point_status(states,env),goals=point_status(goals,env),reachability=reach)
        lengths_by_set[label]=lengths
    for epoch in [400,800]:
        trace=dict(np.load(ROOT/f'decline_analysis/trace_{epoch}.npz'))
        positions=trace['position'][trace['active']]
        result['traces'][epoch]=point_status(positions,env)
    stage_lengths=lengths_by_set['matched_stage_759'];result['stage_budget_counts']={}
    for label in sorted(set(r['stage'] for r in stage)):
        mask=np.array([r['stage']==label for r in stage]);budget=sets['matched_stage_759'][2]
        result['stage_budget_counts'][label]=dict(probes=int(mask.sum()),reference_over_remaining=int((stage_lengths[mask]>budget[mask]).sum()),
            reference_over_fresh30=int((stage_lengths[mask]>30).sum()))
    original_lengths=lengths_by_set['original_600'];result['original_by_category']={}
    for c,name in enumerate(['same_room','adjacent','diagonal']):
        selected=pairs['categories']==c
        result['original_by_category'][name]=dict(count=int(selected.sum()),over30=int((original_lengths[selected]>30).sum()),max_steps=int(original_lengths[selected].max()))
    result['notes']=['Reachability uses validated native noiseless .05-grid edges and exact float32 success predicate.',
        'Off-grid stage states attach by a collision-free geometric segment to one of their eight nearest free nodes; step counts are approximate there.',
        'Goal-square overlap with a wall is valid: only a reachable point within the square is required. Center validity is checked separately.',
        'Finite-horizon success is not guaranteed under Gaussian noise even when a collision-free route exists.']
    path=ROOT/'analyses/goal_validity/goal_validity_audit.json';path.write_text(json.dumps(result,indent=2))
    np.savez_compressed(ROOT/'analyses/goal_validity/goal_validity_reference_lengths.npz',**lengths_by_set)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
