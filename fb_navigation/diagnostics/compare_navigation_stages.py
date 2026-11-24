"""Monte Carlo rollout metrics shared by the Four Rooms evaluator."""
import jax
import jax.numpy as jnp

from fb_navigation.environments.continuous_four_rooms_jax import Point, _move


def mc_rollout(learner,env,horizon=150,epsilon=.02):
    actions=jnp.array([(a.x,a.y) for a in env.actions]);walls=tuple(env.wall_blocks)
    @jax.jit
    def rollout(params,starts,goals,budgets,keys):
        tasks=learner.backward_network.apply(params['b'],learner.featuriser.transform(goals))
        n=len(starts);limits=jnp.stack([jnp.full(n,30),jnp.full(n,horizon),budgets],axis=1)
        def step(carry,t):
            position,hits,keys,collisions,repeats,longest,run,previous,previous_blocked=carry
            split=jax.vmap(lambda k:jax.random.split(k,4))(keys)
            keys,ek,ak,mk=[split[:,i] for i in range(4)]
            f=learner.forward_network.apply(params['f'],learner.featuriser.transform(position),tasks)
            greedy=jnp.argmax(jnp.einsum('bda,bd->ba',f,tasks),axis=1)
            explore=jax.vmap(lambda k:jax.random.uniform(k)<epsilon)(ek)
            random=jax.vmap(lambda k:jax.random.randint(k,(),0,4))(ak)
            selected=jnp.where(explore,random,greedy)
            def move(p,a,key):
                nxt,_,blocked=_move(Point(*p),Point(*actions[a]),key,env._noise,env.size,walls)
                return jnp.array([nxt.x,nxt.y]),blocked
            following,blocked=jax.vmap(move)(position,selected,mk)
            active=hits>horizon;blocked=blocked&active
            eligible=active[:,None]&(t<limits)
            repeated=blocked&previous_blocked&(selected==previous)
            run=jnp.where(repeated,run+1,jnp.where(blocked,1,0))
            collisions+=eligible&blocked[:,None];repeats+=eligible&repeated[:,None]
            longest=jnp.where(eligible,jnp.maximum(longest,run[:,None]),longest)
            position=jnp.where(active[:,None],following,position)
            reached=jnp.all(abs(position-goals)<=env.threshold_distance,axis=1)
            hits=jnp.where(active&reached,t+1,hits)
            return (position,hits,keys,collisions,repeats,longest,run,selected,blocked),None
        counters=jnp.zeros((n,3),jnp.int32)
        initial=(starts,jnp.full(n,horizon+1,jnp.int32),keys,counters,counters,counters,
                 jnp.zeros(n,jnp.int32),jnp.full(n,-1,jnp.int32),jnp.zeros(n,bool))
        final,_=jax.lax.scan(step,initial,jnp.arange(horizon))
        return dict(hits=final[1],collisions=final[3],repeats=final[4],longest=final[5])
    return rollout
