using QuickPOMDPs
using Combinatorics
using POMDPSimulators
using POMDPPolicies
using Distributions
using POMDPModelTools
using QMDP
using Distances
using BeliefUpdaters
using Plots
#using POMDPModels
using PointBasedValueIteration
using POMDPs
using Random # for AbstractRNG

#_x__G__x

function simulate(pos1, pos2, self, a)
    #return pos1_p, pos2_p
    if self==1
        if pos1 == goal_pos
            pos1_p = goal_pos
        else
            pos1_p = pos1 + a
            if (pos1_p < 0) || (pos1_p > width) || pos1_p == pos2
                pos1_p = pos1
            end
        end
        pos2_p = pos2 + 1
        if (pos2_p < 0) || (pos2_p > width) || pos2_p == goal_pos || pos2_p == pos1_p
            pos2_p = pos2
        end
    else
        if pos2 == goal_pos
            pos2_p = goal_pos
        else
            pos2_p = pos2 + a
            if (pos2_p < 0) || (pos2_p > width) || pos2_p == pos1
                pos2_p = pos2
            end
        end
        pos1_p = pos1 + 1
        if (pos1_p < 0) || (pos1_p > width) || pos1_p == goal_pos || pos1_p == pos2_p
            pos1_p = pos1
        end
    end
    return (pos1_p, pos2_p)
end

#given this state, and this self, what would happen? How would our belief change?
function get_next_states(state,self,a)
    (pos1_given_1, pos2_given_1) = simulate(state[1], state[2], 1, a)
    (pos1_given_2, pos2_given_2) = simulate(state[1], state[2], 2, a)
    #what is the likelihood of this happening, given my belief self=1 or self=2
    lik = [0,0]
    #if they predict same, equal likelihood
    if (pos1_given_1 == pos1_given_2) && (pos2_given_1 == pos2_given_2)
        lik = [0.5,0.5]
    #otherwise, 1 for true
    else
        lik[self] = 1
    end
    b1 = state[3] * lik[1]
    b2 = (1-state[3]) * lik[2]
    b = b1/(b1+b2)
    if self==1
        return [((pos1_given_1, pos2_given_1, b), 1)]
    else
        return [((pos1_given_2, pos2_given_2, b), 1)]
    end
end

function get_states()
    frontier = [init_state]
    all_states = [init_state]
    transitions = Dict()
    ugh=1
    while length(frontier)>0
        s = pop!(frontier)
        for a in actions
            #for this state, what states could we transition to?
            all_next_states = []
            if s[3]>0
                append!(all_next_states, [(sp, p*s[3]) for (sp,p) in get_next_states(s,1,a)])
            end
            if s[3] < 1
                append!(all_next_states, [(sp, p*(1-s[3])) for (sp,p) in get_next_states(s,2,a)])
            end
            transitions[(s,a)] = Dict()
            for (sp, p) in all_next_states
                if !(sp in keys(transitions[(s,a)]))
                    transitions[(s,a)][sp] = 0
                end
                transitions[(s,a)][sp] = transitions[(s,a)][sp] + p
                if !(sp in all_states)
                    push!(all_states, sp)
                    push!(frontier, sp)
                end
            end
        end
    end
    return all_states, transitions
end

function expected_reward(s, a)
    er = 0
    if (s[1] == goal_pos)
        er = er + s[3]
    else
        er = er - s[3]
    end
    if (s[2] == goal_pos)
        er = er + (1 - s[3])
    else
        er = er - (1 - s[3])
    end
    return er
end


function value_iteration(states, transitions; max_iterations=1000, tolerance=1e-6)
    # Initialization
    V = Dict()
    for s in states
        V[s] = 0
    end
    actions = [-1, 1]
    for iteration in 1:max_iterations
        delta = 0.0
        # Update the value function for each state
        for (i,s) in enumerate(states)
            v = V[s]
            # Find the maximum Q-value over actions
            max_q_value = -Inf
            for a in actions
                er = expected_reward(s, a)
                q_value = er + (0.95 * sum(p * V[sp] for (sp,p) in transitions[(s,a)]))
                max_q_value = max(max_q_value, q_value)
            end
            # Update the value function
            V[s] = max_q_value
            delta = max(delta, abs(v - V[s]))
        end
        # Check for convergence
        if delta < tolerance
            break
        end
    end
    return V
end
 
width = 8
init_state = (2,8,0.5)
actions = [-1, 1]
goal_pos = 5
(states, transitions) = get_states()

v = value_iteration(states, transitions)
print(v)

s = init_state

evs = []
for a in actions
    ev = 0
    for (sp, p) in transitions[(s,a)]
        ev = ev + p*v[sp]
    end
    push!(evs, ev)
end
print(evs)


function render_grid_world(s)
    #x, y = sort(unique(a)), sort(unique(b)) # the lattice x and y
    #row1, row2, row3, row4
    nrows = 5#height
    ncols = width
    a = fill(-1, nrows, ncols)
    a[3,:] .= 0
    a[3, 5] = 2
    for pos in s[1:2]
        r = 3#pos[2]
        c = pos
        a[r, c] = 1
    end
    col = cgrad([:black, :gray,:red,:green])
    gui(heatmap(a, color=col, clim=(-1,2)))
end

render_grid_world(s)

#=
#get more control over stepping thru sim, or just save images
f = true
for (s, a, r) in stepthrough(pomdp, policy, "s,a,r", max_steps=8)
    global f
    if f
        @show s
        println()
        render_grid_world(s)
        while read(stdin, Char) != ' '
            continue
        end
        f = false
    end
    sp = transition_func(s, a)
    @show a
    @show sp
    @show r
    println()
    render_grid_world(sp)
    while read(stdin, Char) != ' '
        continue
    end
end
=#




# test grid world
function test_render()
    s = get_state_space()[1]
    println(s)
    render_grid_world(WS2str(s))
end
