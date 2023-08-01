#=
test:
- Julia version: 
- Author: cedric
- Date: 2023-05-11
=#
using Gen

possible_action_mappings = []
for dir1=1:4
    for dir2=1:4
        for dir3=1:4
            for dir4=1:4
                action_mapping = [dir1, dir2, dir3, dir4]
                if length(Set(action_mapping)) == length(action_mapping)
                   append!(possible_action_mappings, [action_mapping])
                end
            end
        end
    end
end
n_act_map = length(possible_action_mappings)
println("possible_action_mapping: $possible_action_mappings, count=$n_act_map")
# 1 is up 2 is right 3 is down 4 is left


function get_list_from_cartesian_inds(cartesian_indexes::Vector{CartesianIndex{2}})
    all_candidates_pos = []
    for i=1:length(cartesian_indexes)
        candidate_pos = [cartesian_indexes[i][1], cartesian_indexes[i][2]]
        append!(all_candidates_pos, [candidate_pos])
    end
    return all_candidates_pos
end

function get_dir_from_act(act::Int, action_mapping::Vector{Int})
    true_act = action_mapping[act]
    if true_act == 1 # up
        dir = [-1, 0]
    elseif true_act == 4 # right
        dir = [0, 1]
    elseif true_act == 2 # down
        dir = [1, 0]
    else true_act == 3 # left
        dir = [0, -1]
    end
    return dir
end

@gen function generative_model_no_switch(initial_map::Array{Int}, actions::Array{Int64}, noisy::Bool, infer_action_mapping::Bool,
    switch::Bool, infer_proba_switch::Bool)

    n_steps = length(actions)
    cartesian_indexes = findall(x -> x==8, initial_map);
    n_candidates = length(cartesian_indexes)
    # sample agent identity and action mapping
    if infer_action_mapping
        action_mapping_id ~ categorical(1 / n_act_map * ones(n_act_map))
    else
        action_mapping_id = 1
    end
    action_mapping = possible_action_mappings[action_mapping_id]
    println("action mapping: $action_mapping")
    # fill the initial candidate positions
    candidates_pos = get_list_from_cartesian_inds(cartesian_indexes)
    all_candidate_positions = []
    append!(all_candidate_positions, [candidates_pos])
    println("initial candidate positions: $all_candidate_positions")
    map = initial_map
    if noisy
        noise ~ beta(1, 15)
    else
        noise = 0
    end
    if infer_proba_switch
        proba_switch ~ beta(1, 15)
    else
        proba_switch = 0.1
    end

    for step=1:n_steps
        # sample the agent
        if step==1
            agent_id = {(:agent_id, step)}~ categorical(1/n_candidates * ones(n_candidates))
            println("Sampled agent id: $agent_id")
        else
            if switch  # consider possible agent change
                sample_switch = {(:sample, step)} ~ bernoulli(proba_switch)
                if sample_switch
                    agent_id = {(:agent_id, step)}~ categorical(1/n_candidates * ones(n_candidates))
                    println("Agent switch: $agent_id")
                end
            end
        end
        # update agent position first
        prev_pos = all_candidate_positions[step][agent_id]
        # sample action
        act = actions[step]
        dir = get_dir_from_act(act, action_mapping)
        # in case of noise, perturb action
        if noise > 0
           sample = {(:sample, step)} ~ bernoulli(noise)
           if sample  # if perturbed action
               act = {(:act, step, agent_id)} ~ categorical(1/4 * ones(4))
               if act == actions[step]
                   dir = [0 0]
               else
                  dir = get_dir_from_act(act, action_mapping)
               end
           end
        end
        # compute dir
        candidate_next_pos = [prev_pos[1] + dir[1], prev_pos[2] + dir[2]]
        if map[candidate_next_pos[1], candidate_next_pos[2]] in [0, 3]  # agent can encounter the goal
            new_agent_pos = candidate_next_pos
            # update map
            map[prev_pos[1], prev_pos[2]] = 0
            map[new_agent_pos[1], new_agent_pos[2]] = 8
        else
            new_agent_pos = prev_pos
        end


        #update other agent positions
        new_candidate_positions = []
        for i_candidate=1:n_candidates
            if i_candidate != agent_id
                prev_pos = all_candidate_positions[step][i_candidate]
                # sample action
                act = {(:act, step, i_candidate)} ~ categorical(1/4 * ones(4))
                # compute dir
                dir = get_dir_from_act(act, possible_action_mappings[1])
                candidate_next_pos = [prev_pos[1] + dir[1], prev_pos[2] + dir[2]]
                if map[candidate_next_pos[1], candidate_next_pos[2]] == 0  # candidates cannot move onto the goal
                    new_pos = candidate_next_pos
                    # update map
                    map[prev_pos[1], prev_pos[2]] = 0
                    map[new_pos[1], new_pos[2]] = 8
                else
                    new_pos = prev_pos
                end
                append!(new_candidate_positions, [new_pos])
            else
                append!(new_candidate_positions, [new_agent_pos])
            end

        end
        append!(all_candidate_positions, [new_candidate_positions])
        println("new_candidate_positions: $new_candidate_positions")
    end
end

inputs = [1 1 1 1 1 1 1 1 1;
          1 8 0 0 0 0 0 8 1;
          1 1 0 0 0 0 0 1 1;
          1 1 0 0 0 0 0 1 1;
          1 0 0 0 3 0 0 0 1;
          1 1 0 0 0 0 0 1 1;
          1 1 0 0 0 0 0 1 1;
          1 8 0 0 0 0 0 8 1;
          1 1 1 1 1 1 1 1 1]

actions = [1 4 3 2 1 4 3 1 2 4]
# up, right, left, down, up, right, left, up, down, right
infer_action_mapping = true
noisy = true
switch = true
infer_proba_switch = true
generative_model_no_switch(inputs, actions, noisy, infer_action_mapping, switch, infer_proba_switch)