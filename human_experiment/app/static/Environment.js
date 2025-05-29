AVATAR_NUM = 8
GOAL_NUM = 2

//add noisy Option
//deal w contingency bounds


class Game {
    #board;
    #level_count;
    #step_limit;
    #avatarPosition;
    #ns_positions;
    #possible_levels = [];
    #gameTypes = [];
    #gameType;
    #action_counts = [];
    #current_action_count = [];
    #avatar_start_position;
    #ns_interactions = [];
    #current_ns_interactions;
    #wall_interactions = [];
    #current_wall_interactions;
    #maps = [];
    #current_map;
    #self_start_locs = [];
    #self_locs = [];
    #current_self_locs = [];
    #action_lists = [];
    #current_action_list = [];
    #ns_locs = []
    #current_ns_locs = [];
    #num_levels;
    #shuffle_key_maps = [];
    #current_shuffle_key_map; 
    #mockSelf;
    #change_agent_every;
    #avatar_noise;
    #goal_position;
    #poss_goal_positions = [];
    #current_poss_goal_positions = [];
    #poss_goal_interactions = [];
    #current_poss_goal_interactions;
    #goal_positions = [];
    #reached_goal = [];
    #contingency_directions = [];
    #contingency_bounds = [];

    constructor(gameType) {
        this.#num_levels = 40; 

        if (gameType == "change_agent_10") {
            this.#change_agent_every = 10;
        } else {
            this.#change_agent_every = 7;
        }
        this.#gameType = gameType;
        this.#level_count = 0;
        this.#step_limit = 150;

        if (gameType == "contingency_noisy") {
            this.#avatar_noise = 1/3
        } else {
            this.#avatar_noise = 0
        }


        if (gameType === "logic" || gameType === "logic_perturbed") {
            logic_levels(this.#possible_levels);
        } else if (gameType === "contingency" || gameType === "contingency_perturbed" || gameType === "change_agent" || gameType === "change_agent_10"
            || gameType === "shuffle_keys" || gameType === "change_agent_perturbed" || gameType === "contingency_noisy") {
            contingency_levels(this.#possible_levels);
        } else if (gameType === "logic_u") {
            logic_u_levels(this.#possible_levels);
        } else if (gameType === "contingency_u" || gameType === "change_agent_u" || gameType === "shuffle_keys_u") {
            contingency_u_levels(this.#possible_levels);
        } else if (game_type === 'contingency_2') {
            contingency_2_levels(this.#possible_levels);
        } else if (game_type === 'contingency_6') {
            contingency_6_levels(this.#possible_levels);
        } else if (game_type === 'contingency_8') {
            contingency_8_levels(this.#possible_levels);
        }
        this.reset_game(gameType)
    }

    reset_game(gameType) {
        //set map
        this.#gameType = gameType;
        let rn = rand(this.#possible_levels.length);
        this.setBoard(JSON.parse(JSON.stringify(this.getLevel(rn))));
        this.#current_map = JSON.parse(JSON.stringify(this.#board));
        this.setGoalPositions()
        this.setAvatarPos(random_avatar_pos(this.#board));
        this.#avatar_start_position = this.#avatarPosition

        //no idea what's going on here
        if (gameType === "change_agent_perturbed" || gameType === "contingency_perturbed") {  // Construct the mock self
            this.#num_levels = 54;
        } else if (gameType === "logic_perturbed") {
            this.#num_levels = 150;
        }

        if (gameType === "shuffle_keys" || gameType === "shuffle_keys_u") {
            this.shuffle_key_mappings()
        }
        
        this.#current_action_count = 0
        this.#current_wall_interactions = 0
        this.#current_poss_goal_interactions = 0;
        this.#current_ns_interactions = 0
        this.#current_self_locs = [];
        this.#current_action_list = [];
        this.#current_self_locs.push(JSON.parse(JSON.stringify(this.#avatarPosition)));
        this.#current_ns_locs = [];
        this.computeNsPositions();
        this.#current_ns_locs.push(JSON.parse(JSON.stringify(this.#ns_positions)));
        if (gameType !== "logic" && gameType !== "logic_u") {
            this.setContingencyDirectionsAndBounds()
        }
    }

    setGoalPositions() {
        //if only one goal, poss_goal_positions has length 1
        let poss_goals = []
        for (var r=0; r<this.#board.length; r++) {
            for (var c=0; c<this.#board[r].length; c++) {
                if (this.#board[r][c] == GOAL_NUM) {
                    poss_goals.push([r,c])
                }
            }
        }
        this.#current_poss_goal_positions = poss_goals
        this.#goal_position = poss_goals[rand(poss_goals.length)]
    }

    isNextToGoal() {
        let dist_vertical = this.#goal_position[0] - this.getAvatarPos()[0];
        let dist_horiz = this.#goal_position[1] - this.getAvatarPos()[1];

        return ((dist_horiz === 0 && (dist_vertical === -1 || dist_vertical === 1)) || (
            dist_vertical === 0 && (dist_horiz === 1 || dist_horiz === -1)));
    }

    addToMaps(oldMap) {
        this.#maps.push(JSON.parse(JSON.stringify(oldMap)));
    }

    getGameType() {
        return this.#gameType;
    }

    getNumLevels() {
        return this.#num_levels;
    }

    getBoard() {
        return this.#board;
    }

    getNsPositions() {
        return this.#ns_positions;
    }

    //compute based on current poss selves and self pos
    computeNsPositions() {
        let poss_avatars = []
        for (var r=0; r<this.#board.length; r++) {
            for (var c=0; c<this.#board[r].length; c++) {
                if (this.#board[r][c] == AVATAR_NUM) {
                    poss_avatars.push([r,c])
                }
            }
        }
        var s = (a) => arraysEqual(a, this.#avatarPosition);
        poss_avatars.splice(poss_avatars.findIndex(s), 1);

        this.#ns_positions = poss_avatars;
    }

    getAvatarPos() {
        return this.#avatarPosition;
    }

    getLevelCount() {
        return this.#level_count;
    }

    getCurrentActionCount() {
        return this.#current_action_count
    }

    getLevels() {
        return this.#possible_levels;
    }

    getLevel(levelNo) {
        return this.#possible_levels[levelNo];
    }

    setBoard(board) {
        this.#board = board;
    }

    setAvatarPos(pos) {
        this.#avatarPosition = pos;
    }

    incrementLevelCount() {
        this.#level_count++;
    }
    decrementLevelCount() {
        this.#level_count--;
    }

    incrementActionCount() {
        this.#current_action_count++;
    }

    nextLevel() {
        //save data from last time
        this.#gameTypes.push(this.#gameType)
        this.#maps.push(this.#current_map)
        this.#self_start_locs.push(JSON.parse(JSON.stringify(this.#avatar_start_position)));
        this.#self_locs.push(deepCopy(this.#current_self_locs));
        this.#action_lists.push(deepCopy(this.#current_action_list));
        this.#ns_locs.push(deepCopy(this.#current_ns_locs));
        this.#goal_positions.push(this.#goal_position)
        this.#poss_goal_positions.push(this.#current_poss_goal_positions)
        this.#action_counts.push(this.#current_action_count);
        this.#wall_interactions.push(this.#current_wall_interactions);
        this.#poss_goal_interactions.push(this.#current_poss_goal_interactions);
        this.#ns_interactions.push(this.#current_ns_interactions);
        this.#shuffle_key_maps.push(this.#current_shuffle_key_map)
        //reset game
        this.reset_game(this.#gameType)
        //increment level count
        this.incrementLevelCount();
    }

    shuffle_key_mappings() {
        let random_array = [0, 1, 2, 3];
        this.#current_shuffle_key_map = rshuffle(random_array);
    }

    // Some of ns sprites will oscillate up and some will oscillate down
    move_ns_contingency() {
        //ns_positions should be currently up to date
        //let's first update the board
        
        //pos is [r,c]
        for (let i = 0; i < this.#ns_positions.length; i++) {
            //let curr = this.#ns_positions[i]; //computeNsPositions()?
            var curr_pos =  this.#ns_positions[i].slice(0); //is this going to cause problems? when we change ns positions

            var can_move = false

            if (this.#contingency_directions[i] === 0) {  // horizontal move
                let rn = rand(2); // 0 = left, 1 = right
                
                //make sure we stay within this avatar's bounds
                //at max, go left
                if (this.#ns_positions[i][1] >= this.#contingency_bounds[i][1][1]) {
                    rn = 0;
                }
                //at min, go right
                if (this.#ns_positions[i][1] <= this.#contingency_bounds[i][1][0]) {
                    rn = 1;
                }

                if (rn === 1) { // right
                    if (this.canMoveNs(3, this.#ns_positions[i]) === 1) {
                        can_move = true
                        this.#ns_positions[i] = [this.#ns_positions[i][0], this.#ns_positions[i][1] + 1];
                    }
                } else if (rn === 0) { // left
                    if (this.canMoveNs(2, this.#ns_positions[i]) === 1) {
                        can_move = true
                        this.#ns_positions[i] = [this.#ns_positions[i][0], this.#ns_positions[i][1] - 1];
                    }
                }
            } else if (this.#contingency_directions[i] === 1) {  // vertical move
                let rn = rand(2); // 0 = up, 1 = down
                //at max, go down
                if (this.#ns_positions[i][0] <= this.#contingency_bounds[i][0][0]) {
                    rn = 1;
                }
                //at max, go up
                if (this.#ns_positions[i][0] >= this.#contingency_bounds[i][0][1]) {
                    rn = 0;
                }

                if (rn === 1) { // down
                    if (this.canMoveNs(1, this.#ns_positions[i]) === 1) {
                        can_move = true
                        this.#ns_positions[i] = [this.#ns_positions[i][0] + 1, this.#ns_positions[i][1]];
                    }
                } else if (rn === 0) { // up
                    if (this.canMoveNs(0, this.#ns_positions[i]) === 1) {
                        can_move=true
                        this.#ns_positions[i] = [this.#ns_positions[i][0] - 1, this.#ns_positions[i][1]];
                    }
                }
            } else {
                assert(false)
            }
            if (can_move) {
                //restore whatever was at old coor in board
                if (this.#current_poss_goal_positions.find(el => (el[0] === curr_pos[0]) && (el[1] === curr_pos[1]))) {
                    this.#board[curr_pos[0]][curr_pos[1]] = GOAL_NUM;
                } else {
                    this.#board[curr_pos[0]][curr_pos[1]] = 0;
                }
                //update board w new coor
                this.#board[this.#ns_positions[i][0]][this.#ns_positions[i][1]] = AVATAR_NUM;
            }
        }

        if ((this.#gameType === 'change_agent_perturbed') || (this.#gameType === 'contingency_perturbed')) { // Move mock self as well, if it exists
            if (this.getLevelCount() > 33) {
                this.#mockSelf.move(this.getBoard());
            }
        }
    }

    setContingencyDirectionsAndBounds() {
        let directions = []
        let bounds = [] //[[xmin, xmax], [ymin, ymax]]
        let v = 2;
        let h = 2;


        for (let i = 0; i < this.#ns_positions.length; i++) {
            let rn = rand(2); // 0 or 1.
            if (v === 0) {
                rn = 1;
            } else if (h === 0) {
                rn = 0;
            }
            // 0 = horizontal, 1 = vertical.
            directions.push(rn);
            if (rn === 0) {
                v--;
            } else {
                h--;
            }

            bounds.push([[this.#ns_positions[i][0]-4, this.#ns_positions[i][0]+4], [this.#ns_positions[i][1]-4, this.#ns_positions[i][1]+4]])
        }
        this.#contingency_directions = directions;
        this.#contingency_bounds = bounds;
    }



    change_agent() {
        if (this.getCurrentActionCount() % this.#change_agent_every !== 0) {
            return;
        }

        let temp = this.#avatarPosition.slice(0);

        let rn = rand(this.#ns_positions.length);
        this.setAvatarPos(this.#ns_positions[rn]);
        this.#ns_positions[rn] = temp;

        if (this.#mockSelf != null) {
            this.#mockSelf.start_navigating();
        }
    }

    move_ns_change_agent() {
        this.change_agent();
        let action_pos_dict = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        let cc = [0, 0, 0, 0];
        var rn;
        var stay = false;
        for (let i = 0; i < this.#ns_positions.length; i++) {  // Each non-self

            do {
                rn = rand(4); // 0 = left, 1 = right, 2 = up, 3 = down

                if (cc[rn] === 0) {
                    cc[rn]++;
                }

                if (cc[0] !== 0 && cc[1] !== 0 && cc[2] !== 0 && cc[3] !== 0) { // stay, cannot move anywhere
                    stay = true;
                    break;
                }

            } while (this.canMoveNs(rn, this.#ns_positions[i]) === 0); // iterate if ns cannot move

            if (!stay) {
                if (this.#current_poss_goal_positions.find(el => (el[0] === this.#ns_positions[i][0]) && (el[1] === this.#ns_positions[i][1]))) {
                    this.#board[this.#ns_positions[i][0]][this.#ns_positions[i][1]] = GOAL_NUM; // old coor
                } else {
                    this.#board[this.#ns_positions[i][0]][this.#ns_positions[i][1]] = 0; // old coor
                    // set avatar's old position to grass
                }

                this.#ns_positions[i] = [this.#ns_positions[i][0] + action_pos_dict[rn][0],
                    this.#ns_positions[i][1] + action_pos_dict[rn][1]];

                this.#board[this.#ns_positions[i][0]][this.#ns_positions[i][1]] = AVATAR_NUM;
            }

        }

        if ((this.#gameType === 'change_agent_perturbed') || (this.#gameType === 'contingency_perturbed')) { // Move mock self as well, if it exists
            if (this.getLevelCount() > 33) {
                this.#mockSelf.move(this.getBoard());
            }
        }
    }

    /*
    *
    * up = 0
    * down = 1
    * left = 2
    * right = 3
    *
    */
   //return 0 if same level, 1 if new level
    step(direction) {
        this.#current_action_list.push(direction)
        this.incrementActionCount();

        if (this.getGameType() === "shuffle_keys" || this.getGameType() === "shuffle_keys_u") {
            direction = this.#current_shuffle_key_map[direction];
        }

        // *-*-*-*-*-* Move the non-selves *-*-*-*-*-* //
        if (this.getGameType() === "contingency" || this.getGameType() === "contingency_perturbed" || this.getGameType() === "contingency_u" || this.getGameType() === "shuffle_keys" || this.getGameType() === "shuffle_keys_u" || this.getGameType() === "contingency_noisy" || this.getGameType() === "contingency_8" || this.getGameType() === "contingency_6" || this.getGameType() === "contingency_2") {
            this.move_ns_contingency(); // move non-self sprites
        } else if (this.getGameType() === "change_agent" || this.getGameType() === "change_agent_perturbed" || this.getGameType() === "change_agent_u" ||  this.getGameType() === "change_agent_10" ) {
            this.move_ns_change_agent(); // move non-self sprites
        }
        this.#current_ns_locs.push(deepCopy(this.#ns_positions));
        //nonselves don't move in logic game

        // *-*-*-*-*-* Move the avatar *-*-*-*-*-* //
        var avatar_move = this.move_avatar(direction)
        if (avatar_move == 2) {
            //this.nextLevel();
            this.#reached_goal.push(1)
            return 1; //success
        }

        if (this.#current_action_count > this.#step_limit) {
            //this.nextLevel();
            this.#reached_goal.push(0)
            return 2; //failure
        }
        return 0;
    }

    move_avatar(direction) {
        //with noise prob, sample a new random direction
        if (Math.random() < this.#avatar_noise) {
            direction = Math.floor(Math.random() * 4);
        }

        let can_move = this.canMove(direction) 
        //can move, update board and avatar pos
        if (can_move === 1 || can_move == 2 || can_move == 3) {
            let x = this.getAvatarPos()[0]
            let y = this.getAvatarPos()[1]

            let [new_x, new_y] = get_new_xy(x, y, direction)
            if (this.#current_poss_goal_positions.find(el => (el[0] === x) && (el[1] === y))) {
                this.#board[x][y] = GOAL_NUM;
            } else {
                this.#board[x][y] = 0;
            }
            this.#board[new_x][new_y] = AVATAR_NUM;
            this.#avatarPosition = [new_x, new_y];
            if (can_move == 3) {
                this.#current_poss_goal_interactions++;
            }
        //cannot move, increment interaction counts
        } else {
            if (can_move === 0) { // Wall
                this.#current_wall_interactions++;
            } else if (can_move === -1) { // Non-Self
                this.#current_ns_interactions++;
            }
        }
        //record new self loc
        this.#current_self_locs.push(deepCopy(this.#avatarPosition));
        //at goal?
        return can_move
    }

    // returns 0 if the avatar cannot move to the specified location, -1 if there is a self, 1 if avatar can move
    // and 2 if avatar reaches goal
    canMove(direction) {
        let x = this.getAvatarPos()[0]
        let y = this.getAvatarPos()[1]
        var next = 0;
        var next_pos = []
        switch (direction) {
            case 0: // up
                next_pos = [x - 1, y]
                next = this.getBoard()[x - 1][y];
                break;
            case 1: // down
            next_pos = [x + 1, y]
                next = this.getBoard()[x + 1][y];
                break;
            case 2: // left
                next_pos = [x, y - 1]
                next = this.getBoard()[x][y - 1];
                break;
            case 3: // right
                next_pos = [x, y + 1]
                next = this.getBoard()[x][y + 1];
                break;
        }

        // If there is a wall return 0. If there is ns, return -1.
        if (next === 1) {
                return 0;
        } else if (next === AVATAR_NUM) {
                return -1;
        } else if (next === 0) { // There is grass, can move
            return 1;
        } else if (next === GOAL_NUM) { // There is a possible goal, can move
            if (next_pos[0] === this.#goal_position[0] && next_pos[1] === this.#goal_position[1]) { // Reached true Goal!
                return 2;
            } else {
                return 3;
            }
        }
    }

    // returns 0 if the ns cannot move to the specified location, 1 if ns can move
    canMoveNs(direction, ns) {
        let x = ns[0]
        let y = ns[1]
        var next = 0;
        var next_pos = []
        switch (direction) {
            case 0: // up
                    next_pos = [x - 1, y]
                next = this.getBoard()[x - 1][y];
                break;
            case 1: // down
                next_pos = [x + 1, y]
                next = this.getBoard()[x + 1][y];
                break;
            case 2: // left
                next_pos = [x, y - 1]
                next = this.getBoard()[x][y - 1];
                break;
            case 3: // right
                 next_pos = [x, y + 1]
                next = this.getBoard()[x][y + 1];
                break;
        }

        if (next === 1 || next === AVATAR_NUM || next === 3) { // Don't allow to move to goal
            return 0;
        } else if (next === 0) { // There is grass, can move
            return 1;
        } else if (next === GOAL_NUM) { //possible goal, can move if not the real goal
            if (next_pos[0] === this.#goal_position[0] && next_pos[1] === this.#goal_position[1]) { // true goal
                return 0;
            } else {
                return 1;
            }
        }
    }


    getData() {
        var datamap = {
            "steps": this.#action_counts,
            "game_type": this.#gameTypes,
            "wall_interactions": this.#wall_interactions,
            "ns_interactions": this.#ns_interactions,
            "map": this.#maps,
            "self_start_locs": this.#self_start_locs,
            "self_locs": this.#self_locs,
            "ns_locs": this.#ns_locs,
            "n_levels": this.getNumLevels(),
            "poss_goal_interactions": this.#poss_goal_interactions,
            "poss_goal_positions": this.#poss_goal_positions,
            "goal_positions": this.#goal_positions,
            "shuffle_key_maps": this.#shuffle_key_maps,
            "action_lists": this.#action_lists,
            "reached_goal": this.#reached_goal
        };

        var data = {"data": datamap}
        return data;
    }
}

// Returns random avatar position.
// If mockSelf is true, return the position of the mock self instead
function random_avatar_pos(board, mockSelf = false) {
    let poss_avatars = []
    for (var r=0; r<board.length; r++) {
        for (var c=0; c<board[r].length; c++) {
            if (board[r][c] == AVATAR_NUM) {
                poss_avatars.push([r,c])
            }
        }
    }
    return poss_avatars[rand(poss_avatars.length)]
}




function rshuffle(array) {
    let currentIndex = array.length, randomIndex;

    // While there remain elements to shuffle...
    while (currentIndex !== 0) {

        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        // And swap it with the current element.
        [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
    }

    return array;
}

function rand(level_amt) {
    return Math.floor(Math.random() * level_amt);
}

function arraysEqual(a, b) {
    for (var i = 0; i < a.length; ++i) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function logic_levels(levels, perturbed = false) {
    var level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]];
    if (perturbed) {
        level[1][4] = AVATAR_NUM;
    }

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    if (perturbed) {
        level[7][4] = AVATAR_NUM;
    }

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    if (perturbed) {
        level[1][4] = AVATAR_NUM;
    }

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, AVATAR_NUM, 0, 0, 0, 1, 1, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    if (perturbed) {
        level[1][4] = AVATAR_NUM;
    }
    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 1, 1, AVATAR_NUM, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    if (perturbed) {
        level[7][4] = AVATAR_NUM;
    }

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 1, 1, AVATAR_NUM, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, AVATAR_NUM, 1, 1, 0, 1, 1, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    if (perturbed) {
        level[4][1] = AVATAR_NUM;
    }

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 1, 1, AVATAR_NUM, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, AVATAR_NUM, 0, 0, 0, 1, 1, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    if (perturbed) {
        level[4][7] = AVATAR_NUM;
    }

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 1, 1, AVATAR_NUM, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    if (perturbed) {
        level[4][7] = AVATAR_NUM;
    }
    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    if (perturbed) {
        level[4][1] = AVATAR_NUM;
    }

    levels.push(deepCopy(level));
}


//add levels with goal uncertainty
function logic_u_levels(levels) {
    var level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]];

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, AVATAR_NUM, 0, 0, 0, 1, 1, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 1, 1, AVATAR_NUM, 1],
        [1, 1, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, AVATAR_NUM, 0, 0, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 1, 1, AVATAR_NUM, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, AVATAR_NUM, 1, 1, 0, 1, 1, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 1, 1, AVATAR_NUM, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, AVATAR_NUM, 0, 0, 0, 1, 1, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 1, 1, AVATAR_NUM, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    levels.push(deepCopy(level));

    // ---------------------------------

    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, GOAL_NUM, 0, 0, 1, 1],
        [1, AVATAR_NUM, 1, 1, 0, 0, 0, AVATAR_NUM, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];

    levels.push(deepCopy(level));
}

function contingency_levels(levels) {
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
}

function contingency_u_levels(levels) {
    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ];
    levels.push(deepCopy(level));
    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ];
    levels.push(deepCopy(level));
    level = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ];
    levels.push(deepCopy(level));
}


function contingency_2_levels(levels) {
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
}

function contingency_6_levels(levels) {
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
}

function contingency_8_levels(levels) {
    levels.push([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, GOAL_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, AVATAR_NUM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]);
}

function get_new_xy(x, y, direction) {
    let new_x = x;
    let new_y = y;
    switch (direction) {
        case 0:
            new_x--;
            break;
        case 1:
            new_x++;
            break;
        case 2:
            new_y--;
            break;
        case 3:
            new_y++;
            break;
    }

    return ([new_x, new_y]);
}

const deepCopy = (arr) => {
    let copy = [];
    arr.forEach(elem => {
        if (Array.isArray(elem)) {
            copy.push(deepCopy(elem))
        } else {
            if (typeof elem === 'object') {
                copy.push(deepCopyObject(elem))
            } else {
                copy.push(elem)
            }
        }
    })
    return copy;
}