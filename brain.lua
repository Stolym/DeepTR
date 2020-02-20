
Brain={}
Brain.__index=Brain

function Brain:create(balance)
    local brain={}             -- our new object
    setmetatable(brain, Brain)  -- make Brain handle lookup
    brain.memory={}
    brain.actions={}
    brain.q_table={}
    brain._lreward=0
    brain._saction=0
    brain._ereward=0
    brain._done=0
    brain.epsilon=1
    brain.gamma=0.55
    brain.alpha=0.2
    return brain
end

function Brain:_action(index)
    for k, v in pairs(self.actions[index]) do
        set_joint_state(0, k, v)
        set_joint_state(1, k, v)
    end
end

function Brain:add_state(state)
    if self:index_state(state) ~= 0 then 
        return 0
    end
    table.insert(self.memory, state)
    return 1
end

function Brain:random_action()
    local _action = {}
    
    for i = 0, 19 do
        table.insert(_action, math.random(1, 4))
    end
    self:add_action(_action)
    return (self:index_action(_action))
end

function Brain:add_action(_action)
    if self:index_action(_action) ~= 0 then 
        return 0
    end
    table.insert(self.actions, _action)
    return 1
end

function Brain:index_action(_action)
    local index=0
    local match=true
    for k, v in pairs(self.actions) do
        match=true
        for l, m in pairs(self.actions[k]) do
            if m ~= _action[l] then 
                match=false
            end
        end
        if match == true then
            index=k
        end
    end
    return index
end

function Brain:index_state(state)
    local index=0
    local match=true
    for k, v in pairs(self.memory) do
        match=true
        for l, m in pairs(self.memory[k]) do
            if m ~= state[l] then 
                match=false
            end
        end
        if match == true then
            index=k
        end
    end
    return index
end

local function fround(n)
    return n % 1 >= 0.5 and math.ceil(n) or math.floor(n)
end

local function round(t)
    return fround(t*10)*0.1
end

function Brain:get_state()
    local state = {}
    local ret = ""
    local body_info = get_body_info(0, 0)
    local bx, by, bz = get_body_angular_vel(0, 0)
    local cx, cy, cz = get_body_linear_vel(0, 0)

    table.insert(state, round(body_info.pos.x))
    table.insert(state, round(body_info.pos.y))
    table.insert(state, round(body_info.pos.z))
    table.insert(state, round(body_info.sides.x))
    table.insert(state, round(body_info.sides.y))
    table.insert(state, round(body_info.sides.z))
    table.insert(state, round(body_info.rot.r0))
    table.insert(state, round(body_info.rot.r1))
    table.insert(state, round(body_info.rot.r2))
    table.insert(state, round(body_info.rot.r3))
    table.insert(state, round(body_info.rot.r4))
    table.insert(state, round(body_info.rot.r5))
    table.insert(state, round(body_info.rot.r6))
    table.insert(state, round(body_info.rot.r7))
    table.insert(state, round(body_info.rot.r8))
    table.insert(state, round(body_info.rot.r9))
    table.insert(state, round(body_info.rot.r10))
    table.insert(state, round(body_info.rot.r11))
    table.insert(state, round(body_info.rot.r12))
    table.insert(state, round(body_info.rot.r13))
    table.insert(state, round(body_info.rot.r14))
    table.insert(state, round(body_info.rot.r15))
    table.insert(state, round(bx))
    table.insert(state, round(by))
    table.insert(state, round(bz))
    table.insert(state, round(cx))
    table.insert(state, round(cy))
    table.insert(state, round(cz))

    for i = 0, 19 do
        local dismember_info, fracture_info = 0, 0
        local joint_info = get_joint_info(0, i)
		local x, y, z = get_joint_pos(0, i)
		local pos2 = get_joint_pos2(0, i)
		local radius = get_joint_radius(0, i)
        if get_joint_dismember(0, i) == true then dismember_info=1 else dismember_info=0 end
        if get_joint_fracture(0, i) == true then fracture_info=1 else fracture_info=0 end
        table.insert(state, (joint_info["state"] - 1))
        table.insert(state, dismember_info)
        table.insert(state, fracture_info)
        table.insert(state, round(x))
        table.insert(state, round(y))
        table.insert(state, round(z))
        for key,value in pairs(pos2) do 
            table.insert(state, round(value))
        end
        table.insert(state, round(radius))
    end
    self:add_state(state)
    return state
end

function Brain:render()
    local reward = self:reward()
    self:update_reward()
    self:update_q_table(self:get_state())
    return self:get_state(), reward, self:done(), self:info()
end

function Brain:state()
    return self:index_state(self:get_state())
end

function Brain:update_reward()
    local pinfo=get_player_info(0)
    local einfo=get_player_info(1)
    self._lreward = pinfo["score"]
    self._ereward = einfo["score"]
end

function Brain:reward()
    local binfo = get_body_info (0, 0)
    local pinfo=get_player_info(0)
    local einfo=get_player_info(1)
    local reward=(pinfo["score"] - self._lreward) - (einfo["score"] - self._ereward)
    if binfo.pos.z > 1 then
        if  binfo.pos.z - 1 > 1.5 then reward = reward + 1 else reward = reward + binfo.pos.z - 1.5 end
    else
        reward = reward - (1 - binfo.pos.z)
    end
    return reward
end

function Brain:update_q_table(state)
    local sindex = self:index_state(state)

    if self.q_table[sindex] == nil then
        local nq_table = {}
        for i = 0, table.getn(self.actions) do
            table.insert(nq_table, 0)
        end
        self.q_table[sindex] = nq_table
    end
    if table.getn(self.q_table[sindex]) ~= table.getn(self.actions) then
        for i = table.getn(self.q_table[sindex]), table.getn(self.actions) do
            table.insert(self.q_table[sindex], 0)
        end
    end
end

local function argmax(_table)
    local value=_table[1]
    local vindex=0

    for k, v in pairs(_table) do
        if v > value then
            vindex = k
            value = v
        end
    end
    echo(ret)
    return vindex
end

function Brain:choose_epsilon_action(state)
    local sindex = self:index_state(state)
    local besp = false

    self:update_q_table(state)
    if math.random(0, 10000) / 10000 < self.epsilon then
        self._saction = self:random_action()
        besp = true
    else
        self._saction = argmax(self.q_table[sindex])
    end
    if table.getn(self.actions) <= self._saction and besp == false then
        self._saction = self:random_action()
    end
    self:_action(self._saction)
end

function Brain:done()
    return (0.0)
end

function Brain:info()
    return (0.0)
end

local engine = false
local train = false
local logic = {}
local tick = os.clock()
local _state = {}
local epoch = 0
local _timer = 1

local function frame()
    local x = os.clock()

    if engine == true and train == false then
        logic:choose_epsilon_action(_state)
        run_frames(30)
        tick = x
        train = true
    end

    if x - tick > _timer and engine == true and train == true then
        tick = x
        _next_state, reward, done, info = logic:render()
        local _sindex = logic:index_state(_state)
        local _nsindex = logic:index_state(_next_state)

        old_value = logic.q_table[_sindex][logic._saction]
        next_max = math.max(unpack(logic.q_table[_nsindex]))
        new_value = (1 - logic.alpha) * old_value + logic.alpha * (reward + logic.gamma * next_max)
        logic.q_table[_sindex][logic._saction] = new_value
        _state = _next_state
        train = false
    end
end

local function maintext()
    set_color(0, 0, 0, 1)
    draw_text("DeepTR Actions Memory ".. table.getn(logic.actions) .. " State Memory " .. table.getn(logic.memory), 100, 100)
    draw_text("Reward ".. logic:reward(), 100, 120)
    draw_text("Epsilon ".. logic.epsilon, 100, 160)
    draw_text("Launch Game Iteration ".. epoch, 100, 140)
end

local function launch()
    logic._lreward = 0
    logic._ereward = 0
    _state = logic:get_state()
    engine = true
    epoch = epoch + 1
    logic.epsilon = logic.epsilon * 0.88
    if epoch == 2000 then _timer = 2 end
    if epoch == 10000 then _timer = 4 end
end

local function _end()
    engine = false
    start_new_game()
end

local function start()
    run_cmd("clear")
    logic=Brain:create()
    add_hook("draw2d", "maintext", maintext)
    add_hook("draw2d", "frame", frame)
    add_hook("new_game", "launch", launch)
    add_hook("end_game", "launch", _end)
    launch()
end
remove_hook("end_game", "launch")
start()
