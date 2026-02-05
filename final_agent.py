import nengo
import nengo.spa as spa
import numpy as np
import grid

# Environment Setup
mymap = """
#########
#  M   R#
#R#R#B#R#
# # # # #
#G Y   R#
#########
"""

class Cell(grid.Cell):
    def color(self):
        if self.wall: return 'black'
        colors = {1:'green', 2:'red', 3:'blue', 4:'magenta', 5:'yellow'}
        return colors.get(self.cellcolor, None)

    def load(self, char):
        self.wall = False
        self.cellcolor = 0
        if char == '#': self.wall = True
        mapping = {'G':1, 'R':2, 'B':3, 'M':4, 'Y':5}
        if char in mapping: self.cellcolor = mapping[char]

world = grid.World(Cell, map=mymap, directions=4)
body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)

col_values = {
    0: [0, 0, 0],        # Empty
    1: [0.2, 0.8, 0.2],  # Green
    2: [0.8, 0.2, 0.2],  # Red
    3: [0.2, 0.2, 0.8],  # Blue
    4: [0.8, 0.2, 0.8],  # Magenta
    5: [0.8, 0.8, 0.2],  # Yellow
}

# Robot Interface
def move(t, x):
    speed, rotation = x
    dt = 0.001
    body.turn(rotation * dt * 25.0)
    body.go_forward(speed * dt * 5.0 if speed > 0.2 else 0)

def detect(t):
    angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
    return [body.detect(d, max_distance=4)[0] for d in angles]

# RGB
def current_color_sensor(t):
    c = col_values.get(body.cell.cellcolor, [0, 0, 0])
    noise = np.random.normal(0, 0.1, 3)  # Rumore ridotto
    return np.clip(np.array(c) + noise, 0, 1)

# SPA Model
D = 64
model = spa.SPA()

with model:
    
    model.vocab = spa.Vocabulary(D)
    model.vocab.parse("RED + GREEN + BLUE + YELLOW + MAGENTA + NONE")
    
    
    model.current_color = spa.State(D, vocab=model.vocab, feedback=0.6)
    model.memory = spa.State(D, vocab=model.vocab, feedback=0.95)
    model.transition = spa.State(D, vocab=model.vocab, feedback=0.8)  # AUMENTATO da 0.5
    
    
    model.current_rgb = nengo.Node(current_color_sensor)
    
    # Classifier
    targets_rgb = {
        'RED': np.array([0.8, 0.2, 0.2]),
        'GREEN': np.array([0.2, 0.8, 0.2]),
        'BLUE': np.array([0.2, 0.2, 0.8]),
        'MAGENTA': np.array([0.8, 0.2, 0.8]),
        'YELLOW': np.array([0.8, 0.8, 0.2])
    }
    
    classifier_ens = nengo.Ensemble(800, 3, radius=1.5)
    nengo.Connection(model.current_rgb, classifier_ens)
    
    def classify_robust(x):
        
        if np.linalg.norm(x) < 0.15:  
            return model.vocab.parse('NONE').v
        
        best_name = 'NONE'
        best_score = 0.25  
        
        for name, rgb in targets_rgb.items():
            
            similarity = np.dot(x, rgb) / (np.linalg.norm(x) * np.linalg.norm(rgb) + 1e-9)
            if similarity > best_score:
                best_score = similarity
                best_name = name
        
        return model.vocab.parse(best_name).v
    
    nengo.Connection(classifier_ens, model.current_color.input,
                    function=classify_robust, synapse=0.05)
    
    # Cortical Binding
    model.cortical = spa.Cortical(
        spa.Actions("transition = memory * current_color")
    )
    
    # Counters
    colors_to_count = ['GREEN', 'BLUE', 'YELLOW', 'MAGENTA', 'RED']
    integrators = {}
    
    for c_name in colors_to_count:
        # Detector ensemble
        det = nengo.Ensemble(300, D, label=f"Det_{c_name}")
        nengo.Connection(model.transition.output, det)
        
        
        count = nengo.Ensemble(300, 1, label=f"Count_{c_name}")  
        nengo.Connection(count, count, synapse=0.1, transform=1.0)
        
        # Target pattern
        target_v = model.vocab.parse(f"RED * {c_name}").v
        
        # Increment function 
        def make_increment(target):
            def increment(x):
                sim = np.dot(x, target)
                if sim > 0.5:  
                    return 1.0  
                elif sim > 0.3:
                    return 0.3  
                return 0
            return increment
        
        nengo.Connection(det, count,
                        function=make_increment(target_v),
                        synapse=0.01)
        
        integrators[c_name] = count
    
    # Basal Ganglia
    actions = spa.Actions(
        "dot(current_color, RED) - dot(current_color, NONE) --> memory = RED",
        "dot(current_color, GREEN) - dot(current_color, NONE) --> memory = GREEN",
        "dot(current_color, BLUE) - dot(current_color, NONE) --> memory = BLUE",
        "dot(current_color, YELLOW) - dot(current_color, NONE) --> memory = YELLOW",
        "dot(current_color, MAGENTA) - dot(current_color, NONE) --> memory = MAGENTA",
        "0.2 --> memory = memory"
    )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
    
    # Debug
    debug_state = {
        'last_print': -10,
        'last_color': 'NONE',
        'transition_count': {'GREEN': 0, 'BLUE': 0, 'YELLOW': 0, 'MAGENTA': 0, 'RED': 0}
    }
    
    def print_status(t):
        if t - debug_state['last_print'] >= 2.0:  
            debug_state['last_print'] = t
            cellcolor = body.cell.cellcolor
            color_names = {0:'NONE', 1:'GREEN', 2:'RED', 3:'BLUE', 4:'MAGENTA', 5:'YELLOW'}
            current_name = color_names.get(cellcolor, 'UNKNOWN')
            rgb = col_values.get(cellcolor, [0,0,0])
            
            # Detect transitions
            if current_name != 'NONE' and current_name != debug_state['last_color']:
                if debug_state['last_color'] == 'RED':
                    debug_state['transition_count'][current_name] = debug_state['transition_count'].get(current_name, 0) + 1
                    print(f"\n>>> TRANSITION DETECTED: RED → {current_name} (count: {debug_state['transition_count'][current_name]})")
                debug_state['last_color'] = current_name
            
            print(f"t={t:.1f}s | Cell: {current_name} | RGB: [{rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f}]")
        return 0
    
    status_node = nengo.Node(print_status)
    
    counter_state = {'last_print': -10}
    
    def print_counts(t, x):
        if t - counter_state['last_print'] >= 5.0:  
            counter_state['last_print'] = t
            print(f"\n{'='*50}")
            print(f"COUNTERS at t={t:.1f}s:")
            for name, val in zip(colors_to_count, x):
                print(f"  RED→{name:7s}: {val:.2f}")
            print(f"{'='*50}")
        return 0
    
    debug_print_node = nengo.Node(print_counts, size_in=len(colors_to_count))
    for i, name in enumerate(colors_to_count):
        nengo.Connection(integrators[name], debug_print_node[i], synapse=0.05)
    
    # Monitor pattern matching in real-time
    similarity_state = {'last_print': -10}
    
    def monitor_similarity(t, trans_vec):
        if t - similarity_state['last_print'] >= 3.0:  
            similarity_state['last_print'] = t
            
            
            sims = {}
            for c_name in colors_to_count:
                target = model.vocab.parse(f'RED * {c_name}').v
                sim = np.dot(trans_vec, target)
                sims[c_name] = sim
            
            max_name = max(sims, key=sims.get)
            max_sim = sims[max_name]
            
            if max_sim > 0.25:  
                print(f"  [Pattern Match] RED→{max_name}: sim={max_sim:.3f}")
                print(f"  [All sims] " + ", ".join([f"{n}:{s:.2f}" for n, s in sims.items()]))
        
        return 0
    
    sim_monitor = nengo.Node(monitor_similarity, size_in=D)
    nengo.Connection(model.transition.output, sim_monitor, synapse=None)
    
    # Output node GUI
    debug_output_node = nengo.Node(size_in=len(colors_to_count))
    for i, name in enumerate(colors_to_count):
        nengo.Connection(integrators[name], debug_output_node[i], synapse=0.05)
    
    # Navigation
    env = grid.GridNode(world, dt=0.005)
    radar = nengo.Ensemble(300, 3, radius=4)
    movement = nengo.Node(move, size_in=2)
    
    nengo.Connection(nengo.Node(detect), radar)
    nengo.Connection(radar, movement[0], function=lambda x: x[1] - 0.5)
    nengo.Connection(radar, movement[1], function=lambda x: x[2] - x[0])
    
    # Exploration noise
    def add_exploration(t):
        return [np.random.uniform(-0.08, 0.08), np.random.uniform(-0.12, 0.12)]
    
    explore = nengo.Node(add_exploration)
    nengo.Connection(explore[0], movement[0], synapse=0.01)
    nengo.Connection(explore[1], movement[1], synapse=0.01)