# Agent Configuration
AGENT_NAME = "orchestrator"

# Model Configuration
MODEL_NAME = "gemini-1.5-flash-preview"
RETRY_ATTEMPTS = 3

# App Configuration
APP_NAME = "story_orchestrator"

# Environment Configuration
GOOGLE_CLOUD_LOCATION = "global"
GOOGLE_GENAI_USE_VERTEXAI = "True"

# World Configuration
MAX_NPCS = 1000
DEFAULT_TICK_DELTA_HOURS = 1.0
INITIAL_GAME_TIME = 0.0

# Simulation
EMOTION_DECAY_RATE = 0.05
EVENT_IMPACT_SCALE = 1.0

# Background Tick Configuration
BACKGROUND_TICK_ENABLED = False  # Disabled by default (pull-based only)
BACKGROUND_TICK_INTERVAL_SECONDS = 5.0  # Real seconds between auto-ticks
BACKGROUND_TICK_DELTA_HOURS = 1.0  # In-game hours per auto-tick

# Memory
MAX_RECENT_MEMORIES = 10

# Interaction Engine
INTERACTION_RATE = 0.3  # Base probability multiplier for NPC-NPC interactions
MAX_INTERACTIONS_PER_LOCATION = 10  # Cap interactions per location per tick
MIN_ENERGY_FOR_INTERACTION = 0.1  # NPCs below this energy won't interact

# Relationship Engine
RELATIONSHIP_DECAY_RATE = 0.01  # Weak ties fade toward zero per tick
RELATIONSHIP_DELTA_SCALE = 1.0  # Global multiplier for relationship changes

# Vitality Engine (Energy & Health dynamics)
ENERGY_DRAIN_PER_TICK = 0.01  # Passive energy expenditure per tick
ENERGY_REGEN_BASE = 0.03  # Base energy recovery per tick (scaled by environment)
HEALTH_REGEN_RATE = 0.005  # Passive health recovery per tick (scaled by safety)
DANGER_HEALTH_DRAIN = 0.02  # Health drain per tick in unsafe environments
DANGER_SAFETY_THRESHOLD = 0.3  # Safety below this causes health drain
HEALTH_ENERGY_CAP_THRESHOLD = 0.5  # Below this health, energy is capped
COLLAPSE_ENERGY_THRESHOLD = 0.0  # At this energy, NPC collapses
DEATH_HEALTH_THRESHOLD = 0.0  # At this health, NPC is incapacitated
INTERACTION_ENERGY_COST = 0.01  # Base energy cost per interaction
CONFLICT_ENERGY_COST = 0.03  # Energy cost for conflict interactions
CONFLICT_HEALTH_DAMAGE = 0.08  # Health damage from conflict interactions
INTIMIDATION_HEALTH_DAMAGE = 0.03  # Health damage from intimidation
