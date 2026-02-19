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
