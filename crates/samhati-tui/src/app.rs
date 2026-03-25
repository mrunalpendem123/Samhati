use chrono::Local;
use sysinfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Chat,
    Dashboard,
    Models,
    Wallet,
    Settings,
}

impl Tab {
    pub const ALL: [Tab; 5] = [
        Tab::Chat,
        Tab::Dashboard,
        Tab::Models,
        Tab::Wallet,
        Tab::Settings,
    ];

    pub fn index(self) -> usize {
        match self {
            Tab::Chat => 0,
            Tab::Dashboard => 1,
            Tab::Models => 2,
            Tab::Wallet => 3,
            Tab::Settings => 4,
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Tab::Chat => "Chat",
            Tab::Dashboard => "Dashboard",
            Tab::Models => "Models",
            Tab::Wallet => "Wallet",
            Tab::Settings => "Settings",
        }
    }

    pub fn next(self) -> Tab {
        Tab::ALL[(self.index() + 1) % Tab::ALL.len()]
    }

    pub fn prev(self) -> Tab {
        Tab::ALL[(self.index() + Tab::ALL.len() - 1) % Tab::ALL.len()]
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: String,
    pub confidence: Option<f32>,
    pub n_nodes: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub domain: String,
    pub size_gb: f32,
    pub params: String,      // e.g. "1.5B", "3B", "7B"
    pub quant: String,       // e.g. "Q4_K_M", "Q8_0", "F16"
    pub smti_bonus: String,
    pub min_ram_gb: f32,     // minimum RAM to run
    pub installed: bool,
    pub active: bool,
    pub recommended: bool,   // auto-detected as good fit for this device
}

#[derive(Debug, Clone)]
pub struct TxEntry {
    pub timestamp: String,
    pub tx_type: String,
    pub amount: f64,
    pub status: String,
}

/// Display info for a swarm node shown on the dashboard.
#[derive(Debug, Clone)]
pub struct SwarmNodeDisplay {
    pub id: String,
    pub url: String,
    pub model: String,
    pub elo: i32,
    pub rounds: u64,
    pub wins: u64,
}

/// Summary of the last swarm round for display.
#[derive(Debug, Clone)]
pub struct SwarmRoundDisplay {
    pub winner: String,
    pub confidence: f32,
    pub n_nodes: usize,
    pub total_time_ms: u64,
}

pub struct App {
    pub tab: Tab,
    pub running: bool,

    // Unified identity
    pub node_id: String, // iroh NodeId (hex-encoded Ed25519 pubkey)

    // Chat state
    pub chat_input: String,
    pub chat_messages: Vec<ChatMessage>,
    pub chat_loading: bool,
    pub chat_scroll: u16,

    // Dashboard state
    pub node_running: bool,
    pub elo_score: i32,
    pub smti_balance: f64,
    pub smti_earned_today: f64,
    pub inferences_served: u64,
    pub uptime_secs: u64,
    pub peers_connected: u32,
    pub current_model: String,
    pub elo_history: Vec<u64>,

    // Models state
    pub models: Vec<ModelInfo>,
    pub selected_model_idx: usize,

    // Wallet state — REAL Solana
    pub wallet: Option<crate::wallet::SolanaWallet>,
    pub wallet_pubkey: String,
    pub wallet_short: String,
    pub sol_balance: f64,         // real SOL balance from devnet
    pub pending_rewards: f64,
    pub wallet_status: String,    // status messages (airdrop, errors)
    pub tx_history: Vec<TxEntry>,

    // Download / node state
    pub download_progress: Option<f64>, // None = not downloading, Some(0-100) = progress
    pub download_status: String,        // "Downloading Qwen2.5-3B... 45%"
    pub node_error: String,             // error messages from node start

    // Swarm state
    pub swarm_nodes: Vec<SwarmNodeDisplay>,
    pub last_round_result: Option<SwarmRoundDisplay>,

    // Domain demand (what queries the network is getting)
    pub demand: crate::swarm::DemandStats,
    pub demand_updated: bool,

    // Pending Solana settlement
    pub pending_settlement: Option<crate::settlement::RoundPayload>,

    // Remote node input (Models tab: press 'r' to add friend's node)
    pub adding_remote_node: bool,
    pub remote_url_input: String,

    // Settings
    pub api_endpoint: String,
    pub max_vram_gb: f64,
    pub selected_setting: usize,
    pub editing_setting: bool,
    pub setting_input: String,
}

impl App {
    pub fn new() -> Self {
        Self {
            tab: Tab::Chat,
            running: true,

            node_id: String::new(),

            chat_input: String::new(),
            chat_messages: vec![
                ChatMessage {
                    role: "assistant".into(),
                    content: "Welcome to Samhati! I'm running on the decentralized mesh network. Ask me anything.".into(),
                    timestamp: Local::now().format("%H:%M").to_string(),
                    confidence: Some(0.99),
                    n_nodes: Some(3),
                },
            ],
            chat_loading: false,
            chat_scroll: 0,

            node_running: true,
            elo_score: 1547,
            smti_balance: 42.583,
            smti_earned_today: 3.21,
            inferences_served: 1_284,
            uptime_secs: 14_832,
            peers_connected: 7,
            current_model: "Qwen2.5-3B".into(),
            elo_history: vec![1480, 1495, 1502, 1510, 1498, 1520, 1535, 1542, 1538, 1547],

            models: detect_models(),
            selected_model_idx: 0,

            wallet: None,
            wallet_pubkey: "loading...".into(),
            wallet_short: "...".into(),
            sol_balance: 0.0,
            pending_rewards: 0.0,
            wallet_status: String::new(),
            tx_history: vec![], // populated from real Solana devnet

            download_progress: None,
            download_status: String::new(),
            node_error: String::new(),

            swarm_nodes: Vec::new(),
            last_round_result: None,
            demand: crate::swarm::DemandStats::load(),
            demand_updated: false,
            pending_settlement: None,

            adding_remote_node: false,
            remote_url_input: String::new(),

            api_endpoint: "http://localhost:8000".into(),
            max_vram_gb: 24.0,
            selected_setting: 0,
            editing_setting: false,
            setting_input: String::new(),
        }
    }

    pub fn format_uptime(&self) -> String {
        let h = self.uptime_secs / 3600;
        let m = (self.uptime_secs % 3600) / 60;
        format!("{}h {}m", h, m)
    }
}

/// Auto-detect available RAM and recommend models that fit this device.
/// Follows the whitepaper's cascade: smaller models for low-end devices,
/// specialist models get domain bonuses (1.5x SMTI).
fn detect_models() -> Vec<ModelInfo> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total_ram_gb = sys.total_memory() as f64 / 1_073_741_824.0;

    let mut models = vec![
        // ── Tiny — runs on anything ─────────────────────────────────
        m("Qwen2.5-0.5B",           "0.5B", "General",   0.4, 1.0, "1.0x"),
        m("Qwen2.5-Coder-0.5B",     "0.5B", "Code",      0.4, 1.0, "1.5x"),
        m("Gemma-3-1B",              "1B",   "General",   0.7, 2.0, "1.0x"),
        m("SmolLM2-1.7B",            "1.7B", "General",   1.0, 2.0, "1.0x"),

        // ── Small — laptops ─────────────────────────────────────────
        m("Qwen2.5-1.5B",            "1.5B", "General",   1.0, 2.0, "1.0x"),
        m("Qwen2.5-Coder-1.5B",      "1.5B", "Code",      1.0, 2.0, "1.5x"),
        m("Qwen2.5-Math-1.5B",       "1.5B", "Math",      1.0, 2.0, "1.5x"),
        m("Qwen2.5-3B",              "3B",   "General",   1.8, 4.0, "1.0x"),
        m("Qwen2.5-Coder-3B",        "3B",   "Code",      1.8, 4.0, "1.5x"),
        m("Llama-3.2-3B",            "3B",   "General",   1.8, 4.0, "1.0x"),
        m("Phi-4-mini-3.8B",         "3.8B", "Reasoning", 2.2, 4.0, "1.5x"),
        m("Gemma-3-4B",              "4B",   "General",   2.5, 6.0, "1.0x"),

        // ── Medium — 16GB+ laptops ──────────────────────────────────
        m("Qwen2.5-7B",              "7B",   "General",   4.4, 8.0, "1.0x"),
        m("Qwen2.5-Coder-7B",        "7B",   "Code",      4.4, 8.0, "1.5x"),
        m("Qwen2.5-Math-7B",         "7B",   "Math",      4.4, 8.0, "1.5x"),
        m("DeepSeek-Coder-V2-Lite",  "7B",   "Code",      4.4, 8.0, "1.5x"),
        m("Llama-3.1-8B",            "8B",   "General",   4.7, 8.0, "1.0x"),
        m("Mistral-7B-v0.3",         "7B",   "General",   4.1, 8.0, "1.0x"),

        // ── Large — 32GB+ machines ──────────────────────────────────
        m("Qwen2.5-14B",             "14B",  "General",   8.7, 16.0, "1.3x"),
        m("Qwen2.5-Coder-14B",       "14B",  "Code",      8.7, 16.0, "1.5x"),
        m("DeepSeek-R1-Distill-14B",  "14B",  "Reasoning", 8.7, 16.0, "1.8x"),
    ];

    // Auto-detect: mark models that fit in 70% of RAM (leave room for OS)
    let usable_ram = total_ram_gb * 0.7;
    for model in &mut models {
        if (model.min_ram_gb as f64) <= usable_ram {
            model.recommended = true;
        }
    }

    // Auto-select: find the best 3B general model as default active
    for model in &mut models {
        if model.name == "Qwen2.5-3B" && model.recommended {
            model.active = true;
            model.installed = true;
        }
    }

    // Also mark largest fitting model as installed
    let mut best_idx = None;
    let mut best_size: f32 = 0.0;
    for (i, model) in models.iter().enumerate() {
        if model.recommended && model.size_gb > best_size && !model.active {
            best_size = model.size_gb;
            best_idx = Some(i);
        }
    }
    if let Some(idx) = best_idx {
        models[idx].installed = true;
    }

    models
}

fn m(name: &str, params: &str, domain: &str, size_gb: f32, min_ram_gb: f32, bonus: &str) -> ModelInfo {
    ModelInfo {
        name: name.into(), params: params.into(), quant: "Q4_K_M".into(),
        domain: domain.into(), size_gb, min_ram_gb,
        smti_bonus: bonus.into(), installed: false, active: false, recommended: false,
    }
}
