use chrono::Local;

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
    pub smti_bonus: String,
    pub installed: bool,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct TxEntry {
    pub timestamp: String,
    pub tx_type: String,
    pub amount: f64,
    pub status: String,
}

pub struct App {
    pub tab: Tab,
    pub running: bool,

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

    // Wallet state
    pub wallet_pubkey: String,
    pub pending_rewards: f64,
    pub tx_history: Vec<TxEntry>,

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
            current_model: "LLaMA-3.3-70B".into(),
            elo_history: vec![1480, 1495, 1502, 1510, 1498, 1520, 1535, 1542, 1538, 1547],

            models: vec![
                ModelInfo {
                    name: "LLaMA-3.3-70B".into(),
                    domain: "General".into(),
                    size_gb: 38.5,
                    smti_bonus: "1.5x".into(),
                    installed: true,
                    active: true,
                },
                ModelInfo {
                    name: "Mistral-7B-v0.3".into(),
                    domain: "General".into(),
                    size_gb: 4.1,
                    smti_bonus: "1.0x".into(),
                    installed: true,
                    active: false,
                },
                ModelInfo {
                    name: "DeepSeek-R1-Distill-32B".into(),
                    domain: "Reasoning".into(),
                    size_gb: 18.2,
                    smti_bonus: "1.8x".into(),
                    installed: false,
                    active: false,
                },
                ModelInfo {
                    name: "Qwen-2.5-Coder-14B".into(),
                    domain: "Code".into(),
                    size_gb: 8.7,
                    smti_bonus: "1.3x".into(),
                    installed: false,
                    active: false,
                },
                ModelInfo {
                    name: "BioMistral-7B".into(),
                    domain: "Medical".into(),
                    size_gb: 4.1,
                    smti_bonus: "2.0x".into(),
                    installed: false,
                    active: false,
                },
            ],
            selected_model_idx: 0,

            wallet_pubkey: "smti1q7x8...k4f2m9".into(),
            pending_rewards: 8.74,
            tx_history: vec![
                TxEntry {
                    timestamp: "2026-03-23 14:22".into(),
                    tx_type: "reward".into(),
                    amount: 0.42,
                    status: "confirmed".into(),
                },
                TxEntry {
                    timestamp: "2026-03-23 11:05".into(),
                    tx_type: "reward".into(),
                    amount: 0.38,
                    status: "confirmed".into(),
                },
                TxEntry {
                    timestamp: "2026-03-22 22:47".into(),
                    tx_type: "claim".into(),
                    amount: 5.10,
                    status: "confirmed".into(),
                },
                TxEntry {
                    timestamp: "2026-03-22 09:30".into(),
                    tx_type: "stake".into(),
                    amount: -10.00,
                    status: "confirmed".into(),
                },
                TxEntry {
                    timestamp: "2026-03-21 18:12".into(),
                    tx_type: "reward".into(),
                    amount: 1.23,
                    status: "confirmed".into(),
                },
            ],

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
