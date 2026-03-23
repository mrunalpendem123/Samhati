/// A 32-byte node identifier, matching the swarm peer identity.
pub type NodeId = [u8; 32];

/// Helper to create a NodeId from a single byte (useful in tests).
pub fn node_id_from_byte(b: u8) -> NodeId {
    let mut id = [0u8; 32];
    id[0] = b;
    id
}

/// Display a NodeId as a hex-encoded short prefix.
pub fn node_id_short(id: &NodeId) -> String {
    format!("{:02x}{:02x}..{:02x}{:02x}", id[0], id[1], id[30], id[31])
}
