#!/usr/bin/env python3
"""Initialize the Samhati protocol on Solana devnet.

Sets your identity key as the coordinator authority.
Only needs to be run ONCE per program deployment.

Usage: python3 scripts/init_protocol.py
"""

import json, hashlib, struct, os, sys
import urllib.request
from pathlib import Path

RPC = "https://api.devnet.solana.com"
PROGRAM_ID = "AB7cSMLv1J7J28DKLMbzo2tyNp1kZSmE67a6Heoa5Mkr"

# ── Helpers ──────────────────────────────────────────────────────────

ALPHABET = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, 'big')
    r = b''
    while n > 0:
        n, rem = divmod(n, 58)
        r = ALPHABET[rem:rem+1] + r
    for b in data:
        if b == 0: r = ALPHABET[0:1] + r
        else: break
    return r.decode()

def b58decode(s: str) -> bytes:
    n = 0
    for c in s.encode():
        n = n * 58 + ALPHABET.index(c)
    return n.to_bytes(32, 'big')

def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(RPC, body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def find_pda(seeds: list, program_id: bytes) -> tuple:
    for bump in range(255, -1, -1):
        h = hashlib.sha256()
        for seed in seeds:
            h.update(seed)
        h.update(bytes([bump]))
        h.update(program_id)
        h.update(b"ProgramDerivedAddress")
        candidate = h.digest()
        # Check if off Ed25519 curve (simplified: try to create a point)
        try:
            from nacl.bindings import crypto_sign_ed25519_pk_to_curve25519
            crypto_sign_ed25519_pk_to_curve25519(candidate)
            continue  # on curve, not valid PDA
        except:
            return candidate, bump
    # Fallback without nacl: just use first bump
    h = hashlib.sha256()
    for seed in seeds:
        h.update(seed)
    h.update(bytes([255]))
    h.update(program_id)
    h.update(b"ProgramDerivedAddress")
    return h.digest(), 255

# ── Main ─────────────────────────────────────────────────────────────

def main():
    # Load identity
    identity_path = Path.home() / ".samhati" / "identity.json"
    if not identity_path.exists():
        print("ERROR: No identity found. Run the TUI first to generate one.")
        sys.exit(1)

    with open(identity_path) as f:
        key_bytes = json.load(f)

    secret = bytes(key_bytes[:32])
    public = bytes(key_bytes[32:64])
    pubkey = b58encode(public)
    program_id = b58decode(PROGRAM_ID)

    print(f"Authority: {pubkey}")
    print(f"Program:   {PROGRAM_ID}")

    # Get correct PDA from solana CLI
    import subprocess
    pda_result = subprocess.run(
        ["solana", "find-program-derived-address", PROGRAM_ID, "string:config"],
        capture_output=True, text=True
    )
    config_addr = pda_result.stdout.strip()
    config_pda = b58decode(config_addr)
    print(f"Config PDA: {config_addr}")

    resp = rpc("getAccountInfo", [config_addr, {"encoding": "base64"}])
    if resp.get("result", {}).get("value") is not None:
        print("Protocol already initialized!")
        return

    print("\nInitializing protocol...")

    # Use solana CLI to send the transaction (handles serialization correctly)
    # Build instruction data: anchor discriminator + base_emission_per_round (u64)
    disc = hashlib.sha256(b"global:initialize").digest()[:8]
    base_emission = 1000  # 1000 SMTI per round
    ix_data = disc + struct.pack("<Q", base_emission)
    ix_data_b58 = b58encode(ix_data)

    # For smti_mint and reward_vault, use placeholder addresses (our own pubkey for now)
    # These will be updated when the real SMTI token is minted
    smti_mint = pubkey  # placeholder
    reward_vault = pubkey  # placeholder
    system_program = "11111111111111111111111111111111"

    # Write instruction data to temp file for solana CLI
    import base64, tempfile, subprocess

    ix_data_base64 = base64.b64encode(ix_data).decode()

    # Use solana program invoke via raw transaction
    # Actually, let's use anchor CLI if available
    print(f"Instruction data (hex): {ix_data.hex()}")
    print(f"Config PDA: {config_addr}")
    print(f"SMTI mint (placeholder): {smti_mint}")
    print(f"Reward vault (placeholder): {reward_vault}")

    # Try using solana CLI to send raw instruction
    # solana program invoke doesn't exist, so we'll use a direct RPC call

    # Build the transaction manually using ed25519
    from nacl.signing import SigningKey as NaClSigningKey

    signing_key = NaClSigningKey(secret)

    # Get recent blockhash
    bh_resp = rpc("getLatestBlockhash", [{"commitment": "finalized"}])
    blockhash = b58decode(bh_resp["result"]["value"]["blockhash"])

    # Build message
    # Accounts: authority(signer,writable), config(writable), smti_mint(readonly), reward_vault(readonly), system_program(readonly)
    accounts = [
        public,                          # 0: authority (signer, writable)
        config_pda,                      # 1: config PDA (writable)
        public,                          # 2: smti_mint placeholder (readonly)
        public,                          # 3: reward_vault placeholder (readonly)
        b58decode(system_program),       # 4: system program (readonly)
        program_id,                      # 5: program id
    ]

    # Message header
    msg = bytearray()
    msg.append(1)   # num_required_signatures
    msg.append(0)   # num_readonly_signed_accounts
    msg.append(2)   # num_readonly_unsigned_accounts (system_program + program_id)

    # Account keys (unique, in order: signer writable, writable, readonly, programs)
    # Deduplicate: public appears 3 times (authority, smti_mint, reward_vault)
    unique_keys = []
    key_indices = {}

    # Signer+writable first
    unique_keys.append(public)
    key_indices[public] = 0

    # Writable (config PDA)
    unique_keys.append(config_pda)
    key_indices[bytes(config_pda)] = 1

    # Readonly unsigned (system_program, program_id)
    sys_prog = b58decode(system_program)
    unique_keys.append(sys_prog)
    key_indices[bytes(sys_prog)] = 2

    unique_keys.append(program_id)
    key_indices[bytes(program_id)] = 3

    # Compact array of keys
    msg.append(len(unique_keys))
    for k in unique_keys:
        msg.extend(k)

    # Recent blockhash
    msg.extend(blockhash)

    # Instructions (1 instruction)
    msg.append(1)  # num instructions

    # Program ID index
    msg.append(3)  # program_id is at index 3

    # Account indices: [authority=0, config=1, smti_mint=0, reward_vault=0, system_program=2]
    msg.append(5)  # 5 accounts
    msg.append(0)  # authority
    msg.append(1)  # config
    msg.append(0)  # smti_mint (same as authority for placeholder)
    msg.append(0)  # reward_vault (same as authority for placeholder)
    msg.append(2)  # system_program

    # Instruction data
    if len(ix_data) < 128:
        msg.append(len(ix_data))
    else:
        msg.append((len(ix_data) & 0x7F) | 0x80)
        msg.append(len(ix_data) >> 7)
    msg.extend(ix_data)

    # Sign
    signature = signing_key.sign(bytes(msg)).signature

    # Full transaction
    tx = bytearray()
    tx.append(1)  # num signatures
    tx.extend(signature)
    tx.extend(msg)

    # Send
    tx_b64 = base64.b64encode(bytes(tx)).decode()
    send_resp = rpc("sendTransaction", [tx_b64, {"encoding": "base64", "skipPreflight": True}])

    if "result" in send_resp:
        print(f"\nSuccess! Tx: {send_resp['result']}")
        print("Protocol initialized with your key as coordinator authority.")
    else:
        err = send_resp.get("error", {}).get("message", "unknown")
        print(f"\nFailed: {err}")
        if "logs" in str(send_resp):
            logs = send_resp.get("error", {}).get("data", {}).get("logs", [])
            for log in logs:
                print(f"  {log}")

if __name__ == "__main__":
    main()
