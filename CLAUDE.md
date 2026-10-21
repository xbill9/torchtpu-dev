# CLAUDE.md — torchtpu-dev

Monorepo of TorchTPU Gemma 4 serving rigs. Each `tpu-pytorch-*` directory is a self-contained rig with its own `CLAUDE.md`; read that one before changing anything inside it, and run its `make` targets from inside it.

- **Rig names** follow `~/gemma4-dev/NAMING.md`. The directory name is also the plugin name, the MCP server name, and the stem of the skill name (`<rig>-management`), so a rename means sweeping the old name out of the whole rig (`grep -rn <old> <rig>`) and updating `.claude-plugin/marketplace.json` here.
- **Forks are copies, not links.** A fix in one rig's `server.py` does not reach its siblings; apply it to each and run `make skill` in each.
- **`colab/`** is outside the naming scheme: serving code only, copied from the rigs.
- **Root `.claude-plugin/marketplace.json`** lists every rig. Validate with `claude plugin validate .` here and inside each rig.
