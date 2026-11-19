# Sourced by the setup/serve scripts: read the root .env the way the Python apps do (python-dotenv), so a
# model set there is the one that gets pulled or downloaded. A variable already set in the shell wins,
# so `NLI_MODEL=... ./setup.sh` still overrides. Values become shell variables, not exported: a script
# exports only what its own child process needs (keys never reach pip or ollama).
_env_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
_env_re='^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
_env_quoted='^"(.*)"$|^'"'"'(.*)'"'"'$'
if [ -f "$_env_file" ]; then
  while IFS= read -r _env_line || [ -n "$_env_line" ]; do
    [[ $_env_line =~ $_env_re ]] || continue
    _env_key=${BASH_REMATCH[2]}
    _env_val=${BASH_REMATCH[3]%$'\r'}
    if [[ $_env_val =~ $_env_quoted ]]; then _env_val=${BASH_REMATCH[1]}${BASH_REMATCH[2]}; fi
    [ -n "${!_env_key+set}" ] || printf -v "$_env_key" '%s' "$_env_val"
  done < "$_env_file"
fi
unset _env_file _env_re _env_quoted _env_line _env_key _env_val
