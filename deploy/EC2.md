# Deploy on EC2 (manual steps)

Everything runs on one instance: Ollama on the host, Langfuse and the agent in Docker.

## 1. Launch the instance
- AMI: Ubuntu 22.04/24.04. Size: at least 16 GB RAM (`t3.xlarge`), 50 GB disk. For speed use a GPU instance (`g5.xlarge`) with NVIDIA drivers.
- Security group inbound, **your IP only**: 22 (SSH), 3000 (Langfuse UI), 8000 (agent API).
- Do **not** open 11434 (Ollama), 5432, 6379, 8123, 9000 or 9090 to the internet.

## 2. Install Docker and Ollama
```bash
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER   # log out/in after
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl edit ollama   # add: [Service] Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
ollama pull llama3.1:8b
```
Containers reach Ollama through the Docker bridge, so Ollama must listen on 0.0.0.0. The security group is what keeps it private.

## 3. Get the code and configure
```bash
git clone <your-repo-url> agent-cost-eval && cd agent-cost-eval
cp .env.example .env
```
Edit `.env`: set `LANGFUSE_PUBLIC_URL=http://<ec2-public-ip>:3000`. Leave the two Langfuse keys empty for now.
In `docker-compose.yml`, replace every `# CHANGEME` secret, and set `NEXTAUTH_URL` to the same public URL.

## 4. Start Langfuse and create keys
```bash
docker compose up -d
```
Open `http://<ec2-public-ip>:3000`, sign up, create an organization and project, then **Settings > API Keys > Create**. Put the keys in `.env` as `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.

## 5. Register the token price (once)
```bash
LANGFUSE_HOST=http://localhost:3000 uv run python -m agent_cost_eval.pricing
```
Ollama is free, so without a registered price Langfuse shows tokens but no dollar cost. The prices in `.env` are placeholders: set `INPUT_PRICE_PER_TOKEN` and `OUTPUT_PRICE_PER_TOKEN` to what you want to model (for example a hosted model's price) before this step.

## 6. Start the agent
```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.agent.yml up -d --build
curl localhost:8000/health
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"Why was I charged twice this month?","user_id":"user-alice","session_id":"s1"}'
```
The response includes `trace_url`. To fill the dashboard with data run the 5 sample questions:
```bash
LANGFUSE_HOST=http://localhost:3000 uv run python -m agent_cost_eval.run
```

## 7. Check it worked
Follow "Reading token cost in Langfuse" in the README. If the step after `get_invoices` shows few input tokens, raise `NUM_CTX` in `.env` and restart the agent.

## Troubleshooting
- **Costs show as $0 or empty:** the price isn't registered (step 5), or the model name in `.env` differs from the registered one.
- **`/ask` returns 502:** the run failed. The `detail` field has the error, and the trace shows where.
- **Agent can't reach Ollama:** check `OLLAMA_HOST` listens on 0.0.0.0 (step 2) and `curl http://localhost:11434/api/tags` works on the host.
