"""
Offline checks for src/agent_orchestrator.py - no AWS account or model calls needed.

What is real and what is simulated:
  - DynamoDB: moto, with the tables from infrastructure/starter_stack.yaml and the records from
    infrastructure/seed_data.py. All tool code runs against it unchanged.
  - Bedrock (guardrail), bedrock-runtime (apply_guardrail) and AgentCore Memory: botocore Stubbers,
    so every request is validated against the real API model; responses are minimal valid ones.
  - The AgentCore CLI helper and apply_observability_config(): recorded, not executed.
  - The LLMs: a scripted stand-in per agent decides which tools to call. Everything behind the tool
    calls (DynamoDB, WorkflowState, parallel retrieval, dedup) is the project code. The routing
    decisions of Claude itself can only be checked live (python src/agent_orchestrator.py test).

Run from the project folder:
    .venv/Scripts/python.exe offline_checks/test_project_offline.py
"""

import io
import json
import os
import re
import sys
import threading
import time
from contextlib import redirect_stdout
from datetime import date, datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT, os.path.join(ROOT, "infrastructure")]

# Placeholder credentials so that nothing can ever reach a real AWS account
os.environ.update({"AWS_ACCESS_KEY_ID": "offline-test", "AWS_SECRET_ACCESS_KEY": "offline-test",
                   "AWS_DEFAULT_REGION": "us-east-1", "AWS_REGION": "us-east-1",
                   "AGENT_TRACING_ENABLED": "false", "AGENT_LOG_TO_CLOUDWATCH": "false"})
for var in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "GUARDRAIL_ID", "GUARDRAIL_VERSION",
            "RETURNS_KB_ID", "SHIPPING_KB_ID", "WARRANTY_KB_ID", "AGENTCORE_RUNTIME_ARN"):
    os.environ.pop(var, None)

from moto import mock_aws  # noqa: E402

mock = mock_aws()
mock.start()

import boto3  # noqa: E402
import botocore.session  # noqa: E402
from botocore.stub import Stubber  # noqa: E402

import config  # noqa: E402

PROJECT = config.PROJECT_NAME
config._exports_cache = {           # what the stack exports (config reads these instead of AWS)
    f"{PROJECT}-OrdersTable": f"{PROJECT}-orders",
    f"{PROJECT}-CustomersTable": f"{PROJECT}-customers",
    f"{PROJECT}-WorkflowStateTable": f"{PROJECT}-workflow-state",
    f"{PROJECT}-AgentCoreRoleArn": "arn:aws:iam::123456789012:role/udacity-agentcore-role",
    f"{PROJECT}-AgentLogGroup": "/aws/bedrock/agentcore/udacity-agentcore",
}
config.ACCOUNT_ID = "123456789012"

# Tables exactly as starter_stack.yaml defines them, then the project's own seed data
ddb = boto3.client("dynamodb", region_name="us-east-1")
for name, keys in ((f"{PROJECT}-orders", [("customer_id", "HASH"), ("order_id", "RANGE")]),
                   (f"{PROJECT}-customers", [("customer_id", "HASH")]),
                   (f"{PROJECT}-workflow-state", [("session_id", "HASH")])):
    ddb.create_table(TableName=name, BillingMode="PAY_PER_REQUEST",
                     AttributeDefinitions=[{"AttributeName": k, "AttributeType": "S"} for k, _ in keys],
                     KeySchema=[{"AttributeName": k, "KeyType": t} for k, t in keys])
import seed_data  # noqa: E402
with redirect_stdout(io.StringIO()):
    seed_data.seed_customers()
    seed_data.seed_orders()

import agent_utils  # noqa: E402
agent_utils._real_stdout = io.StringIO()   # silence the terminal trace UI during the checks
import agent_orchestrator as ao  # noqa: E402

REAL_CLIENT = boto3.client
MODELS = botocore.session.get_session()
checks = []


def ok(label):
    checks.append(label)
    print(f"  PASS  {label}")


def text(tool_result):
    return tool_result["content"][0]["text"]


def data(tool_result):
    """Strands serializes a tool's dict return value as JSON text."""
    return json.loads(text(tool_result))


def minimal(shape):
    """Smallest response that satisfies an output shape (required members only)."""
    kind = shape.type_name
    if kind == "structure":
        return {name: minimal(shape.members[name]) for name in shape.required_members}
    if kind == "list":
        return []
    if kind == "map":
        return {}
    if kind == "string":
        return shape.enum[0] if shape.enum else "x" * max(1, shape.metadata.get("min", 1))
    if kind in ("integer", "long"):
        return max(1, shape.metadata.get("min", 1))
    if kind in ("float", "double"):
        return 1.0
    if kind == "boolean":
        return False
    if kind == "timestamp":
        return datetime.now(timezone.utc)
    return b"x"


class Stubbed:
    """A real botocore client behind a Stubber: requests are validated against the API model and
    recorded; respond(op, params) supplies each response (validated too)."""

    def __init__(self, service, respond):
        self.client = REAL_CLIENT(service, region_name="us-east-1")
        self.model = MODELS.get_service_model(service)
        self.stub = Stubber(self.client)
        self.stub.activate()
        self.respond, self.calls = respond, []

    def output(self, op):
        return minimal(self.model.operation_model(op).output_shape)

    def __getattr__(self, attr):
        op = self.client.meta.method_to_api_mapping[attr]

        def call(**params):
            self.calls.append((attr, params))
            self.stub.add_response(attr, self.respond(self, op, params), params)
            return getattr(self.client, attr)(**params)
        return call


# ---- Scripted stand-ins for the LLMs ------------------------------------------------------------
RealAgent = ao.Agent
created = []


def _id(pattern, prompt):
    match = re.search(pattern, prompt)
    return match.group(1) if match else None


def orchestrator_script(agent, prompt):
    sid, cid = _id(r"\[Session ID: (\S+)\]", prompt), _id(r"\[Customer ID: (\S+)\]", prompt)
    request = prompt.split("] ", 2)[-1]
    low = request.lower()
    agent.tool.initialize_session(session_id=sid, customer_id=cid)                       # rule 1
    if re.search(r"\$\d", request) and ("%" in request or "discount" in low):
        pass                                                                              # rule 5
    elif re.search(r"ORD-\d+", request) or "refund" in low:                              # rule 2
        agent.tool.route_to_inventory_agent(session_id=sid, customer_id=cid, request=request)
        agent.tool.route_to_refund_agent(session_id=sid, customer_id=cid, request=request)
    elif any(word in low for word in ("policy", "warranty", "shipping")):                # rule 3
        agent.tool.route_to_policy_agent(session_id=sid, request=request)
    elif "tier" in low:                                                                   # rule 4
        agent.tool.route_to_inventory_agent(session_id=sid, customer_id=cid, request=request)
    return text(agent.tool.route_to_communication_agent(                                  # rule 6
        session_id=sid, customer_id=cid, original_request=request))


def inventory_script(agent, prompt):
    cid, order_id = _id(r"Customer ID: (\S+)", prompt), _id(r"(ORD-\d+)", prompt)
    facts = [text(agent.tool.get_customer_tier(customer_id=cid))]
    facts.append(text(agent.tool.check_order_status(customer_id=cid, order_id=order_id)) if order_id
                 else text(agent.tool.list_customer_orders(customer_id=cid)))
    return "FACTS\n" + "\n".join(facts)


def refund_script(agent, prompt):
    sid, cid = _id(r"Session ID: (\S+)", prompt), _id(r"Customer ID: (\S+)", prompt)
    context = text(agent.tool.get_inventory_context(session_id=sid))
    order_id = _id(r"(ORD-\d+)", context)
    if not order_id:
        return "DECISION: NEEDS INFO"
    return "DECISION\n" + text(agent.tool.initiate_refund(customer_id=cid, order_id=order_id,
                                                          reason="Customer asked to return it"))


def policy_script(agent, prompt):
    return "POLICY ANSWER from:\n" + text(agent.tool.search_all_policies(query=prompt))


def communication_script(agent, prompt):
    context = text(agent.tool.get_full_workflow_context(session_id=_id(r"Session ID: (\S+)", prompt)))
    reference = _id(r"(RET-[0-9A-F]{8})", context)
    return "Hi! Here is what we found." + (f" Your return reference is {reference}." if reference else "")


def retriever_script(agent, prompt):
    (name,) = agent.tool_names
    return text(getattr(agent.tool, name)(query=prompt))


SCRIPTS = {"OrchestratorAgent": orchestrator_script, "InventoryAgent": inventory_script,
           "RefundAgent": refund_script, "PolicyAgent": policy_script,
           "CommunicationAgent": communication_script}


class ScriptedAgent(RealAgent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("callback_handler", None)
        super().__init__(*args, **kwargs)
        created.append(self)

    def __call__(self, prompt, **kwargs):
        return SCRIPTS.get(self.name, retriever_script)(self, prompt)


ao.Agent = ScriptedAgent

# ---- Fake Knowledge Bases (the tiers document is indexed in all three, as seed_data.py does) ----
TIERS = "PREMIUM TIER: extended 60-day return window, free expedited shipping, 3-year electronics warranty."
PASSAGES = {
    "KB-RETURNS":  [("Customers may return most items within 30 days of delivery. Premium: 60 days.", 0.81,
                     "returns/return_policy.txt"), (TIERS, 0.62, "returns/customer_tiers.txt")],
    "KB-SHIPPING": [("Premium tier customers receive free expedited shipping on all orders.", 0.55,
                     "shipping/shipping_policy.txt"), (TIERS, 0.58, "shipping/customer_tiers.txt")],
    "KB-WARRANTY": [("Premium tier customers receive an automatic 3-year warranty on all Electronics.", 0.49,
                     "warranty/warranty_policy.txt"), (TIERS, 0.66, "warranty/customer_tiers.txt")],
}
active, peak, kb_lock, broken_kb = [0], [0], threading.Lock(), set()


def fake_retrieve(kb_id, query, top_k=3):
    with kb_lock:
        active[0] += 1
        peak[0] = max(peak[0], active[0])
    try:
        time.sleep(0.2)                               # network latency, to observe the parallelism
        if kb_id in broken_kb:
            return [{"text": "Knowledge base retrieval failed: ThrottlingException", "source": "error", "score": 0.0}]
        return [{"text": t, "source": f"s3://policy-bucket/policies/{src}", "score": s}
                for t, s, src in PASSAGES.get(kb_id, [])][:top_k]
    finally:
        with kb_lock:
            active[0] -= 1


ao.retrieve_from_knowledge_base = fake_retrieve
os.environ.update({"RETURNS_KB_ID": "KB-RETURNS", "SHIPPING_KB_ID": "KB-SHIPPING", "WARRANTY_KB_ID": "KB-WARRANTY"})

print("\nAgents, tools and WorkflowState")

# 1) Models, temperatures, names and tools match the design
inventory, refund, policy, communication = (ao.build_inventory_agent(), ao.build_refund_agent(),
                                            ao.build_policy_agent(), ao.build_communication_agent())
orchestrator = ao.build_orchestrator_agent(inventory, refund, policy, communication)
by_name = {a.name: a for a in created}
expected = {"InventoryAgent": (config.WORKER_MODEL_ID, 0.1), "RefundAgent": (config.WORKER_MODEL_ID, 0.1),
            "ReturnsPolicyRetrieverAgent": (config.WORKER_MODEL_ID, 0.0),
            "ShippingPolicyRetrieverAgent": (config.WORKER_MODEL_ID, 0.0),
            "WarrantyPolicyRetrieverAgent": (config.WORKER_MODEL_ID, 0.0),
            "PolicyAgent": (config.WORKER_MODEL_ID, 0.2), "CommunicationAgent": (config.WORKER_MODEL_ID, 0.3),
            "OrchestratorAgent": (config.ORCHESTRATOR_MODEL_ID, 0.0)}
for name, (model_id, temperature) in expected.items():
    cfg = by_name[name].model.config
    assert (cfg["model_id"], cfg["temperature"]) == (model_id, temperature), (name, cfg)
assert inventory.tool_names == ["check_order_status", "get_customer_tier", "list_customer_orders"]
assert refund.tool_names == ["get_inventory_context", "initiate_refund"]
assert policy.tool_names == ["search_all_policies"] and communication.tool_names == ["get_full_workflow_context"]
assert orchestrator.tool_names == ["initialize_session", "route_to_inventory_agent", "route_to_policy_agent",
                                   "route_to_refund_agent", "route_to_communication_agent"]
assert type(orchestrator.tool_executor).__name__ == "SequentialToolExecutor"
ok("8 agents (5 + 3 retrievers): models, temperatures, names and tool sets as designed")

# 2) Every tool documents its purpose, parameters and return value
for agent in created:
    for spec in agent.tool_registry.get_all_tools_config().values():
        assert "Returns" in spec["description"] or "Returns" in (spec.get("description") or ""), spec["name"]
        for prop, schema in spec["inputSchema"]["json"]["properties"].items():
            assert not schema.get("description", "").startswith("Parameter "), (spec["name"], prop)
ok("every tool has a docstring with its parameters and return value")

# 3) The orchestrator prompt carries all six routing rules
prompt = orchestrator.system_prompt
for phrase in ("call initialize_session", "route_to_inventory_agent first, then\n   route_to_refund_agent",
               "call route_to_policy_agent", "Never send\n   them to route_to_policy_agent",
               "Math or calculation questions: do not call the inventory, policy or refund agents",
               "as the very last step: call route_to_communication_agent", "never write the customer-facing reply"):
    assert phrase in prompt, phrase
ok("orchestrator system prompt enforces rules 1-6 and forbids writing the reply itself")

# 4) InventoryAgent tools against the seeded tables
order = data(inventory.tool.check_order_status(customer_id="CUST-001", order_id="ORD-27176"))
facts = order["order"]
assert order["found"] and facts["days_since_order"] == 12 and facts["days_since_delivery"] == 7
assert facts["checked_on"] == date.today().isoformat() and facts["product_name"] == "Wireless Headphones Pro"
assert "No order ORD-27176 found for customer CUST-002" in text(
    inventory.tool.check_order_status(customer_id="CUST-002", order_id="ORD-27176"))
customer = data(inventory.tool.get_customer_tier(customer_id="CUST-001"))["customer"]
assert customer["tier"] == "Premium" and "email" not in customer and customer["total_orders"] == 47  # Decimal -> int
listing = data(inventory.tool.list_customer_orders(customer_id="CUST-001"))
assert [o["order_id"] for o in listing["orders"]] == ["ORD-27177", "ORD-27176", "ORD-27180"] and listing["order_count"] == 3
ok("InventoryAgent: order facts with day counts, other customers' orders invisible, tier without email, newest-first list")

# 5) RefundAgent: the model decides, initiate_refund enforces the tier windows in code
def attempt(customer_id, order_id):
    return data(refund.tool.initiate_refund(customer_id=customer_id, order_id=order_id, reason="test"))


orders_table = boto3.resource("dynamodb", region_name="us-east-1").Table(f"{PROJECT}-orders")
first = attempt("CUST-001", "ORD-27176")
record = orders_table.get_item(Key={"customer_id": "CUST-001", "order_id": "ORD-27176"})["Item"]
assert first["success"] and first["refund_amount"] == "149.99" and first["return_reference"].startswith("RET-")
assert record["status"] == "return_initiated" and record["return_reference"] == first["return_reference"]
again = attempt("CUST-001", "ORD-27176")["reason"]                                   # no double return
assert "already in progress" in again and first["return_reference"] in again
assert "Outside the 60-day Premium return window (90 days" in attempt("CUST-001", "ORD-27180")["reason"]
assert "Outside the 30-day Standard return window (40 days" in attempt("CUST-002", "ORD-28001")["reason"]
assert attempt("CUST-002", "ORD-28002")["refund_amount"] == "179.98"                 # 2 x 89.99
assert attempt("CUST-003", "ORD-29001")["success"]                                   # 35 days: Premium only
assert "only delivered orders" in attempt("CUST-001", "ORD-27177")["reason"]         # shipped
assert "only delivered orders" in attempt("CUST-003", "ORD-29003")["reason"]         # cancelled
assert "No order ORD-30001 found for customer CUST-001" in attempt("CUST-001", "ORD-30001")["reason"]
ok("RefundAgent: Premium 60 / Standard 30 days from delivery, delivered-only, no double returns, DynamoDB updated")

# 6) PolicyAgent: three retriever sub-agents in parallel, merged + deduplicated + ranked
t0 = time.time()
combined = text(policy.tool.search_all_policies(query="What is the return window for premium customers?"))
elapsed = time.time() - t0
assert peak[0] == 3 and elapsed < 0.55, (peak[0], elapsed)
assert combined.count("PREMIUM TIER") == 1 and "(4 unique" in combined                # tiers doc deduplicated
scores = [float(s) for s in re.findall(r"score (\d\.\d+)", combined)]
assert scores == sorted(scores, reverse=True) and scores[1] == 0.66                   # best copy kept
assert not agent_utils._TraceWriter._suppress_parallel.is_set()                      # stdout restored
broken_kb.add("KB-SHIPPING")
partial = text(policy.tool.search_all_policies(query="shipping options"))
broken_kb.clear()
assert "[Shipping KB] retrieval failed" in partial and "Returns KB |" in partial and "Warranty KB |" in partial
ok(f"PolicyAgent: 3 retrievers ran concurrently ({elapsed:.2f}s for 3 x 0.2s), duplicate tiers passage "
   "merged, ranked by score, one failing KB reported without losing the others")

# 7) End-to-end through the orchestrator's routing tools and WorkflowState
def run(customer_id, message, session_id):
    return orchestrator(f"[Session ID: {session_id}] [Customer ID: {customer_id}] {message}")


state_table = boto3.resource("dynamodb", region_name="us-east-1").Table(f"{PROJECT}-workflow-state")
reply = run("CUST-004", "I want to return my phone case from order ORD-30001", "sess-ret")   # 15 days, Standard
state = state_table.get_item(Key={"session_id": "sess-ret"})["Item"]
assert int(state["version"]) == 3 and "policy_agent" not in state                      # init + 3 workers
assert '"success": true' in state["refund_agent"] and "RET-" in reply
reply = run("CUST-003", "How much are 5 items at $29.99 with 10% off?", "sess-math")
state = state_table.get_item(Key={"session_id": "sess-math"})["Item"]
assert [c for c in ao.AGENT_COLUMNS if c in state] == ["communication_agent"] and int(state["version"]) == 1
reply = run("CUST-004", "What is the return policy for premium customers?", "sess-ret")      # follow-up
state = state_table.get_item(Key={"session_id": "sess-ret"})["Item"]
assert [c for c in ao.AGENT_COLUMNS if c in state] == ["policy_agent", "communication_agent"]
assert int(state["version"]) == 6                                                      # reset + 2 workers
ok("routing tools: read -> run worker -> locked write; return/math/policy flows leave the expected "
   "columns; a follow-up in the same session starts with clean columns")

print("\nDeployment functions (request shapes validated against the API models)")

# 8) create_guardrail(): policies, STANDARD topic tier, numbered version, behaviour checks
def bedrock_responses(fake, op, params):
    resp = fake.output(op)
    if op == "ListGuardrails":
        return {"guardrails": []}
    if op == "CreateGuardrail":
        resp.update(guardrailId="gr0offline1", version="DRAFT")
    if op == "CreateGuardrailVersion":
        resp.update(guardrailId="gr0offline1", version="1")
    if op == "GetGuardrail":
        resp.update(status="READY", version="1")
    return resp


def runtime_responses(fake, op, params):
    blocked = any(w in params["content"][0]["text"]["text"] for w in ("advertised price", "Best Buy", "sue"))
    return {"usage": {k: 0 for k in ("topicPolicyUnits", "contentPolicyUnits", "wordPolicyUnits",
                                    "sensitiveInformationPolicyUnits", "sensitiveInformationPolicyFreeUnits",
                                    "contextualGroundingPolicyUnits")},
            "action": "GUARDRAIL_INTERVENED" if blocked else "NONE",
            "outputs": [{"text": "blocked"}] if blocked else [], "assessments": []}


bedrock = Stubbed("bedrock", bedrock_responses)
ao.bedrock_runtime = Stubbed("bedrock-runtime", runtime_responses)
boto3.client = lambda service, *a, **k: bedrock if service == "bedrock" else REAL_CLIENT(service, *a, **k)
try:
    out = io.StringIO()
    with redirect_stdout(out):
        guardrail = ao.create_guardrail()
finally:
    boto3.client = REAL_CLIENT
assert guardrail == ("gr0offline1", "1")
request = dict(bedrock.calls)["create_guardrail"]
filters = {f["type"]: (f["inputStrength"], f["outputStrength"]) for f in request["contentPolicyConfig"]["filtersConfig"]}
assert filters == {"SEXUAL": ("HIGH", "HIGH"), "VIOLENCE": ("HIGH", "HIGH"), "HATE": ("HIGH", "HIGH"),
                   "INSULTS": ("MEDIUM", "MEDIUM"), "MISCONDUCT": ("MEDIUM", "MEDIUM")}
pii = {p["type"]: p["action"] for p in request["sensitiveInformationPolicyConfig"]["piiEntitiesConfig"]}
assert pii == {"CREDIT_DEBIT_CARD_NUMBER": "BLOCK", "US_SOCIAL_SECURITY_NUMBER": "BLOCK",
               "EMAIL": "ANONYMIZE", "PHONE": "ANONYMIZE"}
topics = request["topicPolicyConfig"]["topicsConfig"]
topic_shape = MODELS.get_service_model("bedrock").operation_model("CreateGuardrail").input_shape.members[
    "topicPolicyConfig"].members["topicsConfig"].member
assert [t["type"] for t in topics] == ["DENY"] * 3 and request["topicPolicyConfig"]["tierConfig"] == {"tierName": "STANDARD"}
for topic in topics:                                   # string limits botocore itself doesn't check
    assert re.fullmatch(topic_shape.members["name"].metadata["pattern"], topic["name"])
    assert len(topic["definition"]) <= topic_shape.members["definition"].metadata["max"]
    assert len(topic["examples"]) <= 5 and all(len(e) <= 100 for e in topic["examples"])
joined = " ".join((t["name"] + " " + t["definition"]).lower() for t in topics)
assert all(word in joined for word in ("competitor", "pricing", "legal"))
assert request["crossRegionConfig"] == {"guardrailProfileIdentifier": "us.guardrail.v1:0"}
assert request["wordPolicyConfig"] == {"managedWordListsConfig": [{"type": "PROFANITY"}]}
assert "[FAIL]" not in out.getvalue() and out.getvalue().count("[OK  ]") == 5
ok("create_guardrail(): content/PII/topic/word policies, STANDARD tier + us.guardrail.v1:0, "
   "numbered version 1, math allowed and negotiation/competitor/legal blocked")

# 9) configure_memory(): SESSION_SUMMARY strategy, 7-day expiry, name the test looks for
def memory_responses(fake, op, params):
    if op == "ListMemories":
        return {"memories": []}
    resp = fake.output(op)
    resp["memory"] = minimal(fake.model.operation_model(op).output_shape.members["memory"])  # optional member
    resp["memory"].update(id=f"{config.MEMORY_NAME}-AbCdEf1234", status="ACTIVE",
                          arn=f"arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/{config.MEMORY_NAME}-AbCdEf1234")
    return resp


ao.agentcore_control = Stubbed("bedrock-agentcore-control", memory_responses)
with redirect_stdout(io.StringIO()):
    memory_arn = ao.configure_memory("arn:runtime")
request = dict(ao.agentcore_control.calls)["create_memory"]
strategy = request["memoryStrategies"][0]["summaryMemoryStrategy"]
assert request["name"] == config.MEMORY_NAME and request["eventExpiryDuration"] == 7 and request["description"]
assert strategy["namespaceTemplates"] == ["/summaries/{actorId}/{sessionId}"] and memory_arn.endswith("AbCdEf1234")
ok("configure_memory(): summaryMemoryStrategy, eventExpiryDuration=7, named and described, returns the ARN")

# 10) deploy_to_agentcore_runtime(): AgentCore CLI calls and the runtime environment
import agentcore_cli  # noqa: E402

recorded, arns = {}, iter([None, "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/udacity_agentcore_runtime-X1"])
agentcore_cli.stack_name = lambda: "AgentCore-udacity-default"
agentcore_cli.deployed_runtime_arn = lambda: next(arns)
agentcore_cli.stage_runtime_code = lambda: recorded.setdefault("staged", True)
agentcore_cli.configure_runtime = lambda **kwargs: recorded.setdefault("configure", kwargs)
agentcore_cli.deploy = lambda verbose=False: recorded.setdefault("deployed", True)
ao.wait_for_runtime_ready = lambda control, runtime_id, timeout=300: "READY"
with redirect_stdout(io.StringIO()):
    runtime_arn = ao.deploy_to_agentcore_runtime(orchestrator, "gr0offline1", "1")
cfg = recorded["configure"]
assert runtime_arn.endswith("runtime/udacity_agentcore_runtime-X1") and recorded["staged"] and recorded["deployed"]
assert (cfg["network_mode"], cfg["protocol"], cfg["execution_role_arn"]) == (
    "PUBLIC", "HTTP", "arn:aws:iam::123456789012:role/udacity-agentcore-role")
assert cfg["env_vars"] == {"AWS_REGION": "us-east-1", "PROJECT_NAME": PROJECT, "RETURNS_KB_ID": "KB-RETURNS",
                           "SHIPPING_KB_ID": "KB-SHIPPING", "WARRANTY_KB_ID": "KB-WARRANTY",
                           "AGENT_LOG_GROUP": "/aws/bedrock/agentcore/udacity-agentcore",
                           "GUARDRAIL_ID": "gr0offline1", "GUARDRAIL_VERSION": "1"}
ok("deploy_to_agentcore_runtime(): configure_runtime(PUBLIC, HTTP, stack role, 8 env vars) -> deploy -> ARN")

# 11) configure_observability(): the loggingConfiguration, and the failure path
from agent_observability import validate_logging_configuration  # noqa: E402

seen = {}
ao.apply_observability_config = lambda arn, cfg: seen.update(cfg) or {"log_group": cfg["cloudWatchConfig"]["logGroupName"]}
out = io.StringIO()
with redirect_stdout(out):
    ao.configure_observability(runtime_arn)
validate_logging_configuration(seen)
assert seen == {"cloudWatchConfig": {"logGroupName": "/aws/bedrock/agentcore/udacity-agentcore", "logLevel": "INFO",
                                     "enabled": True}, "xRayConfig": {"enabled": True, "samplingRate": 1.0}}
assert "100% sampling" in out.getvalue()


def failing(arn, cfg):
    raise RuntimeError("AccessDenied on xray:UpdateTraceSegmentDestination")


ao.apply_observability_config = failing
out = io.StringIO()
with redirect_stdout(out):
    ao.configure_observability(runtime_arn)
assert "[Note] Observability configuration failed: AccessDenied" in out.getvalue()
ok("configure_observability(): CloudWatch INFO + X-Ray 100%, passed to apply_observability_config inside try/except")

# 12) The guardrail reaches every agent's model once GUARDRAIL_ID / VERSION are known
os.environ.update({"GUARDRAIL_ID": "gr0offline1", "GUARDRAIL_VERSION": "1"})
created.clear()
graph = ao.build_agent_graph()
guarded = {a.name: (a.model.config.get("guardrail_id"), a.model.config.get("guardrail_version")) for a in created
           if a.name in SCRIPTS}
assert set(guarded.values()) == {("gr0offline1", "1")} and len(guarded) == 5
ok("build_agent_graph(): guardrail gr0offline1 v1 attached to all 5 agents' models")

mock.stop()
print(f"\nAll {len(checks)} offline project checks passed.")
