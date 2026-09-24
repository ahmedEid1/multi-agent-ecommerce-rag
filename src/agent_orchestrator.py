"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # guardrail + runtime + memory + observability
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)

Deployment uses the AgentCore CLI (`agentcore`, npm package @aws/agentcore,
https://github.com/aws/agentcore-cli) through the helper src/agentcore_cli.py -
see the README for the prerequisites (Node.js 20+, uv).
"""

import boto3
import json
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from decimal import Decimal

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from strands.tools.executors import SequentialToolExecutor
from boto3.dynamodb.conditions import Key

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_writer, _real_stdout, _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_runtime   = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client  = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb          = boto3.resource('dynamodb', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> dict | None:
    """Read the current WorkflowState for a session."""
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """Update WorkflowState with optimistic locking; on a version conflict, re-read and retry."""
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                ) from None
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# The four worker columns of a WorkflowState record
AGENT_COLUMNS = ('inventory_agent', 'policy_agent', 'refund_agent', 'communication_agent')


def _reset_agent_columns(session_id: str, max_retries: int = 3) -> int:
    """
    Start a new request in an existing session (a follow-up message in `chat`):
    remove the previous request's worker columns so the CommunicationAgent only
    sees findings for the current request. Same optimistic locking as
    _update_workflow_state(). Returns the new version.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    for attempt in range(max_retries):
        version = int((_read_workflow_state(session_id) or {}).get('version', 0))
        try:
            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression='REMOVE ' + ', '.join(AGENT_COLUMNS) + ' SET version = :new_version',
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues={':new_version': version + 1, ':expected_version': version},
            )
            return version + 1
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"WorkflowState reset failed after {max_retries} retries (session: {session_id})")


# ─────────────────────────────────────────────────────
#  HELPERS SHARED BY THE AGENT TOOLS
# ─────────────────────────────────────────────────────

# Return windows by customer tier (days, counted from delivery as in the Returns policy)
STANDARD_RETURN_WINDOW_DAYS = 30
PREMIUM_RETURN_WINDOW_DAYS  = 60


def _plain(value):
    """DynamoDB returns numbers as Decimal; convert them so tool results serialize cleanly."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _days_since(iso_date: str) -> int | None:
    """Whole days from a YYYY-MM-DD date to today (None if the date is missing or invalid)."""
    try:
        return (date.today() - date.fromisoformat(iso_date)).days
    except (TypeError, ValueError):
        return None


def _order_facts(item: dict) -> dict:
    """An order record plus the date arithmetic the RefundAgent needs. These are facts, not
    decisions, and computing them in code keeps date math out of the LLM."""
    order = _plain(item)
    order['checked_on'] = date.today().isoformat()
    order['days_since_order'] = _days_since(order.get('order_date', ''))
    if order.get('status') == 'delivered':
        # The table records the estimated delivery date; returns count from delivery
        order['days_since_delivery'] = _days_since(order.get('estimated_delivery', ''))
    return order


def _return_window_days(tier: str) -> int:
    return PREMIUM_RETURN_WINDOW_DAYS if str(tier).lower() == 'premium' else STANDARD_RETURN_WINDOW_DAYS


def _ask(agent: Agent, prompt: str) -> str:
    """Run a worker agent on a fresh conversation. Workers share context through WorkflowState,
    not chat history, so one request (or customer) never leaks into the next."""
    agent.messages.clear()
    return str(agent(prompt)).strip()


# ═══════════════════════════════════════════════════════
#  AGENTS
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  INVENTORY AGENT - order and customer facts (DynamoDB)
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    model = BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.1)

    system_prompt = """You are NovaMart's InventoryAgent. You gather facts from the order and customer
records. You never decide anything.

- For a specific order, call check_order_status(customer_id, order_id).
- For any order, return or account question, also call get_customer_tier(customer_id): the
  RefundAgent needs the tier.
- If no order ID is given, or the customer asks about their orders, call list_customer_orders(customer_id).

Report the facts exactly as the tools return them: order ID, product, status, order date, estimated
delivery date, days since order / delivery, price, quantity, and the customer's name and tier.
Do NOT say whether a return is eligible or approved, do NOT quote policy, and never invent data.
If a record is not found, say so plainly (an order is only visible to the customer who placed it)."""

    # NOTE: the Orders table has a COMPOSITE key (customer_id = partition key,
    # order_id = sort key), so a get_item needs BOTH values. That is why this
    # tool takes customer_id as well as order_id.
    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up one order in DynamoDB and report its status, product, dates
        and amount. Reports facts only - it does NOT decide return eligibility.

        Args:
            customer_id: The customer's unique identifier (e.g. CUST-001)
            order_id: The order identifier (e.g. ORD-27176)

        Returns:
            Order record (order_id, status, product_name, order_date, price, ...)
            or a not-found message
        """
        item = dynamodb.Table(config.ORDERS_TABLE).get_item(
            Key={'customer_id': customer_id, 'order_id': order_id}).get('Item')
        if not item:
            return {'found': False, 'message': f"No order {order_id} found for customer {customer_id}."}
        return {'found': True, 'order': _order_facts(item)}

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        item = dynamodb.Table(config.CUSTOMERS_TABLE).get_item(Key={'customer_id': customer_id}).get('Item')
        if not item:
            return {'found': False, 'message': f"No customer {customer_id} found."}
        customer = _plain(item)
        customer.pop('email', None)  # not needed by any agent: keep contact PII out of the conversation
        return {'found': True, 'customer': customer}

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        query = {'KeyConditionExpression': Key('customer_id').eq(customer_id)}
        items = []
        while True:
            page = table.query(**query)
            items.extend(page.get('Items', []))
            if 'LastEvaluatedKey' not in page:
                break
            query['ExclusiveStartKey'] = page['LastEvaluatedKey']
        orders = sorted((_order_facts(item) for item in items),
                        key=lambda order: order.get('order_date', ''), reverse=True)
        return {'customer_id': customer_id, 'order_count': len(orders), 'orders': orders}

    return Agent(name='InventoryAgent', model=model, system_prompt=system_prompt,
                 tools=[check_order_status, get_customer_tier, list_customer_orders])


# ───────────────────────────────────────────────────────
#  REFUND AGENT - return decisions, enforced in code
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    model = BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.1)

    system_prompt = f"""You are NovaMart's RefundAgent. You decide whether an order can be returned and, when
the customer asked for a return or refund and it is allowed, you initiate it.

Decision process, in this order:
1. Always call get_inventory_context(session_id) first and base the decision only on those facts
   (order status, delivery date, days since delivery, customer tier). If they are missing, answer
   NEEDS INFO and do not guess.
2. Apply the return window for the customer's tier: Standard = {STANDARD_RETURN_WINDOW_DAYS} days,
   Premium = {PREMIUM_RETURN_WINDOW_DAYS} days, counted from delivery (days_since_delivery; use
   days_since_order only when no delivery date is known).
3. Only delivered orders can be returned: processing or shipped orders are not returnable yet, and
   cancelled or already-returned orders cannot be returned.
4. If the order is eligible and the customer asked to return or refund it, call
   initiate_refund(customer_id, order_id, reason) exactly once; it re-checks the rules and may refuse.
   If the customer only asked about the order's status, initiate nothing and say until when a return
   would be possible.

Answer in this format:
DECISION: APPROVED | DENIED | NOT REQUESTED | NEEDS INFO
RULE: the rule applied, with the numbers (e.g. "Premium window 60 days; delivered 7 days ago")
RETURN REFERENCE: the reference from initiate_refund, or none
NEXT STEPS: what the customer should do next"""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        inventory = (_read_workflow_state(session_id) or {}).get('inventory_agent')
        return {'session_id': session_id, 'inventory_agent': inventory} if inventory else {}

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        orders = dynamodb.Table(config.ORDERS_TABLE)
        item = orders.get_item(Key={'customer_id': customer_id, 'order_id': order_id}).get('Item')
        if not item:
            return {'success': False, 'reason': f"No order {order_id} found for customer {customer_id}."}
        order = _order_facts(item)
        customer = dynamodb.Table(config.CUSTOMERS_TABLE).get_item(
            Key={'customer_id': customer_id}).get('Item') or {}
        tier = customer.get('tier', 'Standard')
        window = _return_window_days(tier)
        days = order.get('days_since_delivery', order.get('days_since_order'))

        # The model decides, the code enforces: nothing outside the rules is ever written
        if order.get('status') == 'return_initiated':
            return {'success': False,
                    'reason': f"A return is already in progress for order {order_id} "
                              f"(reference {order.get('return_reference')})."}
        if order.get('status') != 'delivered':
            return {'success': False,
                    'reason': f"Order {order_id} is '{order.get('status')}' - only delivered orders can be returned."}
        if days is None or days > window:
            return {'success': False,
                    'reason': f"Outside the {window}-day {tier} return window ({days} days since delivery)."}

        reference = f"RET-{uuid.uuid4().hex[:8].upper()}"
        try:
            orders.update_item(
                Key={'customer_id': customer_id, 'order_id': order_id},
                UpdateExpression=('SET #status = :returning, return_reference = :ref, '
                                  'return_reason = :reason, return_initiated_at = :at'),
                ConditionExpression='#status = :delivered',   # never return the same order twice
                ExpressionAttributeNames={'#status': 'status'},  # "status" is a DynamoDB reserved word
                ExpressionAttributeValues={
                    ':returning': 'return_initiated', ':delivered': 'delivered', ':ref': reference,
                    ':reason': reason, ':at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                },
            )
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            return {'success': False,
                    'reason': f"Order {order_id} is no longer 'delivered' - a return may already be in progress."}

        amount = Decimal(str(order.get('price', '0'))) * Decimal(str(order.get('quantity', '1')))
        return {
            'success': True,
            'return_reference': reference,
            'order_id': order_id,
            'product_name': order.get('product_name'),
            'refund_amount': f"{amount:.2f}",
            'customer_tier': tier,
            'return_window_days': window,
            'days_since_delivery': days,
            'instructions': [
                "A prepaid return shipping label will be emailed to you.",
                "Pack the item securely in its original packaging with all accessories.",
                "Drop it off at any authorized carrier location.",
                "The refund goes to the original payment method within 5-7 business days of the return arriving.",
            ],
        }

    return Agent(name='RefundAgent', model=model, system_prompt=system_prompt,
                 tools=[get_inventory_context, initiate_refund])


# ───────────────────────────────────────────────────────
#  POLICY AGENT - multi-agent RAG over three knowledge bases
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    # Passages the retriever tools found during the current search_all_policies call, per domain.
    # search_all_policies merges them, so duplicates across KBs are removed (customer_tiers.txt,
    # for example, is indexed in all three).
    found    = {}
    failures = {}
    found_lock = threading.Lock()

    def _retrieve(domain: str, kb_id: str, query: str) -> str:
        passages = retrieve_from_knowledge_base(kb_id, query)
        with found_lock:
            for passage in passages:
                if passage.get('source') == 'error':
                    failures[domain] = passage['text']
                else:
                    found.setdefault(domain, []).append(passage)
        return format_kb_results(passages)

    retriever_prompt = """You are NovaMart's {name}. Call {tool} exactly once with the question unchanged,
then return the passages it found word for word, with their scores and sources. Do not answer the
question, summarise or add anything. If the tool finds nothing, reply "No relevant {domain} policy
passages found."."""

    def _retriever_agent(name: str, domain: str, retrieve_tool) -> Agent:
        # Retrieval is deterministic: temperature 0.0. callback_handler=None keeps the three threads from
        # streaming into stdout at once (the chat proxy only suppresses them in chat mode); their passages
        # are printed in order by trace.kb_result() after the parallel run.
        return Agent(name=name,
                     model=BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION,
                                        temperature=0.0),
                     system_prompt=retriever_prompt.format(name=name, domain=domain,
                                                           tool=retrieve_tool.tool_name),
                     tools=[retrieve_tool],
                     callback_handler=None)

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base.

        Args:
            query: The customer's policy question

        Returns:
            The most relevant passages with relevance scores and source documents
        """
        return _retrieve('Returns', config.RETURNS_KB_ID, query)

    returns_retriever = _retriever_agent('ReturnsPolicyRetrieverAgent', 'returns', retrieve_returns_policy)

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base.

        Args:
            query: The customer's policy question

        Returns:
            The most relevant passages with relevance scores and source documents
        """
        return _retrieve('Shipping', config.SHIPPING_KB_ID, query)

    shipping_retriever = _retriever_agent('ShippingPolicyRetrieverAgent', 'shipping', retrieve_shipping_policy)

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base.

        Args:
            query: The customer's policy question

        Returns:
            The most relevant passages with relevance scores and source documents
        """
        return _retrieve('Warranty', config.WARRANTY_KB_ID, query)

    warranty_retriever = _retriever_agent('WarrantyPolicyRetrieverAgent', 'warranty', retrieve_warranty_policy)

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }
        with found_lock:
            found.clear()
            failures.clear()

        # Trace: the parallel KB dispatch (kb_start also suppresses stdout for every thread,
        # including Strands' own streaming threads, until kb_done)
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """Run one retriever sub-agent; returns (domain, result_text), printed later by kb_result()."""
            try:
                return domain, _ask(agent, query)
            except Exception as exc:          # one failing KB must not sink the other two
                logger.warning("%s retriever failed: %s", domain, exc)
                with found_lock:
                    failures[domain] = str(exc)
                return domain, f"[{domain} retriever error: {exc}]"

        # Run all three retrievers in parallel
        results = {}
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(_run_retriever, domain, agent, query)
                           for domain, agent in retrievers.items()]
                for future in as_completed(futures):
                    domain, text = future.result()
                    results[domain] = text
        finally:
            # All KBs responded: this also lifts the stdout suppression set by kb_start
            trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        # Merge -> deduplicate (keep the best-scoring copy) -> rank by relevance
        best = {}
        for domain, passages in found.items():
            for passage in passages:
                key = ' '.join(passage['text'].split()).lower()
                if key not in best or passage['score'] > best[key]['score']:
                    best[key] = {**passage, 'domain': domain}
        ranked = sorted(best.values(), key=lambda passage: passage['score'], reverse=True)

        lines = [f"Policy passages from {len(retrievers)} knowledge bases searched in parallel "
                 f"({len(ranked)} unique, most relevant first):"]
        for i, passage in enumerate(ranked, 1):
            lines.append(f"\n[{i}] {passage['domain']} KB | score {passage['score']:.3f} | "
                         f"{passage['source']}\n{passage['text'].strip()}")
        for domain in retrievers:
            if domain in failures:
                lines.append(f"\n[{domain} KB] retrieval failed: {failures[domain]}")
            elif domain not in found:
                lines.append(f"\n[{domain} KB] no relevant passages")
        return "\n".join(lines)

    model = BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.2)

    system_prompt = """You are NovaMart's PolicyAgent. You answer questions about NovaMart's return,
shipping and warranty policies.

1. Always call search_all_policies(query) first, once, with the customer's question. It searches the
   Returns, Shipping and Warranty knowledge bases in parallel.
2. Answer only from the passages it returns and name the source document for each fact
   (e.g. return_policy.txt). Give exact numbers: days, prices, years.
3. If passages disagree (for example two different warranty lengths), say so and quote both instead
   of picking one.
4. If the passages don't answer the question, say that the policy documents don't cover it. Never
   fill gaps from general knowledge.
5. You explain policy text only. You cannot see customer accounts or orders, so never state which
   tier a particular customer has."""

    return Agent(name='PolicyAgent', model=model, system_prompt=system_prompt, tools=[search_all_policies])


# ───────────────────────────────────────────────────────
#  COMMUNICATION AGENT - the customer-facing reply
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    # Temperature 0.3: a warm, natural tone
    model = BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.3)

    system_prompt = """You are NovaMart's CommunicationAgent. You write the final message the customer reads.

1. Always call get_full_workflow_context(session_id) first. It holds everything the specialist agents
   found for this request: inventory_agent (order and account facts), policy_agent (policy answer),
   refund_agent (return decision and reference). Columns that are missing were not needed.
2. Include every relevant finding and nothing else: never invent order details, dates, amounts,
   return references or policy terms. If a return was approved, give the return reference and the
   next steps; if it was denied, explain the rule kindly and offer an alternative (a warranty claim,
   or contacting support).
3. Calculation requests (no specialist findings): work it out step by step. Calculate the exact
   total first and round currency only at the end (half up).
4. Tone: warm, empathetic, professional and concise. Address the customer by name when it is known.
   Plain text: no agent names, tool names, session IDs or JSON.
5. Never include card numbers, SSNs or other sensitive personal data."""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        state = _read_workflow_state(session_id)
        if not state:
            return {'found': False, 'message': f"No WorkflowState found for session {session_id}."}
        state = _plain(state)
        state.pop('ttl', None)
        return state

    return Agent(name='CommunicationAgent', model=model, system_prompt=system_prompt,
                 tools=[get_full_workflow_context])


# ───────────────────────────────────────────────────────
#  ORCHESTRATOR AGENT - routing + WorkflowState
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    # Temperature 0.0: deterministic routing
    model = BedrockModel(model_id=config.ORCHESTRATOR_MODEL_ID, region_name=config.AWS_REGION, temperature=0.0)

    system_prompt = """You are NovaMart's OrchestratorAgent. You route each customer request to specialist agents.
You never answer the customer yourself.

Every message starts with "[Session ID: <id>] [Customer ID: <id>]". Pass exactly those values to the tools.

Routing rules:
1. Every request, always: call initialize_session(session_id, customer_id) first.
2. Order status, return or refund requests: call route_to_inventory_agent first, then
   route_to_refund_agent (the RefundAgent decides using the inventory facts).
3. Questions about what a policy means (return windows, shipping rates or times, warranty terms):
   call route_to_policy_agent.
4. Account questions ("what is my tier?", "am I premium?"): call route_to_inventory_agent. Never send
   them to route_to_policy_agent - it only knows policy text, not customer data.
5. Math or calculation questions: do not call the inventory, policy or refund agents; go straight to rule 6.
6. Every request, always, as the very last step: call route_to_communication_agent(session_id,
   customer_id, original_request) with the customer's original message, even if a specialist failed.

A request can match several rules (for example a return plus a policy question): call each matching
specialist once, in rule order, one tool at a time.
CRITICAL: never write the customer-facing reply yourself. After route_to_communication_agent returns,
reply with exactly its text, unchanged."""

    # Each routing tool follows the same pattern:
    #   1. read the current WorkflowState  (_read_workflow_state)
    #   2. invoke the worker agent
    #   3. write its result back with optimistic locking
    #      (_update_workflow_state(session_id, {'<column>': text}, expected_version))
    # The terminal trace UI shows each step: trace.step_start() before the worker runs and
    # trace.step_done() after the write.
    def _route(session_id: str, column: str, worker: Agent, prompt: str) -> str:
        state = _read_workflow_state(session_id)
        if state is None:
            return f"No WorkflowState for session {session_id} - call initialize_session first."
        version = int(state['version'])
        trace.step_start(column)
        trace.agent_section(_AGENT_META[column][1])
        result = _ask(worker, prompt)
        _update_workflow_state(session_id, {column: result}, expected_version=version)
        trace.step_done(column, version)
        return result

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        return _route(session_id, 'inventory_agent', inventory_agent,
                      f"Customer ID: {customer_id}\nCustomer request: {request}\n"
                      f"Look up the order and account facts this request needs.")

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        return _route(session_id, 'policy_agent', policy_agent, request)

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        return _route(session_id, 'refund_agent', refund_agent,
                      f"Session ID: {session_id}\nCustomer ID: {customer_id}\nCustomer request: {request}")

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        return _route(session_id, 'communication_agent', communication_agent,
                      f"Session ID: {session_id}\nCustomer ID: {customer_id}\n"
                      f"Customer's message: {original_request}\nWrite the reply to the customer.")

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            _create_workflow_state(session_id, customer_id)
            return f"WorkflowState created for session {session_id} (customer {customer_id}, version 0)."
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            # A follow-up message in the same chat session: start this request with clean columns
            version = _reset_agent_columns(session_id)
            return (f"Session {session_id} already exists; previous findings cleared for this request "
                    f"(WorkflowState version {version}).")

    # SequentialToolExecutor: even if the model asks for several tools in one turn they run in
    # order, so the RefundAgent never reads WorkflowState before the InventoryAgent has written it
    return Agent(name='OrchestratorAgent', model=model, system_prompt=system_prompt,
                 tools=[initialize_session, route_to_inventory_agent, route_to_policy_agent,
                        route_to_refund_agent, route_to_communication_agent],
                 tool_executor=SequentialToolExecutor())


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT TOOLING - AgentCore CLI
#
#  The runtime is deployed with the AgentCore CLI (`agentcore`, npm package
#  @aws/agentcore - https://github.com/aws/agentcore-cli) through the
#  helper src/agentcore_cli.py:
#
#    agentcore_cli.stage_runtime_code()   copies this file, its helper modules
#                                         and config.py to build/runtime/ with a
#                                         pyproject.toml of the runtime deps
#    agentcore_cli.configure_runtime()    writes the runtime settings (network
#                                         mode, protocol, execution role, env
#                                         vars) to agentcore/agentcore.json
#    agentcore_cli.deploy()               runs `agentcore deploy -y`: the CLI
#                                         downloads arm64 / Python 3.12 wheels
#                                         with uv, zips them with the code
#                                         (direct code deployment) and creates
#                                         or updates the runtime via CDK
#    agentcore_cli.deployed_runtime_arn() reads the ARN the CLI recorded
#
#  Inside the runtime this same file is the entry point: it is started with
#  no command-line argument and serves HTTP (see run_serve). The marker file
#  written next to it by stage_runtime_code() tells __main__ to do so.
# ═══════════════════════════════════════════════════════

_RUNTIME_MARKER = '.agentcore-runtime'         # written by agentcore_cli.stage_runtime_code()
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
#  GUARDRAIL + RUNTIME DEPLOYMENT
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            if guardrail_version == 'DRAFT':    # created earlier but never published
                guardrail_version = bedrock_client.create_guardrail_version(
                    guardrailIdentifier=guardrail_id, description='NovaMart production policy')['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    # One DENY topic per config.GUARDRAIL_BLOCKED_TOPICS entry (definitions are at most 200 characters,
    # examples at most 5 x 100). "Pricing negotiations" is defined narrowly so that arithmetic with a
    # price and discount the customer already states (the project's math scenario) stays allowed.
    topic_details = {
        'competitor products': (
            "Questions about, comparisons with or recommendations of other retailers' products, prices "
            "or services, such as Amazon, Walmart or Best Buy.",
            ["Is Amazon selling these headphones cheaper than you?",
             "Should I buy this laptop from Best Buy instead?",
             "How does Walmart's return policy compare to yours?"]),
        'pricing negotiations': (
            "Haggling, or asking NovaMart to lower or change an advertised price or to price match. "
            "Calculating a total from a price and discount the customer states is not negotiation.",
            ["Can you knock $20 off the advertised price?",
             "Will you match the lower price I found elsewhere?",
             "I'll only order if you give me 30% off."]),
        'legal threats': (
            "Threats of legal action against NovaMart, such as suing, involving a lawyer or taking "
            "the company to court.",
            ["I'm going to sue NovaMart if I don't get a refund.",
             "My lawyer will be contacting you about this order.",
             "I'll take you to small claims court."]),
    }
    topics_config = [
        {'name': topic.title(), 'definition': topic_details[topic][0],
         'examples': topic_details[topic][1], 'type': 'DENY'}
        for topic in config.GUARDRAIL_BLOCKED_TOPICS
    ]

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description='NovaMart customer support: harmful content, PII, off-topic subjects and profanity',
        contentPolicyConfig={'filtersConfig': [
            {'type': 'SEXUAL',     'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
            {'type': 'VIOLENCE',   'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
            {'type': 'HATE',       'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
            {'type': 'INSULTS',    'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
            {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
        ]},
        sensitiveInformationPolicyConfig={'piiEntitiesConfig': [
            {'type': 'CREDIT_DEBIT_CARD_NUMBER',  'action': 'BLOCK'},
            {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
            {'type': 'EMAIL',                     'action': 'ANONYMIZE'},
            {'type': 'PHONE',                     'action': 'ANONYMIZE'},
        ]},
        topicPolicyConfig={'topicsConfig': topics_config, 'tierConfig': {'tierName': 'STANDARD'}},
        crossRegionConfig={'guardrailProfileIdentifier': 'us.guardrail.v1:0'},  # required by the STANDARD tier
        wordPolicyConfig={'managedWordListsConfig': [{'type': 'PROFANITY'}]},
        blockedInputMessaging=("Sorry, I can't help with that request. I'm happy to help with your NovaMart "
                               "orders, returns, shipping or warranty questions."),
        blockedOutputsMessaging=("Sorry, I can't share that response. Please contact NovaMart support if you "
                                 "need more help."),
    )
    guardrail_id = response['guardrailId']
    print(f"  Guardrail created: {guardrail_id}")

    # Publish an immutable numbered version: the runtime must never run on the editable DRAFT
    guardrail_version = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id, description='NovaMart production policy')['version']
    for _ in range(30):
        status = bedrock_client.get_guardrail(guardrailIdentifier=guardrail_id,
                                              guardrailVersion=guardrail_version)['status']
        if status == 'READY':
            break
        time.sleep(2)
    print(f"  Guardrail version {guardrail_version} published ({status})")

    verify_guardrail(guardrail_id, guardrail_version)
    return guardrail_id, guardrail_version


# Requests the project requires the guardrail to allow or block: (source, text, expected)
GUARDRAIL_CHECKS = [
    ('INPUT',  "How much are 5 items at $29.99 with 10% off?",                          'ALLOWED'),
    ('OUTPUT', "5 x $29.99 = $149.95. With 10% off ($14.995) the total is $134.96.",   'ALLOWED'),
    ('INPUT',  "Can you lower the advertised price of the headphones by $20 for me?",   'BLOCKED'),
    ('INPUT',  "Is Best Buy selling this keyboard cheaper than NovaMart?",              'BLOCKED'),
    ('INPUT',  "If I don't get my money back I will sue NovaMart.",                      'BLOCKED'),
]


def verify_guardrail(guardrail_id: str, guardrail_version: str) -> bool:
    """Run GUARDRAIL_CHECKS through bedrock-runtime.apply_guardrail() and print the results.
    The math scenario (question and answer) must pass; negotiation, competitor and legal-threat
    examples must be blocked. Prints a warning instead of failing the deployment."""
    ok = True
    print("  Guardrail checks:")
    for source, text, expected in GUARDRAIL_CHECKS:
        try:
            response = bedrock_runtime.apply_guardrail(
                guardrailIdentifier=guardrail_id, guardrailVersion=guardrail_version,
                source=source, content=[{'text': {'text': text}}])
            actual = 'BLOCKED' if response.get('action') == 'GUARDRAIL_INTERVENED' else 'ALLOWED'
        except Exception as exc:
            actual = f"ERROR ({exc.__class__.__name__})"
        ok = ok and actual == expected
        mark = 'OK  ' if actual == expected else 'FAIL'
        print(f"    [{mark}] {source:<6} expected {expected:<7} got {actual:<7} {text}")
    if not ok:
        print("  WARNING: the guardrail does not behave as specified - adjust the topic definitions.")
    return ok


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime with
    the AgentCore CLI (src/agentcore_cli.py wraps it).

    AgentCore does not serialize Python objects, so `orchestrator_agent` is
    not uploaded directly. Instead the staging step copies this
    file, which doubles as the HTTP entry point (see run_serve), together
    with its helper modules and config.py to build/runtime/. `agentcore
    deploy` then packages that directory with arm64 dependencies and creates
    or updates the runtime ("direct code deployment"). Re-running is safe:
    an unchanged runtime is left alone, a changed one is updated in place.

    The guardrail is attached by environment variables: inside the runtime
    build_agent_graph() reads GUARDRAIL_ID / GUARDRAIL_VERSION and applies
    them to every agent's model (see _apply_guardrail), exactly as `test`
    and `chat` do locally.

    Returns:
        The AgentCore Runtime ARN
    """
    import agentcore_cli

    runtime_name = config.AGENTCORE_RUNTIME_NAME
    print(f"  AWS Account: {config.ACCOUNT_ID}  |  Region: {config.AWS_REGION}")
    print(f"  Runtime: {runtime_name}  |  CLI project: agentcore/agentcore.json "
          f"(stack {agentcore_cli.stack_name()})")
    previous_arn = agentcore_cli.deployed_runtime_arn()
    if previous_arn:
        print(f"  Runtime already deployed - updating it: {previous_arn}")

    # Stage the code the CLI packages (src modules + config.py + pyproject.toml).
    agentcore_cli.stage_runtime_code()

    # 1. Runtime environment: inside the runtime, config.py and _apply_guardrail() read these
    runtime_env = {
        'AWS_REGION':        config.AWS_REGION,
        'PROJECT_NAME':      config.PROJECT_NAME,
        'RETURNS_KB_ID':     config.RETURNS_KB_ID,
        'SHIPPING_KB_ID':    config.SHIPPING_KB_ID,
        'WARRANTY_KB_ID':    config.WARRANTY_KB_ID,
        'AGENT_LOG_GROUP':   config.AGENT_LOG_GROUP,
        'GUARDRAIL_ID':      guardrail_id,
        'GUARDRAIL_VERSION': guardrail_version,
    }
    missing = [name for name, value in runtime_env.items() if not value]
    if missing:
        # configure_runtime() skips empty values; re-run deploy once they exist
        print(f"  WARNING: not set yet, so not passed to the runtime: {', '.join(missing)}")

    # 2. Runtime settings -> agentcore/agentcore.json
    agentcore_cli.configure_runtime(env_vars=runtime_env,
                                    network_mode='PUBLIC',
                                    protocol='HTTP',
                                    execution_role_arn=config.AGENTCORE_ROLE_ARN)
    # 3. `agentcore deploy -y` (packages build/runtime/ and creates or updates the runtime via CDK)
    agentcore_cli.deploy()
    # 4. The ARN the CLI recorded
    runtime_arn = agentcore_cli.deployed_runtime_arn()
    if not runtime_arn:
        raise RuntimeError("The AgentCore CLI did not record a runtime ARN (agentcore/.cli/deployed-state.json)")

    print(f"  Runtime deployed: {runtime_arn}")
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(agentcore_control, runtime_arn.split('/')[-1])
    print(' ready.')
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    response = agentcore_control.create_memory(
        name=memory_name,
        description=('NovaMart customer support: a running summary of each customer session, so a '
                     'returning customer does not have to repeat themselves. Events kept 7 days.'),
        eventExpiryDuration=7,
        memoryStrategies=[{'summaryMemoryStrategy': {
            'name': 'SessionSummary',
            'description': 'Summarizes each support conversation per customer and session',
            # namespaceTemplates replaces the legacy `namespaces` parameter in the current API
            'namespaceTemplates': ['/summaries/{actorId}/{sessionId}'],
        }}],
        clientToken=str(uuid.uuid4()),
    )

    # Wait until the memory resource is ACTIVE and return its ARN.
    memory = response['memory']
    print(f"  Memory created: {memory['arn']}  (status: {memory['status']})")
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory['status'] != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        if memory['status'] == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory['status'] == 'ACTIVE' else f" status {memory['status']}")
    return memory['arn']


# ═══════════════════════════════════════════════════════
#  OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    logging_configuration = {
        'cloudWatchConfig': {'logGroupName': config.AGENT_LOG_GROUP, 'logLevel': 'INFO', 'enabled': True},
        # 100% while developing, so every request reaches the Service Map (use ~5% in production)
        'xRayConfig':       {'enabled': True, 'samplingRate': 1.0},
    }
    try:
        summary = apply_observability_config(runtime_arn, logging_configuration)
        print(f"  CloudWatch Logs : {summary['log_group']} (level "
              f"{logging_configuration['cloudWatchConfig']['logLevel']})")
        print(f"  X-Ray tracing   : {logging_configuration['xRayConfig']['samplingRate']:.0%} sampling "
              f"(CloudWatch Transaction Search)")
    except Exception as e:
        print(f"[Note] Observability configuration failed: {e}")


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline: agents -> guardrail -> runtime -> memory -> observability."""
    print("\n" + "="*60)
    print("  Deploying the NovaMart multi-agent system")
    print("="*60 + "\n")

    # Fail fast if the AgentCore CLI (used by the runtime step) is missing.
    import agentcore_cli
    print(f"AgentCore CLI: {agentcore_cli.cli_version()} ({agentcore_cli.cli_path()})\n")

    print("Step 1/5: Building agent graph...")
    orchestrator = build_agent_graph()
    print("  All 5 agents initialized\n")

    print("Step 2/5: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/5: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/5: Configuring Memory...")
    print(f"  Memory: {configure_memory(runtime_arn)}\n")

    print("Step 5/5: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print("\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    print("  Then try the deployed runtime:")
    print("  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"")
    print("  or with the CLI:  agentcore invoke \"What is the return policy for premium customers?\"")
    print("  (agentcore status / agentcore logs show the deployed runtime and its logs)\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query):
            response = orchestrator(prompt)
        print(f"Response: {response}")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat that shows each agent step, the parallel KB calls and the WorkflowState."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input("  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input):
                response = orchestrator(prompt)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── Resolve the final customer-facing text ────────────────────
        final_state = _read_workflow_state(session_id) or {}
        comm_result = final_state.get('communication_agent', '')
        text = _strip_xml_tags(comm_result or str(response))

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        with tracer.trace_request(session_id, customer_id, prompt):
            response = _orchestrator()(enriched)

        state = _read_workflow_state(session_id) or {}
        text  = _strip_xml_tags(state.get('communication_agent', '') or str(response))
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Guardrail + AgentCore Runtime + Memory + observability")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")
