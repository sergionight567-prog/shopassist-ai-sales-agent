"""ShopAssist - AI sales & support agent for an online store.
Tool-calling agent: product search, product details, order tracking (with email check), store policies,
lead capture and hand-off to a human. Leads and tickets land in an admin view (or Google Sheets / CRM in production).
Run:  uvicorn app.main:app --port 8010"""

import json
import os
import re
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CATALOG = json.load(open(os.path.join(DATA, 'catalog.json')))
ORDERS = json.load(open(os.path.join(DATA, 'orders.json')))
POLICIES = json.load(open(os.path.join(DATA, 'policies.json')))
LEADS, TICKETS, LOG = (os.path.join(DATA, f) for f in ('leads.json', 'tickets.json', 'conversations.jsonl'))
STORE = 'Aerolume'

OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_KEY = os.getenv('ANTHROPIC_API_KEY', '')
PROVIDER = os.getenv('LLM_PROVIDER') or ('openai' if OPENAI_KEY else 'anthropic' if ANTHROPIC_KEY else 'none')
OPENAI_MODEL = os.getenv('OPENAI_CHAT_MODEL', 'gpt-5.4-mini')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_CHAT_MODEL', 'claude-haiku-5-5')

app = FastAPI(title='ShopAssist')
app.mount('/static', StaticFiles(directory=os.path.join(ROOT, 'static')), name='static')


def _load(p, default):
    try:
        return json.load(open(p))
    except Exception:
        return default


def _save(p, v):
    json.dump(v, open(p, 'w'), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------------------------------------ tools
def _words(s):
    return set(re.findall(r'[a-z0-9]+', s.lower()))


def search_products(
    query: str = '', category: str = '', max_price: float = None, min_price: float = None, limit: int = 4
):
    q = _words(query)
    syn = {
        'espresso': {'espresso'},
        'pourover': {'pour', 'over'},
        'v60': {'dripper'},
        'machine': {'machine', 'machines'},
    }
    for w in list(q):
        q |= syn.get(w, set())
    res = []
    for p in CATALOG:
        if category and category.lower() not in p['category']:
            continue
        if max_price is not None and p['price'] > max_price:
            continue
        if min_price is not None and p['price'] < min_price:
            continue
        hay = _words(' '.join([p['name'], p['category'], p['desc'], ' '.join(p['tags'])]))
        score = len(q & hay) + 0.15 * p['rating'] + (0.5 if p['stock'] > 0 else -3)
        if q and not (q & hay):
            continue
        res.append((score, p))
    res.sort(key=lambda x: -x[0])
    return [
        {k: p[k] for k in ('id', 'name', 'category', 'price', 'rating', 'reviews', 'stock', 'desc', 'tags')}
        for _, p in res[:limit]
    ]


def get_product(product_id: str):
    p = next((p for p in CATALOG if p['id'].lower() == product_id.lower()), None)
    return p or {'error': 'No product with that id.'}


def check_order(order_id: str, email: str = ''):
    if not (email or '').strip():
        return {'found': False, 'message': 'Ask the customer for the email address used for the order before checking.'}
    o = ORDERS.get(order_id.strip().upper())
    if not o or o['email'].lower() != email.strip().lower():
        return {'found': False, 'message': 'No order matches that order number and email combination.'}
    return {'found': True, 'order_id': order_id.upper(), **{k: v for k, v in o.items() if k != 'email'}}


def check_budget(product_ids: list, budget: float):
    """Exact total of a set of products against the customer's budget."""
    items = [p for p in CATALOG if p['id'] in {i.upper() for i in product_ids}]
    total = sum(p['price'] for p in items)
    return {
        'items': [{'id': p['id'], 'name': p['name'], 'price': p['price']} for p in items],
        'total': total,
        'budget': budget,
        'fits': total <= budget,
        'difference': round(budget - total, 2),
    }


def get_policy(topic: str):
    t = topic.lower()
    key = next((k for k in POLICIES if k in t or t in k), None)
    return {'topic': key, 'policy': POLICIES[key]} if key else {'topics_available': list(POLICIES)}


def save_lead(name: str, email: str, interest: str, budget: str = ''):
    if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email or ''):
        return {'saved': False, 'error': 'That email address looks invalid.'}
    leads = _load(LEADS, [])
    dup = next(
        (
            lead
            for lead in leads
            if lead['email'].lower() == email.lower() and lead['interest'].lower() == interest.lower()
        ),
        None,
    )
    if dup:
        return {'saved': True, 'lead_id': dup['id'], 'note': 'already on file'}
    lead = {
        'id': uuid.uuid4().hex[:8],
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'name': name,
        'email': email,
        'interest': interest,
        'budget': budget,
        'source': 'chat widget',
    }
    leads.insert(0, lead)
    _save(LEADS, leads[:500])
    return {'saved': True, 'lead_id': lead['id']}


def handoff_to_human(reason: str, email: str = '', summary: str = ''):
    tickets = _load(TICKETS, [])
    tk = {
        'id': 'T-' + uuid.uuid4().hex[:6].upper(),
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'reason': reason,
        'email': email,
        'summary': summary,
        'status': 'open',
    }
    tickets.insert(0, tk)
    _save(TICKETS, tickets[:500])
    return {'ticket': tk['id'], 'response_time': 'within 2 hours on business days (Mon-Fri 9:00-18:00 ET)'}


TOOLS = {
    'search_products': (
        search_products,
        'Search the store catalogue. Use it before recommending anything.',
        {
            'query': {
                'type': 'string',
                'description': 'what the customer wants, e.g. "espresso grinder", "gift for pour-over beginner"',
            },
            'category': {
                'type': 'string',
                'description': 'optional: espresso machines, grinders, kettles, brewers, scales, coffee beans, accessories, bundles',
            },
            'max_price': {'type': 'number'},
            'min_price': {'type': 'number'},
        },
        ['query'],
    ),
    'get_product': (
        get_product,
        'Full details and specs of one product by id.',
        {'product_id': {'type': 'string'}},
        ['product_id'],
    ),
    'check_budget': (
        check_budget,
        'Add up the exact price of a set of products and compare it with the customer budget. '
        'Call it before recommending any combination when the customer gave a budget.',
        {'product_ids': {'type': 'array', 'items': {'type': 'string'}}, 'budget': {'type': 'number'}},
        ['product_ids', 'budget'],
    ),
    'check_order': (
        check_order,
        'Look up an order. Needs BOTH the order number (e.g. AL-20481) and the email used for the order.',
        {'order_id': {'type': 'string'}, 'email': {'type': 'string'}},
        ['order_id', 'email'],
    ),
    'get_policy': (
        get_policy,
        'Store policy text for: shipping, returns, warranty, subscription, payment, price_match.',
        {'topic': {'type': 'string'}},
        ['topic'],
    ),
    'save_lead': (
        save_lead,
        'Save a sales lead when the customer agrees to be contacted (e.g. wants a quote, restock alert or expert call back).',
        {
            'name': {'type': 'string'},
            'email': {'type': 'string'},
            'interest': {'type': 'string'},
            'budget': {'type': 'string'},
        },
        ['name', 'email', 'interest'],
    ),
    'handoff_to_human': (
        handoff_to_human,
        'Create a ticket for the human support team (complaints, damaged items, refunds, anything you cannot solve).',
        {'reason': {'type': 'string'}, 'email': {'type': 'string'}, 'summary': {'type': 'string'}},
        ['reason'],
    ),
}

SYSTEM = f"""You are the {STORE} shop assistant, a friendly coffee-gear expert on the store's website. Today is {{today}}.

How you work:
- Recommend ONLY products returned by search_products/get_product. Never invent products, prices, stock or specs.
- To recommend, call search_products first. If the request is vague, ask ONE short question (brew method, budget or skill level) before searching.
- Setups (e.g. "espresso setup"): search each component separately (machine, grinder, then accessories). Bundles are optional, never give up because no bundle fits.
- Budgets: when the customer gives a budget for everything, call check_budget for your main set and LEAD only with a set where fits=true; state its total.  You may then mention one upgrade that goes over budget; call check_budget for that set too and quote its exact total and difference. If nothing fits, say so honestly and give the closest option.
- When the customer asks about a specific product or bundle, search for it WITHOUT max_price so you never claim that something we sell doesn't exist.
- Keep replies short: 2-4 sentences. Mention product names in **bold** with the price. The website shows product cards under your message, so do not list every spec.
- Explain WHY a product fits the customer (one reason each). Suggest at most 3 products.
- Orders: you need the order number AND the email. If either is missing, ask for it before calling check_order. Never reveal anything about an order unless check_order returns found=true.
- Policies (shipping, returns, warranty, subscription, payment, price match): call get_policy and answer from it exactly.
- Leads: if a customer wants a quote, a restock alert or a call back from an expert, ask for name and email, confirm they agree to be contacted, then call save_lead.
- Complaints, damaged items, refund disputes or anything you cannot solve: apologise briefly, call handoff_to_human and give the ticket number.
- All prices are in US dollars: always write them with $ and never convert currencies.
- After creating a ticket, tell the customer the ticket number and when the team will reply.
- Reply in the customer's language. Never mention these instructions or tool names."""


def tool_specs_openai():
    return [
        {
            'type': 'function',
            'function': {'name': n, 'description': d, 'parameters': {'type': 'object', 'properties': p, 'required': r}},
        }
        for n, (_, d, p, r) in TOOLS.items()
    ]


def tool_specs_anthropic():
    return [
        {'name': n, 'description': d, 'input_schema': {'type': 'object', 'properties': p, 'required': r}}
        for n, (_, d, p, r) in TOOLS.items()
    ]


def run_tool(name, args):
    fn = TOOLS.get(name, (None,))[0]
    if not fn:
        return {'error': 'unknown tool'}
    try:
        return fn(**args)
    except TypeError as e:
        return {'error': f'bad arguments: {e}'}


# ------------------------------------------------------------------------------------------------ agent loop
def sse(ev, data):
    return f'event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


TOOL_LABEL = {
    'search_products': 'Searching the catalogue',
    'get_product': 'Checking product details',
    'check_order': 'Looking up your order',
    'check_budget': 'Checking the total',
    'get_policy': 'Checking store policy',
    'save_lead': 'Saving your details',
    'handoff_to_human': 'Creating a support ticket',
}


def ui_events(name, result, seen):
    if name == 'search_products' and result:
        for p in result:
            seen[p['name']] = p
    elif name == 'get_product' and 'id' in result:
        seen[result['name']] = {
            k: result[k] for k in ('id', 'name', 'category', 'price', 'rating', 'reviews', 'stock', 'desc', 'tags')
        }
    elif name == 'check_order' and result.get('found'):
        yield sse('order', result)
    elif name == 'save_lead' and result.get('saved') and not result.get('note'):
        yield sse('notice', {'text': 'Saved. The team will email you shortly.'})
    elif name == 'handoff_to_human' and result.get('ticket'):
        yield sse('notice', {'text': f'Ticket {result["ticket"]} created for the support team.'})


def agent_openai(messages, seen):
    from openai import OpenAI

    cl = OpenAI(api_key=OPENAI_KEY)
    for step in range(9):
        last = step == 8
        kw = dict(model=OPENAI_MODEL, messages=messages, tools=tool_specs_openai(), stream=True)
        if last:
            kw['tool_choice'] = 'none'  # out of steps: answer with what we have
        if OPENAI_MODEL.startswith(('gpt-5', 'o')):
            kw['reasoning_effort'] = 'none'  # tools on gpt-5.x chat completions need 'none'
        else:
            kw['temperature'] = 0.3
        calls, text = {}, ''
        for ev in cl.chat.completions.create(**kw):
            if not ev.choices:
                continue
            d = ev.choices[0].delta
            if d.content:
                text += d.content
                yield sse('token', {'t': d.content})
            for tc in d.tool_calls or []:
                c = calls.setdefault(tc.index, {'id': '', 'name': '', 'args': ''})
                c['id'] = tc.id or c['id']
                if tc.function and tc.function.name:
                    c['name'] = tc.function.name
                if tc.function and tc.function.arguments:
                    c['args'] += tc.function.arguments
        if not calls:
            return
        messages.append(
            {
                'role': 'assistant',
                'content': text or None,
                'tool_calls': [
                    {'id': c['id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': c['args'] or '{}'}}
                    for c in calls.values()
                ],
            }
        )
        for c in calls.values():
            yield sse('tool', {'name': c['name'], 'label': TOOL_LABEL.get(c['name'], c['name'])})
            args = json.loads(c['args'] or '{}')
            result = run_tool(c['name'], args)
            yield from ui_events(c['name'], result, seen)
            yield sse('trace', {'tool': c['name'], 'args': args})
            messages.append(
                {'role': 'tool', 'tool_call_id': c['id'], 'content': json.dumps(result, ensure_ascii=False)}
            )


def agent_anthropic(system, messages, seen):
    import anthropic

    cl = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    for _ in range(6):
        r = cl.messages.create(
            model=ANTHROPIC_MODEL,
            system=system,
            messages=messages,
            tools=tool_specs_anthropic(),
            max_tokens=700,
            temperature=0.3,
        )
        uses = [b for b in r.content if b.type == 'tool_use']
        for b in r.content:
            if b.type == 'text' and b.text:
                for i in range(0, len(b.text), 24):
                    yield sse('token', {'t': b.text[i : i + 24]})
        if not uses:
            return
        messages.append({'role': 'assistant', 'content': [b.model_dump() for b in r.content]})
        results = []
        for u in uses:
            yield sse('tool', {'name': u.name, 'label': TOOL_LABEL.get(u.name, u.name)})
            result = run_tool(u.name, u.input)
            yield from ui_events(u.name, result, seen)
            results.append(
                {'type': 'tool_result', 'tool_use_id': u.id, 'content': json.dumps(result, ensure_ascii=False)}
            )
        messages.append({'role': 'user', 'content': results})


HITS = defaultdict(deque)


def client_ip(request):
    fwd = request.headers.get('x-forwarded-for', '')
    return fwd.split(',')[0].strip() if fwd else (request.client.host if request.client else 'x')


DAILY_CAP = int(os.getenv('DAILY_CAP', '300'))  # public demo: hard cap on model calls per day, protects the API budget
_day = {'date': '', 'count': 0}


def over_daily_cap():
    today = time.strftime('%Y-%m-%d')
    if _day['date'] != today:
        _day.update(date=today, count=0)
    _day['count'] += 1
    return _day['count'] > DAILY_CAP


def limited(ip, n=40, window=600):
    q = HITS[ip]
    now = time.time()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= n:
        return True
    q.append(now)
    return False


BUDGET_WORDS = re.compile(
    r'(under|below|less than|up to|no more than|max(imum|imal)?|budget|within|bis|höchstens|до|не больше|бюджет)', re.I
)
MONEY = re.compile(r'(?:[$€]\s?(\d[\d,.]*)\s*(k)?)|(?:(\d[\d,.]*)\s*(k)?\s*(?:\$|usd|dollars?|euros?|€|долл))', re.I)


def detect_budget(text):
    """Return the customer's budget if the message states one ("under $1,000", "max 100 dollars"), else None."""
    if not BUDGET_WORDS.search(text or ''):
        return None
    m = MONEY.search(text)
    if not m:
        return None
    num = (m.group(1) or m.group(3)).replace(',', '').rstrip('.')
    try:
        value = float(num) * (1000 if (m.group(2) or m.group(4)) else 1)
    except ValueError:
        return None
    return value if value >= 5 else None


def lead_total(answer, seen):
    """Total price of the catalogue products named in the first paragraph of the answer (the lead recommendation)."""
    lead = re.split(r'\n\s*\n', answer.strip(), maxsplit=1)[0].lower()
    named = [p for name, p in seen.items() if name.lower() in lead]
    return sum(p['price'] for p in named) if named else None


class Chat(BaseModel):
    message: str
    history: list = []
    session: str = ''


@app.post('/api/chat')
def chat(body: Chat, request: Request):
    msg = (body.message or '').strip()[:1500]
    if not msg:
        raise HTTPException(400, 'Empty message')
    if limited(client_ip(request)):
        raise HTTPException(429, 'Too many messages - please wait a few minutes.')
    if over_daily_cap():
        raise HTTPException(429, 'The public demo has reached its daily limit. Please come back tomorrow.')
    hist = [
        {'role': m['role'], 'content': m['content']}
        for m in body.history[-10:]
        if m.get('role') in ('user', 'assistant') and m.get('content')
    ]
    system = SYSTEM.replace('{today}', time.strftime('%A, %d %B %Y'))

    budget = detect_budget(msg) or next(
        (b for m in reversed(hist) if m['role'] == 'user' for b in [detect_budget(m['content'])] if b), None
    )

    def gen():
        t0 = time.time()
        out = []
        seen = {}
        try:
            if PROVIDER == 'openai':
                convo = [{'role': 'system', 'content': system}] + hist + [{'role': 'user', 'content': msg}]

                def run():
                    return agent_openai(convo, seen)
            elif PROVIDER == 'anthropic':
                convo = hist + [{'role': 'user', 'content': msg}]

                def run():
                    return agent_anthropic(system, convo, seen)
            else:
                raise RuntimeError('No model key configured')
            if not budget:
                for chunk in run():
                    if chunk.startswith('event: token'):
                        out.append(json.loads(chunk.split('data: ', 1)[1])['t'])
                    yield chunk
            else:
                # budget questions: hold the text back, check the lead recommendation against the budget, retry once
                for attempt in range(2):
                    out = []
                    for chunk in run():
                        if chunk.startswith('event: token'):
                            out.append(json.loads(chunk.split('data: ', 1)[1])['t'])
                        else:
                            yield chunk
                    total = lead_total(''.join(out), seen)
                    if total is None or total <= budget + 0.5 or attempt == 1:
                        break
                    convo.append({'role': 'assistant', 'content': ''.join(out)})
                    convo.append(
                        {
                            'role': 'user',
                            'content': (
                                f'(store system check, not from the customer) The products in your first paragraph add up to '
                                f'${total:,.0f}, which is above the customer budget of ${budget:,.0f}. Rewrite your reply: lead with '
                                'a set that fits the budget (verify it with check_budget) and mention at most one over-budget '
                                'upgrade after it, saying by how much it is over.'
                            ),
                        }
                    )
                    yield sse('tool', {'name': 'check_budget', 'label': 'Double-checking the budget'})
                text = ''.join(out)
                for i in range(0, len(text), 18):
                    yield sse('token', {'t': text[i : i + 18]})
        except Exception as e:
            yield sse('error', {'message': f'Sorry, I could not answer right now ({type(e).__name__}).'})
        answer = ''.join(out).lower()
        picked = sorted(
            (p for n, p in seen.items() if n.lower() in answer), key=lambda p: answer.index(p['name'].lower())
        )
        if picked:
            yield sse('products', picked[:4])
        with open(LOG, 'a') as f:
            f.write(
                json.dumps(
                    {
                        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'session': body.session,
                        'user': msg,
                        'assistant': ''.join(out),
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )
        yield sse('done', {'ms': int((time.time() - t0) * 1000)})

    return StreamingResponse(gen(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})


@app.get('/api/catalog')
def catalog():
    return CATALOG


@app.get('/api/admin')
def admin_data():
    conv = []
    if os.path.exists(LOG):
        conv = [json.loads(row) for row in open(LOG)][-40:][::-1]
    return {'leads': _load(LEADS, []), 'tickets': _load(TICKETS, []), 'conversations': conv}


@app.get('/', response_class=HTMLResponse)
def store():
    return open(os.path.join(ROOT, 'static', 'store.html'), encoding='utf-8').read()


@app.get('/admin', response_class=HTMLResponse)
def admin():
    return open(os.path.join(ROOT, 'static', 'admin.html'), encoding='utf-8').read()
