"""gemma_meridian_demo_v2.py - Gemma 4 Good Hackathon submission.

Gemma 4 (e4b) as a clinical decision-support assistant for binding affinity
prediction. Uses Meridian v_clean_005c (32.4M params, PDBbind Pearson 0.7773)
as a function-calling tool, with analytic Van't Hoff + Gibbs thermodynamic
consistency by construction.

This is the SECOND version, upgraded May 17 from the original April 20 demo:
- Backbone model: v33-alpha -> v_clean_005c (today's analytic thermo fix)
- Adds full thermodynamic profile (pKd, dG, dH, mTdS) with physical consistency
- Adds SmartCalibrator output (per-target ESM-similarity calibration with 80%CI)
- Adds 3 antimalarial drugs for neglected-disease story
- Adds arbitrary-SMILES predict tool (for advanced users)

Flow:
    user query
      -> Gemma 4 sees tool list, decides to call meridian tool
      -> POST http://127.0.0.1:7891/predict[_drug]
      -> tool result fed back to Gemma
      -> Gemma writes calibrated clinical-language answer citing:
          pKd value, 80% confidence interval (nM and pKd scale),
          thermodynamic profile (enthalpy- vs entropy-driven),
          calibration source (direct anchor vs ESM-extrapolated),
          structural provenance (AlphaFold/KLIFS coverage),
          safety caveats, never hallucinates numbers.

Usage:
    python gemma_meridian_demo_v2.py                      # interactive REPL
    python gemma_meridian_demo_v2.py --query "..."        # one-shot
    python gemma_meridian_demo_v2.py --demo               # scripted demo cases
"""
from __future__ import annotations
import argparse, json, os, sys, time
import urllib.request, urllib.error

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')
MERIDIAN_URL = os.environ.get('MERIDIAN_URL', 'http://127.0.0.1:7891')
MODEL = os.environ.get('GEMMA_MODEL', 'gemma4:e4b')
NUM_GPU = int(os.environ.get('GEMMA_NUM_GPU', '999'))

MERIDIAN_MODE = os.environ.get('MERIDIAN_MODE', 'live').lower()
import pathlib as _pl
CACHE_PATH = os.environ.get('MERIDIAN_CACHE_PATH', str(_pl.Path(__file__).parent / 'cached_responses.json'))

_CACHE = None
def _get_cache():
    global _CACHE
    if _CACHE is None:
        try:
            with open(CACHE_PATH) as f:
                _CACHE = json.load(f)
            print(f'[cache] loaded {len(_CACHE)} entries from {CACHE_PATH}', flush=True)
        except Exception as e:
            print(f'[cache] FAILED to load {CACHE_PATH}: {e}', file=sys.stderr)
            _CACHE = {}
    return _CACHE

def _cache_lookup(tool_name, args):
    cache = _get_cache()
    key = f'{tool_name}:{json.dumps(args, sort_keys=True)}'
    if key in cache:
        return cache[key]
    return {'error': f'cache miss for {key}', 'hint': str([k for k in cache.keys() if k.startswith(tool_name)])[:300]}


SYSTEM_PROMPT = """You are a careful clinical decision-support assistant for drug-target binding analysis. You have access to Meridian v_clean_005c, a physics-informed binding affinity predictor with built-in Van't Hoff and Gibbs equation consistency.

HARD RULES:
1. NEVER guess a pKd, Kd, IC50, dG, dH, or any binding/thermodynamic value. ALWAYS call the tool. If the drug or target isn't accessible to the tool, say so explicitly and decline to estimate.
2. ALWAYS report:
   - pKd_calibrated and its 80% confidence interval (pkd_80ci_low, pkd_80ci_high) in both pKd units and nM.
   - The calibration_source ('direct_anchor_*' means anchored to a real measurement; 'esm_weighted_top=*' means extrapolated by ESM-similarity to a related target; calibration_extrapolated=true means the prediction is outside the model's anchored range).
   - The thermodynamic decomposition (dG, dH, mTdS) and explain whether binding is enthalpy-driven (large negative dH, small mTdS), entropy-driven (small dH, large negative mTdS), or balanced.
3. NEVER provide individualized medical advice. Frame everything as research/literature context, not a clinical recommendation. Always recommend consulting peer-reviewed primary sources for clinical decisions.
4. If multiple drugs are mentioned, call the tool once per drug.
5. pKd scale interpretation: 4-6 weak, 6-8 moderate, 8-10 tight, >10 very tight. Higher = stronger binding.
6. Thermodynamic interpretation: dG = -RT*ln(10)*pKd by physics. dG = dH + mTdS by Gibbs. These hold by construction in Meridian.

Voice: precise, calm, uses concrete numbers, never marketing language. Cite uncertainty honestly.
"""

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'predict_drug_binding',
            'description': 'Predict the binding affinity of a curated-panel drug against a protein target using Meridian v_clean_005c. Returns calibrated pKd (with 80% confidence interval in pKd and nM units), the full thermodynamic decomposition (dG, dH, mTdS in kcal/mol, all physically consistent via Van\'t Hoff and Gibbs), the calibration provenance (anchored vs ESM-extrapolated), structural data flags (AlphaFold/KLIFS coverage), and drug-likeness properties (MW, logP, TPSA, Lipinski). If target_uniprot is omitted, uses the drug\'s primary curated target.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'drug_name': {
                        'type': 'string',
                        'description': 'Drug name from the curated panel. Available: DSM265 (antimalarial, PfDHODH inhibitor), Atovaquone (antimalarial, PfCytB), Pyrimethamine (antimalarial, PfDHFR), Imatinib (oncology, BCR-ABL), Aspirin, Ibuprofen, Caffeine, Methamphetamine, Penicillin_G, Trimethoprim, Testosterone, Celecoxib, Fluoxetine, Haloperidol, Atorvastatin, Enalapril, Alprenolol. Call list_curated_drugs first if unsure.',
                    },
                    'target_uniprot': {
                        'type': 'string',
                        'description': 'Optional UniProt ID of the target protein (e.g. Q08210 for PfDHODH, P00519 for BCR-ABL). If omitted, uses the drug\'s primary curated target.',
                    },
                },
                'required': ['drug_name'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'predict_smiles_binding',
            'description': 'Predict binding affinity for an arbitrary SMILES (drug-like molecule) against a UniProt-identified protein target. Use this when the user provides a SMILES string directly or the drug is not in the curated panel. Returns the same rich prediction bundle as predict_drug_binding.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'smiles': {'type': 'string', 'description': 'Canonical SMILES of the small molecule.'},
                    'uniprot': {'type': 'string', 'description': 'UniProt accession of the target protein.'},
                },
                'required': ['smiles', 'uniprot'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'list_curated_drugs',
            'description': 'List all drugs in the Meridian curated panel with their primary protein targets. Call this if the user asks about available drugs or you need to disambiguate a drug name.',
            'parameters': {'type': 'object', 'properties': {}},
        },
    },
]

def _post(url, payload, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                  headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())

def execute_tool(name, args):
    if MERIDIAN_MODE == 'cached':
        return _cache_lookup(name, args)
    try:
        if name == 'predict_drug_binding':
            payload = {'drug_name': args.get('drug_name', '')}
            if args.get('target_uniprot'):
                payload['target_uniprot'] = args['target_uniprot']
            return _post(f'{MERIDIAN_URL}/predict_drug', payload)
        if name == 'predict_smiles_binding':
            return _post(f'{MERIDIAN_URL}/predict', {'smiles': args.get('smiles', ''), 'uniprot': args.get('uniprot', '')})
        if name == 'list_curated_drugs':
            return _get(f'{MERIDIAN_URL}/drugs')
        return {'error': f'unknown tool: {name}'}
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else str(e)
        return {'error': f'meridian {e.code}: {body}'}
    except Exception as e:
        return {'error': f'meridian failure: {e}'}

def gemma_chat(messages, tools=None):
    payload = {'model': MODEL, 'messages': messages, 'stream': False,
               'options': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 64, 'num_predict': 1024}}
    if tools: payload['tools'] = tools
    if NUM_GPU != 999: payload['options']['num_gpu'] = NUM_GPU
    return _post(f'{OLLAMA_URL}/api/chat', payload, timeout=300)

def run_query(user_query, max_tool_rounds=4, verbose=True):
    t_start = time.time()
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_query},
    ]
    tool_trace = []
    for round_i in range(max_tool_rounds):
        if verbose: print(f'  [round {round_i}] calling Gemma {MODEL}...', flush=True)
        t0 = time.time()
        r = gemma_chat(messages, tools=TOOLS)
        gemma_dt = time.time() - t0
        msg = r.get('message', {})
        tool_calls = msg.get('tool_calls') or []
        content = msg.get('content', '')
        if verbose: print(f'    gemma {gemma_dt:.1f}s tool_calls={len(tool_calls)} content={len(content)}c', flush=True)
        assistant_turn = {'role': 'assistant', 'content': content}
        if tool_calls: assistant_turn['tool_calls'] = tool_calls
        messages.append(assistant_turn)
        if not tool_calls:
            return {'final_content': content, 'rounds': round_i + 1, 'tool_trace': tool_trace,
                    'wall_seconds': time.time() - t_start}
        for tc in tool_calls:
            fn = tc.get('function', {})
            tname = fn.get('name')
            targs = fn.get('arguments') or {}
            if isinstance(targs, str):
                try: targs = json.loads(targs)
                except: pass
            if verbose: print(f'    -> tool {tname}({targs})', flush=True)
            t0 = time.time()
            tresult = execute_tool(tname, targs)
            tool_dt = time.time() - t0
            if verbose: print(f'       returned in {tool_dt*1000:.0f}ms', flush=True)
            tool_trace.append({'name': tname, 'args': targs, 'result': tresult, 'seconds': tool_dt})
            messages.append({'role': 'tool', 'name': tname, 'content': json.dumps(tresult, default=str)})
    return {'final_content': '[max rounds reached]', 'rounds': max_tool_rounds, 'tool_trace': tool_trace,
            'wall_seconds': time.time() - t_start}

DEMO_QUERIES = [
    "What is the predicted binding affinity of DSM265 against PfDHODH? Is the binding entropy-driven or enthalpy-driven? What does this mean for malaria drug development?",
    "Compare Imatinib's binding to BCR-ABL with its expected off-target binding to a serotonin transporter. Should we worry about off-target effects?",
    "I'm looking at this compound: Cc1cn(-c2ncnc(N3CCCC3C(F)(F)F)c2N)nc1-c1ccc(C(F)(F)F)cc1. How does it bind to PfDHODH (Q08210)?",
    "What antimalarial drugs do you have in your curated panel? Which target is most druggable based on the model?",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', help='one-shot query')
    ap.add_argument('--demo', action='store_true', help='run scripted demo cases')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    # Sanity
    if MERIDIAN_MODE == 'cached':
        c = _get_cache()
        print(f'[meridian] CACHED mode, {len(c)-1} entries loaded from {CACHE_PATH}', flush=True)
        print(f'[meridian] (queries outside the cached set will return a cache-miss error)', flush=True)
    else:
      try:
        h = _get(f'{MERIDIAN_URL}/health')
        print(f'[meridian] {h.get("model", "unknown")}', flush=True)
        print(f'[meridian] {h.get("analytic_thermo", "")}', flush=True)
      except Exception as e:
        print(f'FATAL: Meridian server unreachable at {MERIDIAN_URL}: {e}', file=sys.stderr); sys.exit(1)
    try:
        ot = _get(f'{OLLAMA_URL}/api/tags')
        models = [m['name'] for m in ot.get('models', [])]
        if MODEL not in models:
            print(f'FATAL: Gemma model {MODEL} not in Ollama. Available: {models[:5]}', file=sys.stderr); sys.exit(1)
        print(f'[ollama] {MODEL} ready', flush=True)
    except Exception as e:
        print(f'FATAL: Ollama unreachable at {OLLAMA_URL}: {e}', file=sys.stderr); sys.exit(1)

    if args.demo:
        for i, q in enumerate(DEMO_QUERIES, 1):
            print(f'\n{"="*70}\nDEMO {i}/{len(DEMO_QUERIES)}: {q}\n{"="*70}', flush=True)
            r = run_query(q, verbose=not args.quiet)
            print(f'\n--- Gemma answer ({r["wall_seconds"]:.1f}s, {r["rounds"]} rounds, {len(r["tool_trace"])} tool calls) ---')
            print(r['final_content'])
        return
    if args.query:
        r = run_query(args.query, verbose=not args.quiet)
        print(f'\n--- Gemma answer ({r["wall_seconds"]:.1f}s) ---')
        print(r['final_content'])
        return

    # REPL
    print('\nMeridian + Gemma 4 interactive REPL. Type a question, blank line to quit.\n', flush=True)
    while True:
        try: q = input('> ').strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if not q: break
        r = run_query(q, verbose=not args.quiet)
        print(f'\n--- Gemma ({r["wall_seconds"]:.1f}s) ---\n{r["final_content"]}\n')

if __name__ == '__main__':
    main()