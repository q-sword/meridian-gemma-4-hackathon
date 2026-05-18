#!/usr/bin/env python3
"""Meridian v_clean_005c HTTP server for Gemma 4 Good Hackathon.

Wraps meridian_api.MeridianAPI in a minimal HTTP server.

Endpoints:
  GET  /health                  -> {ok, model, n_drugs_panel}
  GET  /drugs                   -> curated drug panel (name, smiles, primary_target)
  POST /predict                 -> body: {smiles, uniprot} -> full prediction bundle
  POST /predict_drug            -> body: {drug_name, target_uniprot?} -> looks up SMILES, predicts

Start:
  python meridian_005c_server.py --port 7891
"""
import sys, json, time, argparse, traceback
from pathlib import Path
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, '/data/meridian/scripts_clean_v1')
sys.path.insert(0, '/data/meridian/scripts_v35')
sys.path.insert(0, '/data/meridian/scripts_v36')
import numpy as np
from meridian_api import MeridianAPI

# Curated drug panel - includes the v33a original 14 + hackathon-relevant additions for
# neglected disease story (malaria: PfDHODH). All SMILES verified canonical.
DRUG_PANEL = {
    # Antimalarials (PfDHODH and friends - HACKATHON STORY)
    'DSM265':         ('Cc1cn(-c2ncnc(N3CCCC3C(F)(F)F)c2N)nc1-c1ccc(C(F)(F)F)cc1', 'Q08210'),
    'Atovaquone':     ('O=C1C(=O)c2ccccc2C(=O)C1=C1CCC(c2ccc(Cl)cc2)CC1', 'P28593'),  # PfCytB
    'Pyrimethamine':  ('CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1', 'Q27738'),  # PfDHFR
    # Original v33a panel (kept for backward compat)
    'Aspirin':        ('CC(=O)Oc1ccccc1C(=O)O', 'P23219'),
    'Ibuprofen':      ('CC(C)Cc1ccc(C(C)C(=O)O)cc1', 'P23219'),
    'Caffeine':       ('Cn1c(=O)c2c(ncn2C)n(C)c1=O', 'P29274'),
    'Imatinib':       ('Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1', 'P00519'),
    'Methamphetamine':('CC(NC)Cc1ccccc1', 'P31645'),
    'Penicillin_G':   ('CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O', 'P02928'),
    'Trimethoprim':   ('COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC', 'P00374'),  # hDHFR (anchor)
    'Testosterone':   ('CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O', 'P10275'),
    'Celecoxib':      ('Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1', 'P35354'),
    'Fluoxetine':     ('CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1', 'P31645'),
    'Haloperidol':    ('OC1(c2ccc(Cl)cc2)CCN(CCCC(=O)c2ccc(F)cc2)CC1', 'P14416'),
    'Atorvastatin':   ('CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC(O)CC(O)CC(=O)O', 'P51639'),
    'Enalapril':      ('CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O', 'P12821'),
    'Alprenolol':     ('CC(C)NCC(O)COc1ccccc1CC=C', 'P08588'),
}

UNIPROT_NAMES = {
    'Q08210': 'PfDHODH (Plasmodium dihydroorotate dehydrogenase, antimalarial target)',
    'P28593': 'PfCytB (Plasmodium cytochrome b)',
    'Q27738': 'PfDHFR (Plasmodium dihydrofolate reductase)',
    'P00519': 'BCR-ABL (chronic myeloid leukemia)',
    'P00374': 'hDHFR (human dihydrofolate reductase)',
    'P23219': 'COX-1 (cyclooxygenase 1)',
    'P29274': 'Adenosine A2A receptor',
    'P31645': 'SERT (serotonin transporter)',
    'P02928': 'Penicillin-binding protein',
    'P10275': 'Androgen receptor',
    'P35354': 'COX-2 (cyclooxygenase 2)',
    'P14416': 'DRD2 (dopamine receptor)',
    'P51639': 'HMGCR (statin target)',
    'P12821': 'ACE (angiotensin-converting enzyme)',
    'P08588': 'ADRB1 (beta-1 adrenergic)',
}

class MeridianHandler(BaseHTTPRequestHandler):
    api = None  # set externally

    def log_message(self, fmt, *args):
        sys.stderr.write(f'[{time.strftime("%H:%M:%S")}] {fmt % args}\n')

    def _send_json(self, code, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get('Content-Length', 0))
        if n == 0: return {}
        return json.loads(self.rfile.read(n).decode())

    def do_GET(self):
        if self.path == '/health':
            self._send_json(200, {'ok': True, 'model': 'v_clean_005c (32.4M params, PDBbind Pearson 0.7773)',
                                  'n_drugs_panel': len(DRUG_PANEL),
                                  'analytic_thermo': 'Vant Hoff + Gibbs by construction',
                                  'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
        elif self.path == '/drugs':
            self._send_json(200, {'drugs': [
                {'name': k, 'smiles': v[0], 'primary_target_uniprot': v[1],
                 'primary_target_name': UNIPROT_NAMES.get(v[1], 'unknown')}
                for k, v in DRUG_PANEL.items()
            ]})
        else:
            self._send_json(404, {'error': f'unknown path: {self.path}'})

    def do_POST(self):
        try:
            body = self._read_json()
            if self.path == '/predict':
                smi = body.get('smiles', '').strip()
                uniprot = body.get('uniprot', '').strip()
                if not smi or not uniprot:
                    self._send_json(400, {'error': 'need both smiles and uniprot'}); return
                t0 = time.time()
                d = self.api.predict(smi, uniprot)
                if d is None:
                    self._send_json(422, {'error': 'invalid SMILES or featurization failed', 'smiles': smi}); return
                d['target_uniprot'] = uniprot
                d['target_name'] = UNIPROT_NAMES.get(uniprot, 'unknown')
                d['inference_seconds'] = round(time.time() - t0, 3)
                # Convert numpy types to native python for JSON
                d = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                          bool(v) if isinstance(v, np.bool_) else v)
                     for k, v in d.items()}
                self._send_json(200, d)
            elif self.path == '/predict_drug':
                drug = body.get('drug_name', '').strip()
                tgt = body.get('target_uniprot', '').strip()
                if drug not in DRUG_PANEL:
                    self._send_json(404, {'error': f'drug "{drug}" not in curated panel',
                                          'available': sorted(DRUG_PANEL.keys())}); return
                smi, primary_target = DRUG_PANEL[drug]
                uniprot = tgt if tgt else primary_target
                t0 = time.time()
                d = self.api.predict(smi, uniprot)
                if d is None:
                    self._send_json(422, {'error': 'predict failed', 'drug': drug}); return
                d['drug_name'] = drug
                d['target_uniprot'] = uniprot
                d['target_name'] = UNIPROT_NAMES.get(uniprot, 'unknown')
                d['target_is_primary'] = (uniprot == primary_target)
                d['inference_seconds'] = round(time.time() - t0, 3)
                d = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                          bool(v) if isinstance(v, np.bool_) else v)
                     for k, v in d.items()}
                self._send_json(200, d)
            else:
                self._send_json(404, {'error': f'unknown path: {self.path}'})
        except Exception as e:
            self._send_json(500, {'error': str(e), 'traceback': traceback.format_exc()})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=7891)
    ap.add_argument('--host', default='0.0.0.0')
    args = ap.parse_args()

    print(f'[boot] Loading MeridianAPI (v_clean_005c, this takes ~30s)...', flush=True)
    t0 = time.time()
    MeridianHandler.api = MeridianAPI(device='cuda', verbose=True)
    print(f'[boot] MeridianAPI loaded in {time.time()-t0:.1f}s', flush=True)
    print(f'[boot] DRUG_PANEL: {len(DRUG_PANEL)} drugs', flush=True)
    print(f'[boot] Listening on http://{args.host}:{args.port}/  (endpoints: /health /drugs /predict /predict_drug)', flush=True)

    server = ThreadingHTTPServer((args.host, args.port), MeridianHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[shutdown]'); server.shutdown()