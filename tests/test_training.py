import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_two_round_cpu_compression_and_stock_transport(tmp_path):
    command=[sys.executable,str(ROOT/'train.py'),'--dataset','synthetic','--clients','2','--rounds','2',
        '--epochs','2','--warmup-epochs','1','--server-epochs','1','--batch-size','8',
        '--min-clusters','2','--max-clusters','3','--window','1','--patience','1',
        '--score-tolerance','100','--output',str(tmp_path/'run')]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((tmp_path/'run/results.json').read_text())
    assert all(r['parameters_changed'] and r['server_self_compression'] for r in report['rounds'])
    assert report['rounds'][-1]['clusters_next_round']==3
    assert all(r['upstream_protobuf_bytes']>0 and r['downstream_protobuf_bytes']>0 for r in report['rounds'])
    assert (tmp_path/'run/compressed.h5').is_file()
