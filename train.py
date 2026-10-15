"""Run FedCompress sequentially using stock Flower messages."""
import argparse
import functools
import json
import os
from pathlib import Path
import platform
import random
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None, *, driver=None, client_id=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['fedavg','client','fedcompress'], default='fedcompress')
    p.add_argument('--dataset', choices=['synthetic','cifar10','cifar100','pathmnist','spcm','voxforge'], default='cifar10')
    p.add_argument('--data-root', type=Path, default=Path('datasets'), help='TFDS cache and prepared PathMNIST root')
    p.add_argument('--ood-dir', type=Path, help='Prepared unlabeled images or mono 16 kHz WAVs')
    p.add_argument('--output', type=Path, default=Path('runs/fedcompress'))
    p.add_argument('--clients', type=int, default=20)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--participation', type=float, default=1.)
    p.add_argument('--epochs', type=int, default=10, help='Total local epochs including warmup')
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--server-epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--cluster-lr', type=float, default=.0005)
    p.add_argument('--server-lr', type=float, default=.0001)
    p.add_argument('--temperature', type=float, default=2.)
    p.add_argument('--min-clusters', type=int, default=8)
    p.add_argument('--max-clusters', type=int, default=30)
    p.add_argument('--cluster-step', type=int, default=1)
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--score-tolerance', type=float, default=.01)
    p.add_argument('--compression-step', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--init-model', type=Path)
    args = p.parse_args(argv)
    if min(args.clients,args.rounds,args.epochs,args.batch_size,args.threads,args.min_clusters,args.cluster_step,args.compression_step) < 1:
        p.error('Counts and intervals must be positive')
    if not 0 < args.participation <= 1 or args.min_clusters > args.max_clusters: p.error('Invalid participation or cluster bounds')
    if args.method != 'fedavg' and not 0 <= args.warmup_epochs < args.epochs:
        p.error('Clustering needs 0 <= warmup-epochs < epochs')
    if args.method == 'fedcompress' and args.server_epochs < 1: p.error('FedCompress requires server epochs')
    if args.method == 'fedcompress' and args.dataset != 'synthetic' and (args.ood_dir is None or not args.ood_dir.is_dir()):
        p.error('FedCompress with real data requires --ood-dir containing prepared unlabeled images or WAVs')
    output = args.output.resolve()
    if client_id is None and output.exists() and any(output.iterdir()): p.error('Output directory must be empty or new')
    os.environ['TFDS_DATA_DIR'] = str(args.data_root.resolve())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(args.threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(args.threads)
    if args.device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    import tensorflow as tf
    import flwr as fl
    from flwr.common import serde
    from client import _Client
    from server import _Server
    from network import get_model
    from data import get_cifar10, get_cifar100, get_pathmnist, get_spcm, get_voxforge
    tf.keras.utils.set_random_seed(args.seed)
    if args.device == 'cuda' and not tf.config.list_physical_devices('GPU'): p.error('CUDA requested but no TensorFlow GPU is available')
    audio_dataset = args.dataset in ('spcm','voxforge')
    model_loader = get_model('yamnet' if audio_dataset else 'resnet20')
    if args.dataset == 'synthetic':
        from synthetic import get_data, get_ood
        loader = functools.partial(get_data, seed=args.seed)
        ood_loader = get_ood
    else:
        loaders = {'cifar10':get_cifar10,'cifar100':get_cifar100,'pathmnist':get_pathmnist,
                   'spcm':get_spcm,'voxforge':get_voxforge}
        loader = functools.partial(loaders[args.dataset], seed=args.seed)
        from recovered_data import load_ood_audio, load_ood_images
        ood_loader = functools.partial(load_ood_audio if audio_dataset else load_ood_images, args.ood_dir)
    output.mkdir(parents=True,exist_ok=True)
    clustered = args.method != 'fedavg'
    if client_id is not None:
        return _Client(int(client_id),args.clients,model_loader,loader,args.batch_size,clustered,args.seed)
    server = _Server(model_loader,loader,args.rounds,num_clients=args.clients,participation=args.participation,
        init_model_fp=str(args.init_model.resolve()) if args.init_model else None,
        model_save_dir=lambda compressed: str(output / ('compressed.h5' if compressed else 'aggregated.h5')),
        server_compression=args.method=='fedcompress',client_compression=clustered,
        clients_config={'lr':args.lr,'epochs':args.warmup_epochs if clustered else args.epochs,
                        'compression_lr':args.cluster_lr,'compression_epochs':args.epochs-args.warmup_epochs if clustered else 0},
        server_compression_config={'compression_step':args.compression_step,'epochs':args.server_epochs,
            'learning_rate':args.server_lr,'batch_size':args.batch_size,'temperature':args.temperature,'data_loader':ood_loader,'seed':args.seed},
        cluster_search_config={'init_num_clusters':args.min_clusters,'max_num_clusters':args.max_clusters,
            'cluster_update_step':args.cluster_step,'init_rounds':1,'window':args.window,
            'patience':args.patience,'limit':args.score_tolerance*100})
    strategy=server.strategy
    parameters=strategy.initial_parameters
    if driver is None: strategy.evaluate(0,parameters)
    report={'backend':'sequential-stock-flower-npz',
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'runtime':{'python':platform.python_version(),'tensorflow':tf.__version__,'flower':fl.__version__,'numpy':np.__version__},'rounds':[]}
    if driver is not None:
        from native_transport import run_native
        return run_native(driver,server,args.rounds,args.clients,args.seed,output,report)
    class Manager:
        def num_available(self): return args.clients
        def sample(self,num_clients,min_num_clients=None):
            from types import SimpleNamespace
            return [SimpleNamespace(cid=str(i)) for i in sorted(random.sample(range(args.clients),num_clients))]
    for rnd in range(1,args.rounds+1):
        before=fl.common.parameters_to_ndarrays(strategy.parameters or parameters)
        results=[]; upstream=downstream=0
        for proxy,instruction in strategy.configure_fit(rnd,parameters,Manager()):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(args.seed+rnd*args.clients+int(proxy.cid))
            client=_Client(int(proxy.cid),args.clients,model_loader,loader,args.batch_size,clustered,args.seed)
            proto=serde.fit_ins_to_proto(instruction); downstream+=proto.ByteSize()
            response=client.to_client().fit(serde.fit_ins_from_proto(proto))
            proto=serde.fit_res_to_proto(response); upstream+=proto.ByteSize()
            results.append((proxy,serde.fit_res_from_proto(proto)))
        parameters,metrics=strategy.aggregate_fit(rnd,results,[])
        loss,evaluation=strategy.evaluate(rnd,parameters)
        after=fl.common.parameters_to_ndarrays(strategy.parameters)
        row={'round':rnd,'loss':float(loss),'accuracy':float(evaluation.get('compressed_accuracy',evaluation['accuracy'])),
             'effective_rank':float(metrics['val_score']),'clusters_next_round':server.num_clusters,
             'upstream_protobuf_bytes':upstream,'downstream_protobuf_bytes':downstream,
             'parameters_changed':any(not np.array_equal(a,b) for a,b in zip(before,after)),
             'client_examples':[r.num_examples for _,r in results],
             'server_self_compression':args.method=='fedcompress' and rnd in server.compression_rnds}
        report['rounds'].append(row)
        (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    np.savez_compressed(output/'final-parameters.npz',*fl.common.parameters_to_ndarrays(strategy.parameters))
    return report


if __name__=='__main__': main()
